"""
aFRR activation volumes — actual activated balancing energy (SRL) per 15-min interval.

Source: netztransparenz.de Data Service API (OAuth2 required).
Endpoint: /data/NrvSaldo/AktivierteSRL/{quality}/{start}/{end}
Provides 15-min activated MW (positive = upward, negative = downward) per TSO and for Germany.

Credentials: NTP_CLIENT_ID / NTP_CLIENT_SECRET env vars.
Register free at https://api-portal.netztransparenz.de/
"""
from __future__ import annotations

import io
import logging
import os
from datetime import datetime, timezone

import pandas as pd
import requests

from lib.data.cache import get_or_build_dataframe, make_cache_key

logger = logging.getLogger(__name__)

_TOKEN_URL = "https://identity.netztransparenz.de/users/connect/token"
_API_BASE = "https://ds.netztransparenz.de/api/v1"
_DATE_FMT = "%Y-%m-%dT%H:%M:%S"


def _get_token() -> str:
    """Obtain OAuth2 access token using client credentials flow."""
    client_id = os.environ.get("NTP_CLIENT_ID", "")
    client_secret = os.environ.get("NTP_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        raise RuntimeError(
            "NTP_CLIENT_ID and NTP_CLIENT_SECRET must be set. "
            "Register at https://api-portal.netztransparenz.de/"
        )
    resp = requests.post(
        _TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def _fetch_month(token: str, year: int, month: int, quality: str) -> pd.DataFrame:
    """Fetch one month of activated SRL data."""
    start = datetime(year, month, 1, tzinfo=timezone.utc)
    if month == 12:
        end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        end = datetime(year, month + 1, 1, tzinfo=timezone.utc)

    url = (
        f"{_API_BASE}/data/NrvSaldo/AktivierteSRL/{quality}"
        f"/{start.strftime(_DATE_FMT)}/{end.strftime(_DATE_FMT)}"
    )
    resp = requests.get(
        url,
        headers={"Authorization": f"Bearer {token}"},
        timeout=60,
    )
    if resp.status_code == 404:
        return pd.DataFrame()
    resp.raise_for_status()

    df = pd.read_csv(
        io.StringIO(resp.text),
        sep=";",
        decimal=",",
        na_values=["N.A.", "N.E.", ""],
    )
    return df


def _build_afrr_activations(start: str, end: str) -> pd.DataFrame:
    """Fetch activated SRL data from netztransparenz.de and return clean DataFrame.

    Returns DataFrame indexed by UTC timestamp with columns:
        pos_mw — average activated upward power (MW) for Germany
        neg_mw — average activated downward power (MW) for Germany
    """
    token = _get_token()

    start_dt = pd.Timestamp(start, tz="UTC")
    end_dt = pd.Timestamp(end, tz="UTC")

    frames = []
    current = start_dt.replace(day=1)
    while current < end_dt:
        year, month = current.year, current.month
        # Try quality-assured first, fall back to operational
        for quality in ("Qualitaetsgesichert", "Betrieblich"):
            df = _fetch_month(token, year, month, quality)
            if not df.empty:
                logger.info(f"aFRR activations {year}-{month:02d}: {len(df)} rows ({quality})")
                frames.append(df)
                break
        else:
            logger.warning(f"aFRR activations {year}-{month:02d}: no data")

        # Advance to next month
        if month == 12:
            current = pd.Timestamp(year + 1, 1, 1, tz="UTC")
        else:
            current = pd.Timestamp(year, month + 1, 1, tz="UTC")

    if not frames:
        return pd.DataFrame(columns=["pos_mw", "neg_mw"])

    raw = pd.concat(frames, ignore_index=True)

    # Parse timestamps: "DD.MM.YYYY" + "HH:MM" + "UTC"
    raw["timestamp"] = pd.to_datetime(
        raw["Datum"] + " " + raw["von"],
        format="%d.%m.%Y %H:%M",
    ).dt.tz_localize("UTC")

    result = pd.DataFrame({
        "timestamp": raw["timestamp"],
        "pos_mw": pd.to_numeric(raw["Deutschland (Positiv)"], errors="coerce").fillna(0),
        "neg_mw": pd.to_numeric(raw["Deutschland (Negativ)"], errors="coerce").fillna(0),
    })
    result = result.set_index("timestamp").sort_index()
    result = result[(result.index >= start_dt) & (result.index < end_dt)]
    return result


def fetch_afrr_activations(
    start: str = "2023-01-01",
    end: str = "2025-12-31",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch activated aFRR (SRL) volumes for Germany.

    Args:
        start: ISO date string (e.g. "2024-01-01").
        end: ISO date string (exclusive).
        force_refresh: bypass cache.

    Returns:
        DataFrame indexed by UTC timestamp with columns pos_mw, neg_mw.
        Resolution: 15 minutes. Values are average activated MW for each interval.
    """
    cache_key = make_cache_key(
        "afrr_activations",
        start=start,
        end=end,
        source="netztransparenz_srl_v1",
    )
    return get_or_build_dataframe(
        cache_key=cache_key,
        builder=lambda: _build_afrr_activations(start=start, end=end),
        ttl_hours=24 * 30,
        force_refresh=force_refresh,
        metadata={"source": f"{_API_BASE}/data/NrvSaldo/AktivierteSRL"},
    )


def compute_afrr_daily_fec(
    activations: pd.DataFrame,
    contracted_pos_mw: float = 2000,
    contracted_neg_mw: float = 1800,
    bess_mw: float = 1.0,
    bess_duration_h: float = 2.0,
) -> pd.Series:
    """Compute daily FEC from aFRR activations for a contracted BESS.

    Assumes proportional activation: a BESS contracted for `bess_mw` out of
    the total contracted pool receives a proportional share of activations.

    Args:
        activations: DataFrame from fetch_afrr_activations.
        contracted_pos_mw: total contracted positive aFRR capacity (MW).
        contracted_neg_mw: total contracted negative aFRR capacity (MW).
        bess_mw: BESS contracted capacity (MW).
        bess_duration_h: BESS duration (hours).

    Returns:
        Series indexed by date with daily FEC values.
    """
    dt_h = 0.25  # 15-min intervals
    bess_mwh = bess_mw * bess_duration_h

    df = activations.copy()
    df["date"] = df.index.date

    daily = df.groupby("date").agg({"pos_mw": "sum", "neg_mw": "sum"})

    # Energy throughput per day for 1 MW BESS (proportional share)
    daily["bess_pos_mwh"] = daily["pos_mw"] * dt_h / contracted_pos_mw * bess_mw
    daily["bess_neg_mwh"] = daily["neg_mw"] * dt_h / contracted_neg_mw * bess_mw

    # FEC = total energy throughput / (2 * capacity)
    # pos = discharge, neg = charge; each direction is half a cycle
    daily["fec"] = (daily["bess_pos_mwh"] + daily["bess_neg_mwh"]) / (2 * bess_mwh)

    return daily["fec"]
