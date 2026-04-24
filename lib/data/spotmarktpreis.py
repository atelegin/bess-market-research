"""
DE Spotmarktpreis loader — netztransparenz.de EEG §3 Nr. 42a.

The Spotmarktpreis is the volume-weighted average across all spot prices
on EPEX and EXAA for each hour — covering BOTH day-ahead auction clearing
AND intraday auction clearing. When intraday volumes are small (typical
quiet days) it collapses to DA. When intraday-only clears at a scarcity
event (solar forecast miss, weather shock, tight balance), Spotmarktpreis
diverges dramatically from DA — by hundreds or thousands of EUR/MWh in
the tail.

**Empirical check (full 2024, 8783 hours vs EnergyCharts DA)**:
  * mean |diff| = 1.0 EUR/MWh
  * 24 hours with |diff| > 1 EUR/MWh
  * worst day 2024-06-26: Spotmarktpreis peak 2097 vs DA 107 → +1989 delta
    (scarcity event captured in intraday only)

So Spotmarktpreis is a usable proxy for **intraday-inclusive spot pricing**
— much closer to "what a BESS trader with ID access actually faces" than
DA alone. Not a continuous-intraday (ID1/ID3) feed, but the best free
alternative we have for historical DE data (EPEX ID historical is
€5-15k/yr, not in project budget).

Source: netztransparenz.de Data Service API, endpoint
``/data/Spotmarktpreise/{start}/{end}``. OAuth2 via NTP_CLIENT_ID /
NTP_CLIENT_SECRET env vars (same as ``lib.data.afrr_activations``).

Resolution: hourly. For use as a 15-min LP input, step-expand 4×.
"""
from __future__ import annotations

import io
import logging
import os

import pandas as pd
import requests

from lib.data.cache import get_or_build_dataframe, make_cache_key

logger = logging.getLogger(__name__)

_TOKEN_URL = "https://identity.netztransparenz.de/users/connect/token"
_API_BASE = "https://ds.netztransparenz.de/api/v1"


def _get_token() -> str:
    client_id = os.environ.get("NTP_CLIENT_ID", "")
    client_secret = os.environ.get("NTP_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        raise RuntimeError(
            "NTP_CLIENT_ID and NTP_CLIENT_SECRET must be set. "
            "Register free at https://api-portal.netztransparenz.de/"
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


def _build_spotmarktpreis(start: str, end: str) -> pd.DataFrame:
    """Fetch the raw hourly series from netztransparenz and return a
    UTC-indexed DataFrame with column ``price_eur_mwh``.
    """
    token = _get_token()
    url = f"{_API_BASE}/data/Spotmarktpreise/{start}/{end}"
    resp = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=60)
    resp.raise_for_status()
    raw = pd.read_csv(io.StringIO(resp.text), sep=";", decimal=",")
    raw["timestamp"] = pd.to_datetime(
        raw["Datum"] + " " + raw["von"], format="%d.%m.%Y %H:%M",
    ).dt.tz_localize("UTC")
    # Source column is EUR/ct/kWh — convert to EUR/MWh (×10).
    raw["price_eur_mwh"] = pd.to_numeric(
        raw["Spotmarktpreis in ct/kWh"], errors="coerce"
    ) * 10.0
    out = raw[["timestamp", "price_eur_mwh"]].dropna()
    return out.set_index("timestamp").sort_index()


def fetch_spotmarktpreis(
    start: str = "2023-01-01",
    end: str = "2025-12-31T23:00:00",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Hourly DE Spotmarktpreis for [start, end], volume-weighted across
    EPEX + EXAA DA + intraday auctions.

    Args:
        start: ISO date or datetime string. Inclusive.
        end: ISO datetime string. Inclusive.
        force_refresh: bypass parquet cache.

    Returns:
        DataFrame indexed by UTC timestamp with column ``price_eur_mwh``.
    """
    cache_key = make_cache_key(
        "spotmarktpreis",
        start=start,
        end=end,
        source="netztransparenz_eeg_42a_v1",
    )
    return get_or_build_dataframe(
        cache_key=cache_key,
        builder=lambda: _build_spotmarktpreis(start=start, end=end),
        ttl_hours=24 * 30,
        force_refresh=force_refresh,
        metadata={"source": f"{_API_BASE}/data/Spotmarktpreise"},
    )
