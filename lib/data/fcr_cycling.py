"""
FCR cycling estimate from grid frequency data.

Source: 1-second grid frequency measurements from TransnetBW (Continental Europe),
available via power-grid-frequency.org / OSF (https://osf.io/m43tg/).

FCR (Frequency Containment Reserve) responds proportionally to frequency deviations:
- Deadband: ±10 mHz (no activation)
- Full activation: ±200 mHz
- Proportional between deadband and full activation

Result: ~0.25 FEC/day for a 1 MW / 2h BESS, stable across years.
Validated against M5BAT study (0.28 EFC/day over 4 years of real FCR operation).
"""
from __future__ import annotations

import io
import logging
import zipfile

import numpy as np
import pandas as pd
import requests

from lib.data.cache import get_or_build_dataframe, make_cache_key

logger = logging.getLogger(__name__)

# OSF project: Pre-Processed Power Grid Frequency Time Series
_OSF_BASE = "https://api.osf.io/v2/nodes/m43tg/files/osfstorage/"

# FCR droop parameters (Continental Europe)
DEADBAND_HZ = 0.010
FULL_ACTIVATION_HZ = 0.200


def _navigate_osf(path_parts: list[str]) -> dict:
    """Navigate OSF folder structure and return API response dict."""
    r = requests.get(_OSF_BASE, timeout=30)
    data = r.json()
    for part in path_parts:
        for item in data.get("data", []):
            if item["attributes"]["name"] == part and item["attributes"]["kind"] == "folder":
                folder_url = item["relationships"]["files"]["links"]["related"]["href"]
                r = requests.get(folder_url, timeout=30)
                data = r.json()
                break
    return data


def _get_month_download_url(year: int, month: int) -> str | None:
    """Get OSF download URL for a specific month's frequency data."""
    data = _navigate_osf(["Data", "Continental Europe", "Germany", str(year), f"{month:02d}"])
    for item in data.get("data", []):
        if item["attributes"]["name"].endswith(".csv.zip"):
            return item["links"]["download"]
    return None


def _compute_fcr_from_frequency(zip_content: bytes, bess_duration_h: float = 2.0) -> pd.DataFrame:
    """Compute FCR cycling from frequency data zip file.

    Returns DataFrame with columns: date, fec (daily FEC for 1 MW BESS).
    """
    z = zipfile.ZipFile(io.BytesIO(zip_content))
    csv_name = z.namelist()[0]

    df = pd.read_csv(
        z.open(csv_name), skiprows=1, header=None,
        names=["timestamp", "freq_mhz"], low_memory=False,
    )
    df["freq_mhz"] = pd.to_numeric(df["freq_mhz"], errors="coerce")
    df = df.dropna(subset=["freq_mhz"])

    # Values are mHz deviation from 50 Hz
    abs_delta = np.abs(df["freq_mhz"].values / 1000.0)

    # Proportional droop with deadband
    activation = np.where(
        abs_delta <= DEADBAND_HZ, 0.0,
        np.clip((abs_delta - DEADBAND_HZ) / (FULL_ACTIVATION_HZ - DEADBAND_HZ), 0, 1),
    )

    dt_h = 1.0 / 3600.0  # 1 second in hours
    bess_mwh = 1.0 * bess_duration_h  # 1 MW

    df["energy_mwh"] = activation * dt_h
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date

    daily = df.groupby("date")["energy_mwh"].sum()
    daily_fec = daily / (2 * bess_mwh)

    return daily_fec.reset_index().rename(columns={"energy_mwh": "fec"})


def fetch_fcr_monthly_cycling(
    year: int,
    month: int,
    bess_duration_h: float = 2.0,
) -> pd.DataFrame | None:
    """Fetch one month of FCR cycling data.

    Returns DataFrame with columns: date, fec (daily FEC for 1 MW BESS).
    """
    cache_key = make_cache_key(
        "fcr_cycling",
        year=year, month=month,
        duration_h=bess_duration_h,
        source="osf_transnetbw_v1",
    )

    def _build():
        url = _get_month_download_url(year, month)
        if url is None:
            logger.warning(f"FCR frequency data not available for {year}-{month:02d}")
            return pd.DataFrame(columns=["date", "fec"])
        logger.info(f"Downloading FCR frequency data {year}-{month:02d}...")
        r = requests.get(url, timeout=120, allow_redirects=True)
        r.raise_for_status()
        return _compute_fcr_from_frequency(r.content, bess_duration_h)

    return get_or_build_dataframe(
        cache_key=cache_key,
        builder=_build,
        ttl_hours=24 * 365,  # frequency data is historical, never changes
        metadata={"source": "osf.io/m43tg", "area": "Continental Europe", "tso": "TransnetBW"},
    )


def compute_fcr_annual_average(
    year: int,
    sample_months: list[int] | None = None,
    bess_duration_h: float = 2.0,
) -> float | None:
    """Compute average daily FCR cycling for a year (sampled quarterly).

    Args:
        year: Year (2011-2020 available).
        sample_months: Months to sample. Default: [1, 4, 7, 10].
        bess_duration_h: BESS duration in hours.

    Returns:
        Average FEC/day, or None if no data available.
    """
    if sample_months is None:
        sample_months = [1, 4, 7, 10]

    monthly_avgs = []
    for month in sample_months:
        df = fetch_fcr_monthly_cycling(year, month, bess_duration_h)
        if df is not None and not df.empty:
            monthly_avgs.append(df["fec"].mean())

    if not monthly_avgs:
        return None
    return float(np.mean(monthly_avgs))
