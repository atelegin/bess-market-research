"""
Ancillary service prices — FCR and aFRR capacity auctions.

Source: regelleistung.net public Excel downloads (no auth required).
Provides weekly prices and annualised revenue estimates per MW.
"""

import requests
import pandas as pd
import numpy as np
import io
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

BASE_URL = "https://www.regelleistung.net/apps/cpp-publisher/api/v1/download/tenders/files"


def fetch_fcr_weekly_prices(year: int) -> pd.DataFrame | None:
    """
    Fetch FCR settlement prices per 4h block and aggregate to weekly average.
    Returns DataFrame with columns: [week_start, eur_mw_4h] (average EUR/MW per 4h block).
    """
    cache_path = CACHE_DIR / f"fcr_weekly_{year}.csv"
    if cache_path.exists():
        df = pd.read_csv(cache_path, parse_dates=["week_start"])
        return df

    url = f"{BASE_URL}/RESULT_OVERVIEW_CAPACITY_MARKET_FCR_{year}-01-01_{year}-12-31.xlsx"
    try:
        r = requests.get(url, timeout=60)
        if r.status_code != 200:
            return None
        df = pd.read_excel(io.BytesIO(r.content))
        df["date"] = pd.to_datetime(df["DATE_FROM"])
        price_col = "GERMANY_SETTLEMENTCAPACITY_PRICE_[EUR/MW]"
        df["price"] = pd.to_numeric(df[price_col], errors="coerce")
        df = df.dropna(subset=["price"])
        df["week_start"] = df["date"].dt.to_period("W").apply(lambda p: p.start_time)
        weekly = df.groupby("week_start")["price"].mean().reset_index()
        weekly.columns = ["week_start", "eur_mw_4h"]
        weekly.to_csv(cache_path, index=False)
        return weekly
    except Exception as e:
        logger.warning(f"FCR weekly {year}: {e}")
        return None


def fetch_afrr_weekly_prices(year: int) -> pd.DataFrame | None:
    """
    Fetch aFRR capacity prices per 4h block and aggregate to weekly average.
    Returns DataFrame with columns: [week_start, eur_mw_h] (average EUR/MW/h).
    """
    cache_path = CACHE_DIR / f"afrr_weekly_{year}.csv"
    if cache_path.exists():
        df = pd.read_csv(cache_path, parse_dates=["week_start"])
        return df

    url = f"{BASE_URL}/RESULT_OVERVIEW_CAPACITY_MARKET_aFRR_{year}-01-01_{year}-12-31.xlsx"
    try:
        r = requests.get(url, timeout=60)
        if r.status_code != 200:
            return None
        df = pd.read_excel(io.BytesIO(r.content))
        df["date"] = pd.to_datetime(df["DATE_FROM"])
        avg_col = "GERMANY_AVERAGE_CAPACITY_PRICE_[(EUR/MW)/h]"
        df["price"] = pd.to_numeric(df[avg_col], errors="coerce")
        df = df.dropna(subset=["price"])
        df["week_start"] = df["date"].dt.to_period("W").apply(lambda p: p.start_time)
        weekly = df.groupby("week_start")["price"].mean().reset_index()
        weekly.columns = ["week_start", "eur_mw_h"]
        weekly.to_csv(cache_path, index=False)
        return weekly
    except Exception as e:
        logger.warning(f"aFRR weekly {year}: {e}")
        return None


def fetch_fcr_annual_revenue(year: int) -> float | None:
    """
    Fetch FCR capacity auction results and compute annual revenue in kEUR/MW.

    FCR is auctioned in 4h blocks. The settlement capacity price (EUR/MW per 4h block)
    summed over all blocks gives the maximum annual revenue for 100% FCR participation.

    For a 2h BESS doing 2 cycles/day, FCR participation is limited (~30-50% of time),
    so we apply a participation factor.
    """
    cache_path = CACHE_DIR / f"fcr_revenue_{year}.csv"
    if cache_path.exists():
        df = pd.read_csv(cache_path)
        return df["annual_keur_mw"].iloc[0]

    url = f"{BASE_URL}/RESULT_OVERVIEW_CAPACITY_MARKET_FCR_{year}-01-01_{year}-12-31.xlsx"
    try:
        r = requests.get(url, timeout=60)
        if r.status_code != 200:
            logger.warning(f"FCR {year}: HTTP {r.status_code}")
            return None
        df = pd.read_excel(io.BytesIO(r.content))
        prices = pd.to_numeric(
            df["GERMANY_SETTLEMENTCAPACITY_PRICE_[EUR/MW]"], errors="coerce"
        ).dropna()

        # Full participation revenue (100% of time on FCR)
        full_annual_keur = prices.sum() / 1000

        # BESS participation factor: a 2h battery can't do FCR 100% of time
        # (needs to cycle for arbitrage). Typical ~35% of hours on FCR.
        participation = 0.35
        annual_keur = full_annual_keur * participation

        pd.DataFrame({"annual_keur_mw": [annual_keur]}).to_csv(cache_path, index=False)
        logger.info(f"FCR {year}: {annual_keur:.0f} kEUR/MW/yr (full={full_annual_keur:.0f})")
        return annual_keur
    except Exception as e:
        logger.warning(f"FCR {year} fetch failed: {e}")
        return None


def fetch_afrr_annual_revenue(year: int) -> dict | None:
    """
    Fetch aFRR capacity auction results and compute annual revenue in kEUR/MW.

    aFRR is auctioned in 4h blocks with EUR/MW/h prices.
    Returns dict with afrr_cap and afrr_energy estimates.
    """
    cache_path = CACHE_DIR / f"afrr_revenue_{year}.csv"
    if cache_path.exists():
        df = pd.read_csv(cache_path)
        return {
            "afrr_cap": df["afrr_cap_keur"].iloc[0],
            "afrr_energy": df["afrr_energy_keur"].iloc[0],
        }

    url = f"{BASE_URL}/RESULT_OVERVIEW_CAPACITY_MARKET_aFRR_{year}-01-01_{year}-12-31.xlsx"
    try:
        r = requests.get(url, timeout=60)
        if r.status_code != 200:
            logger.warning(f"aFRR {year}: HTTP {r.status_code}")
            return None
        df = pd.read_excel(io.BytesIO(r.content))

        avg_col = "GERMANY_AVERAGE_CAPACITY_PRICE_[(EUR/MW)/h]"
        prices = pd.to_numeric(df[avg_col], errors="coerce").dropna()

        # Each row is a 4h block at EUR/MW/h. Annual capacity revenue = sum * 4 / 1000
        # (4h per block, price is per hour)
        n_blocks = len(prices)
        hours_per_block = 4
        annual_cap_keur = (prices.sum() * hours_per_block) / 1000

        # BESS participation: ~40% of time on aFRR capacity
        participation = 0.40
        afrr_cap = annual_cap_keur * participation

        # aFRR energy revenue estimate: ~8-10% of capacity revenue
        # (energy is called ~10% of time, paid at energy price)
        afrr_energy = afrr_cap * 0.10

        result = {"afrr_cap_keur": afrr_cap, "afrr_energy_keur": afrr_energy}
        pd.DataFrame([result]).to_csv(cache_path, index=False)
        logger.info(f"aFRR {year}: cap={afrr_cap:.0f}, energy={afrr_energy:.0f} kEUR/MW/yr")
        return {"afrr_cap": afrr_cap, "afrr_energy": afrr_energy}
    except Exception as e:
        logger.warning(f"aFRR {year} fetch failed: {e}")
        return None
