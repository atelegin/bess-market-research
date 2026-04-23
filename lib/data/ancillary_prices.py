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

from lib.data.cache import CACHE_DIR

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


def fetch_afrr_energy_prices(year: int) -> pd.DataFrame | None:
    """
    Fetch aFRR activation (energy) market clearing prices per 15-min product.

    Source: regelleistung.net RESULT_OVERVIEW_ENERGY_MARKET_aFRR_{year}.xlsx.
    German aFRR settles 96 products per day per direction (POS/NEG), each a
    15-min interval. Columns include MIN / AVERAGE / MARGINAL energy price
    in EUR/MWh.

    Returns a DataFrame indexed by UTC timestamp with columns::

        pos_avg_eur_mwh, pos_marginal_eur_mwh,
        neg_avg_eur_mwh, neg_marginal_eur_mwh

    The index is the interval start (15-min resolution). Both directions in
    the same timestamp index.
    """
    cache_path = CACHE_DIR / f"afrr_energy_prices_{year}_v2.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    url = f"{BASE_URL}/RESULT_OVERVIEW_ENERGY_MARKET_aFRR_{year}-01-01_{year}-12-31.xlsx"
    try:
        r = requests.get(url, timeout=120)
        if r.status_code != 200:
            logger.warning(f"aFRR energy prices {year}: HTTP {r.status_code}")
            return None
        raw = pd.read_excel(io.BytesIO(r.content))
    except Exception as e:
        logger.warning(f"aFRR energy prices {year} fetch failed: {e}")
        return None

    # Parse PRODUCT = "{DIR}_{NNN}" where DIR ∈ {POS, NEG} and NNN = 1..96
    # identifies the 15-min block inside the day (UTC, local German time may
    # differ but the publisher returns CET/CEST-aligned delivery day; we
    # standardise at UTC using the delivery date + 15-min offset, accepting
    # a small alignment skew — sufficient for annual aggregates).
    parts = raw["PRODUCT"].astype(str).str.split("_", expand=True)
    raw["direction"] = parts[0]
    raw["block_idx"] = pd.to_numeric(parts[1], errors="coerce").astype("Int64")
    raw = raw.dropna(subset=["block_idx"])

    minutes_offset = (raw["block_idx"].astype(int) - 1) * 15
    raw["timestamp"] = pd.to_datetime(raw["DELIVERY_DATE"]) + pd.to_timedelta(
        minutes_offset, unit="m"
    )

    avg_col = "GERMANY_AVERAGE_ENERGY_PRICE_[EUR/MWh]"
    marg_col = "GERMANY_MARGINAL_ENERGY_PRICE_[EUR/MWh]"
    raw["avg"] = pd.to_numeric(raw[avg_col], errors="coerce")
    raw["marg"] = pd.to_numeric(raw[marg_col], errors="coerce")

    pos = raw[raw["direction"] == "POS"].set_index("timestamp")[["avg", "marg"]]
    pos.columns = ["pos_avg_eur_mwh", "pos_marginal_eur_mwh"]
    neg = raw[raw["direction"] == "NEG"].set_index("timestamp")[["avg", "marg"]]
    neg.columns = ["neg_avg_eur_mwh", "neg_marginal_eur_mwh"]

    out = pos.join(neg, how="outer").sort_index()
    # Localise to UTC so downstream join with activations (UTC-indexed) aligns.
    out.index = out.index.tz_localize("UTC", nonexistent="shift_forward",
                                       ambiguous="NaT")
    out = out[~out.index.isna()]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache_path)
    return out


def compute_afrr_energy_revenue_real(
    year: int,
    contracted_pos_mw: float = 2000.0,
    contracted_neg_mw: float = 1800.0,
    price_basis: str = "avg",
) -> dict[str, float] | None:
    """
    Real aFRR activation-energy revenue per MW of BESS contracted capacity.

    Replaces the ``afrr_cap * 0.04`` proxy with a pairwise calculation of
    activated energy × clearing price, summed over all 15-min intervals for
    the year.

    **Sign convention** (regelleistung.net, German aFRR):
      * POS direction: positive avg price → operator is PAID when activated
        upward (discharges energy into grid).
      * NEG direction: negative avg price → operator PAYS TSO when activated
        downward (absorbs surplus energy). Counter-intuitive but correct:
        operators bid negative NEG prices because the energy absorbed has
        positive alternative value (wholesale resale later). A BESS-only
        view of NEG revenue is therefore conjugate with its wholesale
        resale — the stacked LP (A2) reconciles this properly. Treating
        aFRR-energy revenue in isolation will read "negative" in years when
        wholesale spreads justified paying for charge energy (2023, 2025).

    Empirical DE values at default pool sizes (pay-as-bid "avg" basis)::

        2023: POS +370  NEG -465  net -95 kEUR/MW  (high wholesale → willing to pay)
        2024: POS +358  NEG -338  net +20 kEUR/MW  (balanced)
        2025: POS +337  NEG -346  net  -9 kEUR/MW  (mild paid-to-charge)

    Per MW of contracted capacity, earnings for a pro-rata-activated BESS::

        rev_per_MW = sum_t( activated_MW[t] / pool_MW * price[t] * 0.25h )

    Per MW of contracted capacity, earnings for a pro-rata-activated BESS::

        rev_per_MW = sum_t( activated_MW[t] / pool_MW * price[t] * 0.25h )

    Args:
        year: calendar year, 2023..2025 supported.
        contracted_pos_mw: TSO-contracted POS aFRR pool (MW). Defaults match
            2024 SO GL steady-state.
        contracted_neg_mw: TSO-contracted NEG aFRR pool (MW).
        price_basis: "avg" (weighted average of activated bids — realistic
            proxy for what an average operator earns) or "marginal" (max
            accepted bid price). NB: "marginal" is saturated at the ±15000
            EUR/MWh regulatory cap across most intervals, so "avg" is the
            honest default.

    Returns dict with::

        pos_keur_per_mw, neg_keur_per_mw, net_keur_per_mw,
        symmetric_keur_per_mw   # net assuming 1 MW reserved on both directions
    """
    from lib.data.afrr_activations import fetch_afrr_activations

    prices = fetch_afrr_energy_prices(year)
    if prices is None or prices.empty:
        return None
    activations = fetch_afrr_activations(
        start=f"{year}-01-01", end=f"{year + 1}-01-01",
    )
    if activations.empty:
        return None

    joined = activations.join(prices, how="inner")
    if joined.empty:
        logger.warning(f"aFRR energy {year}: zero overlap between activations and prices")
        return None

    suffix = "marginal" if price_basis == "marginal" else "avg"
    pos_price = joined[f"pos_{suffix}_eur_mwh"].astype(float)
    neg_price = joined[f"neg_{suffix}_eur_mwh"].astype(float)

    dt_h = 0.25  # 15-min intervals
    # Per-MW-contracted activation energy and revenue for POS direction.
    pos_activated_mwh_per_mw = (joined["pos_mw"] / contracted_pos_mw) * dt_h
    neg_activated_mwh_per_mw = (joined["neg_mw"] / contracted_neg_mw) * dt_h

    pos_rev_eur = (pos_activated_mwh_per_mw * pos_price).sum()
    neg_rev_eur = (neg_activated_mwh_per_mw * neg_price).sum()

    pos_keur = pos_rev_eur / 1000.0
    neg_keur = neg_rev_eur / 1000.0
    return {
        "pos_keur_per_mw": pos_keur,
        "neg_keur_per_mw": neg_keur,
        "net_keur_per_mw": pos_keur + neg_keur,
        "symmetric_keur_per_mw": pos_keur + neg_keur,
    }


def fetch_afrr_annual_revenue(year: int, use_real_energy: bool = True) -> dict | None:
    """
    Fetch aFRR capacity auction results and compute annual revenue in kEUR/MW.

    aFRR capacity is auctioned in 4h blocks with EUR/MW/h prices. Energy
    revenue defaults to the **real** activations × clearing-price calc
    (``compute_afrr_energy_revenue_real``, 2026-04 rework per ROADMAP note
    ``trader-aging-aware`` A1.1). Pass ``use_real_energy=False`` to fall
    back to the legacy ``afrr_cap * 0.04`` proxy (preserved for debugging).

    Cache bumped to ``_v2`` so legacy proxy numbers don't mask the new data.
    """
    cache_path = CACHE_DIR / f"afrr_revenue_{year}_v2.csv"
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
        hours_per_block = 4
        annual_cap_keur = (prices.sum() * hours_per_block) / 1000

        # BESS participation: ~40% of time on aFRR capacity
        participation = 0.40
        afrr_cap = annual_cap_keur * participation

        if use_real_energy:
            energy_real = compute_afrr_energy_revenue_real(year)
            if energy_real is not None:
                afrr_energy = energy_real["net_keur_per_mw"]
            else:
                logger.warning(
                    f"aFRR {year}: real energy calc failed, falling back to proxy"
                )
                afrr_energy = afrr_cap * 0.04
        else:
            # Legacy proxy: activation_ratio × capacity_revenue.
            afrr_energy = afrr_cap * 0.04

        result = {"afrr_cap_keur": afrr_cap, "afrr_energy_keur": afrr_energy}
        pd.DataFrame([result]).to_csv(cache_path, index=False)
        logger.info(f"aFRR {year}: cap={afrr_cap:.0f}, energy={afrr_energy:.0f} kEUR/MW/yr")
        return {"afrr_cap": afrr_cap, "afrr_energy": afrr_energy}
    except Exception as e:
        logger.warning(f"aFRR {year} fetch failed: {e}")
        return None
