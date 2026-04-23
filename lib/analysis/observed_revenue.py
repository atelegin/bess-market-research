"""
Observed historical revenue — Tier 1/2 ground truth, independent of
third-party benchmark indices (CH, LCP, enspired, suena, RWTH).

Philosophy (Note 4 A1.2 follow-up, 2026-04-23): third-party revenue
benchmarks disagree by up to ~100 kEUR/MW/yr. Using any single index as
"truth" bakes that benchmark's methodology into our calibration. Instead,
we build our own ground-truth layer from public sources an independent
researcher can reproduce:

  * **Tier 1 — TSO-published auction & activation data**: regelleistung.net
    FCR/aFRR capacity auctions, netztransparenz.de aFRR activation volumes,
    EnergyCharts DE-LU day-ahead prices.
  * **Tier 2 — deterministic computation on Tier 1**: measured AS revenue
    with documented participation factors; wholesale via LP dispatch on
    observed DE prices with a documented strategy.

Third-party benchmarks (CH et al.) then sit in **Tier 3** — context for
"how does our ground truth compare to published indices" rather than
calibration anchors.

``observed_total_revenue`` is the entry point. It returns what a reference
BESS actually earned per MW of nameplate in a given historical year,
built entirely from Tier 1/2 sources.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from lib.data.ancillary_prices import (
    fetch_afrr_annual_revenue,
    fetch_fcr_annual_revenue,
)
from lib.data.day_ahead_prices import fetch_day_ahead_prices, prices_to_daily_arrays
from lib.models.ancillary import HISTORICAL_YEARS_WITH_MEASURED_DATA
from lib.models.dispatch import annual_revenue, dispatch_day
from lib.models.projection import id_da_ratio

logger = logging.getLogger(__name__)

# Default dispatch config for wholesale backtest. Matches Note 1's
# precompute baseline to keep comparability. Changes here shift our
# ground-truth wholesale estimate by several kEUR/MW/yr — document at
# call sites.
DEFAULT_RTE = 0.85
DEFAULT_MAX_CYCLES = 2


def _wholesale_da_revenue_keur_per_mw(
    year: int, duration_h: float, rte: float, max_cycles: int,
) -> Optional[float]:
    """Annual DA arbitrage revenue per MW from LP dispatch on observed prices."""
    try:
        prices_df = fetch_day_ahead_prices(
            start=f"{year}-01-01", end=f"{year}-12-31",
        )
    except Exception as exc:
        logger.warning(f"observed_total_revenue({year}): DA fetch failed — {exc}")
        return None

    daily = prices_to_daily_arrays(prices_df, resolution_minutes=60)
    year_key = str(year)
    if year_key not in daily:
        logger.warning(f"observed_total_revenue({year}): no daily prices in {year_key}")
        return None

    results = [
        dispatch_day(p, duration_h=duration_h, rte=rte, max_cycles=max_cycles)
        for p in daily[year_key]
    ]
    # annual_revenue returns EUR per MW; convert to kEUR/MW/yr.
    return annual_revenue(results) / 1000.0


def observed_total_revenue(
    year: int,
    duration_h: float = 2.0,
    rte: float = DEFAULT_RTE,
    max_cycles: int = DEFAULT_MAX_CYCLES,
) -> Optional[dict[str, float]]:
    """
    Reproducible historical per-MW revenue from Tier 1/2 sources only.

    Composition::

        da            — LP dispatch on observed DE-LU day-ahead prices,
                        duration/RTE/cycles-per-day as given
        id            — da * id_da_ratio(year) (empirical split; a proper
                        ID-continuous dispatch is out of scope here)
        fcr           — regelleistung.net capacity auctions × 0.35
                        participation factor (our documented assumption)
        afrr_cap      — regelleistung.net × 0.40 participation
        afrr_energy   — real netztransparenz activations × regelleistung
                        energy-market clearing prices (see
                        ``compute_afrr_energy_revenue_real``). Can be
                        negative in years when wholesale spreads justified
                        paying to charge via NEG activations (2023, 2025).
        total         — sum of the above

    **Important caveats**:
      * Participation factors 0.35 FCR / 0.40 aFRR are our assumptions, not
        measured operator behaviour. Documented in the fetch functions.
      * **aFRR↔wholesale conjugate coupling is NOT fully accounted for here.**
        When aFRR_energy is negative (operator pays to charge via NEG
        activations), the absorbed energy has positive resale value in
        wholesale. Our stand-alone LP dispatch on DA prices does NOT see
        this "free charge" input — it solves wholesale as an independent
        market. In years with wide spreads (2023) this omission is
        material: e.g. 2023 observed total is ~146 kEUR vs CH ~230, a 35%
        gap largely explained by the uncounted coupling. The stacked-market
        LP (Note 4 A2) closes the gap by treating aFRR reservation and
        wholesale arbitrage as a joint optimisation. Until then, historical
        Tier 1/2 totals under-count by the magnitude of the conjugate
        coupling.
      * Only years in ``HISTORICAL_YEARS_WITH_MEASURED_DATA`` have complete
        regelleistung+netztransparenz data. Other years return ``None``.

    Returns ``None`` if any component source is unavailable.

    Third-party indices (CH, LCP, enspired, suena, RWTH) may differ by
    ±100 kEUR/MW/yr in some months — this function does NOT calibrate to
    any of them. See ``benchmark-reconciliation`` for the methodology
    comparison across indices.
    """
    if year not in HISTORICAL_YEARS_WITH_MEASURED_DATA:
        logger.warning(
            f"observed_total_revenue({year}): year not in "
            f"HISTORICAL_YEARS_WITH_MEASURED_DATA = "
            f"{HISTORICAL_YEARS_WITH_MEASURED_DATA}"
        )
        return None

    # NOTE: this function's AS↔wholesale conjugate coupling is *not*
    # captured. For a coupling-aware number use
    # ``observed_total_revenue_stacked`` below — that runs the joint
    # stacked-market LP which properly models aFRR reservation and
    # activation against wholesale arbitrage.

    da = _wholesale_da_revenue_keur_per_mw(year, duration_h, rte, max_cycles)
    if da is None:
        return None
    id_rev = da * id_da_ratio(year)

    fcr = fetch_fcr_annual_revenue(year)
    afrr = fetch_afrr_annual_revenue(year, use_real_energy=True)
    if fcr is None or afrr is None:
        logger.warning(
            f"observed_total_revenue({year}): AS data unavailable — "
            f"fcr={fcr}, afrr={afrr}"
        )
        return None

    dur_scale = min(duration_h / 2.0, 1.0)
    fcr_scaled = float(fcr) * dur_scale
    afrr_cap_scaled = float(afrr["afrr_cap"]) * dur_scale
    afrr_energy_scaled = float(afrr["afrr_energy"]) * dur_scale

    total = da + id_rev + fcr_scaled + afrr_cap_scaled + afrr_energy_scaled
    return {
        "year": year,
        "duration_h": duration_h,
        "da": round(da, 1),
        "id": round(id_rev, 1),
        "fcr": round(fcr_scaled, 1),
        "afrr_cap": round(afrr_cap_scaled, 1),
        "afrr_energy": round(afrr_energy_scaled, 1),
        "total": round(total, 1),
    }


def observed_total_revenue_stacked(
    year: int,
    duration_h: float = 2.0,
    max_cycles: float = 1.5,
    rte: float = DEFAULT_RTE,
    afrr_reserve_duration_hours: float = 0.25,
    power_mw: float = 1.0,
    max_afrr_participation: float = 0.40,
) -> Optional[dict[str, float]]:
    """
    Coupling-aware Tier 2 per-MW annual revenue via joint LP (Note 4 A2.3).

    Runs the joint DA + ID + aFRR cap + aFRR energy LP day-by-day over
    the year. Unlike ``observed_total_revenue`` this routes aFRR
    reservation and activation *through the same LP* as wholesale, so
    the aFRR↔wholesale conjugate coupling is captured correctly — the
    "free charge from NEG activations" feeds the LP's SoC budget and
    gets priced against the wholesale resale opportunity.

    Calibration to match realistic operator (default)
    -------------------------------------------------
    At full participation (``max_afrr_participation=1.0``) the LP
    reserves ~90 % of nameplate for aFRR and the annual number lands
    2.2× above published benchmarks (CH, LCP, enspired, suena, RWTH).
    Five independent benchmarks converging at the same level strongly
    suggests our full-participation LP is *wrong* for the typical
    operator — not that every trader leaves millions on the table.

    The gap is closed by a participation cap that encodes real-world
    frictions the LP doesn't model:
      * day-ahead bid competition (offered/demand ≈ 0.55 in 2024)
      * risk-averse bid placement (avoiding deep scarcity activations)
      * maintenance and operational withholding
      * imperfect intraday/activation forecasts
      * market-design limits (can't bid every block of the year)

    Empirically calibrated to CH: at ``max_afrr_participation = 0.40``
    the LP matches 2023 (~245 vs CH 230, +6 %) and 2025 (~257 vs CH 236,
    +9 %) within benchmark fan tolerance. 0.40 also matches the
    historical participation factor ROADMAP Note 1 calibration quoted
    (40 % on aFRR capacity auctions). This is the honest realistic
    default. Raise toward 1.0 for the "LP-optimal upper bound" used in
    Note 4's optimizer-gap narrative (~2.2× uplift at full
    participation).

    Note: ID prices default to DA-proxy — netztransparenz AEP is TSO's
    imbalance settlement price, not a tradable ID market, so using it
    as an ID price lets the LP exploit non-tradable volatility. Real
    intraday market data (EPEX ID1/ID3, XBID) is not ingested yet;
    adding it would lift the calibrated participation level (since
    genuine intraday trades real-world operators DO capture).

    Requires complete regelleistung + netztransparenz + EnergyCharts
    data for the year. Returns ``None`` if the runner solves zero days.
    """
    from lib.analysis.stacked_year_runner import run_stacked_year

    if year not in HISTORICAL_YEARS_WITH_MEASURED_DATA:
        logger.warning(
            f"observed_total_revenue_stacked({year}): year not in "
            f"HISTORICAL_YEARS_WITH_MEASURED_DATA"
        )
        return None

    result = run_stacked_year(
        year=year, duration_h=duration_h, max_cycles=max_cycles,
        power_mw=power_mw, rte=rte,
        afrr_reserve_duration_hours=afrr_reserve_duration_hours,
        max_afrr_participation=max_afrr_participation,
    )
    if result.days_solved == 0:
        return None

    out = result.revenue_by_stream_annual()
    out.update({
        "year": year,
        "duration_h": duration_h,
        "days_solved": result.days_solved,
        "days_skipped": result.days_skipped,
        "annual_fec": round(result.annual_fec, 1),
        "mean_r_pos_mw": round(result.mean_r_pos_mw, 3),
        "mean_r_neg_mw": round(result.mean_r_neg_mw, 3),
    })
    return out
