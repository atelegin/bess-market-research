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
