"""
Simple BESS dispatch — LP optimisation, array in → revenue out.

Minimal interface for quick revenue estimates: pass a daily price array,
get back revenue/MWh/cycles. For detailed SoC tracking and strategy
comparison use dispatch_detailed.py instead.
"""

import numpy as np
from scipy.optimize import linprog
from dataclasses import dataclass
from typing import Optional


@dataclass
class DispatchResult:
    revenue_eur_per_mw: float
    charge_mwh: float
    discharge_mwh: float
    cycles_used: float
    n_slots: int


def dispatch_day(
    prices: np.ndarray,
    duration_h: float = 2.0,
    rte: float = 0.85,
    max_cycles: int = 2,
    p_rated: float = 1.0,       # normalised to 1 MW
) -> DispatchResult:
    """
    Optimal dispatch of a battery over one day given 15-min prices.

    Parameters
    ----------
    prices : array of shape (T,), EUR/MWh per 15-min slot.
             T = 96 for a full day, but handles any length.
    duration_h : battery duration in hours (E_max = p_rated * duration_h)
    rte : round-trip efficiency
    max_cycles : max full-cycle-equivalents per day
    p_rated : rated power in MW (default 1 for per-MW calculation)

    Returns
    -------
    DispatchResult with revenue in EUR (per p_rated MW) and diagnostics.
    """
    T = len(prices)
    if T == 0:
        return DispatchResult(0.0, 0.0, 0.0, 0.0, 0)

    dt = 24.0 / T  # hours per slot (0.25 for 96 slots, 1.0 for 24)
    eta_c = rte ** 0.5
    eta_d = rte ** 0.5
    e_max = p_rated * duration_h

    # Decision variables: x = [c_0..c_{T-1}, d_0..d_{T-1}]
    # Objective: maximise Σ p_t * (d_t - c_t) * dt
    # linprog minimises, so negate
    c_obj = prices * dt          # cost of charging
    d_obj = -prices * dt         # revenue from discharging
    c_vec = np.concatenate([c_obj, d_obj])  # minimise this

    # --- Inequality constraints: A_ub @ x <= b_ub ---
    A_rows = []
    b_rows = []

    # SoC constraints: for each time step t (1..T), SoC_t must be in [0, e_max]
    # SoC_t = SoC_0 + Σ_{s=0}^{t-1} (eta_c * c_s * dt - d_s * dt / eta_d)
    # SoC_0 = e_max / 2
    soc_0 = e_max / 2.0

    for t in range(1, T + 1):
        # SoC_t <= e_max  →  Σ (eta_c*c_s*dt) - Σ (d_s*dt/eta_d) <= e_max - soc_0
        row_upper = np.zeros(2 * T)
        row_upper[:t] = eta_c * dt          # charge contributes +
        row_upper[T:T + t] = -dt / eta_d    # discharge contributes -
        A_rows.append(row_upper)
        b_rows.append(e_max - soc_0)

        # SoC_t >= 0  →  -Σ (eta_c*c_s*dt) + Σ (d_s*dt/eta_d) <= soc_0
        row_lower = np.zeros(2 * T)
        row_lower[:t] = -eta_c * dt
        row_lower[T:T + t] = dt / eta_d
        A_rows.append(row_lower)
        b_rows.append(soc_0)

    # Cycle budget: Σ d_t * dt <= max_cycles * e_max
    row_cycle = np.zeros(2 * T)
    row_cycle[T:] = dt
    A_rows.append(row_cycle)
    b_rows.append(max_cycles * e_max)

    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)

    # --- Equality constraint: SoC_T = SoC_0 (return to starting SoC) ---
    A_eq_row = np.zeros(2 * T)
    A_eq_row[:T] = eta_c * dt
    A_eq_row[T:] = -dt / eta_d
    A_eq = A_eq_row.reshape(1, -1)
    b_eq = np.array([0.0])  # net SoC change = 0

    # --- Bounds: 0 <= c_t, d_t <= p_rated ---
    bounds = [(0, p_rated)] * (2 * T)

    # --- Solve ---
    result = linprog(
        c_vec,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
        options={"presolve": True},
    )

    if not result.success:
        return DispatchResult(0.0, 0.0, 0.0, 0.0, T)

    c_opt = result.x[:T]
    d_opt = result.x[T:]

    revenue = np.sum((d_opt - c_opt) * prices * dt)
    charge_mwh = np.sum(c_opt * dt)
    discharge_mwh = np.sum(d_opt * dt)
    cycles = discharge_mwh / e_max if e_max > 0 else 0

    return DispatchResult(
        revenue_eur_per_mw=revenue / p_rated,
        charge_mwh=charge_mwh,
        discharge_mwh=discharge_mwh,
        cycles_used=cycles,
        n_slots=T,
    )


def dispatch_year(
    daily_prices: list[np.ndarray],
    **kwargs,
) -> list[DispatchResult]:
    """
    Run dispatch for each day in a year.

    Parameters
    ----------
    daily_prices : list of arrays, one per day (each shape (96,) or (24,))

    Returns
    -------
    list of DispatchResult, one per day
    """
    return [dispatch_day(p, **kwargs) for p in daily_prices]


def annual_revenue(results: list[DispatchResult]) -> float:
    """Sum daily revenues → annual EUR/MW."""
    return sum(r.revenue_eur_per_mw for r in results)


def best_market_per_day(
    da_daily: list[np.ndarray],
    id_daily: list[np.ndarray],
    **kwargs,
) -> dict:
    """
    Approach A: for each day, pick the market with higher LP revenue.

    Returns dict with:
      - annual_da: EUR/MW from days where DA won
      - annual_id: EUR/MW from days where ID won
      - annual_total: combined
      - da_day_count, id_day_count
      - daily_results: list of (market, DispatchResult) tuples
    """
    assert len(da_daily) == len(id_daily), "DA and ID must have same number of days"

    total_da = 0.0
    total_id = 0.0
    da_days = 0
    id_days = 0
    daily = []

    for da_prices, id_prices in zip(da_daily, id_daily):
        r_da = dispatch_day(da_prices, **kwargs)
        r_id = dispatch_day(id_prices, **kwargs)

        if r_da.revenue_eur_per_mw >= r_id.revenue_eur_per_mw:
            total_da += r_da.revenue_eur_per_mw
            da_days += 1
            daily.append(("DA", r_da))
        else:
            total_id += r_id.revenue_eur_per_mw
            id_days += 1
            daily.append(("ID", r_id))

    return {
        "annual_da": total_da,
        "annual_id": total_id,
        "annual_total": total_da + total_id,
        "da_day_count": da_days,
        "id_day_count": id_days,
        "daily_results": daily,
    }
