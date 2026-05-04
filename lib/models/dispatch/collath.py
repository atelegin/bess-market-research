"""
L7 dispatch: Collath-style piecewise-linear calendar + cyclic aging IN OBJECTIVE.

Single-market one-price LP that ports the structural form of Collath et al.
(2023)'s `MpcOptArbitrageCalendarCyclicOnePrice` into pure scipy.linprog,
calibrated against our Note 3 eve_lf280k Wang+Naumann kernel rather than
their Sony LFP coefficients.

Why this exists
---------------
L1-L6 in our progressive stack are all SCALAR shadow-cost forms (per-
interval ``α(state) × |power|``). The bridge analysis showed gap 2:
scalar shadow forms cap at ~43% of Collath capture even at matched FEC,
because the LP cannot trade off cycle SHAPE (DoD, mean SoC, throughput
distribution) inside the optimisation. L7 closes that by putting the
piecewise-linear (cyclic, calendar) functions directly into the LP
objective.

Calendar: ``qloss_cal(mean_SoC[t])`` per timestep, piecewise-linear over
SoC breakpoints. Encoded as upper envelope (``f_cal[t] >= a_i × SoC[t] +
b_i for each segment i``). This is exact for convex parts and slightly
over-estimates fade in concave high-SoC regions (~10–30% local error
above SoC=0.7), making the LP more conservative there. Acceptable for
first pass; switch to MILP (SOS-2) if precision matters.

Cyclic: ``qloss_cyc(throughput_in_4h_window)`` per 4-hour bucket,
piecewise-linear over 28 throughput breakpoints. Convex on eve_lf280k
(verified), so the upper-envelope encoding is exact.

Both are calibrated with ``scripts/collath_benchmark/collath_calibrate_eve_lf280k.py``;
output stored in ``collath_eve_lf280k_coefficients.npz``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linprog


@dataclass
class CollathDispatchResult:
    power: np.ndarray            # MW signed (+ discharge, − charge)
    soc: np.ndarray              # MWh absolute
    revenue: float
    aging_cost_calendar: float
    aging_cost_cyclic: float
    profit: float
    fec: float
    success: bool
    message: str = ""


def load_collath_coefficients(
    path: Path | str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load piecewise-linear breakpoints calibrated against eve_lf280k.

    Returns ``(soc_brk, cal_fade_per_year, throughput_brk, cyc_fade_per_window)``.
    Fade values are FRACTIONAL SoH loss; calendar is per-year, cyclic is
    per 4-hour window.
    """
    if path is None:
        path = (Path(__file__).resolve().parents[2]
                / "scripts" / "collath_benchmark"
                / "collath_eve_lf280k_coefficients.npz")
    data = np.load(path)
    return (
        data["soc_breakpoints"], data["cal_fade"],
        data["throughput_breakpoints"], data["cyc_fade"],
    )


# segment_lines lifted to lib.models.dispatch.piecewise_common; import as the
# canonical shared helper to avoid duplication across Collath / Rainflow.
from lib.models.dispatch.piecewise_common import segment_lines as _segment_lines  # noqa: E402


def optimize_arbitrage_collath(
    prices: np.ndarray,            # (T,) EUR/MWh
    energy_mwh: float,
    power_mw: float = 1.0,
    rte: float = 0.90,
    soc_min_frac: float = 0.0,
    soc_max_frac: float = 1.0,
    initial_soc_mwh: float | None = None,
    dt_hours: float = 0.25,
    capex_eur_per_mwh: float = 100_000.0,
    warranty_floor: float = 0.80,
    cycle_window_hours: float = 4.0,
    coefficient_path: Path | str | None = None,
    aging_cost_scale: float = 1.0,
    calendar_scale: float = 1.0,
    cyclic_scale: float = 1.0,
) -> CollathDispatchResult:
    """One-price LP with L7 piecewise-linear calendar+cyclic aging in objective.

    Args mostly mirror :func:`lib.models.dispatch.arbitrage.optimize_arbitrage_horizon`.
    Additional:

    * ``capex_eur_per_mwh``: monetisation factor; the cost of one unit of
      fade fraction is ``capacity_mwh × CAPEX / (1 − warranty_floor)``,
      matching Collath's ``c_capacity / (1 − floor) × aging_cost``
      monetisation structure.
    * ``cycle_window_hours``: the bucket length over which cyclic
      throughput is measured, default 4 h (Collath default). Must
      divide ``T × dt_hours`` evenly.
    * ``coefficient_path``: where to load the calibrated piecewise-
      linear coefficients (defaults to ``collath_eve_lf280k_coefficients.npz``).
    * ``aging_cost_scale``: multiplier for both calendar and cyclic
      penalties — diagnostic knob to sweep aging-cost magnitude. Default
      1.0 = honest physics.
    """
    T = len(prices)
    eta = np.sqrt(rte)
    soc_min = soc_min_frac * energy_mwh
    soc_max = soc_max_frac * energy_mwh
    soc_init = (
        float(initial_soc_mwh) if initial_soc_mwh is not None
        else 0.5 * energy_mwh
    )
    soc_init = float(np.clip(soc_init, soc_min, soc_max))

    soc_brk, cal_fade_yr, thr_brk, cyc_fade_window = load_collath_coefficients(
        coefficient_path
    )

    # Subtract SoC=0 baseline from calendar curve — only the SoC-dependent
    # delta enters the LP (the constant baseline is a sunk cost and can't
    # be optimised away). Convert per-year fade to per-timestep fade.
    cal_delta_yr = cal_fade_yr - cal_fade_yr[0]
    timesteps_per_year = (8760.0 / dt_hours)
    cal_delta_per_step = cal_delta_yr / timesteps_per_year

    # Cost factor: one unit of fade fraction × capacity = €
    cost_per_unit_fade = (
        energy_mwh * capex_eur_per_mwh / max(1.0 - warranty_floor, 1e-6)
    ) * aging_cost_scale

    # Calendar penalty per-step is in SoC-fraction terms (soc[t] / energy_mwh).
    # Convert SoC breakpoints to absolute MWh for LP variable scaling.
    soc_brk_mwh = soc_brk * energy_mwh
    a_cal_step, b_cal_step = _segment_lines(soc_brk_mwh, cal_delta_per_step)
    n_seg_cal = len(a_cal_step)

    # Cyclic per 4h window. Throughput breakpoints in CSV are per-unit-
    # capacity; scale to absolute MWh for LP.
    thr_brk_mwh = thr_brk * energy_mwh
    a_cyc, b_cyc = _segment_lines(thr_brk_mwh, cyc_fade_window)
    n_seg_cyc = len(a_cyc)

    # Cycle window structure
    steps_per_window = int(round(cycle_window_hours / dt_hours))
    if T % steps_per_window != 0:
        # Last partial window is allowed; adjust upward
        n_windows = (T + steps_per_window - 1) // steps_per_window
    else:
        n_windows = T // steps_per_window

    # Decision variables (scipy linprog: minimise c^T x):
    #   charge[t] (T)         non-neg, ≤ P_max
    #   discharge[t] (T)      non-neg, ≤ P_max
    #   f_cal[t] (T)          non-neg, calendar fade per step at this SoC
    #   thru_char[h] (n_w)    non-neg, sum of charge over window h
    #   thru_dis[h] (n_w)     non-neg, sum of discharge over window h
    #   f_cyc_char[h] (n_w)   non-neg
    #   f_cyc_dis[h] (n_w)    non-neg
    n_vars = T + T + T + n_windows * 4
    idx_c = slice(0, T)
    idx_d = slice(T, 2 * T)
    idx_fcal = slice(2 * T, 3 * T)
    idx_thr_c = slice(3 * T, 3 * T + n_windows)
    idx_thr_d = slice(3 * T + n_windows, 3 * T + 2 * n_windows)
    idx_fcyc_c = slice(3 * T + 2 * n_windows, 3 * T + 3 * n_windows)
    idx_fcyc_d = slice(3 * T + 3 * n_windows, 3 * T + 4 * n_windows)

    c = np.zeros(n_vars)
    c[idx_c] = dt_hours * (prices / eta)        # buying cost
    c[idx_d] = -dt_hours * (prices * eta)        # selling revenue (negative)
    c[idx_fcal] = cost_per_unit_fade * calendar_scale       # calendar fade penalty
    c[idx_fcyc_c] = cost_per_unit_fade * cyclic_scale       # cyclic charge penalty
    c[idx_fcyc_d] = cost_per_unit_fade * cyclic_scale       # cyclic discharge penalty
    # thru_char and thru_dis don't directly enter objective (only via f_cyc)

    A_rows: list[np.ndarray] = []
    b_rows: list[float] = []

    # SoC dynamics: SoC[t] = soc_init + sum_{tau<=t}(c[tau] η − d[tau]/η) dt
    # bounds: soc_min ≤ SoC[t] ≤ soc_max
    soc_coef_for_t = np.zeros((T, n_vars))
    for t in range(T):
        for tau in range(t + 1):
            soc_coef_for_t[t, idx_c.start + tau] = eta * dt_hours
            soc_coef_for_t[t, idx_d.start + tau] = -dt_hours / eta
        # SoC[t] ≤ soc_max → cumulative ≤ soc_max - soc_init
        A_rows.append(soc_coef_for_t[t].copy())
        b_rows.append(soc_max - soc_init)
        # SoC[t] ≥ soc_min → -cumulative ≤ soc_init - soc_min
        A_rows.append(-soc_coef_for_t[t].copy())
        b_rows.append(soc_init - soc_min)

    # Calendar: f_cal[t] ≥ a_i × SoC[t] + b_i  for each segment i
    # SoC[t] = soc_init + cumulative(c, d). Move SoC expression into constraint:
    # f_cal[t] - a_i × SoC[t] ≥ b_i
    # → -f_cal[t] + a_i × cumulative(c, d) ≤ -(b_i - a_i × soc_init)
    for t in range(T):
        for i in range(n_seg_cal):
            row = np.zeros(n_vars)
            row[idx_fcal.start + t] = -1.0
            row[:n_vars] += a_cal_step[i] * soc_coef_for_t[t]
            A_rows.append(row)
            b_rows.append(-(b_cal_step[i] - a_cal_step[i] * soc_init))

    # Cyclic throughput accumulators per window.
    # thru_char[h] = sum over t in window h of charge[t] × dt
    # thru_dis[h]  = sum over t in window h of discharge[t] × dt
    for h in range(n_windows):
        t_start = h * steps_per_window
        t_end = min(t_start + steps_per_window, T)
        # equality constraint (handle as two ≤ rows)
        for sign in (1, -1):
            row_c = np.zeros(n_vars)
            row_c[idx_thr_c.start + h] = sign * 1.0
            for t in range(t_start, t_end):
                row_c[idx_c.start + t] = -sign * dt_hours
            A_rows.append(row_c)
            b_rows.append(0.0)

            row_d = np.zeros(n_vars)
            row_d[idx_thr_d.start + h] = sign * 1.0
            for t in range(t_start, t_end):
                row_d[idx_d.start + t] = -sign * dt_hours
            A_rows.append(row_d)
            b_rows.append(0.0)

    # Cyclic piecewise-linear penalty:
    # f_cyc_char[h] ≥ a_j × thru_char[h] + b_j  for each segment j
    # → -f_cyc_char[h] + a_j × thru_char[h] ≤ -b_j
    for h in range(n_windows):
        for j in range(n_seg_cyc):
            row = np.zeros(n_vars)
            row[idx_fcyc_c.start + h] = -1.0
            row[idx_thr_c.start + h] = a_cyc[j]
            A_rows.append(row)
            b_rows.append(-b_cyc[j])

            row = np.zeros(n_vars)
            row[idx_fcyc_d.start + h] = -1.0
            row[idx_thr_d.start + h] = a_cyc[j]
            A_rows.append(row)
            b_rows.append(-b_cyc[j])

    A_ub = np.asarray(A_rows)
    b_ub = np.asarray(b_rows)
    bounds = (
        [(0.0, power_mw)] * (2 * T)              # charge, discharge
        + [(0.0, None)] * T                       # f_cal
        + [(0.0, None)] * (4 * n_windows)         # thru_c, thru_d, f_cyc_c, f_cyc_d
    )

    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        zeros = np.zeros(T)
        return CollathDispatchResult(
            power=zeros, soc=np.full(T, soc_init),
            revenue=0.0, aging_cost_calendar=0.0, aging_cost_cyclic=0.0,
            profit=0.0, fec=0.0, success=False, message=res.message,
        )

    x = res.x
    charge = x[idx_c]
    discharge = x[idx_d]
    f_cal = x[idx_fcal]
    f_cyc_c = x[idx_fcyc_c]
    f_cyc_d = x[idx_fcyc_d]

    power = discharge - charge
    soc_changes = (charge * eta - discharge / eta) * dt_hours
    soc = soc_init + np.cumsum(soc_changes)
    revenue = float(np.sum(
        discharge * dt_hours * prices * eta
        - charge * dt_hours * prices / eta
    ))
    aging_cal = float(cost_per_unit_fade * calendar_scale * f_cal.sum())
    aging_cyc = float(cost_per_unit_fade * cyclic_scale * (f_cyc_c.sum() + f_cyc_d.sum()))
    return CollathDispatchResult(
        power=power, soc=soc,
        revenue=revenue,
        aging_cost_calendar=aging_cal,
        aging_cost_cyclic=aging_cyc,
        profit=revenue - aging_cal - aging_cyc,
        fec=float(np.sum(discharge * dt_hours)) / energy_mwh,
        success=True,
    )
