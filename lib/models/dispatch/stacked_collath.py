"""
L7 stacked-market LP: piecewise-linear calendar+cyclic aging in objective
across DA + ID + aFRR.

Fork of :func:`lib.models.dispatch.stacked.optimize_day_stacked` extended
with two L7 cost components:

* **Calendar**: per-timestep penalty piecewise-linear in SoC[t]. Encoded
  as ``f_cal[t] >= a_i × SoC[t] + b_i`` for each segment i (upper-envelope
  LP relaxation; over-estimates fade in concave high-SoC regions).
* **Cyclic**: per-4h-block penalty piecewise-linear in throughput
  (separately for charging and discharging halves). Encoded as
  ``f_cyc_char[h] >= a_j × thru_char[h] + b_j``. Throughput in window h
  is the sum of charge (or discharge) flows from DA + ID + aFRR
  activation channels.

Coefficients are loaded from the eve_lf280k calibration in
``scripts/collath_benchmark/collath_eve_lf280k_coefficients.npz``. SoH-
dependent breakpoints are NOT yet supported — uses SoH=1 calibration
throughout, so accuracy degrades for aged cells. Acceptable for the
note's L7 supplement; full SoH-grid calibration is a follow-up.
"""
from __future__ import annotations

from math import sqrt
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from lib.models.dispatch.collath import load_collath_coefficients
from lib.models.dispatch.piecewise_common import segment_lines
from lib.models.dispatch.stacked import (
    BLOCKS_PER_DAY,
    DT_HOURS_DEFAULT,
    PERIODS_PER_BLOCK,
    PERIODS_PER_DAY,
    StackedDispatchResult,
    _failed_result,
)


def optimize_day_stacked_collath(
    prices_da: np.ndarray,
    prices_id: np.ndarray,
    afrr_cap_pos_price: np.ndarray,
    afrr_cap_neg_price: np.ndarray,
    afrr_energy_pos_price: np.ndarray,
    afrr_energy_neg_price: np.ndarray,
    afrr_activation_rate_pos: np.ndarray,
    afrr_activation_rate_neg: np.ndarray,
    energy_mwh: float,
    power_mw: float = 1.0,
    rte: float = 0.85,
    soc_min_frac: float = 0.05,
    soc_max_frac: float = 0.95,
    max_cycles: float = 2.0,
    dt_hours: float = DT_HOURS_DEFAULT,
    afrr_reserve_duration_hours: float = 0.25,
    max_afrr_participation: float = 1.0,
    initial_soc_mwh: float | None = None,
    cyclic_soc: bool = True,
    final_soc_target_mwh: float | None = None,
    capex_eur_per_mwh: float = 100_000.0,
    warranty_floor: float = 0.80,
    cycle_window_hours: float = 4.0,
    coefficient_path: Path | str | None = None,
    calendar_scale: float = 1.0,
    cyclic_scale: float = 1.0,
) -> StackedDispatchResult:
    """Same signature as :func:`optimize_day_stacked` minus
    ``wear_cost_eur_per_mwh`` (replaced by L7 in-objective terms),
    plus L7 calibration / scaling parameters."""
    T = len(prices_da)
    PB = PERIODS_PER_BLOCK
    if T % PB != 0:
        return _failed_result(
            T, energy_mwh, 0.5 * energy_mwh,
            f"prices_da length {T} not a multiple of PERIODS_PER_BLOCK={PB}",
        )
    B = T // PB
    eta = sqrt(rte)

    soc_min = soc_min_frac * energy_mwh
    soc_max = soc_max_frac * energy_mwh
    soc_init = (
        float(initial_soc_mwh) if initial_soc_mwh is not None
        else 0.5 * (soc_min + soc_max)
    )
    soc_init = float(np.clip(soc_init, soc_min, soc_max))
    usable_energy_mwh = max(energy_mwh * (soc_max_frac - soc_min_frac), 1e-9)

    # Load + prepare L7 piecewise-linear coefficients
    soc_brk_unit, cal_fade_yr, thr_brk_unit, cyc_fade_window = load_collath_coefficients(
        coefficient_path
    )
    cal_delta_yr = cal_fade_yr - cal_fade_yr[0]    # subtract SoC=0 baseline
    timesteps_per_year = 8760.0 / dt_hours
    cal_delta_per_step = cal_delta_yr / timesteps_per_year
    cost_per_unit_fade = (
        energy_mwh * capex_eur_per_mwh / max(1.0 - warranty_floor, 1e-6)
    )
    soc_brk_mwh = soc_brk_unit * energy_mwh
    a_cal, b_cal = segment_lines(soc_brk_mwh, cal_delta_per_step)
    n_seg_cal = len(a_cal)

    thr_brk_mwh = thr_brk_unit * energy_mwh
    a_cyc, b_cyc = segment_lines(thr_brk_mwh, cyc_fade_window)
    n_seg_cyc = len(a_cyc)

    # Cycle window structure: each window covers 4h = 16 timesteps = 1 block
    steps_per_window = int(round(cycle_window_hours / dt_hours))
    if T % steps_per_window != 0:
        return _failed_result(
            T, energy_mwh, soc_init,
            f"T={T} not divisible by cycle window steps {steps_per_window}",
        )
    n_windows = T // steps_per_window

    # --- Variable layout ----------------------------------------------------
    # Existing dispatch vars (4T + 2B):
    #   0..T-1            c_da
    #   T..2T-1           d_da
    #   2T..3T-1          c_id
    #   3T..4T-1          d_id
    #   4T..4T+B-1        r_pos block
    #   4T+B..4T+2B-1     r_neg block
    # New L7 vars (T + 4 × n_windows):
    #   4T+2B..5T+2B-1    f_cal[t]            calendar fade per step
    #   5T+2B+0..n_w-1    thru_char[h]        sum charge in window h
    #   ...               thru_dis[h]
    #   ...               f_cyc_char[h]
    #   ...               f_cyc_dis[h]
    base = 4 * T + 2 * B
    n_vars = base + T + 4 * n_windows

    idx_c_da = slice(0, T)
    idx_d_da = slice(T, 2 * T)
    idx_c_id = slice(2 * T, 3 * T)
    idx_d_id = slice(3 * T, 4 * T)
    idx_r_pos_block = slice(4 * T, 4 * T + B)
    idx_r_neg_block = slice(4 * T + B, 4 * T + 2 * B)
    idx_fcal = slice(base, base + T)
    idx_thr_c = slice(base + T, base + T + n_windows)
    idx_thr_d = slice(base + T + n_windows, base + T + 2 * n_windows)
    idx_fcyc_c = slice(base + T + 2 * n_windows, base + T + 3 * n_windows)
    idx_fcyc_d = slice(base + T + 3 * n_windows, base + T + 4 * n_windows)

    block_of = np.arange(T) // PB
    window_of = np.arange(T) // steps_per_window

    # --- Objective ---------------------------------------------------------
    c = np.zeros(n_vars)
    c[idx_c_da] = dt_hours * (prices_da / eta)
    c[idx_d_da] = -dt_hours * (prices_da * eta)
    c[idx_c_id] = dt_hours * (prices_id / eta)
    c[idx_d_id] = -dt_hours * (prices_id * eta)
    block_hours = PB * dt_hours
    for b in range(B):
        c[idx_r_pos_block.start + b] = -afrr_cap_pos_price[b] * block_hours
        c[idx_r_neg_block.start + b] = -afrr_cap_neg_price[b] * block_hours
        lo = b * PB
        hi = (b + 1) * PB
        coeff_pos = (
            afrr_energy_pos_price[lo:hi] * afrr_activation_rate_pos[lo:hi]
        ).sum() * dt_hours
        coeff_neg = (
            afrr_energy_neg_price[lo:hi] * afrr_activation_rate_neg[lo:hi]
        ).sum() * dt_hours
        c[idx_r_pos_block.start + b] += -coeff_pos
        c[idx_r_neg_block.start + b] += -coeff_neg
    # L7 penalties
    c[idx_fcal] = cost_per_unit_fade * calendar_scale
    c[idx_fcyc_c] = cost_per_unit_fade * cyclic_scale
    c[idx_fcyc_d] = cost_per_unit_fade * cyclic_scale

    A_rows: list[np.ndarray] = []
    b_rows: list[float] = []

    # 1) Power bounds per interval (incl. aFRR reservation)
    for t in range(T):
        b = block_of[t]
        row = np.zeros(n_vars)
        row[idx_d_da.start + t] = 1.0
        row[idx_d_id.start + t] = 1.0
        row[idx_r_pos_block.start + b] = 1.0
        A_rows.append(row); b_rows.append(power_mw)

        row = np.zeros(n_vars)
        row[idx_c_da.start + t] = 1.0
        row[idx_c_id.start + t] = 1.0
        row[idx_r_neg_block.start + b] = 1.0
        A_rows.append(row); b_rows.append(power_mw)

    # 2) SoC dynamics (cumulative). Keep soc_row template for L7 calendar.
    soc_row_for_t = []
    for t in range(T):
        row = np.zeros(n_vars)
        for tau in range(t + 1):
            bb = block_of[tau]
            row[idx_c_da.start + tau] += eta * dt_hours
            row[idx_c_id.start + tau] += eta * dt_hours
            row[idx_d_da.start + tau] += -dt_hours / eta
            row[idx_d_id.start + tau] += -dt_hours / eta
            row[idx_r_neg_block.start + bb] += afrr_activation_rate_neg[tau] * eta * dt_hours
            row[idx_r_pos_block.start + bb] += -afrr_activation_rate_pos[tau] * dt_hours / eta
        soc_row_for_t.append(row)
        # SoC bounds
        A_rows.append(row.copy()); b_rows.append(soc_max - soc_init)
        A_rows.append(-row.copy()); b_rows.append(soc_init - soc_min)

    # 3) aFRR SoC reservation (same as base dispatch)
    reserve_h = max(0.0, afrr_reserve_duration_hours)
    if reserve_h > 0:
        for t in range(T):
            b = block_of[t]
            soc_row = soc_row_for_t[t]
            row = -soc_row.copy()
            row[idx_r_pos_block.start + b] += reserve_h / eta
            A_rows.append(row); b_rows.append(soc_init - soc_min)
            row = soc_row.copy()
            row[idx_r_neg_block.start + b] += reserve_h * eta
            A_rows.append(row); b_rows.append(soc_max - soc_init)

    # 4) Cycle cap
    cycle_row = np.zeros(n_vars)
    cycle_row[idx_d_da] = dt_hours
    cycle_row[idx_d_id] = dt_hours
    for b in range(B):
        lo = b * PB
        hi = (b + 1) * PB
        cycle_row[idx_r_pos_block.start + b] = afrr_activation_rate_pos[lo:hi].sum() * dt_hours
    A_rows.append(cycle_row); b_rows.append(max_cycles * usable_energy_mwh)

    # 5) End-of-horizon SoC constraint
    if cyclic_soc or final_soc_target_mwh is not None:
        end_row = soc_row_for_t[-1]
        if cyclic_soc and final_soc_target_mwh is None:
            tol = 0.1 * energy_mwh
            A_rows.append(end_row.copy()); b_rows.append(tol)
            A_rows.append(-end_row.copy()); b_rows.append(tol)
        else:
            delta = float(final_soc_target_mwh) - soc_init
            tol = 0.02 * energy_mwh
            A_rows.append(end_row.copy()); b_rows.append(delta + tol)
            A_rows.append(-end_row.copy()); b_rows.append(-delta + tol)

    # 6) L7 CALENDAR: f_cal[t] >= a_i × SoC[t] + b_i for each segment
    #    SoC[t] = soc_init + soc_row_for_t[t] · x
    #    → -f_cal[t] + a_i × (soc_row · x) <= -(b_i - a_i × soc_init)
    if calendar_scale > 0:
        for t in range(T):
            soc_row = soc_row_for_t[t]
            for i in range(n_seg_cal):
                row = a_cal[i] * soc_row.copy()
                row[idx_fcal.start + t] = -1.0
                A_rows.append(row)
                b_rows.append(-(b_cal[i] - a_cal[i] * soc_init))

    # 7) L7 CYCLIC throughput accumulators per window
    #    thru_char[h] = sum over t in window h of (c_da[t] + c_id[t] + a_neg[t]) × dt
    #    a_neg[t] = α_neg[t] × r_neg[block(t)]
    #    Similarly thru_dis[h] = (d_da + d_id + a_pos) × dt
    if cyclic_scale > 0:
        for h in range(n_windows):
            t_start = h * steps_per_window
            t_end = min(t_start + steps_per_window, T)
            for sign in (1, -1):
                row_c = np.zeros(n_vars)
                row_c[idx_thr_c.start + h] = sign * 1.0
                for t in range(t_start, t_end):
                    bb = block_of[t]
                    row_c[idx_c_da.start + t] = -sign * dt_hours
                    row_c[idx_c_id.start + t] = -sign * dt_hours
                    row_c[idx_r_neg_block.start + bb] += -sign * (
                        afrr_activation_rate_neg[t] * dt_hours
                    )
                A_rows.append(row_c); b_rows.append(0.0)

                row_d = np.zeros(n_vars)
                row_d[idx_thr_d.start + h] = sign * 1.0
                for t in range(t_start, t_end):
                    bb = block_of[t]
                    row_d[idx_d_da.start + t] = -sign * dt_hours
                    row_d[idx_d_id.start + t] = -sign * dt_hours
                    row_d[idx_r_pos_block.start + bb] += -sign * (
                        afrr_activation_rate_pos[t] * dt_hours
                    )
                A_rows.append(row_d); b_rows.append(0.0)

        # 8) L7 cyclic piecewise-linear penalty
        for h in range(n_windows):
            for j in range(n_seg_cyc):
                row = np.zeros(n_vars)
                row[idx_fcyc_c.start + h] = -1.0
                row[idx_thr_c.start + h] = a_cyc[j]
                A_rows.append(row); b_rows.append(-b_cyc[j])

                row = np.zeros(n_vars)
                row[idx_fcyc_d.start + h] = -1.0
                row[idx_thr_d.start + h] = a_cyc[j]
                A_rows.append(row); b_rows.append(-b_cyc[j])

    A_ub = np.asarray(A_rows)
    b_ub = np.asarray(b_rows)

    r_cap = min(power_mw, max_afrr_participation * power_mw)
    bounds = (
        [(0.0, power_mw)] * (4 * T)              # c_da, d_da, c_id, d_id
        + [(0.0, r_cap)] * (2 * B)                # r_pos, r_neg block
        + [(0.0, None)] * T                       # f_cal
        + [(0.0, None)] * (4 * n_windows)         # thru_c, thru_d, f_cyc_c, f_cyc_d
    )

    result = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not result.success:
        return _failed_result(
            T, energy_mwh, soc_init,
            f"linprog failed: {result.message}",
        )

    x = result.x
    c_da = x[idx_c_da]
    d_da = x[idx_d_da]
    c_id = x[idx_c_id]
    d_id = x[idx_d_id]
    r_pos_block = x[idx_r_pos_block]
    r_neg_block = x[idx_r_neg_block]
    r_pos = r_pos_block[block_of]
    r_neg = r_neg_block[block_of]
    a_pos = afrr_activation_rate_pos * r_pos
    a_neg = afrr_activation_rate_neg * r_neg

    soc_changes = (c_da + c_id + a_neg) * eta * dt_hours - (d_da + d_id + a_pos) / eta * dt_hours
    soc = soc_init + np.cumsum(soc_changes)

    revenue_da = float(np.sum(d_da * dt_hours * prices_da * eta - c_da * dt_hours * prices_da / eta))
    revenue_id = float(np.sum(d_id * dt_hours * prices_id * eta - c_id * dt_hours * prices_id / eta))
    revenue_afrr_cap_pos = float(np.sum(r_pos_block * afrr_cap_pos_price) * block_hours)
    revenue_afrr_cap_neg = float(np.sum(r_neg_block * afrr_cap_neg_price) * block_hours)
    revenue_afrr_energy_pos = float(np.sum(a_pos * dt_hours * afrr_energy_pos_price))
    revenue_afrr_energy_neg = float(np.sum(a_neg * dt_hours * afrr_energy_neg_price))
    revenue_total = (
        revenue_da + revenue_id
        + revenue_afrr_cap_pos + revenue_afrr_cap_neg
        + revenue_afrr_energy_pos + revenue_afrr_energy_neg
    )

    total_discharge_mwh = float(np.sum((d_da + d_id + a_pos) * dt_hours))
    fec = total_discharge_mwh / energy_mwh

    return StackedDispatchResult(
        charge_da=c_da, discharge_da=d_da,
        charge_id=c_id, discharge_id=d_id,
        r_pos=r_pos, r_neg=r_neg, a_pos=a_pos, a_neg=a_neg,
        soc=soc,
        revenue_da=revenue_da, revenue_id=revenue_id,
        revenue_afrr_cap_pos=revenue_afrr_cap_pos,
        revenue_afrr_cap_neg=revenue_afrr_cap_neg,
        revenue_afrr_energy_pos=revenue_afrr_energy_pos,
        revenue_afrr_energy_neg=revenue_afrr_energy_neg,
        revenue_total=revenue_total,
        full_equivalent_cycles=fec,
        success=True,
    )
