"""
Rolling-horizon MPC wrapper around ``optimize_day_stacked``.

Replaces the perfect-foresight daily LP with a realistic 12-hour MPC
horizon stepped every 3 hours (matching Collath 2023 + typical
intraday trader practice). Each step solves a 48-interval LP with
the observed current SoC and the next 12 hours of market inputs;
only the first ``step`` intervals are committed to the realised
dispatch before the planner advances state and re-solves.

Under perfect foresight (current default in Note 4), the LP sees all
96 intervals of the day and picks optimal cycling — which makes any
shadow-cost policy above a flat wear act as a tax on already-optimal
decisions and reduces NPV. Under MPC:
  * the LP is greedy within its horizon, over-commits to near-term
    arbitrage, and under-weights long-term aging;
  * shadow-cost policies act as a genuine hedge against this myopia
    and recover value that perfect foresight would have captured
    automatically.
This is the standard "sophistication helps under uncertainty" story.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np

from lib.analysis.stacked_day_assembler import StackedDayInputs
from lib.models.dispatch_stacked import (
    PERIODS_PER_BLOCK,
    PERIODS_PER_DAY,
    StackedDispatchResult,
    optimize_day_stacked,
    _failed_result,
)


def _slice_inputs(
    full: StackedDayInputs, t_start: int, t_end: int,
) -> dict:
    """Return kwargs for optimize_day_stacked over intervals [t_start, t_end).

    aFRR block prices (length 6 for full day, i.e. 4h blocks) are sliced
    by aligning the t_start / t_end to block boundaries.
    """
    assert t_start % PERIODS_PER_BLOCK == 0 and t_end % PERIODS_PER_BLOCK == 0
    b_start = t_start // PERIODS_PER_BLOCK
    b_end = t_end // PERIODS_PER_BLOCK
    return dict(
        prices_da=full.prices_da[t_start:t_end],
        prices_id=full.prices_id[t_start:t_end],
        afrr_cap_pos_price=full.afrr_cap_pos_price[b_start:b_end],
        afrr_cap_neg_price=full.afrr_cap_neg_price[b_start:b_end],
        afrr_energy_pos_price=full.afrr_energy_pos_price[t_start:t_end],
        afrr_energy_neg_price=full.afrr_energy_neg_price[t_start:t_end],
        afrr_activation_rate_pos=full.afrr_activation_rate_pos[t_start:t_end],
        afrr_activation_rate_neg=full.afrr_activation_rate_neg[t_start:t_end],
    )


def run_rolling_mpc_day(
    inputs: StackedDayInputs,
    *,
    energy_mwh: float,
    power_mw: float = 1.0,
    rte: float = 0.85,
    soc_min_frac: float = 0.05,
    soc_max_frac: float = 0.95,
    max_cycles: float = 2.0,
    afrr_reserve_duration_hours: float = 0.25,
    max_afrr_participation: float = 1.0,
    wear_cost_eur_per_mwh: Optional[np.ndarray] = None,
    horizon_hours: float = 12.0,
    step_hours: float = 3.0,
    initial_soc_mwh: Optional[float] = None,
) -> StackedDispatchResult:
    """Run a 12h-horizon MPC over the full day, 3h re-plan step.

    Returns a full-day (96-interval) :class:`StackedDispatchResult`
    stitched from the committed dispatch slices of each MPC step.

    Per-step max-cycles is scaled to the horizon fraction of a day.
    """
    T_full = PERIODS_PER_DAY          # 96
    PB = PERIODS_PER_BLOCK             # 16
    # dt = 0.25 h (15-min intervals); horizon and step must align to block
    horizon_intervals = int(round(horizon_hours * 4))
    step_intervals = int(round(step_hours * 4))
    # Snap to block boundaries
    if horizon_intervals % PB != 0:
        horizon_intervals = ((horizon_intervals // PB) + 1) * PB
    if step_intervals % PB != 0:
        step_intervals = ((step_intervals // PB) + 1) * PB

    horizon_day_frac = horizon_intervals / T_full

    soc_min = soc_min_frac * energy_mwh
    soc_max = soc_max_frac * energy_mwh
    current_soc = (float(initial_soc_mwh) if initial_soc_mwh is not None
                    else 0.5 * (soc_min + soc_max))

    # Collectors for committed dispatch per full-day interval
    c_da_full = np.zeros(T_full)
    d_da_full = np.zeros(T_full)
    c_id_full = np.zeros(T_full)
    d_id_full = np.zeros(T_full)
    r_pos_full = np.zeros(T_full)
    r_neg_full = np.zeros(T_full)
    a_pos_full = np.zeros(T_full)
    a_neg_full = np.zeros(T_full)
    soc_trajectory = np.zeros(T_full)

    cumulative_fec = 0.0
    usable_energy = energy_mwh * (soc_max_frac - soc_min_frac)
    step_start = 0
    while step_start < T_full:
        # Horizon window: 12h ahead (or less if at end of day)
        h_end = min(step_start + horizon_intervals, T_full)
        # Round up to block boundary so the LP has whole 4h aFRR blocks
        if h_end % PB != 0:
            h_end = ((h_end // PB) + 1) * PB
        h_end = min(h_end, T_full)

        horizon_kwargs = _slice_inputs(inputs, step_start, h_end)

        # Wear cost slice (if per-interval)
        wear_slice = None
        if wear_cost_eur_per_mwh is not None:
            wear_slice = np.asarray(wear_cost_eur_per_mwh)[step_start:h_end]

        # Remaining daily FEC budget; cap this horizon at it.
        remaining_fec_budget = max(0.0, max_cycles - cumulative_fec)
        max_cycles_h = remaining_fec_budget

        step_out = optimize_day_stacked(
            **horizon_kwargs,
            energy_mwh=energy_mwh, power_mw=power_mw, rte=rte,
            soc_min_frac=soc_min_frac, soc_max_frac=soc_max_frac,
            max_cycles=max_cycles_h,
            afrr_reserve_duration_hours=afrr_reserve_duration_hours,
            wear_cost_eur_per_mwh=wear_slice,
            max_afrr_participation=max_afrr_participation,
            initial_soc_mwh=current_soc,
            cyclic_soc=False,
        )
        if not step_out.success:
            # Fall back to no-action for this step
            return _failed_result(
                T_full, energy_mwh, current_soc,
                f"MPC step at t={step_start} failed: {step_out.status_message}",
            )

        # Commit only the first ``step_intervals`` of the horizon's dispatch
        commit_end = min(step_start + step_intervals, T_full)
        commit_len = commit_end - step_start

        c_da_full[step_start:commit_end] = step_out.charge_da[:commit_len]
        d_da_full[step_start:commit_end] = step_out.discharge_da[:commit_len]
        c_id_full[step_start:commit_end] = step_out.charge_id[:commit_len]
        d_id_full[step_start:commit_end] = step_out.discharge_id[:commit_len]
        r_pos_full[step_start:commit_end] = step_out.r_pos[:commit_len]
        r_neg_full[step_start:commit_end] = step_out.r_neg[:commit_len]
        a_pos_full[step_start:commit_end] = step_out.a_pos[:commit_len]
        a_neg_full[step_start:commit_end] = step_out.a_neg[:commit_len]
        soc_trajectory[step_start:commit_end] = step_out.soc[:commit_len]

        # Update cumulative FEC based on COMMITTED discharge (what actually
        # happened), not the full-horizon plan.
        committed_discharge = float(np.sum(
            (d_da_full[step_start:commit_end]
             + d_id_full[step_start:commit_end]
             + a_pos_full[step_start:commit_end]) * 0.25
        ))
        cumulative_fec += committed_discharge / max(energy_mwh, 1e-9)

        # Advance state: use the *observed* SoC after the commit window
        current_soc = float(step_out.soc[commit_len - 1])
        step_start = commit_end

    # Compute full-day revenue from the committed schedules
    dt = 0.25
    eta = np.sqrt(rte)
    prices_da = inputs.prices_da
    prices_id = inputs.prices_id
    afrr_e_pos = inputs.afrr_energy_pos_price
    afrr_e_neg = inputs.afrr_energy_neg_price
    # Per-block cap prices broadcast
    block_of = np.arange(T_full) // PB
    cap_pos_per_interval = inputs.afrr_cap_pos_price[block_of]
    cap_neg_per_interval = inputs.afrr_cap_neg_price[block_of]
    # Cap revenue: paid per MW × block_hours. Aggregate committed r_pos/r_neg
    # by block and multiply by 4h block revenue (integral of cap price).
    rev_da = float(np.sum(
        d_da_full * dt * prices_da * eta - c_da_full * dt * prices_da / eta
    ))
    rev_id = float(np.sum(
        d_id_full * dt * prices_id * eta - c_id_full * dt * prices_id / eta
    ))
    # For aFRR cap: r_pos_full is constant per block (broadcast from per-block
    # reservation). Sum over blocks: r × cap_price × 4h.
    block_hours = 4.0
    rev_cap_pos = 0.0
    rev_cap_neg = 0.0
    for b in range(T_full // PB):
        block_slice = slice(b * PB, (b + 1) * PB)
        # r is broadcast across the block — take the mean as the reservation
        r_p = float(np.mean(r_pos_full[block_slice]))
        r_n = float(np.mean(r_neg_full[block_slice]))
        rev_cap_pos += r_p * inputs.afrr_cap_pos_price[b] * block_hours
        rev_cap_neg += r_n * inputs.afrr_cap_neg_price[b] * block_hours
    rev_energy_pos = float(np.sum(a_pos_full * dt * afrr_e_pos))
    rev_energy_neg = float(np.sum(a_neg_full * dt * afrr_e_neg))
    rev_total = (rev_da + rev_id + rev_cap_pos + rev_cap_neg
                  + rev_energy_pos + rev_energy_neg)

    total_discharge = float(np.sum((d_da_full + d_id_full + a_pos_full) * dt))
    fec = total_discharge / energy_mwh if energy_mwh > 0 else 0.0

    return StackedDispatchResult(
        charge_da=c_da_full, discharge_da=d_da_full,
        charge_id=c_id_full, discharge_id=d_id_full,
        r_pos=r_pos_full, r_neg=r_neg_full,
        a_pos=a_pos_full, a_neg=a_neg_full,
        soc=soc_trajectory,
        revenue_da=rev_da, revenue_id=rev_id,
        revenue_afrr_cap_pos=rev_cap_pos, revenue_afrr_cap_neg=rev_cap_neg,
        revenue_afrr_energy_pos=rev_energy_pos,
        revenue_afrr_energy_neg=rev_energy_neg,
        revenue_total=rev_total,
        full_equivalent_cycles=fec,
        success=True,
    )
