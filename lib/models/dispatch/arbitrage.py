"""
Single-price single-market dispatch for the Collath head-to-head benchmark.

This is a stripped-down LP/MPC for *intraday-continuous arbitrage only*
that mirrors the Collath et al. (2023) MPC framework's structure:

  * one price series (buy = sell at the same clearing price)
  * one decision power per interval (discharge positive, charge negative)
  * no aFRR, no DA, no two-channel decomposition
  * cyclic SoC NOT enforced (MPC propagates state explicitly)
  * round-trip efficiency on AC-side (Collath uses 0.90)
  * optional flat aging cost €/MWh of throughput
  * SoC bounds [0, energy] (Collath default — full nameplate accessible)

We reuse this same one-price LP for our perfect-foresight (full-day
horizon) and rolling-horizon-MPC (12h horizon, 30-min commit step,
matching Collath's SIM_TO_OPT_RATIO=2) variants. The four-way bar
chart in the Collath section uses these identically structured calls
to isolate the foresight gap from the model-structure gap.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


@dataclass
class ArbitrageDispatchResult:
    power: np.ndarray          # MW, signed: + discharge, − charge
    soc: np.ndarray            # MWh absolute
    revenue: float             # EUR
    aging_cost: float          # EUR (price × throughput × wear factor)
    profit: float              # revenue − aging_cost
    fec: float                 # full-equivalent cycles
    success: bool
    message: str = ""


def optimize_arbitrage_horizon(
    prices: np.ndarray,            # (T,) EUR/MWh
    energy_mwh: float,
    power_mw: float = 1.0,
    rte: float = 0.90,             # Collath default
    soc_min_frac: float = 0.0,
    soc_max_frac: float = 1.0,
    initial_soc_mwh: float | None = None,
    final_soc_target_mwh: float | None = None,
    cyclic_soc: bool = False,
    wear_cost_eur_per_mwh: "float | np.ndarray" = 0.0,
    max_cycles: float | None = None,
    dt_hours: float = 0.25,
) -> ArbitrageDispatchResult:
    """One-price, one-market arbitrage LP over a horizon of T intervals.

    Decision variables: charge[t], discharge[t] (each ≥ 0, ≤ P_max).
    Single round-trip efficiency η = sqrt(rte) on each side (delivers
    matches Collath's 0.90 RTE on AC-side).

    ``wear_cost_eur_per_mwh`` can be a scalar (uniform L3-style flat wear)
    or a length-T vector (L5 hour-varying ADP shadow, L6 physics-from-duty
    refined per-interval cost).
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

    n_vars = 2 * T
    idx_c = slice(0, T)
    idx_d = slice(T, 2 * T)

    wear_vec = np.broadcast_to(
        np.asarray(wear_cost_eur_per_mwh, dtype=float), (T,)
    ).copy()

    # Objective: minimise (-revenue + aging_cost). Charging at price[t]
    # costs price[t] × charge[t] × dt, discharging at price[t] earns
    # price[t] × discharge[t] × η × dt (efficiency on discharge).
    c = np.zeros(n_vars)
    c[idx_c] = dt_hours * (prices / eta)        # cost to charge
    c[idx_d] = -dt_hours * (prices * eta)        # reward for discharge
    if np.any(wear_vec > 0):
        c[idx_c] += dt_hours * wear_vec
        c[idx_d] += dt_hours * wear_vec

    A_rows: list[np.ndarray] = []
    b_rows: list[float] = []

    # SoC dynamics: cumulative ((c × η − d / η) × dt) × prefix-sum
    for t in range(T):
        row = np.zeros(n_vars)
        for tau in range(t + 1):
            row[idx_c.start + tau] += eta * dt_hours
            row[idx_d.start + tau] += -dt_hours / eta
        # Upper bound: cumulative Δ ≤ soc_max − soc_init
        A_rows.append(row.copy())
        b_rows.append(soc_max - soc_init)
        # Lower bound: cumulative Δ ≥ soc_min − soc_init
        A_rows.append(-row.copy())
        b_rows.append(soc_init - soc_min)

    # Cycle cap (optional). Throughput = sum of discharge × dt.
    if max_cycles is not None:
        cycle_row = np.zeros(n_vars)
        cycle_row[idx_d] = dt_hours
        A_rows.append(cycle_row)
        b_rows.append(max_cycles * energy_mwh)

    # Optional terminal constraint
    if cyclic_soc or final_soc_target_mwh is not None:
        end_row = np.zeros(n_vars)
        for tau in range(T):
            end_row[idx_c.start + tau] += eta * dt_hours
            end_row[idx_d.start + tau] += -dt_hours / eta
        if cyclic_soc and final_soc_target_mwh is None:
            tol = 0.05 * energy_mwh
            A_rows.append(end_row.copy()); b_rows.append(tol)
            A_rows.append(-end_row.copy()); b_rows.append(tol)
        else:
            delta = float(final_soc_target_mwh) - soc_init
            tol = 0.02 * energy_mwh
            A_rows.append(end_row.copy()); b_rows.append(delta + tol)
            A_rows.append(-end_row.copy()); b_rows.append(-delta + tol)

    A_ub = np.asarray(A_rows)
    b_ub = np.asarray(b_rows)
    bounds = [(0.0, power_mw)] * (2 * T)

    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        zeros = np.zeros(T)
        return ArbitrageDispatchResult(
            power=zeros, soc=np.full(T, soc_init),
            revenue=0.0, aging_cost=0.0, profit=0.0, fec=0.0,
            success=False, message=res.message,
        )

    x = res.x
    charge = x[idx_c]
    discharge = x[idx_d]
    power = discharge - charge   # signed
    soc_changes = (charge * eta - discharge / eta) * dt_hours
    soc = soc_init + np.cumsum(soc_changes)
    revenue = float(np.sum(
        discharge * dt_hours * prices * eta
        - charge * dt_hours * prices / eta
    ))
    aging_cost = float(np.sum(
        (charge + discharge) * dt_hours * wear_vec
    ))
    return ArbitrageDispatchResult(
        power=power, soc=soc,
        revenue=revenue, aging_cost=aging_cost,
        profit=revenue - aging_cost,
        fec=float(np.sum(discharge * dt_hours)) / energy_mwh,
        success=True,
    )


def run_arbitrage_rolling_mpc(
    prices: np.ndarray,            # (T_full,) all-day prices
    energy_mwh: float,
    power_mw: float = 1.0,
    rte: float = 0.90,
    soc_min_frac: float = 0.0,
    soc_max_frac: float = 1.0,
    initial_soc_mwh: float | None = None,
    wear_cost_eur_per_mwh: float = 0.0,
    horizon_intervals: int = 48,    # 12h × 4
    step_intervals: int = 2,         # 30 min (Collath SIM_TO_OPT_RATIO=2)
    dt_hours: float = 0.25,
) -> ArbitrageDispatchResult:
    """Roll a one-price LP over the full horizon with H-step lookahead and
    S-interval commit. Mirrors Collath's structure exactly."""
    T_full = len(prices)
    soc_init = (
        float(initial_soc_mwh) if initial_soc_mwh is not None
        else 0.5 * energy_mwh
    )
    current_soc = float(np.clip(
        soc_init, soc_min_frac * energy_mwh, soc_max_frac * energy_mwh,
    ))

    power_full = np.zeros(T_full)
    soc_full = np.zeros(T_full)
    revenue_total = 0.0
    aging_total = 0.0

    step_start = 0
    while step_start < T_full:
        h_end = min(step_start + horizon_intervals, T_full)
        prices_h = prices[step_start:h_end]
        out = optimize_arbitrage_horizon(
            prices=prices_h, energy_mwh=energy_mwh, power_mw=power_mw,
            rte=rte, soc_min_frac=soc_min_frac, soc_max_frac=soc_max_frac,
            initial_soc_mwh=current_soc, cyclic_soc=False,
            wear_cost_eur_per_mwh=wear_cost_eur_per_mwh,
            dt_hours=dt_hours,
        )
        if not out.success:
            return out
        commit_end = min(step_start + step_intervals, T_full)
        commit_len = commit_end - step_start
        power_full[step_start:commit_end] = out.power[:commit_len]
        soc_full[step_start:commit_end] = out.soc[:commit_len]
        # Compute committed-window revenue/aging on actual prices
        eta = np.sqrt(rte)
        for t in range(commit_len):
            p = power_full[step_start + t]
            pr = prices[step_start + t]
            if p > 0:        # discharge
                revenue_total += p * eta * pr * dt_hours
                aging_total += abs(p) * dt_hours * wear_cost_eur_per_mwh
            elif p < 0:      # charge
                revenue_total += p / eta * pr * dt_hours
                aging_total += abs(p) * dt_hours * wear_cost_eur_per_mwh
        current_soc = float(soc_full[commit_end - 1])
        step_start = commit_end

    discharge_total = float(np.sum(np.where(power_full > 0, power_full, 0))) * dt_hours
    return ArbitrageDispatchResult(
        power=power_full, soc=soc_full,
        revenue=revenue_total, aging_cost=aging_total,
        profit=revenue_total - aging_total,
        fec=discharge_total / energy_mwh if energy_mwh > 0 else 0.0,
        success=True,
    )
