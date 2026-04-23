"""
Stacked-market LP dispatch — DA + ID + aFRR capacity + aFRR energy, joint.

Solves a single linear program for daily BESS dispatch across all four
German revenue streams. Replaces the single-market ``optimize_day`` (DA
only) and the DA+ID overlay ``run_dispatch_with_intraday_overlay_for_period``
when the analysis needs aFRR reservation and its conjugate wholesale
coupling (cf. Note 4 A2, "aFRR↔wholesale coupling" in
``lib/analysis/observed_revenue.py``).

Markets
-------
* **DA (day-ahead)** — hourly auction, prices held constant across the 4
  15-min slots of each hour.
* **ID (intraday)** — 15-min continuous + auction; we model as 15-min
  residual adjustments to DA energy scheduling.
* **aFRR capacity** — 4h blocks, reserved MW per direction (POS, NEG).
  Operator bids ``r_pos`` / ``r_neg`` and is paid
  ``cap_price × r × block_hours`` per block.
* **aFRR activation** — per-15-min energy delivery following TSO's
  historical activation ratio. Reservation ``r`` scaled by fraction
  ``α[t] ∈ [0, 1]`` yields delivered MW ``a[t] = α[t] × r``.

Key physics encoded
-------------------
1.  **Reservation blocks power** — bidding ``r_pos`` for a 4h block means
    across every 15-min of that block, ``d_da + d_id + r_pos ≤ P_max``,
    regardless of whether TSO activates. Likewise for ``r_neg`` and charge.
2.  **SoC reservation headroom** — to be able to deliver full ``r_pos``
    for ``reserve_hours`` continuously (default 0.25h ≈ one 15-min SLA
    horizon), SoC must stay above ``soc_min + r_pos × reserve_hours / η``.
    Symmetric for ``r_neg`` at the top. Parameterised so stricter (1h+)
    or looser (zero) policies can be tested.
3.  **Activations consume energy** — a_pos discharges, a_neg charges.
    Accounted in SoC dynamics and cycle cap.
4.  **Cycle cap spans everything** — FECs from DA + ID + aFRR activations
    must not exceed ``max_cycles × usable_energy``.
5.  **Conjugate coupling is captured** — when NEG activation prices are
    negative (operator pays to absorb), the LP trades off the direct cost
    against the higher SoC it enables, which later gets discharged at DA
    peaks. This is the "free fuel from aFRR NEG" channel that a standalone
    DA LP misses.

Inputs
------
Each 15-min input array has length 96; aFRR block inputs have length 6.
Prices in EUR/MWh (energy) or EUR/MW/h (capacity). See
``optimize_day_stacked`` for full signature.

Outputs
-------
Returns per-interval dispatch decisions, revenue split by stream, and
cycle usage. See ``StackedDispatchResult``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt

import numpy as np
from scipy.optimize import linprog


PERIODS_PER_DAY = 96
BLOCKS_PER_DAY = 6
PERIODS_PER_BLOCK = PERIODS_PER_DAY // BLOCKS_PER_DAY  # 16 × 15-min = 4h
DT_HOURS_DEFAULT = 0.25


@dataclass
class StackedDispatchResult:
    # Per-interval MW schedules (length 96)
    charge_da: np.ndarray
    discharge_da: np.ndarray
    charge_id: np.ndarray
    discharge_id: np.ndarray
    # Per-interval aFRR state (r broadcast from per-block, a = α × r)
    r_pos: np.ndarray
    r_neg: np.ndarray
    a_pos: np.ndarray
    a_neg: np.ndarray
    # SoC trajectory (length 96), MWh absolute
    soc: np.ndarray
    # Revenue split (EUR, full day)
    revenue_da: float
    revenue_id: float
    revenue_afrr_cap_pos: float
    revenue_afrr_cap_neg: float
    revenue_afrr_energy_pos: float
    revenue_afrr_energy_neg: float
    revenue_total: float
    # Diagnostics
    full_equivalent_cycles: float
    success: bool = True
    status_message: str = ""

    def revenue_by_stream(self) -> dict[str, float]:
        return {
            "da": self.revenue_da,
            "id": self.revenue_id,
            "afrr_cap_pos": self.revenue_afrr_cap_pos,
            "afrr_cap_neg": self.revenue_afrr_cap_neg,
            "afrr_energy_pos": self.revenue_afrr_energy_pos,
            "afrr_energy_neg": self.revenue_afrr_energy_neg,
            "total": self.revenue_total,
        }


def _failed_result(periods: int, energy_mwh: float, soc_init: float,
                   message: str) -> StackedDispatchResult:
    zeros = np.zeros(periods)
    return StackedDispatchResult(
        charge_da=zeros, discharge_da=zeros,
        charge_id=zeros, discharge_id=zeros,
        r_pos=zeros, r_neg=zeros,
        a_pos=zeros, a_neg=zeros,
        soc=np.full(periods, soc_init),
        revenue_da=0.0, revenue_id=0.0,
        revenue_afrr_cap_pos=0.0, revenue_afrr_cap_neg=0.0,
        revenue_afrr_energy_pos=0.0, revenue_afrr_energy_neg=0.0,
        revenue_total=0.0,
        full_equivalent_cycles=0.0,
        success=False,
        status_message=message,
    )


def optimize_day_stacked(
    prices_da: np.ndarray,
    prices_id: np.ndarray,
    afrr_cap_pos_price: np.ndarray,      # (6,) EUR/MW/h per 4h block
    afrr_cap_neg_price: np.ndarray,      # (6,)
    afrr_energy_pos_price: np.ndarray,   # (96,) EUR/MWh
    afrr_energy_neg_price: np.ndarray,   # (96,) typically negative
    afrr_activation_rate_pos: np.ndarray,  # (96,) α ∈ [0, 1]
    afrr_activation_rate_neg: np.ndarray,  # (96,)
    energy_mwh: float,
    power_mw: float = 1.0,
    rte: float = 0.85,
    soc_min_frac: float = 0.05,
    soc_max_frac: float = 0.95,
    max_cycles: float = 2.0,
    dt_hours: float = DT_HOURS_DEFAULT,
    afrr_reserve_duration_hours: float = 0.25,
    wear_cost_eur_per_mwh: np.ndarray | None = None,
    max_afrr_participation: float = 1.0,
) -> StackedDispatchResult:
    """
    Joint LP across DA + ID + aFRR cap + aFRR activation for one day.

    See module docstring for physics. Returns ``StackedDispatchResult``; if
    the LP is infeasible (misconfigured inputs) the result carries
    ``success=False`` and zero schedules.
    """
    T = PERIODS_PER_DAY
    B = BLOCKS_PER_DAY
    PB = PERIODS_PER_BLOCK
    eta = sqrt(rte)

    soc_min = soc_min_frac * energy_mwh
    soc_max = soc_max_frac * energy_mwh
    soc_init = 0.5 * (soc_min + soc_max)
    usable_energy_mwh = max(energy_mwh * (soc_max_frac - soc_min_frac), 1e-9)

    # Input shape checks (fail fast)
    for name, arr, expected in [
        ("prices_da", prices_da, T),
        ("prices_id", prices_id, T),
        ("afrr_energy_pos_price", afrr_energy_pos_price, T),
        ("afrr_energy_neg_price", afrr_energy_neg_price, T),
        ("afrr_activation_rate_pos", afrr_activation_rate_pos, T),
        ("afrr_activation_rate_neg", afrr_activation_rate_neg, T),
        ("afrr_cap_pos_price", afrr_cap_pos_price, B),
        ("afrr_cap_neg_price", afrr_cap_neg_price, B),
    ]:
        if len(arr) != expected:
            return _failed_result(
                T, energy_mwh, soc_init,
                f"{name} has length {len(arr)}, expected {expected}",
            )

    # Variable layout — single flat vector for linprog:
    #   0..T-1           : c_da      (charge DA, MW)
    #   T..2T-1          : d_da      (discharge DA, MW)
    #   2T..3T-1         : c_id
    #   3T..4T-1         : d_id
    #   4T..4T+B-1       : r_pos     (aFRR capacity POS per block, MW)
    #   4T+B..4T+2B-1    : r_neg
    n_vars = 4 * T + 2 * B

    idx_c_da = slice(0, T)
    idx_d_da = slice(T, 2 * T)
    idx_c_id = slice(2 * T, 3 * T)
    idx_d_id = slice(3 * T, 4 * T)
    idx_r_pos_block = slice(4 * T, 4 * T + B)
    idx_r_neg_block = slice(4 * T + B, 4 * T + 2 * B)

    # Map an interval t → its 4h block index b = t // PB
    block_of = np.arange(T) // PB

    # --- Objective (minimise negative revenue) ---------------------------
    c = np.zeros(n_vars)
    # DA revenue: discharge earns price, charge pays price/eta (eta loss)
    c[idx_c_da] = dt_hours * (prices_da / eta)                   # cost of charging at DA
    c[idx_d_da] = -dt_hours * (prices_da * eta)                  # reward for discharging to DA
    # ID same structure
    c[idx_c_id] = dt_hours * (prices_id / eta)
    c[idx_d_id] = -dt_hours * (prices_id * eta)
    # aFRR cap: paid per MW per block hour. Block hours = PB * dt = 4 (by default).
    block_hours = PB * dt_hours
    for b in range(B):
        c[idx_r_pos_block.start + b] = -afrr_cap_pos_price[b] * block_hours
        c[idx_r_neg_block.start + b] = -afrr_cap_neg_price[b] * block_hours
    # aFRR energy: activation a_pos[t] = r_pos[block(t)] × α_pos[t]; revenue
    # at energy price e_pos[t]. Total POS energy rev = Σ_t e_pos[t] × r_pos[b(t)] × α_pos[t] × dt
    # → objective contribution on r_pos per block:
    #        -Σ_{t in block} e_pos[t] × α_pos[t] × dt
    # Similarly NEG — but operator PAYS (neg price): the negative price times
    # positive α and positive r produces negative revenue already, so LP sees
    # correctly that large r_neg at highly-negative prices is costly.
    for b in range(B):
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

    # Wear / shadow cost: per-interval EUR per MWh of throughput
    if wear_cost_eur_per_mwh is not None:
        if len(wear_cost_eur_per_mwh) != T:
            return _failed_result(
                T, energy_mwh, soc_init,
                f"wear_cost_eur_per_mwh length {len(wear_cost_eur_per_mwh)}, expected {T}",
            )
        wear = np.asarray(wear_cost_eur_per_mwh)
        # Wear hits both charge and discharge throughput.
        c[idx_c_da] += dt_hours * wear
        c[idx_d_da] += dt_hours * wear
        c[idx_c_id] += dt_hours * wear
        c[idx_d_id] += dt_hours * wear

    # --- Inequality constraints (A_ub × x ≤ b_ub) ------------------------
    A_rows: list[np.ndarray] = []
    b_rows: list[float] = []

    # 1) Power bounds per interval: reservation blocks power regardless of
    #    activation — must be able to deliver full reservation if called.
    #    Discharge side:  d_da[t] + d_id[t] + r_pos[b(t)] ≤ P_max
    #    Charge side:     c_da[t] + c_id[t] + r_neg[b(t)] ≤ P_max
    for t in range(T):
        b = block_of[t]
        row = np.zeros(n_vars)
        row[idx_d_da.start + t] = 1.0
        row[idx_d_id.start + t] = 1.0
        row[idx_r_pos_block.start + b] = 1.0
        A_rows.append(row)
        b_rows.append(power_mw)

        row = np.zeros(n_vars)
        row[idx_c_da.start + t] = 1.0
        row[idx_c_id.start + t] = 1.0
        row[idx_r_neg_block.start + b] = 1.0
        A_rows.append(row)
        b_rows.append(power_mw)

    # 2) SoC dynamics at every interval:
    #    soc[t] = soc_init + Σ_{τ≤t} ((c_da+c_id+a_neg)×η - (d_da+d_id+a_pos)/η) × dt
    #    where a_pos[τ] = α_pos[τ] × r_pos[b(τ)],  a_neg[τ] = α_neg[τ] × r_neg[b(τ)]
    #    soc_min ≤ soc[t] ≤ soc_max  →  Δ_cum[t] ∈ [soc_min − soc_init, soc_max − soc_init]
    for t in range(T):
        row = np.zeros(n_vars)
        for tau in range(t + 1):
            b = block_of[tau]
            row[idx_c_da.start + tau] += eta * dt_hours
            row[idx_c_id.start + tau] += eta * dt_hours
            row[idx_d_da.start + tau] += -dt_hours / eta
            row[idx_d_id.start + tau] += -dt_hours / eta
            # Activation contributions
            row[idx_r_neg_block.start + b] += afrr_activation_rate_neg[tau] * eta * dt_hours
            row[idx_r_pos_block.start + b] += -afrr_activation_rate_pos[tau] * dt_hours / eta

        # Upper SoC: row × x ≤ soc_max − soc_init
        A_rows.append(row.copy())
        b_rows.append(soc_max - soc_init)
        # Lower SoC: −row × x ≤ soc_init − soc_min
        A_rows.append(-row.copy())
        b_rows.append(soc_init - soc_min)

    # 3) SoC reservation headroom for aFRR: reserve enough energy to deliver
    #    full r_pos for ``afrr_reserve_duration_hours``; similarly top headroom
    #    for r_neg.
    #    At any interval t:
    #       soc[t] − r_pos[b(t)] × reserve_h / η ≥ soc_min
    #       soc[t] + r_neg[b(t)] × reserve_h × η ≤ soc_max
    #    Express soc[t] as (row defined above) + soc_init and substitute.
    reserve_h = max(0.0, afrr_reserve_duration_hours)
    if reserve_h > 0:
        for t in range(T):
            b = block_of[t]
            # Reconstruct the soc[t] row (accumulated flows to time t)
            soc_row = np.zeros(n_vars)
            for tau in range(t + 1):
                bb = block_of[tau]
                soc_row[idx_c_da.start + tau] += eta * dt_hours
                soc_row[idx_c_id.start + tau] += eta * dt_hours
                soc_row[idx_d_da.start + tau] += -dt_hours / eta
                soc_row[idx_d_id.start + tau] += -dt_hours / eta
                soc_row[idx_r_neg_block.start + bb] += afrr_activation_rate_neg[tau] * eta * dt_hours
                soc_row[idx_r_pos_block.start + bb] += -afrr_activation_rate_pos[tau] * dt_hours / eta

            # Constraint: −soc_row + (r_pos[b] × reserve_h / η) ≤ soc_init − soc_min
            row = -soc_row.copy()
            row[idx_r_pos_block.start + b] += reserve_h / eta
            A_rows.append(row)
            b_rows.append(soc_init - soc_min)

            # Constraint: soc_row + (r_neg[b] × reserve_h × η) ≤ soc_max − soc_init
            row = soc_row.copy()
            row[idx_r_neg_block.start + b] += reserve_h * eta
            A_rows.append(row)
            b_rows.append(soc_max - soc_init)

    # 4) Cycle cap: total discharge energy (DA + ID + pos activation) ≤ max_cycles × usable_energy
    cycle_row = np.zeros(n_vars)
    cycle_row[idx_d_da] = dt_hours
    cycle_row[idx_d_id] = dt_hours
    for b in range(B):
        lo = b * PB
        hi = (b + 1) * PB
        cycle_row[idx_r_pos_block.start + b] = afrr_activation_rate_pos[lo:hi].sum() * dt_hours
    A_rows.append(cycle_row)
    b_rows.append(max_cycles * usable_energy_mwh)

    # 5) End-of-day SoC ≈ start (loose tolerance, keeps LP well-posed)
    end_row = np.zeros(n_vars)
    for tau in range(T):
        bb = block_of[tau]
        end_row[idx_c_da.start + tau] += eta * dt_hours
        end_row[idx_c_id.start + tau] += eta * dt_hours
        end_row[idx_d_da.start + tau] += -dt_hours / eta
        end_row[idx_d_id.start + tau] += -dt_hours / eta
        end_row[idx_r_neg_block.start + bb] += afrr_activation_rate_neg[tau] * eta * dt_hours
        end_row[idx_r_pos_block.start + bb] += -afrr_activation_rate_pos[tau] * dt_hours / eta
    tolerance_mwh = 0.1 * energy_mwh
    A_rows.append(end_row.copy())
    b_rows.append(tolerance_mwh)
    A_rows.append(-end_row.copy())
    b_rows.append(tolerance_mwh)

    A_ub = np.asarray(A_rows)
    b_ub = np.asarray(b_rows)

    # Bounds: all variables ≥ 0. Power vars ≤ power_mw. aFRR reservation
    # capped at ``max_afrr_participation × P_max`` — reflects real-world
    # bid-win rates, risk-averse placement, and operational withholding
    # that keep operators below the LP-unconstrained optimum. Default 1.0
    # gives the unconstrained upper bound; 0.5 approximates typical
    # German 2023-2025 bid competition (demand/offered ≈ 0.55).
    r_cap = min(power_mw, max_afrr_participation * power_mw)
    bounds = (
        [(0.0, power_mw)] * (4 * T)
        + [(0.0, r_cap)] * (2 * B)
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

    # Broadcast block-level reservations to per-interval
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
