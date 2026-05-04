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
from typing import Optional

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
    # Stage 2 shortfall diagnostics — populated only when called with
    # locked r_pos / r_neg (i.e. as Stage 2 of two-stage dispatch). Each
    # array has length T (96 for full-day). Energy in MWh; cost in EUR.
    shortfall_pos_mwh: Optional[np.ndarray] = None
    shortfall_neg_mwh: Optional[np.ndarray] = None
    shortfall_cost_eur: float = 0.0
    # Market-aware bid-vs-expected diagnostics. When ``bid_win_rate_lp``
    # < 1.0 the LP optimised for expected outcomes; r_pos / a_pos /
    # r_neg / a_neg above are the *bids* (full-clear values). The
    # *_expected fields are the LP's expected delivery (= bid ×
    # bid_win_rate_lp). Lifecycle SoH / FEC accounting uses the expected
    # values; cap revenue numbers above are also at expected. None when
    # the LP ran in 100 %-clearing-naive mode.
    bid_win_rate_lp_used: float = 1.0
    r_pos_expected: Optional[np.ndarray] = None
    r_neg_expected: Optional[np.ndarray] = None
    a_pos_expected: Optional[np.ndarray] = None
    a_neg_expected: Optional[np.ndarray] = None
    # ADR-002 MVP — auction-access cost charged on the bid commitment.
    # Already subtracted from ``revenue_total``; surfaced separately so
    # the lifecycle accounting / reconciliation can audit the deduction.
    bid_hurdle_cost_eur: float = 0.0

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


@dataclass
class TwoStageDispatchResult:
    """Result of :func:`optimize_day_two_stage` — wraps both stage outputs.

    The headline economics live in ``stage2`` (realised dispatch under
    locked aFRR commitments). ``stage1`` carries the forecast-driven
    commitment LP for diagnostics and the smoke-test invariant that
    Stage 1's ``r_pos`` does not trivially saturate at nameplate.
    """

    stage1: StackedDispatchResult       # forecast-driven DA-aFRR commit
    stage2: StackedDispatchResult       # realised LP with r_pos/r_neg locked

    @property
    def success(self) -> bool:
        return self.stage1.success and self.stage2.success

    @property
    def revenue_total(self) -> float:
        return self.stage2.revenue_total

    @property
    def stage1_r_pos_block(self) -> np.ndarray:
        return self.stage1.r_pos[::PERIODS_PER_BLOCK]

    @property
    def stage1_r_neg_block(self) -> np.ndarray:
        return self.stage1.r_neg[::PERIODS_PER_BLOCK]


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
        shortfall_pos_mwh=None,
        shortfall_neg_mwh=None,
        shortfall_cost_eur=0.0,
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
    afrr_wear_premium_eur_per_mwh: float = 0.0,
    max_afrr_participation: float = 1.0,
    # Two-stage Stage 2 inputs (ADR-001). When the caller passes locked
    # per-block reservations (typical use: Stage 2 of optimize_day_two_stage),
    # the LP's r_pos[b] / r_neg[b] are pinned to those values via equality
    # bounds, and the SoC reservation constraint becomes SOFT: shortfall
    # variables shortfall_pos[t], shortfall_neg[t] absorb infeasibility
    # when realised α × locked-r exceeds the SoC headroom that Stage 1's
    # forecast α implied. Each MWh of shortfall costs
    # ``mfrr_penalty_*_eur_per_mwh × α_*[t] × dt`` — i.e. the operator pays
    # mFRR-imbalance only on the activated fraction of the missed
    # reservation. DE 2024 mFRR imbalance ≈ €200–800/MWh; €500/MWh central.
    # In single-stage mode (locked params None), penalties are ignored and
    # the SoC reservation constraint is hard.
    r_pos_locked_per_block: np.ndarray | None = None,
    r_neg_locked_per_block: np.ndarray | None = None,
    mfrr_penalty_pos_eur_per_mwh: float = 500.0,
    mfrr_penalty_neg_eur_per_mwh: float = 500.0,
    # Market-aware bidding (ex-ante auction-clearing probability). When
    # < 1.0, the LP optimises *expected* revenue knowing only fraction
    # ``bid_win_rate_lp`` of each block's r commitment will clear. The
    # cap-revenue and aFRR-energy-revenue coefficients on r_pos / r_neg
    # are scaled by ``bid_win_rate_lp`` (since only the cleared fraction
    # is collected); SoC dynamics use expected drain (clearing-probability-
    # weighted activation); cycle-cap binds on expected throughput. Power
    # and SoC reservation-headroom constraints stay at full r — operator
    # must hold capacity for the full bid in case it clears (auction-
    # settles-before-delivery sequencing). When 1.0 (default), the LP is
    # 100 %-clearing-naive (current behaviour). For risk-analysis use
    # the post-LP stochastic Bernoulli realisation in
    # :func:`lib.analysis.lifecycle_npv.simulate_lifecycle`.
    bid_win_rate_lp: float = 1.0,
    # ADR-002 MVP — auction-access marginal cost. Positive value adds
    # ``afrr_bid_hurdle_eur_per_mw_h × block_hours × (r_pos + r_neg)`` to
    # the LP objective and subtracts the same amount from reported
    # ``revenue_total``. Models any continuous economic discipline on the
    # bid (bid-prep / fleet competition / opportunity cost / risk
    # premium) — paid per MW × hour of bid placed, regardless of
    # clearing. Retained as a sensitivity / sign-sanity knob; not the
    # central calibration lever (see ADR-002c). In two-stage Stage 2
    # (locked mode), the LP does NOT see the hurdle (sunk on locked
    # bid); reported revenue still subtracts ``hurdle × block_hours ×
    # (r_locked_pos + r_locked_neg)`` so the lifecycle headline is net.
    afrr_bid_hurdle_eur_per_mw_h: float = 0.0,
    # ADR-002c — settlement model: must-bid floor + activation-energy
    # capture haircut.
    #
    # ``r_min_per_block_mw`` (must-bid floor) raises the lower bound on
    # ``r_pos[b]`` and ``r_neg[b]`` from 0 to ``r_min`` per direction.
    # Forces the LP to bid into cap auction every block — represents a
    # fleet-style pre-allocation policy (operators contract a fraction
    # of nameplate to aFRR year-round and accept the lower
    # clearing-time-averaged revenue rather than picking only spike
    # blocks). Closes the cap-revenue under-capture gap that emerged
    # after ``max_afrr_participation`` was retired. Applied only in
    # single-stage / Stage 1 (Stage 2 inherits the locked bid).
    #
    # ``afrr_energy_capture_factor`` (merit-order capture haircut)
    # multiplies the LP's α-driven revenue, SoC drain, cycle-cap, and
    # mFRR-penalty terms by an additional fraction representing the
    # share of cleared activations the asset is actually dispatched for.
    # In reality the TSO calls activation merit-order across qualified
    # providers; only the cheapest are dispatched for any given
    # activation event. Default 1.0 = full dispatch (current behaviour).
    # Calibrated value (probably 0.10–0.30) cuts model energy-revenue
    # over-collection that arises from spike-day price-times-α capture.
    # Cap leg unaffected (operators are paid for capacity reservation
    # regardless of dispatch).
    r_min_per_block_mw: float = 0.0,
    afrr_energy_capture_factor: float = 1.0,
    initial_soc_mwh: float | None = None,
    cyclic_soc: bool = True,
    final_soc_target_mwh: float | None = None,
    periods_per_block_override: int | None = None,
) -> StackedDispatchResult:
    """
    Joint LP across DA + ID + aFRR cap + aFRR activation.

    Variable horizon: the LP now derives ``T`` from input length (any
    multiple of ``PERIODS_PER_BLOCK``), enabling intra-day rolling-
    horizon MPC when ``T < PERIODS_PER_DAY``.

    MPC mode (``cyclic_soc=False`` + explicit ``initial_soc_mwh``): the
    LP accepts an arbitrary starting SoC and does NOT force the final
    SoC back to the starting value — the caller is responsible for
    carrying the observed final SoC into the next MPC step. Optional
    ``final_soc_target_mwh`` imposes an equality constraint on final
    SoC (useful for terminal-cost shaping).

    See module docstring for physics. Returns ``StackedDispatchResult``.
    """
    T = len(prices_da)
    PB = periods_per_block_override or PERIODS_PER_BLOCK
    if T % PB != 0:
        # Round T up to full block, zero-pad aFRR slot — unusual case.
        T_padded = ((T // PB) + 1) * PB
        if T_padded != T:
            # Fail fast rather than silent padding
            return _failed_result(
                PERIODS_PER_DAY, energy_mwh, 0.5 * energy_mwh,
                f"prices_da length {T} not a multiple of PERIODS_PER_BLOCK={PB}",
            )
    B = T // PB
    eta = sqrt(rte)

    # ADR-002c: combined activation effectiveness = clearing × dispatch.
    # Applied to all α-driven terms (energy revenue, SoC dynamics α,
    # cycle-cap α, reservation-headroom α, mFRR penalty). Cap revenue
    # uses ``bid_win_rate_lp`` only (cap is paid for cleared capacity
    # regardless of dispatch).
    activation_eff = float(bid_win_rate_lp) * float(afrr_energy_capture_factor)

    soc_min = soc_min_frac * energy_mwh
    soc_max = soc_max_frac * energy_mwh
    soc_init = (
        float(initial_soc_mwh) if initial_soc_mwh is not None
        else 0.5 * (soc_min + soc_max)
    )
    # Clamp to envelope (for MPC: carry-over SoC might be barely outside
    # the envelope due to floating-point; clamp rather than fail).
    soc_init = float(np.clip(soc_init, soc_min, soc_max))
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

    # Two-stage Stage 2 indicator: r_pos / r_neg per-block bounded above by
    # the locked Stage-1 commitment. The LP variable is the *delivered*
    # fraction; under-delivery is allowed (paying mFRR-imbalance penalty
    # on the shortfall = locked − delivered) so Stage 2 stays feasible
    # when realised α exceeds Stage-1 forecast. Cap revenue is paid on
    # the locked commitment regardless of delivery (D−1 auction settles
    # before realisations are known); the LP coefficient on r_pos[b] in
    # Stage 2 reflects only the energy revenue + penalty trade-off.
    use_locked_stage2 = (
        r_pos_locked_per_block is not None
        or r_neg_locked_per_block is not None
    )
    if use_locked_stage2:
        if r_pos_locked_per_block is None or r_neg_locked_per_block is None:
            return _failed_result(
                T, energy_mwh, 0.5 * energy_mwh,
                "r_pos_locked_per_block and r_neg_locked_per_block must both be provided",
            )
        if len(r_pos_locked_per_block) != B or len(r_neg_locked_per_block) != B:
            return _failed_result(
                T, energy_mwh, 0.5 * energy_mwh,
                f"locked block arrays must have length {B}, got "
                f"{len(r_pos_locked_per_block)} / {len(r_neg_locked_per_block)}",
            )
        r_pos_locked_arr = np.asarray(r_pos_locked_per_block, dtype=float)
        r_neg_locked_arr = np.asarray(r_neg_locked_per_block, dtype=float)
    else:
        r_pos_locked_arr = None
        r_neg_locked_arr = None

    # Variable layout — single flat vector for linprog:
    #   0..T-1           : c_da      (charge DA, MW)
    #   T..2T-1          : d_da      (discharge DA, MW)
    #   2T..3T-1         : c_id
    #   3T..4T-1         : d_id
    #   4T..4T+B-1       : r_pos     (aFRR POS reservation per block, MW)
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
    # In Stage 2 (locked) mode the cap revenue is *paid on the locked
    # commitment* (settled at D−1 before realisations), independent of the
    # LP-chosen delivered r — so it is added as a constant outside the LP.
    # Market-aware mode: cap coefficient is scaled by bid_win_rate_lp
    # (only the cleared fraction collects cap revenue ex-ante).
    block_hours = PB * dt_hours
    if not use_locked_stage2:
        for b in range(B):
            c[idx_r_pos_block.start + b] = (
                -afrr_cap_pos_price[b] * block_hours * bid_win_rate_lp
            )
            c[idx_r_neg_block.start + b] = (
                -afrr_cap_neg_price[b] * block_hours * bid_win_rate_lp
            )
        # ADR-002 MVP — auction-access marginal cost on each MW × block-hour
        # of bid placed (paid regardless of clearing). Disciplines the LP's
        # otherwise unconstrained urge to bid full nameplate when
        # ``max_afrr_participation`` is retired. In Stage-2 (locked) mode the
        # bid is sunk at Stage 1 — no LP coefficient added here; the
        # post-solve revenue accounting subtracts the same per-MW cost on
        # the locked bid.
        if afrr_bid_hurdle_eur_per_mw_h:
            hurdle_per_mw_block = afrr_bid_hurdle_eur_per_mw_h * block_hours
            for b in range(B):
                c[idx_r_pos_block.start + b] += hurdle_per_mw_block
                c[idx_r_neg_block.start + b] += hurdle_per_mw_block
    # aFRR energy: activation a_pos[t] = r_pos[block(t)] × α_pos[t]; revenue
    # at energy price e_pos[t]. Total POS energy rev = Σ_t e_pos[t] × r_pos[b(t)] × α_pos[t] × dt
    # → objective contribution on r_pos per block:
    #        -Σ_{t in block} e_pos[t] × α_pos[t] × dt
    # Similarly NEG — but operator PAYS (neg price): the negative price times
    # positive α and positive r produces negative revenue already, so LP sees
    # correctly that large r_neg at highly-negative prices is costly.
    #
    # Stage 2 (locked) shortfall mechanic: shortfall = locked - r_pos
    # (≥ 0 by upper bound), penalty cost per MWh of activated shortfall =
    # mfrr_penalty × α × dt. Linear in r_pos with coefficient
    # -Σ α × penalty × dt (the LP wants r_pos high to avoid penalty), plus
    # a constant `locked × Σ α × penalty × dt` that we drop here and add
    # back to revenue accounting after the solve.
    # Market-aware mode: aFRR-energy revenue is scaled by bid_win_rate_lp
    # (only the cleared fraction delivers and collects energy revenue).
    for b in range(B):
        lo = b * PB
        hi = (b + 1) * PB
        # ADR-002c: energy revenue scales by activation_eff
        # (= bid_win_rate_lp × afrr_energy_capture_factor); only the
        # cleared-and-dispatched fraction earns activation revenue.
        coeff_pos = (
            afrr_energy_pos_price[lo:hi] * afrr_activation_rate_pos[lo:hi]
        ).sum() * dt_hours * activation_eff
        coeff_neg = (
            afrr_energy_neg_price[lo:hi] * afrr_activation_rate_neg[lo:hi]
        ).sum() * dt_hours * activation_eff
        c[idx_r_pos_block.start + b] += -coeff_pos
        c[idx_r_neg_block.start + b] += -coeff_neg
        if use_locked_stage2:
            # mFRR-imbalance penalty: paid on the activated dispatched
            # fraction of (locked − delivered). Penalty coefficient on
            # r_pos[b] uses activation_eff (matches the LP-aware
            # cap/energy semantics + dispatch capture).
            pen_pos_b = (
                mfrr_penalty_pos_eur_per_mwh
                * afrr_activation_rate_pos[lo:hi].sum()
                * dt_hours
                * activation_eff
            )
            pen_neg_b = (
                mfrr_penalty_neg_eur_per_mwh
                * afrr_activation_rate_neg[lo:hi].sum()
                * dt_hours
                * activation_eff
            )
            c[idx_r_pos_block.start + b] += -pen_pos_b
            c[idx_r_neg_block.start + b] += -pen_neg_b

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

    # Stream-differentiated wear premium for aFRR (Idea 2): aFRR cycles are
    # TSO-driven and hit at unforecastable SoC moments → higher per-MWh
    # stress vs operator-controlled arbitrage. Charge an extra
    # ``afrr_wear_premium_eur_per_mwh`` on aFRR-activation throughput
    # (a_pos = α_pos × r_pos and a_neg = α_neg × r_neg). Wired through
    # the per-block r_pos / r_neg coefficient.
    if afrr_wear_premium_eur_per_mwh > 0:
        for b in range(B):
            lo = b * PB
            hi = (b + 1) * PB
            extra_pos = afrr_activation_rate_pos[lo:hi].sum() * dt_hours \
                * afrr_wear_premium_eur_per_mwh
            extra_neg = afrr_activation_rate_neg[lo:hi].sum() * dt_hours \
                * afrr_wear_premium_eur_per_mwh
            c[idx_r_pos_block.start + b] += extra_pos
            c[idx_r_neg_block.start + b] += extra_neg

    # --- Inequality constraints (A_ub × x ≤ b_ub) ------------------------
    A_rows: list[np.ndarray] = []
    b_rows: list[float] = []

    # 1) Power bounds per interval: reservation blocks power regardless of
    #    activation — must be able to deliver full reservation if called.
    #    Discharge side:  d_da[t] + d_id[t] + r_pos[b(t)] ≤ P_max
    #    Charge side:     c_da[t] + c_id[t] + r_neg[b(t)] ≤ P_max
    #    Stage 2 (locked) — power is blocked by the *locked* commitment
    #    (operator must hold capacity even if they choose to under-deliver),
    #    so the LP variable r_pos[b] drops out of the power inequality and
    #    the bound becomes P_max − r_locked[b].
    for t in range(T):
        b = block_of[t]
        row = np.zeros(n_vars)
        row[idx_d_da.start + t] = 1.0
        row[idx_d_id.start + t] = 1.0
        if use_locked_stage2:
            d_rhs = power_mw - float(r_pos_locked_arr[b])
        else:
            row[idx_r_pos_block.start + b] = 1.0
            d_rhs = power_mw
        A_rows.append(row)
        b_rows.append(d_rhs)

        row = np.zeros(n_vars)
        row[idx_c_da.start + t] = 1.0
        row[idx_c_id.start + t] = 1.0
        if use_locked_stage2:
            c_rhs = power_mw - float(r_neg_locked_arr[b])
        else:
            row[idx_r_neg_block.start + b] = 1.0
            c_rhs = power_mw
        A_rows.append(row)
        b_rows.append(c_rhs)

    # 2) SoC dynamics at every interval:
    #    soc[t] = soc_init + Σ_{τ≤t} ((c_da+c_id+a_neg)×η - (d_da+d_id+a_pos)/η) × dt
    #    where a_pos[τ] = α_pos[τ] × r_pos[b(τ)],  a_neg[τ] = α_neg[τ] × r_neg[b(τ)]
    #    soc_min ≤ soc[t] ≤ soc_max  →  Δ_cum[t] ∈ [soc_min − soc_init, soc_max − soc_init]
    #
    #    Market-aware mode: a_pos / a_neg in SoC dynamics are scaled by
    #    bid_win_rate_lp (expected activation under ex-ante clearing
    #    probability). The reservation headroom constraint below keeps
    #    full r — operator must hold capacity in case the bid clears.
    for t in range(T):
        row = np.zeros(n_vars)
        for tau in range(t + 1):
            b = block_of[tau]
            row[idx_c_da.start + tau] += eta * dt_hours
            row[idx_c_id.start + tau] += eta * dt_hours
            row[idx_d_da.start + tau] += -dt_hours / eta
            row[idx_d_id.start + tau] += -dt_hours / eta
            # Activation contributions (expected, under ex-ante clearing
            # × dispatch capture — ADR-002c).
            row[idx_r_neg_block.start + b] += (
                afrr_activation_rate_neg[tau] * eta * dt_hours * activation_eff
            )
            row[idx_r_pos_block.start + b] += (
                -afrr_activation_rate_pos[tau] * dt_hours / eta * activation_eff
            )

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
    #
    #    In Stage 2 (locked) the constraint applies to the LP-chosen
    #    r_pos / r_neg (the *delivered* fraction). When realised SoC can't
    #    sustain the locked commitment, the LP simply lowers r_pos / r_neg
    #    to satisfy the headroom, paying the mFRR shortfall penalty
    #    (handled in objective) on the locked − delivered gap. Equivalent
    #    to ADR-001 §Open issue 1 option (a) but with the cleaner LP
    #    structure of soft *commitment* instead of soft *headroom*.
    reserve_h = max(0.0, afrr_reserve_duration_hours)
    if reserve_h > 0:
        for t in range(T):
            b = block_of[t]
            # Reconstruct the soc[t] row (accumulated flows to time t).
            # α × r terms are expected drain — scaled by bid_win_rate_lp
            # to match the (expected) SoC trajectory the LP optimises.
            # The reservation-headroom term itself uses FULL r (operator
            # must reserve capacity in case the bid clears).
            soc_row = np.zeros(n_vars)
            for tau in range(t + 1):
                bb = block_of[tau]
                soc_row[idx_c_da.start + tau] += eta * dt_hours
                soc_row[idx_c_id.start + tau] += eta * dt_hours
                soc_row[idx_d_da.start + tau] += -dt_hours / eta
                soc_row[idx_d_id.start + tau] += -dt_hours / eta
                soc_row[idx_r_neg_block.start + bb] += (
                    afrr_activation_rate_neg[tau] * eta * dt_hours * activation_eff
                )
                soc_row[idx_r_pos_block.start + bb] += (
                    -afrr_activation_rate_pos[tau] * dt_hours / eta * activation_eff
                )

            row = -soc_row.copy()
            row[idx_r_pos_block.start + b] += reserve_h / eta
            A_rows.append(row)
            b_rows.append(soc_init - soc_min)

            row = soc_row.copy()
            row[idx_r_neg_block.start + b] += reserve_h * eta
            A_rows.append(row)
            b_rows.append(soc_max - soc_init)

    # 4) Cycle cap: total discharge energy (DA + ID + pos activation) ≤ max_cycles × usable_energy.
    # Market-aware mode: aFRR activation contribution scaled by
    # bid_win_rate_lp (expected throughput from cleared fraction).
    cycle_row = np.zeros(n_vars)
    cycle_row[idx_d_da] = dt_hours
    cycle_row[idx_d_id] = dt_hours
    for b in range(B):
        lo = b * PB
        hi = (b + 1) * PB
        cycle_row[idx_r_pos_block.start + b] = (
            afrr_activation_rate_pos[lo:hi].sum() * dt_hours * activation_eff
        )
    A_rows.append(cycle_row)
    b_rows.append(max_cycles * usable_energy_mwh)

    # 5) End-of-horizon SoC constraint.
    #    - cyclic_soc=True (day-LP default): end ≈ start, loose tolerance
    #    - final_soc_target_mwh set: end = target (tight, for MPC
    #      terminal shaping)
    #    - otherwise: no final-SoC constraint (open-ended MPC horizon)
    if cyclic_soc or final_soc_target_mwh is not None:
        # End-of-horizon SoC built from the same cumulative flows as
        # constraint #2; α × r terms scaled by bid_win_rate_lp (expected).
        end_row = np.zeros(n_vars)
        for tau in range(T):
            bb = block_of[tau]
            end_row[idx_c_da.start + tau] += eta * dt_hours
            end_row[idx_c_id.start + tau] += eta * dt_hours
            end_row[idx_d_da.start + tau] += -dt_hours / eta
            end_row[idx_d_id.start + tau] += -dt_hours / eta
            end_row[idx_r_neg_block.start + bb] += (
                afrr_activation_rate_neg[tau] * eta * dt_hours * activation_eff
            )
            end_row[idx_r_pos_block.start + bb] += (
                -afrr_activation_rate_pos[tau] * dt_hours / eta * activation_eff
            )
        if cyclic_soc and final_soc_target_mwh is None:
            tolerance_mwh = 0.1 * energy_mwh
            A_rows.append(end_row.copy())
            b_rows.append(tolerance_mwh)
            A_rows.append(-end_row.copy())
            b_rows.append(tolerance_mwh)
        else:  # target specified
            delta = float(final_soc_target_mwh) - soc_init
            tol = 0.02 * energy_mwh
            A_rows.append(end_row.copy())
            b_rows.append(delta + tol)
            A_rows.append(-end_row.copy())
            b_rows.append(-delta + tol)

    A_ub = np.asarray(A_rows)
    b_ub = np.asarray(b_rows)

    # Bounds: all variables ≥ 0. Power vars ≤ power_mw. aFRR reservation
    # capped at ``max_afrr_participation × P_max`` — reflects real-world
    # bid-win rates, risk-averse placement, and operational withholding
    # that keep operators below the LP-unconstrained optimum. Default 1.0
    # gives the unconstrained upper bound; 0.5 approximates typical
    # German 2023-2025 bid competition (demand/offered ≈ 0.55).
    #
    # Stage 2 mode (locked block reservations passed): equality bounds
    # pin r_pos[b] / r_neg[b] to Stage-1 values, overriding the cap.
    r_cap = min(power_mw, max_afrr_participation * power_mw)
    bounds = [(0.0, power_mw)] * (4 * T)
    if use_locked_stage2:
        for b in range(B):
            bounds.append((0.0, float(r_pos_locked_arr[b])))
        for b in range(B):
            bounds.append((0.0, float(r_neg_locked_arr[b])))
    else:
        # ADR-002c: must-bid floor on r_pos / r_neg per direction. Clamp
        # r_min to r_cap to keep bounds feasible.
        r_min = max(0.0, min(float(r_min_per_block_mw), r_cap))
        bounds += [(r_min, r_cap)] * (2 * B)

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

    # Broadcast block-level reservations to per-interval. r_pos / r_neg
    # remain the *bid* (= LP variable, what the operator commits to in
    # the auction). a_pos / a_neg = α × r are the *full-clear*
    # activations (what the operator delivers if their bid clears).
    r_pos = r_pos_block[block_of]
    r_neg = r_neg_block[block_of]
    a_pos = afrr_activation_rate_pos * r_pos
    a_neg = afrr_activation_rate_neg * r_neg

    market_aware = bid_win_rate_lp < 1.0 or activation_eff < 1.0
    if market_aware:
        # Expected values: what the LP optimised for at ex-ante clearing
        # × dispatch capture. r_*_expected uses bid_win_rate_lp (cap
        # leg = cleared capacity); a_*_expected uses activation_eff
        # (energy / SoC / cycle leg = cleared-and-dispatched activation).
        r_pos_expected = float(bid_win_rate_lp) * r_pos
        r_neg_expected = float(bid_win_rate_lp) * r_neg
        a_pos_expected = activation_eff * a_pos
        a_neg_expected = activation_eff * a_neg
    else:
        r_pos_expected = None
        r_neg_expected = None
        a_pos_expected = None
        a_neg_expected = None

    # SoC trajectory uses expected α drain (× activation_eff) — matches
    # the LP's SoC dynamics constraint.
    a_pos_soc = a_pos_expected if a_pos_expected is not None else a_pos
    a_neg_soc = a_neg_expected if a_neg_expected is not None else a_neg
    soc_changes = (
        (c_da + c_id + a_neg_soc) * eta * dt_hours
        - (d_da + d_id + a_pos_soc) / eta * dt_hours
    )
    soc = soc_init + np.cumsum(soc_changes)

    revenue_da = float(np.sum(d_da * dt_hours * prices_da * eta - c_da * dt_hours * prices_da / eta))
    revenue_id = float(np.sum(d_id * dt_hours * prices_id * eta - c_id * dt_hours * prices_id / eta))
    # aFRR energy revenue: under market-aware mode, only the cleared
    # fraction is delivered, so revenue accrues on a_*_expected.
    revenue_afrr_energy_pos = float(
        np.sum(a_pos_soc * dt_hours * afrr_energy_pos_price)
    )
    revenue_afrr_energy_neg = float(
        np.sum(a_neg_soc * dt_hours * afrr_energy_neg_price)
    )

    shortfall_pos_arr: Optional[np.ndarray] = None
    shortfall_neg_arr: Optional[np.ndarray] = None
    shortfall_cost = 0.0
    if use_locked_stage2:
        # Cap revenue is settled at D−1 on the locked commitment,
        # independent of LP-chosen delivery. Under expected mode
        # (bid_win_rate_lp < 1.0) only that fraction of bids actually
        # clears, so reported cap revenue is at expected = p × bid.
        revenue_afrr_cap_pos = float(
            np.sum(r_pos_locked_arr * afrr_cap_pos_price)
            * block_hours * bid_win_rate_lp
        )
        revenue_afrr_cap_neg = float(
            np.sum(r_neg_locked_arr * afrr_cap_neg_price)
            * block_hours * bid_win_rate_lp
        )
        # Per-interval shortfall in MW = α[t] × (locked − delivered).
        # The reported shortfall (in MW) is on the bid; the cost is on
        # the cleared-and-activated fraction = α × shortfall × p.
        shortfall_pos_arr = (
            afrr_activation_rate_pos
            * (r_pos_locked_arr[block_of] - r_pos_block[block_of])
        )
        shortfall_neg_arr = (
            afrr_activation_rate_neg
            * (r_neg_locked_arr[block_of] - r_neg_block[block_of])
        )
        # ADR-002c: shortfall cost is on the cleared-and-dispatched
        # fraction = activation_eff × α × (locked − delivered).
        shortfall_cost = float(
            (shortfall_pos_arr * mfrr_penalty_pos_eur_per_mwh).sum() * dt_hours
            * activation_eff
            + (shortfall_neg_arr * mfrr_penalty_neg_eur_per_mwh).sum() * dt_hours
            * activation_eff
        )
    else:
        # Cap revenue scales by bid_win_rate_lp (only cleared bids
        # collect cap; LP optimised expected, so we report expected).
        revenue_afrr_cap_pos = float(
            np.sum(r_pos_block * afrr_cap_pos_price) * block_hours * bid_win_rate_lp
        )
        revenue_afrr_cap_neg = float(
            np.sum(r_neg_block * afrr_cap_neg_price) * block_hours * bid_win_rate_lp
        )

    # ADR-002 MVP — auction-access cost on the bid commitment. In Stage-2
    # (locked) mode the bid is the locked Stage-1 commitment; in
    # single-stage mode the bid is the LP-chosen r_pos_block / r_neg_block.
    if afrr_bid_hurdle_eur_per_mw_h:
        if use_locked_stage2:
            bid_for_hurdle_pos = r_pos_locked_arr
            bid_for_hurdle_neg = r_neg_locked_arr
        else:
            bid_for_hurdle_pos = r_pos_block
            bid_for_hurdle_neg = r_neg_block
        bid_hurdle_cost_eur = float(
            (bid_for_hurdle_pos + bid_for_hurdle_neg).sum()
            * block_hours * afrr_bid_hurdle_eur_per_mw_h
        )
    else:
        bid_hurdle_cost_eur = 0.0

    revenue_total = (
        revenue_da + revenue_id
        + revenue_afrr_cap_pos + revenue_afrr_cap_neg
        + revenue_afrr_energy_pos + revenue_afrr_energy_neg
        - shortfall_cost
        - bid_hurdle_cost_eur
    )

    # FEC binds on expected throughput when market-aware (matches the
    # cycle-cap constraint inside the LP).
    total_discharge_mwh = float(
        np.sum((d_da + d_id + a_pos_soc) * dt_hours)
    )
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
        shortfall_pos_mwh=shortfall_pos_arr,
        shortfall_neg_mwh=shortfall_neg_arr,
        shortfall_cost_eur=shortfall_cost,
        bid_win_rate_lp_used=float(bid_win_rate_lp),
        r_pos_expected=r_pos_expected,
        r_neg_expected=r_neg_expected,
        a_pos_expected=a_pos_expected,
        a_neg_expected=a_neg_expected,
        bid_hurdle_cost_eur=bid_hurdle_cost_eur,
    )


def optimize_day_two_stage(
    *,
    prices_da: np.ndarray,
    prices_id: np.ndarray,
    afrr_cap_pos_price: np.ndarray,
    afrr_cap_neg_price: np.ndarray,
    afrr_energy_pos_price: np.ndarray,         # realised
    afrr_energy_neg_price: np.ndarray,         # realised
    afrr_activation_rate_pos: np.ndarray,      # realised α
    afrr_activation_rate_neg: np.ndarray,      # realised α
    afrr_activation_forecast_pos: np.ndarray,  # Stage 1 input (regime-conditional mean α)
    afrr_activation_forecast_neg: np.ndarray,
    afrr_energy_forecast_pos: np.ndarray,      # Stage 1 input (regime-conditional mean price)
    afrr_energy_forecast_neg: np.ndarray,
    energy_mwh: float,
    power_mw: float = 1.0,
    rte: float = 0.85,
    soc_min_frac: float = 0.05,
    soc_max_frac: float = 0.95,
    max_cycles: float = 2.0,
    dt_hours: float = DT_HOURS_DEFAULT,
    afrr_reserve_duration_hours: float = 1.0,
    wear_cost_eur_per_mwh: np.ndarray | None = None,
    afrr_wear_premium_eur_per_mwh: float = 0.0,
    mfrr_penalty_pos_eur_per_mwh: float = 500.0,
    mfrr_penalty_neg_eur_per_mwh: float = 500.0,
    bid_win_rate_lp: float = 1.0,                # ex-ante clearing probability for Stage 1
    # ADR-002 MVP — auction-access marginal cost on r_bid; applied in
    # Stage 1 (disciplines bid choice). Stage 2 receives the locked bid
    # and re-applies the same hurdle as a fixed deduction from realised
    # revenue (sunk on the locked commitment) so the lifecycle headline
    # is net of the hurdle.
    afrr_bid_hurdle_eur_per_mw_h: float = 0.0,
    # ADR-002c — settlement model: must-bid floor + activation-energy
    # capture haircut. Stage 1 sees the must-bid floor (the LP commits
    # at least r_min per direction every block); Stage 2 inherits the
    # locked Stage-1 r values, so r_min has no direct Stage-2 effect.
    # Both stages apply the capture factor to all α-driven terms.
    r_min_per_block_mw: float = 0.0,
    afrr_energy_capture_factor: float = 1.0,
    # Phase 2.3 stochastic-recourse mode: when both award masks are
    # provided (length-B 0/1 arrays from a Bernoulli realisation),
    # Stage 2 receives ``r_locked = r_bid × award_mask`` per block per
    # direction — un-cleared blocks have r_locked=0 so Stage 2 can
    # re-trade the freed power on DA/ID. Stage 2 also runs at
    # ``bid_win_rate_lp=1.0`` because clearing has already realised
    # deterministically. This is the upper-bound risk model with
    # recourse (BCM-clears-before-intraday sequencing).
    stage2_award_mask_pos: np.ndarray | None = None,
    stage2_award_mask_neg: np.ndarray | None = None,
    initial_soc_mwh: float | None = None,
    cyclic_soc: bool = True,
    final_soc_target_mwh: float | None = None,
) -> TwoStageDispatchResult:
    """Two-stage DA-aFRR-commit → intraday re-optimisation (ADR-001).

    Stage 1 — at D−1 08:00, the operator commits aFRR per-block
    reservation r_pos[b], r_neg[b] for day D using:
      * realised DA-day prices (assumed forecastable from forwards)
      * realised aFRR capacity prices (auction clears at D−1 08:00)
      * **forecast** α and aFRR energy prices (regime-conditional mean)
      * intraday prices proxied by DA (no ID information yet)
    No exogenous cap on r — ``max_afrr_participation`` is set to 1.0;
    the natural cap emerges from the SoC reservation headroom constraint
    given ``afrr_reserve_duration_hours`` (typically 1 h).

    Stage 2 — at D−1 18:00 onwards, operator re-optimises full DA / ID /
    aFRR-activation dispatch under the locked Stage-1 r commitments,
    using **realised** α and energy prices. SoC reservation constraint
    becomes soft via shortfall variables: shortfall × α × penalty
    captures mFRR-imbalance cost for delivery shortfall when realised α
    exceeds the Stage-1 forecast that justified the commitment.

    Returns :class:`TwoStageDispatchResult` with both stage outputs;
    ``stage2`` carries the realised dispatch and net revenue
    (after imbalance penalty), ``stage1`` carries the commitment LP
    diagnostics. Smoke-test invariant: ``stage1.r_pos`` should not
    trivially saturate at ``power_mw`` — the SoC reservation headroom
    enforces an emergent commitment cap.
    """
    stage1 = optimize_day_stacked(
        prices_da=prices_da,
        prices_id=prices_da,                                     # ID = DA proxy at D−1
        afrr_cap_pos_price=afrr_cap_pos_price,
        afrr_cap_neg_price=afrr_cap_neg_price,
        afrr_energy_pos_price=afrr_energy_forecast_pos,
        afrr_energy_neg_price=afrr_energy_forecast_neg,
        afrr_activation_rate_pos=afrr_activation_forecast_pos,
        afrr_activation_rate_neg=afrr_activation_forecast_neg,
        energy_mwh=energy_mwh,
        power_mw=power_mw,
        rte=rte,
        soc_min_frac=soc_min_frac,
        soc_max_frac=soc_max_frac,
        max_cycles=max_cycles,
        dt_hours=dt_hours,
        afrr_reserve_duration_hours=afrr_reserve_duration_hours,
        wear_cost_eur_per_mwh=wear_cost_eur_per_mwh,
        afrr_wear_premium_eur_per_mwh=afrr_wear_premium_eur_per_mwh,
        max_afrr_participation=1.0,                              # no exogenous cap; let LP decide
        bid_win_rate_lp=bid_win_rate_lp,                          # market-aware ex-ante clearing
        afrr_bid_hurdle_eur_per_mw_h=afrr_bid_hurdle_eur_per_mw_h,  # ADR-002 MVP discipline on Stage 1 bid
        r_min_per_block_mw=r_min_per_block_mw,                    # ADR-002c must-bid floor (Stage 1 only)
        afrr_energy_capture_factor=afrr_energy_capture_factor,    # ADR-002c capture haircut
        initial_soc_mwh=initial_soc_mwh,
        cyclic_soc=cyclic_soc,
        final_soc_target_mwh=final_soc_target_mwh,
    )
    if not stage1.success:
        return TwoStageDispatchResult(stage1=stage1, stage2=stage1)

    # Extract per-block r commitments (LP returns r broadcast to per-interval).
    # In recourse mode, multiply by the award mask (per block per direction)
    # — un-cleared blocks have r_locked=0, freeing power for Stage 2 DA/ID
    # re-trading. Stage 2 then runs at bid_win_rate_lp=1.0 because clearing
    # has been realised deterministically.
    r_pos_locked = stage1.r_pos[::PERIODS_PER_BLOCK].copy()
    r_neg_locked = stage1.r_neg[::PERIODS_PER_BLOCK].copy()
    stage2_bid_win_rate_lp = bid_win_rate_lp
    if stage2_award_mask_pos is not None:
        if len(stage2_award_mask_pos) != len(r_pos_locked):
            return TwoStageDispatchResult(stage1=stage1, stage2=stage1)
        r_pos_locked = r_pos_locked * np.asarray(stage2_award_mask_pos, dtype=float)
        stage2_bid_win_rate_lp = 1.0
    if stage2_award_mask_neg is not None:
        if len(stage2_award_mask_neg) != len(r_neg_locked):
            return TwoStageDispatchResult(stage1=stage1, stage2=stage1)
        r_neg_locked = r_neg_locked * np.asarray(stage2_award_mask_neg, dtype=float)
        stage2_bid_win_rate_lp = 1.0

    stage2 = optimize_day_stacked(
        prices_da=prices_da,
        prices_id=prices_id,                                     # realised ID
        afrr_cap_pos_price=afrr_cap_pos_price,
        afrr_cap_neg_price=afrr_cap_neg_price,
        afrr_energy_pos_price=afrr_energy_pos_price,             # realised
        afrr_energy_neg_price=afrr_energy_neg_price,             # realised
        afrr_activation_rate_pos=afrr_activation_rate_pos,       # realised
        afrr_activation_rate_neg=afrr_activation_rate_neg,       # realised
        energy_mwh=energy_mwh,
        power_mw=power_mw,
        rte=rte,
        soc_min_frac=soc_min_frac,
        soc_max_frac=soc_max_frac,
        max_cycles=max_cycles,
        dt_hours=dt_hours,
        afrr_reserve_duration_hours=afrr_reserve_duration_hours,
        wear_cost_eur_per_mwh=wear_cost_eur_per_mwh,
        afrr_wear_premium_eur_per_mwh=afrr_wear_premium_eur_per_mwh,
        max_afrr_participation=1.0,                              # locked dominates
        r_pos_locked_per_block=r_pos_locked,
        r_neg_locked_per_block=r_neg_locked,
        bid_win_rate_lp=stage2_bid_win_rate_lp,
        afrr_bid_hurdle_eur_per_mw_h=afrr_bid_hurdle_eur_per_mw_h,  # ADR-002 — sunk on locked bid; deducted in revenue accounting
        # ADR-002c — Stage 2 does NOT need the must-bid floor (locked
        # bid from Stage 1 is the binding constraint). Capture haircut
        # passes through so realised Stage 2 energy revenue + SoC + FEC
        # see the merit-order dispatch fraction.
        r_min_per_block_mw=0.0,
        afrr_energy_capture_factor=afrr_energy_capture_factor,
        mfrr_penalty_pos_eur_per_mwh=mfrr_penalty_pos_eur_per_mwh,
        mfrr_penalty_neg_eur_per_mwh=mfrr_penalty_neg_eur_per_mwh,
        initial_soc_mwh=initial_soc_mwh,
        cyclic_soc=cyclic_soc,
        final_soc_target_mwh=final_soc_target_mwh,
    )
    return TwoStageDispatchResult(stage1=stage1, stage2=stage2)
