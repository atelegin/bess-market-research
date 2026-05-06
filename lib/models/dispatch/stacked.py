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
    # ADR-006 (Model B): separate IDA1 auction trade stream. Zeros when
    # ``prices_ida1`` not passed — backward-compatible with the Model A+
    # single-stream dispatch where ``charge_id/discharge_id`` is the
    # combined IDA2-with-IDA1-fallback proxy.
    charge_ida1: np.ndarray
    discharge_ida1: np.ndarray
    charge_id: np.ndarray
    discharge_id: np.ndarray
    # Per-interval aFRR state (r broadcast from per-block, a = α × r)
    r_pos: np.ndarray
    r_neg: np.ndarray
    a_pos: np.ndarray
    a_neg: np.ndarray
    # ADR-006: FCR reservation (symmetric, capacity-only product). r_fcr is
    # broadcast from per-block to per-interval (length 96); zeros when FCR
    # is disabled (no fcr_cap_price passed). Zero-net-energy assumption: no
    # SoC dynamics contribution and no cycle-cap consumption — frequency
    # activations are stochastic and approximately net to zero over a block.
    r_fcr: np.ndarray
    # SoC trajectory (length 96), MWh absolute
    soc: np.ndarray
    # Revenue split (EUR, full day)
    revenue_da: float
    revenue_ida1: float                # ADR-006 (Model B): 0.0 when IDA1 disabled
    revenue_id: float                  # IDA2 in Model B; combined ID stream in Model A+
    revenue_afrr_cap_pos: float
    revenue_afrr_cap_neg: float
    revenue_afrr_energy_pos: float
    revenue_afrr_energy_neg: float
    revenue_fcr_cap: float
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
    r_fcr_expected: Optional[np.ndarray] = None
    # ADR-002 MVP — auction-access cost charged on the bid commitment.
    # Already subtracted from ``revenue_total``; surfaced separately so
    # the lifecycle accounting / reconciliation can audit the deduction.
    bid_hurdle_cost_eur: float = 0.0
    # ADR-006 — FCR-side auction-access cost (mirrors aFRR mechanic).
    fcr_bid_hurdle_cost_eur: float = 0.0

    def revenue_by_stream(self) -> dict[str, float]:
        return {
            "da": self.revenue_da,
            "ida1": self.revenue_ida1,
            "id": self.revenue_id,
            "afrr_cap_pos": self.revenue_afrr_cap_pos,
            "afrr_cap_neg": self.revenue_afrr_cap_neg,
            "afrr_energy_pos": self.revenue_afrr_energy_pos,
            "afrr_energy_neg": self.revenue_afrr_energy_neg,
            "fcr_cap": self.revenue_fcr_cap,
            "total": self.revenue_total,
        }


@dataclass
class ThreeStageDispatchResult:
    """Result of :func:`optimize_day_three_stage` — Model B multi-window
    dispatch with honest information-set partition.

    The headline economics live in ``stage2b`` (realised dispatch under
    locked aFRR commitments and locked IDA1 trades). ``stage1`` carries
    the aFRR-commit LP under D−1 morning forecasts; ``stage2a`` carries
    the IDA1-commit LP at D−1 16:00 under realised IDA1 + forecast IDA2.
    DA / IDA2 / activation lines from ``stage1`` and ``stage2a`` are
    "scratch" — they exist because the LP needs to schedule them to make
    the IDA1/aFRR commit decisions, but only the relevant commitment
    variables are locked forward.
    """

    stage1: StackedDispatchResult       # D−1 08:00 aFRR commit (forecasts only)
    stage2a: StackedDispatchResult      # D−1 16:00 IDA1 commit (realised IDA1 + forecast IDA2)
    stage2b: StackedDispatchResult      # D−1 22:30+ IDA2 + activation (all realised)

    @property
    def success(self) -> bool:
        return self.stage1.success and self.stage2a.success and self.stage2b.success

    @property
    def revenue_total(self) -> float:
        return self.stage2b.revenue_total

    @property
    def stage1_r_pos_block(self) -> np.ndarray:
        return self.stage1.r_pos[::PERIODS_PER_BLOCK]

    @property
    def stage1_r_neg_block(self) -> np.ndarray:
        return self.stage1.r_neg[::PERIODS_PER_BLOCK]

    @property
    def stage1_r_fcr_block(self) -> np.ndarray:
        return self.stage1.r_fcr[::PERIODS_PER_BLOCK]

    @property
    def stage2a_c_ida1(self) -> np.ndarray:
        """IDA1 charge trades committed at D−1 16:00 — locked into Stage 2b."""
        return self.stage2a.charge_ida1

    @property
    def stage2a_d_ida1(self) -> np.ndarray:
        """IDA1 discharge trades committed at D−1 16:00 — locked into Stage 2b."""
        return self.stage2a.discharge_ida1


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
        charge_ida1=zeros, discharge_ida1=zeros,
        charge_id=zeros, discharge_id=zeros,
        r_pos=zeros, r_neg=zeros,
        a_pos=zeros, a_neg=zeros,
        r_fcr=zeros,
        soc=np.full(periods, soc_init),
        revenue_da=0.0, revenue_ida1=0.0, revenue_id=0.0,
        revenue_afrr_cap_pos=0.0, revenue_afrr_cap_neg=0.0,
        revenue_afrr_energy_pos=0.0, revenue_afrr_energy_neg=0.0,
        revenue_fcr_cap=0.0,
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
    # ADR-009 (price-conditional bid hurdle): when not None, an array
    # of length B (one value per 4 h block) overrides the scalar above.
    # Caller passes ``λ × mean(|IDA1[t]|)`` per block to make the
    # bid-prep cost scale with intraday price level — high in
    # peak-demand blocks (LP becomes more conservative when intraday
    # opportunity cost is high), low in calm blocks. Mirrors ISEA's
    # IDA1 ± 50 % margin rule. None (default) = use scalar above.
    afrr_bid_hurdle_per_block_eur_per_mw_h: np.ndarray | None = None,
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
    # ADR-006 — full FCR LP integration (replaces phantom additive layer).
    #
    # FCR is a symmetric, capacity-only product on the same 4 h block grid as
    # aFRR. One LP variable per block ``r_fcr[b]`` represents the symmetric
    # reservation — operator must hold ``r_fcr`` MW of both up- and
    # down-regulation continuously through the block.
    #
    # ``fcr_cap_price`` (B,) EUR/MW/h per 4 h block — clearing price.
    # When ``None`` (default) FCR is disabled: r_fcr is bounded to [0, 0]
    # and the LP behaves identically to the pre-ADR-006 dispatch. Existing
    # callers that don't pass this param see no behaviour change.
    #
    # ``fcr_reserve_duration_hours`` (default 0.5) — symmetric SoC headroom
    # the operator must hold to deliver full r_fcr in either direction
    # continuously for that horizon. Industry standard SLA for FCR. Looser
    # than aFRR's typical 1 h because frequency activations are short and
    # symmetric on average; 30 min is the regulator-floor sustained-delivery
    # requirement (PRL pre-qualification).
    #
    # ``r_min_fcr_per_block_mw`` — must-bid floor for FCR (mirrors
    # ``r_min_per_block_mw`` for aFRR). Single-stage / Stage 1 only.
    #
    # ``r_fcr_locked_per_block`` — Stage 2 lock from Stage 1's FCR
    # commitment. Cap revenue is settled at D−1 on the locked bid; LP
    # variable is bounded above by locked value. No FCR activation /
    # shortfall / mFRR penalty mechanics — FCR is treated as zero-net-energy
    # with no SoC dynamics contribution and no cycle-cap consumption.
    # Frequency activations average to ≈ 0 net energy over a 4 h block; this
    # is the standard simplification used by Schäfer (regelleistung-online),
    # ISEA, and the published BESS revenue-index methodologies.
    #
    # ``fcr_bid_hurdle_eur_per_mw_h`` — auction-access marginal cost on the
    # FCR bid (mirrors ``afrr_bid_hurdle_eur_per_mw_h``). Single-stage /
    # Stage 1 LP sees the hurdle; Stage 2 deducts it post-solve from the
    # locked commitment.
    fcr_cap_price: np.ndarray | None = None,
    fcr_reserve_duration_hours: float = 0.5,
    r_min_fcr_per_block_mw: float = 0.0,
    r_fcr_locked_per_block: np.ndarray | None = None,
    fcr_bid_hurdle_eur_per_mw_h: float = 0.0,
    # Model B (multi-window IDA1 + IDA2 LP). When ``prices_ida1`` is
    # provided, the LP gets a separate IDA1 trade pair (c_ida1[t],
    # d_ida1[t]) priced at the IDA1 auction clearing prices. Existing
    # ``prices_id`` then represents IDA2 (auction at D−1 22:30) by
    # convention, but for backward-compat the parameter name is unchanged
    # — the existing single-stream Model A+ usage continues to work
    # because c_ida1 / d_ida1 are bounded to [0, 0] when prices_ida1
    # is None.
    #
    # ``c_ida1_locked_per_interval`` / ``d_ida1_locked_per_interval``
    # (length T) — Stage 2b lock from Stage 2a's IDA1 commitment. When
    # provided, the LP variables c_ida1/d_ida1 are pinned via equality
    # bounds. IDA1 cap revenue is settled at the time of the IDA1 auction
    # clearing (D−1 16:00), so Stage 2b gets it as a fixed accounting
    # entry; the LP optimises the remaining freedom (DA continuous +
    # IDA2 + α activation). When None, the LP variables are free.
    prices_ida1: np.ndarray | None = None,
    c_ida1_locked_per_interval: np.ndarray | None = None,
    d_ida1_locked_per_interval: np.ndarray | None = None,
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
        or r_fcr_locked_per_block is not None
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
        if r_fcr_locked_per_block is not None:
            if len(r_fcr_locked_per_block) != B:
                return _failed_result(
                    T, energy_mwh, 0.5 * energy_mwh,
                    f"r_fcr_locked_per_block length {len(r_fcr_locked_per_block)} != {B}",
                )
            r_fcr_locked_arr = np.asarray(r_fcr_locked_per_block, dtype=float)
        else:
            r_fcr_locked_arr = np.zeros(B)
    else:
        r_pos_locked_arr = None
        r_neg_locked_arr = None
        r_fcr_locked_arr = None

    # FCR enabled when caller passes a non-None price array (single-stage /
    # Stage 1) or locked commitments (Stage 2). When disabled, the r_fcr
    # variable still exists in the LP but is bounded to [0, 0] so the LP
    # reduces to the pre-ADR-006 dispatch.
    fcr_enabled = fcr_cap_price is not None or r_fcr_locked_arr is not None
    if fcr_cap_price is not None and len(fcr_cap_price) != B:
        return _failed_result(
            T, energy_mwh, 0.5 * energy_mwh,
            f"fcr_cap_price length {len(fcr_cap_price)} != {B}",
        )

    # Model B: IDA1 enabled when caller passes prices_ida1 OR locked IDA1
    # trades. When disabled, c_ida1/d_ida1 LP variables are bounded to
    # [0, 0] so the LP reduces to the pre-Model-B (Model A+) dispatch.
    ida1_enabled = (
        prices_ida1 is not None
        or c_ida1_locked_per_interval is not None
        or d_ida1_locked_per_interval is not None
    )
    if prices_ida1 is not None and len(prices_ida1) != T:
        return _failed_result(
            T, energy_mwh, 0.5 * energy_mwh,
            f"prices_ida1 length {len(prices_ida1)} != {T}",
        )
    if c_ida1_locked_per_interval is not None:
        if len(c_ida1_locked_per_interval) != T:
            return _failed_result(
                T, energy_mwh, 0.5 * energy_mwh,
                f"c_ida1_locked_per_interval length {len(c_ida1_locked_per_interval)} != {T}",
            )
        c_ida1_locked_arr = np.asarray(c_ida1_locked_per_interval, dtype=float)
    else:
        c_ida1_locked_arr = None
    if d_ida1_locked_per_interval is not None:
        if len(d_ida1_locked_per_interval) != T:
            return _failed_result(
                T, energy_mwh, 0.5 * energy_mwh,
                f"d_ida1_locked_per_interval length {len(d_ida1_locked_per_interval)} != {T}",
            )
        d_ida1_locked_arr = np.asarray(d_ida1_locked_per_interval, dtype=float)
    else:
        d_ida1_locked_arr = None

    # Variable layout — single flat vector for linprog:
    #   0..T-1            : c_da      (charge DA, MW)
    #   T..2T-1           : d_da      (discharge DA, MW)
    #   2T..3T-1          : c_ida1    (Model B: IDA1 charge — 0 when disabled)
    #   3T..4T-1          : d_ida1    (Model B: IDA1 discharge — 0 when disabled)
    #   4T..5T-1          : c_id      (IDA2 in Model B; combined ID in Model A+)
    #   5T..6T-1          : d_id
    #   6T..6T+B-1        : r_pos     (aFRR POS reservation per block, MW)
    #   6T+B..6T+2B-1     : r_neg     (aFRR NEG reservation per block, MW)
    #   6T+2B..6T+3B-1    : r_fcr     (FCR symmetric reservation per block, MW)
    n_vars = 6 * T + 3 * B

    idx_c_da = slice(0, T)
    idx_d_da = slice(T, 2 * T)
    idx_c_ida1 = slice(2 * T, 3 * T)
    idx_d_ida1 = slice(3 * T, 4 * T)
    idx_c_id = slice(4 * T, 5 * T)
    idx_d_id = slice(5 * T, 6 * T)
    idx_r_pos_block = slice(6 * T, 6 * T + B)
    idx_r_neg_block = slice(6 * T + B, 6 * T + 2 * B)
    idx_r_fcr_block = slice(6 * T + 2 * B, 6 * T + 3 * B)

    # Map an interval t → its 4h block index b = t // PB
    block_of = np.arange(T) // PB

    # --- Objective (minimise negative revenue) ---------------------------
    c = np.zeros(n_vars)
    # DA revenue: discharge earns price, charge pays price/eta (eta loss)
    c[idx_c_da] = dt_hours * (prices_da / eta)                   # cost of charging at DA
    c[idx_d_da] = -dt_hours * (prices_da * eta)                  # reward for discharging to DA
    # IDA1 (Model B). When disabled (no prices_ida1), c_ida1/d_ida1 are
    # bounded to [0, 0] so the objective coefficients are inert. Setting
    # them anyway keeps the LP-vector structure homogeneous.
    if prices_ida1 is not None:
        prices_ida1_arr = np.asarray(prices_ida1, dtype=float)
        c[idx_c_ida1] = dt_hours * (prices_ida1_arr / eta)
        c[idx_d_ida1] = -dt_hours * (prices_ida1_arr * eta)
    else:
        prices_ida1_arr = None
    # ID (= IDA2 in Model B; combined IDA2-with-IDA1-fallback in Model A+).
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
        # ADR-006: FCR cap revenue (no energy/activation leg by simplification).
        # Single-stage / Stage 1: r_fcr is an LP variable; cap revenue per
        # block scaled by bid_win_rate_lp matches the aFRR mechanic.
        if fcr_cap_price is not None:
            for b in range(B):
                c[idx_r_fcr_block.start + b] = (
                    -float(fcr_cap_price[b]) * block_hours * bid_win_rate_lp
                )
        # ADR-002 MVP — auction-access marginal cost on each MW × block-hour
        # of bid placed (paid regardless of clearing). Disciplines the LP's
        # otherwise unconstrained urge to bid full nameplate when
        # ``max_afrr_participation`` is retired. In Stage-2 (locked) mode the
        # bid is sunk at Stage 1 — no LP coefficient added here; the
        # post-solve revenue accounting subtracts the same per-MW cost on
        # the locked bid.
        # ADR-009: per-block hurdle array overrides the scalar.
        if afrr_bid_hurdle_per_block_eur_per_mw_h is not None:
            _hurdle_arr = np.asarray(
                afrr_bid_hurdle_per_block_eur_per_mw_h, dtype=float
            )
            if _hurdle_arr.shape != (B,):
                raise ValueError(
                    f"afrr_bid_hurdle_per_block_eur_per_mw_h shape "
                    f"{_hurdle_arr.shape} != ({B},)"
                )
        elif afrr_bid_hurdle_eur_per_mw_h:
            _hurdle_arr = np.full(B, float(afrr_bid_hurdle_eur_per_mw_h))
        else:
            _hurdle_arr = None
        if _hurdle_arr is not None:
            for b in range(B):
                c[idx_r_pos_block.start + b] += _hurdle_arr[b] * block_hours
                c[idx_r_neg_block.start + b] += _hurdle_arr[b] * block_hours
        # ADR-006: FCR-side bid hurdle (mirrors aFRR mechanic).
        if fcr_bid_hurdle_eur_per_mw_h and fcr_cap_price is not None:
            fcr_hurdle_per_mw_block = fcr_bid_hurdle_eur_per_mw_h * block_hours
            for b in range(B):
                c[idx_r_fcr_block.start + b] += fcr_hurdle_per_mw_block
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
        # Wear hits both charge and discharge throughput across all
        # auction streams (DA, IDA1, IDA2). When IDA1 is disabled the
        # bounded-zero LP variables make this term inert.
        c[idx_c_da] += dt_hours * wear
        c[idx_d_da] += dt_hours * wear
        c[idx_c_ida1] += dt_hours * wear
        c[idx_d_ida1] += dt_hours * wear
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
    #    Discharge side:  d_da[t] + d_id[t] + r_pos[b(t)] + r_fcr[b(t)] ≤ P_max
    #    Charge side:     c_da[t] + c_id[t] + r_neg[b(t)] + r_fcr[b(t)] ≤ P_max
    #    FCR is symmetric — r_fcr appears on BOTH inequalities (operator must
    #    hold capacity in both directions for the full block).
    #    Stage 2 (locked) — power is blocked by the *locked* commitment
    #    (operator must hold capacity even if they choose to under-deliver),
    #    so the LP variable r_pos[b] / r_fcr[b] drops out of the power
    #    inequality and the bound becomes P_max − (r_pos_locked + r_fcr_locked).
    for t in range(T):
        b = block_of[t]
        row = np.zeros(n_vars)
        row[idx_d_da.start + t] = 1.0
        row[idx_d_ida1.start + t] = 1.0   # Model B: IDA1 discharge contributes
        row[idx_d_id.start + t] = 1.0
        if use_locked_stage2:
            d_rhs = power_mw - float(r_pos_locked_arr[b]) - float(r_fcr_locked_arr[b])
        else:
            row[idx_r_pos_block.start + b] = 1.0
            row[idx_r_fcr_block.start + b] = 1.0
            d_rhs = power_mw
        A_rows.append(row)
        b_rows.append(d_rhs)

        row = np.zeros(n_vars)
        row[idx_c_da.start + t] = 1.0
        row[idx_c_ida1.start + t] = 1.0   # Model B: IDA1 charge contributes
        row[idx_c_id.start + t] = 1.0
        if use_locked_stage2:
            c_rhs = power_mw - float(r_neg_locked_arr[b]) - float(r_fcr_locked_arr[b])
        else:
            row[idx_r_neg_block.start + b] = 1.0
            row[idx_r_fcr_block.start + b] = 1.0
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
            row[idx_c_ida1.start + tau] += eta * dt_hours    # Model B: IDA1 charge → SoC
            row[idx_c_id.start + tau] += eta * dt_hours
            row[idx_d_da.start + tau] += -dt_hours / eta
            row[idx_d_ida1.start + tau] += -dt_hours / eta   # Model B: IDA1 discharge → SoC
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
    reserve_h_fcr = max(0.0, fcr_reserve_duration_hours) if fcr_enabled else 0.0
    if reserve_h > 0 or reserve_h_fcr > 0:
        for t in range(T):
            b = block_of[t]
            # Reconstruct the soc[t] row (accumulated flows to time t).
            # α × r terms are expected drain — scaled by bid_win_rate_lp
            # to match the (expected) SoC trajectory the LP optimises.
            # The reservation-headroom term itself uses FULL r (operator
            # must reserve capacity in case the bid clears). FCR
            # contributes nothing to SoC dynamics here — zero-net-energy
            # by ADR-006 simplification — but adds symmetric reservation
            # headroom on both sides.
            soc_row = np.zeros(n_vars)
            for tau in range(t + 1):
                bb = block_of[tau]
                soc_row[idx_c_da.start + tau] += eta * dt_hours
                soc_row[idx_c_ida1.start + tau] += eta * dt_hours    # Model B: IDA1 charge → SoC
                soc_row[idx_c_id.start + tau] += eta * dt_hours
                soc_row[idx_d_da.start + tau] += -dt_hours / eta
                soc_row[idx_d_ida1.start + tau] += -dt_hours / eta   # Model B: IDA1 discharge → SoC
                soc_row[idx_d_id.start + tau] += -dt_hours / eta
                soc_row[idx_r_neg_block.start + bb] += (
                    afrr_activation_rate_neg[tau] * eta * dt_hours * activation_eff
                )
                soc_row[idx_r_pos_block.start + bb] += (
                    -afrr_activation_rate_pos[tau] * dt_hours / eta * activation_eff
                )

            # FCR-side adjustment: in Stage 2 locked mode, the operator must
            # hold SoC headroom for the locked FCR bid (no FCR shortfall
            # mechanic — symmetric capacity is fully obligated). Move the
            # locked × reserve_h_fcr term to the RHS as a constant. In
            # Stage 1 / single-stage, r_fcr is the LP variable.
            if reserve_h_fcr > 0 and use_locked_stage2:
                fcr_headroom_const = float(r_fcr_locked_arr[b]) * reserve_h_fcr / eta
                fcr_top_const = float(r_fcr_locked_arr[b]) * reserve_h_fcr * eta
            else:
                fcr_headroom_const = 0.0
                fcr_top_const = 0.0

            # Bottom: soc[t] − r_pos × reserve_h_a/η − r_fcr × reserve_h_f/η ≥ soc_min
            row = -soc_row.copy()
            if reserve_h > 0:
                row[idx_r_pos_block.start + b] += reserve_h / eta
            if reserve_h_fcr > 0 and not use_locked_stage2:
                row[idx_r_fcr_block.start + b] += reserve_h_fcr / eta
            A_rows.append(row)
            b_rows.append(soc_init - soc_min - fcr_headroom_const)

            # Top: soc[t] + r_neg × reserve_h_a × η + r_fcr × reserve_h_f × η ≤ soc_max
            row = soc_row.copy()
            if reserve_h > 0:
                row[idx_r_neg_block.start + b] += reserve_h * eta
            if reserve_h_fcr > 0 and not use_locked_stage2:
                row[idx_r_fcr_block.start + b] += reserve_h_fcr * eta
            A_rows.append(row)
            b_rows.append(soc_max - soc_init - fcr_top_const)

    # 4) Cycle cap: total discharge energy (DA + IDA1 + IDA2/ID +
    # pos activation) ≤ max_cycles × usable_energy.
    # Market-aware mode: aFRR activation contribution scaled by
    # bid_win_rate_lp (expected throughput from cleared fraction).
    cycle_row = np.zeros(n_vars)
    cycle_row[idx_d_da] = dt_hours
    cycle_row[idx_d_ida1] = dt_hours       # Model B: IDA1 discharge counts toward cycle cap
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
            end_row[idx_c_ida1.start + tau] += eta * dt_hours    # Model B: IDA1 charge → SoC
            end_row[idx_c_id.start + tau] += eta * dt_hours
            end_row[idx_d_da.start + tau] += -dt_hours / eta
            end_row[idx_d_ida1.start + tau] += -dt_hours / eta   # Model B: IDA1 discharge → SoC
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
    # Bounds for c_da, d_da (positions 0..2T): [0, P_max].
    bounds = [(0.0, power_mw)] * (2 * T)
    # Model B: bounds for c_ida1, d_ida1 (positions 2T..4T).
    # When IDA1 is enabled, free in [0, P_max]. When locked (Stage 2b),
    # equality bounds pin to the Stage 2a values. When disabled (no
    # prices_ida1, no locks), bound to [0, 0] so the LP reduces to
    # Model A+ behaviour.
    if c_ida1_locked_arr is not None:
        for t in range(T):
            v = float(c_ida1_locked_arr[t])
            bounds.append((v, v))
    elif ida1_enabled:
        bounds += [(0.0, power_mw)] * T
    else:
        bounds += [(0.0, 0.0)] * T
    if d_ida1_locked_arr is not None:
        for t in range(T):
            v = float(d_ida1_locked_arr[t])
            bounds.append((v, v))
    elif ida1_enabled:
        bounds += [(0.0, power_mw)] * T
    else:
        bounds += [(0.0, 0.0)] * T
    # Bounds for c_id, d_id (positions 4T..6T): [0, P_max].
    bounds += [(0.0, power_mw)] * (2 * T)
    if use_locked_stage2:
        for b in range(B):
            bounds.append((0.0, float(r_pos_locked_arr[b])))
        for b in range(B):
            bounds.append((0.0, float(r_neg_locked_arr[b])))
        # ADR-006: FCR locked from Stage 1 (zero when FCR not in use).
        for b in range(B):
            bounds.append((0.0, float(r_fcr_locked_arr[b])))
    else:
        # ADR-002c: must-bid floor on r_pos / r_neg per direction. Clamp
        # r_min to r_cap to keep bounds feasible.
        r_min = max(0.0, min(float(r_min_per_block_mw), r_cap))
        bounds += [(r_min, r_cap)] * (2 * B)
        # ADR-006: FCR bounds. Disabled (no fcr_cap_price) → [0, 0]. Enabled
        # → [r_min_fcr, P_max] (FCR doesn't compete with aFRR via the
        # max_afrr_participation knob — that knob is being retired anyway).
        if fcr_enabled:
            r_fcr_min = max(0.0, min(float(r_min_fcr_per_block_mw), power_mw))
            bounds += [(r_fcr_min, power_mw)] * B
        else:
            bounds += [(0.0, 0.0)] * B

    result = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not result.success:
        return _failed_result(
            T, energy_mwh, soc_init,
            f"linprog failed: {result.message}",
        )

    x = result.x
    c_da = x[idx_c_da]
    d_da = x[idx_d_da]
    c_ida1 = x[idx_c_ida1]
    d_ida1 = x[idx_d_ida1]
    c_id = x[idx_c_id]
    d_id = x[idx_d_id]
    r_pos_block = x[idx_r_pos_block]
    r_neg_block = x[idx_r_neg_block]
    r_fcr_block = x[idx_r_fcr_block]

    # Broadcast block-level reservations to per-interval. r_pos / r_neg /
    # r_fcr remain the *bid* (= LP variable, what the operator commits to
    # in the auction). a_pos / a_neg = α × r are the *full-clear*
    # activations (what the operator delivers if their aFRR bid clears).
    # FCR has no activation by simplification (zero-net-energy).
    r_pos = r_pos_block[block_of]
    r_neg = r_neg_block[block_of]
    r_fcr = r_fcr_block[block_of]
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
        r_fcr_expected = float(bid_win_rate_lp) * r_fcr
    else:
        r_pos_expected = None
        r_neg_expected = None
        a_pos_expected = None
        a_neg_expected = None
        r_fcr_expected = None

    # SoC trajectory uses expected α drain (× activation_eff) — matches
    # the LP's SoC dynamics constraint. Charge / discharge across all
    # auction streams (DA, IDA1, IDA2/ID) contributes to the SoC walk.
    a_pos_soc = a_pos_expected if a_pos_expected is not None else a_pos
    a_neg_soc = a_neg_expected if a_neg_expected is not None else a_neg
    soc_changes = (
        (c_da + c_ida1 + c_id + a_neg_soc) * eta * dt_hours
        - (d_da + d_ida1 + d_id + a_pos_soc) / eta * dt_hours
    )
    soc = soc_init + np.cumsum(soc_changes)

    revenue_da = float(np.sum(d_da * dt_hours * prices_da * eta - c_da * dt_hours * prices_da / eta))
    if prices_ida1_arr is not None:
        revenue_ida1 = float(
            np.sum(d_ida1 * dt_hours * prices_ida1_arr * eta
                   - c_ida1 * dt_hours * prices_ida1_arr / eta)
        )
    else:
        revenue_ida1 = 0.0
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
        # ADR-006: FCR cap revenue settled on locked commitment when
        # fcr_cap_price provided. Stage 2 receives both the locked array
        # and the cap price (caller passes both).
        if fcr_cap_price is not None:
            revenue_fcr_cap = float(
                np.sum(r_fcr_locked_arr * np.asarray(fcr_cap_price, dtype=float))
                * block_hours * bid_win_rate_lp
            )
        else:
            revenue_fcr_cap = 0.0
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
        # ADR-006: FCR cap revenue from LP-chosen r_fcr.
        if fcr_cap_price is not None:
            revenue_fcr_cap = float(
                np.sum(r_fcr_block * np.asarray(fcr_cap_price, dtype=float))
                * block_hours * bid_win_rate_lp
            )
        else:
            revenue_fcr_cap = 0.0

    # ADR-002 MVP — auction-access cost on the bid commitment. In Stage-2
    # (locked) mode the bid is the locked Stage-1 commitment; in
    # single-stage mode the bid is the LP-chosen r_pos_block / r_neg_block.
    # ADR-009: per-block hurdle array overrides scalar in revenue accounting.
    if afrr_bid_hurdle_per_block_eur_per_mw_h is not None:
        _accrued_hurdle = np.asarray(
            afrr_bid_hurdle_per_block_eur_per_mw_h, dtype=float
        )
    elif afrr_bid_hurdle_eur_per_mw_h:
        _accrued_hurdle = np.full(B, float(afrr_bid_hurdle_eur_per_mw_h))
    else:
        _accrued_hurdle = None
    if _accrued_hurdle is not None:
        if use_locked_stage2:
            bid_for_hurdle_pos = r_pos_locked_arr
            bid_for_hurdle_neg = r_neg_locked_arr
        else:
            bid_for_hurdle_pos = r_pos_block
            bid_for_hurdle_neg = r_neg_block
        bid_hurdle_cost_eur = float(
            ((bid_for_hurdle_pos + bid_for_hurdle_neg) * _accrued_hurdle).sum()
            * block_hours
        )
    else:
        bid_hurdle_cost_eur = 0.0

    # ADR-006: FCR-side bid hurdle (mirrors aFRR mechanic).
    if fcr_bid_hurdle_eur_per_mw_h and fcr_enabled:
        if use_locked_stage2:
            bid_for_hurdle_fcr = r_fcr_locked_arr
        else:
            bid_for_hurdle_fcr = r_fcr_block
        fcr_bid_hurdle_cost_eur = float(
            bid_for_hurdle_fcr.sum() * block_hours * fcr_bid_hurdle_eur_per_mw_h
        )
    else:
        fcr_bid_hurdle_cost_eur = 0.0

    revenue_total = (
        revenue_da + revenue_ida1 + revenue_id
        + revenue_afrr_cap_pos + revenue_afrr_cap_neg
        + revenue_afrr_energy_pos + revenue_afrr_energy_neg
        + revenue_fcr_cap
        - shortfall_cost
        - bid_hurdle_cost_eur
        - fcr_bid_hurdle_cost_eur
    )

    # FEC binds on expected throughput when market-aware (matches the
    # cycle-cap constraint inside the LP). All auction discharge streams
    # contribute (DA + IDA1 + IDA2/ID + activation).
    total_discharge_mwh = float(
        np.sum((d_da + d_ida1 + d_id + a_pos_soc) * dt_hours)
    )
    fec = total_discharge_mwh / energy_mwh

    return StackedDispatchResult(
        charge_da=c_da, discharge_da=d_da,
        charge_ida1=c_ida1, discharge_ida1=d_ida1,
        charge_id=c_id, discharge_id=d_id,
        r_pos=r_pos, r_neg=r_neg, a_pos=a_pos, a_neg=a_neg,
        r_fcr=r_fcr,
        soc=soc,
        revenue_da=revenue_da, revenue_ida1=revenue_ida1, revenue_id=revenue_id,
        revenue_afrr_cap_pos=revenue_afrr_cap_pos,
        revenue_afrr_cap_neg=revenue_afrr_cap_neg,
        revenue_afrr_energy_pos=revenue_afrr_energy_pos,
        revenue_afrr_energy_neg=revenue_afrr_energy_neg,
        revenue_fcr_cap=revenue_fcr_cap,
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
        r_fcr_expected=r_fcr_expected,
        bid_hurdle_cost_eur=bid_hurdle_cost_eur,
        fcr_bid_hurdle_cost_eur=fcr_bid_hurdle_cost_eur,
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
    # ADR-009 (price-conditional bid hurdle): see optimize_day_stacked.
    afrr_bid_hurdle_per_block_eur_per_mw_h: np.ndarray | None = None,
    # ADR-002c — settlement model: must-bid floor + activation-energy
    # capture haircut. Stage 1 sees the must-bid floor (the LP commits
    # at least r_min per direction every block); Stage 2 inherits the
    # locked Stage-1 r values, so r_min has no direct Stage-2 effect.
    # Both stages apply the capture factor to all α-driven terms.
    r_min_per_block_mw: float = 0.0,
    afrr_energy_capture_factor: float = 1.0,
    # ADR-006: FCR full LP integration. Stage 1 chooses r_fcr per block
    # using fcr_cap_price (cap auction clears at D−1 alongside aFRR);
    # Stage 2 inherits the locked Stage-1 r_fcr commitment. When
    # ``fcr_cap_price is None``, FCR is disabled and the LP behaves
    # identically to the pre-ADR-006 dispatch (backward-compatible).
    # ``fcr_reserve_duration_hours`` is the symmetric SoC headroom
    # horizon (default 0.5 h matches German PRL pre-qualification).
    fcr_cap_price: np.ndarray | None = None,
    fcr_reserve_duration_hours: float = 0.5,
    r_min_fcr_per_block_mw: float = 0.0,
    fcr_bid_hurdle_eur_per_mw_h: float = 0.0,
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
        max_afrr_participation=max_afrr_participation,
        bid_win_rate_lp=bid_win_rate_lp,                          # market-aware ex-ante clearing
        afrr_bid_hurdle_eur_per_mw_h=afrr_bid_hurdle_eur_per_mw_h,  # ADR-002 MVP discipline on Stage 1 bid
        afrr_bid_hurdle_per_block_eur_per_mw_h=afrr_bid_hurdle_per_block_eur_per_mw_h,  # ADR-009
        r_min_per_block_mw=r_min_per_block_mw,                    # ADR-002c must-bid floor (Stage 1 only)
        afrr_energy_capture_factor=afrr_energy_capture_factor,    # ADR-002c capture haircut
        # ADR-006: FCR enabled when caller passes fcr_cap_price.
        fcr_cap_price=fcr_cap_price,
        fcr_reserve_duration_hours=fcr_reserve_duration_hours,
        r_min_fcr_per_block_mw=r_min_fcr_per_block_mw,
        fcr_bid_hurdle_eur_per_mw_h=fcr_bid_hurdle_eur_per_mw_h,
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
    r_fcr_locked = stage1.r_fcr[::PERIODS_PER_BLOCK].copy()
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
        max_afrr_participation=max_afrr_participation,           # honoured at Stage 1; locked bid takes over here
        r_pos_locked_per_block=r_pos_locked,
        r_neg_locked_per_block=r_neg_locked,
        bid_win_rate_lp=stage2_bid_win_rate_lp,
        afrr_bid_hurdle_eur_per_mw_h=afrr_bid_hurdle_eur_per_mw_h,  # ADR-002 — sunk on locked bid; deducted in revenue accounting
        afrr_bid_hurdle_per_block_eur_per_mw_h=afrr_bid_hurdle_per_block_eur_per_mw_h,  # ADR-009
        # ADR-002c — Stage 2 does NOT need the must-bid floor (locked
        # bid from Stage 1 is the binding constraint). Capture haircut
        # passes through so realised Stage 2 energy revenue + SoC + FEC
        # see the merit-order dispatch fraction.
        r_min_per_block_mw=0.0,
        afrr_energy_capture_factor=afrr_energy_capture_factor,
        # ADR-006: FCR locked Stage 2. Cap revenue settled on D−1 locked
        # commitment; LP variable bounded by locked array.
        fcr_cap_price=fcr_cap_price,
        fcr_reserve_duration_hours=fcr_reserve_duration_hours,
        r_fcr_locked_per_block=r_fcr_locked,
        # FCR bid hurdle sunk on locked bid (deducted post-solve like aFRR).
        fcr_bid_hurdle_eur_per_mw_h=fcr_bid_hurdle_eur_per_mw_h,
        mfrr_penalty_pos_eur_per_mwh=mfrr_penalty_pos_eur_per_mwh,
        mfrr_penalty_neg_eur_per_mwh=mfrr_penalty_neg_eur_per_mwh,
        initial_soc_mwh=initial_soc_mwh,
        cyclic_soc=cyclic_soc,
        final_soc_target_mwh=final_soc_target_mwh,
    )
    return TwoStageDispatchResult(stage1=stage1, stage2=stage2)


# ─────────────────────────────────────────────────────────────────────────
# Model B (multi-window IDA1 + IDA2) — three-stage dispatch
# ─────────────────────────────────────────────────────────────────────────


def forecast_ida_from_da(prices_da: np.ndarray) -> np.ndarray:
    """Baseline IDA forecast at D−1 morning: use DA prices as the proxy.

    DA cleared at D−1 12:00; IDA1 clears at D−1 16:00, IDA2 at D−1 22:30.
    Before any IDA auction prints, DA is the best public predictor.
    Empirical: corr(DA, IDA1) ≈ 0.93, corr(DA, IDA2) ≈ 0.94 on DE 2024–2026
    15-min slots; mean(|DA − IDA|) ≈ €7/MWh.

    Returns a copy of ``prices_da`` (same shape, EUR/MWh).
    """
    return np.asarray(prices_da, dtype=float).copy()


def forecast_ida2_from_ida1(prices_ida1: np.ndarray) -> np.ndarray:
    """Stage 2a IDA2 forecast at D−1 16:00: use realised IDA1 as the proxy.

    IDA1 has just cleared at D−1 16:00; IDA2 doesn't print until D−1 22:30.
    Empirical: corr(IDA1, IDA2) ≈ 0.95, mean |IDA1 − IDA2| ≈ €4/MWh on
    DE 2024–2026 15-min slots. IDA1 is the best public predictor of IDA2
    available at the IDA1-decision moment.

    Returns a copy of ``prices_ida1`` (same shape, EUR/MWh).

    A regression-based refinement (e.g. ``IDA2_hat = a + b·DA + c·IDA1 +
    d·load_residual``) is a candidate for future work; the identity proxy
    is the cleanest defensible baseline that doesn't leak realised IDA2
    information into the Stage 2a LP.
    """
    return np.asarray(prices_ida1, dtype=float).copy()


def optimize_day_three_stage(
    *,
    prices_da: np.ndarray,
    prices_id: np.ndarray,                          # realised IDA2 prices (Model B convention)
    prices_ida1: np.ndarray,                         # realised IDA1 prices
    afrr_cap_pos_price: np.ndarray,
    afrr_cap_neg_price: np.ndarray,
    afrr_energy_pos_price: np.ndarray,               # realised
    afrr_energy_neg_price: np.ndarray,               # realised
    afrr_activation_rate_pos: np.ndarray,            # realised α
    afrr_activation_rate_neg: np.ndarray,            # realised α
    afrr_activation_forecast_pos: np.ndarray,        # Stage 1 + Stage 2a input
    afrr_activation_forecast_neg: np.ndarray,
    afrr_energy_forecast_pos: np.ndarray,            # Stage 1 + Stage 2a input
    afrr_energy_forecast_neg: np.ndarray,
    # Forecasts of IDA prices at each decision moment. If None, falls back to
    # DA-proxy / IDA1-proxy heuristics defined above.
    prices_ida1_forecast: np.ndarray | None = None,        # Stage 1 only
    prices_ida2_forecast_stage1: np.ndarray | None = None,  # Stage 1 only
    prices_ida2_forecast_stage2a: np.ndarray | None = None, # Stage 2a only
    energy_mwh: float = 1.0,
    power_mw: float = 1.0,
    rte: float = 0.85,
    soc_min_frac: float = 0.05,
    soc_max_frac: float = 0.95,
    max_cycles: float = 2.0,
    dt_hours: float = DT_HOURS_DEFAULT,
    afrr_reserve_duration_hours: float = 1.0,
    fcr_cap_price: np.ndarray | None = None,
    fcr_reserve_duration_hours: float = 0.5,
    r_min_fcr_per_block_mw: float = 0.0,
    fcr_bid_hurdle_eur_per_mw_h: float = 0.0,
    wear_cost_eur_per_mwh: np.ndarray | None = None,
    afrr_wear_premium_eur_per_mwh: float = 0.0,
    mfrr_penalty_pos_eur_per_mwh: float = 500.0,
    mfrr_penalty_neg_eur_per_mwh: float = 500.0,
    bid_win_rate_lp: float = 1.0,
    afrr_bid_hurdle_eur_per_mw_h: float = 0.0,
    # ADR-009 (price-conditional bid hurdle): see optimize_day_stacked.
    afrr_bid_hurdle_per_block_eur_per_mw_h: np.ndarray | None = None,
    r_min_per_block_mw: float = 0.0,
    afrr_energy_capture_factor: float = 1.0,
    # Hard cap on aFRR participation (Option 1 candidate). LP's r_pos[b] /
    # r_neg[b] upper-bounded at ``max_afrr_participation × power_mw``.
    # Default 1.0 = LP-unconstrained; <1.0 captures real-fleet
    # operational reserve (BoS / outages / arb-side capacity reservation).
    max_afrr_participation: float = 1.0,
    initial_soc_mwh: float | None = None,
    cyclic_soc: bool = True,
    final_soc_target_mwh: float | None = None,
    stage2b_perfect_foresight: bool = True,
) -> ThreeStageDispatchResult:
    """Three-stage dispatch — Model B multi-window IDA1 + IDA2 with honest
    information-set partition.

    * **Stage 1 (D−1 08:00):** aFRR cap auction commit. LP sees realised
      DA + aFRR cap prices (cap auction cleared at 08:00); forecasts of
      α, aFRR energy, IDA1, IDA2. Output: r_pos[b], r_neg[b], r_fcr[b].
    * **Stage 2a (D−1 16:00):** IDA1 auction commit. LP sees Stage 1
      r commitments locked; realised IDA1 prices; forecasts of IDA2,
      α, aFRR energy. Output: c_ida1[t], d_ida1[t]. (DA / IDA2 /
      activation lines are scratch — re-optimised in Stage 2b.)
    * **Stage 2b (D−1 22:30+):** IDA2 auction + α activation final.

      Two information-set modes:

      - ``stage2b_perfect_foresight=True`` (default; back-compat): LP sees
        Stage 1 r locked, Stage 2a IDA1 trades locked; **all prices and
        α realised** — the LP makes day-of-delivery decisions with
        perfect foresight on activation rate and aFRR-energy prices.
        Reported ``revenue_afrr_energy_*`` is the realised revenue
        because LP and reality agree.

      - ``stage2b_perfect_foresight=False`` (ADR-008 imperfect-foresight
        mode): LP makes Stage 2b decisions on **forecast α + forecast
        aFRR-energy** (same forecasts Stage 2a used). Auction prices
        (DA, IDA1, IDA2) remain realised because they cleared before
        delivery. The LP's reported ``revenue_afrr_energy_*`` is the
        operator's *expected* revenue under their forecasts; the caller
        must use :func:`recompute_realised_afrr_energy_revenue` against
        the realised arrays to obtain the cash-realised value
        (locked-bid × realised activation × realised energy price).
        See ADR-008 (Option A') for the editorial rationale.

    The forbidden-by-construction case is feeding realised IDA2 into the
    Stage 2a LP — it would let the solver "cheat" with inter-auction
    perfect foresight. Stage 2a uses ``prices_ida2_forecast_stage2a``
    (default: identity-proxy of realised IDA1 via
    :func:`forecast_ida2_from_ida1`); the realised IDA2 is reserved for
    Stage 2b only.

    Returns :class:`ThreeStageDispatchResult` with all three stage outputs;
    headline economics live in ``stage2b``.
    """
    # ── Forecasts (default heuristics) ─────────────────────────────────
    if prices_ida1_forecast is None:
        prices_ida1_forecast = forecast_ida_from_da(prices_da)
    if prices_ida2_forecast_stage1 is None:
        prices_ida2_forecast_stage1 = forecast_ida_from_da(prices_da)
    if prices_ida2_forecast_stage2a is None:
        prices_ida2_forecast_stage2a = forecast_ida2_from_ida1(prices_ida1)

    common_kw = dict(
        afrr_cap_pos_price=afrr_cap_pos_price,
        afrr_cap_neg_price=afrr_cap_neg_price,
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
        bid_win_rate_lp=bid_win_rate_lp,
        afrr_bid_hurdle_eur_per_mw_h=afrr_bid_hurdle_eur_per_mw_h,
        afrr_bid_hurdle_per_block_eur_per_mw_h=afrr_bid_hurdle_per_block_eur_per_mw_h,
        afrr_energy_capture_factor=afrr_energy_capture_factor,
        fcr_cap_price=fcr_cap_price,
        fcr_reserve_duration_hours=fcr_reserve_duration_hours,
        fcr_bid_hurdle_eur_per_mw_h=fcr_bid_hurdle_eur_per_mw_h,
        initial_soc_mwh=initial_soc_mwh,
        cyclic_soc=cyclic_soc,
        final_soc_target_mwh=final_soc_target_mwh,
    )

    # ── Stage 1 (D−1 08:00): aFRR cap commit ──────────────────────────
    stage1 = optimize_day_stacked(
        prices_da=prices_da,
        prices_id=prices_ida2_forecast_stage1,    # Stage 1 IDA2 forecast
        prices_ida1=prices_ida1_forecast,          # Stage 1 IDA1 forecast
        afrr_energy_pos_price=afrr_energy_forecast_pos,
        afrr_energy_neg_price=afrr_energy_forecast_neg,
        afrr_activation_rate_pos=afrr_activation_forecast_pos,
        afrr_activation_rate_neg=afrr_activation_forecast_neg,
        max_afrr_participation=max_afrr_participation,
        r_min_per_block_mw=r_min_per_block_mw,
        r_min_fcr_per_block_mw=r_min_fcr_per_block_mw,
        **common_kw,
    )
    if not stage1.success:
        return ThreeStageDispatchResult(stage1=stage1, stage2a=stage1, stage2b=stage1)

    # Lock aFRR commitments per-block.
    r_pos_locked = stage1.r_pos[::PERIODS_PER_BLOCK].copy()
    r_neg_locked = stage1.r_neg[::PERIODS_PER_BLOCK].copy()
    r_fcr_locked = stage1.r_fcr[::PERIODS_PER_BLOCK].copy()

    # ── Stage 2a (D−1 16:00): IDA1 commit, aFRR locked ────────────────
    # LP sees realised DA + realised IDA1 + forecast IDA2 + forecast α +
    # forecast aFRR energy. aFRR (cap + FCR) locked from Stage 1.
    stage2a = optimize_day_stacked(
        prices_da=prices_da,
        prices_id=prices_ida2_forecast_stage2a,    # Stage 2a IDA2 forecast
        prices_ida1=prices_ida1,                    # IDA1 realised
        afrr_energy_pos_price=afrr_energy_forecast_pos,
        afrr_energy_neg_price=afrr_energy_forecast_neg,
        afrr_activation_rate_pos=afrr_activation_forecast_pos,
        afrr_activation_rate_neg=afrr_activation_forecast_neg,
        max_afrr_participation=max_afrr_participation,
        r_pos_locked_per_block=r_pos_locked,
        r_neg_locked_per_block=r_neg_locked,
        r_fcr_locked_per_block=r_fcr_locked,
        mfrr_penalty_pos_eur_per_mwh=mfrr_penalty_pos_eur_per_mwh,
        mfrr_penalty_neg_eur_per_mwh=mfrr_penalty_neg_eur_per_mwh,
        # Floors only meaningful in Stage 1; Stage 2a inherits aFRR locks.
        r_min_per_block_mw=0.0,
        r_min_fcr_per_block_mw=0.0,
        **common_kw,
    )
    if not stage2a.success:
        return ThreeStageDispatchResult(stage1=stage1, stage2a=stage2a, stage2b=stage2a)

    # Lock IDA1 trades from Stage 2a.
    c_ida1_locked = stage2a.charge_ida1.copy()
    d_ida1_locked = stage2a.discharge_ida1.copy()

    # ── Stage 2b (D−1 22:30+): IDA2 + α activation ────────────────────
    # Information-set choice (ADR-008):
    # * PF=True (default): LP sees realised α + realised aFRR-energy →
    #   day-of-delivery decisions with perfect foresight on stochastic
    #   streams. Reported aFRR-energy revenue == realised cash.
    # * PF=False: LP sees forecast α + forecast aFRR-energy (same as
    #   Stage 2a inputs). Auction prices (DA, IDA1, IDA2) are realised
    #   because they clear before delivery. Reported aFRR-energy revenue
    #   is the operator's *expected* revenue under their forecasts; the
    #   caller must run :func:`recompute_realised_afrr_energy_revenue`
    #   against the realised arrays to convert to cash.
    if stage2b_perfect_foresight:
        s2b_energy_pos = afrr_energy_pos_price
        s2b_energy_neg = afrr_energy_neg_price
        s2b_alpha_pos = afrr_activation_rate_pos
        s2b_alpha_neg = afrr_activation_rate_neg
    else:
        s2b_energy_pos = afrr_energy_forecast_pos
        s2b_energy_neg = afrr_energy_forecast_neg
        s2b_alpha_pos = afrr_activation_forecast_pos
        s2b_alpha_neg = afrr_activation_forecast_neg
    stage2b = optimize_day_stacked(
        prices_da=prices_da,
        prices_id=prices_id,                        # IDA2 realised (auction cleared 22:30 D-1)
        prices_ida1=prices_ida1,                    # realised; LP variable will be pinned
        afrr_energy_pos_price=s2b_energy_pos,
        afrr_energy_neg_price=s2b_energy_neg,
        afrr_activation_rate_pos=s2b_alpha_pos,
        afrr_activation_rate_neg=s2b_alpha_neg,
        max_afrr_participation=max_afrr_participation,
        r_pos_locked_per_block=r_pos_locked,
        r_neg_locked_per_block=r_neg_locked,
        r_fcr_locked_per_block=r_fcr_locked,
        c_ida1_locked_per_interval=c_ida1_locked,
        d_ida1_locked_per_interval=d_ida1_locked,
        mfrr_penalty_pos_eur_per_mwh=mfrr_penalty_pos_eur_per_mwh,
        mfrr_penalty_neg_eur_per_mwh=mfrr_penalty_neg_eur_per_mwh,
        r_min_per_block_mw=0.0,
        r_min_fcr_per_block_mw=0.0,
        **common_kw,
    )
    return ThreeStageDispatchResult(stage1=stage1, stage2a=stage2a, stage2b=stage2b)


def compute_soc_violation_penalty(
    *,
    stage_result,                                  # StackedDispatchResult
    afrr_activation_forecast_pos: np.ndarray,      # (T,) α the LP saw
    afrr_activation_forecast_neg: np.ndarray,
    afrr_activation_realised_pos: np.ndarray,      # (T,) what actually happened
    afrr_activation_realised_neg: np.ndarray,
    energy_mwh: float,
    soc_min_frac: float,
    soc_max_frac: float,
    rte: float,
    bid_win_rate_lp: float = 1.0,
    afrr_energy_capture_factor: float = 1.0,
    dt_hours: float = DT_HOURS_DEFAULT,
    mfrr_penalty_pos_eur_per_mwh: float = 500.0,
    mfrr_penalty_neg_eur_per_mwh: float = 500.0,
) -> tuple[float, dict]:
    """Compute the realised-α SoC trajectory drift vs the LP's planned
    trajectory and apply mFRR-imbalance penalty for any [soc_min, soc_max]
    violations under realised conditions (ADR-010).

    Differential formulation: ``realised_soc[t] = LP_planned_soc[t] +
    cumsum(differential_changes[t])`` where
    ``differential_changes = (Δ_neg × eta − Δ_pos / eta) × dt`` with
    ``Δ_pos = activation_eff × (α_realised_pos − α_forecast_pos) × r_pos``
    and similarly for neg. By construction the drift is zero in PF mode
    (forecast == realised, penalty = 0). In noPF mode the drift
    accumulates over the day and the penalty € reflects the realised
    mFRR-imbalance cost the operator would have paid.
    """
    eta = float(rte) ** 0.5
    activation_eff = float(bid_win_rate_lp) * float(afrr_energy_capture_factor)

    delta_alpha_pos = (
        np.asarray(afrr_activation_realised_pos, dtype=float)
        - np.asarray(afrr_activation_forecast_pos, dtype=float)
    )
    delta_alpha_neg = (
        np.asarray(afrr_activation_realised_neg, dtype=float)
        - np.asarray(afrr_activation_forecast_neg, dtype=float)
    )
    r_pos = np.asarray(stage_result.r_pos, dtype=float)
    r_neg = np.asarray(stage_result.r_neg, dtype=float)
    delta_changes = (
        delta_alpha_neg * r_neg * activation_eff * eta * dt_hours
        - delta_alpha_pos * r_pos * activation_eff / eta * dt_hours
    )
    soc_drift = np.cumsum(delta_changes)
    soc_realised = np.asarray(stage_result.soc, dtype=float) + soc_drift

    soc_min = float(soc_min_frac) * float(energy_mwh)
    soc_max = float(soc_max_frac) * float(energy_mwh)
    violation_below = np.maximum(0.0, soc_min - soc_realised)
    violation_above = np.maximum(0.0, soc_realised - soc_max)

    penalty_eur = float(
        violation_below.sum() * mfrr_penalty_neg_eur_per_mwh
        + violation_above.sum() * mfrr_penalty_pos_eur_per_mwh
    )
    return penalty_eur, {
        "soc_violation_below_mwh_total": float(violation_below.sum()),
        "soc_violation_above_mwh_total": float(violation_above.sum()),
        "soc_violation_below_max_mwh": float(violation_below.max()) if violation_below.size else 0.0,
        "soc_violation_above_max_mwh": float(violation_above.max()) if violation_above.size else 0.0,
        "n_intervals_below_floor": int((violation_below > 0).sum()),
        "n_intervals_above_ceiling": int((violation_above > 0).sum()),
    }


def recompute_realised_afrr_energy_revenue(
    *,
    r_pos_per_interval: np.ndarray,
    r_neg_per_interval: np.ndarray,
    afrr_activation_realised_pos: np.ndarray,
    afrr_activation_realised_neg: np.ndarray,
    afrr_energy_realised_pos: np.ndarray,
    afrr_energy_realised_neg: np.ndarray,
    bid_win_rate_lp: float = 1.0,
    afrr_energy_capture_factor: float = 1.0,
    dt_hours: float = DT_HOURS_DEFAULT,
) -> tuple[float, float]:
    """Recompute aFRR-energy revenue for an LP whose decisions were made
    on forecast α + forecast aFRR-energy (ADR-008 imperfect-foresight Stage 2b).

    Mirrors the LP-internal formula from :func:`optimize_day_stacked`:
    ``revenue = Σ (r × α × activation_eff × dt × energy_price)`` per direction,
    where ``activation_eff = bid_win_rate_lp × afrr_energy_capture_factor``.
    The locked bid (``r_pos`` / ``r_neg`` from the LP's chosen reservation)
    is paired with **realised** α + **realised** energy price to produce
    the cash-realised revenue, even though the LP's internal accounting
    used forecasts.

    Returns ``(rev_pos_eur, rev_neg_eur)`` for the day. Caller substitutes
    these for ``stage2b.revenue_afrr_energy_pos`` and
    ``stage2b.revenue_afrr_energy_neg`` in the realised gross.
    """
    activation_eff = float(bid_win_rate_lp) * float(afrr_energy_capture_factor)
    a_pos_realised = (
        np.asarray(afrr_activation_realised_pos, dtype=float)
        * np.asarray(r_pos_per_interval, dtype=float)
        * activation_eff
    )
    a_neg_realised = (
        np.asarray(afrr_activation_realised_neg, dtype=float)
        * np.asarray(r_neg_per_interval, dtype=float)
        * activation_eff
    )
    rev_pos = float(
        np.sum(a_pos_realised * dt_hours
               * np.asarray(afrr_energy_realised_pos, dtype=float))
    )
    rev_neg = float(
        np.sum(a_neg_realised * dt_hours
               * np.asarray(afrr_energy_realised_neg, dtype=float))
    )
    return rev_pos, rev_neg
