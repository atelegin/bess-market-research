"""
Lifecycle NPV simulator for policy comparison (Note 4 A4).

Runs an annual loop over N years for a given shadow-cost policy. Each
day: market data → policy gives wear-cost vector → stacked-market LP
solves → dispatch accumulates → SoH degrades. Year-end: discount annual
revenue to present. Lifetime NPV = discounted sum.

Designed for side-by-side policy comparison: same price series, same
market mechanics, same degradation physics — the *only* difference
between runs is the policy's wear cost vector.

Price data rotation
-------------------
We have observed 2023 and 2025 regelleistung + netztransparenz +
EnergyCharts data (2024 has an unreliable DA fetch). The simulator
rotates through a list of available "template years" for each year in
the simulation horizon; by default ``[2023, 2025]`` alternating.
Physical realism of using old prices for a future year is modest but
sufficient for *relative* policy comparison which is what Note 4
cares about.

Degradation
-----------
Uses the same :func:`lib.models.dispatch.adp.solver.degradation_per_day` formula
as the DP, to keep policies consistent with the backward-induction
model. Daily SoH decrement = linear-in-FEC + mild age acceleration +
calendar fade. At end-of-life (SoH ≤ warranty_floor) the year terminates
and remaining years contribute zero revenue.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import numpy as np

from lib.analysis.stacked_day_assembler import (
    DEFAULT_POOL_NEG_MW,
    DEFAULT_POOL_POS_MW,
    assemble_day_inputs,
)
from lib.analysis.stacked_year_runner import _prefetch_year_frames
from lib.models.dispatch.adp.shadow_cost import ShadowCostPolicy
from lib.models.dispatch.adp.solver import degradation_per_day
from lib.models.dispatch.stacked import (
    PERIODS_PER_BLOCK,
    optimize_day_stacked,
    optimize_day_two_stage,
)

logger = logging.getLogger(__name__)


@dataclass
class LifecycleResult:
    """Per-policy lifetime simulation output."""

    policy_name: str
    n_years: int
    initial_soh: float
    warranty_floor: float
    discount_rate: float
    annual_revenue_eur: np.ndarray       # shape (n_years,), nominal EUR/MW
    annual_fec: np.ndarray               # shape (n_years,)
    end_of_year_soh: np.ndarray          # shape (n_years,)
    days_solved: int
    days_skipped: int
    lifetime_npv_eur: float              # discounted to present, EUR/MW
    years_to_floor: Optional[int] = None  # first year SoH hit floor; None if never
    # Per-stream annual revenue (shape (n_years,), EUR/MW). Enables
    # per-component market-trend sensitivity (Note 1 differential decline).
    annual_revenue_da_eur: Optional[np.ndarray] = None
    annual_revenue_id_eur: Optional[np.ndarray] = None
    annual_revenue_afrr_cap_eur: Optional[np.ndarray] = None
    annual_revenue_afrr_energy_eur: Optional[np.ndarray] = None
    # Optional dispatch logs for diagnostics. Populated only when
    # ``collect_diagnostics_year`` passed to simulate_lifecycle.
    diagnostic_days: Optional[list] = None  # list[DayDiagnosticData] — first collected year

    def revenue_by_year_keur(self) -> np.ndarray:
        return self.annual_revenue_eur / 1000.0


def _rotation_year(sim_year_idx: int, template_years: tuple[int, ...]) -> int:
    return template_years[sim_year_idx % len(template_years)]


def simulate_lifecycle(
    policy: ShadowCostPolicy,
    n_years: int = 15,
    template_years: tuple[int, ...] = (2023, 2025),
    initial_soh: float = 1.0,
    warranty_floor: float = 0.80,
    discount_rate: float = 0.07,
    power_mw: float = 1.0,
    duration_h: float = 2.0,
    rte: float = 0.85,
    max_cycles: float = 2.0,
    max_afrr_participation: float = 0.40,
    afrr_reserve_duration_hours: float = 1.0,
    pool_pos_mw: float = DEFAULT_POOL_POS_MW,
    pool_neg_mw: float = DEFAULT_POOL_NEG_MW,
    fade_per_fec_at_soh_1: float = 3.3e-5,
    calendar_fade_per_day: float = 2e-5,
    use_physics_degradation: bool = False,
    physics_preset_name: str = "eve_lf280k",
    physics_temperature_c: float = 25.0,
    # Joint scale on the Note 3 physics kernel's ``k_cal`` and ``k_cyc``.
    # Default 1.0 = academic Naumann / Stanford calibration (Note 3 tripwire
    # gated). 0.66 = EVE LF280K manufacturer-endurance anchor (6000 cycles
    # to 80 % retention at 25 °C / 0.5 C / 1.0 DoD). Note 4 headline uses
    # 0.66 because the EVE 280 Ah prismatic asset class is not the same
    # physical generation as the Stanford K2 18650 cells the academic
    # kernel is fit to.
    physics_kernel_scale: float = 1.0,
    max_days_per_year: Optional[int] = None,
    prefetched_frames: Optional[dict[int, dict]] = None,
    collect_diagnostics_year: Optional[int] = None,
    yearly_revenue_scale: Optional[np.ndarray] = None,
    yearly_price_scale_wholesale: Optional[np.ndarray] = None,
    yearly_price_scale_afrr_cap: Optional[np.ndarray] = None,
    yearly_price_scale_afrr_energy: Optional[np.ndarray] = None,
    # Optional per-year multiplier on the policy's wear-cost vector,
    # applied after `policy.wear_cost()` returns. Use case: ADP-derived
    # shadow costs (M4 / M5 in Note 4) calibrated on a fixed regime go
    # stale under a price trajectory; scaling the wear vector by the
    # year-specific wholesale price multiplier (K_wholesale[y]) restores
    # the linear price-level dependence of "save SoC for high-spread
    # hours" without re-solving the intraday DP each year.
    yearly_wear_scale: Optional[np.ndarray] = None,
    dispatch_mode: str = "pf",                # 'pf' | 'rolling_12h_3h' | 'two_stage'
    rolling_horizon_hours: float = 12.0,
    rolling_step_hours: float = 3.0,
    bid_win_rate: float = 1.0,                # post-LP realised-outcome haircut on aFRR cap+energy (NOT market-aware): see in-loop block
    afrr_wear_premium_eur_per_mwh: float = 0.0,  # extra wear cost on aFRR-activation throughput (Idea 2)
    bid_win_mode: str = "deterministic",       # "deterministic" (uniform haircut) | "stochastic" (no-recourse Bernoulli) | "stochastic_recourse" (Stage 1 → Bernoulli draw → Stage 2 with awarded; only with bid_win_in_lp=True + dispatch_mode='two_stage')
    bid_win_rng_seed: Optional[int] = None,    # reproducibility for stochastic mode
    bid_win_in_lp: bool = False,               # market-aware: pass bid_win_rate to LP as ex-ante clearing prob (skip post-LP haircut)
    # Per-year `bid_win_rate` trajectory for fleet-saturation modelling.
    # When provided as a length-`n_years` array, multiplies the constant
    # `bid_win_rate` per-year (e.g. anchor at 1.0 for 2024 then decay
    # toward 0.2 by 2030 as BESS fleet overbuild ratio passes 1.5 — GB
    # DCL precedent). The effective per-year `bid_win_rate(y) =
    # bid_win_rate × yearly_bid_win_rate_scale[y]`, clamped to [0, 1].
    # Default None = no per-year decay (= constant bid_win_rate).
    yearly_bid_win_rate_scale: Optional[np.ndarray] = None,
    # ADR-002 MVP — auction-access marginal cost (EUR per MW × hour of bid
    # placed). Disciplines aFRR over-commitment after retiring
    # ``max_afrr_participation``. Calibrated against a 2D plausibility
    # corridor (Note 1 ex-FCR annual revenue ≈ 160 k€/MW/yr AND aFRR share
    # ex-FCR in [40, 55]%) — see ADR-002 and
    # ``precompute_hurdle_calibration.py``. Default 0.0 = no hurdle (legacy).
    afrr_bid_hurdle_eur_per_mw_h: float = 0.0,
    # ADR-002c — aFRR settlement model: must-bid floor (raises cap
    # capture toward fleet level) + capture haircut (haircuts activation
    # revenue toward merit-order-realistic level). Calibrated against
    # the 2D plausibility corridor (level + composition) plus a Test 3
    # on cap : energy ratio (~10 : 1 fleet target). Defaults retain
    # legacy behaviour. See ADR-002-cap-energy-diagnosis.md.
    r_min_per_block_mw: float = 0.0,
    afrr_energy_capture_factor: float = 1.0,
    # FCR phantom layer (paper-grade calibration). FCR is a separate
    # symmetric capacity-only product with very low activation rate
    # (~0 energy contribution). We model it as a fixed annual revenue
    # add + a small SoC reservation reduction, since the dispatch LP
    # itself does not endogenize FCR (deferred to ADR-006). Defaults
    # to 0 = off. Calibrated values for 2 h DE 2024:
    #   fcr_revenue_keur_per_mw_per_year=36-40 (Note 1 hist_bars 2024)
    #   fcr_soc_reservation_mw=0.10-0.15 (typical fleet allocation)
    # The trajectory `yearly_fcr_revenue_scale[year]` decays as fleet
    # overbuild ratio passes 1.4 (already true in 2024 — FCR saturated
    # ahead of aFRR). See ADR-002c §"FCR phantom layer".
    fcr_revenue_keur_per_mw_per_year: float = 0.0,
    fcr_soc_reservation_mw: float = 0.0,
    yearly_fcr_revenue_scale: Optional[np.ndarray] = None,
    # Two-stage dispatch (ADR-001) parameters. Used only when dispatch_mode='two_stage'.
    two_stage_n_regimes: int = 3,
    two_stage_alpha_forecast: str = "regime_mean",   # "regime_mean" | "perfect"
    two_stage_mfrr_penalty_pos_eur_per_mwh: float = 500.0,
    two_stage_mfrr_penalty_neg_eur_per_mwh: float = 500.0,
    # ADR-001 v1.2 — Picard fixed-point iteration for two-pass policies
    # under two-stage dispatch. When > 0, instead of v1.1's bounded
    # Stage-2-only re-pricing (Stage-1 r locked from pass-1), iterate
    # full Stage-1+Stage-2 with refined wear until aFRR commitments
    # stabilise or max iterations reached. Only fires when policy
    # uses_two_pass=True and dispatch_mode='two_stage'. Convergence
    # criterion: ||r_new - r_old||_inf < two_stage_picard_tol_mw.
    two_stage_picard_max_iter: int = 0,
    two_stage_picard_tol_mw: float = 0.01,
) -> LifecycleResult:
    """
    Simulate lifetime revenue under a shadow-cost policy.

    Args:
        policy: A :class:`ShadowCostPolicy` that produces per-day wear
            cost vectors given ``(soh_current, day_of_year)``.
        n_years: Horizon in years (typical 10–15 for LFP).
        template_years: Historical years whose market data is replayed
            in rotation. Default ``(2023, 2025)`` — both have complete
            regelleistung + netztransparenz + EnergyCharts coverage.
        initial_soh: Starting state of health.
        warranty_floor: Below this SoH, year terminates and subsequent
            years contribute zero revenue.
        discount_rate: Annual discount (default 7% — typical merchant-
            BESS hurdle rate).
        power_mw / duration_h / rte: Asset specs (nominal, 2h LFP 1 MW).
        max_cycles: Daily cycling cap in the LP (same for all policies
            — differentiation comes from the wear cost vector).
        max_afrr_participation: aFRR reservation cap (same as Note 1
            calibrated ≈ 0.40).
        fade_per_fec_at_soh_1 / calendar_fade_per_day: Degradation
            parameters (shared with ADP solver).
        max_days_per_year: Cap days per year for fast smoke testing
            (default None = full year).
        prefetched_frames: If provided, reuse these per-year frame
            dicts instead of re-fetching. Keyed by ``template_year``.
            Multi-policy comparisons should pass the same dict to all
            calls for speed + identical inputs.
        collect_diagnostics_year: 0-based year index whose dispatch
            days should be retained on the result for diagnostic
            signal computation. Memory-cheap: one year × 96 floats × 4
            arrays × ~365 days ≈ 5 MB.
        yearly_revenue_scale: Optional per-year revenue multiplier
            (shape ``(n_years,)``), typically from Note 1's
            ``project_full_stack`` output — reflects fleet-size-driven
            market saturation / demand growth over the operating
            horizon. ``None`` (default) = flat 2023/2025 rotation
            (no market drift). Multiplies the LP's revenue *after*
            dispatch; trader does not see the future prices and does
            not adjust dispatch. Captures the NPV-level implication
            of a declining market without re-solving the LP.

    Returns:
        :class:`LifecycleResult` with per-year revenue, SoH trajectory,
        FEC count, and discounted lifetime NPV.
    """
    annual_revenue = np.zeros(n_years)
    annual_revenue_da = np.zeros(n_years)
    annual_revenue_id = np.zeros(n_years)
    annual_revenue_afrr_cap = np.zeros(n_years)
    annual_revenue_afrr_energy = np.zeros(n_years)
    end_of_year_soh = np.zeros(n_years)
    annual_fec = np.zeros(n_years)
    days_solved = 0
    days_skipped = 0
    years_to_floor: Optional[int] = None

    current_soh = float(initial_soh)
    diagnostic_days: list = []

    # RNG for stochastic bid-win clearing (used per-day per-block in the day loop).
    # Persistent across days within one simulate_lifecycle call so that the
    # full lifetime sees independent draws; reproducible via bid_win_rng_seed.
    _bid_win_rng = np.random.default_rng(bid_win_rng_seed)

    # Dispatch mode dispatcher
    use_rolling = dispatch_mode in ("rolling_12h_3h", "rolling")
    use_two_stage = dispatch_mode == "two_stage"
    if use_rolling and use_two_stage:
        raise ValueError("dispatch_mode cannot be both rolling and two_stage")
    if use_rolling:
        from lib.analysis.rolling_horizon import run_rolling_mpc_day
        # 2-pass policies (DoD rainflow / physics-from-duty) cannot run inside
        # rolling MPC — they require observing the full-day dispatch then
        # re-pricing. Disable transparently with a warning.
        if getattr(policy, "uses_two_pass", False):
            logger.warning(
                f"Policy {policy.name} requires 2-pass; disabled in rolling-horizon mode"
            )

    # Two-stage dispatch (ADR-001): regime classification + α / energy
    # forecasts per template year. Computed once; reused per day.
    regime_by_year: dict[int, "RegimeClassification"] = {}
    if use_two_stage:
        from lib.analysis.afrr_forecast import (
            perfect_foresight_forecast,
            regime_conditional_alpha_forecast,
            regime_conditional_energy_price_forecast,
        )
        from lib.models.price_regime import fit_regimes
        if max_afrr_participation != 1.0:
            logger.info(
                "dispatch_mode='two_stage': overriding max_afrr_participation "
                f"({max_afrr_participation}) → 1.0 (commitment cap is emergent "
                "from the Stage 1 LP given afrr_reserve_duration_hours)."
            )

    # Optional Note 3 physics kernel
    physics_preset = None
    if use_physics_degradation:
        from lib.models.degradation.simple import PRESETS
        physics_preset = PRESETS[physics_preset_name]
        from lib.analysis.physics_degradation import physics_degradation_per_day

    # Prefetch frames per template year once.
    frames_by_year: dict[int, dict] = dict(prefetched_frames) if prefetched_frames else {}
    for ty in template_years:
        if ty not in frames_by_year:
            frames_by_year[ty] = _prefetch_year_frames(ty)

    if use_two_stage:
        for ty in template_years:
            da_frame = frames_by_year[ty].get("da")
            if da_frame is None or da_frame.empty:
                raise ValueError(
                    f"two_stage requires DA frame for template year {ty}; got empty/None"
                )
            regime_by_year[ty] = fit_regimes(
                prices=da_frame["price_eur_mwh"],
                n_regimes=two_stage_n_regimes,
            )

    for y_idx in range(n_years):
        if current_soh <= warranty_floor + 1e-9:
            # Warranty floor hit; no further operation
            end_of_year_soh[y_idx] = current_soh
            if years_to_floor is None:
                years_to_floor = y_idx
            continue

        # Per-year `bid_win_rate` trajectory (fleet-saturation modelling).
        # When `yearly_bid_win_rate_scale` is provided, multiplies the
        # constant `bid_win_rate` per-year. Clamped to [0, 1] for safety.
        if yearly_bid_win_rate_scale is not None:
            scale_y = float(yearly_bid_win_rate_scale[y_idx]) \
                if y_idx < len(yearly_bid_win_rate_scale) else 1.0
            bid_win_rate_y = float(np.clip(bid_win_rate * scale_y, 0.0, 1.0))
        else:
            bid_win_rate_y = float(bid_win_rate)

        template = _rotation_year(y_idx, template_years)
        frames = frames_by_year[template]
        start_date = date(template, 1, 1)
        end_date = date(template + 1, 1, 1)
        current = start_date
        day_idx = 0
        year_revenue = 0.0
        year_rev_da = 0.0
        year_rev_id = 0.0
        year_rev_afrr_cap = 0.0
        year_rev_afrr_energy = 0.0
        year_fec = 0.0
        while current < end_date:
            if max_days_per_year is not None and day_idx >= max_days_per_year:
                break
            inputs = assemble_day_inputs(
                target_date=current,
                pool_pos_mw=pool_pos_mw, pool_neg_mw=pool_neg_mw,
                da_frame=frames.get("da"), id_frame=frames.get("id"),
                spot_frame=frames.get("spot"),
                afrr_cap_frame=frames.get("afrr_cap"),
                afrr_energy_frame=frames.get("afrr_energy"),
                activations_frame=frames.get("activations"),
                ida_frame=frames.get("ida"),
            )
            if inputs is None:
                days_skipped += 1
                current += timedelta(days=1)
                day_idx += 1
                continue

            # Optional per-year per-stream price scaling — lets dispatch
            # adapt to projected market evolution. Note 1's per-stream
            # trajectory: aFRR cap collapses, wholesale grows. A rational
            # trader observing 2030 prices reallocates dispatch from aFRR
            # to wholesale; this scaling models that adaptation by
            # presenting the LP with year-y prices instead of the template
            # year prices. Shadow costs remain calibrated to 2024.
            if (yearly_price_scale_wholesale is not None
                    or yearly_price_scale_afrr_cap is not None
                    or yearly_price_scale_afrr_energy is not None):
                w = (yearly_price_scale_wholesale[y_idx]
                     if yearly_price_scale_wholesale is not None else 1.0)
                ac = (yearly_price_scale_afrr_cap[y_idx]
                      if yearly_price_scale_afrr_cap is not None else 1.0)
                ae = (yearly_price_scale_afrr_energy[y_idx]
                      if yearly_price_scale_afrr_energy is not None else 1.0)
                inputs = type(inputs)(
                    date=inputs.date,
                    prices_da=inputs.prices_da * w,
                    prices_id=inputs.prices_id * w,
                    afrr_cap_pos_price=inputs.afrr_cap_pos_price * ac,
                    afrr_cap_neg_price=inputs.afrr_cap_neg_price * ac,
                    afrr_energy_pos_price=inputs.afrr_energy_pos_price * ae,
                    afrr_energy_neg_price=inputs.afrr_energy_neg_price * ae,
                    afrr_activation_rate_pos=inputs.afrr_activation_rate_pos,
                    afrr_activation_rate_neg=inputs.afrr_activation_rate_neg,
                )

            # Policy-derived per-interval wear cost + optional LP overrides
            day_of_year = (current - start_date).days + 1
            wear = policy.wear_cost(
                soh_current=current_soh,
                day_of_year=day_of_year,
                periods_per_day=96,
                duration_h=duration_h,
            )
            if yearly_wear_scale is not None:
                wear = np.asarray(wear, dtype=float) * float(yearly_wear_scale[y_idx])
            overrides = policy.lp_overrides(
                soh_current=current_soh, day_of_year=day_of_year,
            )

            # SoH-derated usable energy (SoC window scales with SoH)
            usable_energy_mwh = power_mw * duration_h * current_soh

            # Market-aware bid_win_rate: when bid_win_in_lp is True, the LP
            # optimises expected revenue at ex-ante clearing probability
            # (cap/energy coefficients × bid_win_rate; SoC dynamics use
            # expected drain). The post-LP haircut block below is skipped
            # in this mode to avoid double-counting.
            bid_win_lp_eff = bid_win_rate_y if bid_win_in_lp else 1.0

            lp_kwargs = dict(
                energy_mwh=usable_energy_mwh, power_mw=power_mw, rte=rte,
                max_cycles=max_cycles,
                afrr_reserve_duration_hours=afrr_reserve_duration_hours,
                max_afrr_participation=max_afrr_participation,
                wear_cost_eur_per_mwh=wear,
                afrr_wear_premium_eur_per_mwh=afrr_wear_premium_eur_per_mwh,
                bid_win_rate_lp=bid_win_lp_eff,
                afrr_bid_hurdle_eur_per_mw_h=afrr_bid_hurdle_eur_per_mw_h,
                r_min_per_block_mw=r_min_per_block_mw,
                afrr_energy_capture_factor=afrr_energy_capture_factor,
            )
            lp_kwargs.update(overrides)  # policy-level LP parameter overrides

            if use_rolling:
                # Rolling-horizon MPC: 12h forecast, 3h commit, 8 re-plans/day
                day_out = run_rolling_mpc_day(
                    inputs=inputs,
                    energy_mwh=usable_energy_mwh, power_mw=power_mw, rte=rte,
                    soc_min_frac=overrides.get("soc_min_frac", 0.05),
                    soc_max_frac=overrides.get("soc_max_frac", 0.95),
                    max_cycles=max_cycles,
                    afrr_reserve_duration_hours=afrr_reserve_duration_hours,
                    max_afrr_participation=max_afrr_participation,
                    wear_cost_eur_per_mwh=wear,
                    horizon_hours=rolling_horizon_hours,
                    step_hours=rolling_step_hours,
                )
            elif use_two_stage:
                regime = regime_by_year[template]
                if two_stage_alpha_forecast == "perfect":
                    alpha_pos_fcst, alpha_neg_fcst = perfect_foresight_forecast(
                        inputs.afrr_activation_rate_pos,
                        inputs.afrr_activation_rate_neg,
                    )
                    e_pos_fcst, e_neg_fcst = perfect_foresight_forecast(
                        inputs.afrr_energy_pos_price,
                        inputs.afrr_energy_neg_price,
                    )
                else:
                    alpha_pos_fcst, alpha_neg_fcst = regime_conditional_alpha_forecast(
                        target_date=current,
                        regime_classification=regime,
                        activations_frame=frames["activations"],
                        pool_pos_mw=pool_pos_mw,
                        pool_neg_mw=pool_neg_mw,
                    )
                    e_pos_fcst, e_neg_fcst = regime_conditional_energy_price_forecast(
                        target_date=current,
                        regime_classification=regime,
                        energy_frame=frames["afrr_energy"],
                    )
                # Phase 2.3 stochastic-recourse: draw per-block per-direction
                # Bernoulli BEFORE Stage 2 — un-cleared blocks zero out
                # r_locked so Stage 2 can re-trade the freed power on
                # DA / ID. Only meaningful with bid_win_in_lp=True (the
                # bidder rationally accounts for clearing probability).
                stage2_award_pos = None
                stage2_award_neg = None
                # Gate: stochastic_recourse fires when per-year effective
                # clearing rate < 1.0 (either constant bid_win_rate < 1.0
                # or yearly_bid_win_rate_scale brings effective < 1.0).
                if (
                    bid_win_in_lp
                    and bid_win_rate_y < 1.0
                    and bid_win_mode == "stochastic_recourse"
                ):
                    stage2_award_pos = _bid_win_rng.binomial(
                        1, bid_win_rate_y, size=6
                    ).astype(float)
                    stage2_award_neg = _bid_win_rng.binomial(
                        1, bid_win_rate_y, size=6
                    ).astype(float)

                two_stage = optimize_day_two_stage(
                    **inputs.as_kwargs(),
                    afrr_activation_forecast_pos=alpha_pos_fcst,
                    afrr_activation_forecast_neg=alpha_neg_fcst,
                    afrr_energy_forecast_pos=e_pos_fcst,
                    afrr_energy_forecast_neg=e_neg_fcst,
                    energy_mwh=usable_energy_mwh, power_mw=power_mw, rte=rte,
                    max_cycles=max_cycles,
                    afrr_reserve_duration_hours=afrr_reserve_duration_hours,
                    wear_cost_eur_per_mwh=wear,
                    afrr_wear_premium_eur_per_mwh=afrr_wear_premium_eur_per_mwh,
                    mfrr_penalty_pos_eur_per_mwh=two_stage_mfrr_penalty_pos_eur_per_mwh,
                    mfrr_penalty_neg_eur_per_mwh=two_stage_mfrr_penalty_neg_eur_per_mwh,
                    bid_win_rate_lp=bid_win_lp_eff,
                    afrr_bid_hurdle_eur_per_mw_h=afrr_bid_hurdle_eur_per_mw_h,
                    r_min_per_block_mw=r_min_per_block_mw,
                    afrr_energy_capture_factor=afrr_energy_capture_factor,
                    stage2_award_mask_pos=stage2_award_pos,
                    stage2_award_mask_neg=stage2_award_neg,
                )
                day_out = two_stage.stage2
            else:
                day_out = optimize_day_stacked(**inputs.as_kwargs(), **lp_kwargs)
            if not day_out.success:
                days_skipped += 1
                current += timedelta(days=1)
                day_idx += 1
                continue

            # Two-pass policies (DoD rainflow / physics-from-duty): re-price
            # throughput based on observed first-pass dispatch, then re-solve.
            # Disabled in rolling-horizon mode (architectural: 2-pass needs
            # full-day SoC observation, which RH commits incrementally).
            # In two-stage mode, three paths depending on
            # ``two_stage_picard_max_iter``:
            #   0 (v1.1, default) — Stage-2 only re-pricing under locked
            #       Stage-1 r commitments. Bounded approximation; ignores
            #       that Stage-1 should commit less aFRR if it knew the
            #       true (pass-2) wear.
            #   ≥1 (v1.2 Picard) — iterate full Stage-1+Stage-2 with
            #       refined wear until r commitments stabilise (or max
            #       iter). Closes the Stage-1 ↔ Stage-2 ↔ wear
            #       fixed-point empirically; convergence not guaranteed
            #       theoretically but well-behaved on realistic inputs.
            if (getattr(policy, "uses_two_pass", False)
                    and not use_rolling):
                if use_two_stage and two_stage_picard_max_iter >= 1:
                    # Picard fixed-point: iterate two_stage with refined wear.
                    current_wear = wear  # initial pass-1 wear
                    r_prev = None
                    for _picard_iter in range(two_stage_picard_max_iter):
                        two_stage_iter = optimize_day_two_stage(
                            **inputs.as_kwargs(),
                            afrr_activation_forecast_pos=alpha_pos_fcst,
                            afrr_activation_forecast_neg=alpha_neg_fcst,
                            afrr_energy_forecast_pos=e_pos_fcst,
                            afrr_energy_forecast_neg=e_neg_fcst,
                            energy_mwh=usable_energy_mwh, power_mw=power_mw, rte=rte,
                            max_cycles=max_cycles,
                            afrr_reserve_duration_hours=afrr_reserve_duration_hours,
                            wear_cost_eur_per_mwh=current_wear,
                            afrr_wear_premium_eur_per_mwh=afrr_wear_premium_eur_per_mwh,
                            mfrr_penalty_pos_eur_per_mwh=two_stage_mfrr_penalty_pos_eur_per_mwh,
                            mfrr_penalty_neg_eur_per_mwh=two_stage_mfrr_penalty_neg_eur_per_mwh,
                            bid_win_rate_lp=bid_win_lp_eff,
                            afrr_bid_hurdle_eur_per_mw_h=afrr_bid_hurdle_eur_per_mw_h,
                            r_min_per_block_mw=r_min_per_block_mw,
                            afrr_energy_capture_factor=afrr_energy_capture_factor,
                            stage2_award_mask_pos=stage2_award_pos,
                            stage2_award_mask_neg=stage2_award_neg,
                        )
                        if not two_stage_iter.success:
                            break
                        day_out_iter = two_stage_iter.stage2
                        # Refine wear from this iteration's observed dispatch
                        current_wear = policy.refine_wear_cost(
                            first_pass_soc=day_out_iter.soc,
                            energy_mwh=usable_energy_mwh,
                            soh_current=current_soh,
                            day_of_year=day_of_year,
                            periods_per_day=96,
                            duration_h=duration_h,
                            day_result=day_out_iter,
                        )
                        # Track Stage-1 r commitments for convergence check
                        r_curr = np.concatenate([
                            two_stage_iter.stage1.r_pos[::PERIODS_PER_BLOCK],
                            two_stage_iter.stage1.r_neg[::PERIODS_PER_BLOCK],
                        ])
                        # Update accepted state to latest iteration
                        day_out = day_out_iter
                        two_stage = two_stage_iter
                        if (r_prev is not None
                                and float(np.max(np.abs(r_curr - r_prev)))
                                < two_stage_picard_tol_mw):
                            break
                        r_prev = r_curr
                else:
                    refined_wear = policy.refine_wear_cost(
                        first_pass_soc=day_out.soc,
                        energy_mwh=usable_energy_mwh,
                        soh_current=current_soh,
                        day_of_year=day_of_year,
                        periods_per_day=96,
                        duration_h=duration_h,
                        day_result=day_out,
                    )
                    if use_two_stage:
                        # v1.1 Stage-2-only re-pricing under locked Stage-1 r.
                        r_pos_locked = two_stage.stage1.r_pos[::PERIODS_PER_BLOCK].copy()
                        r_neg_locked = two_stage.stage1.r_neg[::PERIODS_PER_BLOCK].copy()
                        if stage2_award_pos is not None:
                            r_pos_locked = r_pos_locked * np.asarray(stage2_award_pos, dtype=float)
                        if stage2_award_neg is not None:
                            r_neg_locked = r_neg_locked * np.asarray(stage2_award_neg, dtype=float)
                        stage2_bid_win_rate_lp = bid_win_lp_eff
                        if stage2_award_pos is not None or stage2_award_neg is not None:
                            stage2_bid_win_rate_lp = 1.0
                        day_out_refined = optimize_day_stacked(
                            **inputs.as_kwargs(),
                            energy_mwh=usable_energy_mwh, power_mw=power_mw, rte=rte,
                            max_cycles=max_cycles,
                            afrr_reserve_duration_hours=afrr_reserve_duration_hours,
                            wear_cost_eur_per_mwh=refined_wear,
                            afrr_wear_premium_eur_per_mwh=afrr_wear_premium_eur_per_mwh,
                            mfrr_penalty_pos_eur_per_mwh=two_stage_mfrr_penalty_pos_eur_per_mwh,
                            mfrr_penalty_neg_eur_per_mwh=two_stage_mfrr_penalty_neg_eur_per_mwh,
                            max_afrr_participation=1.0,
                            r_pos_locked_per_block=r_pos_locked,
                            r_neg_locked_per_block=r_neg_locked,
                            bid_win_rate_lp=stage2_bid_win_rate_lp,
                            afrr_bid_hurdle_eur_per_mw_h=afrr_bid_hurdle_eur_per_mw_h,
                            r_min_per_block_mw=0.0,
                            afrr_energy_capture_factor=afrr_energy_capture_factor,
                        )
                    else:
                        lp_kwargs["wear_cost_eur_per_mwh"] = refined_wear
                        day_out_refined = optimize_day_stacked(
                            **inputs.as_kwargs(), **lp_kwargs,
                        )
                    if day_out_refined.success:
                        day_out = day_out_refined

            # ── bid-win post-decision realised-outcome haircut ─────
            # IMPORTANT METHODOLOGICAL CAVEAT (P1.1 reviewer comment):
            # `bid_win_rate` is applied AFTER the LP / two-stage dispatch
            # has already chosen its r_pos / r_neg commitments under an
            # implicit assumption of 100 % auction clearing. We then scale
            # the *realised* a_pos / a_neg / r_pos / r_neg / SoC / FEC by
            # `bid_win_rate` (deterministic) or per-block Bernoulli draws
            # (stochastic). This represents a **conservative
            # realised-outcome scenario** — what an operator who bids
            # everything and clears at rate `bid_win_rate` actually sees
            # downstream — NOT a market-aware optimiser that would
            # explicitly optimise for clearing probability.
            #
            # A market-aware version would feed `bid_win_rate` into the
            # LP itself: multiply the cap-price coefficient by it (since
            # only that fraction of bid cap revenue is collected) and
            # scale the activation rate by it (since only that fraction
            # of activation energy is delivered). That would change the
            # LP's optimal r_pos / r_neg — a less-aggressively-bidding
            # operator who knows only half their bids clear may commit
            # less power per block. Implementing that is a future
            # extension — relevant once the calibration goal is "rational
            # auction-aware operator NPV" rather than "post-LP realised
            # outcome under uncertainty."
            #
            # Wholesale (DA + ID) is take-or-leave at clearing price and
            # is not subject to bid-win uncertainty in the same way.
            #
            # PHYSICAL SCALING (paper-critical fix vs revenue-only haircut):
            # When bid_win_rate < 1.0, only that fraction of aFRR
            # commitments actually clears. Two modes:
            #   - "deterministic": uniform haircut bid_win_rate on a_pos,
            #     a_neg, and per-block r_pos / r_neg. Returns expected
            #     outcome (= mean over many stochastic realisations).
            #   - "stochastic": per-block per-direction Bernoulli(bid_win_rate)
            #     clearing. Captures bid-uncertainty variance for risk
            #     analysis (P10/P50/P90 NPV distributions over multiple
            #     seeds).
            # Either way, revenue, FEC, SoC trajectory, and the day_out
            # passed to the physics kernel are recomputed with realised
            # aFRR flows. Conservative interpretation: SoC headroom freed
            # by un-cleared bids is left unused (no intraday re-trade
            # modelled).
            # Gate: post-LP realisation block runs when (a) legacy mode
            # without LP awareness OR (b) LP-aware mode + stochastic
            # Bernoulli (= conservative no-recourse risk-realisation
            # layer; freed power from un-cleared blocks is left unused).
            # Under LP-aware + deterministic, the LP already optimised
            # at expected — no further scaling.
            # Gate: post-LP realisation block fires when per-year effective
            # clearing rate < 1.0 (constant bid_win_rate or yearly_bid_win_rate_scale
            # both reduce bid_win_rate_y below 1).
            run_post_lp_realisation = bid_win_rate_y < 1.0 and (
                not bid_win_in_lp
                or bid_win_mode == "stochastic"
            )
            if run_post_lp_realisation:
                import numpy as _np
                from math import sqrt as _sqrt
                eta = _sqrt(rte)
                dt_hours = 0.25
                steps_per_block = 96 // 6   # 16 intervals per 4-hour aFRR block
                if bid_win_mode == "stochastic":
                    cleared_pos_block = _bid_win_rng.binomial(1, bid_win_rate_y, size=6).astype(float)
                    cleared_neg_block = _bid_win_rng.binomial(1, bid_win_rate_y, size=6).astype(float)
                    bid_win_pos_per_t = _np.repeat(cleared_pos_block, steps_per_block)
                    bid_win_neg_per_t = _np.repeat(cleared_neg_block, steps_per_block)
                else:
                    cleared_pos_block = _np.full(6, float(bid_win_rate_y))
                    cleared_neg_block = _np.full(6, float(bid_win_rate_y))
                    bid_win_pos_per_t = _np.full(96, float(bid_win_rate_y))
                    bid_win_neg_per_t = _np.full(96, float(bid_win_rate_y))
                # Reconstruct day-start SoC from cumsum inversion
                first_flow_in = (
                    day_out.charge_da[0] + day_out.charge_id[0]
                    + day_out.a_neg[0]
                ) * eta * dt_hours
                first_flow_out = (
                    day_out.discharge_da[0] + day_out.discharge_id[0]
                    + day_out.a_pos[0]
                ) / eta * dt_hours
                soc_init_today = float(day_out.soc[0]) - (first_flow_in - first_flow_out)
                # Realised aFRR flows (per-interval mask broadcast from per-block)
                a_pos_real = day_out.a_pos * bid_win_pos_per_t
                a_neg_real = day_out.a_neg * bid_win_neg_per_t
                r_pos_real = day_out.r_pos * bid_win_pos_per_t
                r_neg_real = day_out.r_neg * bid_win_neg_per_t
                # Recompute SoC trajectory with realised aFRR
                soc_changes_real = (
                    (day_out.charge_da + day_out.charge_id + a_neg_real) * eta
                    - (day_out.discharge_da + day_out.discharge_id + a_pos_real) / eta
                ) * dt_hours
                soc_real = soc_init_today + _np.cumsum(soc_changes_real)
                # Recompute FEC (total discharge / energy)
                fec_real = float(_np.sum(
                    (day_out.discharge_da + day_out.discharge_id + a_pos_real)
                    * dt_hours
                )) / max(usable_energy_mwh, 1e-9)
                # Recompute aFRR revenue with realised clearings.
                # Energy revenue: a_pos × price × dt, summed over intervals.
                # Cap revenue: per-block r × cap_price × block_hours, summed.
                block_hours = steps_per_block * dt_hours
                cap_pos_real = float(_np.sum(
                    cleared_pos_block * inputs.afrr_cap_pos_price *
                    day_out.r_pos[::steps_per_block] * block_hours
                )) if bid_win_mode == "stochastic" else (
                    day_out.revenue_afrr_cap_pos * bid_win_rate_y
                )
                cap_neg_real = float(_np.sum(
                    cleared_neg_block * inputs.afrr_cap_neg_price *
                    day_out.r_neg[::steps_per_block] * block_hours
                )) if bid_win_mode == "stochastic" else (
                    day_out.revenue_afrr_cap_neg * bid_win_rate_y
                )
                energy_pos_real = float(_np.sum(
                    a_pos_real * dt_hours * inputs.afrr_energy_pos_price
                ))
                energy_neg_real = float(_np.sum(
                    a_neg_real * dt_hours * inputs.afrr_energy_neg_price
                ))
                from dataclasses import replace as _dc_replace
                # Recompute revenue_total with realised aFRR revenue so
                # downstream diagnostic records (P2.7 reviewer comment) reflect
                # bid-win-adjusted owner economics rather than LP-committed.
                revenue_total_real = (
                    day_out.revenue_da + day_out.revenue_id
                    + cap_pos_real + cap_neg_real
                    + energy_pos_real + energy_neg_real
                )
                day_out_realised = _dc_replace(
                    day_out,
                    a_pos=a_pos_real, a_neg=a_neg_real,
                    r_pos=r_pos_real, r_neg=r_neg_real,
                    soc=soc_real,
                    revenue_afrr_cap_pos=cap_pos_real,
                    revenue_afrr_cap_neg=cap_neg_real,
                    revenue_afrr_energy_pos=energy_pos_real,
                    revenue_afrr_energy_neg=energy_neg_real,
                    revenue_total=revenue_total_real,
                    full_equivalent_cycles=fec_real,
                )
            else:
                day_out_realised = day_out

            cap_pos_rev = day_out_realised.revenue_afrr_cap_pos
            cap_neg_rev = day_out_realised.revenue_afrr_cap_neg
            energy_pos_rev = day_out_realised.revenue_afrr_energy_pos
            energy_neg_rev = day_out_realised.revenue_afrr_energy_neg
            day_total_adjusted = (
                day_out_realised.revenue_da + day_out_realised.revenue_id
                + cap_pos_rev + cap_neg_rev + energy_pos_rev + energy_neg_rev
            )

            year_revenue += day_total_adjusted
            year_rev_da += day_out_realised.revenue_da
            year_rev_id += day_out_realised.revenue_id
            year_rev_afrr_cap += (cap_pos_rev + cap_neg_rev)
            year_rev_afrr_energy += (energy_pos_rev + energy_neg_rev)
            year_fec += day_out_realised.full_equivalent_cycles
            days_solved += 1

            # Collect diagnostic-day record if this is the target year.
            # Use the REALISED day_out (post bid-win scaling) so diagnostic
            # signals reflect actual cycling, not LP-committed cycling.
            if collect_diagnostics_year is not None and y_idx == collect_diagnostics_year:
                from lib.analysis.aging_aware_diagnostics import (
                    day_diagnostic_from_stacked,
                )
                diagnostic_days.append(
                    day_diagnostic_from_stacked(
                        day_result=day_out_realised, inputs=inputs,
                        target_date=current, power_mw=power_mw,
                        energy_mwh=usable_energy_mwh,
                    )
                )

            # Degrade SoH for today's REALISED throughput
            if use_physics_degradation and physics_preset is not None:
                delta = physics_degradation_per_day(
                    day_result=day_out_realised,
                    energy_mwh=usable_energy_mwh,
                    soh_current=current_soh,
                    preset=physics_preset,
                    temperature_c=physics_temperature_c,
                    kernel_scale=physics_kernel_scale,
                )
            else:
                delta = degradation_per_day(
                    intensity=day_out_realised.full_equivalent_cycles,
                    soh=current_soh,
                    fade_per_fec_at_soh_1=fade_per_fec_at_soh_1,
                    calendar_fade_per_day=calendar_fade_per_day,
                )
            current_soh -= delta
            if current_soh < warranty_floor:
                current_soh = warranty_floor
                if years_to_floor is None:
                    years_to_floor = y_idx
                break

            current += timedelta(days=1)
            day_idx += 1

        annual_revenue[y_idx] = year_revenue
        annual_revenue_da[y_idx] = year_rev_da
        annual_revenue_id[y_idx] = year_rev_id
        annual_revenue_afrr_cap[y_idx] = year_rev_afrr_cap
        annual_revenue_afrr_energy[y_idx] = year_rev_afrr_energy
        annual_fec[y_idx] = year_fec
        end_of_year_soh[y_idx] = current_soh

    # FCR phantom layer (paper-grade calibration; ADR-002c §"FCR phantom").
    # Adds fixed annual FCR revenue (capacity-only, very low energy α) for
    # years until warranty-floor is hit, scaled by SoH (capacity-derated).
    # Optional `yearly_fcr_revenue_scale` array applies fleet-saturation
    # decay (FCR pool already at overbuild ratio 1.4 in 2024, so trajectory
    # decays faster than aFRR). The phantom does NOT modify the dispatch
    # LP — endogenous FCR integration is ADR-006 (deferred). The fixed
    # SoC reservation cost (`fcr_soc_reservation_mw`) is NOT applied here
    # either; deferred to ADR-006 because it requires modifying the LP's
    # usable_energy_mwh per-day, which would re-introduce the calibration
    # journey. For now treat phantom FCR as a pure additive revenue
    # contribution; the SoC competition cost is a known un-modelled
    # residual.
    if fcr_revenue_keur_per_mw_per_year > 0:
        fcr_per_year = np.zeros(n_years)
        for y in range(n_years):
            if y >= (years_to_floor or n_years):
                continue    # asset retired; no FCR revenue
            soh_y = float(end_of_year_soh[y]) if end_of_year_soh[y] > 0 else 1.0
            scale_y = (
                float(yearly_fcr_revenue_scale[y])
                if yearly_fcr_revenue_scale is not None
                else 1.0
            )
            fcr_per_year[y] = fcr_revenue_keur_per_mw_per_year * 1000.0 * soh_y * scale_y
        annual_revenue = annual_revenue + fcr_per_year

    # Optional per-year revenue scaling (e.g. Note 1 market-saturation trend).
    # Applied post-dispatch: the trader doesn't see future price evolution;
    # the scaling captures NPV-level consequences only.
    if yearly_revenue_scale is not None:
        scale = np.asarray(yearly_revenue_scale, dtype=float)
        if scale.shape != (n_years,):
            raise ValueError(
                f"yearly_revenue_scale must have shape ({n_years},), "
                f"got {scale.shape}"
            )
        annual_revenue = annual_revenue * scale

    # Discount annual revenue to present (year 1 discounted by 1/(1+r)^1, etc.)
    years = np.arange(1, n_years + 1)
    discount_factors = (1.0 + discount_rate) ** -years
    lifetime_npv = float((annual_revenue * discount_factors).sum())

    return LifecycleResult(
        policy_name=policy.name,
        n_years=n_years,
        initial_soh=initial_soh,
        warranty_floor=warranty_floor,
        discount_rate=discount_rate,
        annual_revenue_eur=annual_revenue,
        annual_revenue_da_eur=annual_revenue_da,
        annual_revenue_id_eur=annual_revenue_id,
        annual_revenue_afrr_cap_eur=annual_revenue_afrr_cap,
        annual_revenue_afrr_energy_eur=annual_revenue_afrr_energy,
        annual_fec=annual_fec,
        end_of_year_soh=end_of_year_soh,
        days_solved=days_solved,
        days_skipped=days_skipped,
        lifetime_npv_eur=lifetime_npv,
        years_to_floor=years_to_floor,
        diagnostic_days=diagnostic_days if diagnostic_days else None,
    )


def monte_carlo_lifecycle(
    policy: ShadowCostPolicy,
    n_realizations: int = 30,
    base_seed: int = 42,
    **simulate_kwargs,
) -> dict:
    """Monte-Carlo over stochastic bid-clearing realisations.

    Runs ``simulate_lifecycle(policy, bid_win_mode='stochastic',
    bid_win_rng_seed=base_seed+i)`` for ``i in range(n_realizations)``,
    collects lifetime NPVs, and returns a distribution summary.

    Captures bid-uncertainty variance: per-block per-direction Bernoulli
    clearing with probability ``bid_win_rate`` (passed via
    ``simulate_kwargs``). SoH evolution per realisation reflects realised
    cycling, so EOL year and life extension are seed-dependent.

    Returns dict with keys ``samples`` (np.ndarray of NPVs), ``mean``,
    ``std``, ``p10``, ``p25``, ``p50``, ``p75``, ``p90``, ``cvar90``
    (mean of bottom 10 % — conditional value at risk).
    """
    simulate_kwargs.setdefault("bid_win_mode", "stochastic")
    simulate_kwargs.pop("bid_win_rng_seed", None)
    samples = np.zeros(n_realizations)
    eol_samples = np.zeros(n_realizations, dtype=int)
    fec_samples = np.zeros(n_realizations)
    for i in range(n_realizations):
        result = simulate_lifecycle(
            policy=policy, bid_win_rng_seed=base_seed + i,
            **simulate_kwargs,
        )
        samples[i] = result.lifetime_npv_eur
        eol_samples[i] = result.years_to_floor or simulate_kwargs.get("n_years", 10)
        fec_samples[i] = result.annual_fec.sum()
    n_tail = max(1, n_realizations // 10)
    sorted_samples = np.sort(samples)
    return {
        "samples": samples,
        "eol_samples": eol_samples,
        "fec_samples": fec_samples,
        "mean": float(np.mean(samples)),
        "std": float(np.std(samples)),
        "p10": float(np.percentile(samples, 10)),
        "p25": float(np.percentile(samples, 25)),
        "p50": float(np.percentile(samples, 50)),
        "p75": float(np.percentile(samples, 75)),
        "p90": float(np.percentile(samples, 90)),
        "cvar90": float(np.mean(sorted_samples[:n_tail])),  # avg of bottom 10%
        "n_realizations": n_realizations,
        "policy_name": policy.name,
    }


def _simulate_one_policy(args):
    """Module-level worker for ProcessPoolExecutor (must be picklable).

    Workers re-prefetch frames from the on-disk cache — cheaper than
    pickling and forwarding a large dict over the IPC boundary, and
    avoids macOS spawn-vs-fork issues in Python 3.14+.
    """
    name, policy, simulate_kwargs = args
    template_years = simulate_kwargs.get("template_years", (2023, 2025))
    shared_frames = {ty: _prefetch_year_frames(ty) for ty in template_years}
    result = simulate_lifecycle(
        policy=policy,
        prefetched_frames=shared_frames,
        **simulate_kwargs,
    )
    return name, result


def compare_policies(
    policies: dict[str, ShadowCostPolicy],
    n_workers: int = 1,
    **simulate_kwargs,
) -> dict[str, LifecycleResult]:
    """
    Run :func:`simulate_lifecycle` for each policy sharing prefetched
    market frames. Returns results keyed by policy name.

    All policies see identical market inputs — the only differentiation
    is their wear cost vector. This is the critical property for
    Note 4's comparison claims.

    Args:
        policies: Mapping of policy name → :class:`ShadowCostPolicy`.
        n_workers: Number of worker processes for parallel execution.
            ``1`` (default) keeps sequential single-process behaviour.
            ``> 1`` spawns a :class:`ProcessPoolExecutor` and runs each
            policy in parallel. Each worker re-prefetches market frames
            from the on-disk cache (~free since data is already cached).
            Bound by the slowest single-policy run; for full 7-policy
            ablation on M-series Mac, n_workers=4 ≈ 3-4× wall-clock
            speed-up.
    """
    if n_workers <= 1:
        # Prefetch once, reuse across all policies (sequential path).
        template_years = simulate_kwargs.get("template_years", (2023, 2025))
        shared_frames = {ty: _prefetch_year_frames(ty) for ty in template_years}
        results: dict[str, LifecycleResult] = {}
        for name, policy in policies.items():
            logger.info(f"compare_policies: simulating {name}…")
            results[name] = simulate_lifecycle(
                policy=policy,
                prefetched_frames=shared_frames,
                **simulate_kwargs,
            )
        return results

    from concurrent.futures import ProcessPoolExecutor, as_completed
    logger.info(f"compare_policies: parallel run, n_workers={n_workers}, "
                f"{len(policies)} policies")
    args_list = [(name, policy, simulate_kwargs) for name, policy in policies.items()]
    results: dict[str, LifecycleResult] = {}
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for name, result in ex.map(_simulate_one_policy, args_list):
            results[name] = result
            logger.info(f"compare_policies: completed {name}")
    return results
