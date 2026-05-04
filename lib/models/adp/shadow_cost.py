"""
Shadow-cost policies for degradation-aware dispatch (Note 4 A3).

Each policy produces the per-interval ``wear_cost_eur_per_mwh`` vector that
feeds ``lib.models.dispatch.stacked.optimize_day_stacked``. The vector
represents the economic cost of cycling at each 15-min slot: the LP
subtracts it from revenue, suppressing marginal trades that don't clear
their wear cost.

Four policies, increasing in sophistication and meant for side-by-side
comparison in Note 4:

1. :class:`NaivePolicy` — ``wear_cost ≡ 0``. The trader prices nothing;
   every spread above variable cost clears.

2. :class:`DepreciationProxyPolicy` — flat scalar
   ``CAPEX / lifetime_throughput``. This is the "depreciation cost as a
   proxy for revenue lost to aging" that Kumtepeli, Hesse, Morstyn,
   Nosratabadi, Aunedi, Howey (2024, arXiv:2403.10617) argue against.
   Symmetric across charge/discharge and constant in time.

3. :class:`AgingAwareDepreciationPolicy` — closed-form state-dependent
   approximation (A3.1 stepping-stone):
   ``λ(SoH) = base × (1 − SoH) / (SoH − warranty_floor)``. Scales the
   depreciation proxy by the *scarcity* of remaining SoH headroom: when
   the battery has plenty of life left, shadow cost is low; as it
   approaches the warranty floor, the shadow cost diverges. Flat in time
   within a given day, but updates day-by-day as SoH drifts down.

4. :class:`ADPPolicy` — wraps a solved ADP DP (A3.3) and returns the
   per-state shadow cost ``∂V/∂SoH`` at the current ``(SoH, regime)``.
   The regime is looked up from a :class:`RegimeClassification` given
   the calendar date, or can be passed as an override for sensitivity
   studies. See the solver module for structural limitations — the
   A3.3 implementation is a principled DP reference, not yet the full
   Holtorf-Shin SoC-aware formulation.

All policies expose the same ``wear_cost(soh_current, day_of_year,
periods_per_day, duration_h) -> np.ndarray`` interface so the Note 4
dispatcher can swap them at LP-call time.
"""
from __future__ import annotations

import datetime as _dt
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


class ShadowCostPolicy(ABC):
    """Interface: maps (current asset state, calendar) → per-interval EUR/MWh.

    The returned array is what an aging-aware dispatcher subtracts from
    energy revenue in its LP objective. Length ``periods_per_day`` (96 by
    default). Values are non-negative by convention; negative shadow cost
    makes no physical sense (cycling can't be *rewarded* by wear).

    Additionally, policies may override LP-level parameters (SoC envelope,
    cycle cap) via :meth:`lp_overrides`. Most policies return an empty
    dict — the exception is a constraint-based policy such as
    :class:`SoCWindowPolicy`.

    Two-pass policies (ROADMAP channel (b) rainflow proxy) set
    :attr:`uses_two_pass` ``= True``. The runner then solves the LP once
    with the first-pass wear, reads the resulting SoC trajectory, asks
    the policy to :meth:`refine_wear_cost`, and re-solves. This captures
    "deep cycles cost more" without full cycle-counting MILP.
    """

    name: str
    uses_two_pass: bool = False

    @abstractmethod
    def wear_cost(
        self,
        soh_current: float,
        day_of_year: int = 1,
        periods_per_day: int = 96,
        duration_h: float = 2.0,
    ) -> np.ndarray:
        """Per-interval wear cost in EUR/MWh throughput for this day.

        Args:
            soh_current: State of health in [warranty_floor, 1.0]. Outside
                this range a policy may clamp or extrapolate; document per
                subclass.
            day_of_year: 1..365/366. Some policies (ADP) use this as
                season index.
            periods_per_day: Time grid; 96 for 15-min dispatch.
            duration_h: BESS energy duration; informs throughput-per-MWh
                accounting in some policies.
        """

    def lp_overrides(
        self, soh_current: float, day_of_year: int = 1,
    ) -> dict:
        """Optional per-day overrides of LP parameters.

        Return a dict with any of: ``soc_min_frac``, ``soc_max_frac``,
        ``max_cycles``, ``max_afrr_participation``. Defaults to empty
        (no overrides). Constraint-based policies (SoC window, hard
        cycle cap) use this channel to impose structure the LP wouldn't
        otherwise see.
        """
        return {}

    def refine_wear_cost(
        self,
        first_pass_soc: np.ndarray,
        energy_mwh: float,
        soh_current: float,
        day_of_year: int = 1,
        periods_per_day: int = 96,
        duration_h: float = 2.0,
        day_result=None,
    ) -> np.ndarray:
        """Second-pass wear vector given the first-pass SoC trajectory.

        Only called by the runner when :attr:`uses_two_pass` is ``True``.
        Default implementation returns the first-pass wear (no-op) — safe
        to inherit for single-pass policies.

        ``day_result`` is the first-pass :class:`StackedDispatchResult`;
        unused by default, but the ``ProgressivePolicy`` physics-from-
        duty layer reads it.
        """
        return self.wear_cost(
            soh_current=soh_current, day_of_year=day_of_year,
            periods_per_day=periods_per_day, duration_h=duration_h,
        )


@dataclass(frozen=True)
class NaivePolicy(ShadowCostPolicy):
    """Revenue-max dispatch. No cycle pricing — every positive spread clears."""

    name: str = "naive"

    def wear_cost(
        self,
        soh_current: float,
        day_of_year: int = 1,
        periods_per_day: int = 96,
        duration_h: float = 2.0,
    ) -> np.ndarray:
        return np.zeros(periods_per_day)


@dataclass(frozen=True)
class DepreciationProxyPolicy(ShadowCostPolicy):
    """Flat CAPEX/lifetime-throughput scalar — the Kumtepeli "poor proxy".

    The proxy is constant across calendar time, SoH, and SoC — it prices
    one MWh of throughput at ``capex_eur_per_mwh / lifetime_throughput_ratio``
    regardless of the asset's economic state. Kumtepeli et al. (2024)
    quantify the revenue gap between this and the correct
    forgone-future-revenue framing.

    Args:
        capex_eur_per_mwh: Installed CAPEX per MWh of energy capacity
            (typical 2026: €90–115 for 2h BESS).
        lifetime_throughput_ratio: Total full-equivalent-cycles over
            design life (typical 5,000–7,000 for LFP).
    """

    name: str = "depreciation_proxy"
    capex_eur_per_mwh: float = 100_000.0
    lifetime_throughput_ratio: float = 6_000.0

    @property
    def scalar_eur_per_mwh(self) -> float:
        return self.capex_eur_per_mwh / max(self.lifetime_throughput_ratio, 1.0)

    def wear_cost(
        self,
        soh_current: float,
        day_of_year: int = 1,
        periods_per_day: int = 96,
        duration_h: float = 2.0,
    ) -> np.ndarray:
        return np.full(periods_per_day, self.scalar_eur_per_mwh)


@dataclass(frozen=True)
class AgingAwareDepreciationPolicy(ShadowCostPolicy):
    """Closed-form state-dependent approximation (A3.1 stepping-stone).

    Prices each MWh of throughput as::

        λ(SoH) = base_eur_per_mwh * scarcity(SoH)
        scarcity(SoH) = (1 - SoH) / max(SoH - warranty_floor, epsilon)

    Interpretation:
      * Fresh battery (SoH ≈ 1.0): scarcity small → cheap to cycle.
      * Near warranty floor (SoH ≈ warranty_floor): scarcity diverges →
        cycling is very expensive; operator should avoid it unless
        revenue far exceeds typical spreads.
      * Below warranty floor: policy clamps scarcity to the ``epsilon``-
        bounded value (cycling is priced at the divergent limit, not
        negative or undefined).

    This is **not** the state-dependent opportunity cost from full ADP; it's
    a one-dimensional approximation that captures "cycles cost more as the
    asset ages" without solving the DP. Same plumbing as the ADP policy —
    drop-in replaceable — so the LP-side integration is identical.

    Args:
        base_eur_per_mwh: Anchor for the depreciation proxy at SoH = 0.90
            (mid-life). A sensible default is the flat Kumtepeli proxy.
        warranty_floor: SoH level at which the warranty bites (typical 0.80).
        epsilon: Numerical safety to avoid division-by-zero at the floor.

    Reference: plan ``~/.claude/plans/eager-enchanting-glade.md`` §A3.
    """

    name: str = "aging_aware_depreciation"
    base_eur_per_mwh: float = 16.67  # = 100000 / 6000 ≈ depreciation proxy
    warranty_floor: float = 0.80
    epsilon: float = 0.005

    def wear_cost(
        self,
        soh_current: float,
        day_of_year: int = 1,
        periods_per_day: int = 96,
        duration_h: float = 2.0,
    ) -> np.ndarray:
        headroom = max(soh_current - self.warranty_floor, self.epsilon)
        consumed = max(1.0 - soh_current, 0.0)
        # Anchor the policy to base at SoH = 0.90 (mid-life):
        #   headroom_ref = 0.90 - 0.80 = 0.10
        #   consumed_ref = 1.0 - 0.90 = 0.10
        #   ratio_ref = 1.0
        # So at SoH = 0.90 the wear cost exactly equals ``base_eur_per_mwh``.
        # Below 0.90: ratio > 1 (cost rises); above 0.90: ratio < 1 (cost falls).
        ratio = consumed / headroom
        return np.full(periods_per_day, self.base_eur_per_mwh * ratio)


@dataclass(frozen=True)
class SoCWindowPolicy(ShadowCostPolicy):
    """Constraint-based aging-aware dispatch (ROADMAP channel (d)).

    Rather than pricing cycles via a shadow cost in the LP objective,
    this policy *constrains* the LP to a tighter SoC envelope — e.g.
    20–80 % of usable energy. The LP then still maximises revenue
    subject to that envelope, but cannot cycle past the high-stress
    extremes.

    Rationale (per ROADMAP trader-aging-aware §Key questions channel
    (d)): "SoC-window constraints that cap high-stress operation
    directly". This is how real commercial operators typically
    implement aging-awareness — warranty terms often impose hard SoC
    limits rather than internal shadow costs.

    Shadow cost returned is zero — the policy does not subtract from
    revenue. The mechanism is entirely the narrower envelope.

    Args:
        soc_min_frac: Lower SoC bound (fraction of usable energy).
        soc_max_frac: Upper SoC bound.
    """

    name: str = "soc_window"
    soc_min_frac: float = 0.20
    soc_max_frac: float = 0.80

    def wear_cost(
        self,
        soh_current: float,
        day_of_year: int = 1,
        periods_per_day: int = 96,
        duration_h: float = 2.0,
    ) -> np.ndarray:
        return np.zeros(periods_per_day)

    def lp_overrides(
        self, soh_current: float, day_of_year: int = 1,
    ) -> dict:
        return {
            "soc_min_frac": self.soc_min_frac,
            "soc_max_frac": self.soc_max_frac,
        }


@dataclass(frozen=True)
class DoDAwareRainflowProxyPolicy(ShadowCostPolicy):
    """Two-pass DoD-aware shadow cost (ROADMAP channel (b), proxy).

    A rainflow-grade policy would identify full charge-discharge cycles
    in the resulting SoC trace and price each by its depth. That requires
    either cycle-counting MILP or LP+rainflow fixed-point iteration —
    both expensive and beyond Note 4's scope.

    The proxy here captures the same mechanism ("deep cycles damage more
    than shallow cycles") with a two-pass LP:

    1. **Pass 1.** Solve the daily LP with the baseline flat wear
       (same level as :class:`DepreciationProxyPolicy`). Observe the
       per-interval SoC trajectory the LP would use in a plain
       proxy-priced world.
    2. **Pass 2.** Price each interval's throughput by::

           wear[t] = base × (1 + dod_multiplier × dev[t] ** dod_exponent)

       where ``dev[t] = |SoC(t)/E_usable − 0.5|`` ∈ [0, 0.5]. A throughput
       period the first-pass LP chose to spend at extreme SoC (deep
       discharge bottom, full-to-top charging) is now expensive; middle-
       SoC throughput stays at baseline.

    The second solve then shifts cycling away from extremes — the
    continuous-penalty analogue of :class:`SoCWindowPolicy`'s hard cutoff.

    Not claimed to be a convergent fixed point; the second-pass SoC
    differs from the first-pass one. Two passes is a calibrated empirical
    compromise (one more solve per day = ~2× runtime; extra passes show
    <0.5 % lifetime-NPV delta in our tests).

    Args:
        base_eur_per_mwh: Mid-SoC baseline (same as the depreciation
            proxy — typical ``CAPEX / lifetime_throughput ≈ 16.67``).
        dod_multiplier: Controls steepness of the SoC-deviation penalty.
            With default ``dod_exponent = 1``, a value of ``8`` makes
            throughput at ``|SoC − 0.5| = 0.5`` five times more
            expensive than at ``SoC = 0.5``.
        dod_exponent: Nonlinearity in the deviation term (1 = linear,
            2 = quadratic). Higher exponents concentrate the penalty at
            the extremes.
    """

    name: str = "dod_aware_rainflow"
    base_eur_per_mwh: float = 16.67
    dod_multiplier: float = 8.0
    dod_exponent: float = 1.0
    uses_two_pass: bool = True

    def wear_cost(
        self,
        soh_current: float,
        day_of_year: int = 1,
        periods_per_day: int = 96,
        duration_h: float = 2.0,
    ) -> np.ndarray:
        return np.full(periods_per_day, self.base_eur_per_mwh)

    def refine_wear_cost(
        self,
        first_pass_soc: np.ndarray,
        energy_mwh: float,
        soh_current: float,
        day_of_year: int = 1,
        periods_per_day: int = 96,
        duration_h: float = 2.0,
        day_result=None,
    ) -> np.ndarray:
        if energy_mwh <= 0 or len(first_pass_soc) != periods_per_day:
            return self.wear_cost(
                soh_current=soh_current, day_of_year=day_of_year,
                periods_per_day=periods_per_day, duration_h=duration_h,
            )
        soc_frac = np.clip(first_pass_soc / energy_mwh, 0.0, 1.0)
        dev = np.abs(soc_frac - 0.5)
        multiplier = 1.0 + self.dod_multiplier * (dev ** self.dod_exponent)
        return self.base_eur_per_mwh * multiplier


class ADPPolicy(ShadowCostPolicy):
    """Shadow cost from a solved backward-induction DP (A3.4 online lookup).

    Wraps a solved :class:`lib.models.adp.solver.ADPSolver` and exposes its
    per-state shadow cost through the ``ShadowCostPolicy`` interface. At
    each call, finds the grid bucket for the current ``SoH`` and the
    regime for the given calendar day, then emits a flat
    ``wear_cost_eur_per_mwh`` vector for the day.

    Construction
    ------------
    ::

        from lib.models.price_regime import fit_regimes
        from lib.models.adp.solver import ADPSolver, default_grids, empirical_daily_revenue_curve

        rc = fit_regimes(da_price_series)
        rev_curve = empirical_daily_revenue_curve(
            daily_revenue=..., regime_labels=rc.regime_labels,
            n_regimes=rc.n_regimes, action_grid=default_grids().action_grid,
        )
        solver = ADPSolver(rc, rev_curve, default_grids())
        adp_result = solver.solve()
        policy = ADPPolicy(solver, adp_result, rc, year_start=date(2024, 1, 1))

    Args:
        solver: The fitted :class:`ADPSolver` (retains the state/action grids).
        result: The :class:`ADPResult` from ``solver.solve()``.
        regime_classification: The regime classifier fit on the historical
            price series. Used to look up the regime for a given
            ``day_of_year``.
        year_start: Calendar anchor for the ``day_of_year`` index — date 0
            corresponds to Jan 1 of this year. Defaults to the first date
            in ``regime_classification.regime_labels``.
        regime_override: If provided, use this fixed regime index instead
            of the calendar lookup. Useful for sensitivity / per-regime
            policy comparisons.
    """

    name: str = "adp"

    def __init__(
        self,
        solver,
        result,
        regime_classification,
        year_start: Optional[_dt.date] = None,
        regime_override: Optional[int] = None,
    ) -> None:
        self._solver = solver
        self._result = result
        self._regime = regime_classification
        self._regime_override = regime_override
        if year_start is None:
            # Infer from the regime labels index (first date)
            first_label_date = next(iter(regime_classification.regime_labels.index))
            if isinstance(first_label_date, _dt.date):
                year_start = first_label_date
            else:  # pandas Timestamp
                year_start = _dt.date(first_label_date.year, 1, 1)
        self._year_start = year_start

    def _soh_idx(self, soh_current: float) -> int:
        grid = self._solver.grids.soh_grid
        # Clamp to grid range, then snap to nearest-at-or-below.
        if soh_current <= grid[0]:
            return 0
        if soh_current >= grid[-1]:
            return len(grid) - 1
        # np.searchsorted side='right' → idx where grid[idx-1] < soh <= grid[idx]
        idx = int(np.searchsorted(grid, soh_current, side="right")) - 1
        return max(0, idx)

    def _regime_for_day(self, day_of_year: int) -> int:
        if self._regime_override is not None:
            return int(self._regime_override)
        target = self._year_start + _dt.timedelta(days=int(day_of_year) - 1)
        labels = self._regime.regime_labels
        # Labels index is date-like; try direct lookup, fallback to nearest
        if target in labels.index:
            return int(labels.loc[target])
        # Fallback: use stationary-distribution mode (most likely regime)
        return int(np.argmax(self._regime.stationary))

    def wear_cost(
        self,
        soh_current: float,
        day_of_year: int = 1,
        periods_per_day: int = 96,
        duration_h: float = 2.0,
    ) -> np.ndarray:
        soh_idx = self._soh_idx(soh_current)
        reg_idx = self._regime_for_day(day_of_year)
        scalar = float(self._result.shadow_cost[soh_idx, reg_idx])
        return np.full(periods_per_day, scalar)


class ADPPolicyIntraday(ShadowCostPolicy):
    """Hour-varying shadow cost from intraday ADP (Holtorf-Shin style).

    Wraps an :class:`lib.models.adp.solver_intraday.IntradayADPResult`.
    At online-lookup time, snaps (SoH, regime) to grid, returns 24-hour
    shadow cost vector expanded to 96 × 15-min intervals.

    The hour-varying signal is what the simplified-state DP cannot
    produce: same shadow cost at peak vs trough, same at fresh vs aged.
    This policy is the principled Holtorf-Shin reference and the
    empirical answer to "do we need full intraday DP for Note 4?".

    Physics-informed aging layer (Note 3 kernel)
    ---------------------------------------------
    When :attr:`physics_wear_eur_per_mwh` is supplied, the policy adds
    the SoH-dependent physics wear (from the Note 3 Wang+Naumann
    calibrated kernel) on top of the DP's arbitrage-derived shadow
    cost. This closes methodology simplification #2 for the intraday DP
    cleanly — the DP stays a pure arbitrage planner; the aging cost
    enters at online-lookup time as a per-MWh-throughput additive
    term::

        shadow_cost[t] = |∂V/∂SoC|[t]  +  physics_wear(SoH)

    First component (hour-varying): opportunity cost of depleting SoC at
    this hour. Second (SoH-varying): physics cost of one MWh of
    throughput at current SoH. LP sees the combined signal and
    suppresses cycling when either dominates.

    The earlier "aging-inside-Bellman" experiment flattened the
    ``|∂V/∂SoC|`` signal (DP policy pre-emptively suppressed cycling,
    shrinking the gradient), so the LP saw a weaker wear signal and
    cycled MORE — a counterproductive coupling. Additive integration
    preserves both signals independently.
    """

    name: str = "adp_intraday"

    def __init__(
        self,
        intraday_result,
        grids,
        regime_classification,
        year_start: Optional[_dt.date] = None,
        regime_override: Optional[int] = None,
        physics_wear_eur_per_mwh: Optional[np.ndarray] = None,
    ) -> None:
        self._result = intraday_result
        self._grids = grids
        self._regime = regime_classification
        self._regime_override = regime_override
        self._physics_wear = (
            np.asarray(physics_wear_eur_per_mwh, dtype=float)
            if physics_wear_eur_per_mwh is not None else None
        )
        if self._physics_wear is not None:
            if self._physics_wear.shape != grids.soh_grid.shape:
                raise ValueError(
                    f"physics_wear_eur_per_mwh shape {self._physics_wear.shape} "
                    f"must match grids.soh_grid shape {grids.soh_grid.shape}"
                )
        if year_start is None:
            first_label_date = next(iter(regime_classification.regime_labels.index))
            if isinstance(first_label_date, _dt.date):
                year_start = first_label_date
            else:
                year_start = _dt.date(first_label_date.year, 1, 1)
        self._year_start = year_start

    def _soh_idx(self, soh_current: float) -> int:
        grid = self._grids.soh_grid
        if soh_current <= grid[0]:
            return 0
        if soh_current >= grid[-1]:
            return len(grid) - 1
        idx = int(np.searchsorted(grid, soh_current, side="right")) - 1
        return max(0, idx)

    def _regime_for_day(self, day_of_year: int) -> int:
        if self._regime_override is not None:
            return int(self._regime_override)
        target = self._year_start + _dt.timedelta(days=int(day_of_year) - 1)
        labels = self._regime.regime_labels
        if target in labels.index:
            return int(labels.loc[target])
        return int(np.argmax(self._regime.stationary))

    def wear_cost(
        self,
        soh_current: float,
        day_of_year: int = 1,
        periods_per_day: int = 96,
        duration_h: float = 2.0,
    ) -> np.ndarray:
        soh_idx = self._soh_idx(soh_current)
        reg_idx = self._regime_for_day(day_of_year)
        hourly = self._result.shadow_cost[soh_idx, reg_idx]  # (24,) arbitrage only
        # Expand to periods_per_day by repeating each hour's value
        intervals_per_hour = periods_per_day // 24
        remainder = periods_per_day - 24 * intervals_per_hour
        vector = np.repeat(hourly, intervals_per_hour)
        if remainder > 0:
            vector = np.concatenate([vector, np.full(remainder, hourly[-1])])
        # Add Note 3 physics-kernel-derived aging cost (SoH-varying, constant
        # across hours within a day). See class docstring for rationale.
        if self._physics_wear is not None:
            vector = vector + float(self._physics_wear[soh_idx])
        return vector


class ProgressivePolicy(ShadowCostPolicy):
    """Progressively stacked shadow-cost policy.

    Each flag enables **one additional layer** on top of the previous
    ones; layers either add to the per-interval wear vector, modify
    the LP-level envelope (SoC window), or refine the wear in a second
    LP pass (DoD rainflow). This is the canonical Note 4 framing: each
    level includes **everything from the previous levels**, so empirical
    comparisons show incremental value per layer rather than an
    apples-to-oranges race between alternative formulas.

    Layer order (as used in Note 4)::

        L1  Naive                    — no layers
        L2  + SoC window              — hard constraint 20–80 %
        L3  + Flat wear               — CAPEX ÷ lifetime throughput
        L4  + Scarcity scaling        — multiply flat by (1-SoH)/(SoH-floor)
        L5  + DoD rainflow            — 2-pass: weight extreme-SoC throughput
        L6  + Intraday ADP shadow     — add hour-varying |∂V/∂SoC|[h]
        L7  + Physics wear (Note 3)   — add SoH-dependent aging per MWh

    All layers coexist. No "alternative" policies in the note — every
    policy is the one before plus one more layer.

    Args:
        name: Human-readable identifier.
        use_soc_window: Toggle the LP SoC envelope constraint.
        soc_min_frac, soc_max_frac: Envelope bounds.
        flat_base_eur_per_mwh: Flat throughput cost (0 disables).
        use_scarcity: Scale flat cost by ``(1−SoH)/max(SoH−floor, ε)``.
        warranty_floor, epsilon: Scarcity denominator parameters.
        dod_multiplier, dod_exponent: DoD rainflow penalty shape. 0 disables.
        adp_result, adp_grids, adp_regime, adp_year_start: Intraday ADP
            solved result + lookup metadata. ``None`` disables.
        physics_wear_eur_per_mwh: Per-SoH physics wear array. ``None`` disables.
    """

    def __init__(
        self,
        *,
        name: str,
        use_soc_window: bool = False,
        soc_min_frac: float = 0.20,
        soc_max_frac: float = 0.80,
        flat_base_eur_per_mwh: float = 0.0,
        use_scarcity: bool = False,
        warranty_floor: float = 0.80,
        epsilon: float = 0.005,
        dod_multiplier: float = 0.0,
        dod_exponent: float = 1.0,
        adp_result=None,
        adp_grids=None,
        adp_regime=None,
        adp_year_start: Optional[_dt.date] = None,
        adp_scale: float = 1.0,
        physics_wear_eur_per_mwh: Optional[np.ndarray] = None,
        physics_weight: float = 1.0,
        physics_subtract_flat_base: bool = False,
        use_physics_from_duty: bool = False,
        physics_preset=None,
        physics_temperature_c: float = 25.0,
        physics_capex_eur_per_mwh: float = 100_000.0,
        physics_max_wear_eur_per_mwh: float = 500.0,
        physics_kernel_scale: float = 1.0,
    ) -> None:
        self.name = name
        self._use_soc_window = use_soc_window
        self._soc_min_frac = soc_min_frac
        self._soc_max_frac = soc_max_frac
        self._flat_base = float(flat_base_eur_per_mwh)
        self._use_scarcity = use_scarcity
        self._warranty_floor = warranty_floor
        self._epsilon = epsilon
        self._dod_multiplier = float(dod_multiplier)
        self._dod_exponent = float(dod_exponent)
        self._adp_result = adp_result
        self._adp_grids = adp_grids
        self._adp_regime = adp_regime
        self._adp_scale = float(adp_scale)
        self._physics_wear = (
            np.asarray(physics_wear_eur_per_mwh, dtype=float)
            if physics_wear_eur_per_mwh is not None else None
        )
        self._physics_weight = float(physics_weight)
        self._physics_subtract_flat_base = physics_subtract_flat_base
        self._use_physics_from_duty = use_physics_from_duty
        self._physics_preset = physics_preset
        self._physics_temperature_c = physics_temperature_c
        self._physics_capex = physics_capex_eur_per_mwh
        self._physics_max_wear = physics_max_wear_eur_per_mwh
        self._physics_kernel_scale = float(physics_kernel_scale)
        self.uses_two_pass = (
            self._dod_multiplier > 0.0 or self._use_physics_from_duty
        )
        if adp_result is not None:
            if adp_grids is None or adp_regime is None:
                raise ValueError(
                    "adp_result requires adp_grids and adp_regime to enable the ADP layer"
                )
            if adp_year_start is None:
                first_label_date = next(iter(adp_regime.regime_labels.index))
                if isinstance(first_label_date, _dt.date):
                    adp_year_start = first_label_date
                else:
                    adp_year_start = _dt.date(first_label_date.year, 1, 1)
        self._adp_year_start = adp_year_start

    def _soh_idx(self, soh_current: float, grid) -> int:
        if soh_current <= grid[0]:
            return 0
        if soh_current >= grid[-1]:
            return len(grid) - 1
        return max(0, int(np.searchsorted(grid, soh_current, side="right")) - 1)

    def _regime_for_day(self, day_of_year: int) -> int:
        if self._adp_regime is None:
            return 0
        target = self._adp_year_start + _dt.timedelta(days=int(day_of_year) - 1)
        labels = self._adp_regime.regime_labels
        if target in labels.index:
            return int(labels.loc[target])
        return int(np.argmax(self._adp_regime.stationary))

    def _base_wear(
        self, soh_current: float, periods_per_day: int,
    ) -> np.ndarray:
        base = self._flat_base
        if self._use_scarcity and base > 0.0:
            # Bounded scarcity: ``factor = 1 + (1-SoH) / (1-floor)`` — grows
            # monotonically from 1.0 at fresh cell to 2.0 at the warranty
            # floor. Gradual SoH-dependent wear increase (replaces the
            # earlier unbounded ``(1-SoH)/(SoH-floor)`` form, which was zero
            # at fresh cell — caused over-cycling early and NPV loss vs L3).
            floor = self._warranty_floor
            scarcity_factor = 1.0 + max(0.0, 1.0 - soh_current) / max(
                1.0 - floor, 1e-6
            )
            base = base * scarcity_factor
        return np.full(periods_per_day, base)

    def wear_cost(
        self,
        soh_current: float,
        day_of_year: int = 1,
        periods_per_day: int = 96,
        duration_h: float = 2.0,
    ) -> np.ndarray:
        v = self._base_wear(soh_current, periods_per_day)

        # ADP intraday layer (hour-varying opportunity cost)
        if self._adp_result is not None:
            soh_idx = self._soh_idx(soh_current, self._adp_grids.soh_grid)
            reg_idx = self._regime_for_day(day_of_year)
            hourly = self._adp_result.shadow_cost[soh_idx, reg_idx]   # (24,)
            iph = periods_per_day // 24
            vec = np.repeat(hourly, iph)
            remainder = periods_per_day - len(vec)
            if remainder > 0:
                vec = np.concatenate([vec, np.full(remainder, hourly[-1])])
            v = v + self._adp_scale * vec

        # Physics aging layer (SoH-varying, constant across intervals).
        # When ``physics_subtract_flat_base`` is True, only the DELTA above
        # the flat base is added — avoids double-counting with L3/L4 layers.
        if self._physics_wear is not None:
            soh_idx = self._soh_idx(soh_current, self._adp_grids.soh_grid
                                    if self._adp_grids is not None
                                    else np.linspace(self._warranty_floor, 1.0, len(self._physics_wear)))
            phys = float(self._physics_wear[soh_idx])
            if self._physics_subtract_flat_base:
                phys = max(0.0, phys - self._flat_base)
            v = v + self._physics_weight * phys

        return v

    def lp_overrides(
        self, soh_current: float, day_of_year: int = 1,
    ) -> dict:
        if self._use_soc_window:
            return {
                "soc_min_frac": self._soc_min_frac,
                "soc_max_frac": self._soc_max_frac,
            }
        return {}

    def refine_wear_cost(
        self,
        first_pass_soc: np.ndarray,
        energy_mwh: float,
        soh_current: float,
        day_of_year: int = 1,
        periods_per_day: int = 96,
        duration_h: float = 2.0,
        day_result=None,
    ) -> np.ndarray:
        """Second-pass wear given the first-pass dispatch.

        Composes in order:
        1. Base wear from :meth:`wear_cost` (flat/scarcity/ADP/physics-1D).
        2. Duty-based physics wear (additive) — requires ``day_result``.
        3. DoD rainflow multiplicative penalty on SoC extremes.
        """
        base = self.wear_cost(
            soh_current=soh_current, day_of_year=day_of_year,
            periods_per_day=periods_per_day, duration_h=duration_h,
        )
        # 2. Duty-based physics wear — calls the Note 3 kernel on the
        # observed duty and adds the resulting scalar EUR/MWh cost.
        if (self._use_physics_from_duty and day_result is not None
                and self._physics_preset is not None):
            from lib.analysis.physics_wear_lookup import physics_wear_from_duty
            phys_scalar = physics_wear_from_duty(
                day_result=day_result, energy_mwh=energy_mwh,
                soh_current=soh_current, preset=self._physics_preset,
                temperature_c=self._physics_temperature_c,
                capex_eur_per_mwh=self._physics_capex,
                warranty_floor=self._warranty_floor,
                epsilon=self._epsilon,
                max_wear_eur_per_mwh=self._physics_max_wear,
                kernel_scale=self._physics_kernel_scale,
            )
            base = base + self._physics_weight * phys_scalar

        # 3. DoD rainflow multiplicative penalty on extreme-SoC throughput.
        if self._dod_multiplier > 0.0 and energy_mwh > 0 \
                and len(first_pass_soc) == periods_per_day:
            soc_frac = np.clip(first_pass_soc / energy_mwh, 0.0, 1.0)
            dev = np.abs(soc_frac - 0.5)
            multiplier = 1.0 + self._dod_multiplier * (dev ** self._dod_exponent)
            base = base * multiplier
        return base


class RainflowPolicy(ShadowCostPolicy):
    """Rainflow sidebar: per-cycle DoD penalty via 2-pass post-hoc rainflow extraction.

    Methodology sibling of :class:`ProgressivePolicy` (does NOT stack on
    L1–L6 — it is an alternative to L4/L5/L6, not an addition). Mirror
    of the Collath sidebar but with a structurally different penalty
    target: instead of throughput-per-window (Collath), the second-pass
    wear is derived from rainflow-extracted cycle DoDs evaluated against
    a convex per-cycle aging function f(DoD) calibrated against the
    Note 3 Wang+Naumann kernel at ``kernel_scale = 0.66`` (Note 4
    manufacturer anchor — same as L6). This is channel (b) in the
    [ROADMAP] taxonomy: *"a rainflow-based piecewise cost that depends
    on DoD of each cycle"*. Built around Shi-Xu et al. (2018,
    arXiv:1703.07968) — convex rainflow-cycle cost in SoC trajectory.

    Pipeline per day:
      1. Pass 1: solve LP with a small bootstrap wear (so the LP has a
         reason to leave power on the table when revenue is thin) and
         the SoC envelope active.
      2. Extract rainflow cycles ``(DoD, mean_SoC)`` from the first-pass
         SoC trajectory.
      3. Evaluate piecewise-linear f(DoD) at each cycle, sum, age-scale,
         convert to scalar EUR/MWh, cap at ``max_wear``.
      4. Pass 2: solve LP again with the refined scalar wear.

    See :func:`lib.analysis.rainflow_wear.rainflow_wear_from_duty` for
    the second-pass formula and
    :doc:`scripts/rainflow_sidebar/rainflow_calibrate_eve_lf280k.py` for
    the calibration that fills ``rainflow_coeffs``.

    Args:
        name: Human-readable identifier (default "Rainflow_sidebar").
        rainflow_coeffs: dict with ``dod_breakpoints`` and ``cycle_fade``
            arrays from the calibration ``.npz``.
        soc_min_frac, soc_max_frac: SoC envelope bounds. Default
            [0.20, 0.80] matches L2.
        bootstrap_wear_eur_per_mwh: Pass-1 warm-start wear. Default
            €30/MWh — small enough to let the LP cycle, large enough
            to avoid degenerate "fill the cap" first passes.
        capex_eur_per_mwh: CAPEX anchor for monetisation. Default
            €100k/MWh, same as L6.
        warranty_floor: SoH below which warranty void. Default 0.80.
        max_wear_eur_per_mwh: Pass-2 cap to keep LP numerics tractable
            near the warranty floor (default €500/MWh, same as L6).
        age_accel_slope: Per-unit SoH-loss acceleration on the per-
            cycle fade. Default 2.5, same as L6.
    """

    uses_two_pass: bool = True

    def __init__(
        self,
        *,
        name: str = "Rainflow_sidebar",
        rainflow_coeffs: dict,
        soc_min_frac: float = 0.20,
        soc_max_frac: float = 0.80,
        bootstrap_wear_eur_per_mwh: float = 30.0,
        capex_eur_per_mwh: float = 100_000.0,
        warranty_floor: float = 0.80,
        epsilon: float = 0.005,
        max_wear_eur_per_mwh: float = 500.0,
        age_accel_slope: float = 2.5,
    ) -> None:
        self.name = name
        if "dod_breakpoints" not in rainflow_coeffs or "cycle_fade" not in rainflow_coeffs:
            raise ValueError(
                "rainflow_coeffs must contain 'dod_breakpoints' and 'cycle_fade'"
            )
        self._coeffs = {
            "dod_breakpoints": np.asarray(rainflow_coeffs["dod_breakpoints"], dtype=float),
            "cycle_fade": np.asarray(rainflow_coeffs["cycle_fade"], dtype=float),
        }
        self._soc_min_frac = float(soc_min_frac)
        self._soc_max_frac = float(soc_max_frac)
        self._bootstrap = float(bootstrap_wear_eur_per_mwh)
        self._capex = float(capex_eur_per_mwh)
        self._warranty_floor = float(warranty_floor)
        self._epsilon = float(epsilon)
        self._max_wear = float(max_wear_eur_per_mwh)
        self._age_accel_slope = float(age_accel_slope)

    def wear_cost(
        self,
        soh_current: float,
        day_of_year: int = 1,
        periods_per_day: int = 96,
        duration_h: float = 2.0,
    ) -> np.ndarray:
        """Pass-1 bootstrap wear (flat scalar)."""
        return np.full(periods_per_day, self._bootstrap)

    def lp_overrides(
        self, soh_current: float, day_of_year: int = 1,
    ) -> dict:
        return {
            "soc_min_frac": self._soc_min_frac,
            "soc_max_frac": self._soc_max_frac,
        }

    def refine_wear_cost(
        self,
        first_pass_soc: np.ndarray,
        energy_mwh: float,
        soh_current: float,
        day_of_year: int = 1,
        periods_per_day: int = 96,
        duration_h: float = 2.0,
        day_result=None,
    ) -> np.ndarray:
        """Pass-2 refined wear from rainflow-cycle aging."""
        if day_result is None or energy_mwh <= 0:
            return self.wear_cost(soh_current, day_of_year, periods_per_day, duration_h)
        from lib.analysis.rainflow_wear import rainflow_wear_from_duty
        scalar = rainflow_wear_from_duty(
            day_result=day_result,
            energy_mwh=energy_mwh,
            soh_current=soh_current,
            rainflow_coeffs=self._coeffs,
            capex_eur_per_mwh=self._capex,
            warranty_floor=self._warranty_floor,
            epsilon=self._epsilon,
            max_wear_eur_per_mwh=self._max_wear,
            age_accel_slope=self._age_accel_slope,
        )
        return np.full(periods_per_day, scalar)


# Convenience factory for Note 4 policy comparisons.
def default_policies_for_comparison(
    capex_eur_per_mwh: float = 100_000.0,
    lifetime_throughput_ratio: float = 6_000.0,
    warranty_floor: float = 0.80,
) -> dict[str, ShadowCostPolicy]:
    """Returns the three built policies keyed by name.

    Use this in Note 4 precompute to run side-by-side dispatch under
    identical market conditions and compare lifetime NPVs. The ADP policy
    is not included yet (A3.3 stub); ``AgingAwareDepreciation`` stands in
    as the "best available state-aware" proxy.
    """
    base = capex_eur_per_mwh / max(lifetime_throughput_ratio, 1.0)
    return {
        "naive": NaivePolicy(),
        "depreciation_proxy": DepreciationProxyPolicy(
            capex_eur_per_mwh=capex_eur_per_mwh,
            lifetime_throughput_ratio=lifetime_throughput_ratio,
        ),
        "aging_aware_depreciation": AgingAwareDepreciationPolicy(
            base_eur_per_mwh=base,
            warranty_floor=warranty_floor,
        ),
    }
