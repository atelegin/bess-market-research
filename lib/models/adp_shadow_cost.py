"""
Shadow-cost policies for degradation-aware dispatch (Note 4 A3).

Each policy produces the per-interval ``wear_cost_eur_per_mwh`` vector that
feeds ``lib.models.dispatch_stacked.optimize_day_stacked``. The vector
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
    """

    name: str

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


class ADPPolicy(ShadowCostPolicy):
    """Shadow cost from a solved backward-induction DP (A3.4 online lookup).

    Wraps a solved :class:`lib.models.adp_solver.ADPSolver` and exposes its
    per-state shadow cost through the ``ShadowCostPolicy`` interface. At
    each call, finds the grid bucket for the current ``SoH`` and the
    regime for the given calendar day, then emits a flat
    ``wear_cost_eur_per_mwh`` vector for the day.

    Construction
    ------------
    ::

        from lib.models.price_regime import fit_regimes
        from lib.models.adp_solver import ADPSolver, default_grids, empirical_daily_revenue_curve

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

    Wraps an :class:`lib.models.adp_solver_intraday.IntradayADPResult`.
    At online-lookup time, snaps (SoH, regime) to grid, returns 24-hour
    shadow cost vector expanded to 96 × 15-min intervals.

    The hour-varying signal is what the simplified-state DP cannot
    produce: same shadow cost at peak vs trough, same at fresh vs aged.
    This policy is the principled Holtorf-Shin reference and the
    empirical answer to "do we need full intraday DP for Note 4?".
    """

    name: str = "adp_intraday"

    def __init__(
        self,
        intraday_result,
        grids,
        regime_classification,
        year_start: Optional[_dt.date] = None,
        regime_override: Optional[int] = None,
    ) -> None:
        self._result = intraday_result
        self._grids = grids
        self._regime = regime_classification
        self._regime_override = regime_override
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
        hourly = self._result.shadow_cost[soh_idx, reg_idx]  # (24,)
        # Expand to periods_per_day by repeating each hour's value
        intervals_per_hour = periods_per_day // 24
        remainder = periods_per_day - 24 * intervals_per_hour
        vector = np.repeat(hourly, intervals_per_hour)
        if remainder > 0:
            vector = np.concatenate([vector, np.full(remainder, hourly[-1])])
        return vector


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
