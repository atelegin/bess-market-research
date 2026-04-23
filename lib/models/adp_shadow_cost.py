"""
Shadow-cost policies for degradation-aware dispatch (Note 4 A3).

Each policy produces the per-interval ``wear_cost_eur_per_mwh`` vector that
feeds ``lib.models.dispatch_stacked.optimize_day_stacked``. The vector
represents the economic cost of cycling at each 15-min slot: the LP
subtracts it from revenue, suppressing marginal trades that don't clear
their wear cost.

Three policies, increasing in sophistication and meant for side-by-side
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

4. :class:`ADPPolicy` — full state-dependent via offline dynamic
   programming (A3.3). Returns ``∂V/∂SoC`` from a backward-induction
   solution, capturing the Kumtepeli/Howey "forgone future revenue"
   interpretation. **Placeholder in A3.1**; implementation lands in A3.3.

All policies expose the same ``wear_cost(soh_current, day_of_year,
periods_per_day, duration_h) -> np.ndarray`` interface so the Note 4
dispatcher can swap them at LP-call time.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class ShadowCostPolicy(ABC):
    """Interface: maps (current asset state, calendar) → per-interval EUR/MWh.

    The returned array is what an aging-aware dispatcher subtracts from
    energy revenue in its LP objective. Length ``periods_per_day`` (96 by
    default). Values are non-negative by convention; negative shadow cost
    makes no physical sense (cycling can't be *rewarded* by wear).
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


class ADPPolicy(ShadowCostPolicy):
    """Full state-dependent shadow cost from offline backward-induction DP.

    **Placeholder (A3.1)**: the implementation lands in A3.3 along with
    :mod:`lib.models.adp_solver`. The interface will match the other
    policies so Note 4's lifecycle loop can swap policies freely.

    Intended behaviour: pre-computes :math:`V(SoC, SoH, regime, season)` by
    backward value iteration over a lifetime horizon; at online-lookup
    time, returns :math:`\\partial V / \\partial SoC` evaluated on the grid
    slice for the current ``(SoH, regime, season)`` → per-interval wear
    cost vector. Matches the construction in Holtorf & Shin (2026,
    arXiv:2603.21089) with simplified grid resolution.
    """

    name = "adp"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "ADPPolicy is stubbed in A3.1; the backward-induction solver "
            "lands in A3.3 (lib.models.adp_solver). Use "
            "AgingAwareDepreciationPolicy for the closed-form stepping-stone."
        )

    def wear_cost(
        self,
        soh_current: float,
        day_of_year: int = 1,
        periods_per_day: int = 96,
        duration_h: float = 2.0,
    ) -> np.ndarray:
        raise NotImplementedError


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
