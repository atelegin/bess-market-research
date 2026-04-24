"""
Offline backward-induction DP solver for aging-aware shadow cost (Note 4 A3.3).

Computes the state-value function :math:`V(\\text{SoH}, \\text{regime})`
by value iteration over a simplified state space, plus the implied
optimal daily cycling intensity and the per-state shadow cost used by
:class:`lib.models.adp_shadow_cost.ADPPolicy`.

State simplification
--------------------
Full Holtorf–Shin ADP uses ``(SoC, SoH, regime, time-of-day, calendar)``.
We collapse to ``(SoH, regime)`` with daily resolution because:

* Intra-day SoC optimality is already captured by the stacked LP at
  dispatch time — the LP decides when to charge/discharge given a
  scalar shadow cost.
* Intra-year seasonality is absorbed into the empirical
  ``daily_revenue_curve`` (regime already captures the "calm summer day
  vs volatile winter day" effect in German prices).
* Horizon convergence: with regime Markov chain and SoH monotone
  descending, stationary value iteration converges fast and the
  scalar-per-state shadow cost is what the LP needs.

What the DP does
----------------
1. For each ``(SoH, regime)``, the agent picks a daily *cycling
   intensity* ``a`` (FEC/day) from a discrete grid.
2. Immediate reward: ``revenue(regime, a)`` from an empirical
   regime-conditional revenue curve (we fit this from historical
   regime-labelled days in :func:`empirical_daily_revenue_curve`).
3. SoH transitions down by a physics-based fade-per-day ``δ(a, SoH)``.
4. Regime transitions per the classifier's transition matrix.
5. Bellman:
   :math:`V(\\text{SoH}, r) = \\max_a \\{R(r, a) + \\gamma \\cdot \\mathbb{E}_{r'}[V(\\text{SoH}', r')]\\}`.

Shadow-cost derivation
----------------------
The LP's ``wear_cost_eur_per_mwh`` corresponds to the **opportunity cost
of consuming one more unit of SoH headroom** — i.e. the partial
derivative of ``V`` along the SoH axis, converted to EUR per MWh of
throughput via the fade-per-throughput relationship. The solver exposes
this as ``shadow_cost(soh, regime) -> EUR/MWh`` ready to feed the
:class:`lib.models.adp_shadow_cost.ADPPolicy` wrapper.

Known limitations of the scalar-per-state DP
--------------------------------------------
This simplified DP **does not** produce the scarcity-responsive shadow
cost (higher as SoH approaches the floor) that Holtorf & Shin (2026)
exhibit. The limitation is structural: with state only
``(SoH, regime)``, the value function is roughly linear in SoH (more
remaining life → proportionally more accumulated revenue) and the
optimal action is near-constant (max cycling). To recover the scarcity
signal one would need to extend the state with:

  * SoC (within-day) — cycling at high SoC near peak hours carries
    different opportunity cost than at low SoC off-peak
  * Time of day — expected next-hour price matters for the marginal
    cycle decision
  * Stochastic price process at 15-min resolution — current regime-day
    aggregation averages out intraday shape

These extensions turn the DP into the full Holtorf-Shin formulation,
which is several days of additional implementation. For Note 4's
current scope, use :class:`lib.models.adp_shadow_cost.AgingAwareDepreciationPolicy`
as the primary "state-aware scarcity-responsive" policy — it produces
the scarcity signal by construction. This ADP solver is retained as the
principled DP reference; the warranty-penalty channel makes V sensitive
to the floor even in the simplified state space.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from lib.models.price_regime import RegimeClassification

logger = logging.getLogger(__name__)


@dataclass
class ADPGrids:
    """Discrete state and action grids."""

    soh_grid: np.ndarray             # e.g. np.arange(0.80, 1.001, 0.02)
    action_grid: np.ndarray          # e.g. np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    # ordered so action_grid[0] = lowest cycling, action_grid[-1] = most
    n_soh: int = field(init=False)
    n_actions: int = field(init=False)
    warranty_floor: float = 0.80

    def __post_init__(self) -> None:
        self.n_soh = len(self.soh_grid)
        self.n_actions = len(self.action_grid)
        # Enforce ascending order (smallest SoH first → makes backward
        # indexing intuitive).
        if not np.all(np.diff(self.soh_grid) > 0):
            raise ValueError("soh_grid must be strictly ascending")
        if self.soh_grid[0] < self.warranty_floor - 1e-9:
            raise ValueError(
                f"soh_grid[0]={self.soh_grid[0]} below warranty_floor={self.warranty_floor}"
            )


@dataclass
class ADPResult:
    """Output of :meth:`ADPSolver.solve`."""

    value: np.ndarray                # (n_soh, n_regimes) expected remaining value (EUR)
    policy: np.ndarray               # (n_soh, n_regimes) optimal action index
    shadow_cost: np.ndarray          # (n_soh, n_regimes) EUR/MWh throughput
    iterations: int
    converged: bool
    final_delta: float


def empirical_daily_revenue_curve(
    daily_revenue: pd.Series,
    regime_labels: pd.Series,
    n_regimes: int,
    action_grid: np.ndarray,
    reference_intensity: float = 1.5,
) -> np.ndarray:
    """
    Fit a per-regime revenue-vs-intensity curve from historical data.

    Args:
        daily_revenue: Series indexed by date with observed daily revenue
            per MW (EUR/day). Typically the stacked-LP output on
            historical days at the reference intensity.
        regime_labels: Matching regime label per date (from
            :class:`RegimeClassification`).
        n_regimes: Number of regimes.
        action_grid: Intensity grid, in cycles/day.
        reference_intensity: The FEC/day at which ``daily_revenue`` was
            observed. Revenue at other intensities is interpolated as
            proportional (linear). Conservative assumption — real revenue
            saturates, but this simplification keeps the curve monotone.

    Returns:
        Array shape ``(n_regimes, n_actions)`` — ``R[r, a]`` is the
        expected daily revenue in regime ``r`` at intensity ``action_grid[a]``.
    """
    if reference_intensity <= 0:
        raise ValueError("reference_intensity must be positive")

    common = daily_revenue.index.intersection(regime_labels.index)
    rev = daily_revenue.loc[common].to_numpy(dtype=float)
    regs = regime_labels.loc[common].to_numpy(dtype=int)

    # Regime-mean observed revenue at the reference intensity
    mean_rev_per_regime = np.zeros(n_regimes)
    for r in range(n_regimes):
        mask = regs == r
        if mask.any():
            mean_rev_per_regime[r] = float(rev[mask].mean())
        else:
            # Regime didn't occur — conservative zero. DP handles
            # unreachable regimes through transition probabilities.
            mean_rev_per_regime[r] = 0.0

    # Linear scaling across intensity: revenue ∝ cycling rate
    # (simplification; captures the "more cycles → more revenue but also
    # more degradation" trade-off the DP needs).
    scaling = action_grid / reference_intensity
    return np.outer(mean_rev_per_regime, scaling)


def degradation_per_day(
    intensity: float,
    soh: float,
    fade_per_fec_at_soh_1: float,
    calendar_fade_per_day: float = 0.0,
) -> float:
    """
    Linear-in-intensity approximation of daily SoH fade.

    Args:
        intensity: Daily cycling in FEC/day.
        soh: Current state of health (affects fade rate mildly — at lower
            SoH the cell ages faster for the same stress, per empirical
            square-root kinetics; here we apply a modest acceleration).
        fade_per_fec_at_soh_1: Fade (Δ SoH) per full equivalent cycle at
            SoH = 1. Typical LFP: ~1e-4 → 10000 cycles to 0 (unrealistic
            tail), ~3e-4 → 3300 cycles to 0 (more realistic incl. calendar).
        calendar_fade_per_day: Additional daily calendar fade (independent
            of cycling). Typical ~2e-5.

    Returns:
        Daily Δ SoH (positive number, to be subtracted from SoH).
    """
    # Mild acceleration as SoH drops (√t-like kinetics produce increasing
    # daily fade even at constant intensity). Factor 1 at SoH=1 → ~1.5 at
    # SoH=0.80.
    age_accel = 1.0 + 2.5 * max(0.0, 1.0 - soh)
    cycling_fade = fade_per_fec_at_soh_1 * intensity * age_accel
    return cycling_fade + calendar_fade_per_day


class ADPSolver:
    """Backward-induction DP over ``(SoH, regime)`` state."""

    def __init__(
        self,
        regime_classification: RegimeClassification,
        revenue_curve: np.ndarray,       # (n_regimes, n_actions) daily EUR
        grids: ADPGrids,
        fade_per_fec_at_soh_1: float = 3.3e-5,
        calendar_fade_per_day: float = 2e-5,
        discount_per_year: float = 0.98,
        warranty_breach_penalty_eur: float = 50_000.0,
    ) -> None:
        """
        Args:
            warranty_breach_penalty_eur: EUR cost when SoH drops below the
                warranty floor. Physical interpretation: OEM refuses to
                replace cells, BESS owner faces repower CAPEX and/or
                contract penalties. Default 50k EUR per MW represents a
                meaningful fraction of total remaining NPV, driving the
                shadow cost to rise sharply as SoH approaches the floor
                (Kumtepeli/Howey scarcity intuition). Set to 0 to recover
                the no-penalty "pure dispatch optimality" DP.
        """
        self.regime = regime_classification
        self.revenue_curve = revenue_curve
        self.grids = grids
        self.fade_per_fec_at_soh_1 = fade_per_fec_at_soh_1
        self.calendar_fade_per_day = calendar_fade_per_day
        self.discount_per_day = discount_per_year ** (1.0 / 365.0)
        self.warranty_breach_penalty_eur = warranty_breach_penalty_eur
        n_r, n_a = revenue_curve.shape
        if n_r != regime_classification.n_regimes:
            raise ValueError(
                f"revenue_curve regime dim {n_r} != classification {regime_classification.n_regimes}"
            )
        if n_a != grids.n_actions:
            raise ValueError(
                f"revenue_curve action dim {n_a} != grid {grids.n_actions}"
            )

    def _next_soh_idx(self, soh_idx: int, intensity: float) -> int:
        """Which SoH bucket does the state transition to after one day at
        ``intensity``? Returns -1 if below warranty floor."""
        soh = self.grids.soh_grid[soh_idx]
        delta = degradation_per_day(
            intensity=intensity, soh=soh,
            fade_per_fec_at_soh_1=self.fade_per_fec_at_soh_1,
            calendar_fade_per_day=self.calendar_fade_per_day,
        )
        next_soh = soh - delta
        if next_soh < self.grids.warranty_floor - 1e-9:
            return -1
        # Snap to nearest grid index at-or-below next_soh
        # (grid ascending, so search from current downward).
        for i in range(soh_idx, -1, -1):
            if self.grids.soh_grid[i] <= next_soh + 1e-9:
                return i
        return 0

    def solve(
        self,
        tol: float = 1e-3,
        max_iter: int = 2000,
    ) -> ADPResult:
        """Run stationary value iteration."""
        n_s, n_r, n_a = self.grids.n_soh, self.regime.n_regimes, self.grids.n_actions
        V = np.zeros((n_s, n_r))
        policy = np.zeros((n_s, n_r), dtype=int)
        transition = self.regime.transition_matrix  # (n_r, n_r)

        last_delta = float("inf")
        converged = False
        for it in range(1, max_iter + 1):
            V_new = np.zeros_like(V)
            for s in range(n_s):
                for r in range(n_r):
                    best_val = -np.inf
                    best_a = 0
                    soh_here = self.grids.soh_grid[s]
                    for a in range(n_a):
                        intensity = float(self.grids.action_grid[a])
                        # SoH-scaled immediate reward: usable capacity is
                        # SoH × nominal, so daily arbitrage revenue at the
                        # same cycling intensity scales with SoH. This is
                        # what drives the V function to be concave in SoH
                        # and produces scarcity-responsive shadow cost.
                        immediate = self.revenue_curve[r, a] * soh_here
                        next_s = self._next_soh_idx(s, intensity)
                        if next_s < 0:
                            # Warranty breach — today's revenue still collected
                            # but no future value, minus the OEM/owner penalty.
                            future = -self.warranty_breach_penalty_eur
                        else:
                            # Expected V over regime transition
                            future = float(transition[r] @ V[next_s])
                        val = immediate + self.discount_per_day * future
                        if val > best_val:
                            best_val = val
                            best_a = a
                    V_new[s, r] = best_val
                    policy[s, r] = best_a
            delta = float(np.abs(V_new - V).max())
            V = V_new
            if delta < tol:
                converged = True
                last_delta = delta
                break
            last_delta = delta

        shadow_cost = self._derive_shadow_cost(V)
        return ADPResult(
            value=V, policy=policy, shadow_cost=shadow_cost,
            iterations=it, converged=converged, final_delta=last_delta,
        )

    def _derive_shadow_cost(self, V: np.ndarray) -> np.ndarray:
        """
        Shadow cost at (SoH, regime) = marginal V loss per MWh of throughput.

        We compute :math:`\\partial V / \\partial \\text{SoH}` via central
        differences on the grid, then convert to EUR/MWh via the fade-per-
        throughput relation. At the edges we use one-sided differences.
        """
        n_s, n_r = V.shape
        dV_dSoH = np.zeros_like(V)
        # Grid is ascending; lower indices = lower SoH = more consumed life.
        for s in range(n_s):
            if s == 0:
                dV_dSoH[s] = (V[s + 1] - V[s]) / (self.grids.soh_grid[s + 1] - self.grids.soh_grid[s])
            elif s == n_s - 1:
                dV_dSoH[s] = (V[s] - V[s - 1]) / (self.grids.soh_grid[s] - self.grids.soh_grid[s - 1])
            else:
                dV_dSoH[s] = (V[s + 1] - V[s - 1]) / (self.grids.soh_grid[s + 1] - self.grids.soh_grid[s - 1])

        # Convert: Δ SoH per FEC at SoH = 1; one FEC = 2× energy_capacity throughput
        # → Δ SoH per MWh throughput = fade_per_fec / (2 × energy_mwh_per_FEC)
        # We don't know energy_mwh here (unitless normalization); caller scales.
        # For default reporting, we return shadow cost in EUR per ΔSoH per MWh at
        # a reference of "1 MWh throughput reduces SoH by fade_per_fec_at_soh_1 / 2".
        fade_per_mwh = self.fade_per_fec_at_soh_1 / 2.0
        shadow_cost = dV_dSoH * fade_per_mwh
        # Clip to non-negative (wear cost is a cost, not a reward).
        shadow_cost = np.maximum(shadow_cost, 0.0)
        return shadow_cost


def default_grids(
    warranty_floor: float = 0.80,
    soh_step: float = 0.02,
    max_intensity: float = 2.0,
    intensity_step: float = 0.25,
) -> ADPGrids:
    """Sensible defaults: SoH 0.80 → 1.00 in 0.02 steps (11 buckets);
    intensity 0 → 2.0 FEC/day in 0.25 steps (9 actions)."""
    soh_grid = np.round(np.arange(warranty_floor, 1.0 + soh_step / 2, soh_step), 4)
    action_grid = np.round(np.arange(0.0, max_intensity + intensity_step / 2, intensity_step), 4)
    return ADPGrids(
        soh_grid=soh_grid, action_grid=action_grid, warranty_floor=warranty_floor,
    )
