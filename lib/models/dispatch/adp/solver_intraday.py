"""
Intraday backward-induction DP — extends the scalar DP with SoC and
hour-of-day state (Note 4 A3-extension, Holtorf-Shin style).

The scalar ``(SoH, regime)`` DP in :mod:`lib.models.dispatch.adp.solver` produces
a nearly-flat shadow cost because all intra-day price variation is
averaged into a single daily revenue number. Real ADP dispatch needs a
shadow cost that *varies by hour* — pricing cycling higher when expected
future prices are better (save SoC for the peak), lower when spread
opportunities are spent.

This module solves a 2D DP ``(SoC, hour-of-day)`` for each ``(SoH, regime)``
slice, using per-regime hourly mean price profiles fit from historical
DA data. The output is a 3D shadow-cost lookup
``shadow_cost[SoH_idx, regime_idx, hour]`` that :class:`ADPPolicyIntraday`
looks up online at LP call time.

Simplifications vs full Holtorf-Shin
------------------------------------
* Deterministic hourly price means per regime (no stochastic transitions
  within the day — regime is fixed for the day).
* SoH within a day assumed constant (fade happens at day boundaries).
* Cyclic SoC boundary (end-of-day SoC ≈ start-of-day SoC) ensures the
  steady-state daily cycle is well-posed.
* Action space: charge/discharge MW at 5 levels, symmetric about zero.

These simplifications keep the DP tractable (<1 s per slice) while still
producing genuine hour-dependent shadow cost — the core Holtorf-Shin
mechanic.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from lib.models.dispatch.adp.solver import ADPGrids, default_grids, degradation_per_day
from lib.models.price_regime import RegimeClassification

logger = logging.getLogger(__name__)


@dataclass
class IntradayADPResult:
    """Output of :meth:`IntradayADPSolver.solve`."""
    # shadow_cost: (n_soh, n_regime, 24) EUR/MWh throughput at each hour
    shadow_cost: np.ndarray
    # value_table: (n_soh, n_regime, n_soc, 24) V for each state at each hour
    value_table: np.ndarray
    n_hours: int = 24


def fit_hourly_price_profiles(
    prices: pd.Series,
    regime_classification: RegimeClassification,
) -> np.ndarray:
    """
    Fit mean DA price per hour-of-day per regime (24 × n_regimes table).

    Groups observations by (date → regime label) and hour, returns
    mean EUR/MWh per bucket.
    """
    if prices.index.tz is None:
        local = prices.tz_localize("UTC")
    else:
        local = prices.tz_convert("UTC")
    df = pd.DataFrame({"price": local.astype(float)})
    df["date"] = local.index.normalize().date
    df["hour"] = local.index.hour
    regime_map = regime_classification.regime_labels
    df["regime"] = df["date"].map(regime_map)
    df = df.dropna(subset=["regime"])
    df["regime"] = df["regime"].astype(int)

    n_r = regime_classification.n_regimes
    profiles = np.zeros((n_r, 24))
    for r in range(n_r):
        sub = df[df["regime"] == r]
        if sub.empty:
            continue
        hourly = sub.groupby("hour")["price"].mean()
        for h in range(24):
            if h in hourly.index:
                profiles[r, h] = float(hourly.loc[h])
    return profiles


def fit_hourly_price_scenarios(
    prices: pd.Series,
    regime_classification: RegimeClassification,
    n_scenarios: int = 20,
    random_state: int = 42,
) -> np.ndarray:
    """
    Per-regime bootstrap of historical daily price curves — returns shape
    ``(n_regimes, n_scenarios, 24)``. Each scenario is the 24-hour DA
    price vector from one historical day that was classified into this
    regime.

    Used by :class:`IntradayADPSolver` in "full-stochastic" mode:
    closes the deterministic-hourly-means simplification (Note 4
    methodology #1) by letting the DP take expectation over real daily
    realisations rather than averaging prices away. Because the intra-
    day optimisation is nonlinear in price (depth of daily spread →
    optimal cycle depth), expectation over daily realisations produces
    a different — and richer — shadow-cost signal than the mean-price
    DP.

    Args:
        n_scenarios: Number of bootstrap samples per regime. 20 is a
            reasonable default; increase for tighter expectations at
            O(n) runtime cost.
        random_state: Seed for reproducibility.
    """
    if prices.index.tz is None:
        local = prices.tz_localize("UTC")
    else:
        local = prices.tz_convert("UTC")
    df = pd.DataFrame({"price": local.astype(float)})
    df["date"] = local.index.normalize().date
    df["hour"] = local.index.hour
    regime_map = regime_classification.regime_labels
    df["regime"] = df["date"].map(regime_map)
    df = df.dropna(subset=["regime"])
    df["regime"] = df["regime"].astype(int)

    # Pivot to one row per day × 24 columns
    daily = df.pivot_table(
        index=["date", "regime"], columns="hour", values="price",
    )
    daily = daily.dropna()  # drop days with missing hours
    if daily.shape[0] == 0:
        raise ValueError("No complete daily price curves found")
    daily["regime"] = daily.index.get_level_values("regime")

    rng = np.random.default_rng(random_state)
    n_r = regime_classification.n_regimes
    scenarios = np.zeros((n_r, n_scenarios, 24))
    for r in range(n_r):
        sub = daily[daily["regime"] == r]
        if len(sub) == 0:
            # Fallback: repeat regime-mean across scenarios
            mean_prof = daily.drop(columns="regime").mean(axis=0).values
            scenarios[r] = np.tile(mean_prof, (n_scenarios, 1))
            continue
        # Sample with replacement
        idx = rng.integers(0, len(sub), size=n_scenarios)
        picked = sub.drop(columns="regime").iloc[idx].values  # (n_scenarios, 24)
        scenarios[r] = picked
    return scenarios


class IntradayADPSolver:
    """Backward-induction DP over ``(SoC, hour)`` per ``(SoH, regime)`` slice."""

    def __init__(
        self,
        regime_classification: RegimeClassification,
        hourly_price_profiles: np.ndarray,   # (n_regimes, 24) EUR/MWh
        grids: ADPGrids | None = None,
        rte: float = 0.85,
        energy_mwh: float = 2.0,
        power_mw: float = 1.0,
        fade_per_fec_at_soh_1: float = 3.3e-5,
        calendar_fade_per_day: float = 2e-5,
        n_soc_buckets: int = 11,
        n_actions: int = 11,     # discharge -P to +P in odd number of steps
        warranty_breach_penalty_eur: float = 200_000.0,
        discount_per_year: float = 0.98,
        physics_wear_eur_per_mwh: Optional[np.ndarray] = None,
    ) -> None:
        """
        Args:
            hourly_price_profiles: shape ``(n_regimes, 24)`` — mean DA
                price per (regime, hour).
            energy_mwh, power_mw: Asset specs. Determines SoC arithmetic
                and action magnitudes.
            n_soc_buckets: SoC grid resolution. 11 = 10 % steps.
            n_actions: Action grid resolution. 11 = power/5 steps from
                −P_max to +P_max (discharge negative, charge positive
                — but LP-side wear cost is on *throughput* so sign is
                absorbed).
            physics_wear_eur_per_mwh: Optional per-SoH wear cost array
                (shape ``(len(grids.soh_grid),)``) from
                :func:`lib.analysis.physics_wear_lookup.physics_wear_cost_per_mwh`.
                **Deprecated mode** (kept for experimentation): when set,
                the Bellman inner loop subtracts ``wear × |throughput|``
                from the hourly reward. Empirically this flattens the
                output ``|∂V/∂SoC|`` signal because the DP policy
                already suppresses cycling internally — so the LP-side
                wear number becomes smaller than baseline and cycling
                increases. The preferred integration path is to keep
                the DP pure arbitrage and add the physics wear at the
                online :class:`ADPPolicyIntraday.wear_cost` layer
                (``shadow = |∂V/∂SoC| + physics_wear(SoH)``). See
                Note 4 methodology expander for the derivation.
        """
        self.regime = regime_classification
        self.profiles = hourly_price_profiles
        self.grids = grids if grids is not None else default_grids()
        self.rte = rte
        self.energy_mwh = energy_mwh
        self.power_mw = power_mw
        self.fade_per_fec_at_soh_1 = fade_per_fec_at_soh_1
        self.calendar_fade_per_day = calendar_fade_per_day
        self.warranty_breach_penalty_eur = warranty_breach_penalty_eur
        self.discount_per_day = discount_per_year ** (1.0 / 365.0)

        self.soc_grid = np.linspace(0.0, energy_mwh, n_soc_buckets)
        self.action_grid = np.linspace(-power_mw, power_mw, n_actions)
        self.physics_wear = (
            np.asarray(physics_wear_eur_per_mwh, dtype=float)
            if physics_wear_eur_per_mwh is not None else None
        )
        if self.physics_wear is not None:
            if self.physics_wear.shape != self.grids.soh_grid.shape:
                raise ValueError(
                    f"physics_wear_eur_per_mwh shape {self.physics_wear.shape} "
                    f"must match soh_grid shape {self.grids.soh_grid.shape}"
                )
        # Detect stochastic mode: 3D profiles (n_regimes, n_scenarios, 24) vs
        # deterministic 2D (n_regimes, 24). Stochastic averages shadow cost
        # over sampled daily realisations — closes methodology simplification
        # #1 (deterministic hourly means).
        if self.profiles.ndim == 3:
            self._stochastic = True
            self._n_scenarios = int(self.profiles.shape[1])
            if self.profiles.shape != (regime_classification.n_regimes,
                                        self._n_scenarios, 24):
                raise ValueError(
                    f"hourly_price_profiles stochastic mode must be shape "
                    f"({regime_classification.n_regimes}, n_scenarios, 24), "
                    f"got {self.profiles.shape}"
                )
        else:
            self._stochastic = False
            self._n_scenarios = 1
        # Sanity: accept (n_regimes, 24) or (n_regimes, n_scenarios, 24)
        if hourly_price_profiles.ndim == 2:
            if hourly_price_profiles.shape != (regime_classification.n_regimes, 24):
                raise ValueError(
                    f"hourly_price_profiles deterministic mode must be shape "
                    f"({regime_classification.n_regimes}, 24), "
                    f"got {hourly_price_profiles.shape}"
                )
        elif hourly_price_profiles.ndim != 3:
            raise ValueError(
                f"hourly_price_profiles must be 2D (n_regimes, 24) or 3D "
                f"(n_regimes, n_scenarios, 24), got ndim={hourly_price_profiles.ndim}"
            )

    def _next_soc(self, soc: float, action: float, dt_h: float = 1.0) -> float:
        """Charge (positive action) or discharge (negative action) 1 hour.
        Respects SoC bounds via clip."""
        eta = np.sqrt(self.rte)
        # action > 0 = charge: SoC += action * eta * dt
        # action < 0 = discharge: SoC += action / eta * dt (still negative)
        if action >= 0:
            delta = action * eta * dt_h
        else:
            delta = action / eta * dt_h
        return float(np.clip(soc + delta, 0.0, self.energy_mwh))

    def _hourly_revenue(self, action: float, price: float, dt_h: float = 1.0) -> float:
        """EUR revenue this hour: price × discharge_MW × dt (minus charge cost)."""
        eta = np.sqrt(self.rte)
        if action >= 0:
            # Charging: pay price per MWh drawn from grid
            return -price * action * dt_h
        else:
            # Discharging: earn price per MWh delivered
            return -price * action * eta * dt_h * eta / eta  # simplified; just price × -action × eta
            # Actually: delivered_mwh = -action × eta × dt_h; revenue = price × delivered
            # With -action > 0 for discharge: revenue = price × (-action) × eta × dt

    def _solve_one_price_path(
        self,
        prices_24: np.ndarray,
        wear_per_mwh: float,
    ) -> np.ndarray:
        """Inner DP for a single 24-hour price vector. Returns V[soc, hour]."""
        n_soc = len(self.soc_grid)
        n_a = len(self.action_grid)
        V = np.zeros((n_soc, 24))
        eta = np.sqrt(self.rte)
        for _iter in range(120):
            V_new = np.zeros_like(V)
            for h in range(23, -1, -1):
                h_next = (h + 1) % 24
                for s in range(n_soc):
                    soc = self.soc_grid[s]
                    best = -np.inf
                    for a_idx in range(n_a):
                        action = self.action_grid[a_idx]
                        next_soc = soc + (action * eta if action >= 0 else action / eta)
                        if next_soc < -1e-9 or next_soc > self.energy_mwh + 1e-9:
                            continue
                        next_soc = float(np.clip(next_soc, 0.0, self.energy_mwh))
                        if action >= 0:
                            rev = -prices_24[h] * action
                        else:
                            rev = -prices_24[h] * action * eta
                        if wear_per_mwh > 0.0:
                            rev -= wear_per_mwh * abs(action)
                        next_val = np.interp(next_soc, self.soc_grid, V[:, h_next])
                        val = rev + next_val
                        if val > best:
                            best = val
                    V_new[s, h] = best if best > -np.inf else 0.0
            delta = np.abs(V_new - V).max()
            V = V_new
            if delta < 1e-3:
                break
        return V

    def _simulate_slice(
        self, soh: float, regime: int, soh_idx: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Value iteration on a single (SoH, regime) slice — returns V[soc, hour]
        and shadow_cost[hour].

        In **deterministic** mode (2D profiles): solves the cyclic DP on the
        regime-mean hourly price profile.

        In **stochastic** mode (3D profiles): solves the DP separately for
        each scenario (each a full historical 24-hour price vector drawn
        from regime ``regime``), then averages the resulting
        ``V[soc, hour]``. Shadow cost is derived from the averaged ``V``.
        Closes methodology simplification #1.

        When :attr:`physics_wear` is set, also subtracts
        ``physics_wear[soh_idx] × |throughput|`` from the reward — the
        arbitrage-vs-aging tradeoff enters the Bellman directly.
        """
        wear_per_mwh = (
            float(self.physics_wear[soh_idx]) if self.physics_wear is not None
            else 0.0
        )
        if self._stochastic:
            # Average V over bootstrap scenarios. V is linear in a scenario's
            # optimal daily dispatch plan — the per-scenario max is nonlinear
            # in prices, so averaging V (not prices) captures the realised
            # daily-spread variability in the shadow cost signal.
            V_sum = np.zeros((len(self.soc_grid), 24))
            n_k = self._n_scenarios
            for k in range(n_k):
                V_sum += self._solve_one_price_path(
                    self.profiles[regime, k], wear_per_mwh,
                )
            V = V_sum / n_k
        else:
            V = self._solve_one_price_path(
                self.profiles[regime], wear_per_mwh,
            )

        # Shadow cost per hour = ∂V/∂SoC (EUR per MWh of SoC held).
        # Central difference on SoC axis at middle bucket — this represents
        # "marginal value of saving one more MWh of SoC at this hour".
        # NOTE: This is PURE DP output — no exogenous scarcity scaling.
        # If the DP produces ~flat shadow cost across SoH levels, that is
        # the honest empirical finding: simplified state even with intraday
        # resolution doesn't deliver scarcity response. The companion
        # AgingAwareDepreciationPolicy does deliver it (by construction).
        mid = len(self.soc_grid) // 2
        dV_dSoC = np.zeros(24)
        for h in range(24):
            if mid == 0:
                dV_dSoC[h] = (V[mid + 1, h] - V[mid, h]) / (self.soc_grid[mid + 1] - self.soc_grid[mid])
            elif mid == len(self.soc_grid) - 1:
                dV_dSoC[h] = (V[mid, h] - V[mid - 1, h]) / (self.soc_grid[mid] - self.soc_grid[mid - 1])
            else:
                dV_dSoC[h] = (V[mid + 1, h] - V[mid - 1, h]) / (self.soc_grid[mid + 1] - self.soc_grid[mid - 1])
        # Wear cost is a cost — take absolute value. V increases with SoC, so
        # |∂V/∂SoC| measures the opportunity cost of one MWh of SoC wasted
        # to cycling at this hour.
        shadow_cost_per_hour = np.abs(dV_dSoC)
        return V, shadow_cost_per_hour

    def solve(self) -> IntradayADPResult:
        """Solve all (SoH, regime) slices → 3D shadow cost table.

        Optimisation: when ``physics_wear is None`` (the common case —
        L5 uses an additive physics layer at online lookup time, not
        inside the DP), the inner DP is SoH-independent. We solve once
        per regime and broadcast across the SoH axis. Big speed-up for
        stochastic mode (11× fewer DP solves).
        """
        n_soh = len(self.grids.soh_grid)
        n_r = self.regime.n_regimes
        n_soc = len(self.soc_grid)
        value_table = np.zeros((n_soh, n_r, n_soc, 24))
        shadow_cost = np.zeros((n_soh, n_r, 24))

        if self.physics_wear is None:
            # SoH-independent DP — solve once per regime.
            for r_idx in range(n_r):
                V, sc = self._simulate_slice(
                    soh=1.0, regime=r_idx, soh_idx=0,
                )
                for s_idx in range(n_soh):
                    value_table[s_idx, r_idx] = V
                    shadow_cost[s_idx, r_idx] = sc
        else:
            for s_idx, soh in enumerate(self.grids.soh_grid):
                for r_idx in range(n_r):
                    V, sc = self._simulate_slice(
                        soh=soh, regime=r_idx, soh_idx=s_idx,
                    )
                    value_table[s_idx, r_idx] = V
                    shadow_cost[s_idx, r_idx] = sc
        return IntradayADPResult(
            shadow_cost=shadow_cost,
            value_table=value_table,
            n_hours=24,
        )
