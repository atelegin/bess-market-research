"""
Detailed BESS dispatch — LP optimisation with configurable strategies.

Supports conservative/aggressive strategies, SoC limits, minimum spread
thresholds, intraday overlay, and per-interval charge/discharge tracking.
For quick revenue estimates use the simpler dispatch.py instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd
from scipy.optimize import linprog


@dataclass(frozen=True)
class DispatchStrategy:
    name: str
    label: str
    max_cycles: float
    soc_min_frac: float
    soc_max_frac: float
    min_spread_eur_mwh: float
    max_p_rate: float | None = None
    max_avg_p_rate_6h: float | None = None
    # Wirkleistungsgradient in %/min of installed power, per BDEW TAB-MS v1
    # §10.2.4.1 (default 22 %/min, adjustable range 6–22 %/min).
    max_ramp_pct_per_min: float | None = None

    @property
    def usable_energy_fraction(self) -> float:
        return self.soc_max_frac - self.soc_min_frac


CONSERVATIVE_STRATEGY = DispatchStrategy(
    name="conservative",
    label="Conservative",
    max_cycles=1.0,
    soc_min_frac=0.20,
    soc_max_frac=0.80,
    min_spread_eur_mwh=20.0,
)

AGGRESSIVE_STRATEGY = DispatchStrategy(
    name="aggressive",
    label="Aggressive",
    max_cycles=2.0,
    soc_min_frac=0.05,
    soc_max_frac=0.95,
    min_spread_eur_mwh=5.0,
)


def infer_timestep_hours(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 1.0
    deltas = index.to_series().diff().dropna().dt.total_seconds() / 3600
    return float(deltas.mode().iloc[0])


def _count_active_segments(values: np.ndarray, tolerance: float = 1e-6) -> int:
    active = values > tolerance
    if not active.any():
        return 0
    return int(active[0]) + int(np.count_nonzero(active[1:] & ~active[:-1]))


def optimize_day(
    prices: np.ndarray,
    energy_mwh: float,
    rte: float,
    soc_min_frac: float,
    soc_max_frac: float,
    max_cycles: float,
    power_mw: float = 1.0,
    dt_hours: float = 1.0,
    min_spread_eur_mwh: float = 0.0,
    max_p_rate: float | None = None,
    max_avg_p_rate_6h: float | None = None,
    max_ramp_pct_per_min: float | None = None,
) -> dict[str, object]:
    periods = len(prices)
    eta = sqrt(rte)
    soc_min = soc_min_frac * energy_mwh
    soc_max = soc_max_frac * energy_mwh
    soc_init = (soc_min + soc_max) / 2

    # Apply instantaneous P-rate limit
    effective_power = power_mw
    if max_p_rate is not None:
        effective_power = min(power_mw, max_p_rate * energy_mwh)
    usable_energy_mwh = max(energy_mwh * (soc_max_frac - soc_min_frac), 1e-9)
    throughput_penalty = min_spread_eur_mwh / 2

    c = np.concatenate(
        [
            dt_hours * (prices / eta + throughput_penalty),
            -dt_hours * (prices * eta - throughput_penalty),
        ]
    )

    a_ub: list[np.ndarray] = []
    b_ub: list[float] = []

    for t in range(periods):
        row_upper = np.zeros(2 * periods)
        row_upper[: t + 1] = eta * dt_hours
        row_upper[periods : periods + t + 1] = -dt_hours / eta
        a_ub.append(row_upper)
        b_ub.append(soc_max - soc_init)

        row_lower = np.zeros(2 * periods)
        row_lower[: t + 1] = -eta * dt_hours
        row_lower[periods : periods + t + 1] = dt_hours / eta
        a_ub.append(row_lower)
        b_ub.append(soc_init - soc_min)

    cycle_limit_row = np.zeros(2 * periods)
    cycle_limit_row[periods:] = dt_hours
    a_ub.append(cycle_limit_row)
    b_ub.append(max_cycles * usable_energy_mwh)

    final_soc_row = np.zeros(2 * periods)
    final_soc_row[:periods] = eta * dt_hours
    final_soc_row[periods:] = -dt_hours / eta
    tolerance_mwh = 0.1 * energy_mwh
    a_ub.append(final_soc_row)
    b_ub.append(tolerance_mwh)
    a_ub.append(-final_soc_row)
    b_ub.append(tolerance_mwh)

    bounds = [(0.0, effective_power)] * (2 * periods)

    # Rolling 6-hour average P-rate constraint
    if max_avg_p_rate_6h is not None:
        window_slots = int(np.ceil(6.0 / dt_hours))
        max_energy_in_window = max_avg_p_rate_6h * energy_mwh * 6.0
        for t in range(max(1, periods - window_slots + 1)):
            end = min(t + window_slots, periods)
            row = np.zeros(2 * periods)
            row[t:end] = dt_hours          # charge power * dt = energy
            row[periods + t:periods + end] = dt_hours  # discharge power * dt = energy
            a_ub.append(row)
            b_ub.append(max_energy_in_window)

    # Leistungsgradient — bound |P_{t+1} − P_t| where P_t = discharge_t − charge_t.
    # BDEW TAB-MS v1 default 22 %/min, adjustable 6–22 %/min. aFRR prequalification
    # derating (5-min rule) lives in lib.models.ancillary.afrr_prequal_fraction.
    if max_ramp_pct_per_min is not None and periods >= 2:
        dt_minutes = dt_hours * 60.0
        delta_max = (max_ramp_pct_per_min / 100.0) * effective_power * dt_minutes
        for t in range(periods - 1):
            # (P_{t+1} − P_t) ≤ delta_max
            row_up = np.zeros(2 * periods)
            row_up[t] = 1.0                 # +charge_t
            row_up[t + 1] = -1.0            # −charge_{t+1}
            row_up[periods + t] = -1.0      # −discharge_t
            row_up[periods + t + 1] = 1.0   # +discharge_{t+1}
            a_ub.append(row_up)
            b_ub.append(delta_max)
            # (P_t − P_{t+1}) ≤ delta_max
            a_ub.append(-row_up)
            b_ub.append(delta_max)

    result = linprog(
        c=c,
        A_ub=np.asarray(a_ub),
        b_ub=np.asarray(b_ub),
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        zeros = np.zeros(periods)
        return {
            "revenue": 0.0,
            "charge": zeros,
            "discharge": zeros,
            "soc": np.full(periods, soc_init),
            "cycles": 0.0,
            "full_equivalent_cycles": 0.0,
            "avg_dod": 0.0,
        }

    decision = result.x
    charge = decision[:periods]
    discharge = decision[periods:]
    soc = soc_init + np.cumsum(charge * eta * dt_hours - discharge / eta * dt_hours)

    revenue = float(
        np.sum(discharge * dt_hours * (prices * eta - throughput_penalty))
        - np.sum(charge * dt_hours * (prices / eta + throughput_penalty))
    )

    discharged_energy_mwh = float(np.sum(discharge) * dt_hours)
    charge_segments = _count_active_segments(charge)
    discharge_segments = _count_active_segments(discharge)
    cycle_events = float(max(charge_segments, discharge_segments))
    avg_dod = min(discharged_energy_mwh / (cycle_events * energy_mwh), soc_max_frac - soc_min_frac) if cycle_events else 0.0
    full_equivalent_cycles = discharged_energy_mwh / energy_mwh

    return {
        "revenue": revenue,
        "charge": charge,
        "discharge": discharge,
        "soc": soc,
        "cycles": cycle_events,
        "full_equivalent_cycles": full_equivalent_cycles,
        "avg_dod": avg_dod,
    }


def _expand_step_profile(
    source_index: pd.DatetimeIndex,
    values: np.ndarray,
    target_index: pd.DatetimeIndex,
) -> np.ndarray:
    series = pd.Series(values, index=source_index)
    expanded = series.reindex(target_index, method="ffill")
    return expanded.to_numpy(dtype=float)


def run_dispatch_with_intraday_overlay_for_period(
    day_ahead_price_frame: pd.DataFrame,
    intraday_price_frame: pd.DataFrame,
    strategy: DispatchStrategy,
    energy_mwh: float,
    rte: float,
    power_mw: float = 1.0,
) -> pd.DataFrame:
    records: list[dict[str, float | pd.Timestamp]] = []
    day_ahead_prices = day_ahead_price_frame.sort_index()
    intraday_prices = intraday_price_frame.sort_index()
    common_days = sorted(
        set(day_ahead_prices.index.normalize()).intersection(set(intraday_prices.index.normalize()))
    )
    eta = sqrt(rte)

    for day in common_days:
        da_group = day_ahead_prices[day_ahead_prices.index.normalize() == day]
        id_group = intraday_prices[intraday_prices.index.normalize() == day]
        if da_group.empty or id_group.empty:
            continue

        da_dt_hours = infer_timestep_hours(da_group.index)
        id_dt_hours = infer_timestep_hours(id_group.index)
        da_result = optimize_day(
            prices=da_group["price_eur_mwh"].to_numpy(dtype=float),
            energy_mwh=energy_mwh,
            rte=rte,
            soc_min_frac=strategy.soc_min_frac,
            soc_max_frac=strategy.soc_max_frac,
            max_cycles=strategy.max_cycles,
            power_mw=power_mw,
            dt_hours=da_dt_hours,
            min_spread_eur_mwh=strategy.min_spread_eur_mwh,
            max_p_rate=strategy.max_p_rate,
            max_avg_p_rate_6h=strategy.max_avg_p_rate_6h,
            max_ramp_pct_per_min=strategy.max_ramp_pct_per_min,
        )
        id_result = optimize_day(
            prices=id_group["price_eur_mwh"].to_numpy(dtype=float),
            energy_mwh=energy_mwh,
            rte=rte,
            soc_min_frac=strategy.soc_min_frac,
            soc_max_frac=strategy.soc_max_frac,
            max_cycles=strategy.max_cycles,
            power_mw=power_mw,
            dt_hours=id_dt_hours,
            min_spread_eur_mwh=strategy.min_spread_eur_mwh,
            max_p_rate=strategy.max_p_rate,
            max_avg_p_rate_6h=strategy.max_avg_p_rate_6h,
            max_ramp_pct_per_min=strategy.max_ramp_pct_per_min,
        )

        da_charge_on_id_grid = _expand_step_profile(da_group.index, da_result["charge"], id_group.index)
        da_discharge_on_id_grid = _expand_step_profile(da_group.index, da_result["discharge"], id_group.index)
        da_price_on_id_grid = _expand_step_profile(
            da_group.index,
            da_group["price_eur_mwh"].to_numpy(dtype=float),
            id_group.index,
        )
        da_grid_flow = da_discharge_on_id_grid * eta - da_charge_on_id_grid / eta
        id_prices = id_group["price_eur_mwh"].to_numpy(dtype=float)
        settlement_adjustment = float(np.sum(da_grid_flow * (da_price_on_id_grid - id_prices) * id_dt_hours))
        total_revenue = float(id_result["revenue"] + settlement_adjustment)
        day_ahead_revenue = float(da_result["revenue"])
        overlay_revenue = float(total_revenue - day_ahead_revenue)

        records.append(
            {
                "date": pd.Timestamp(day).tz_localize(None),
                "revenue_eur_per_mw": total_revenue,
                "revenue_day_ahead_eur_per_mw": day_ahead_revenue,
                "revenue_intraday_eur_per_mw": overlay_revenue,
                "revenue_intraday_execution_eur_per_mw": float(id_result["revenue"]),
                "cycles": float(id_result["cycles"]),
                "full_equivalent_cycles": float(id_result["full_equivalent_cycles"]),
                "avg_dod": float(id_result["avg_dod"]),
                "price_min_eur_mwh": float(id_group["price_eur_mwh"].min()),
                "price_max_eur_mwh": float(id_group["price_eur_mwh"].max()),
                "price_spread_eur_mwh": float(id_group["price_eur_mwh"].max() - id_group["price_eur_mwh"].min()),
            }
        )

    return pd.DataFrame(records).set_index("date").sort_index()


def run_dispatch_for_period(
    price_frame: pd.DataFrame,
    strategy: DispatchStrategy,
    energy_mwh: float,
    rte: float,
    power_mw: float = 1.0,
) -> pd.DataFrame:
    records: list[dict[str, float | pd.Timestamp]] = []
    prices = price_frame.sort_index()
    for day, group in prices.groupby(prices.index.normalize()):
        dt_hours = infer_timestep_hours(group.index)
        result = optimize_day(
            prices=group["price_eur_mwh"].to_numpy(dtype=float),
            energy_mwh=energy_mwh,
            rte=rte,
            soc_min_frac=strategy.soc_min_frac,
            soc_max_frac=strategy.soc_max_frac,
            max_cycles=strategy.max_cycles,
            power_mw=power_mw,
            dt_hours=dt_hours,
            min_spread_eur_mwh=strategy.min_spread_eur_mwh,
            max_p_rate=strategy.max_p_rate,
            max_avg_p_rate_6h=strategy.max_avg_p_rate_6h,
            max_ramp_pct_per_min=strategy.max_ramp_pct_per_min,
        )
        records.append(
            {
                "date": pd.Timestamp(day).tz_localize(None),
                "revenue_eur_per_mw": float(result["revenue"]),
                "cycles": float(result["cycles"]),
                "full_equivalent_cycles": float(result["full_equivalent_cycles"]),
                "avg_dod": float(result["avg_dod"]),
                "price_min_eur_mwh": float(group["price_eur_mwh"].min()),
                "price_max_eur_mwh": float(group["price_eur_mwh"].max()),
                "price_spread_eur_mwh": float(group["price_eur_mwh"].max() - group["price_eur_mwh"].min()),
            }
        )
    return pd.DataFrame(records).set_index("date").sort_index()
