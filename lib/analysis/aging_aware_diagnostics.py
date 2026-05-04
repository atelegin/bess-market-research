"""
Owner-facing aging-aware dispatch diagnostics (Note 4 C).

Five signals an owner can compute from a month or year of BESS dispatch
logs to detect whether the trader is genuinely pricing cycles or just
running a throughput-penalty dressed up as aging-aware optimisation.
Each signal is a pure function over per-day dispatch traces + market
context; the LP-native :class:`lib.models.dispatch.stacked.StackedDispatchResult`
plus the matching :class:`lib.analysis.stacked_day_assembler.StackedDayInputs`
collapse cleanly into the :class:`DayDiagnosticData` record this module
consumes.

The five signals (per ROADMAP ``trader-aging-aware`` §Diagnostic):

1. **DoD vs spread decile** — naive optimiser cycles to similar depth
   regardless of daily spread; aging-aware skips marginal DoD on weak
   spread days.
2. **SoC-hours histogram** — aging-aware resolves ambiguous moments
   toward low-stress SoC (40–60 %); naive parks at whatever SoC the
   last trade left.
3. **Revenue per cycle by spread quartile** — aging-aware earns
   strictly higher EUR/FEC on low-spread days because marginal
   unprofitable cycles are skipped.
4. **C-rate histogram vs P-rate envelope** — naive pushes to P-max on
   every cleared trade; aging-aware holds back on marginal cycles
   because high C-rate drives fade.
5. **Ancillary-vs-arbitrage mix** — flags the revenue regime (Kawollek
   point): EUR/MWh is the wrong denominator on aFRR-heavy assets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as Date
from typing import Iterable, Optional

import numpy as np
import pandas as pd


@dataclass
class DayDiagnosticData:
    """Per-day payload the diagnostic signals operate on.

    Built from a daily LP solve plus the market inputs that fed it.
    Keep raw (no normalisations) — signals may rescale per their needs.
    """
    date: Date
    # Net power flow per 15-min interval (discharge positive, charge negative), MW
    power_mw_signed: np.ndarray   # shape (96,)
    # SoC trace (MWh absolute) per 15-min interval
    soc_mwh: np.ndarray           # shape (96,)
    # Prices per 15-min interval (EUR/MWh)
    prices_da: np.ndarray         # shape (96,)
    # Asset specs
    power_mw: float
    energy_mwh: float
    # Per-day aggregate stats
    daily_revenue_eur: float
    full_equivalent_cycles: float
    # Revenue breakdown (for signal 5 — ancillary vs arbitrage mix)
    revenue_breakdown: dict[str, float] = field(default_factory=dict)


def day_diagnostic_from_stacked(
    day_result, inputs, target_date: Date, power_mw: float, energy_mwh: float,
) -> DayDiagnosticData:
    """Collapse (StackedDispatchResult, StackedDayInputs) → DayDiagnosticData."""
    net_power = (
        day_result.discharge_da + day_result.discharge_id + day_result.a_pos
        - day_result.charge_da - day_result.charge_id - day_result.a_neg
    )
    return DayDiagnosticData(
        date=target_date,
        power_mw_signed=np.asarray(net_power, dtype=float),
        soc_mwh=np.asarray(day_result.soc, dtype=float),
        prices_da=np.asarray(inputs.prices_da, dtype=float),
        power_mw=float(power_mw),
        energy_mwh=float(energy_mwh),
        daily_revenue_eur=float(day_result.revenue_total),
        full_equivalent_cycles=float(day_result.full_equivalent_cycles),
        revenue_breakdown={
            "da": float(day_result.revenue_da),
            "id": float(day_result.revenue_id),
            "afrr_cap_pos": float(day_result.revenue_afrr_cap_pos),
            "afrr_cap_neg": float(day_result.revenue_afrr_cap_neg),
            "afrr_energy_pos": float(day_result.revenue_afrr_energy_pos),
            "afrr_energy_neg": float(day_result.revenue_afrr_energy_neg),
        },
    )


def _daily_spread(prices: np.ndarray, low_pct: float = 5.0, high_pct: float = 95.0) -> float:
    return float(np.percentile(prices, high_pct) - np.percentile(prices, low_pct))


def _daily_dod(soc_mwh: np.ndarray, energy_mwh: float, fec: float) -> float:
    """Mean cycle depth (MWh per FEC, normalised to energy capacity).

    Approximates "average DoD per cycle" as total throughput divided by
    FEC count × capacity. Returns fraction of nominal capacity (0..1).
    """
    if fec <= 1e-9:
        return 0.0
    # Total throughput ≈ 2 × energy_mwh × FEC (both directions). Per-cycle
    # depth is throughput / (2 × FEC) / energy_mwh.
    soc_span = soc_mwh.max() - soc_mwh.min()
    # Fraction of nominal — capped at 1.0 for numerical safety.
    return min(1.0, soc_span / max(energy_mwh, 1e-9))


def dod_by_spread_decile(days: list[DayDiagnosticData]) -> pd.DataFrame:
    """Signal 1. Mean DoD per day grouped by spread decile.

    Returns DataFrame with columns ``spread_decile`` (0..9) and
    ``mean_dod``. Aging-aware dispatch shows DoD rising with spread
    decile (skips deep cycles on weak days); naive shows flat DoD.
    """
    if not days:
        return pd.DataFrame(columns=["spread_decile", "mean_dod", "n_days"])
    records = []
    for d in days:
        records.append({
            "date": d.date,
            "spread": _daily_spread(d.prices_da),
            "dod": _daily_dod(d.soc_mwh, d.energy_mwh, d.full_equivalent_cycles),
        })
    df = pd.DataFrame(records)
    df["spread_decile"] = pd.qcut(df["spread"], q=10, labels=False, duplicates="drop")
    summary = (
        df.groupby("spread_decile")
        .agg(mean_dod=("dod", "mean"), n_days=("date", "count"))
        .reset_index()
    )
    return summary


def soc_hours_histogram(
    days: list[DayDiagnosticData], n_bins: int = 10,
) -> pd.DataFrame:
    """Signal 2. SoC-hours distribution pooled across all days.

    For each day, SoC is normalised by current usable energy, classified
    into ``n_bins`` buckets (default 10 = 10 % each), and hours per
    bucket counted. Aging-aware → mid-band concentration; naive →
    extremes.
    """
    if not days:
        return pd.DataFrame(columns=["soc_bin_pct", "hours"])
    # Bin edges 0..1 in n_bins equal steps
    edges = np.linspace(0, 1, n_bins + 1)
    counts = np.zeros(n_bins)
    dt_h = 0.25  # 15-min intervals
    for d in days:
        frac = d.soc_mwh / max(d.energy_mwh, 1e-9)
        bin_idx = np.clip(np.digitize(frac, edges) - 1, 0, n_bins - 1)
        for b in range(n_bins):
            counts[b] += float((bin_idx == b).sum()) * dt_h
    labels = [f"{int(edges[i] * 100)}–{int(edges[i + 1] * 100)} %" for i in range(n_bins)]
    return pd.DataFrame({"soc_bin_pct": labels, "hours": counts})


def revenue_per_cycle_by_quartile(days: list[DayDiagnosticData]) -> pd.DataFrame:
    """Signal 3. EUR per FEC grouped by daily-spread quartile.

    Aging-aware dispatch skips marginal low-spread cycles → higher EUR/FEC
    on bottom-quartile days. Naive cycles equally everywhere → roughly
    flat EUR/FEC by spread quartile (or even lower on low-spread days
    because same FEC earns less).
    """
    if not days:
        return pd.DataFrame(columns=["spread_quartile", "eur_per_fec", "n_days"])
    records = []
    for d in days:
        if d.full_equivalent_cycles <= 1e-9:
            continue
        records.append({
            "spread": _daily_spread(d.prices_da),
            "eur_per_fec": d.daily_revenue_eur / d.full_equivalent_cycles,
        })
    if not records:
        return pd.DataFrame(columns=["spread_quartile", "eur_per_fec", "n_days"])
    df = pd.DataFrame(records)
    df["spread_quartile"] = pd.qcut(df["spread"], q=4, labels=False, duplicates="drop")
    return (
        df.groupby("spread_quartile")
        .agg(eur_per_fec=("eur_per_fec", "mean"), n_days=("eur_per_fec", "count"))
        .reset_index()
    )


def crate_histogram(
    days: list[DayDiagnosticData], n_bins: int = 5,
) -> pd.DataFrame:
    """Signal 4. C-rate distribution pooled across all days.

    Per 15-min interval, C-rate = |net power| / energy. Bucketed into
    ``n_bins`` bands spanning 0 to max observed (capped at 1.0 C for
    BESS). Aging-aware: mass at lower C-rates (preserves cells);
    naive: mass piled at P-max whenever cleared.
    """
    if not days:
        return pd.DataFrame(columns=["crate_bin", "hours"])
    edges = np.linspace(0, 1.0, n_bins + 1)
    counts = np.zeros(n_bins)
    dt_h = 0.25
    for d in days:
        crate = np.abs(d.power_mw_signed) / max(d.energy_mwh, 1e-9)
        crate = np.clip(crate, 0, 1.0)
        bin_idx = np.clip(np.digitize(crate, edges) - 1, 0, n_bins - 1)
        for b in range(n_bins):
            counts[b] += float((bin_idx == b).sum()) * dt_h
    labels = [f"{edges[i]:.2f}–{edges[i + 1]:.2f} C" for i in range(n_bins)]
    return pd.DataFrame({"crate_bin": labels, "hours": counts})


def ancillary_vs_arbitrage_mix(days: list[DayDiagnosticData]) -> dict[str, float]:
    """Signal 5. Pooled revenue share by stream (Kawollek regime flag).

    When AS share > ~50 % on a lifetime basis, EUR/MWh is the wrong
    denominator for dispatch quality comparisons (Kawollek); use
    EUR/lost-capacity or €/MW-reserved instead.

    Returns dict: ``{total, da, id, afrr_cap, afrr_energy, as_share}``.
    """
    totals = {"da": 0.0, "id": 0.0, "afrr_cap": 0.0, "afrr_energy": 0.0}
    for d in days:
        rb = d.revenue_breakdown
        totals["da"] += rb.get("da", 0.0)
        totals["id"] += rb.get("id", 0.0)
        totals["afrr_cap"] += rb.get("afrr_cap_pos", 0.0) + rb.get("afrr_cap_neg", 0.0)
        totals["afrr_energy"] += rb.get("afrr_energy_pos", 0.0) + rb.get("afrr_energy_neg", 0.0)
    total = sum(totals.values())
    as_total = totals["afrr_cap"] + totals["afrr_energy"]
    as_share = as_total / total if total else 0.0
    return {
        "total": total,
        "da": totals["da"],
        "id": totals["id"],
        "afrr_cap": totals["afrr_cap"],
        "afrr_energy": totals["afrr_energy"],
        "as_share": as_share,
    }


@dataclass
class DiagnosticReport:
    """Bundle of all five signals for a policy's dispatch record."""
    policy_name: str
    n_days: int
    dod_vs_spread: pd.DataFrame
    soc_hours: pd.DataFrame
    revenue_per_cycle_quartile: pd.DataFrame
    crate_hist: pd.DataFrame
    ancillary_mix: dict[str, float]


def compute_all_signals(
    policy_name: str, days: list[DayDiagnosticData],
) -> DiagnosticReport:
    """Convenience: run all five signals on one policy's dispatch log."""
    return DiagnosticReport(
        policy_name=policy_name,
        n_days=len(days),
        dod_vs_spread=dod_by_spread_decile(days),
        soc_hours=soc_hours_histogram(days),
        revenue_per_cycle_quartile=revenue_per_cycle_by_quartile(days),
        crate_hist=crate_histogram(days),
        ancillary_mix=ancillary_vs_arbitrage_mix(days),
    )
