"""
Pre-compute all results for Note 3 — "What Actually Drives Degradation".

Runs the detailed degradation model across:
  - baseline duty per preset
  - lever sweeps (DoD, SoC dwell, C-rate, temperature, chemistry)
  - lifetime NPV (20 yr, 8% discount, Note 2 DE DA+ID baseline)

Run once:  python notes/degradation-drivers/precompute.py
Results saved to notes/degradation-drivers/data/precomputed.pkl
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from lib.models.degradation.simple import PRESETS, CellPreset
from lib.models.degradation.detailed import (
    DutyCycle,
    cell_soh_detailed,
    lifecycle_value_detailed,
    project_capacity_detailed,
)

DATA_DIR = Path(__file__).parent / "data"

# Baseline duty anchor: Note 2 average operator (DE DA+ID)
BASELINE_DUTY = dict(
    fec_per_year=730.0,  # 2 c/d
    mean_dod=0.80,
    mean_soc=0.55,
    mean_crate=0.50,
    mean_temp_C=25.0,
)

# Year-1 revenue anchor: Note 2 DE DA+ID 2026 frontier (kEUR/MW/yr)
YEAR1_REVENUE_KEUR = 120.0
LIFETIME_YEARS = 20
DISCOUNT_RATE = 0.08
MARKET_DECLINE = -0.02  # ~2%/yr revenue decline from Note 1


def baseline_duty(**overrides) -> DutyCycle:
    kwargs = {**BASELINE_DUTY, **overrides}
    return DutyCycle.from_mean(**kwargs)


def npv_kEUR(duty: DutyCycle, preset: CellPreset) -> float:
    df = lifecycle_value_detailed(
        year1_revenue=YEAR1_REVENUE_KEUR,
        duty=duty,
        preset=preset,
        years=LIFETIME_YEARS,
        discount_rate=DISCOUNT_RATE,
        annual_market_decline=MARKET_DECLINE,
        n_mc=200,
    )
    return float(df["discounted_revenue_eur_per_mw"].sum())


def soh_trajectory(duty: DutyCycle, preset: CellPreset, years: int = LIFETIME_YEARS) -> pd.DataFrame:
    xs = np.arange(0.0, years + 0.01, 0.25)
    rows = []
    rng = np.random.default_rng(1)
    for t in xs:
        if t <= 0:
            rows.append({"year": 0.0, "p10": 1.0, "p50": 1.0, "p90": 1.0})
            continue
        p10, p50, p90 = project_capacity_detailed(
            duty=duty, years=t, preset=preset, n_mc=500, return_kind="distribution", rng=rng
        )
        rows.append({"year": float(t), "p10": p10, "p50": p50, "p90": p90})
    return pd.DataFrame(rows)


def lever_sweep() -> pd.DataFrame:
    """Chart A — symmetric tornado: each lever moved down and up from baseline."""
    preset = PRESETS["baseline_fleet"]
    npv_base = npv_kEUR(baseline_duty(), preset)

    # (lever label, group, low-value override, low-step label, high-value override, high-step label)
    #
    # Imbalance lever: weakest-cell proxy. String imbalance shifts the weakest
    # cell to deeper effective DoD and higher SoC dwell than the string mean.
    # Spread converts directly to DoD inflation and SoC shift. Range anchored
    # to field literature (Schimpe 2018 / Reniers 2019): pack-vs-cell fade gap
    # ~10–20% — mapped to 2 pp (active monitoring) → 10 pp (no monitoring).
    specs = [
        ("Temperature",  "given",    {"mean_temp_C": 15.0},                       "−10 °C (15 °C)",   {"mean_temp_C": 30.0},                       "+5 °C (30 °C)"),
        ("Cycles / day", "dispatch", {"fec_per_year": 365.0},                     "−1 c/d (FCR)",     {"fec_per_year": 900.0},                     "+0.5 c/d (arb)"),
        ("C-rate",       "dispatch", {"mean_crate": 0.25},                        "−0.25 (0.25C)",    {"mean_crate": 1.00},                        "+0.50 (1.00C)"),
        ("DoD",          "dispatch", {"mean_dod": 0.60},                          "−20 pp (60%)",     {"mean_dod": 0.95},                          "+15 pp (95%)"),
        ("Rest SoC",     "dispatch", {"mean_soc": 0.40},                          "−15 pp (40%)",     {"mean_soc": 0.75},                          "+20 pp (75%)"),
        ("Imbalance",    "given",    {"mean_dod": 0.82, "mean_soc": 0.58},        "2 pp (monitored)", {"mean_dod": 0.90, "mean_soc": 0.65},        "10 pp (unmanaged)"),
    ]

    rows = []
    for lever, group, lo_over, lo_label, hi_over, hi_label in specs:
        lo_delta = npv_kEUR(baseline_duty(**lo_over), preset) - npv_base
        hi_delta = npv_kEUR(baseline_duty(**hi_over), preset) - npv_base
        rows.append({
            "lever": lever,
            "group": group,
            "low_delta": lo_delta,
            "low_label": lo_label,
            "high_delta": hi_delta,
            "high_label": hi_label,
            "span": abs(lo_delta) + abs(hi_delta),
        })

    df = pd.DataFrame(rows).sort_values("span").reset_index(drop=True)
    return df


def soh_panels() -> dict:
    """Supporting 4-panel: baseline / deep-DoD / high-SoC / C-rate discipline."""
    preset = PRESETS["baseline_fleet"]
    scenarios = {
        "baseline": baseline_duty(),
        "deep_dod": baseline_duty(mean_dod=0.95),
        "high_soc": baseline_duty(mean_soc=0.85),
        "c_rate_discipline": baseline_duty(mean_crate=0.25),
    }
    return {k: soh_trajectory(d, preset) for k, d in scenarios.items()}


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    import json
    response_curves_path = DATA_DIR.parent / "charts" / "prototype_d3_data.json"
    with open(response_curves_path) as f:
        response_curves = json.load(f)

    payload = {
        "baseline_duty": BASELINE_DUTY,
        "lifetime_years": LIFETIME_YEARS,
        "year1_revenue_keur": YEAR1_REVENUE_KEUR,
        "discount_rate": DISCOUNT_RATE,
        "market_decline": MARKET_DECLINE,
        "baseline_npv_keur": npv_kEUR(baseline_duty(), PRESETS["baseline_fleet"]),
        "lever_sweep": lever_sweep(),
        "soh_panels": soh_panels(),
        "response_curves": response_curves,
    }

    path = DATA_DIR / "precomputed.pkl"
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    size_mb = path.stat().st_size / 1e6
    print(f"Saved to {path} ({size_mb:.3f} MB)")
    print()
    print("Lever sweep preview:")
    print(payload["lever_sweep"].to_string(index=False))


if __name__ == "__main__":
    main()
