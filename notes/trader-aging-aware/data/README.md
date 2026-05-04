# Precomputed artifacts (v4.2 paper-grade)

All pkls here are loaded by `../app.py`. Filenames keep the `_v42` suffix
to flag the calibration vintage; if a future v4.3 lands, regenerate
alongside instead of overwriting.

| file | shape | written by | size |
| --- | --- | --- | ---: |
| `precomputed_v42_5methods_10y.pkl` | `{n_years, template_years, results: {M1..M5: LifecycleResult}, diagnostics, ablation_*}` | `precompute/policies.py` | ~6 MB |
| `precomputed_v42_recalibrated_10y.pkl` | same shape, with Note-1 trajectory per-year price scaling | `precompute/recalibrated_10y.py` | ~6 MB |
| `anchor_2026-01_v42.pkl` … `anchor_2026-04_v42.pkl` | `{year, month, results: {policy: {daily_logs}}}` | `precompute/anchor_month.py --month {1..4}` | ~400 KB each |

The `_v42` headline numbers (M1 → M5 = +36.5 % NPV uplift) live in
`precomputed_v42_5methods_10y.pkl`. The recalibrated pkl shows how that
uplift compresses under the Note-1 fleet-saturation trajectory.

Anchor-month pkls feed the "One month, five policies" trader-portal
section — fresh-cell (SoH=1.0) dispatch logs over a single recent
calendar month, so policy footprints can be compared without multi-year
age drift.
