# Precompute scripts (v4.2 paper-grade)

These three scripts regenerate every artifact loaded by `app.py`. All run
from the repo root with `.venv/bin/python`.

| script | output (in `../data/`) | runtime |
| --- | --- | ---: |
| `policies.py` | `precomputed_v42_5methods_10y.pkl` (canonical M1–M5 headline, 10 y × single template year 2025, flat market) | ~5 min |
| `recalibrated_10y.py` | `precomputed_v42_recalibrated_10y.pkl` (same M-stack, with Note-1 mid-case per-year revenue trajectory) | ~5 min |
| `anchor_month.py --month {1,2,3,4}` | `anchor_2026-{MM}_v42.pkl` (one calendar month, all 5 policies, fresh-cell dispatch logs) | ~1 min/month |

Calibration knobs (identical across all three): `bid_win_rate=1.0`,
`capture_factor=0.10`, `r_min=0.20`, `bid_hurdle=20`, `kernel_scale=0.66`
(EVE LF280K), FCR phantom revenue €36 k/MW/yr, two-stage dispatch with
Stage-2 pass-2 physics-from-duty refinement.

`policies.build_policies()` is the shared M1–M5 factory. `anchor_month.py`
defines its own `build_m_stack()` (lighter, no rainflow / diagnostics);
`recalibrated_10y.py` re-imports `build_m_stack` from `anchor_month.py` by
file path so the two stay byte-identical.

To run a full refresh of every artifact app.py needs:

```bash
.venv/bin/python notes/trader-aging-aware/precompute/policies.py
.venv/bin/python notes/trader-aging-aware/precompute/recalibrated_10y.py
for m in 1 2 3 4; do
    .venv/bin/python notes/trader-aging-aware/precompute/anchor_month.py --month $m
done
```

Total wall-clock ≈ 15 min on an M-series Mac.
