# Note 4 policy comparison — 10y × 5 policies (2026-04-24)

## Setup

- 2 h LFP BESS, 1 MW / 2 MWh nominal
- Real 2023 and 2025 DE markets alternating
- Shared market frames across policies (same inputs, only wear cost differs)
- `max_cycles = 2.0` (LP constraint); cycling differentiation via wear cost only
- `max_afrr_participation = 0.40` (CH-calibrated from A1 rework)
- Warranty floor 0.80; discount 7 %/yr; degradation 2e-4 per FEC at SoH=1

## Headline

| policy                      | NPV (k€/MW) | EOL year | Σ FEC | Δ vs naive |
| --------------------------- | ----------: | -------: | ----: | ---------: |
| naive                       |         361 |       Y2 |   761 |       +0 % |
| depreciation_proxy          |         664 |       Y4 |   715 |      +84 % |
| aging_aware_depreciation    |         791 |       Y5 |   698 |     +119 % |
| adp_simplified (SoH+regime) |         396 |       Y2 |   759 |      +10 % |
| **adp_intraday (SoC+hour)** |    **1224** |   **Y8** | **591** |  **+239 %** |

## Reading the table

**Naive (+0%)**: cycles aggressively (761 FEC). Degrades in 2 years, zero revenue Y3+. Worst long-horizon policy.

**Flat depreciation proxy (+84%)**: the Kumtepeli "poor proxy". Cycles less (715 FEC), lives 4 years. Beats naive on lifetime NPV despite earning slightly less per year. But over-penalises uniformly.

**Aging-aware-depreciation formula (+119%)**: closed-form heuristic `λ = base × (1 − SoH)/(SoH − floor)`. Cycles 698 FEC, lives 5 years. Captures scarcity response but formula is not derived — it's a plausible-looking choice.

**ADP simplified (SoH, regime only) (+10%)**: principled backward-induction DP but in a state space too coarse to produce time-of-day signal. Behaves near-naïve. Documents the structural limitation of the simplified DP.

**ADP intraday (SoC, SoH, regime, hour) (+239%)**: full Holtorf-Shin formulation. Hour-varying shadow cost lets the LP cycle only on high-value hours. Cycles the LEAST (591 FEC = −22 % vs formula) yet earns the MOST (1224 k€/MW = +55 % vs formula). Battery lives 7 years at stable ~220 k€/yr revenue.

## Key insight

Hour-varying shadow cost is not a nicety — it's load-bearing. The LP needs to know "this shoulder hour isn't worth a cycle, save the SoC for the 18:00 peak". Closed-form SoH-scarcity alone doesn't tell it that; intraday DP does.

## Reversal of earlier conclusion

Prior commit `3715431` concluded that full Holtorf-Shin wasn't needed for Note 4's narrative because `AgingAwareDepreciationPolicy` carried the scarcity signal. That was wrong. Empirically:

- Formula policy beats naive by +119 %.
- Intraday DP beats naive by +239 % — and beats the formula by +55 %.

The formula gets 60 % of the way to optimal. For Note 4's narrative accuracy, **intraday DP is the primary aging-aware policy**, with the formula positioned as a closed-form approximation that still dramatically outperforms the Kumtepeli "poor proxy".

## Annual revenue profile

```
 year |     naive | depr |  aging-aware | adp_simpl | adp_intraday
  1   |     244   |  239 |     243      |    244    |     218
  2   |     152   |  250 |     247      |    191    |     224
  3   |       0   |  235 |     217      |      0    |     217
  4   |       0   |   40 |     211      |      0    |     224
  5   |       0   |    0 |      14      |      0    |     217
  6   |       0   |    0 |       0      |      0    |     223
  7   |       0   |    0 |       0      |      0    |     216
  8   |       0   |    0 |       0      |      0    |      66
  9   |       0   |    0 |       0      |      0    |       0
 10   |       0   |    0 |       0      |      0    |       0
```

Intraday ADP trades lower Y1 revenue (218 vs naive 244) for 5 extra years of full-revenue operation. Cumulative effect is decisive.

## SoH trajectory

Intraday ADP preserves SoH dramatically longer:
- Year 1 EOL SoH: 0.978 (naive 0.894)
- Year 5 EOL SoH: 0.874 (naive already at floor 0.800 since Y2)
- Year 7 EOL SoH: 0.812 — on the cusp of the floor

The economic value of preservation: EVERY non-aged year of operation earns ~220 k€/MW; losing 5 years = losing 1100 k€/MW nominal, which even with discounting is > 550 k€/MW present-value — a substantial fraction of the total NPV gap.
