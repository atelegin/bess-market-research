# BESS Market Research

Open-source models and Streamlit notes on the economics of battery energy storage in the German power market — revenue outlook, cycling-vs-revenue trade-offs, degradation drivers, and aging-aware dispatch.

## Published notes

| # | Note | App |
|---|---|---|
| 1 | German BESS revenue outlook 2026–2040 | https://de-bess-outlook.streamlit.app/ |
| 2 | Cycles & marginal value | https://de-bess-cycles.streamlit.app/ |
| 3 | What actually drives degradation | https://bess-degradation-drivers.streamlit.app/ |
| 4 | Cost of a cycle: is your optimiser aging-aware? | https://bess-cycle-cost.streamlit.app/ |

Each note in `notes/<slug>/` is a self-contained Streamlit app reading a precomputed `data/precomputed.pkl`. The `precompute*.py` scripts rebuild those artefacts from price data in `lib/data/cache/` plus the models in `lib/`.

## Project layout

```
lib/
  analysis/      Revenue, lifecycle NPV, degradation diagnostics, rolling-horizon helpers
  data/          Price loaders + cached CSVs (DA, ID, aFRR, FCR, clean-horizon indices)
  models/
    dispatch/    Dispatch policies — simple, detailed, stacked, arbitrage,
                 Collath benchmark, piecewise common, aFRR clearing-rate,
                 and the ADP (approximate dynamic programming) sub-package.
    degradation/ Simple + detailed (Wang/Naumann LFP) capacity-fade models
    ancillary.py
    price_regime.py
    projection.py
  shared/        Common Streamlit theme + bundled fonts
notes/
  de-bess-outlook/        Note 1
  cycles-marginal-value/  Note 2
  degradation-drivers/    Note 3
  trader-aging-aware/     Note 4
```

## Running a note

```bash
git clone https://github.com/atelegin/bess-market-research.git
cd bess-market-research
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run notes/de-bess-outlook/app.py    # or any other note
```

To rebuild a note's precomputed artefacts:

```bash
python -m notes.de_bess_outlook.precompute
```

## Tests

```bash
pytest lib/
```

## License

MIT — see `LICENSE`.

Author: Anton Telegin (PO BESS Data Platform, BayWa r.e. Data Services GmbH).
