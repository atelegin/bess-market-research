# BESS Market Research

Energy storage market notes and analysis tools.

## Structure

```
lib/              Shared modelling code (data fetchers, dispatch, projection)
notes/            Market notes — each is a self-contained Streamlit app
  de-bess-best-days/    Revenue concentration analysis (DE)
  de-bess-outlook/      German BESS revenue outlook 2026-2040
  _template/            Starter for new notes
notebooks/        Exploratory work (Jupyter, scratch scripts)
```

## Running a note

```bash
cd bess-market-research
streamlit run notes/de-bess-outlook/app.py
```

## Adding a new note

```bash
cp -r notes/_template notes/XX-your-slug
# Edit app.py and README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add API credentials
```
