"""
Empirical aFRR cap auction clearing-rate analysis from regelleistung.net.

API path: ``/apps/crds/api/v2/tenders/files/``.
File pattern: ``RESULT_LIST_ANONYM_CAPACITY_MARKET_aFRR_DE_{from}_{to}.xlsx.zip``.

Each row = one anonymous bid with: bid price (EUR/MW/h), offered MW,
allocated MW, block (PRODUCT like POS_00_04 / NEG_16_20), country.

For each (year, month, direction, block_start_h) compute:
  * total offered MW
  * total allocated MW
  * **clearing rate = allocated / offered** (publicly-derivable proxy
    for what we call ``bid_win_rate`` in the dispatcher)
  * mean / max bid price weighted by allocated MW

Output: parquet with per-(year, month, dir, block) statistics + annual
aggregates suitable for direct citation in paper.

Public source: regelleistung.net (TSO open data, no auth, no paywall).

Usage::

    .venv/bin/python scripts/regelleistung_clearing_rate.py 2024
    .venv/bin/python scripts/regelleistung_clearing_rate.py 2024 2025
"""
from __future__ import annotations

import io
import logging
import sys
import zipfile
from calendar import monthrange
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import requests


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

API_BASE = "https://www.regelleistung.net/apps/crds/api/v2/tenders/files"
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "external" / "regelleistung_clearing"


def _fetch_month(year: int, month: int) -> pd.DataFrame | None:
    """Download + unzip + parse one month of DE aFRR cap bids."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = DATA_DIR / f"afrr_cap_bids_{year}_{month:02d}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    last_day = monthrange(year, month)[1]
    fname = (
        f"RESULT_LIST_ANONYM_CAPACITY_MARKET_aFRR_DE_"
        f"{year}-{month:02d}-01_{year}-{month:02d}-{last_day:02d}.xlsx.zip"
    )
    url = f"{API_BASE}/{fname}"
    logger.info(f"  Fetching {year}-{month:02d}…")
    try:
        r = requests.get(url, timeout=300)
    except Exception as e:
        logger.warning(f"  {year}-{month:02d}: fetch error {e}")
        return None
    if r.status_code != 200 or len(r.content) < 1000:
        logger.warning(f"  {year}-{month:02d}: HTTP {r.status_code}, "
                       f"size={len(r.content)}")
        return None
    try:
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            inner = zf.namelist()[0]
            with zf.open(inner) as fh:
                df = pd.read_excel(fh)
    except Exception as e:
        logger.warning(f"  {year}-{month:02d}: parse error {e}")
        return None

    df.columns = [c.strip() for c in df.columns]
    rename = {
        "DATE_FROM": "date_from",
        "DATE_TO": "date_to",
        "TYPE_OF_RESERVES": "reserve_type",
        "PRODUCT": "product",
        "CAPACITY_PRICE_[(EUR/MW)/h]": "price_eur_mw_h",
        "OFFERED_CAPACITY_[MW]": "offered_mw",
        "ALLOCATED_CAPACITY_[MW]": "allocated_mw",
        "COUNTRY": "country",
        "NOTE": "note",
    }
    df = df.rename(columns=rename)
    parts = df["product"].astype(str).str.split("_", expand=True)
    df["direction"] = parts[0]
    df["block_start_h"] = pd.to_numeric(parts[1], errors="coerce").astype("Int64")
    df["date_from"] = pd.to_datetime(df["date_from"]).dt.date
    df = df.dropna(subset=["block_start_h"])
    df["year"] = year
    df["month"] = month
    df.to_parquet(cache)
    logger.info(f"    {len(df)} bids in {year}-{month:02d}")
    return df


def fetch_year(year: int) -> pd.DataFrame:
    """Concatenate all 12 months for a given year."""
    parts = []
    for m in range(1, 13):
        df = _fetch_month(year, m)
        if df is not None:
            parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def aggregate(df: pd.DataFrame) -> dict:
    """Compute clearing-rate statistics."""
    if df.empty:
        return {}
    total_offered = df["offered_mw"].sum()
    total_allocated = df["allocated_mw"].sum()
    overall_clearing = total_allocated / total_offered if total_offered else float("nan")

    # Per-bid status
    df = df.assign(
        bid_status=lambda x: x.apply(
            lambda r: "full"
            if r["allocated_mw"] >= r["offered_mw"] - 1e-6
            else ("partial" if r["allocated_mw"] > 1e-6 else "lost"),
            axis=1,
        ),
    )
    bid_status_counts = df["bid_status"].value_counts(normalize=True).to_dict()

    # Per-direction
    by_dir = (
        df.groupby("direction").agg(
            offered_mw=("offered_mw", "sum"),
            allocated_mw=("allocated_mw", "sum"),
            n_bids=("offered_mw", "count"),
        ).reset_index()
    )
    by_dir["clearing_rate"] = by_dir["allocated_mw"] / by_dir["offered_mw"]

    # Per-block per-direction
    by_block = (
        df.groupby(["direction", "block_start_h"]).agg(
            offered_mw=("offered_mw", "sum"),
            allocated_mw=("allocated_mw", "sum"),
            n_bids=("offered_mw", "count"),
        ).reset_index()
    )
    by_block["clearing_rate"] = by_block["allocated_mw"] / by_block["offered_mw"]

    # Per-month
    by_month = (
        df.groupby(["year", "month"]).agg(
            offered_mw=("offered_mw", "sum"),
            allocated_mw=("allocated_mw", "sum"),
            n_bids=("offered_mw", "count"),
        ).reset_index()
    )
    by_month["clearing_rate"] = by_month["allocated_mw"] / by_month["offered_mw"]

    # Quantity-weighted mean clearing price (€/MW/h)
    df["weighted_price"] = df["allocated_mw"] * df["price_eur_mw_h"]
    weighted_price = (
        df["weighted_price"].sum() / df["allocated_mw"].sum()
        if df["allocated_mw"].sum() else float("nan")
    )

    # Conditional clearing rates by bid-price quartile.
    # Low-price quartile (low-opportunity-cost bidders, e.g. BESS) is the
    # right proxy for our LP's effective bid_win_rate, since the LP commits
    # at marginal arbitrage cost which is competitive vs gas peakers.
    by_price = []
    for direction in ("POS", "NEG"):
        sub = df[df["direction"] == direction].copy()
        if sub.empty:
            continue
        sub["price_quartile"] = pd.qcut(
            sub["price_eur_mw_h"], q=4, labels=["Q1_low", "Q2", "Q3", "Q4_high"],
            duplicates="drop",
        )
        for q, qsub in sub.groupby("price_quartile", observed=True):
            offered = qsub["offered_mw"].sum()
            allocated = qsub["allocated_mw"].sum()
            by_price.append({
                "direction": direction,
                "price_quartile": str(q),
                "n_bids": len(qsub),
                "offered_mw": float(offered),
                "allocated_mw": float(allocated),
                "clearing_rate": float(allocated / offered) if offered else float("nan"),
                "min_price": float(qsub["price_eur_mw_h"].min()),
                "max_price": float(qsub["price_eur_mw_h"].max()),
                "median_price": float(qsub["price_eur_mw_h"].median()),
            })

    return {
        "n_bids": len(df),
        "total_offered_mw": float(total_offered),
        "total_allocated_mw": float(total_allocated),
        "overall_clearing_rate": float(overall_clearing),
        "weighted_avg_clearing_price_eur_mw_h": float(weighted_price),
        "bid_status_share": bid_status_counts,
        "by_direction": by_dir.to_dict("records"),
        "by_block": by_block.to_dict("records"),
        "by_month": by_month.to_dict("records"),
        "by_price_quartile": by_price,
    }


def report(stats: dict, year: int):
    print(f"\n{'=' * 70}")
    print(f"  Empirical aFRR cap clearing rates — DE, {year}")
    print(f"{'=' * 70}")
    print(f"\n  N bids: {stats['n_bids']:>10,}")
    print(f"  Total offered: {stats['total_offered_mw']:>12,.0f} MW")
    print(f"  Total allocated: {stats['total_allocated_mw']:>10,.0f} MW")
    print(f"  Overall clearing rate: {stats['overall_clearing_rate'] * 100:>5.1f} %")
    print(f"  Avg clearing price (qty-weighted): {stats['weighted_avg_clearing_price_eur_mw_h']:>5.2f} €/MW/h")

    print(f"\n  Bid status (share of bids):")
    for status, share in stats["bid_status_share"].items():
        print(f"    {status:>10}: {share * 100:>5.1f} %")

    print(f"\n  By direction:")
    print(f"  {'dir':>5} {'offered':>10} {'allocated':>10} {'clear%':>8}")
    for r in stats["by_direction"]:
        print(f"  {r['direction']:>5} "
              f"{r['offered_mw']:>10,.0f} {r['allocated_mw']:>10,.0f} "
              f"{r['clearing_rate'] * 100:>7.1f}%")

    print(f"\n  By block × direction:")
    print(f"  {'dir':>5} {'block_h':>8} {'offered':>10} {'allocated':>10} {'clear%':>8}")
    for r in sorted(stats["by_block"], key=lambda x: (x["direction"], x["block_start_h"])):
        print(f"  {r['direction']:>5} {r['block_start_h']:>8} "
              f"{r['offered_mw']:>10,.0f} {r['allocated_mw']:>10,.0f} "
              f"{r['clearing_rate'] * 100:>7.1f}%")

    print(f"\n  By month:")
    print(f"  {'year':>5} {'month':>5} {'offered':>10} {'allocated':>10} {'clear%':>8}")
    for r in stats["by_month"]:
        print(f"  {r['year']:>5} {r['month']:>5} "
              f"{r['offered_mw']:>10,.0f} {r['allocated_mw']:>10,.0f} "
              f"{r['clearing_rate'] * 100:>7.1f}%")

    print(f"\n  By price quartile (low quartile = competitive bidders ≈ BESS):")
    print(f"  {'dir':>5} {'quartile':>10} {'price_med':>10} {'offered':>10} "
          f"{'allocated':>10} {'clear%':>8}")
    for r in stats["by_price_quartile"]:
        print(f"  {r['direction']:>5} {r['price_quartile']:>10} "
              f"{r['median_price']:>9.2f}  "
              f"{r['offered_mw']:>10,.0f} {r['allocated_mw']:>10,.0f} "
              f"{r['clearing_rate'] * 100:>7.1f}%")


def main():
    args = sys.argv[1:]
    years = [int(a) for a in args] if args else [2024]
    for year in years:
        df = fetch_year(year)
        if df.empty:
            logger.warning(f"No data for {year}")
            continue
        stats = aggregate(df)
        report(stats, year)
        out = DATA_DIR / f"clearing_rate_summary_{year}.json"
        import json
        with open(out, "w") as f:
            # JSON-friendly: convert dicts of records
            json.dump(stats, f, indent=2, default=str)
        logger.info(f"\n  Wrote {out}")


if __name__ == "__main__":
    main()
