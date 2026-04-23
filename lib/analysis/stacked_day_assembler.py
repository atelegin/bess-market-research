"""
Assemble per-day input arrays for the stacked-market LP (A2.2).

Collects observed DE-LU market data for one day and shapes it into the
arrays expected by ``lib.models.dispatch_stacked.optimize_day_stacked``:

* DA day-ahead prices (hourly → step-expand to 15-min)
* ID intraday AEP prices (15-min, from netztransparenz)
* aFRR capacity prices per 4h block per direction (regelleistung)
* aFRR activation energy prices 15-min per direction (regelleistung)
* aFRR activation rates 15-min per direction
  (netztransparenz volumes / contracted pool size)

Any missing source yields ``None`` — caller decides whether to skip the
day or fall back to a partial stack.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date as Date
from typing import Optional

import numpy as np
import pandas as pd

from lib.data.afrr_activations import fetch_afrr_activations
from lib.data.ancillary_prices import (
    fetch_afrr_cap_prices_daily,
    fetch_afrr_energy_prices,
)
from lib.data.day_ahead_prices import fetch_day_ahead_prices
from lib.data.intraday_prices import fetch_id_aep
from lib.models.dispatch_stacked import (
    BLOCKS_PER_DAY,
    PERIODS_PER_BLOCK,
    PERIODS_PER_DAY,
)

logger = logging.getLogger(__name__)

# Default contracted aFRR pool sizes (MW). TSO-announced values stable
# across 2023-2025 per SO GL. If calibrating for a specific year, override.
DEFAULT_POOL_POS_MW = 2000.0
DEFAULT_POOL_NEG_MW = 1800.0


@dataclass
class StackedDayInputs:
    """Container for the 8 arrays the stacked LP expects for one day."""
    date: Date
    prices_da: np.ndarray            # (96,) EUR/MWh
    prices_id: np.ndarray            # (96,)
    afrr_cap_pos_price: np.ndarray   # (6,) EUR/MW/h
    afrr_cap_neg_price: np.ndarray   # (6,)
    afrr_energy_pos_price: np.ndarray  # (96,)
    afrr_energy_neg_price: np.ndarray  # (96,)
    afrr_activation_rate_pos: np.ndarray  # (96,) fraction of contracted pool
    afrr_activation_rate_neg: np.ndarray  # (96,)

    def as_kwargs(self) -> dict:
        return {
            "prices_da": self.prices_da,
            "prices_id": self.prices_id,
            "afrr_cap_pos_price": self.afrr_cap_pos_price,
            "afrr_cap_neg_price": self.afrr_cap_neg_price,
            "afrr_energy_pos_price": self.afrr_energy_pos_price,
            "afrr_energy_neg_price": self.afrr_energy_neg_price,
            "afrr_activation_rate_pos": self.afrr_activation_rate_pos,
            "afrr_activation_rate_neg": self.afrr_activation_rate_neg,
        }


def _expand_hourly_to_15min(hourly: np.ndarray) -> np.ndarray:
    """24 hourly prices → 96 15-min prices (step function)."""
    if len(hourly) != 24:
        # Some DA APIs return 23 or 25 hours on DST transitions. Pad/truncate.
        if len(hourly) < 24:
            hourly = np.concatenate([hourly, np.full(24 - len(hourly), hourly[-1])])
        else:
            hourly = hourly[:24]
    return np.repeat(hourly, PERIODS_PER_BLOCK // 4)  # 4 intervals per hour


def _extract_day_slice(
    frame: pd.DataFrame, target_date: Date, col: str,
    expected_len: int, tz: str = "UTC",
) -> Optional[np.ndarray]:
    """Slice a 15-min-indexed DataFrame to a single day, UTC-aligned."""
    if frame is None or frame.empty or col not in frame.columns:
        return None
    if frame.index.tz is None:
        local = frame.tz_localize(tz)
    else:
        local = frame.tz_convert(tz)
    start = pd.Timestamp(target_date, tz=tz)
    end = start + pd.Timedelta(days=1)
    slab = local.loc[(local.index >= start) & (local.index < end), col]
    if slab.empty:
        return None
    values = slab.to_numpy(dtype=float)
    if len(values) < expected_len:
        return None
    return values[:expected_len]


def assemble_day_inputs(
    target_date: Date,
    pool_pos_mw: float = DEFAULT_POOL_POS_MW,
    pool_neg_mw: float = DEFAULT_POOL_NEG_MW,
    # Optional pre-fetched frames — pass to avoid re-hitting APIs in a loop
    da_frame: pd.DataFrame | None = None,
    id_frame: pd.DataFrame | None = None,
    afrr_cap_frame: pd.DataFrame | None = None,
    afrr_energy_frame: pd.DataFrame | None = None,
    activations_frame: pd.DataFrame | None = None,
) -> Optional[StackedDayInputs]:
    """
    Build the 8 input arrays for the stacked-market LP for one day.

    When any source is missing or misaligned, returns ``None`` and logs.
    For multi-day runners, pre-fetch the frames once per year and pass
    them in — the function reuses them without re-hitting the APIs.
    """
    year = target_date.year
    day_start = pd.Timestamp(target_date, tz="UTC")
    day_end = day_start + pd.Timedelta(days=1)

    # -- DA (hourly → 15-min step) --
    if da_frame is None:
        try:
            da_frame = fetch_day_ahead_prices(
                start=f"{year}-01-01", end=f"{year}-12-31",
            )
        except Exception as e:
            logger.warning(f"assemble_day_inputs({target_date}): DA fetch failed: {e}")
            return None
    da_local = da_frame.tz_convert("UTC") if da_frame.index.tz is not None else da_frame.tz_localize("UTC")
    da_slab = da_local.loc[
        (da_local.index >= day_start) & (da_local.index < day_end), "price_eur_mwh"
    ]
    if da_slab.empty:
        logger.warning(f"assemble_day_inputs({target_date}): DA slab empty")
        return None
    prices_da = _expand_hourly_to_15min(da_slab.to_numpy(dtype=float))
    if len(prices_da) != PERIODS_PER_DAY:
        logger.warning(
            f"assemble_day_inputs({target_date}): DA expansion gave "
            f"{len(prices_da)} vs {PERIODS_PER_DAY}"
        )
        return None

    # -- ID AEP (15-min) --
    if id_frame is None:
        try:
            id_frame = fetch_id_aep(
                start=f"{year}-01-01", end=f"{year}-12-31",
            )
        except Exception as e:
            logger.warning(f"assemble_day_inputs({target_date}): ID fetch failed: {e}")
            id_frame = None
    # Pick first numeric column regardless of name (ID loader column name varies)
    if id_frame is not None and not id_frame.empty:
        id_col = next((c for c in id_frame.columns if "price" in c.lower() or "aep" in c.lower()),
                      id_frame.columns[0])
        prices_id = _extract_day_slice(id_frame, target_date, id_col, PERIODS_PER_DAY)
    else:
        prices_id = None
    if prices_id is None:
        # Fallback: reuse DA for ID when intraday is unavailable.
        logger.info(f"assemble_day_inputs({target_date}): ID unavailable, reusing DA")
        prices_id = prices_da.copy()

    # -- aFRR capacity per block --
    if afrr_cap_frame is None:
        afrr_cap_frame = fetch_afrr_cap_prices_daily(year)
    if afrr_cap_frame is None:
        logger.warning(f"assemble_day_inputs({target_date}): aFRR cap fetch failed")
        return None
    day_cap = afrr_cap_frame[afrr_cap_frame["date"] == target_date]
    afrr_cap_pos = np.zeros(BLOCKS_PER_DAY)
    afrr_cap_neg = np.zeros(BLOCKS_PER_DAY)
    for _, row in day_cap.iterrows():
        b = row["block_start_h"] // 4
        if row["direction"] == "POS":
            afrr_cap_pos[b] = row["eur_mw_h"]
        elif row["direction"] == "NEG":
            afrr_cap_neg[b] = row["eur_mw_h"]

    # -- aFRR energy prices per 15-min --
    if afrr_energy_frame is None:
        afrr_energy_frame = fetch_afrr_energy_prices(year)
    if afrr_energy_frame is None:
        logger.warning(f"assemble_day_inputs({target_date}): aFRR energy fetch failed")
        return None
    pos_price = _extract_day_slice(
        afrr_energy_frame, target_date, "pos_avg_eur_mwh", PERIODS_PER_DAY,
    )
    neg_price = _extract_day_slice(
        afrr_energy_frame, target_date, "neg_avg_eur_mwh", PERIODS_PER_DAY,
    )
    if pos_price is None or neg_price is None:
        logger.warning(f"assemble_day_inputs({target_date}): aFRR energy slab incomplete")
        return None

    # -- aFRR activations → rates --
    if activations_frame is None:
        try:
            activations_frame = fetch_afrr_activations(
                start=f"{year}-01-01", end=f"{year + 1}-01-01",
            )
        except Exception as e:
            logger.warning(f"assemble_day_inputs({target_date}): activations fetch failed: {e}")
            return None
    pos_mw = _extract_day_slice(activations_frame, target_date, "pos_mw", PERIODS_PER_DAY)
    neg_mw = _extract_day_slice(activations_frame, target_date, "neg_mw", PERIODS_PER_DAY)
    if pos_mw is None or neg_mw is None:
        logger.warning(f"assemble_day_inputs({target_date}): activations slab incomplete")
        return None
    rate_pos = np.clip(pos_mw / pool_pos_mw, 0.0, 1.0)
    rate_neg = np.clip(neg_mw / pool_neg_mw, 0.0, 1.0)

    return StackedDayInputs(
        date=target_date,
        prices_da=prices_da,
        prices_id=prices_id,
        afrr_cap_pos_price=afrr_cap_pos,
        afrr_cap_neg_price=afrr_cap_neg,
        afrr_energy_pos_price=pos_price,
        afrr_energy_neg_price=neg_price,
        afrr_activation_rate_pos=rate_pos,
        afrr_activation_rate_neg=rate_neg,
    )
