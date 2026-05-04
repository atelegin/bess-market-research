"""
Regime-conditional forecast for Stage 1 of the two-stage dispatch (ADR-001).

Stage 1 (DA-aFRR commitment, fired at D−1 08:00) sees:
* DA-day prices (assumed forecastable from forwards)
* aFRR capacity prices (auction clears at D−1 08:00 — known)
* aFRR activation rate α and energy prices: forecast only

This module produces those forecasts from per-day market regime
classification (``lib.models.price_regime.fit_regimes``):

  α̂_d[t] = mean over historical days in the same regime as day d of
           the realised α at 15-min slot t
  ê_d[t] = same construction for aFRR energy prices

The forecast captures the bulk of the diurnal + day-type structure
operators rely on in practice (regime-conditional historical means are
the natural baseline for any aFRR forecast — quiet days have low α,
volatile days have high α, but the within-day shape is driven by load
profiles common to all days in a regime).

This is the v1.0 baseline. Future ADRs may upgrade to ARMA / NN-based
forecasters — those plug into the same return signature.
"""
from __future__ import annotations

import logging
from datetime import date as Date, time as Time
from typing import Optional

import numpy as np
import pandas as pd

from lib.models.dispatch_stacked import PERIODS_PER_DAY
from lib.models.price_regime import RegimeClassification

logger = logging.getLogger(__name__)


_ALL_15MIN_SLOTS: tuple[Time, ...] = tuple(
    Time(h, m) for h in range(24) for m in (0, 15, 30, 45)
)
assert len(_ALL_15MIN_SLOTS) == PERIODS_PER_DAY


def _utc_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of frame with a tz-aware UTC DatetimeIndex."""
    if frame.index.tz is None:
        return frame.tz_localize("UTC")
    return frame.tz_convert("UTC")


def _filter_to_regime_days(
    frame: pd.DataFrame,
    regime_classification: RegimeClassification,
    target_regime: int,
) -> pd.DataFrame:
    """Slice ``frame`` (15-min indexed) to dates assigned ``target_regime``.

    ``regime_classification.regime_labels`` is indexed by ``date``;
    ``frame`` may be 15-min indexed across multiple days. We build a
    boolean mask on per-row date.
    """
    labels = regime_classification.regime_labels
    same_regime_dates = set(labels[labels == target_regime].index)
    if not same_regime_dates:
        return frame.iloc[0:0]
    row_dates = pd.Series(frame.index.normalize().date, index=frame.index)
    mask = row_dates.isin(same_regime_dates)
    return frame.loc[mask]


def _tod_mean_to_96(series: pd.Series, fill: float = 0.0) -> np.ndarray:
    """Group ``series`` by HH:MM, take mean, and reindex to all 96 slots."""
    if series.empty:
        return np.full(PERIODS_PER_DAY, fill, dtype=float)
    grouped = series.groupby(series.index.time).mean()
    grouped = grouped.reindex(list(_ALL_15MIN_SLOTS), fill_value=fill)
    return grouped.to_numpy(dtype=float)


def regime_conditional_alpha_forecast(
    target_date: Date,
    regime_classification: RegimeClassification,
    activations_frame: pd.DataFrame,
    pool_pos_mw: float,
    pool_neg_mw: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(α_pos_forecast, α_neg_forecast)`` for ``target_date``.

    Each output is a length-96 array of forecast activation rates per
    15-min slot, equal to the regime-conditional historical mean over
    days in ``activations_frame`` whose regime label matches that of
    ``target_date``.

    Args:
        target_date: Day to forecast for.
        regime_classification: Output of
            :func:`lib.models.price_regime.fit_regimes` over the
            historical horizon.
        activations_frame: 15-min indexed DataFrame with columns
            ``pos_mw`` and ``neg_mw`` (raw activated MW; same shape as
            ``lib.data.afrr_activations.fetch_afrr_activations``).
        pool_pos_mw / pool_neg_mw: Contracted aFRR pool sizes used to
            convert MW activation to fractional α.

    Returns:
        ``(alpha_pos, alpha_neg)`` — both shape (96,), each ∈ [0, 1].

    Notes:
        * Falls back to the all-day mean (ignoring regime) when the
          target date's regime has zero historical days in the frame.
        * Slots with no observations within the regime are filled with
          0.0 (conservative: no expected activation).
    """
    try:
        target_regime = regime_classification.regime_for_date(target_date)
    except KeyError:
        logger.warning(
            f"regime_conditional_alpha_forecast: {target_date} outside "
            f"regime classification window — falling back to full-frame mean."
        )
        target_regime = None

    frame = _utc_index(activations_frame)
    if target_regime is not None:
        sub = _filter_to_regime_days(frame, regime_classification, target_regime)
        if sub.empty:
            sub = frame
    else:
        sub = frame

    alpha_pos_series = (sub["pos_mw"] / pool_pos_mw).clip(0.0, 1.0)
    alpha_neg_series = (sub["neg_mw"] / pool_neg_mw).clip(0.0, 1.0)
    alpha_pos = _tod_mean_to_96(alpha_pos_series, fill=0.0)
    alpha_neg = _tod_mean_to_96(alpha_neg_series, fill=0.0)
    return alpha_pos, alpha_neg


def regime_conditional_energy_price_forecast(
    target_date: Date,
    regime_classification: RegimeClassification,
    energy_frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(price_pos_forecast, price_neg_forecast)`` for ``target_date``.

    Identical construction to
    :func:`regime_conditional_alpha_forecast` but on energy prices.

    Args:
        target_date: Day to forecast for.
        regime_classification: Regime fit on DA prices spanning the
            relevant historical horizon.
        energy_frame: 15-min indexed DataFrame with columns
            ``pos_avg_eur_mwh`` and ``neg_avg_eur_mwh`` (matches the
            output of
            :func:`lib.data.ancillary_prices.fetch_afrr_energy_prices`).

    Returns:
        ``(price_pos, price_neg)`` — both shape (96,), EUR/MWh.

    Notes:
        * NEG energy prices are typically negative (operator paid to
          absorb); we preserve sign and use 0.0 as missing-slot fill.
    """
    try:
        target_regime = regime_classification.regime_for_date(target_date)
    except KeyError:
        logger.warning(
            f"regime_conditional_energy_price_forecast: {target_date} "
            f"outside regime classification window — falling back to "
            f"full-frame mean."
        )
        target_regime = None

    frame = _utc_index(energy_frame)
    if target_regime is not None:
        sub = _filter_to_regime_days(frame, regime_classification, target_regime)
        if sub.empty:
            sub = frame
    else:
        sub = frame

    price_pos = _tod_mean_to_96(sub["pos_avg_eur_mwh"], fill=0.0)
    price_neg = _tod_mean_to_96(sub["neg_avg_eur_mwh"], fill=0.0)
    return price_pos, price_neg


def perfect_foresight_forecast(
    realised_pos: np.ndarray,
    realised_neg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pass-through forecaster: Stage 1 sees realised values.

    Useful as an upper-bound diagnostic — it shows the NPV impact of
    forecast quality by comparing two-stage with this oracle vs the
    regime-mean baseline. If perfect-foresight Stage 1 ≈ regime-mean
    Stage 1, then forecast quality is not the binding constraint and
    further forecast investment offers no economic uplift.
    """
    return np.asarray(realised_pos, dtype=float), np.asarray(realised_neg, dtype=float)
