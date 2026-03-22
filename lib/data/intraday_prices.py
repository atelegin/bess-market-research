"""
Intraday electricity prices — 15-min ID-AEP index.

Source: netztransparenz.de (volume-weighted last-500 MW continuous ID).
Fetches day-by-day via the Highchart grid-data endpoint, caches as parquet.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from time import sleep

import pandas as pd
import requests

from lib.data.cache import get_or_build_dataframe, make_cache_key

ID_AEP_URL = "https://www.netztransparenz.de/DesktopModules/LotesCharts/Services/HighchartService.asmx/GetIdAepGridDataForDay"
ID_AEP_TEMPLATE = "D:/root/Websites/dnn_unb_ext02_relaunch/Web/DesktopModules/LotesCharts/Templates/ExcelTemplate.xlsx"
DEFAULT_TIMEZONE = "Europe/Berlin"


def _empty_price_frame() -> pd.DataFrame:
    return pd.DataFrame(index=pd.DatetimeIndex([], tz=DEFAULT_TIMEZONE), columns=["price_eur_mwh"])


def _request_payload(day: pd.Timestamp) -> dict[str, object]:
    return {
        "dateFrom": day.strftime("%Y-%m-%dT00:00:00"),
        "dateTo": None,
        "asImage": False,
        "diagramType": "line",
        "highChartType": "2",
        "columnColors": None,
        "template": ID_AEP_TEMPLATE,
        "title": "Index Ausgleichsenergiepreis",
        "timezone": DEFAULT_TIMEZONE,
    }


def _parse_grid_data(grid_data: str) -> pd.DataFrame:
    if not grid_data.strip():
        return _empty_price_frame()
    raw = pd.read_csv(StringIO(grid_data), sep=";")
    if raw.empty:
        return _empty_price_frame()

    price_column = next((column for column in raw.columns if "ID AEP" in column), None)
    if price_column is None:
        return _empty_price_frame()

    timestamp = pd.to_datetime(
        raw["Datum"].astype(str) + " " + raw["von"].astype(str),
        format="%d.%m.%Y %H:%M",
        errors="coerce",
    )
    frame = pd.DataFrame(
        {
            "timestamp": timestamp,
            "price_eur_mwh": pd.to_numeric(raw[price_column], errors="coerce"),
        }
    ).dropna(subset=["timestamp", "price_eur_mwh"])

    if frame.empty:
        return _empty_price_frame()

    frame["timestamp"] = frame["timestamp"].dt.tz_localize(
        DEFAULT_TIMEZONE,
        ambiguous="infer",
        nonexistent="shift_forward",
    )
    return frame.set_index("timestamp").sort_index()


def _fetch_id_aep_day(day: pd.Timestamp, max_attempts: int = 4) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.post(
                ID_AEP_URL,
                json=_request_payload(day),
                timeout=60,
            )
            response.raise_for_status()
            payload = response.json()
            inner = json.loads(payload["d"])
            frame = _parse_grid_data(inner.get("gridData", ""))
            start_ts = pd.Timestamp(day.date(), tz=DEFAULT_TIMEZONE)
            end_ts = start_ts + pd.Timedelta(days=1)
            return frame[(frame.index >= start_ts) & (frame.index < end_ts)]
        except requests.RequestException as exc:
            last_error = exc
            if attempt == max_attempts:
                break
            sleep(1.5 * attempt)

    raise RuntimeError(f"Failed to fetch ID-AEP for {day.date().isoformat()} after {max_attempts} attempts") from last_error


def _build_id_aep_frame(start: str, end: str, max_workers: int = 4) -> pd.DataFrame:
    days = pd.date_range(start=start, end=end, freq="D")
    if len(days) == 0:
        return _empty_price_frame()

    with ThreadPoolExecutor(max_workers=min(max_workers, len(days))) as executor:
        frames = [frame for frame in executor.map(_fetch_id_aep_day, days) if not frame.empty]

    if not frames:
        return _empty_price_frame()

    frame = pd.concat(frames).sort_index()
    frame = frame[~frame.index.duplicated(keep="first")]
    return frame


def fetch_id_aep(
    start: str = "2021-01-01",
    end: str = "2025-12-31",
    force_refresh: bool = False,
) -> pd.DataFrame:
    cache_key = make_cache_key(
        "netztransparenz_id_aep",
        start=start,
        end=end,
        source="netztransparenz_id_aep_v1",
    )
    return get_or_build_dataframe(
        cache_key=cache_key,
        builder=lambda: _build_id_aep_frame(start=start, end=end),
        ttl_hours=24 * 30,
        force_refresh=force_refresh,
        metadata={"source": ID_AEP_URL},
    )
