"""
Parquet cache layer — content-addressed caching for DataFrames.

Generates deterministic cache keys from parameters, stores results
as parquet files, and supports TTL-based expiry via JSON metadata.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _sanitize_token(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def make_cache_key(prefix: str, **params: Any) -> str:
    payload = json.dumps(params, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.md5(payload).hexdigest()[:12]
    return f"{_sanitize_token(prefix)}_{digest}"


def cache_path(cache_key: str) -> Path:
    return CACHE_DIR / f"{cache_key}.parquet"


def metadata_path(cache_key: str) -> Path:
    return CACHE_DIR / f"{cache_key}.json"


def is_cache_fresh(cache_key: str, ttl_hours: float | None) -> bool:
    data_path = cache_path(cache_key)
    meta_path = metadata_path(cache_key)
    if not data_path.exists() or not meta_path.exists():
        return False
    if ttl_hours is None:
        return True
    metadata = json.loads(meta_path.read_text())
    created_at = datetime.fromisoformat(metadata["created_at"])
    return datetime.now(timezone.utc) - created_at <= timedelta(hours=ttl_hours)


def write_dataframe(cache_key: str, frame: pd.DataFrame, metadata: dict[str, Any] | None = None) -> pd.DataFrame:
    data_path = cache_path(cache_key)
    meta_path = metadata_path(cache_key)
    frame.to_parquet(data_path)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(frame.shape[0]),
        "columns": list(frame.columns),
    }
    if metadata:
        payload.update(metadata)
    meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return frame


def read_dataframe(cache_key: str) -> pd.DataFrame:
    return pd.read_parquet(cache_path(cache_key))


def get_or_build_dataframe(
    cache_key: str,
    builder: Callable[[], pd.DataFrame],
    ttl_hours: float | None,
    force_refresh: bool = False,
    metadata: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if not force_refresh and is_cache_fresh(cache_key, ttl_hours):
        return read_dataframe(cache_key)
    frame = builder()
    return write_dataframe(cache_key, frame, metadata=metadata)

