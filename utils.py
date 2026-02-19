from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import yaml


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def to_utc_hourly_index(x) -> pd.Series:
    return pd.to_datetime(x, utc=True, errors="coerce").dt.floor("h")


def log_return(p: pd.Series, horizon: int) -> pd.Series:
    return np.log(p.shift(-horizon) / p)


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    e = y_true - y_pred
    return float(np.mean(np.maximum(q * e, (q - 1) * e)))


def prediction_interval_coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean((y >= lo) & (y <= hi)))


def time_series_splits(n: int, n_splits: int, test_size: int, gap: int = 0):
    """
    Expanding-window splits with fixed test_size.
    Yields (train_idx, test_idx).
    """
    total_test = n_splits * test_size
    if n <= total_test + gap:
        raise ValueError("Not enough samples for requested splits")

    last_test_end = n
    for k in range(n_splits):
        test_end = last_test_end - k * test_size
        test_start = test_end - test_size
        train_end = test_start - gap
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        yield train_idx, test_idx
