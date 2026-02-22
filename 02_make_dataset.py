from __future__ import annotations

import argparse
from typing import List

import numpy as np
import pandas as pd

from utils import ensure_dir, load_config, log_return, to_utc_hourly_index


def add_lags(df: pd.DataFrame, col: str, lags: List[int]) -> None:
    for l in lags:
        df[f"{col}_lag{l}"] = df[col].shift(l)


def add_rolling(df: pd.DataFrame, col: str, windows: List[int]) -> None:
    for w in windows:
        df[f"{col}_roll_mean{w}"] = df[col].rolling(w).mean()
        df[f"{col}_roll_std{w}"] = df[col].rolling(w).std()


def add_rsi(df: pd.DataFrame, price_col: str, period: int = 14) -> None:
    """RSI на основі погодинних цін закриття."""
    delta = df[price_col].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    df[f"rsi_{period}"] = 100 - (100 / (1 + rs))


def add_ema_diff(df: pd.DataFrame, price_col: str, fast: int = 12, slow: int = 26) -> None:
    """Нормалізована різниця EMA(fast) - EMA(slow), поділена на ціну."""
    ema_fast = df[price_col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[price_col].ewm(span=slow, adjust=False).mean()
    df[f"ema_diff_{fast}_{slow}"] = (ema_fast - ema_slow) / df[price_col]


def make_symbol_dataset(prices: pd.DataFrame, sym: str, cfg: dict) -> pd.DataFrame:
    pcol = f"price_{sym}"
    # Support both single horizon (int) and multiple horizons (list)
    horizons_raw = cfg["dataset"]["horizon_hours"]
    if isinstance(horizons_raw, list):
        horizons = list(map(int, horizons_raw))
    else:
        horizons = [int(horizons_raw)]
    lags = list(map(int, cfg["dataset"]["lags"]))
    windows = list(map(int, cfg["dataset"]["rolling_windows"]))

    df = prices[["timestamp", pcol]].copy()
    df["r_1h"] = np.log(df[pcol] / df[pcol].shift(1))

    # Generate target columns for ALL horizons in one pass
    for horizon in horizons:
        df[f"y_r_{horizon}h"] = log_return(df[pcol], horizon=horizon)

    add_lags(df, "r_1h", lags)
    add_rolling(df, "r_1h", windows)
    add_rsi(df, pcol, period=14)
    add_ema_diff(df, pcol, fast=12, slow=26)

    df["symbol"] = sym
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="", help="Override output dataset CSV path (default from config)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dir(cfg["data"]["out_dir"])

    prices = pd.read_csv(cfg["data"]["prices_file"])
    prices["timestamp"] = to_utc_hourly_index(prices["timestamp"])
    prices = prices.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")

    # Optional: merge event features if available
    event_path = cfg["data"].get("event_features_file")
    events = None
    if event_path:
        try:
            events = pd.read_csv(event_path)
            events["timestamp"] = to_utc_hourly_index(events["hour"] if "hour" in events.columns else events["timestamp"])
            # standardize column name to timestamp and keep symbol column
            if "hour" in events.columns:
                events = events.drop(columns=["hour"])
            # Prevent dataset row duplication if the event file contains duplicate keys
            if {"timestamp", "symbol"}.issubset(events.columns):
                events = (
                    events.sort_values(["timestamp", "symbol"])
                    .drop_duplicates(subset=["timestamp", "symbol"], keep="last")
                )
        except FileNotFoundError:
            events = None

    frames = [make_symbol_dataset(prices, sym, cfg) for sym in cfg["prices"]["symbols"]]
    ds = pd.concat(frames, axis=0, ignore_index=True)

    if events is not None and len(events) > 0:
        # Merge on (timestamp, symbol)
        ds = ds.merge(events, on=["timestamp", "symbol"], how="left")

    # Drop rows with NaNs due to lag/rolling/forward targets (all horizons)
    horizons_raw = cfg["dataset"]["horizon_hours"]
    horizons = list(map(int, horizons_raw)) if isinstance(horizons_raw, list) else [int(horizons_raw)]
    target_cols = [f"y_r_{h}h" for h in horizons]
    feature_na_cols = [c for c in ds.columns if c.startswith("r_1h_") or c == "r_1h"]
    ds = ds.dropna(subset=target_cols + feature_na_cols)

    # For event features: fill missing (hours with no news) with zeros
    if events is not None and len(events) > 0:
        for c in ds.columns:
            if c.startswith(("news_", "sent_")):
                ds[c] = ds[c].fillna(0.0)

    out_file = args.out.strip() or cfg["data"]["dataset_file"]
    ds.to_csv(out_file, index=False)
    print(f"Saved: {out_file} ({len(ds)} rows, {ds.shape[1]} cols)")


if __name__ == "__main__":
    main()