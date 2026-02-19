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


def add_rsi(df: pd.DataFrame, col: str, period: int = 14) -> None:
    delta = df[col].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, float("nan"))
    df[f"rsi_{period}"] = 100 - (100 / (1 + rs))


def add_ema(df: pd.DataFrame, col: str, spans: List[int]) -> None:
    for s in spans:
        df[f"ema_{s}"] = df[col].ewm(span=s, adjust=False).mean()


def make_symbol_dataset(prices: pd.DataFrame, sym: str, cfg: dict) -> pd.DataFrame:
    pcol = f"price_{sym}"
    horizon = int(cfg["dataset"]["horizon_hours"])
    lags = list(map(int, cfg["dataset"]["lags"]))
    windows = list(map(int, cfg["dataset"]["rolling_windows"]))

    df = prices[["timestamp", pcol]].copy()
    df["r_1h"] = np.log(df[pcol] / df[pcol].shift(1))
    df[f"y_r_{horizon}h"] = log_return(df[pcol], horizon=horizon)

    add_lags(df, "r_1h", lags)
    add_rolling(df, "r_1h", windows)

    # Technical indicators
    add_rsi(df, pcol, period=14)
    add_ema(df, pcol, spans=[12, 26])
    df["ema_cross"] = df["ema_12"] - df["ema_26"]

    # Normalize EMA by price so the signal is scale-invariant
    df["ema_12_norm"] = df["ema_12"] / df[pcol] - 1
    df["ema_26_norm"] = df["ema_26"] / df[pcol] - 1
    df["ema_cross_norm"] = df["ema_cross"] / df[pcol]

    # Drop raw EMA price columns (keep normalized versions)
    df = df.drop(columns=["ema_12", "ema_26", "ema_cross"])

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

    # Optional: merge GDELT event features if available
    event_path = cfg["data"].get("event_features_file")
    events = None
    if event_path:
        try:
            events = pd.read_csv(event_path)
            events["timestamp"] = to_utc_hourly_index(events["hour"] if "hour" in events.columns else events["timestamp"])
            if "hour" in events.columns:
                events = events.drop(columns=["hour"])
            if {"timestamp", "symbol"}.issubset(events.columns):
                events = (
                    events.sort_values(["timestamp", "symbol"])
                    .drop_duplicates(subset=["timestamp", "symbol"], keep="last")
                )
        except FileNotFoundError:
            events = None

    # Optional: merge Fear & Greed Index features if available
    fg_path = cfg["data"].get("fg_features_file")
    fg = None
    if fg_path:
        try:
            fg = pd.read_csv(fg_path)
            fg["timestamp"] = to_utc_hourly_index(fg["timestamp"])
            fg = fg.drop(columns=["fear_greed_class"], errors="ignore")
            if {"timestamp", "symbol"}.issubset(fg.columns):
                fg = (
                    fg.sort_values(["timestamp", "symbol"])
                    .drop_duplicates(subset=["timestamp", "symbol"], keep="last")
                )
            print(f"Loaded Fear & Greed features: {len(fg)} rows, cols={[c for c in fg.columns if c not in ('timestamp','symbol')]}")
        except FileNotFoundError:
            print("fg_features_file not found, skipping Fear & Greed")
            fg = None

    frames = [make_symbol_dataset(prices, sym, cfg) for sym in cfg["prices"]["symbols"]]
    ds = pd.concat(frames, axis=0, ignore_index=True)

    # Merge GDELT event features
    if events is not None and len(events) > 0:
        ds = ds.merge(events, on=["timestamp", "symbol"], how="left")

    # Merge Fear & Greed features
    if fg is not None and len(fg) > 0:
        ds = ds.merge(fg, on=["timestamp", "symbol"], how="left")

    # Drop rows with NaNs due to lag/rolling/forward target
    horizon = int(cfg["dataset"]["horizon_hours"])
    target_col = f"y_r_{horizon}h"
    ds = ds.dropna(subset=[target_col] + [c for c in ds.columns if c.startswith("r_1h_") or c == "r_1h"])

    # Fill missing event features with zeros
    if events is not None and len(events) > 0:
        for c in ds.columns:
            if c.startswith(("news_", "sent_")):
                ds[c] = ds[c].fillna(0.0)

    # Fill missing Fear & Greed features with zeros
    if fg is not None and len(fg) > 0:
        for c in ds.columns:
            if c.startswith("fg_") or c == "fear_greed":
                ds[c] = ds[c].fillna(0.0)
        fg_cols_in_ds = [c for c in ds.columns if c.startswith("fg_") or c == "fear_greed"]
        print(f"Fear & Greed cols in dataset: {fg_cols_in_ds}")

    out_file = args.out.strip() or cfg["data"]["dataset_file"]
    ds.to_csv(out_file, index=False)
    print(f"Saved: {out_file} ({len(ds)} rows, {ds.shape[1]} cols)")


if __name__ == "__main__":
    main()