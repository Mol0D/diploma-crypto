from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from utils import ensure_dir, load_config, pinball_loss, prediction_interval_coverage, time_series_splits


def _assert_no_duplicates(ds: pd.DataFrame, keys: List[str]) -> None:
    if not set(keys).issubset(ds.columns):
        return
    dup = ds.duplicated(subset=keys).sum()
    if dup:
        raise ValueError(f"Dataset has {dup} duplicated rows by keys={keys}.")


def _assert_no_duplicates(ds: pd.DataFrame, keys: List[str]) -> None:
    if not set(keys).issubset(ds.columns):
        return
    dup = ds.duplicated(subset=keys).sum()
    if dup:
        raise ValueError(f"Dataset has {dup} duplicated rows by keys={keys}.")


def train_bundle(ds: pd.DataFrame, feature_cols: List[str], target_col: str, cfg: dict) -> dict:
    X = ds[feature_cols].to_numpy(dtype=float)
    y = ds[target_col].to_numpy(dtype=float)

    n_splits = int(cfg["eval"]["n_splits"])
    test_size = int(cfg["eval"]["test_size"])
    gap = int(cfg["eval"].get("gap", 0))
    splits = list(time_series_splits(len(ds), n_splits=n_splits, test_size=test_size, gap=gap))
    train_idx, valid_idx = splits[0]

    X_train, y_train = X[train_idx], y[train_idx]
    X_valid, y_valid = X[valid_idx], y[valid_idx]

    base_params = dict(cfg["model"]["lgbm_params"])
    quantiles = list(map(float, cfg["model"]["quantiles"]))

    bundle = {"feature_cols": feature_cols, "models": {}, "valid_losses": {}}
    for q in quantiles:
        params = dict(base_params)
        params.update({"objective": "quantile", "alpha": float(q)})
        m = LGBMRegressor(**params)
        m.fit(X_train, y_train)
        pred = m.predict(X_valid)
        bundle["models"][q] = m
        bundle["valid_losses"][q] = pinball_loss(y_valid, pred, q=q)
    return bundle


def eval_bundle(ds: pd.DataFrame, bundle: dict, target_col: str, cfg: dict) -> pd.DataFrame:
    feature_cols = bundle["feature_cols"]
    X = ds[feature_cols].to_numpy(dtype=float)
    y = ds[target_col].to_numpy(dtype=float)

    n_splits = int(cfg["eval"]["n_splits"])
    test_size = int(cfg["eval"]["test_size"])
    gap = int(cfg["eval"].get("gap", 0))
    splits = list(time_series_splits(len(ds), n_splits=n_splits, test_size=test_size, gap=gap))

    rows = []
    qs = sorted(bundle["models"].keys())
    for i, (_, test_idx) in enumerate(splits, start=1):
        X_test = X[test_idx]
        y_test = y[test_idx]
        preds = {q: bundle["models"][q].predict(X_test) for q in qs}
        metrics: Dict[str, float] = {f"pinball_q{q:.2f}": pinball_loss(y_test, preds[q], q=q) for q in qs}
        if 0.05 in preds and 0.95 in preds:
            metrics["pi_90_coverage"] = prediction_interval_coverage(y_test, preds[0.05], preds[0.95])
            metrics["pi_90_avg_width"] = float(np.mean(preds[0.95] - preds[0.05]))
        metrics["split"] = float(i)
        metrics["n_test"] = float(len(test_idx))
        rows.append(metrics)
    return pd.DataFrame(rows).sort_values("split")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--dataset", default="", help="Override input dataset CSV path (default from config)")
    ap.add_argument("--tag", default="", help="Optional run tag appended to output filenames (e.g. finbert/nofinbert)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dir(cfg["eval"]["plots_dir"])
    ensure_dir(cfg["eval"]["models_dir"])

    import os
    from pathlib import Path

    # Resolve relative paths from the current working directory (often repo_root/code/)
    dataset_path = args.dataset.strip() or cfg["data"]["dataset_file"]
    dataset_path = str(Path(dataset_path).expanduser().resolve())
    ds = pd.read_csv(dataset_path, parse_dates=["timestamp"])
    # Keep deterministic ordering for splits
    if {"timestamp", "symbol"}.issubset(ds.columns):
        ds = ds.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    horizon = int(cfg["dataset"]["horizon_hours"])
    target_col = f"y_r_{horizon}h"
    tag = args.tag.strip()
    tag_suffix = f"_{tag}" if tag else ""

    # Sanity checks: allow repeated timestamps (multiple symbols), but not duplicate (timestamp,symbol)
    _assert_no_duplicates(ds, ["timestamp", "symbol"])

    # Sanity checks: allow repeated timestamps (multiple symbols), but not duplicate (timestamp,symbol)
    _assert_no_duplicates(ds, ["timestamp", "symbol"])

    # Define feature sets
    market_cols = [c for c in ds.columns if (
    c.startswith("r_1h") or
    c.startswith("rsi_") or
    c.startswith("ema_")
)]
    event_cols = [c for c in ds.columns if c.startswith(("news_", "sent_", "fg_")) or c == "fear_greed"]

    # In case user reuses an older dataset where event columns have different prefixes,
    # allow a broader fallback.
    if len(event_cols) == 0:
        event_cols = [c for c in ds.columns if ("news" in c.lower() or "sent" in c.lower())]
    base_drop = {"timestamp", "symbol", target_col}
    # keep only numeric columns within these sets
    market_cols = [c for c in market_cols if c in ds.columns and c not in base_drop]
    event_cols = [c for c in event_cols if c in ds.columns and c not in base_drop]

    sets = {
        "market": market_cols,
        "events": event_cols,
        "combined": sorted(list(set(market_cols + event_cols))),
    }

    # Print quick diagnostics so it's obvious whether FinBERT features are present / nonzero
    finbert_cols = [c for c in ds.columns if "finbert" in c.lower()]
    if finbert_cols:
        nz_share = float((ds[finbert_cols].fillna(0.0).abs().sum(axis=1) > 0).mean())
        print(f"FinBERT cols detected: {finbert_cols}")
        print(f"FinBERT nonzero-row share: {nz_share:.3f}")
    else:
        print("FinBERT cols detected: []")

    # Print quick diagnostics so it's obvious whether FinBERT features are present / nonzero
    finbert_cols = [c for c in ds.columns if "finbert" in c.lower()]
    if finbert_cols:
        nz_share = float((ds[finbert_cols].fillna(0.0).abs().sum(axis=1) > 0).mean())
        print(f"FinBERT cols detected: {finbert_cols}")
        print(f"FinBERT nonzero-row share: {nz_share:.3f}")
    else:
        print("FinBERT cols detected: []")

    results = []
    for name, cols in sets.items():
        if len(cols) == 0:
            print(f"Skip {name}: no features detected")
            continue
        print(f"\n=== Feature set: {name} (n_features={len(cols)}) ===")
        if len(cols) <= 30:
            print("Features:", cols)
        else:
            print("First 30 features:", cols[:30])
        bundle = train_bundle(ds, cols, target_col, cfg)
        model_path = f"{cfg['eval']['models_dir']}/lgbm_{name}_{horizon}h{tag_suffix}.joblib"
        joblib.dump(bundle, model_path)
        rep = eval_bundle(ds, bundle, target_col, cfg)
        out_csv = f"{cfg['eval']['plots_dir']}/eval_{horizon}h_{name}{tag_suffix}.csv"
        rep.to_csv(out_csv, index=False)
        print(f"Saved: {out_csv} and {model_path} (dataset={dataset_path})")
        # store summary
        mean_row = rep.drop(columns=["split", "n_test"]).mean(numeric_only=True).to_dict()
        mean_row["set"] = name
        results.append(mean_row)

    if results:
        summary = pd.DataFrame(results).set_index("set")
        out_sum = f"{cfg['eval']['plots_dir']}/eval_{horizon}h_summary{tag_suffix}.csv"
        summary.to_csv(out_sum)
        print(f"Saved: {out_sum}")
        print(summary)


if __name__ == "__main__":
    main()


# додай в кінець 07_compare_feature_sets.py або запусти окремо
import joblib, pandas as pd

bundle = joblib.load("models/lgbm_combined_4h.joblib")
model = bundle["models"][0.5]
fi = pd.Series(model.feature_importances_, index=bundle["feature_cols"])
print(fi.sort_values(ascending=False).head(15))
fg_fi = fi[fi.index.str.startswith("fg_") | (fi.index == "fear_greed")]
print(fg_fi.sort_values(ascending=False))