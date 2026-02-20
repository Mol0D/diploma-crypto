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


def train_bundle(ds: pd.DataFrame, feature_cols: List[str], target_col: str, cfg: dict, horizon: int) -> dict:
    X = ds[feature_cols].to_numpy(dtype=float)
    y = ds[target_col].to_numpy(dtype=float)

    n_splits = int(cfg["eval"]["n_splits"])
    test_size = int(cfg["eval"]["test_size"])
    gap = max(int(cfg["eval"].get("gap", 0)), horizon)
    splits = list(time_series_splits(len(ds), n_splits=n_splits, test_size=test_size, gap=gap))
    train_idx, valid_idx = splits[0]

    X_train, y_train = X[train_idx], y[train_idx]
    X_valid, y_valid = X[valid_idx], y[valid_idx]

    base_params = dict(cfg["model"]["lgbm_params"])
    quantiles = list(map(float, cfg["model"]["quantiles"]))

    bundle = {"feature_cols": feature_cols, "horizon": horizon, "models": {}, "valid_losses": {}}
    for q in quantiles:
        params = dict(base_params)
        params.update({"objective": "quantile", "alpha": float(q)})
        m = LGBMRegressor(**params)
        m.fit(X_train, y_train)
        pred = m.predict(X_valid)
        bundle["models"][q] = m
        bundle["valid_losses"][q] = pinball_loss(y_valid, pred, q=q)
    return bundle


def eval_bundle(ds: pd.DataFrame, bundle: dict, target_col: str, cfg: dict, horizon: int) -> pd.DataFrame:
    feature_cols = bundle["feature_cols"]
    X = ds[feature_cols].to_numpy(dtype=float)
    y = ds[target_col].to_numpy(dtype=float)

    n_splits = int(cfg["eval"]["n_splits"])
    test_size = int(cfg["eval"]["test_size"])
    gap = max(int(cfg["eval"].get("gap", 0)), horizon)
    splits = list(time_series_splits(len(ds), n_splits=n_splits, test_size=test_size, gap=gap))

    rows = []
    qs = sorted(bundle["models"].keys())
    for i, (_, test_idx) in enumerate(splits, start=1):
        X_test = X[test_idx]
        y_test = y[test_idx]
        preds = {q: bundle["models"][q].predict(X_test) for q in qs}
        metrics: Dict[str, float] = {
            f"pinball_q{q:.2f}": pinball_loss(y_test, preds[q], q=q) for q in qs
        }
        if 0.05 in preds and 0.95 in preds:
            metrics["pi_90_coverage"] = prediction_interval_coverage(
                y_test, preds[0.05], preds[0.95]
            )
            metrics["pi_90_avg_width"] = float(np.mean(preds[0.95] - preds[0.05]))
        metrics["split"] = float(i)
        metrics["n_test"] = float(len(test_idx))
        rows.append(metrics)
    return pd.DataFrame(rows).sort_values("split")


def run_for_horizon(ds: pd.DataFrame, horizon: int, cfg: dict, tag_suffix: str, dataset_path: str) -> None:
    target_col = f"y_r_{horizon}h"
    if target_col not in ds.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. "
            f"Run 02_make_dataset.py with horizon_hours including {horizon}."
        )

    # Feature sets — exclude ALL target columns, not just current horizon
    all_target_cols = {c for c in ds.columns if c.startswith("y_r_")}
    base_drop = {"timestamp", "symbol"} | all_target_cols | {c for c in ds.columns if c.startswith("price_")}

    market_cols = [
        c for c in ds.columns
        if (c.startswith("r_1h") or c.startswith("r_1h_")) and c not in base_drop
    ]
    event_cols = [
        c for c in ds.columns
        if c.startswith(("news_", "sent_", "fg_", "fear_", "extreme_")) and c not in base_drop
    ]
    if len(event_cols) == 0:
        event_cols = [
            c for c in ds.columns
            if ("news" in c.lower() or "sent" in c.lower()) and c not in base_drop
        ]

    sets = {
        "market":   market_cols,
        "events":   event_cols,
        "combined": sorted(set(market_cols + event_cols)),
    }

    # Diagnostics
    finbert_cols = [c for c in ds.columns if "finbert" in c.lower()]
    if finbert_cols:
        nz_share = float((ds[finbert_cols].fillna(0.0).abs().sum(axis=1) > 0).mean())
        print(f"  FinBERT cols: {finbert_cols}  nonzero-row share: {nz_share:.3f}")
    else:
        print("  FinBERT cols: []")

    results = []
    for name, cols in sets.items():
        if len(cols) == 0:
            print(f"  Skip '{name}': no features detected")
            continue
        print(f"\n  --- Feature set: {name} (n={len(cols)}, horizon={horizon}h) ---")
        bundle = train_bundle(ds, cols, target_col, cfg, horizon)
        model_path = f"{cfg['eval']['models_dir']}/lgbm_{name}_{horizon}h{tag_suffix}.joblib"
        joblib.dump(bundle, model_path)
        rep = eval_bundle(ds, bundle, target_col, cfg, horizon)
        out_csv = f"{cfg['eval']['plots_dir']}/eval_{horizon}h_{name}{tag_suffix}.csv"
        rep.to_csv(out_csv, index=False)
        print(f"  Saved: {out_csv}")
        mean_row = rep.drop(columns=["split", "n_test"]).mean(numeric_only=True).to_dict()
        mean_row["set"] = name
        results.append(mean_row)

    if results:
        summary = pd.DataFrame(results).set_index("set")
        out_sum = f"{cfg['eval']['plots_dir']}/eval_{horizon}h_summary{tag_suffix}.csv"
        summary.to_csv(out_sum)
        print(f"\n  Summary saved: {out_sum}")
        print(summary.to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--dataset", default="", help="Override input dataset CSV path")
    ap.add_argument("--tag", default="", help="Optional run tag appended to output filenames")
    ap.add_argument("--horizon", type=int, default=None,
                    help="Run only this horizon (default: all from config)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dir(cfg["eval"]["plots_dir"])
    ensure_dir(cfg["eval"]["models_dir"])

    from pathlib import Path
    dataset_path = args.dataset.strip() or cfg["data"]["dataset_file"]
    dataset_path = str(Path(dataset_path).expanduser().resolve())
    ds = pd.read_csv(dataset_path, parse_dates=["timestamp"])

    if {"timestamp", "symbol"}.issubset(ds.columns):
        ds = ds.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    _assert_no_duplicates(ds, ["timestamp", "symbol"])

    horizons_raw = cfg["dataset"]["horizon_hours"]
    all_horizons = (
        list(map(int, horizons_raw))
        if isinstance(horizons_raw, list)
        else [int(horizons_raw)]
    )

    if args.horizon is not None:
        if args.horizon not in all_horizons:
            raise ValueError(f"--horizon {args.horizon} not in config {all_horizons}")
        horizons = [args.horizon]
    else:
        horizons = all_horizons

    tag_suffix = f"_{args.tag.strip()}" if args.tag.strip() else ""

    for horizon in horizons:
        print(f"\n{'='*60}")
        print(f"=== Comparing feature sets: horizon={horizon}h ===")
        print(f"{'='*60}")
        run_for_horizon(ds, horizon, cfg, tag_suffix, dataset_path)

    print("\nDone.")


if __name__ == "__main__":
    main()