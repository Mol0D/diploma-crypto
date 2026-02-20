from __future__ import annotations

import argparse
from typing import List

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from utils import ensure_dir, load_config, pinball_loss, time_series_splits


def train_for_horizon(ds: pd.DataFrame, horizon: int, cfg: dict) -> None:
    target_col = f"y_r_{horizon}h"
    if target_col not in ds.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in dataset. "
            f"Run 02_make_dataset.py with horizon_hours including {horizon}."
        )

    feature_cols = [
        c for c in ds.columns
        if c not in {"timestamp", "symbol"}
        and not c.startswith("price_")
        and not c.startswith("y_r_")  # exclude ALL target columns, not just current
    ]
    X = ds[feature_cols].to_numpy(dtype=float)
    y = ds[target_col].to_numpy(dtype=float)

    n_splits = int(cfg["eval"]["n_splits"])
    test_size = int(cfg["eval"]["test_size"])
    # gap must be >= horizon to avoid look-ahead bias
    gap = max(int(cfg["eval"].get("gap", 0)), horizon)

    splits = list(time_series_splits(len(ds), n_splits=n_splits, test_size=test_size, gap=gap))
    train_idx, valid_idx = splits[0]

    X_train, y_train = X[train_idx], y[train_idx]
    X_valid, y_valid = X[valid_idx], y[valid_idx]

    base_params = dict(cfg["model"]["lgbm_params"])
    quantiles: List[float] = list(map(float, cfg["model"]["quantiles"]))

    bundle = {"feature_cols": feature_cols, "horizon": horizon, "models": {}, "valid_losses": {}}
    for q in quantiles:
        params = dict(base_params)
        params.update({"objective": "quantile", "alpha": float(q)})
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train)
        pred = model.predict(X_valid)
        loss = pinball_loss(y_valid, pred, q=q)
        bundle["models"][q] = model
        bundle["valid_losses"][q] = loss
        print(f"  horizon={horizon}h  q={q:.2f}  valid pinball={loss:.6f}")

    out_path = f"{cfg['eval']['models_dir']}/lgbm_quantiles_{horizon}h.joblib"
    joblib.dump(bundle, out_path)
    print(f"  Saved: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--horizon", type=int, default=None,
                    help="Train only this horizon (default: all from config)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dir(cfg["eval"]["models_dir"])

    ds = pd.read_csv(cfg["data"]["dataset_file"], parse_dates=["timestamp"])

    horizons_raw = cfg["dataset"]["horizon_hours"]
    all_horizons = (
        list(map(int, horizons_raw))
        if isinstance(horizons_raw, list)
        else [int(horizons_raw)]
    )

    if args.horizon is not None:
        if args.horizon not in all_horizons:
            raise ValueError(
                f"--horizon {args.horizon} not in config horizon_hours={all_horizons}"
            )
        horizons = [args.horizon]
    else:
        horizons = all_horizons

    for horizon in horizons:
        print(f"\n=== Training horizon={horizon}h ===")
        train_for_horizon(ds, horizon, cfg)

    print("\nDone.")


if __name__ == "__main__":
    main()