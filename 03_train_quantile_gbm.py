from __future__ import annotations

import argparse
from typing import List

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from utils import ensure_dir, load_config, pinball_loss, time_series_splits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dir(cfg["eval"]["models_dir"])

    ds = pd.read_csv(cfg["data"]["dataset_file"], parse_dates=["timestamp"])
    horizon = int(cfg["dataset"]["horizon_hours"])
    target_col = f"y_r_{horizon}h"

    feature_cols = [c for c in ds.columns if c not in {"timestamp", "symbol", target_col} and not c.startswith("price_")]
    X = ds[feature_cols].to_numpy(dtype=float)
    y = ds[target_col].to_numpy(dtype=float)

    n_splits = int(cfg["eval"]["n_splits"])
    test_size = int(cfg["eval"]["test_size"])
    gap = int(cfg["eval"].get("gap", 0))

    # Use one split for a quick validation during training
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
        print(f"q={q:.2f} valid pinball={loss:.6f}")

    out_path = f"{cfg['eval']['models_dir']}/lgbm_quantiles_{horizon}h.joblib"
    joblib.dump(bundle, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
