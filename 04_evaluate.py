from __future__ import annotations

import argparse
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import (
    ensure_dir,
    load_config,
    pinball_loss,
    prediction_interval_coverage,
    time_series_splits,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dir(cfg["eval"]["plots_dir"])

    horizon = int(cfg["dataset"]["horizon_hours"])
    target_col = f"y_r_{horizon}h"

    ds = pd.read_csv(cfg["data"]["dataset_file"], parse_dates=["timestamp"])
    bundle_path = f"{cfg['eval']['models_dir']}/lgbm_quantiles_{horizon}h.joblib"
    bundle = joblib.load(bundle_path)
    feature_cols: List[str] = bundle["feature_cols"]

    n_splits = int(cfg["eval"]["n_splits"])
    test_size = int(cfg["eval"]["test_size"])
    gap = int(cfg["eval"].get("gap", 0))

    splits = list(time_series_splits(len(ds), n_splits=n_splits, test_size=test_size, gap=gap))

    rows = []
    for i, (train_idx, test_idx) in enumerate(splits, start=1):
        X_test = ds.iloc[test_idx][feature_cols].to_numpy(dtype=float)
        y_test = ds.iloc[test_idx][target_col].to_numpy(dtype=float)

        qs = sorted(bundle["models"].keys())
        preds = {q: bundle["models"][q].predict(X_test) for q in qs}

        metrics: Dict[str, float] = {f"pinball_q{q:.2f}": pinball_loss(y_test, preds[q], q=q) for q in qs}
        if 0.05 in preds and 0.95 in preds:
            metrics["pi_90_coverage"] = prediction_interval_coverage(y_test, preds[0.05], preds[0.95])
            metrics["pi_90_avg_width"] = float(np.mean(preds[0.95] - preds[0.05]))

        metrics["split"] = float(i)
        metrics["n_test"] = float(len(test_idx))
        rows.append(metrics)

    rep = pd.DataFrame(rows).sort_values("split")
    out_csv = f"{cfg['eval']['plots_dir']}/eval_{horizon}h.csv"
    rep.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    print(rep)

    # Plot last split
    _, test_idx = splits[-1]
    X_last = ds.iloc[test_idx][feature_cols].to_numpy(dtype=float)
    y_last = ds.iloc[test_idx][target_col].to_numpy(dtype=float)
    x = np.arange(len(y_last))

    p50 = bundle["models"][0.5].predict(X_last) if 0.5 in bundle["models"] else None
    p05 = bundle["models"][0.05].predict(X_last) if 0.05 in bundle["models"] else None
    p95 = bundle["models"][0.95].predict(X_last) if 0.95 in bundle["models"] else None

    plt.figure(figsize=(10, 4))
    plt.plot(x, y_last, label="true", linewidth=1)
    if p50 is not None:
        plt.plot(x, p50, label="pred q=0.50", linewidth=1)
    if p05 is not None and p95 is not None:
        plt.fill_between(x, p05, p95, alpha=0.2, label="PI(0.05,0.95)")
    plt.title(f"Out-of-sample (last split), horizon={horizon}h")
    plt.legend()
    out_png = f"{cfg['eval']['plots_dir']}/pred_vs_true_{horizon}h.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
