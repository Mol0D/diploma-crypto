# 05c_make_fg_features.py
import pandas as pd
import numpy as np
from utils import ensure_dir, load_config


def main():
    cfg = load_config("config.yaml")
    ensure_dir(cfg["data"]["out_dir"])

    fg = pd.read_csv(cfg["data"].get("fear_greed_file", "data/fear_greed.csv"))
    fg["timestamp"] = pd.to_datetime(fg["timestamp"], utc=True)
    fg = fg.sort_values("timestamp").drop_duplicates(subset=["timestamp"])

    # Розширюємо на годинну сітку
    hourly_index = pd.date_range(
        start=fg["timestamp"].min(),
        end=fg["timestamp"].max() + pd.Timedelta(hours=23),
        freq="h",
        tz="UTC"
    )
    fg_hourly = (
        fg.set_index("timestamp")
        .reindex(hourly_index)
        .ffill()  # денне значення тягнеться на всі години дня
        .reset_index()
        .rename(columns={"index": "timestamp"})
    )

    # Лаги (в годинах)
    fg_hourly["fg_lag24h"]  = fg_hourly["fear_greed"].shift(24)   # 1 день тому
    fg_hourly["fg_lag48h"]  = fg_hourly["fear_greed"].shift(48)   # 2 дні тому
    fg_hourly["fg_lag168h"] = fg_hourly["fear_greed"].shift(168)  # 7 днів тому

    # Зміна індексу за добу
    fg_hourly["fg_delta24h"] = fg_hourly["fear_greed"] - fg_hourly["fg_lag24h"]

    # Бінарні ознаки режиму
    fg_hourly["fg_extreme_fear"] = (fg_hourly["fear_greed"] <= 25).astype(int)
    fg_hourly["fg_greed"]        = (fg_hourly["fear_greed"] >= 60).astype(int)

    # Додаємо symbol — потрібно для merge з датасетом
    rows = []
    for sym in cfg["prices"]["symbols"]:
        d = fg_hourly.copy()
        d["symbol"] = sym
        rows.append(d)

    out = pd.concat(rows, ignore_index=True)
    out_file = cfg["data"].get("fg_features_file", "data/fg_features.csv")
    out.to_csv(out_file, index=False)
    print(f"Saved: {out_file} ({len(out)} rows, {out.shape[1]} cols)")
    print(out.tail(3))


if __name__ == "__main__":
    main()