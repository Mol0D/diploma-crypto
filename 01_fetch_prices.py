from __future__ import annotations

import argparse
import time
from typing import List

import pandas as pd
import requests

from utils import ensure_dir, load_config, to_utc_hourly_index


COINGECKO_IDS = {"BTC": "bitcoin", "ETH": "ethereum"}


def fetch_coingecko_market_chart(coin_id: str, vs_currency: str, days: int) -> pd.DataFrame:
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": vs_currency.lower(), "days": int(days), "interval": "hourly"}
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()

    df = pd.DataFrame(data["prices"], columns=["timestamp_ms", "price"])
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True).dt.floor("h")
    df = df.drop(columns=["timestamp_ms"]).drop_duplicates(subset=["timestamp"], keep="last")
    df = df.sort_values("timestamp")
    return df


def fetch_cryptocompare_histohour(sym: str, currency: str, days: int) -> pd.DataFrame:
    # CryptoCompare: https://min-api.cryptocompare.com/documentation?key=Historical&cat=dataHistohour
    # We need (days*24) points, and we request in chunks (API limit per call).
    url = "https://min-api.cryptocompare.com/data/v2/histohour"
    total_limit = max(24, int(days) * 24)
    limit_per_call = 2000  # CryptoCompare typical max

    frames = []
    to_ts = None
    remaining = total_limit
    while remaining > 0:
        limit = min(limit_per_call, remaining)
        params = {"fsym": sym.upper(), "tsym": currency.upper(), "limit": int(limit)}
        if to_ts is not None:
            params["toTs"] = int(to_ts)
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        j = r.json()
        if j.get("Response") != "Success":
            raise RuntimeError(f"CryptoCompare error: {j}")
        data = j["Data"]["Data"]
        if not data:
            break
        d = pd.DataFrame(data)
        # 'time' is unix seconds
        d["timestamp"] = pd.to_datetime(d["time"], unit="s", utc=True).dt.floor("h")
        d = d[["timestamp", "close"]].rename(columns={"close": "price"})
        frames.append(d)
        # next chunk ends before the earliest timestamp we got
        to_ts = int(pd.to_datetime(d["timestamp"].min(), utc=True).timestamp()) - 3600
        remaining -= len(d)
        time.sleep(0.3)

    df = pd.concat(frames, axis=0, ignore_index=True)
    df = df.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
    return df


def fetch_cryptocompare_histohour(sym: str, currency: str, days: int) -> pd.DataFrame:
    # CryptoCompare: https://min-api.cryptocompare.com/documentation?key=Historical&cat=dataHistohour
    # We need (days*24) points, and we request in chunks (API limit per call).
    url = "https://min-api.cryptocompare.com/data/v2/histohour"
    total_limit = max(24, int(days) * 24)
    limit_per_call = 2000  # CryptoCompare typical max

    frames = []
    to_ts = None
    remaining = total_limit
    while remaining > 0:
        limit = min(limit_per_call, remaining)
        params = {"fsym": sym.upper(), "tsym": currency.upper(), "limit": int(limit)}
        if to_ts is not None:
            params["toTs"] = int(to_ts)
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        j = r.json()
        if j.get("Response") != "Success":
            raise RuntimeError(f"CryptoCompare error: {j}")
        data = j["Data"]["Data"]
        if not data:
            break
        d = pd.DataFrame(data)
        # 'time' is unix seconds
        d["timestamp"] = pd.to_datetime(d["time"], unit="s", utc=True).dt.floor("h")
        d = d[["timestamp", "close"]].rename(columns={"close": "price"})
        frames.append(d)
        # next chunk ends before the earliest timestamp we got
        to_ts = int(pd.to_datetime(d["timestamp"].min(), utc=True).timestamp()) - 3600
        remaining -= len(d)
        time.sleep(0.3)

    df = pd.concat(frames, axis=0, ignore_index=True)
    df = df.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)

    out_dir = ensure_dir(cfg["data"]["out_dir"])
    out_file = cfg["data"]["prices_file"]

    symbols: List[str] = cfg["prices"]["symbols"]
    currency: str = cfg["prices"]["currency"]
    days: int = int(cfg["prices"]["days"])
    source: str = str(cfg["prices"].get("source", "cryptocompare")).lower()
    source: str = str(cfg["prices"].get("source", "cryptocompare")).lower()

    frames = []
    for sym in symbols:
        if source == "coingecko":
            coin_id = COINGECKO_IDS.get(sym)
            if not coin_id:
                raise ValueError(f"Unknown symbol {sym}. Add mapping to COINGECKO_IDS.")
            d = fetch_coingecko_market_chart(coin_id, currency, days)
        elif source == "cryptocompare":
            d = fetch_cryptocompare_histohour(sym, currency, days)
        else:
            raise ValueError(f"Unknown prices.source={source}. Use 'cryptocompare' or 'coingecko'.")

        d = d.rename(columns={"price": f"price_{sym}"})
        frames.append(d.set_index("timestamp"))
        time.sleep(1.2)

    prices = pd.concat(frames, axis=1).reset_index()
    # Ensure deterministic output on re-run
    if "timestamp" in prices.columns:
        prices = prices.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    prices.to_csv(out_file, index=False)
    print(f"Saved: {out_file} ({len(prices)} rows)")


if __name__ == "__main__":
    main()
