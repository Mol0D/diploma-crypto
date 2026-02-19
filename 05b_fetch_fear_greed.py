# 05b_fetch_fear_greed.py
import requests
import pandas as pd
from utils import ensure_dir, load_config

def fetch_fear_greed(days: int) -> pd.DataFrame:
    url = "https://api.alternative.me/fng/"
    params = {"limit": days, "format": "json"}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()["data"]
    
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(
        df["timestamp"].astype(int), unit="s", utc=True
    ).dt.floor("D")
    df["fear_greed"] = df["value"].astype(float)
    df["fear_greed_class"] = df["value_classification"]
    
    return df[["timestamp", "fear_greed", "fear_greed_class"]].sort_values("timestamp")

def main():
    cfg = load_config("config.yaml")
    ensure_dir(cfg["data"]["out_dir"])
    days = int(cfg["prices"]["days"])
    
    df = fetch_fear_greed(days)
    out = cfg["data"].get("fear_greed_file", "data/fear_greed.csv")
    df.to_csv(out, index=False)
    print(f"Saved: {out} ({len(df)} rows)")
    print(df.tail())

if __name__ == "__main__":
    main()