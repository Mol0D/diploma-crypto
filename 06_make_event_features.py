from __future__ import annotations

import argparse

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

from utils import ensure_dir, load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--no-finbert", action="store_true", help="Disable FinBERT sentiment (debug/performance)")
    ap.add_argument("--out", default="", help="Override output CSV path (default from config)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    # Ensure output directories exist (config paths are relative to repo root; script may be run from code/)
    ensure_dir(cfg["data"]["out_dir"])
    if args.out.strip():
        import os
        from pathlib import Path

        out_parent = Path(args.out.strip()).expanduser().resolve().parent
        os.makedirs(out_parent, exist_ok=True)

    news = pd.read_csv(cfg["data"]["news_file"])
    news["timestamp"] = pd.to_datetime(news["timestamp"], utc=True, errors="coerce")
    news = news.dropna(subset=["timestamp"])
    news["hour"] = news["timestamp"].dt.floor("h")

    # Sentiment from titles:
    # 1) VADER (fast baseline)
    vader = SentimentIntensityAnalyzer()
    news["title"] = news["title"].fillna("")
    news["sent_vader"] = news["title"].apply(lambda s: vader.polarity_scores(str(s))["compound"])

    # 2) FinBERT (financial sentiment)
    if args.no_finbert:
        news["sent_finbert"] = 0.0
        news["finbert_pos"] = 0
        news["finbert_neg"] = 0
    else:
        finbert = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            return_all_scores=True,
            truncation=True,
        )

        def finbert_score(text: str) -> float:
            """
            Convert label probabilities to a signed score: P(pos) - P(neg).

            pipeline(return_all_scores=True) output format depends on transformers version:
              - either [[{'label':..., 'score':...}, ...]]  (per item)
              - or [{'label':..., 'score':...}, ...]        (already flattened)
            """
            out = finbert(text[:512])
            if isinstance(out, list) and len(out) > 0 and isinstance(out[0], list):
                scores = out[0]
            else:
                scores = out
            probs = {d["label"].lower(): float(d["score"]) for d in scores}
            return probs.get("positive", 0.0) - probs.get("negative", 0.0)

        news["sent_finbert"] = news["title"].apply(lambda s: finbert_score(str(s)))
        news["finbert_pos"] = (news["sent_finbert"] > 0.05).astype(int)
        news["finbert_neg"] = (news["sent_finbert"] < -0.05).astype(int)

    agg = (
        news.groupby(["hour", "symbol"], as_index=False)
        .agg(
            news_count=("title", "size"),
            sent_vader_mean=("sent_vader", "mean"),
            sent_vader_std=("sent_vader", "std"),
            sent_finbert_mean=("sent_finbert", "mean"),
            sent_finbert_std=("sent_finbert", "std"),
            finbert_pos_count=("finbert_pos", "sum"),
            finbert_neg_count=("finbert_neg", "sum"),
        )
        .sort_values(["symbol", "hour"])
    )

    for c in ["sent_vader_std", "sent_finbert_std"]:
        agg[c] = agg[c].fillna(0.0)

    # Simple lags for event signals
    for col in [
        "news_count",
        "sent_vader_mean",
        "sent_vader_std",
        "sent_finbert_mean",
        "sent_finbert_std",
        "finbert_pos_count",
        "finbert_neg_count",
    ]:
        agg[f"{col}_lag1"] = agg.groupby("symbol")[col].shift(1)
        agg[f"{col}_lag2"] = agg.groupby("symbol")[col].shift(2)

    out_file = args.out.strip() or cfg["data"]["event_features_file"]
    agg.to_csv(out_file, index=False)
    print(f"Saved: {out_file} ({len(agg)} rows)")

    finbert_cols = [c for c in agg.columns if "finbert" in c]
    print(f"Event features include FinBERT cols: {finbert_cols}")
    if finbert_cols:
        nz = float((agg[finbert_cols].abs().sum(axis=1) > 0).mean())
        print(f"FinBERT feature nonzero-row share: {nz:.3f}")


if __name__ == "__main__":
    main()
