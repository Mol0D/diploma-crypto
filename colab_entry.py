from __future__ import annotations

"""
Single entry-point for running the whole pipeline in Google Colab without zipping/unzipping
the entire Prism project (which can break Cyrillic filenames).

Recommended Colab flow:
1) Create a folder, e.g. /content/crypto_diploma
2) Paste the files from Prism's code/ folder into that folder (or upload just code/ as a zip).
3) Run:
   !pip -q install -r requirements.txt
   !python colab_entry.py --all
"""

import argparse
from pathlib import Path


def run(cmd: str) -> None:
    import subprocess

    print(f"\n$ {cmd}")
    subprocess.check_call(cmd, shell=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--fetch-news", action="store_true")
    ap.add_argument("--make-events", action="store_true")
    ap.add_argument("--make-dataset", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--all", action="store_true", help="run fetch -> dataset -> train -> eval")
    args = ap.parse_args()

    if args.all:
        args.fetch = args.fetch_news = args.make_events = args.make_dataset = args.train = args.eval = True

    if args.fetch:
        run(f"python3 01_fetch_prices.py --config {args.config}")
    if args.fetch_news:
        run(f"python3 05_fetch_news_gdelt.py --config {args.config}")
    if args.make_events:
        run(f"python3 06_make_event_features.py --config {args.config}")
    if args.make_dataset:
        run(f"python3 02_make_dataset.py --config {args.config}")
    if args.train:
        run(f"python3 03_train_quantile_gbm.py --config {args.config}")
    if args.eval:
        run(f"python3 04_evaluate.py --config {args.config}")


if __name__ == "__main__":
    main()
