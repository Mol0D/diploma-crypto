#!/usr/bin/env bash
set -euo pipefail

# Run the full pipeline from within the code/ directory.
# Usage:
#   cd code
#   bash run_all.sh
#
# Optional env vars:
#   MAXRECORDS=20 SLEEP=7 RETRIES=3 BACKOFF=2

CONFIG="${CONFIG:-config.yaml}"
MAXRECORDS="${MAXRECORDS:-20}"
SLEEP="${SLEEP:-7}"
RETRIES="${RETRIES:-3}"
BACKOFF="${BACKOFF:-2}"

echo "Using config: ${CONFIG}"
echo "GDELT: maxrecords=${MAXRECORDS} sleep=${SLEEP} retries=${RETRIES} backoff=${BACKOFF}"

python3 -m pip install -r requirements.txt

echo "1) Fetch prices"
python3 01_fetch_prices.py --config "${CONFIG}"

echo "2) Fetch news (GDELT)"
python3 05_fetch_news_gdelt.py --config "${CONFIG}" --maxrecords "${MAXRECORDS}" --sleep "${SLEEP}" --retries "${RETRIES}" --backoff "${BACKOFF}"

echo "3) Build event features (VADER + FinBERT)"
python3 06_make_event_features.py --config "${CONFIG}"

echo "4) Build merged dataset"
python3 02_make_dataset.py --config "${CONFIG}"

echo "5) Compare feature sets (market vs events vs combined)"
python3 07_compare_feature_sets.py --config "${CONFIG}"

echo "Done. Check ../data and ../reports"
