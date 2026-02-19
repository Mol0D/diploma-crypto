from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from typing import Dict, List

import pandas as pd
import requests
import time

from utils import ensure_dir, load_config


def _dt_utc(days_ago: int) -> str:
    dt = datetime.now(timezone.utc) - timedelta(days=int(days_ago))
    return dt.strftime("%Y%m%d%H%M%S")


def fetch_gdelt_docs(query: str, start_dt_utc: str, lang: str, maxrecords: int = 250) -> pd.DataFrame:
    # GDELT 2 DOC API
    # Returns at most maxrecords per request; for a dissertation pipeline we keep it simple and pull a bounded sample.
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    # Normalize query:
    # - GDELT is strict: parentheses are only allowed around OR groups, but OR groups must be parenthesized.
    # - In title-mode we already build queries like: title:(A OR B OR C)
    #   so we must NOT wrap the whole query again, otherwise we end up with:
    #   (title:(A OR B))  -> can trigger "Parentheses may only be used around OR'd statements."
    q = query.strip()
    if q.startswith("title:"):
        # leave as-is; title:(...) wrapping is handled upstream
        pass
    else:
        if " OR " in q and not (q.startswith("(") and q.endswith(")")):
            q = f"({q})"

    params = {
        "query": q,
        "format": "json",
        "mode": "artlist",
        "formatdatetime": "1",
        "startdatetime": start_dt_utc,
        "maxrecords": int(maxrecords),
        "sort": "hybridrel",
        "sourcelang": lang,
    }
    # Friendly user agent can reduce blocking on some public endpoints
    headers = {"User-Agent": "Mozilla/5.0 (compatible; PrismDiplomaBot/1.0)"}
    # Use a (connect_timeout, read_timeout) tuple for better control on flaky endpoints
    r = requests.get(url, params=params, headers=headers, timeout=(15, 120))
    # Handle rate limiting / transient server errors with backoff + retry
    if r.status_code == 429:
        raise RuntimeError("GDELT rate-limited (HTTP 429).")
    if 500 <= int(r.status_code) <= 599:
        raise RuntimeError(f"GDELT server error (HTTP {r.status_code}).")
    r.raise_for_status()
    # GDELT sometimes returns HTML (e.g., rate-limit / transient errors). Guard JSON parsing.
    ctype = (r.headers.get("content-type") or "").lower()
    # Some failures come back as text/html or even as a JSON string that contains an HTML header dump.
    # Detect this early and treat as transient.
    text_snip = (r.text or "")[:300].replace("\n", " ")
    if "unknown error occurred" in (r.text or "").lower():
        raise RuntimeError("GDELT returned an unknown error page; retry later.")
    try:
        j = r.json()
    except Exception as e:
        raise RuntimeError(
            f"GDELT response is not JSON (content-type={ctype}). "
            f"First 300 chars: {text_snip!r}"
        ) from e
    # Some responses return a dict with {"articles": [...]}, others may return a list directly.
    if isinstance(j, dict):
        arts = j.get("articles", [])
    elif isinstance(j, list):
        arts = j
    else:
        arts = []

    # Debug: show top-level keys and a preview of the first item to help diagnose schema drift.
    if bool(int(__import__("os").environ.get("GDELT_DEBUG", "0"))):
        if isinstance(j, dict):
            print(f"[gdelt] top-level keys: {sorted(list(j.keys()))[:25]}")
            if "articles" in j:
                print(f"[gdelt] articles type: {type(j['articles']).__name__} len={len(j['articles']) if isinstance(j['articles'], list) else 'n/a'}")
        else:
            print(f"[gdelt] top-level type: {type(j).__name__}")
        if arts:
            try:
                first = arts[0]
                if isinstance(first, dict):
                    print(f"[gdelt] first article keys: {sorted(list(first.keys()))[:40]}")
                    for k in ["datetime", "seendate", "date", "url", "title"]:
                        if k in first:
                            print(f"[gdelt] first[{k}]={first.get(k)!r}")
            except Exception:
                pass

    if not arts:
        return pd.DataFrame(columns=["datetime", "sourceCountry", "language", "title", "url"])
    df = pd.DataFrame(arts)
    # Normalize a minimal schema (GDELT commonly returns lowercase keys like seendate/sourcecountry)
    rename_map = {
        "seendate": "datetime",
        "sourcecountry": "sourceCountry",
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})

    keep = [c for c in ["datetime", "sourceCountry", "language", "title", "url"] if c in df.columns]
    df = df[keep].copy()
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--maxrecords", type=int, default=20)
    ap.add_argument("--sleep", type=float, default=5.0)
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--backoff", type=float, default=2.0)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--dump-schema", action="store_true", help="print GDELT JSON schema preview")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dir(cfg["data"]["out_dir"])

    days = int(cfg["events"]["days"])
    lang = str(cfg["events"].get("language", "english")).lower()
    start_dt = _dt_utc(days)
    search_mode = str(cfg["events"].get("search_mode", "full")).lower()

    rows = []
    for sym, q in [("BTC", cfg["events"]["keywords_btc"]), ("ETH", cfg["events"]["keywords_eth"])]:
        q2 = q
        # For title-only search, ensure the OR-group is parenthesized *inside* title:(...),
        # and also avoid producing title:(<no OR>) which GDELT may reject.
        if search_mode == "title":
            inner = q.strip()
            if " OR " in inner and not (inner.startswith("(") and inner.endswith(")")):
                inner = f"({inner})"
            q2 = f"title:{inner}"
        else:
            # full-text search; keep query as-is
            q2 = q
        if args.debug:
            print(f"\n[{sym}] query={q2!r} startdatetime={start_dt!r} sourcelang={lang!r} maxrecords={args.maxrecords}")
        if args.dump_schema:
            # Enable schema printing inside fetch_gdelt_docs without changing function signatures everywhere.
            import os

            os.environ["GDELT_DEBUG"] = "1"
        last_err = None
        for attempt in range(int(args.retries) + 1):
            try:
                df = fetch_gdelt_docs(query=q2, start_dt_utc=start_dt, lang=lang, maxrecords=args.maxrecords)
                last_err = None
                break
            except (RuntimeError, requests.exceptions.RequestException) as e:
                # rate limit / transient network errors
                last_err = e
                wait = float(args.sleep) * (float(args.backoff) ** attempt)
                print(f"GDELT temporary error for {sym}. attempt={attempt+1} waiting {wait:.1f}s err={e}")
                time.sleep(wait)
        if last_err is not None:
            raise last_err
        if args.debug or args.dump_schema:
            print(f"[{sym}] fetched rows={len(df)} columns={list(df.columns)}")
        df["symbol"] = sym
        rows.append(df)
        time.sleep(float(args.sleep))

    out = pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame()

    # Parse datetime to UTC (be robust to alternative field names)
    if len(out) > 0:
        if "datetime" in out.columns:
            dt_col = "datetime"
        elif "seendate" in out.columns:
            dt_col = "seendate"
        elif "date" in out.columns:
            dt_col = "date"
        else:
            dt_col = None

        if dt_col is not None:
            out["timestamp"] = pd.to_datetime(out[dt_col], utc=True, errors="coerce")
        else:
            out["timestamp"] = pd.NaT

        out = out.dropna(subset=["timestamp"]).sort_values("timestamp")

        # Deterministic output: drop duplicates so re-runs don't "grow" the dataset
        if "url" in out.columns:
            out = out.drop_duplicates(subset=["symbol", "url"], keep="last")
        elif "title" in out.columns:
            out = out.drop_duplicates(subset=["symbol", "title", "timestamp"], keep="last")

    out_file = cfg["data"]["news_file"]
    # Always overwrite (not append). With the de-duplication above this makes re-runs deterministic.
    out.to_csv(out_file, index=False)
    print(f"Saved: {out_file} ({len(out)} rows) [overwrite]")


if __name__ == "__main__":
    main()
