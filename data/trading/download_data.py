# download_data.py
"""
Download 1-minute stock bars from Alpaca and save as parquet:
  {out_dir}/{TICKER}/{YEAR}.parquet
"""

import os
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed, Adjustment

from .config import TICKERS, API_KEY, API_SECRET


def _feed_enum(feed):
    if feed == "iex":
        return DataFeed.IEX
    if feed == "sip":
        return DataFeed.SIP
    return DataFeed.DELAYED_SIP


def _adj_enum(adj):
    if adj == "raw":
        return Adjustment.RAW
    if adj == "split":
        return Adjustment.SPLIT
    if adj == "dividend":
        return Adjustment.DIVIDEND
    return Adjustment.ALL


def fetch_bars_one_symbol_one_year(
    client, symbol, year, feed, adjustment, max_retries, sleep_sec
):
    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)

    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(1, TimeFrameUnit.Minute),
        start=start,
        end=end,
        feed=feed,
        adjustment=adjustment,
    )

    last_err = None
    for attempt in range(max_retries):
        try:
            bars = client.get_stock_bars(req)
            df = bars.df

            # df may have MultiIndex (symbol,timestamp) or single index
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index()
            else:
                df = df.reset_index()
                if "symbol" not in df.columns:
                    df.insert(0, "symbol", symbol)

            if "timestamp" not in df.columns:
                for cand in ["time", "t"]:
                    if cand in df.columns:
                        df = df.rename(columns={cand: "timestamp"})
                        break

            df = df[df["symbol"] == symbol].copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
            return df

        except Exception as e:
            last_err = e
            backoff = min(30.0, (2**attempt) * sleep_sec)
            print(
                f"[warn] {symbol} {year} attempt {attempt+1}/{max_retries} failed: {e} ; sleep {backoff:.2f}s"
            )
            time.sleep(backoff)

    raise RuntimeError(
        f"Failed to fetch {symbol} {year} after {max_retries} retries. Last error: {last_err}"
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default="data/trading/raw_data")
    p.add_argument("--start_year", type=int, default=2023)
    p.add_argument("--end_year", type=int, default=2025)
    p.add_argument(
        "--feed", type=str, default="iex", choices=["iex", "sip", "delayed_sip"]
    )
    p.add_argument(
        "--adjustment",
        type=str,
        default="split",
        choices=["raw", "split", "dividend", "all"],
    )
    p.add_argument("--max_retries", type=int, default=6)
    p.add_argument("--sleep_sec", type=float, default=0.25)
    p.add_argument("--skip_existing", action="store_true", default=True)

    return p.parse_args()


def main():
    args = parse_args()

    api_key = API_KEY
    api_secret = API_SECRET
    if not api_key or not api_secret:
        raise RuntimeError("Set env vars ALPACA_API_KEY and ALPACA_API_SECRET")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = StockHistoricalDataClient(api_key, api_secret)
    feed = _feed_enum(args.feed)
    adj = _adj_enum(args.adjustment)

    years = list(range(args.start_year, args.end_year + 1))

    for sym in TICKERS:
        sym_dir = out_dir / sym
        sym_dir.mkdir(parents=True, exist_ok=True)

        for y in years:
            out_fp = sym_dir / f"{y}.parquet"
            if args.skip_existing and out_fp.exists():
                print(f"[skip] exists: {out_fp}")
                continue

            print(f"[dl] {sym} {y} feed={args.feed} adj={args.adjustment}")
            df = fetch_bars_one_symbol_one_year(
                client=client,
                symbol=sym,
                year=y,
                feed=feed,
                adjustment=adj,
                max_retries=args.max_retries,
                sleep_sec=args.sleep_sec,
            )
            df.to_parquet(out_fp, index=False)
            print(f"[ok] wrote: {out_fp} rows={len(df)}")

            time.sleep(args.sleep_sec)


if __name__ == "__main__":
    main()
