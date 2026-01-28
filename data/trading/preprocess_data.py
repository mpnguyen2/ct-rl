# data/trading/preprocess_data.py
"""
Preprocess Alpaca parquet minute bars into close-only lag features.

Outputs:
  {out_dir}/train.npz
  {out_dir}/eval.npz

NPZ contains:
  timestamps: [T'] datetime64[ns] (UTC)
  tickers: [A]
  features: [A, T', 23] float32
    features[:, :, 0]  = current close/PRICE_SCALE
    features[:, :, j]  = close at lag LAGS[j] / PRICE_SCALE
  lags: [23] int32
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    TICKERS,
    NY_TZ,
    SESSION_MIN_PER_DAY,
    SESSION_START_HHMM,
    PRICE_SCALE,
    LAGS,
    MAX_LAG,
)


def parse_quarter(label: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    m = re.match(r"^Q([1-4])_(\d{4})$", label)
    if not m:
        raise ValueError(f"Invalid quarter label: {label}")
    q, y = int(m.group(1)), int(m.group(2))
    start_month = 3 * (q - 1) + 1
    start = pd.Timestamp(year=y, month=start_month, day=1, tz="UTC")
    if start_month == 10:
        end = pd.Timestamp(year=y + 1, month=1, day=1, tz="UTC")
    else:
        end = pd.Timestamp(year=y, month=start_month + 3, day=1, tz="UTC")
    return start, end


def parse_month(label: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    m = re.match(r"^(\d{1,2})_(\d{4})$", label)
    if not m:
        raise ValueError(f"Invalid month label: {label}")
    month, year = int(m.group(1)), int(m.group(2))
    start = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
    if month == 12:
        end = pd.Timestamp(year=year + 1, month=1, day=1, tz="UTC")
    else:
        end = pd.Timestamp(year=year, month=month + 1, day=1, tz="UTC")
    return start, end


def parse_range(range_str: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    # Try YYYY-YYYY
    m_year = re.match(r"^(\d{4})-(\d{4})$", range_str)
    if m_year:
        start_year = int(m_year.group(1))
        end_year = int(m_year.group(2))
        start = pd.Timestamp(year=start_year, month=1, day=1, tz="UTC")
        end = pd.Timestamp(year=end_year + 1, month=1, day=1, tz="UTC")
        return start, end

    # Try Qx_YYYY-Qy_YYYY
    parts = range_str.split("-")
    if len(parts) == 2:
        if parts[0].startswith("Q") and parts[1].startswith("Q"):
            s_start, _ = parse_quarter(parts[0])
            _, e_end = parse_quarter(parts[1])
            return s_start, e_end
        # Try MM_YYYY-MM_YYYY
        if "_" in parts[0] and "_" in parts[1]:
            s_start, _ = parse_month(parts[0])
            _, e_end = parse_month(parts[1])
            return s_start, e_end

    # Handle single period (no "-")
    if len(parts) == 1:
        if range_str.startswith("Q"):
            return parse_quarter(range_str)
        elif "_" in range_str:
            return parse_month(range_str)
        elif re.match(r"^\d{4}$", range_str):
            y = int(range_str)
            return pd.Timestamp(year=y, month=1, day=1, tz="UTC"), pd.Timestamp(
                year=y + 1, month=1, day=1, tz="UTC"
            )

    raise ValueError(
        f"Unknown range format: {range_str}. Use 'YYYY-YYYY' or 'Qx_YYYY-Qy_YYYY'."
    )


def load_symbol_years(raw_dir, symbol, start_year, end_year, ts_col, close_col):
    parts = []
    sym_dir = raw_dir / symbol
    for y in range(start_year, end_year + 1):
        fp = sym_dir / f"{y}.parquet"
        if not fp.exists():
            raise FileNotFoundError(fp)
        df = pd.read_parquet(fp, columns=[ts_col, close_col])
        parts.append(df)

    df = pd.concat(parts, axis=0, ignore_index=True)
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    df[close_col] = pd.to_numeric(df[close_col], errors="coerce")
    df = df.sort_values(ts_col, kind="mergesort").reset_index(drop=True)
    return df


def filter_regular_session(df, ts_col):
    # Keep only 09:30 <= time < 16:00 in NY time
    ts_ny = df[ts_col].dt.tz_convert(NY_TZ)
    hh, mm = SESSION_START_HHMM
    t0 = (hh, mm)

    # compute minutes since midnight for filtering
    mins = ts_ny.dt.hour * 60 + ts_ny.dt.minute
    start_m = t0[0] * 60 + t0[1]
    end_m = start_m + SESSION_MIN_PER_DAY  # 390 minutes window

    mask = (mins >= start_m) & (mins < end_m)
    return df.loc[mask].copy()


def build_daily_grid_utc(trading_dates_ny):
    # For each NY date, generate 390 minutes: 09:30..15:59 ET, convert to UTC
    if not trading_dates_ny:
        return pd.DatetimeIndex([], tz="UTC")

    grids = []
    hh, mm = SESSION_START_HHMM
    for d in trading_dates_ny:
        start = pd.Timestamp(d.date()).tz_localize(NY_TZ) + pd.Timedelta(
            hours=hh, minutes=mm
        )
        day_idx = pd.date_range(
            start=start, periods=SESSION_MIN_PER_DAY, freq="min", tz=NY_TZ
        )
        grids.append(day_idx.tz_convert("UTC"))

    if not grids:
        return pd.DatetimeIndex([], tz="UTC")

    return pd.DatetimeIndex(np.concatenate([g.values for g in grids]))


def interpolate_series_to_daily_grid(s: pd.Series, trading_dates_ny):
    """
    Interpolate from original timestamps -> fixed per-day minute grid (390 pts/day).
    Does NOT require exact timestamp matches like reindex does.

    Returns:
      v: shape [D*390] float32 (NaNs for days with zero points)
    """
    # Ensure UTC tz-aware index
    idx_utc = pd.to_datetime(s.index, utc=True)
    s = pd.Series(s.to_numpy(dtype=np.float32, copy=False), index=idx_utc, name=s.name)
    s = s[~s.index.duplicated(keep="last")].sort_index()

    # Convert to NY to compute day + minute offset
    idx_ny = s.index.tz_convert(NY_TZ)

    hh, mm = SESSION_START_HHMM
    start_m = hh * 60 + mm

    # Key by calendar date (no tz headaches)
    s_date = pd.Series(idx_ny.date, index=s.index)  # date object per point
    minutes = (
        idx_ny.hour * 60 + idx_ny.minute
    ) - start_m  # integer minute bin (0..389)
    minutes = minutes.astype(np.int32)

    D = len(trading_dates_ny)
    out = np.full((D, SESSION_MIN_PER_DAY), np.nan, dtype=np.float32)

    # Map trading date -> row index
    trading_date_keys = [pd.Timestamp(d).date() for d in trading_dates_ny]
    date_to_row = {dk: i for i, dk in enumerate(trading_date_keys)}

    # Build a small table for grouping
    df_tmp = pd.DataFrame(
        {"date": np.array(s_date.values, dtype=object), "m": minutes, "y": s.values},
        copy=False,
    )

    # Keep only in-session minute bins
    df_tmp = df_tmp[(df_tmp["m"] >= 0) & (df_tmp["m"] < SESSION_MIN_PER_DAY)]

    # If your timestamps are not minute-aligned (seconds), this still bins by minute.
    # If you suspect "minute-end labeled" bars (first bin is 1 not 0), you can shift:
    # df_tmp["m"] = df_tmp["m"] - 1

    # Sort so "last" wins on duplicate (date, minute)
    df_tmp = df_tmp.sort_values(["date", "m"], kind="mergesort")
    df_tmp = df_tmp.drop_duplicates(subset=["date", "m"], keep="last")

    for dk, g in df_tmp.groupby("date", sort=False):
        row = date_to_row.get(dk, None)
        if row is None:
            continue

        x = g["m"].to_numpy(dtype=np.float32, copy=False)
        y = g["y"].to_numpy(dtype=np.float32, copy=False)

        if x.size == 0:
            continue

        x_grid = np.arange(SESSION_MIN_PER_DAY, dtype=np.float32)

        if x.size == 1:
            # Only one point in the day: fill day constant (or keep NaN if you prefer)
            out[row, :] = y[0]
        else:
            # True interpolation within the day; no cross-day interpolation
            out[row, :] = np.interp(x_grid, x, y).astype(np.float32, copy=False)

    return out.reshape(-1)


def make_close_features(close_AT):
    # close_AT: [A, T] close/PRICE_SCALE after imputation on full grid
    A, T = close_AT.shape
    L = T - MAX_LAG
    if L <= 0:
        raise ValueError(f"Not enough minutes: T={T}, need > {MAX_LAG}")

    feats = np.stack(
        [close_AT[:, (MAX_LAG - lag) : (T - lag)] for lag in LAGS], axis=-1
    ).astype(np.float32, copy=False)
    return feats


def build_split(raw_dir, out_dir, name, start_ts, end_ts, ts_col, close_col):
    # Load each symbol, filter regular session, collect NY trading dates
    sym_series = []
    all_dates = set()

    y_start = start_ts.year
    y_end = (end_ts - pd.Timedelta(seconds=1)).year

    for sym in TICKERS:
        df = load_symbol_years(raw_dir, sym, y_start, y_end, ts_col, close_col)
        df = df[(df[ts_col] >= start_ts) & (df[ts_col] < end_ts)].copy()
        df = filter_regular_session(df, ts_col)

        ts_ny = df[ts_col].dt.tz_convert(NY_TZ)
        dates_ny = pd.DatetimeIndex(ts_ny.dt.normalize().unique())
        for d in dates_ny:
            all_dates.add(pd.Timestamp(d))

        s = pd.Series(
            df[close_col].to_numpy(dtype=np.float32), index=df[ts_col], name=sym
        )
        s = s[~s.index.duplicated(keep="last")].sort_index()
        sym_series.append(s)

    trading_dates_ny = sorted(all_dates)
    if not trading_dates_ny:
        raise RuntimeError(f"No trading dates found for split {name}")

    # Build full minute grid
    grid_utc = build_daily_grid_utc(trading_dates_ny)  # tz-aware UTC
    D = len(trading_dates_ny)
    assert len(grid_utc) == D * SESSION_MIN_PER_DAY

    # Interpolate from original timestamps -> fixed daily grid
    close_list = []

    for s in sym_series:
        s = s.ffill().bfill()
        v = interpolate_series_to_daily_grid(s, trading_dates_ny)  # [D*390]

        # quick check
        nan_frac = float(np.isnan(v).mean()) if v.size > 0 else 1.0
        if nan_frac > 0.01:
            print(
                f"[{name}] warn: {s.name} NaN_frac after interp_to_grid = {nan_frac:.2%}"
            )

        close_list.append(v)

    close_TA = np.stack(close_list, axis=1)  # [T, A]
    close_AT = (close_TA.T) / float(PRICE_SCALE)  # [A, T]

    # Drop days with any ticker fully missing
    close_ADM = close_AT.reshape(len(TICKERS), D, SESSION_MIN_PER_DAY)  # [A, D, 390]
    day_allnan = np.isnan(close_ADM).all(axis=2)  # [A, D]
    bad_day = day_allnan.any(axis=0)  # [D]

    if bad_day.any():
        keep_days = ~bad_day
        print(
            f"[{name}] dropping {int(bad_day.sum())} days with full-day missing for >=1 ticker"
        )

        kept_dates = [d for d, k in zip(trading_dates_ny, keep_days) if k]
        grid_utc = build_daily_grid_utc(kept_dates)
        D = len(kept_dates)
        close_ADM = close_ADM[:, keep_days, :]

    close_AT = close_ADM.reshape(len(TICKERS), D * SESSION_MIN_PER_DAY)

    # Build lagged-close features and timestamps (drop first MAX_LAG minutes)
    features = make_close_features(close_AT)  # [A, T', 23]
    timestamps_out = grid_utc.to_numpy(dtype="datetime64[ns]")[MAX_LAG:]

    nan_frac = float(np.isnan(features).mean())
    print(
        f"[{name}] features={features.shape} timestamps={timestamps_out.shape} NaN_frac={nan_frac:.10f}"
    )
    if nan_frac > 0.0:
        raise RuntimeError(f"[{name}] NaNs remain after imputation. Check gaps.")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_fp = out_dir / f"{name}.npz"
    np.savez_compressed(
        out_fp,
        timestamps=timestamps_out,
        tickers=np.array(TICKERS, dtype=object),
        features=features,
        lags=np.array(LAGS, dtype=np.int32),
    )
    print(f"[{name}] wrote {out_fp}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", type=str, default="data/trading/raw_data")
    p.add_argument("--out_dir", type=str, default="data/trading/processed_data")
    p.add_argument("--train_range", type=str, default="Q3_2023-Q2_2025")
    p.add_argument("--eval_range", type=str, default="Q3_2025")
    p.add_argument("--timestamp_col", type=str, default="timestamp")
    p.add_argument("--close_col", type=str, default="close")
    return p.parse_args()


def main():
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    train_start, train_end = parse_range(args.train_range)
    eval_start, eval_end = parse_range(args.eval_range)

    build_split(
        raw_dir,
        out_dir,
        "train",
        train_start,
        train_end,
        args.timestamp_col,
        args.close_col,
    )
    build_split(
        raw_dir,
        out_dir,
        "eval",
        eval_start,
        eval_end,
        args.timestamp_col,
        args.close_col,
    )


if __name__ == "__main__":
    main()
