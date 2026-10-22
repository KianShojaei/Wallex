"""Candle fetching, normalisation and on-disk caching (CSV) for reproducible backtests."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from wallex_agent.wallex.client import WallexClient, WallexError

log = logging.getLogger(__name__)

RESOLUTION_SECONDS = {"1": 60, "5": 300, "15": 900, "30": 1800, "60": 3600, "180": 10800,
                      "360": 21600, "720": 43200, "D": 86400, "1D": 86400}


def resolution_seconds(res: str) -> int:
    return RESOLUTION_SECONDS[str(res)]


def udf_to_df(payload: dict) -> pd.DataFrame:
    if not payload or payload.get("s") not in (None, "ok") or not payload.get("t"):
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"],
                            index=pd.DatetimeIndex([], name="time", tz="UTC"))
    df = pd.DataFrame({
        "time": pd.to_datetime(payload["t"], unit="s", utc=True),
        "open": pd.to_numeric(payload["o"], errors="coerce"),
        "high": pd.to_numeric(payload["h"], errors="coerce"),
        "low": pd.to_numeric(payload["l"], errors="coerce"),
        "close": pd.to_numeric(payload["c"], errors="coerce"),
        "volume": pd.to_numeric(payload.get("v", [0] * len(payload["t"])), errors="coerce"),
    }).dropna(subset=["close"]).set_index("time").sort_index()
    return df[~df.index.duplicated(keep="last")]


def resample_ohlcv(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    if df.empty:
        return df
    return df.resample(f"{minutes}min").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()


def ensure_resolution(df: pd.DataFrame, resolution: str) -> pd.DataFrame:
    """Wallex UDF may answer a coarse request with finer bars (e.g. '15' -> 1-minute); normalise."""
    if len(df) < 3 or str(resolution) == "1":
        return df
    want = resolution_seconds(resolution)
    spacing = (df.index.to_series().diff().dt.total_seconds()).median()
    if spacing and spacing < want * 0.9:
        return resample_ohlcv(df, want // 60)
    return df


class CandleStore:
    def __init__(self, cache_dir: str = "data/candles"):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, symbol: str, resolution: str) -> Path:
        return self.dir / f"{symbol}_{resolution}.csv"

    def load(self, symbol: str, resolution: str) -> pd.DataFrame:
        p = self.path(symbol, resolution)
        if not p.exists():
            return udf_to_df({})
        df = pd.read_csv(p, parse_dates=["time"], index_col="time")
        df.index = pd.to_datetime(df.index, utc=True)
        return df.sort_index()

    def save(self, symbol: str, resolution: str, df: pd.DataFrame) -> None:
        df.sort_index().to_csv(self.path(symbol, resolution), index_label="time")

    def merge(self, symbol: str, resolution: str, new: pd.DataFrame) -> pd.DataFrame:
        old = self.load(symbol, resolution)
        df = pd.concat([old, new]) if len(old) else new
        df = df[~df.index.duplicated(keep="last")].sort_index()
        self.save(symbol, resolution, df)
        return df


class MarketData:
    """Fetches candles from Wallex with a fallback resolution and keeps the local cache fresh."""

    def __init__(self, client: WallexClient, store: CandleStore, resolution: str = "60",
                 fallback_resolution: str = "60", lookback_bars: int = 300):
        self.client = client
        self.store = store
        self.resolution = str(resolution)
        self.fallback = str(fallback_resolution)
        self.lookback = lookback_bars

    def fetch(self, symbol: str, resolution: Optional[str] = None, bars: Optional[int] = None,
              end_ts: Optional[int] = None) -> pd.DataFrame:
        res = str(resolution or self.resolution)
        bars = bars or self.lookback
        end_ts = int(end_ts or time.time())
        start_ts = end_ts - bars * resolution_seconds(res)
        try:
            df = ensure_resolution(udf_to_df(self.client.candles(symbol, res, start_ts, end_ts)), res)
        except WallexError as e:
            log.warning("candles %s@%s failed: %s", symbol, res, e)
            df = udf_to_df({})
        if df.empty and res != self.fallback:
            log.info("falling back to resolution %s for %s", self.fallback, symbol)
            return self.fetch(symbol, self.fallback, bars, end_ts)
        if not df.empty:
            self.store.merge(symbol, res, df)
        return df

    def history(self, symbol: str, resolution: str, start_ts: int, end_ts: int, chunk_bars: int = 1000) -> pd.DataFrame:
        """Download a long range in chunks (for backtests) and cache it."""
        res = str(resolution)
        step = chunk_bars * resolution_seconds(res)
        cur = start_ts
        frames = []
        while cur < end_ts:
            nxt = min(cur + step, end_ts)
            try:
                frames.append(udf_to_df(self.client.candles(symbol, res, cur, nxt)))
            except WallexError as e:
                log.warning("history chunk %s %s-%s failed: %s", symbol, cur, nxt, e)
            cur = nxt
        df = pd.concat(frames) if frames else udf_to_df({})
        df = df[~df.index.duplicated(keep="last")].sort_index() if len(df) else df
        df = ensure_resolution(df, res)
        if len(df):
            self.store.merge(symbol, res, df)
        return df
