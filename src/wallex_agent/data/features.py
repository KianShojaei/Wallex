"""Technical features computed from OHLCV. Pure pandas/numpy, deterministic."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np
import pandas as pd


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0).where(dn != 0, 100.0)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = close.rolling(n, min_periods=n).mean()
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    return mid - k * sd, mid, mid + k * sd


def compute_indicators(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Append indicator columns. `cfg` is StrategyCfg (duck-typed)."""
    out = df.copy()
    c = out["close"]
    out["ema_fast"] = ema(c, cfg.ema_fast)
    out["ema_slow"] = ema(c, cfg.ema_slow)
    out["ema_trend"] = ema(c, cfg.trend_ema)
    out["rsi"] = rsi(c, cfg.rsi_period)
    lo, mid, hi = bollinger(c, cfg.bb_period, cfg.bb_std)
    out["bb_lo"], out["bb_mid"], out["bb_hi"] = lo, mid, hi
    width = (hi - lo).replace(0, np.nan)
    out["bb_pos"] = ((c - lo) / width).clip(-0.5, 1.5)          # 0 = lower band, 1 = upper band
    out["atr"] = atr(out, cfg.atr_period)
    out["atr_pct"] = out["atr"] / c * 100
    out["ret_1"] = c.pct_change() * 100
    out["ret_6"] = c.pct_change(6) * 100
    out["ret_24"] = c.pct_change(24) * 100
    v = out["volume"]
    out["vol_z"] = (v - v.rolling(48, min_periods=10).mean()) / v.rolling(48, min_periods=10).std(ddof=0).replace(0, np.nan)
    out["vol_z"] = out["vol_z"].fillna(0.0)
    out["realized_vol_24"] = out["ret_1"].rolling(24, min_periods=12).std(ddof=0)
    return out


@dataclass
class Snapshot:
    """Compact, LLM-friendly description of the latest bar."""
    symbol: str
    time: str
    close: float
    rsi: float
    bb_pos: float
    ema_fast: float
    ema_slow: float
    ema_trend: float
    atr_pct: float
    ret_1: float
    ret_6: float
    ret_24: float
    vol_z: float
    realized_vol_24: float
    spread_pct: Optional[float] = None
    change_24h_pct: Optional[float] = None
    quote_volume_24h: Optional[float] = None
    orderbook_imbalance: Optional[float] = None   # (bid_vol - ask_vol)/(bid_vol+ask_vol) in top levels

    def to_dict(self) -> Dict:
        d = asdict(self)
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in d.items()}


def snapshot_from(df: pd.DataFrame, symbol: str, **extra) -> Optional[Snapshot]:
    if df.empty or df["rsi"].isna().iloc[-1] or df["bb_pos"].isna().iloc[-1]:
        return None
    r = df.iloc[-1]
    return Snapshot(symbol=symbol, time=df.index[-1].isoformat(), close=float(r.close), rsi=float(r.rsi),
                    bb_pos=float(r.bb_pos), ema_fast=float(r.ema_fast), ema_slow=float(r.ema_slow),
                    ema_trend=float(r.ema_trend) if not np.isnan(r.ema_trend) else float(r.ema_slow),
                    atr_pct=float(r.atr_pct), ret_1=float(r.ret_1), ret_6=float(r.ret_6),
                    ret_24=float(r.ret_24) if not np.isnan(r.ret_24) else 0.0, vol_z=float(r.vol_z),
                    realized_vol_24=float(r.realized_vol_24) if not np.isnan(r.realized_vol_24) else 0.0, **extra)


def orderbook_imbalance(depth: dict, levels: int = 10) -> Optional[float]:
    bids = sum(x["quantity"] * x["price"] for x in depth.get("bid", [])[:levels])
    asks = sum(x["quantity"] * x["price"] for x in depth.get("ask", [])[:levels])
    tot = bids + asks
    return (bids - asks) / tot if tot else None
