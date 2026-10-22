"""Deterministic signal layer.

Strategy: *AI-gated mean reversion with trend filter* (long-only, spot).
  ENTRY candidate when (all computed on closed candles):
    - RSI(14) below `rsi_oversold`                                  (oversold)
    - price near/below the lower Bollinger band (bb_pos <= 0.15)    (statistical stretch)
    - NOT in a strong downtrend: not (close < EMA_trend and EMA_fast < EMA_slow and ret_24 < -4%)
    - market alive: volume z-score >= volume_zscore_min
  Composite score in [0,1] weights these; only candidates with score >= min_signal_score
  are passed to the AI analyst, which may approve/reject/resize (never widen risk).

  EXIT (handled by the risk manager each loop): stop-loss, take-profit, trailing stop,
  time stop, RSI mean reverted above `rsi_exit`, or challenge-end liquidation.
"""
from __future__ import annotations

from typing import Optional

from wallex_agent.data.features import Snapshot


from wallex_agent.strategy.base import Signal  # noqa: E402  (shared dataclass)


def _clip(x: float, lo=0.0, hi=1.0) -> float:
    return max(lo, min(hi, x))


def evaluate_entry(s: Snapshot, cfg) -> Signal:
    reasons, vetoes = [], []
    # 1) oversold strength (RSI 32 -> 0, RSI 15 -> 1)
    rsi_score = _clip((cfg.rsi_oversold - s.rsi) / 17.0 + 0.0) if s.rsi <= cfg.rsi_oversold else 0.0
    if s.rsi <= cfg.rsi_oversold:
        reasons.append(f"RSI {s.rsi:.1f} <= {cfg.rsi_oversold} (oversold)")
    # 2) distance below lower band (bb_pos 0.15 -> 0, -0.25 -> 1)
    bb_score = _clip((0.15 - s.bb_pos) / 0.4) if s.bb_pos <= 0.15 else 0.0
    if s.bb_pos <= 0.15:
        reasons.append(f"price at/below lower Bollinger band (bb_pos={s.bb_pos:.2f})")
    # 3) trend filter
    strong_down = s.close < s.ema_trend and s.ema_fast < s.ema_slow and s.ret_24 < -4.0
    if strong_down:
        vetoes.append(f"strong downtrend: close<EMA{cfg.trend_ema}, EMA{cfg.ema_fast}<EMA{cfg.ema_slow}, 24h {s.ret_24:.1f}%")
    trend_score = 1.0 if s.close >= s.ema_trend else (0.6 if s.ema_fast >= s.ema_slow else 0.3)
    # 4) liquidity / activity
    if s.vol_z < cfg.volume_zscore_min:
        vetoes.append(f"volume z-score {s.vol_z:.2f} < {cfg.volume_zscore_min} (dead market)")
    if s.spread_pct is not None and s.spread_pct > getattr(cfg, "max_spread_pct", 0.35):
        vetoes.append(f"spread {s.spread_pct:.2f}% too wide")
    # 5) volatility sanity: extreme vol -> smaller score (risk engine also shrinks size)
    vol_pen = 0.0 if s.atr_pct < 1.5 else _clip((s.atr_pct - 1.5) / 3.0) * 0.3
    # 6) order-book confirmation (optional)
    ob = 0.5 if s.orderbook_imbalance is None else _clip(0.5 + s.orderbook_imbalance / 2)
    if s.orderbook_imbalance is not None and s.orderbook_imbalance > 0.2:
        reasons.append(f"bid-heavy order book (imbalance {s.orderbook_imbalance:+.2f})")

    core = 0.45 * rsi_score + 0.35 * bb_score + 0.20 * (trend_score * 0.5 + ob * 0.5)
    score = _clip(core - vol_pen) if (rsi_score > 0 or bb_score > 0) else 0.0
    if rsi_score == 0 and bb_score == 0:
        vetoes.append("no oversold condition")
    return Signal(symbol=s.symbol, side="BUY", score=round(score, 4), reasons=reasons, vetoes=vetoes,
                  snapshot=s.to_dict())


def evaluate_exit(s: Snapshot, cfg) -> Optional[str]:
    """Rule-based exit hint (independent of SL/TP which the risk manager owns)."""
    if s.rsi >= cfg.rsi_exit and s.bb_pos >= 0.5:
        return f"mean reverted: RSI {s.rsi:.1f} >= {cfg.rsi_exit} and bb_pos {s.bb_pos:.2f}"
    return None
