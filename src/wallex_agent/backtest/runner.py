"""Event-driven backtest that reuses the exact live decision core (`Agent.step`).

Data: cached candles (data/candles/SYMBOL_RES.csv) downloaded with `wallex-agent fetch-data`.
Fills: SimulatedBroker at the bar close (+slippage, fees). AI: off | replay | live.
Output: metrics JSON + equity curve CSV + the usual decision log, under logs/backtests/<run>/.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from wallex_agent.ai.analyst import Analyst
from wallex_agent.ai.llm import LLMClient, LLMUnavailable
from wallex_agent.data.features import compute_indicators, snapshot_from
from wallex_agent.data.market_data import CandleStore
from wallex_agent.decision_log.logger import DecisionLog
from wallex_agent.engine.core import Agent
from wallex_agent.execution.broker import SimulatedBroker
from wallex_agent.portfolio.state import Portfolio
from wallex_agent.risk.manager import RiskManager
from wallex_agent.wallex.client import MarketInfo

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    metrics: Dict
    equity: pd.Series
    trades: List[Dict]
    out_dir: Path


def _fake_market(sym: str, price: float) -> MarketInfo:
    return MarketInfo(symbol=sym, base=sym.replace("TMN", ""), quote="TMN", price_precision=0, qty_precision=6,
                      min_qty=0.0, min_notional=0.0, bid=price * 0.9995, ask=price * 1.0005, last=price,
                      quote_volume_24h=1e12, change_24h_pct=0.0, raw={})


def run_backtest(s, symbols: List[str], start: Optional[str], end: Optional[str], ai_mode: str = "off",
                 resolution: Optional[str] = None, out_root: str = "logs/backtests", frames: Optional[Dict[str, pd.DataFrame]] = None) -> BacktestResult:
    res = str(resolution or s.data.resolution)
    store = CandleStore(s.data.candle_cache_dir)
    data: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = frames[sym] if frames and sym in frames else store.load(sym, res)
        if start:
            df = df[df.index >= pd.Timestamp(start, tz="UTC")]
        if end:
            df = df[df.index <= pd.Timestamp(end, tz="UTC")]
        if len(df) < s.strategy.trend_ema + 5:
            log.warning("%s: only %d bars, skipping", sym, len(df)); continue
        data[sym] = compute_indicators(df, s.strategy)
    if not data:
        raise SystemExit("no candle data; run `wallex-agent fetch-data` first")

    run_id = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    out_dir = Path(out_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pf = Portfolio(cash_tmn=s.challenge.initial_capital_tmn, initial_capital=s.challenge.initial_capital_tmn)
    analyst = None
    if ai_mode != "off" and s.strategy.ai_enabled:
        try:
            analyst = Analyst(LLMClient(s.ai_base_url, s.ai_api_key, s.ai_model_fast, s.ai_model_deep, s.execution.ai_cache_dir, mode=ai_mode),
                              s.strategy.ai_min_confidence)
        except LLMUnavailable as e:
            log.warning("AI unavailable in backtest: %s", e)
    # the backtest's "challenge end" is the last bar: forces liquidation -> realized TMN result
    idx = sorted(set().union(*[set(d.index) for d in data.values()]))
    end_dt = idx[-1].to_pydatetime()
    risk = RiskManager(s.risk, end_dt.astimezone(timezone.utc), liquidate_before_min=0)
    dlog = DecisionLog(str(out_dir), run_id="decisions", mode="backtest")
    fee = s.risk.maker_fee_pct if s.execution.order_style == "maker" else s.risk.fee_pct
    agent = Agent(s, pf, risk, analyst, SimulatedBroker(fee, s.risk.slippage_pct), dlog)

    warm = s.strategy.trend_ema
    eq_pts = []
    for ts in idx[warm:]:
        now = ts.timestamp()
        snaps, prices, markets, recent, frames = {}, {}, {}, {}, {}
        for sym, df in data.items():
            if ts not in df.index:
                continue
            i = df.index.get_loc(ts)
            sub = df.iloc[: i + 1]
            frames[sym] = sub
            snap = snapshot_from(sub, sym, spread_pct=0.1)
            if snap is None:
                continue
            px = float(sub["close"].iloc[-1])
            snaps[sym], prices[sym], markets[sym] = snap, px, _fake_market(sym, px)
            if analyst:
                recent[sym] = [{"t": str(j)[:16], "o": float(r.open), "h": float(r.high), "l": float(r.low), "c": float(r.close), "v": float(r.volume)}
                               for j, r in sub.tail(24).iterrows()]
        if not prices:
            continue
        agent.step(now, snaps, prices, markets, recent, frames=frames)
        eq_pts.append((ts, pf.equity(prices)))
    # force flat at the end (challenge rule: realized in TMN)
    if pf.positions:
        last_prices = {sym: float(df["close"].iloc[-1]) for sym, df in data.items()}
        agent.liquidate_all(last_prices, {sym: _fake_market(sym, p) for sym, p in last_prices.items()}, idx[-1].timestamp(), "end of backtest")
        eq_pts.append((idx[-1], pf.cash_tmn))

    eq = pd.Series([e for _, e in eq_pts], index=pd.DatetimeIndex([t for t, _ in eq_pts]), name="equity")
    trades = [json.loads(l) for l in open(dlog.path, encoding="utf-8") if '"kind": "exit"' in l]
    metrics = compute_metrics(eq, trades, pf, res)
    metrics.update({"symbols": list(data), "bars": len(idx), "ai_mode": ai_mode, "resolution": res,
                    "ai_calls": analyst.llm.calls if analyst else 0, "ai_cost_usd": round(analyst.llm.total_cost_usd, 4) if analyst else 0})
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    eq.to_csv(out_dir / "equity.csv", index_label="time")
    pd.DataFrame(trades).to_csv(out_dir / "trades.csv", index=False) if trades else None
    return BacktestResult(metrics, eq, trades, out_dir)


def compute_metrics(eq: pd.Series, trades: List[Dict], pf: Portfolio, resolution: str) -> Dict:
    if eq.empty:
        return {"error": "no equity points"}
    ret = (eq.iloc[-1] / pf.initial_capital - 1) * 100
    peak = eq.cummax()
    mdd = ((peak - eq) / peak * 100).max()
    r = eq.pct_change().dropna()
    bars_per_year = {"1": 525600, "5": 105120, "15": 35040, "30": 17520, "60": 8760, "180": 2920, "360": 1460, "720": 730, "D": 365}.get(resolution, 8760)
    sharpe = float(r.mean() / r.std(ddof=0) * np.sqrt(bars_per_year)) if len(r) > 2 and r.std(ddof=0) > 0 else 0.0
    pnls = [t["pnl_tmn"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    return {
        "initial_capital_tmn": pf.initial_capital, "final_equity_tmn": round(float(eq.iloc[-1])),
        "net_return_pct": round(float(ret), 3), "max_drawdown_pct": round(float(mdd), 3), "sharpe_annualized": round(sharpe, 2),
        "trades": len(trades), "win_rate_pct": round(len(wins) / len(trades) * 100, 1) if trades else 0.0,
        "avg_win_tmn": round(float(np.mean(wins))) if wins else 0, "avg_loss_tmn": round(float(np.mean(losses))) if losses else 0,
        "profit_factor": round(sum(wins) / abs(sum(losses)), 2) if losses and sum(losses) != 0 else (999.0 if wins else 0.0),
        "fees_paid_tmn": round(pf.fees_paid_tmn), "avg_holding_minutes": round(float(np.mean([t["held_minutes"] for t in trades]))) if trades else 0,
        "exit_reasons": pd.Series([t["reason"].split(" (")[0] for t in trades]).value_counts().to_dict() if trades else {},
        "start": str(eq.index[0]), "end": str(eq.index[-1]),
    }
