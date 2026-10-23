"""Portfolio state: positions, cash, equity curve, realized PnL. Persisted as JSON so a crash/restart resumes."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Position:
    symbol: str
    qty: float
    entry_price: float
    entry_time: float                 # unix seconds
    stop_loss: float
    take_profit: float
    trailing_pct: float
    highest: float
    entry_fee_tmn: float = 0.0
    entry_order_id: str = ""
    decision_id: str = ""
    size_factor: float = 1.0
    max_holding_minutes: int = 0
    module: str = ""
    last_add_time: float = 0.0
    adds: int = 0

    @property
    def cost_tmn(self) -> float:
        return self.qty * self.entry_price

    @property
    def paid_tmn(self) -> float:
        """TMN actually spent: notional + in-kind buy fee valued at entry (fee was deducted from qty, not cash)."""
        return self.qty * self.entry_price + self.entry_fee_tmn

    def unrealized_pct(self, price: float) -> float:
        return (price / self.entry_price - 1) * 100


@dataclass
class Portfolio:
    cash_tmn: float
    initial_capital: float
    positions: Dict[str, Position] = field(default_factory=dict)
    realized_pnl_tmn: float = 0.0
    fees_paid_tmn: float = 0.0
    trades: int = 0
    wins: int = 0
    equity_curve: List[List[float]] = field(default_factory=list)   # [ts, equity]
    peak_equity: float = 0.0
    day_start_equity: float = 0.0
    day_key: str = ""
    halted_until_day: str = ""           # daily halt marker
    killed: bool = False
    last_loss_time: Dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------ valuation
    def equity(self, prices: Dict[str, float]) -> float:
        eq = self.cash_tmn
        for p in self.positions.values():
            eq += p.qty * prices.get(p.symbol, p.entry_price)
        return eq

    def exposure_tmn(self, prices: Dict[str, float]) -> float:
        return sum(p.qty * prices.get(p.symbol, p.entry_price) for p in self.positions.values())

    def mark(self, prices: Dict[str, float], now: Optional[float] = None) -> float:
        now = now or time.time()
        eq = self.equity(prices)
        self.equity_curve.append([round(now), round(eq, 2)])
        if len(self.equity_curve) > 20000:
            self.equity_curve = self.equity_curve[-20000:]
        self.peak_equity = max(self.peak_equity or eq, eq)
        dk = time.strftime("%Y-%m-%d", time.gmtime(now))
        if dk != self.day_key:
            self.day_key, self.day_start_equity = dk, eq
        return eq

    def drawdown_pct(self, eq: float) -> float:
        return (self.peak_equity - eq) / self.peak_equity * 100 if self.peak_equity else 0.0

    def daily_drawdown_pct(self, eq: float) -> float:
        return (self.day_start_equity - eq) / self.day_start_equity * 100 if self.day_start_equity else 0.0

    def summary(self, prices: Dict[str, float]) -> Dict:
        eq = self.equity(prices)
        return {"equity_tmn": round(eq), "cash_tmn": round(self.cash_tmn), "exposure_tmn": round(self.exposure_tmn(prices)),
                "return_pct": round((eq / self.initial_capital - 1) * 100, 3), "realized_pnl_tmn": round(self.realized_pnl_tmn),
                "fees_paid_tmn": round(self.fees_paid_tmn), "trades": self.trades, "wins": self.wins,
                "open_positions": [{"symbol": p.symbol, "qty": p.qty, "entry": p.entry_price,
                                    "upnl_pct": round(p.unrealized_pct(prices.get(p.symbol, p.entry_price)), 3)}
                                   for p in self.positions.values()],
                "max_drawdown_pct": round(self.drawdown_pct(eq), 3), "killed": self.killed}

    # ----------------------------------------------------------- persist
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        d = asdict(self)
        d["positions"] = {k: asdict(v) for k, v in self.positions.items()}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "Portfolio":
        d = json.loads(path.read_text(encoding="utf-8"))
        d["positions"] = {k: Position(**v) for k, v in d.get("positions", {}).items()}
        return cls(**d)
