"""Strategy module interface. Every module turns market context into *candidates* (Signal) and
exit hints; sizing, stops and execution always belong to the RiskManager."""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional

import pandas as pd

from wallex_agent.data.features import Snapshot


@dataclass
class Signal:
    symbol: str
    side: str                      # "BUY" (entry candidate)
    score: float                   # 0..1 confidence of the rule layer
    module: str = "generic"
    reasons: List[str] = field(default_factory=list)
    vetoes: List[str] = field(default_factory=list)
    snapshot: Optional[Dict] = None
    # per-module risk overrides (RiskManager clamps them to global limits)
    max_position_pct: Optional[float] = None
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None     # None/0 -> no fixed take-profit (trailing/regime exit only)
    trailing_pct: Optional[float] = None
    max_holding_minutes: Optional[int] = None
    allow_top_up: bool = False
    floor_size_factor: float = 0.0     # minimum size the risk engine takes when the rule layer is confident (AI advises above it)

    @property
    def is_candidate(self) -> bool:
        return not self.vetoes

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Context:
    """Everything a module may look at for one step."""
    now: float
    snapshots: Dict[str, Snapshot]
    frames: Dict[str, pd.DataFrame]            # indicator-enriched OHLCV per symbol (closed bars)
    prices: Dict[str, float]
    quote_asset: str = "TMN"
    usdt_symbol: str = "USDTTMN"


class StrategyModule:
    name = "base"

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def symbols(self, universe: List[str]) -> List[str]:
        return universe

    def candidates(self, ctx: Context) -> List[Signal]:
        return []

    def exit_hint(self, symbol: str, ctx: Context) -> Optional[str]:
        return None

    def position_cap_pct(self, symbol: str) -> Optional[float]:
        """Per-symbol cap (fraction of equity) if the module defines one; used to trim oversized positions."""
        return None
