from __future__ import annotations

from typing import Dict, List

from wallex_agent.strategy.base import StrategyModule
from wallex_agent.strategy.carry import CarryStrategy
from wallex_agent.strategy.mean_reversion import MeanReversionStrategy
from wallex_agent.strategy.trend import TrendStrategy

REGISTRY = {"carry": CarryStrategy, "trend": TrendStrategy, "mean_reversion": MeanReversionStrategy}


def build_modules(modules_cfg: Dict[str, dict]) -> List[StrategyModule]:
    out = []
    for name, cfg in (modules_cfg or {}).items():
        cfg = dict(cfg or {})
        if not cfg.pop("enabled", True):
            continue
        if name not in REGISTRY:
            raise ValueError(f"unknown strategy module '{name}'; known: {list(REGISTRY)}")
        out.append(REGISTRY[name](cfg))
    if not out:
        raise ValueError("no strategy module enabled")
    return out
