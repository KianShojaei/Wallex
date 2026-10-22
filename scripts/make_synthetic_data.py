"""Generate synthetic candle CSVs so the backtester can be exercised without exchange access.

    python scripts/make_synthetic_data.py --symbols BTCTMN,ETHTMN --bars 720 --resolution 60
Real data: `wallex-agent fetch-data --days 30` (needs access to api.wallex.ir).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.conftest import synthetic_candles  # noqa: E402
from wallex_agent.data.market_data import CandleStore  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--symbols", default="BTCTMN,ETHTMN,SOLTMN")
p.add_argument("--bars", type=int, default=720)
p.add_argument("--resolution", default="60")
p.add_argument("--out", default="data/candles")
a = p.parse_args()
store = CandleStore(a.out)
for k, sym in enumerate(a.symbols.split(",")):
    df = synthetic_candles(n=a.bars, seed=10 + k, start_price=[7_000_000_000, 250_000_000, 15_000_000][k % 3], freq_min=int(a.resolution) if a.resolution.isdigit() else 1440)
    store.save(sym, a.resolution, df)
    print(f"{sym}: {len(df)} synthetic bars -> {store.path(sym, a.resolution)}")
