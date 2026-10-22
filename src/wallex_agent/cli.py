"""Command line entry point.

  wallex-agent check                       connectivity: Wallex public API, account (if key), AI gateway
  wallex-agent fetch-data --days 30        download candles for allowed markets into data/candles
  wallex-agent backtest [--ai off|replay|live] [--symbols BTCTMN,ETHTMN] [--start ... --end ...]
  wallex-agent run --mode shadow|live [--ai live|replay|off] [--once] [--mcp]
  wallex-agent status --mode shadow|live   print persisted portfolio state
  wallex-agent liquidate --mode live       flatten everything to TMN now (algorithmic emergency exit)
  wallex-agent report                      summarise decision logs (trades, reasons, AI usage)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from wallex_agent.config import load_settings

console = Console()


def _logging(verbose: bool):
    Path("logs").mkdir(exist_ok=True)
    handlers = [RichHandler(console=console, show_path=False, rich_tracebacks=True),
                logging.FileHandler("logs/agent.log", encoding="utf-8")]
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(asctime)s %(name)s %(message)s", handlers=handlers)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def cmd_check(s, a):
    from wallex_agent.wallex.client import WallexClient, WallexError
    c = WallexClient(s.wallex_base_url, s.wallex_api_key)
    try:
        mk = c.markets()
        console.print(f"[green]Wallex public API ok[/]: {len(mk)} markets")
        t = Table("symbol", "bid", "ask", "spread%", "24h vol (TMN)", "found")
        for sym in s.markets.allowed:
            m = mk.get(sym)
            t.add_row(sym, *( [f"{m.bid:,.0f}", f"{m.ask:,.0f}", f"{m.spread_pct:.3f}", f"{m.quote_volume_24h:,.0f}", "yes"] if m else ["-", "-", "-", "-", "[red]NO[/]"]))
        console.print(t)
        payload = c.candles(s.markets.allowed[0], s.data.resolution, int(time.time()) - 86400, int(time.time()))
        console.print(f"candles {s.markets.allowed[0]}@{s.data.resolution}: status={payload.get('s')} bars={len(payload.get('t', []))}")
    except WallexError as e:
        console.print(f"[red]Wallex public API failed:[/] {e}")
    if s.wallex_api_key:
        try:
            b = c.balances(); console.print("[green]account ok[/] balances:", {k: v for k, v in b.items() if v["value"] > 0})
            console.print("fees:", c.fees())
        except WallexError as e:
            console.print(f"[red]account failed:[/] {e}")
    else:
        console.print("[yellow]WALLEX_API_KEY not set; skipping private endpoints[/]")
    if s.ai_api_key:
        from openai import OpenAI
        try:
            cl = OpenAI(base_url=s.ai_base_url, api_key=s.ai_api_key, timeout=60)
            r = cl.chat.completions.create(model=s.ai_model_fast, messages=[{"role": "user", "content": "Reply with exactly: OK"}], max_completion_tokens=20)
            console.print(f"[green]AI gateway ok[/] ({s.ai_model_fast}): {r.choices[0].message.content!r}")
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]AI gateway failed:[/] {e}")
    else:
        console.print("[yellow]AI_API_KEY not set[/]")


def cmd_fetch(s, a):
    from wallex_agent.data.market_data import CandleStore, MarketData
    from wallex_agent.wallex.client import WallexClient
    md = MarketData(WallexClient(s.wallex_base_url, s.wallex_api_key), CandleStore(s.data.candle_cache_dir))
    end = int(time.time()); start = end - a.days * 86400
    for sym in (a.symbols.split(",") if a.symbols else s.markets.allowed):
        df = md.history(sym, a.resolution or s.data.resolution, start, end)
        console.print(f"{sym}: {len(df)} bars cached -> {md.store.path(sym, a.resolution or s.data.resolution)}")


def cmd_backtest(s, a):
    from wallex_agent.backtest.runner import run_backtest
    syms = a.symbols.split(",") if a.symbols else s.markets.allowed
    r = run_backtest(s, syms, a.start, a.end, ai_mode=a.ai, resolution=a.resolution)
    t = Table("metric", "value", title=f"Backtest {r.out_dir.name} (ai={a.ai})")
    for k, v in r.metrics.items():
        t.add_row(k, json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v))
    console.print(t)
    console.print(f"artifacts: {r.out_dir}/metrics.json, equity.csv, trades.csv, backtest_decisions.jsonl")


def cmd_run(s, a):
    from wallex_agent.engine.loop import run
    if a.mode == "live" and not s.wallex_api_key:
        sys.exit("live mode needs WALLEX_API_KEY")
    if a.mode == "live" and not a.yes:
        console.print("[bold red]LIVE MODE: real orders will be placed on Wallex with the challenge account.[/]")
        if input("type LIVE to continue: ").strip() != "LIVE":
            sys.exit("aborted")
    run(s, a.mode, ai_mode=a.ai, once=a.once, mcp=a.mcp)


def cmd_status(s, a):
    from wallex_agent.portfolio.state import Portfolio
    p = Path(s.execution.state_dir) / f"portfolio_{a.mode}.json"
    if not p.exists():
        sys.exit(f"no state at {p}")
    pf = Portfolio.load(p)
    console.print_json(json.dumps(pf.summary({}), ensure_ascii=False))


def cmd_liquidate(s, a):
    from wallex_agent.engine.loop import build_agent
    client, agent, md, state_path, _ = build_agent(s, a.mode, ai_mode="off")
    markets = client.markets()
    prices = {sym: (markets[sym].mid if sym in markets else p.entry_price) for sym, p in agent.pf.positions.items()}
    agent.liquidate_all(prices, markets, reason="manual liquidation command")
    agent.pf.save(state_path)
    console.print(agent.pf.summary(prices))


def cmd_report(s, a):
    from wallex_agent.backtest.report import summarize_logs
    summarize_logs(Path(a.path or s.execution.decision_log_dir), console)


def main(argv=None):
    p = argparse.ArgumentParser(prog="wallex-agent", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None); p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check").set_defaults(fn=cmd_check)
    f = sub.add_parser("fetch-data"); f.add_argument("--days", type=int, default=30); f.add_argument("--symbols"); f.add_argument("--resolution"); f.set_defaults(fn=cmd_fetch)
    b = sub.add_parser("backtest"); b.add_argument("--ai", choices=["off", "replay", "live"], default="off"); b.add_argument("--symbols"); b.add_argument("--start"); b.add_argument("--end"); b.add_argument("--resolution"); b.set_defaults(fn=cmd_backtest)
    r = sub.add_parser("run"); r.add_argument("--mode", choices=["shadow", "live"], default=None); r.add_argument("--ai", choices=["live", "replay", "off"], default="live"); r.add_argument("--once", action="store_true"); r.add_argument("--mcp", action="store_true"); r.add_argument("-y", "--yes", action="store_true"); r.set_defaults(fn=cmd_run)
    st = sub.add_parser("status"); st.add_argument("--mode", default="shadow"); st.set_defaults(fn=cmd_status)
    lq = sub.add_parser("liquidate"); lq.add_argument("--mode", default="live"); lq.set_defaults(fn=cmd_liquidate)
    rp = sub.add_parser("report"); rp.add_argument("--path"); rp.set_defaults(fn=cmd_report)
    a = p.parse_args(argv)
    _logging(a.verbose)
    s = load_settings(a.config)
    if a.cmd == "run" and a.mode is None:
        a.mode = s.run_mode if s.run_mode in ("shadow", "live") else "shadow"
    a.fn(s, a)


if __name__ == "__main__":
    main()
