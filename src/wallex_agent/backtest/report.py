"""Summarise decision logs (JSONL) for judges: trades, exit reasons, AI usage, rejects."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.table import Table


def summarize_logs(path: Path, console: Console) -> None:
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    if not files:
        console.print(f"no logs under {path}"); return
    for f in files:
        kinds, exits, rejects, pnl, ai_calls = Counter(), Counter(), Counter(), 0.0, 0
        for line in open(f, encoding="utf-8"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            kinds[r["kind"]] += 1
            if r["kind"] == "exit":
                exits[r["reason"].split(" (")[0]] += 1; pnl += r.get("pnl_tmn", 0)
            if r["kind"] == "reject":
                rejects[r.get("stage", "?")] += 1
            if r["kind"] in ("entry", "regime") or (r["kind"] == "reject" and r.get("stage") == "ai"):
                ai_calls += 1
        t = Table(title=str(f))
        t.add_column("item"); t.add_column("value")
        t.add_row("records", json.dumps(dict(kinds)))
        t.add_row("entries / exits", f"{kinds.get('entry', 0)} / {kinds.get('exit', 0)}")
        t.add_row("realized pnl (TMN)", f"{pnl:,.0f}")
        t.add_row("exit reasons", json.dumps(dict(exits), ensure_ascii=False))
        t.add_row("rejects by stage", json.dumps(dict(rejects)))
        t.add_row("decisions that involved the AI", str(ai_calls))
        console.print(t)
