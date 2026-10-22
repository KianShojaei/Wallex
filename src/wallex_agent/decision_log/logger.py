"""Append-only JSONL decision log: one record per *decision*, not just per trade.

Every loop writes what the agent saw and why it acted or did not act, so judges can
(1) replay the reasoning, (2) prove no manual trade happened, (3) audit the AI's role.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional


class DecisionLog:
    def __init__(self, directory: str, run_id: Optional[str] = None, mode: str = "shadow"):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        self.path = self.dir / f"{mode}_{self.run_id}.jsonl"

    def write(self, kind: str, **fields: Any) -> str:
        rec: Dict[str, Any] = {"id": uuid.uuid4().hex[:12], "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(fields.pop("now", None) or time.time())),
                               "kind": kind, **fields}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        return rec["id"]
