"""Optional bridge to the official Wallex MCP server (Streamable HTTP).

    URL     https://api.wallex.ir/mcp
    Auth    header X-API-Key (private tools only)
    Docs    https://api.wallex.ir/llm.txt

The trading loop uses the REST client for determinism and low latency. The MCP
bridge exposes the same exchange capabilities to the LLM analyst as *tools*, so
the analyst can pull extra context (order book, recent trades, market stats) on
demand while reasoning. It is fully optional: if the `mcp` package is missing or
the server is unreachable the agent keeps running with REST data only.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


class WallexMCP:
    def __init__(self, url: str = "https://api.wallex.ir/mcp", api_key: str = ""):
        self.url = url
        self.headers = {"X-API-Key": api_key} if api_key else {}
        self._tools: Optional[List[Dict[str, Any]]] = None

    async def _session(self):
        from mcp import ClientSession  # type: ignore
        from mcp.client.streamable_http import streamablehttp_client  # type: ignore
        return ClientSession, streamablehttp_client

    async def list_tools(self) -> List[Dict[str, Any]]:
        try:
            ClientSession, streamablehttp_client = await self._session()
        except ImportError:
            log.info("mcp package not installed; Wallex MCP bridge disabled (pip install mcp)")
            return []
        async with streamablehttp_client(self.url, headers=self.headers) as (r, w, _):
            async with ClientSession(r, w) as s:
                await s.initialize()
                res = await s.list_tools()
                self._tools = [{"name": t.name, "description": t.description or "",
                                "input_schema": t.inputSchema} for t in res.tools]
                return self._tools

    async def call(self, name: str, arguments: Dict[str, Any]) -> Any:
        ClientSession, streamablehttp_client = await self._session()
        async with streamablehttp_client(self.url, headers=self.headers) as (r, w, _):
            async with ClientSession(r, w) as s:
                await s.initialize()
                res = await s.call_tool(name, arguments)
                parts = []
                for c in res.content:
                    txt = getattr(c, "text", None)
                    if txt is not None:
                        try:
                            parts.append(json.loads(txt))
                        except ValueError:
                            parts.append(txt)
                return parts[0] if len(parts) == 1 else parts

    # sync helpers --------------------------------------------------------
    def list_tools_sync(self) -> List[Dict[str, Any]]:
        try:
            return asyncio.run(self.list_tools())
        except Exception as e:  # noqa: BLE001
            log.warning("Wallex MCP unavailable: %s", e)
            return []

    def call_sync(self, name: str, arguments: Dict[str, Any]) -> Any:
        return asyncio.run(self.call(name, arguments))

    def as_openai_tools(self, allow: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Convert MCP tool specs to OpenAI function-calling format (read-only tools only)."""
        tools = self._tools or []
        out = []
        for t in tools:
            n = t["name"].lower()
            if allow is not None and t["name"] not in allow:
                continue
            # never hand order-placing / withdrawal tools to the LLM: execution belongs to the risk engine
            if any(k in n for k in ("order", "withdraw", "cancel", "deposit")) and "open" not in n and "get" not in n:
                continue
            out.append({"type": "function", "function": {"name": t["name"], "description": t["description"][:900],
                                                         "parameters": t.get("input_schema") or {"type": "object", "properties": {}}}})
        return out
