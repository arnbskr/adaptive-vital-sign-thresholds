"""Local, in-process tool backend.

Thin adapter over the existing MCP-compatible registry in ``src.mcp_server``
(``TOOL_SPECS`` / ``call_tool``). It introduces NO new logic: it simply exposes
the seven specialized tools through the ``ToolClient`` interface so the agent can
treat local and remote backends uniformly. Requires no network and no ``mcp`` SDK.
"""

from __future__ import annotations

from typing import Any

from ..mcp_server import call_tool as _call_tool
from ..mcp_server import list_tools as _list_tools
from ..tool_client import ToolClient


class LocalToolBackend(ToolClient):
    name = "local"

    def list_tools(self) -> list[dict[str, Any]]:
        return _list_tools()

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        return _call_tool(name, arguments or {})

    def is_available(self) -> bool:
        return True
