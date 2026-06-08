"""Remote tool backend over a real MCP ``streamable-http`` server.

This adapts the best of ``src/agent2.py`` (handshake, dynamic tool discovery,
``session.call_tool``) into a synchronous ``ToolClient`` so the otherwise
synchronous agent can drive a genuine network MCP boundary without a fragile
ad-hoc bridge.

Sync/async bridge: the asynchronous MCP session lives on a dedicated event loop
running in a background thread. The streamable-http and ``ClientSession`` async
contexts are entered ONCE (via an ``AsyncExitStack``) on that loop and kept open;
each synchronous ``call_tool`` submits a coroutine to the same loop with
``run_coroutine_threadsafe`` and blocks on the result. Enter and exit therefore
always happen on the loop that owns the anyio streams, which is the supported,
non-fragile pattern.

The ``mcp`` SDK is imported lazily inside ``connect`` so importing this module
never hard-requires it; failures become a clear ``ToolBackendError``.
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

from ..config import MCP_REMOTE_URL
from ..tool_client import ToolBackendError, ToolClient

_CONNECT_TIMEOUT_S = 30.0
_CALL_TIMEOUT_S = 120.0


class _LoopThread:
    """A background thread running a dedicated asyncio event loop."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    def run(self, coro: Any, timeout: float) -> Any:
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def shutdown(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        if not self._loop.is_running():
            self._loop.close()


def _first_text(result: Any) -> str | None:
    content = getattr(result, "content", None) or []
    for item in content:
        text = getattr(item, "text", None)
        if text is not None:
            return text
    return None


def _parse_tool_result(result: Any) -> dict[str, Any]:
    """Normalize an MCP ``CallToolResult`` into the same plain dict the local
    backend returns, so the agent and trace see identical shapes."""

    if getattr(result, "isError", False):
        return {
            "error": "tool_error",
            "message": _first_text(result) or "The MCP tool returned an error.",
        }

    text = _first_text(result)
    if text is not None:
        try:
            return json.loads(text)
        except (ValueError, TypeError):
            pass

    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        # FastMCP wraps scalar returns under a single "result" key.
        if set(structured.keys()) == {"result"}:
            return {"result": structured["result"]}
        return structured

    return {"result": text}


class MCPRemoteBackend(ToolClient):
    name = "mcp_remote"

    def __init__(self, url: str | None = None, connect: bool = True) -> None:
        self.url = url or MCP_REMOTE_URL
        self._bridge: _LoopThread | None = None
        self._stack: Any = None
        self._session: Any = None
        self._tools: list[dict[str, Any]] = []
        self._connected = False
        if connect:
            self.connect()

    # -- lifecycle ---------------------------------------------------------- #

    def connect(self) -> None:
        try:
            from contextlib import AsyncExitStack

            from mcp import ClientSession
            from mcp.client.streamable_http import streamablehttp_client
        except Exception as exc:  # noqa: BLE001 - SDK missing
            raise ToolBackendError(
                f"The 'mcp' SDK is required for the remote backend but is unavailable: {exc}"
            ) from exc

        self._ClientSession = ClientSession
        self._streamablehttp_client = streamablehttp_client
        self._AsyncExitStack = AsyncExitStack
        self._bridge = _LoopThread()
        try:
            self._tools = self._bridge.run(self._aconnect(), timeout=_CONNECT_TIMEOUT_S)
            self._connected = True
        except Exception as exc:  # noqa: BLE001 - connection/handshake failure
            self.close()
            raise ToolBackendError(
                f"Could not connect to the MCP server at {self.url}: {exc}"
            ) from exc

    async def _aconnect(self) -> list[dict[str, Any]]:
        self._stack = self._AsyncExitStack()
        read, write, _ = await self._stack.enter_async_context(
            self._streamablehttp_client(self.url)
        )
        self._session = await self._stack.enter_async_context(
            self._ClientSession(read, write)
        )
        await self._session.initialize()
        listed = await self._session.list_tools()
        return [
            {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema or {},
            }
            for tool in listed.tools
        ]

    def close(self) -> None:
        try:
            if self._stack is not None and self._bridge is not None:
                self._bridge.run(self._stack.aclose(), timeout=10)
        except Exception:  # noqa: BLE001 - best-effort teardown
            pass
        finally:
            self._stack = None
            self._session = None
            self._connected = False
            if self._bridge is not None:
                self._bridge.shutdown()
                self._bridge = None

    # -- ToolClient interface ---------------------------------------------- #

    def is_available(self) -> bool:
        return self._connected

    def list_tools(self) -> list[dict[str, Any]]:
        if not self._connected:
            raise ToolBackendError("Remote MCP backend is not connected.")
        return list(self._tools)

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        if not self._connected or self._bridge is None:
            raise ToolBackendError("Remote MCP backend is not connected.")
        return self._bridge.run(self._acall(name, arguments or {}), timeout=_CALL_TIMEOUT_S)

    async def _acall(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = await self._session.call_tool(name, arguments)
        return _parse_tool_result(result)
