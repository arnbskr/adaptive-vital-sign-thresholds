"""Backend-agnostic tool client for the Phase 2 agent.

The agent must not depend directly on ``src.mcp_server`` (in-process) or on the
real MCP HTTP server. It depends on the small ``ToolClient`` interface defined
here. Two interchangeable backends implement it:

* ``LocalToolBackend``      -- in-process MCP-compatible registry (default).
* ``MCPRemoteBackend``      -- a real MCP server over streamable-http.

Both expose the SAME seven specialized tools and return the SAME plain-dict
outputs, so the agent's deterministic orchestration and the auditable trace
behave identically whichever backend is selected.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .config import DEFAULT_TOOL_BACKEND

# Accepted aliases for each backend (CLI / env / sidebar friendliness).
_LOCAL_ALIASES = {"local", "local_mcp", "in_process", "inprocess", "local-mcp"}
_REMOTE_ALIASES = {"mcp_remote", "remote", "mcp", "mcp-remote", "remote_mcp", "http"}


class ToolBackendError(RuntimeError):
    """Raised when a tool backend cannot be created or reached."""


class ToolClient(ABC):
    """Common interface for calling Phase 2 tools, independent of the backend."""

    #: short, stable backend identifier recorded in the trace ("local" / "mcp_remote")
    name: str = "tool_client"

    @abstractmethod
    def list_tools(self) -> list[dict[str, Any]]:
        """Return the tool catalogue ([{name, description, parameters}, ...])."""

    @abstractmethod
    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Invoke a tool by name with keyword arguments; return a plain dict."""

    def is_available(self) -> bool:
        """Whether the backend is ready to serve calls. Local is always True."""

        return True

    def close(self) -> None:
        """Release any resources (network sessions, threads). No-op by default."""


def is_remote_backend(backend: str | None) -> bool:
    return str(backend or "").strip().lower() in _REMOTE_ALIASES


def get_tool_client(backend: str | None = None) -> ToolClient:
    """Factory: build the requested backend (defaults to ``DEFAULT_TOOL_BACKEND``).

    Backends are imported lazily so that selecting ``local`` never imports the
    optional ``mcp`` SDK, and a remote-connection failure surfaces as a clear
    ``ToolBackendError`` (which the agent can catch to fall back to local).
    """

    requested = str(backend or DEFAULT_TOOL_BACKEND).strip().lower()

    if requested in _LOCAL_ALIASES:
        from .tool_backends.local_backend import LocalToolBackend

        return LocalToolBackend()

    if requested in _REMOTE_ALIASES:
        from .tool_backends.mcp_remote_backend import MCPRemoteBackend

        return MCPRemoteBackend()

    raise ToolBackendError(
        f"Unknown tool backend '{backend}'. Use 'local' or 'mcp_remote'."
    )
