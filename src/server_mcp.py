"""Real MCP network backend server (streamable-http) for Phase 2.

This is the **network execution mode** of the *same* Phase 2 system: it exposes
the exact same tool set as the in-process registry (``src/mcp_server.py``) by
wrapping the shared ``TOOL_SPECS`` -- the seven specialized deterministic tools
plus the demonstration ``calculatrice_medicale``. No tool logic is defined or
duplicated here; the agent reaches these tools through ``MCPRemoteBackend``
exactly as it reaches the local backend, so behaviour is identical.

Run (serves on ``MCP_REMOTE_URL``, default ``http://127.0.0.1:8000/mcp``):

    python src/server_mcp.py          # or: python -m src.server_mcp

Requires the ``mcp`` SDK, a running Ollama (``bge-m3:latest`` + ``qwen2.5:14b``)
and the ChromaDB index at ``data/chroma_db``. Descriptive / non-clinical only.
"""

from __future__ import annotations

import os
import sys
from urllib.parse import urlparse

# Make ``python src/server_mcp.py`` work as a script (root not on sys.path then).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from src.config import MCP_REMOTE_URL  # noqa: E402
from src.mcp_server import TOOL_SPECS  # noqa: E402

# Configure host/port/path from the single source of truth (MCP_REMOTE_URL) so
# client and server stay in sync and nothing is hardcoded in two places.
_parsed = urlparse(MCP_REMOTE_URL)
mcp = FastMCP(
    "icu-trajectory-mcp",
    host=_parsed.hostname or "127.0.0.1",
    port=_parsed.port or 8000,
    streamable_http_path=_parsed.path or "/mcp",
)

# Expose every shared tool (7 specialized + calculatrice_medicale) over MCP by
# wrapping the existing implementations. FastMCP introspects each function's
# signature/docstring for the input schema. Same registry as the local backend.
for _spec in TOOL_SPECS:
    mcp.add_tool(_spec.func, name=_spec.name, description=_spec.description)


if __name__ == "__main__":
    print(f"Starting ICU MCP server (streamable-http) on {MCP_REMOTE_URL} ...")
    print(f"Exposing {len(TOOL_SPECS)} tools: {', '.join(s.name for s in TOOL_SPECS)}")
    mcp.run(transport="streamable-http")
