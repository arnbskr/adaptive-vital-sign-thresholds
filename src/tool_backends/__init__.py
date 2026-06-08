"""Phase 2 tool backends implementing the ``ToolClient`` interface.

* ``local_backend.LocalToolBackend``      -- in-process MCP-compatible registry.
* ``mcp_remote_backend.MCPRemoteBackend`` -- real MCP server over streamable-http.

Import the concrete backends lazily (e.g. via ``src.tool_client.get_tool_client``)
so the optional ``mcp`` SDK is only needed for the remote backend.
"""
