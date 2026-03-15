"""MCP (Model Context Protocol) client for connecting to MCP servers."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client

logger = logging.getLogger(__name__)


class MCPClient:
    """Manages connections to one or more MCP servers and exposes their tools."""

    def __init__(self, server_configs: list[dict]) -> None:
        """
        Args:
            server_configs: List of server config dicts. Each must have a 'name' key
                and either:
                - 'command' + optional 'args'/'env' for stdio transport
                - 'url' for SSE transport
        """
        self._server_configs = server_configs
        self._sessions: dict[str, ClientSession] = {}
        self._tool_to_server: dict[str, str] = {}
        self._definitions: list[dict] = []
        self._exit_stack = AsyncExitStack()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def connect_all(self) -> None:
        """Connect to all configured MCP servers and fetch their tool lists."""
        await self._exit_stack.__aenter__()
        for cfg in self._server_configs:
            name = cfg.get("name", "unnamed")
            try:
                await self._connect_server(name, cfg)
            except Exception:
                logger.exception("Failed to connect to MCP server '%s'", name)

    async def _connect_server(self, name: str, cfg: dict) -> None:
        if "command" in cfg:
            params = StdioServerParameters(
                command=cfg["command"],
                args=cfg.get("args", []),
                env=cfg.get("env"),
            )
            read, write = await self._exit_stack.enter_async_context(
                stdio_client(params)
            )
        elif "url" in cfg:
            read, write = await self._exit_stack.enter_async_context(
                sse_client(cfg["url"])
            )
        else:
            raise ValueError(f"MCP server '{name}' needs 'command' or 'url'")

        session = await self._exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await session.initialize()
        self._sessions[name] = session

        # Fetch and register tools from this server
        tools_result = await session.list_tools()
        for tool in tools_result.tools:
            self._tool_to_server[tool.name] = name
            self._definitions.append(_to_openai_schema(tool))

        logger.info(
            "Connected to MCP server '%s' with %d tool(s)",
            name,
            len(tools_result.tools),
        )

    async def close(self) -> None:
        """Disconnect from all MCP servers."""
        await self._exit_stack.__aexit__(None, None, None)

    # ── Tool interface ────────────────────────────────────────────────────────

    def get_definitions(self) -> list[dict]:
        """Return OpenAI function-calling schemas for all MCP tools."""
        return self._definitions

    async def execute_async(self, name: str, arguments: dict) -> str:
        """Call an MCP tool by name and return the result as a string."""
        server_name = self._tool_to_server.get(name)
        if server_name is None:
            return f"Error: unknown MCP tool '{name}'"

        session = self._sessions[server_name]
        try:
            result = await session.call_tool(name, arguments)
            # Flatten content blocks to a single string
            parts: list[str] = []
            for block in result.content:
                if hasattr(block, "text"):
                    parts.append(block.text)
                else:
                    parts.append(str(block))
            return "\n".join(parts) if parts else ""
        except Exception as exc:
            return f"Error calling MCP tool '{name}': {exc}"

    def execute(self, name: str, arguments: dict) -> str:
        """Synchronous wrapper around execute_async for use in the tool callback."""
        return asyncio.get_event_loop().run_until_complete(
            self.execute_async(name, arguments)
        )

    def owns(self, tool_name: str) -> bool:
        """Return True if this client knows about the given tool."""
        return tool_name in self._tool_to_server


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_openai_schema(tool: Any) -> dict:
    """Convert an MCP Tool object to an OpenAI function-calling schema dict."""
    params = tool.inputSchema if tool.inputSchema else {"type": "object", "properties": {}}
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": params,
        },
    }


def load_mcp_client(config_path: str | None) -> MCPClient | None:
    """Load an MCPClient from a JSON config file, or return None if not configured.

    The JSON file should be an array of server config objects, e.g.:
    [
      {"name": "filesystem", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]},
      {"name": "my-sse-server", "url": "http://localhost:8080/sse"}
    ]
    """
    if not config_path:
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            servers = json.load(f)
        if not isinstance(servers, list):
            raise ValueError("MCP config must be a JSON array of server objects")
        return MCPClient(servers)
    except Exception as exc:
        logger.error("Failed to load MCP config from '%s': %s", config_path, exc)
        return None
