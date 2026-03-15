"""Unit tests for agent/mcp_client.py"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.mcp_client import MCPClient, _to_openai_schema, load_mcp_client


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_tool(name: str, description: str = "A tool", schema: dict | None = None):
    """Build a mock MCP Tool object."""
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.inputSchema = schema or {"type": "object", "properties": {}}
    return tool


def _make_content_block(text: str):
    block = MagicMock()
    block.text = text
    return block


# ── _to_openai_schema ─────────────────────────────────────────────────────────

class TestToOpenAISchema:
    def test_basic_conversion(self):
        tool = _make_tool("my_tool", "Does something")
        schema = _to_openai_schema(tool)
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "my_tool"
        assert schema["function"]["description"] == "Does something"

    def test_parameters_passed_through(self):
        params = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }
        tool = _make_tool("read_file", schema=params)
        schema = _to_openai_schema(tool)
        assert schema["function"]["parameters"] == params

    def test_none_description_becomes_empty_string(self):
        tool = _make_tool("tool", description=None)
        schema = _to_openai_schema(tool)
        assert schema["function"]["description"] == ""

    def test_none_input_schema_becomes_empty_object(self):
        tool = _make_tool("tool")
        tool.inputSchema = None
        schema = _to_openai_schema(tool)
        assert schema["function"]["parameters"] == {"type": "object", "properties": {}}


# ── load_mcp_client ───────────────────────────────────────────────────────────

class TestLoadMcpClient:
    def test_returns_none_when_no_path(self):
        assert load_mcp_client(None) is None

    def test_returns_mcp_client_for_valid_json(self, tmp_path):
        cfg = [{"name": "test", "command": "echo", "args": []}]
        f = tmp_path / "mcp.json"
        f.write_text(json.dumps(cfg))
        client = load_mcp_client(str(f))
        assert isinstance(client, MCPClient)

    def test_returns_none_for_missing_file(self):
        result = load_mcp_client("/nonexistent/path/mcp.json")
        assert result is None

    def test_returns_none_for_invalid_json(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("not valid json {{{")
        result = load_mcp_client(str(f))
        assert result is None

    def test_returns_none_when_json_is_not_array(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text(json.dumps({"name": "server"}))
        result = load_mcp_client(str(f))
        assert result is None

    def test_server_configs_passed_to_client(self, tmp_path):
        cfg = [
            {"name": "s1", "command": "cmd1"},
            {"name": "s2", "url": "http://localhost:8080/sse"},
        ]
        f = tmp_path / "mcp.json"
        f.write_text(json.dumps(cfg))
        client = load_mcp_client(str(f))
        assert client._server_configs == cfg


# ── MCPClient.get_definitions ─────────────────────────────────────────────────

class TestMCPClientGetDefinitions:
    def test_empty_before_connect(self):
        client = MCPClient([])
        assert client.get_definitions() == []

    def test_returns_registered_definitions(self):
        client = MCPClient([])
        tool = _make_tool("list_files", "List files in a directory")
        client._definitions.append(_to_openai_schema(tool))
        defs = client.get_definitions()
        assert len(defs) == 1
        assert defs[0]["function"]["name"] == "list_files"


# ── MCPClient.owns ────────────────────────────────────────────────────────────

class TestMCPClientOwns:
    def test_owns_returns_false_for_unknown_tool(self):
        client = MCPClient([])
        assert client.owns("unknown_tool") is False

    def test_owns_returns_true_after_registration(self):
        client = MCPClient([])
        client._tool_to_server["my_tool"] = "my_server"
        assert client.owns("my_tool") is True

    def test_owns_is_case_sensitive(self):
        client = MCPClient([])
        client._tool_to_server["MyTool"] = "server"
        assert client.owns("mytool") is False


# ── MCPClient.execute_async ───────────────────────────────────────────────────

class TestMCPClientExecuteAsync:
    def _client_with_session(self, tool_name: str, server_name: str = "srv"):
        client = MCPClient([])
        client._tool_to_server[tool_name] = server_name
        session = AsyncMock()
        client._sessions[server_name] = session
        return client, session

    def test_returns_error_for_unknown_tool(self):
        client = MCPClient([])
        result = asyncio.get_event_loop().run_until_complete(
            client.execute_async("nonexistent", {})
        )
        assert "unknown MCP tool" in result

    def test_calls_session_call_tool(self):
        client, session = self._client_with_session("my_tool")
        result_mock = MagicMock()
        result_mock.content = [_make_content_block("hello")]
        session.call_tool = AsyncMock(return_value=result_mock)

        result = asyncio.get_event_loop().run_until_complete(
            client.execute_async("my_tool", {"arg": "val"})
        )

        session.call_tool.assert_called_once_with("my_tool", {"arg": "val"})
        assert result == "hello"

    def test_joins_multiple_content_blocks(self):
        client, session = self._client_with_session("tool")
        result_mock = MagicMock()
        result_mock.content = [
            _make_content_block("line1"),
            _make_content_block("line2"),
        ]
        session.call_tool = AsyncMock(return_value=result_mock)

        result = asyncio.get_event_loop().run_until_complete(
            client.execute_async("tool", {})
        )
        assert result == "line1\nline2"

    def test_returns_empty_string_for_no_content(self):
        client, session = self._client_with_session("tool")
        result_mock = MagicMock()
        result_mock.content = []
        session.call_tool = AsyncMock(return_value=result_mock)

        result = asyncio.get_event_loop().run_until_complete(
            client.execute_async("tool", {})
        )
        assert result == ""

    def test_returns_error_string_on_exception(self):
        client, session = self._client_with_session("tool")
        session.call_tool = AsyncMock(side_effect=RuntimeError("connection lost"))

        result = asyncio.get_event_loop().run_until_complete(
            client.execute_async("tool", {})
        )
        assert "Error calling MCP tool" in result
        assert "connection lost" in result

    def test_content_block_without_text_uses_str(self):
        client, session = self._client_with_session("tool")
        # Use a real object without a .text attribute
        class NoTextBlock:
            def __str__(self):
                return "fallback"
        result_mock = MagicMock()
        result_mock.content = [NoTextBlock()]
        session.call_tool = AsyncMock(return_value=result_mock)

        result = asyncio.get_event_loop().run_until_complete(
            client.execute_async("tool", {})
        )
        assert result == "fallback"


# ── MCPClient.execute (sync wrapper) ─────────────────────────────────────────

class TestMCPClientExecuteSync:
    def test_sync_execute_returns_result(self):
        client = MCPClient([])
        client._tool_to_server["tool"] = "srv"
        session = AsyncMock()
        result_mock = MagicMock()
        result_mock.content = [_make_content_block("sync result")]
        session.call_tool = AsyncMock(return_value=result_mock)
        client._sessions["srv"] = session

        result = client.execute("tool", {})
        assert result == "sync result"

    def test_sync_execute_unknown_tool_returns_error(self):
        client = MCPClient([])
        result = client.execute("ghost", {})
        assert "unknown MCP tool" in result


# ── MCPClient.connect_all (mocked transport) ──────────────────────────────────

class TestMCPClientConnectAll:
    def test_connect_all_skips_server_on_error(self):
        """A failing server should not prevent the client from initialising."""
        client = MCPClient([{"name": "bad", "command": "nonexistent-binary"}])

        async def _run():
            # Patch the stdio_client context manager to raise immediately
            with patch("agent.mcp_client.stdio_client", side_effect=OSError("not found")):
                await client.connect_all()

        asyncio.get_event_loop().run_until_complete(_run())
        # No sessions registered, but no exception raised
        assert client._sessions == {}

    def test_connect_all_registers_tools_from_server(self):
        tool = _make_tool("remote_tool", "A remote tool")
        client = MCPClient([{"name": "srv", "command": "echo"}])

        async def _fake_connect(name, cfg):
            client._sessions[name] = AsyncMock()
            client._tool_to_server[tool.name] = name
            client._definitions.append(_to_openai_schema(tool))

        async def _run():
            with patch.object(client, "_connect_server", side_effect=_fake_connect):
                await client.connect_all()

        asyncio.get_event_loop().run_until_complete(_run())
        assert client.owns("remote_tool")
        assert any(d["function"]["name"] == "remote_tool" for d in client.get_definitions())

    def test_connect_all_raises_for_missing_command_and_url(self):
        client = MCPClient([{"name": "bad_cfg"}])

        async def _run():
            with patch("agent.mcp_client.AsyncExitStack") as mock_stack_cls:
                mock_stack = AsyncMock()
                mock_stack.__aenter__ = AsyncMock(return_value=mock_stack)
                mock_stack.__aexit__ = AsyncMock(return_value=False)
                mock_stack_cls.return_value = mock_stack
                await client.connect_all()

        # Should not raise — error is logged and skipped
        asyncio.get_event_loop().run_until_complete(_run())


# ── REPLSession MCP integration ───────────────────────────────────────────────

class TestREPLSessionMCPIntegration:
    """Tests that REPLSession correctly wires MCP tools into the LLM turn."""

    def _make_config(self, mcp_config=None):
        from agent.config import Config
        return Config(
            api_key="key",
            model="gpt-4o",
            base_url=None,
            system_prompt="sys",
            shell_timeout=30,
            mcp_config=mcp_config,
        )

    def test_no_mcp_when_config_is_none(self):
        from agent.repl import REPLSession
        config = self._make_config(mcp_config=None)
        with patch("agent.repl.LLMClient"), patch("agent.repl.load_mcp_client", return_value=None) as mock_load:
            session = REPLSession(config)
        mock_load.assert_called_once_with(None)
        assert session._mcp is None

    def test_mcp_client_connected_on_init(self):
        from agent.repl import REPLSession
        config = self._make_config(mcp_config="mcp.json")
        mock_mcp = MagicMock()
        mock_mcp.connect_all = AsyncMock()
        mock_mcp.close = AsyncMock()

        with patch("agent.repl.LLMClient"), \
             patch("agent.repl.load_mcp_client", return_value=mock_mcp), \
             patch("agent.repl.asyncio") as mock_asyncio:
            mock_asyncio.get_event_loop.return_value.run_until_complete = MagicMock()
            session = REPLSession(config)

        assert session._mcp is mock_mcp

    def test_mcp_tools_merged_into_definitions(self):
        from agent.repl import REPLSession
        config = self._make_config()
        mock_mcp = MagicMock()
        mock_mcp.get_definitions.return_value = [
            {"type": "function", "function": {"name": "mcp_tool", "description": "", "parameters": {}}}
        ]
        mock_mcp.owns.return_value = False
        mock_mcp.connect_all = AsyncMock()

        captured_tools = []

        def fake_stream(messages, tools, on_token, on_tool_call):
            captured_tools.extend(tools)
            return "response"

        with patch("agent.repl.LLMClient") as MockLLM, \
             patch("agent.repl.load_mcp_client", return_value=mock_mcp), \
             patch("agent.repl.asyncio") as mock_asyncio, \
             patch("agent.repl.formatter"):
            mock_asyncio.get_event_loop.return_value.run_until_complete = MagicMock()
            session = REPLSession(config)
            # Manually set _mcp since asyncio is mocked and connect_all wasn't really called
            session._mcp = mock_mcp
            session._llm = MagicMock()
            session._llm.stream_response.side_effect = fake_stream
            session._submit_turn("hello")

        tool_names = [d["function"]["name"] for d in captured_tools]
        assert "mcp_tool" in tool_names

    def test_mcp_tool_call_routed_to_mcp(self):
        from agent.repl import REPLSession
        config = self._make_config()
        mock_mcp = MagicMock()
        mock_mcp.get_definitions.return_value = []
        mock_mcp.owns.side_effect = lambda name: name == "mcp_tool"
        mock_mcp.execute.return_value = "mcp result"
        mock_mcp.connect_all = AsyncMock()

        captured_callback = {}

        def fake_stream(messages, tools, on_token, on_tool_call):
            captured_callback["fn"] = on_tool_call
            return "response"

        with patch("agent.repl.LLMClient"), \
             patch("agent.repl.load_mcp_client", return_value=mock_mcp), \
             patch("agent.repl.asyncio") as mock_asyncio, \
             patch("agent.repl.formatter"):
            mock_asyncio.get_event_loop.return_value.run_until_complete = MagicMock()
            session = REPLSession(config)
            session._llm = MagicMock()
            session._llm.stream_response.side_effect = fake_stream
            session._submit_turn("hello")

        result = captured_callback["fn"]("mcp_tool", {"arg": "val"})
        mock_mcp.execute.assert_called_once_with("mcp_tool", {"arg": "val"})
        assert result == "mcp result"

    def test_local_tool_call_not_routed_to_mcp(self):
        from agent.repl import REPLSession
        config = self._make_config()
        mock_mcp = MagicMock()
        mock_mcp.get_definitions.return_value = []
        mock_mcp.owns.return_value = False  # MCP doesn't own this tool
        mock_mcp.connect_all = AsyncMock()

        captured_callback = {}

        def fake_stream(messages, tools, on_token, on_tool_call):
            captured_callback["fn"] = on_tool_call
            return "response"

        with patch("agent.repl.LLMClient"), \
             patch("agent.repl.load_mcp_client", return_value=mock_mcp), \
             patch("agent.repl.asyncio") as mock_asyncio, \
             patch("agent.repl.formatter"), \
             patch("agent.repl.ToolRegistry") as MockRegistry:
            mock_asyncio.get_event_loop.return_value.run_until_complete = MagicMock()
            mock_registry_instance = MagicMock()
            mock_registry_instance.execute.return_value = "local result"
            mock_registry_instance.get_definitions.return_value = []
            MockRegistry.return_value = mock_registry_instance

            session = REPLSession(config)
            session._llm = MagicMock()
            session._llm.stream_response.side_effect = fake_stream
            session._submit_turn("hello")

        result = captured_callback["fn"]("read_file", {"path": "/tmp/f"})
        mock_mcp.execute.assert_not_called()
        mock_registry_instance.execute.assert_called_once_with("read_file", {"path": "/tmp/f"})
        assert result == "local result"
