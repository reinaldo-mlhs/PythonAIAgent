"""Unit tests for agent/llm_client.py"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import openai
import pytest

from agent.config import Config
from agent.llm_client import LLMClient, LLMError


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_config() -> Config:
    return Config(
        api_key="test-key",
        model="gpt-4o",
        base_url=None,
        system_prompt="You are a helpful assistant.",
        shell_timeout=30,
        mcp_config=None,
    )


def _text_chunk(content: str) -> SimpleNamespace:
    """Build a fake streaming chunk that carries a text delta."""
    delta = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(delta=delta)
    return SimpleNamespace(choices=[choice])


def _tool_chunk(index: int, tc_id: str, name: str, arguments: str) -> SimpleNamespace:
    """Build a fake streaming chunk that carries a tool-call delta."""
    fn = SimpleNamespace(name=name, arguments=arguments)
    tc_delta = SimpleNamespace(index=index, id=tc_id, function=fn)
    delta = SimpleNamespace(content=None, tool_calls=[tc_delta])
    choice = SimpleNamespace(delta=delta)
    return SimpleNamespace(choices=[choice])


def _empty_chunk() -> SimpleNamespace:
    """Build a fake streaming chunk with no delta content."""
    delta = SimpleNamespace(content=None, tool_calls=None)
    choice = SimpleNamespace(delta=delta)
    return SimpleNamespace(choices=[choice])


# ── Streaming accumulation ────────────────────────────────────────────────────

class TestStreamingAccumulation:
    """Tokens streamed by the API compose to the full response string."""

    def _run_stream(self, chunks: list) -> tuple[str, list[str]]:
        """Run stream_response with the given chunks; return (return_value, tokens_received)."""
        config = _make_config()
        client = LLMClient(config)

        tokens: list[str] = []

        with patch.object(client._client.chat.completions, "create", return_value=iter(chunks)):
            result = client.stream_response(
                messages=[{"role": "user", "content": "hi"}],
                tools=[],
                on_token=tokens.append,
                on_tool_call=lambda name, args: "unused",
            )

        return result, tokens

    def test_single_token_returned(self):
        result, tokens = self._run_stream([_text_chunk("hello")])
        assert result == "hello"
        assert tokens == ["hello"]

    def test_multiple_tokens_concatenated(self):
        chunks = [_text_chunk("foo"), _text_chunk(" "), _text_chunk("bar")]
        result, tokens = self._run_stream(chunks)
        assert result == "foo bar"
        assert tokens == ["foo", " ", "bar"]

    def test_empty_stream_returns_empty_string(self):
        result, tokens = self._run_stream([_empty_chunk()])
        assert result == ""
        assert tokens == []

    def test_on_token_called_for_each_chunk(self):
        words = ["one", " ", "two", " ", "three"]
        chunks = [_text_chunk(w) for w in words]
        result, tokens = self._run_stream(chunks)
        assert tokens == words
        assert result == "".join(words)

    def test_return_value_equals_concatenated_tokens(self):
        words = ["alpha", "beta", "gamma"]
        chunks = [_text_chunk(w) for w in words]
        result, tokens = self._run_stream(chunks)
        assert result == "".join(tokens)


# ── Tool call dispatch ────────────────────────────────────────────────────────

class TestToolCallDispatch:
    """When the model requests a tool, on_tool_call is invoked and the result
    is fed back so the model can continue."""

    def _make_client_with_two_passes(
        self,
        first_pass_chunks: list,
        second_pass_chunks: list,
    ) -> tuple[LLMClient, list]:
        """Return (client, call_log) where call_log records (name, args) pairs."""
        config = _make_config()
        client = LLMClient(config)

        call_count = 0

        def fake_create(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return iter(first_pass_chunks)
            return iter(second_pass_chunks)

        client._client.chat.completions.create = fake_create
        return client

    def test_on_tool_call_invoked_with_name_and_args(self):
        tool_chunks = [
            _tool_chunk(0, "call-1", "read_file", json.dumps({"path": "/tmp/x"})),
        ]
        text_chunks = [_text_chunk("done")]

        config = _make_config()
        client = LLMClient(config)

        call_log: list[tuple[str, dict]] = []
        call_count = 0

        def fake_create(**kwargs):
            nonlocal call_count
            call_count += 1
            return iter(tool_chunks if call_count == 1 else text_chunks)

        client._client.chat.completions.create = fake_create

        client.stream_response(
            messages=[{"role": "user", "content": "read it"}],
            tools=[],
            on_token=lambda t: None,
            on_tool_call=lambda name, args: (call_log.append((name, args)) or "file content"),
        )

        assert len(call_log) == 1
        assert call_log[0][0] == "read_file"
        assert call_log[0][1] == {"path": "/tmp/x"}

    def test_tool_result_appended_to_messages(self):
        """The tool result message must be included in the second API call."""
        tool_chunks = [
            _tool_chunk(0, "call-42", "run_shell", json.dumps({"command": "ls"})),
        ]
        text_chunks = [_text_chunk("ok")]

        config = _make_config()
        client = LLMClient(config)

        captured_messages: list[list[dict]] = []
        call_count = 0

        def fake_create(**kwargs):
            nonlocal call_count
            call_count += 1
            captured_messages.append(kwargs["messages"])
            return iter(tool_chunks if call_count == 1 else text_chunks)

        client._client.chat.completions.create = fake_create

        client.stream_response(
            messages=[{"role": "user", "content": "run ls"}],
            tools=[],
            on_token=lambda t: None,
            on_tool_call=lambda name, args: "file1\nfile2",
        )

        # Second call should include a tool-role message
        second_call_messages = captured_messages[1]
        tool_msgs = [m for m in second_call_messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["content"] == "file1\nfile2"
        assert tool_msgs[0]["tool_call_id"] == "call-42"

    def test_no_tool_call_skips_on_tool_call(self):
        """When the model returns only text, on_tool_call must never be called."""
        config = _make_config()
        client = LLMClient(config)

        client._client.chat.completions.create = lambda **kw: iter([_text_chunk("hello")])

        called = []
        client.stream_response(
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            on_token=lambda t: None,
            on_tool_call=lambda name, args: called.append((name, args)) or "",
        )

        assert called == []


# ── Error wrapping ────────────────────────────────────────────────────────────

class TestErrorWrapping:
    """openai API errors are wrapped in LLMError."""

    def _client_that_raises(self, exc: Exception) -> LLMClient:
        config = _make_config()
        client = LLMClient(config)
        client._client.chat.completions.create = MagicMock(side_effect=exc)
        return client

    def _call(self, client: LLMClient) -> None:
        client.stream_response(
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            on_token=lambda t: None,
            on_tool_call=lambda name, args: "",
        )

    def test_api_error_raises_llm_error(self):
        # openai.APIError requires request and body kwargs
        exc = openai.APIError("bad request", request=MagicMock(), body=None)
        client = self._client_that_raises(exc)
        with pytest.raises(LLMError):
            self._call(client)

    def test_api_connection_error_raises_llm_error(self):
        exc = openai.APIConnectionError(request=MagicMock())
        client = self._client_that_raises(exc)
        with pytest.raises(LLMError):
            self._call(client)

    def test_api_timeout_error_raises_llm_error(self):
        exc = openai.APITimeoutError(request=MagicMock())
        client = self._client_that_raises(exc)
        with pytest.raises(LLMError):
            self._call(client)

    def test_llm_error_message_contains_detail(self):
        exc = openai.APIConnectionError(request=MagicMock())
        client = self._client_that_raises(exc)
        with pytest.raises(LLMError, match="Connection error"):
            self._call(client)

    def test_llm_error_is_not_openai_error(self):
        """LLMError should be a plain exception, not an openai exception."""
        exc = openai.APITimeoutError(request=MagicMock())
        client = self._client_that_raises(exc)
        with pytest.raises(LLMError) as exc_info:
            self._call(client)
        assert not isinstance(exc_info.value, openai.APIError)
