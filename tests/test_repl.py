"""Unit tests for agent/repl.py"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, call

import pytest

from agent.config import Config
from agent.repl import REPLSession, _estimate_tokens


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_config(system_prompt: str = "You are a helpful assistant.") -> Config:
    return Config(
        api_key="test-key",
        model="gpt-4o",
        base_url=None,
        system_prompt=system_prompt,
        shell_timeout=30,
    )


def _make_session(system_prompt: str = "You are a helpful assistant.") -> REPLSession:
    config = _make_config(system_prompt)
    with patch("agent.repl.LLMClient"):
        session = REPLSession(config)
    return session


# ── /clear command ────────────────────────────────────────────────────────────

class TestClearCommand:
    def test_clear_resets_history_to_system_only(self):
        session = _make_session()
        # Add some messages manually
        session._history.append({"role": "user", "content": "hello"})
        session._history.append({"role": "assistant", "content": "hi"})

        with patch("agent.repl.formatter"):
            session._handle_command("/clear")

        assert len(session._history) == 1
        assert session._history[0]["role"] == "system"

    def test_clear_preserves_system_prompt_content(self):
        session = _make_session(system_prompt="Custom prompt")
        session._history.append({"role": "user", "content": "hello"})

        with patch("agent.repl.formatter"):
            session._handle_command("/clear")

        assert session._history[0]["content"] == "Custom prompt"

    def test_clear_returns_true(self):
        session = _make_session()
        with patch("agent.repl.formatter"):
            result = session._handle_command("/clear")
        assert result is True

    def test_clear_on_empty_history_still_works(self):
        session = _make_session()
        with patch("agent.repl.formatter"):
            session._handle_command("/clear")
        assert len(session._history) == 1
        assert session._history[0]["role"] == "system"


# ── /history command ──────────────────────────────────────────────────────────

class TestHistoryCommand:
    def test_history_calls_print_history(self):
        session = _make_session()
        session._history.append({"role": "user", "content": "test"})

        with patch("agent.repl.formatter") as mock_fmt:
            session._handle_command("/history")
            mock_fmt.print_history.assert_called_once_with(session._history)

    def test_history_returns_true(self):
        session = _make_session()
        with patch("agent.repl.formatter"):
            result = session._handle_command("/history")
        assert result is True

    def test_history_passes_current_history_snapshot(self):
        session = _make_session()
        session._history.append({"role": "user", "content": "msg1"})
        session._history.append({"role": "assistant", "content": "resp1"})

        with patch("agent.repl.formatter") as mock_fmt:
            session._handle_command("/history")
            passed = mock_fmt.print_history.call_args[0][0]
            assert any(m["content"] == "msg1" for m in passed)


# ── exit / quit commands ──────────────────────────────────────────────────────

class TestExitQuitCommands:
    @pytest.mark.parametrize("cmd", ["exit", "quit", "EXIT", "QUIT", "Exit"])
    def test_exit_quit_raises_system_exit(self, cmd):
        session = _make_session()
        with patch("agent.repl.formatter"):
            with pytest.raises(SystemExit):
                session._handle_command(cmd)

    def test_exit_raises_system_exit_0(self):
        session = _make_session()
        with patch("agent.repl.formatter"):
            with pytest.raises(SystemExit) as exc_info:
                session._handle_command("exit")
        assert exc_info.value.code == 0

    def test_quit_raises_system_exit_0(self):
        session = _make_session()
        with patch("agent.repl.formatter"):
            with pytest.raises(SystemExit) as exc_info:
                session._handle_command("quit")
        assert exc_info.value.code == 0


# ── Unknown command returns False ─────────────────────────────────────────────

class TestUnknownCommand:
    def test_unknown_command_returns_false(self):
        session = _make_session()
        result = session._handle_command("hello world")
        assert result is False

    def test_unknown_slash_command_returns_false(self):
        session = _make_session()
        result = session._handle_command("/unknown")
        assert result is False


# ── KeyboardInterrupt and EOFError ────────────────────────────────────────────

class TestGracefulShutdown:
    def test_keyboard_interrupt_exits_gracefully(self):
        session = _make_session()
        with patch("agent.repl.formatter"):
            with patch("builtins.input", side_effect=KeyboardInterrupt):
                # run() should return without raising
                session.run()

    def test_eof_error_exits_gracefully(self):
        session = _make_session()
        with patch("agent.repl.formatter"):
            with patch("builtins.input", side_effect=EOFError):
                session.run()

    def test_keyboard_interrupt_does_not_propagate(self):
        session = _make_session()
        with patch("agent.repl.formatter"):
            with patch("builtins.input", side_effect=KeyboardInterrupt):
                try:
                    session.run()
                except KeyboardInterrupt:
                    pytest.fail("KeyboardInterrupt propagated out of run()")


# ── _submit_turn history updates ──────────────────────────────────────────────

class TestSubmitTurn:
    def _session_with_mock_llm(self, response: str = "assistant reply") -> REPLSession:
        session = _make_session()
        session._llm = MagicMock()
        session._llm.stream_response.return_value = response
        return session

    def test_submit_turn_appends_user_message(self):
        session = self._session_with_mock_llm()
        with patch("agent.repl.formatter"):
            session._submit_turn("hello")
        user_msgs = [m for m in session._history if m["role"] == "user"]
        assert any(m["content"] == "hello" for m in user_msgs)

    def test_submit_turn_appends_assistant_message(self):
        session = self._session_with_mock_llm("the answer")
        with patch("agent.repl.formatter"):
            session._submit_turn("question")
        asst_msgs = [m for m in session._history if m["role"] == "assistant"]
        assert any(m["content"] == "the answer" for m in asst_msgs)

    def test_submit_turn_user_before_assistant(self):
        session = self._session_with_mock_llm("reply")
        with patch("agent.repl.formatter"):
            session._submit_turn("user input")
        roles = [m["role"] for m in session._history if m["role"] != "system"]
        assert roles == ["user", "assistant"]

    def test_submit_turn_multiple_turns_accumulate(self):
        session = self._session_with_mock_llm("resp")
        with patch("agent.repl.formatter"):
            session._submit_turn("turn1")
            session._submit_turn("turn2")
        user_msgs = [m for m in session._history if m["role"] == "user"]
        assert len(user_msgs) == 2

    def test_submit_turn_exception_does_not_propagate(self):
        session = _make_session()
        session._llm = MagicMock()
        session._llm.stream_response.side_effect = RuntimeError("boom")
        with patch("agent.repl.formatter"):
            # Should not raise
            session._submit_turn("trigger error")


# ── _truncate_history ─────────────────────────────────────────────────────────

class TestTruncateHistory:
    def _session_with_long_history(self, num_pairs: int, msg_len: int = 200) -> REPLSession:
        """Build a session with num_pairs user+assistant pairs, each message ~msg_len chars."""
        session = _make_session()
        for i in range(num_pairs):
            session._history.append({"role": "user", "content": "u" * msg_len})
            session._history.append({"role": "assistant", "content": "a" * msg_len})
        return session

    def test_truncate_keeps_system_prompt(self):
        # 50 pairs × 200 chars × 2 messages = 20000 chars → ~5000 tokens > 8000 limit
        # Use longer messages to exceed limit
        session = self._session_with_long_history(num_pairs=100, msg_len=400)
        session._truncate_history()
        assert session._history[0]["role"] == "system"

    def test_truncate_reduces_token_count(self):
        session = self._session_with_long_history(num_pairs=100, msg_len=400)
        session._truncate_history()
        assert _estimate_tokens(session._history) <= 8000

    def test_truncate_removes_oldest_messages_first(self):
        session = _make_session()
        # Add messages with identifiable content; use 600 chars each so
        # 50 pairs × 1200 chars / 4 = 15000 tokens >> 8000 limit
        for i in range(50):
            session._history.append({"role": "user", "content": f"user_{i} " + "x" * 600})
            session._history.append({"role": "assistant", "content": f"asst_{i} " + "x" * 600})

        session._truncate_history()

        # The oldest user messages should be gone, newest should remain
        contents = [m["content"] for m in session._history]
        # user_0 should have been removed (oldest)
        assert not any("user_0 " in c for c in contents)

    def test_truncate_noop_when_within_limit(self):
        session = _make_session()
        session._history.append({"role": "user", "content": "short"})
        session._history.append({"role": "assistant", "content": "reply"})
        original_len = len(session._history)
        session._truncate_history()
        assert len(session._history) == original_len

    def test_truncate_preserves_system_as_first_message(self):
        session = self._session_with_long_history(num_pairs=100, msg_len=400)
        session._truncate_history()
        assert session._history[0]["role"] == "system"
        assert session._history[0]["content"] == "You are a helpful assistant."
