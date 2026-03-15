"""Unit tests for agent/formatter.py"""

import io
import pytest
from unittest.mock import patch, MagicMock
from rich.console import Console

from agent import formatter


def _capture_output(func, *args, **kwargs) -> str:
    """Helper: capture rich console output to a string."""
    buf = io.StringIO()
    test_console = Console(file=buf, highlight=False, markup=True)
    with patch.object(formatter, "console", test_console):
        func(*args, **kwargs)
    return buf.getvalue()


class TestPrintWelcome:
    def test_prints_something(self):
        output = _capture_output(formatter.print_welcome)
        assert len(output.strip()) > 0

    def test_contains_agent_text(self):
        output = _capture_output(formatter.print_welcome)
        assert "Agent" in output or "CLI" in output


class TestPrintUser:
    def test_contains_message(self):
        output = _capture_output(formatter.print_user, "hello there")
        assert "hello there" in output

    def test_contains_you_prefix(self):
        output = _capture_output(formatter.print_user, "test")
        assert "You" in output


class TestPrintAssistantFinal:
    def test_contains_message(self):
        output = _capture_output(formatter.print_assistant_final, "plain text response")
        assert "plain text response" in output

    def test_contains_agent_prefix(self):
        output = _capture_output(formatter.print_assistant_final, "response")
        assert "Agent" in output


class TestPrintToolCall:
    def test_contains_tool_name(self):
        output = _capture_output(formatter.print_tool_call, "read_file", {"path": "/tmp/x"})
        assert "read_file" in output

    def test_contains_args(self):
        output = _capture_output(formatter.print_tool_call, "run_shell", {"command": "ls"})
        assert "ls" in output

    def test_contains_tool_prefix(self):
        output = _capture_output(formatter.print_tool_call, "read_file", {})
        assert "Tool" in output


class TestPrintError:
    def test_contains_message(self):
        output = _capture_output(formatter.print_error, "something went wrong")
        assert "something went wrong" in output

    def test_contains_error_prefix(self):
        output = _capture_output(formatter.print_error, "oops")
        assert "Error" in output


class TestPrintHistory:
    def test_prints_user_messages(self):
        messages = [{"role": "user", "content": "hello"}]
        output = _capture_output(formatter.print_history, messages)
        assert "hello" in output

    def test_prints_assistant_messages(self):
        messages = [{"role": "assistant", "content": "hi there"}]
        output = _capture_output(formatter.print_history, messages)
        assert "hi there" in output

    def test_prints_system_messages(self):
        messages = [{"role": "system", "content": "system instruction"}]
        output = _capture_output(formatter.print_history, messages)
        assert "system instruction" in output

    def test_empty_history_prints_header(self):
        output = _capture_output(formatter.print_history, [])
        assert "History" in output


class TestPrintLoading:
    def test_is_context_manager(self):
        # Should not raise; just verify it works as a context manager
        buf = io.StringIO()
        test_console = Console(file=buf, highlight=False)
        with patch.object(formatter, "console", test_console):
            with formatter.print_loading("Working…"):
                pass  # body executes without error
