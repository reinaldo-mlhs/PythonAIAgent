"""Unit tests for agent/tools.py"""

import os
import pytest

from agent.tools import read_file, run_shell, ToolRegistry


class TestReadFile:
    def test_reads_existing_file(self, tmp_path):
        f = tmp_path / "hello.txt"
        f.write_text("hello world")
        assert read_file(str(f)) == "hello world"

    def test_returns_error_string_for_missing_file(self):
        result = read_file("/nonexistent/path/file.txt")
        assert result.startswith("Error")

    def test_reads_multiline_file(self, tmp_path):
        f = tmp_path / "multi.txt"
        content = "line1\nline2\nline3"
        f.write_text(content)
        assert read_file(str(f)) == content

    def test_reads_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("")
        assert read_file(str(f)) == ""


class TestRunShell:
    def test_captures_stdout(self):
        result = run_shell("echo hello")
        assert "hello" in result

    def test_captures_stderr(self):
        # A command that writes to stderr
        result = run_shell("python -c \"import sys; sys.stderr.write('err_output')\"")
        assert "err_output" in result

    def test_returns_error_string_on_timeout(self):
        result = run_shell("python -c \"import time; time.sleep(10)\"", timeout=1)
        assert "timed out" in result.lower() or "Error" in result

    def test_returns_output_for_nonzero_exit(self):
        # A command that exits non-zero still returns output
        result = run_shell("python -c \"import sys; sys.exit(1)\"")
        # Should not raise, just return (possibly empty) string
        assert isinstance(result, str)


class TestToolRegistry:
    def setup_method(self):
        self.registry = ToolRegistry()

    def test_get_definitions_returns_list(self):
        defs = self.registry.get_definitions()
        assert isinstance(defs, list)
        assert len(defs) == 3

    def test_definitions_contain_read_file(self):
        names = [d["function"]["name"] for d in self.registry.get_definitions()]
        assert "read_file" in names

    def test_definitions_contain_run_shell(self):
        names = [d["function"]["name"] for d in self.registry.get_definitions()]
        assert "run_shell" in names

    def test_execute_read_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("registry test")
        result = self.registry.execute("read_file", {"path": str(f)})
        assert result == "registry test"

    def test_execute_run_shell(self):
        result = self.registry.execute("run_shell", {"command": "echo dispatch"})
        assert "dispatch" in result

    def test_execute_unknown_tool_returns_error_string(self):
        result = self.registry.execute("nonexistent_tool", {})
        assert "Error" in result or "unknown" in result.lower()

    def test_execute_returns_error_string_on_exception(self):
        # Pass bad arguments to trigger an exception inside the tool
        result = self.registry.execute("read_file", {})  # missing 'path' arg
        assert isinstance(result, str)
        assert "Error" in result
