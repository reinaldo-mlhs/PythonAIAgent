"""Unit tests for agent/cli.py"""

from __future__ import annotations

import os
from unittest.mock import patch, MagicMock

import pytest
from click.testing import CliRunner

from agent.cli import main
from agent.config import ConfigError


@pytest.fixture
def runner():
    return CliRunner()


class TestHelpFlag:
    def test_help_exits_zero(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0

    def test_help_shows_usage(self, runner):
        result = runner.invoke(main, ["--help"])
        assert "Usage" in result.output

    def test_help_shows_system_prompt_option(self, runner):
        result = runner.invoke(main, ["--help"])
        assert "--system-prompt" in result.output

    def test_help_shows_model_option(self, runner):
        result = runner.invoke(main, ["--help"])
        assert "--model" in result.output


class TestVersionFlag:
    def test_version_exits_zero(self, runner):
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0

    def test_version_shows_version_string(self, runner):
        result = runner.invoke(main, ["--version"])
        # click version_option outputs something like "main, version X.Y.Z"
        assert result.output.strip() != ""


class TestSystemPromptFlag:
    def test_system_prompt_passed_to_load_config(self, runner):
        with patch("agent.cli.load_config") as mock_load, \
             patch("agent.cli.REPLSession") as mock_repl:
            mock_load.return_value = MagicMock()
            mock_repl.return_value.run.return_value = None

            runner.invoke(main, ["--system-prompt", "You are a pirate."])

            mock_load.assert_called_once_with(
                model_override=None,
                system_prompt_override="You are a pirate.",
                mcp_config_override=None,
            )

    def test_system_prompt_default_is_none(self, runner):
        with patch("agent.cli.load_config") as mock_load, \
             patch("agent.cli.REPLSession") as mock_repl:
            mock_load.return_value = MagicMock()
            mock_repl.return_value.run.return_value = None

            runner.invoke(main, [])

            mock_load.assert_called_once_with(
                model_override=None,
                system_prompt_override=None,
                mcp_config_override=None,
            )


class TestModelFlag:
    def test_model_passed_to_load_config(self, runner):
        with patch("agent.cli.load_config") as mock_load, \
             patch("agent.cli.REPLSession") as mock_repl:
            mock_load.return_value = MagicMock()
            mock_repl.return_value.run.return_value = None

            runner.invoke(main, ["--model", "gpt-3.5-turbo"])

            mock_load.assert_called_once_with(
                model_override="gpt-3.5-turbo",
                system_prompt_override=None,
                mcp_config_override=None,
            )

    def test_model_default_is_none(self, runner):
        with patch("agent.cli.load_config") as mock_load, \
             patch("agent.cli.REPLSession") as mock_repl:
            mock_load.return_value = MagicMock()
            mock_repl.return_value.run.return_value = None

            runner.invoke(main, [])

            mock_load.assert_called_once_with(
                model_override=None,
                system_prompt_override=None,
                mcp_config_override=None,
            )

    def test_model_and_system_prompt_together(self, runner):
        with patch("agent.cli.load_config") as mock_load, \
             patch("agent.cli.REPLSession") as mock_repl:
            mock_load.return_value = MagicMock()
            mock_repl.return_value.run.return_value = None

            runner.invoke(main, ["--model", "claude-3", "--system-prompt", "Be concise."])

            mock_load.assert_called_once_with(
                model_override="claude-3",
                system_prompt_override="Be concise.",
                mcp_config_override=None,
            )


class TestMissingApiKey:
    def test_config_error_exits_with_code_1(self, runner):
        with patch("agent.cli.load_config", side_effect=ConfigError("LLM_API_KEY is not set")):
            result = runner.invoke(main, [])
        assert result.exit_code == 1

    def test_config_error_message_printed_to_stderr(self, runner):
        with patch("agent.cli.load_config", side_effect=ConfigError("LLM_API_KEY is not set")):
            result = runner.invoke(main, [])
        # click.echo(..., err=True) goes to stderr; CliRunner merges it into output by default
        assert "LLM_API_KEY" in result.output

    def test_config_error_does_not_start_repl(self, runner):
        with patch("agent.cli.load_config", side_effect=ConfigError("LLM_API_KEY is not set")), \
             patch("agent.cli.REPLSession") as mock_repl:
            runner.invoke(main, [])
        mock_repl.assert_not_called()


class TestMcpConfigFlag:
    def test_mcp_config_passed_to_load_config(self, runner):
        with patch("agent.cli.load_config") as mock_load, \
             patch("agent.cli.REPLSession") as mock_repl:
            mock_load.return_value = MagicMock()
            mock_repl.return_value.run.return_value = None

            runner.invoke(main, ["--mcp-config", "mcp.json"])

            mock_load.assert_called_once_with(
                model_override=None,
                system_prompt_override=None,
                mcp_config_override="mcp.json",
            )

    def test_mcp_config_default_is_none(self, runner):
        with patch("agent.cli.load_config") as mock_load, \
             patch("agent.cli.REPLSession") as mock_repl:
            mock_load.return_value = MagicMock()
            mock_repl.return_value.run.return_value = None

            runner.invoke(main, [])

            mock_load.assert_called_once_with(
                model_override=None,
                system_prompt_override=None,
                mcp_config_override=None,
            )

    def test_help_shows_mcp_config_option(self, runner):
        result = runner.invoke(main, ["--help"])
        assert "--mcp-config" in result.output
