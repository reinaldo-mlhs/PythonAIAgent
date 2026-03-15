"""CLI entry point for the AI agent using click."""

from __future__ import annotations

import sys

import click

from agent.config import ConfigError, load_config
from agent.repl import REPLSession


@click.command()
@click.option("--system-prompt", default=None, help="Override the default system prompt.")
@click.option("--model", default=None, help="Override the LLM model (LLM_MODEL env var).")
@click.option("--mcp-config", default=None, help="Path to MCP servers JSON config file (MCP_CONFIG env var).")
@click.version_option(version="0.1.0")
def main(system_prompt: str | None, model: str | None, mcp_config: str | None) -> None:
    """Start an interactive AI agent REPL session."""
    try:
        config = load_config(
            model_override=model,
            system_prompt_override=system_prompt,
            mcp_config_override=mcp_config,
        )
    except ConfigError as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)

    REPLSession(config).run()
