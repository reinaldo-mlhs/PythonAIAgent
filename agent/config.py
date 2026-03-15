"""Configuration loading for the CLI Python AI Agent."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


class ConfigError(Exception):
    """Raised when required configuration is missing or invalid."""


@dataclass
class Config:
    api_key: str
    model: str
    base_url: str | None
    system_prompt: str
    shell_timeout: int  # seconds
    mcp_config: str | None  # path to MCP servers JSON config file


def load_config(
    model_override: str | None = None,
    system_prompt_override: str | None = None,
    mcp_config_override: str | None = None,
) -> Config:
    """Load configuration from environment variables and optional .env file.

    Args:
        model_override: Value from --model CLI flag; overrides LLM_MODEL env var.
        system_prompt_override: Value from --system-prompt CLI flag; overrides default.

    Returns:
        A populated Config instance.

    Raises:
        ConfigError: If LLM_API_KEY is not set.
    """
    # Load .env file if present (does not override already-set env vars)
    load_dotenv()

    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        raise ConfigError(
            "LLM_API_KEY environment variable is not set. "
            "Please set it before running the agent."
        )

    model = model_override or os.environ.get("LLM_MODEL") or "gpt-4o"
    base_url = os.environ.get("LLM_BASE_URL") or None
    system_prompt = system_prompt_override or "You are a helpful assistant."
    shell_timeout = 30
    mcp_config = mcp_config_override or os.environ.get("MCP_CONFIG") or None

    return Config(
        api_key=api_key,
        model=model,
        base_url=base_url,
        system_prompt=system_prompt,
        shell_timeout=shell_timeout,
        mcp_config=mcp_config,
    )
