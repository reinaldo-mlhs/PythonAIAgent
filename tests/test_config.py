"""Unit tests for agent/config.py"""

import os
import pytest
from unittest.mock import patch

from agent.config import Config, ConfigError, load_config


class TestLoadConfig:
    def test_loads_api_key_from_env(self):
        with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}, clear=True):
            config = load_config()
            assert config.api_key == "test-key"

    def test_raises_config_error_when_api_key_missing(self):
        with patch("agent.config.load_dotenv"), \
             patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ConfigError, match="LLM_API_KEY"):
                load_config()

    def test_default_model_is_gpt4o(self):
        with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}, clear=True):
            config = load_config()
            assert config.model == "gpt-4o"

    def test_model_from_env_var(self):
        with patch.dict(os.environ, {"LLM_API_KEY": "test-key", "LLM_MODEL": "gpt-3.5-turbo"}, clear=True):
            config = load_config()
            assert config.model == "gpt-3.5-turbo"

    def test_model_override_takes_precedence_over_env(self):
        with patch.dict(os.environ, {"LLM_API_KEY": "test-key", "LLM_MODEL": "gpt-3.5-turbo"}, clear=True):
            config = load_config(model_override="claude-3")
            assert config.model == "claude-3"

    def test_base_url_from_env(self):
        with patch.dict(os.environ, {"LLM_API_KEY": "test-key", "LLM_BASE_URL": "http://localhost:11434/v1"}, clear=True):
            config = load_config()
            assert config.base_url == "http://localhost:11434/v1"

    def test_base_url_defaults_to_none(self):
        with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}, clear=True):
            config = load_config()
            assert config.base_url is None

    def test_default_system_prompt(self):
        with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}, clear=True):
            config = load_config()
            assert config.system_prompt == "You are a helpful assistant."

    def test_system_prompt_override(self):
        with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}, clear=True):
            config = load_config(system_prompt_override="You are a pirate.")
            assert config.system_prompt == "You are a pirate."

    def test_default_shell_timeout_is_30(self):
        with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}, clear=True):
            config = load_config()
            assert config.shell_timeout == 30

    def test_returns_config_dataclass(self):
        with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}, clear=True):
            config = load_config()
            assert isinstance(config, Config)

    def test_loads_dotenv_file(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text("LLM_API_KEY=dotenv-key\n")
        # Patch load_dotenv to load from our specific .env file
        from dotenv import load_dotenv as _real_load_dotenv

        def _patched_load_dotenv():
            _real_load_dotenv(dotenv_path=str(env_file), override=True)

        with patch("agent.config.load_dotenv", _patched_load_dotenv):
            with patch.dict(os.environ, {}, clear=True):
                config = load_config()
                assert config.api_key == "dotenv-key"


class TestMcpConfig:
    def test_mcp_config_defaults_to_none(self):
        with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}, clear=True):
            config = load_config()
            assert config.mcp_config is None

    def test_mcp_config_from_env_var(self):
        with patch.dict(os.environ, {"LLM_API_KEY": "test-key", "MCP_CONFIG": "servers.json"}, clear=True):
            config = load_config()
            assert config.mcp_config == "servers.json"

    def test_mcp_config_override_takes_precedence(self):
        with patch.dict(os.environ, {"LLM_API_KEY": "test-key", "MCP_CONFIG": "env.json"}, clear=True):
            config = load_config(mcp_config_override="override.json")
            assert config.mcp_config == "override.json"

    def test_mcp_config_override_without_env(self):
        with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}, clear=True):
            config = load_config(mcp_config_override="my_servers.json")
            assert config.mcp_config == "my_servers.json"
