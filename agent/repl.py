"""REPL session management for the CLI Python AI Agent."""

from __future__ import annotations

import asyncio
import logging

from agent import formatter
from agent.config import Config
from agent.llm_client import LLMClient
from agent.mcp_client import MCPClient, load_mcp_client
from agent.tools import ToolRegistry

logger = logging.getLogger(__name__)

# Approximate token limit for context window management.
_CONTEXT_WINDOW_LIMIT = 8000


def _estimate_tokens(messages: list[dict]) -> int:
    """Approximate total token count for a list of messages."""
    total = 0
    for msg in messages:
        content = msg.get("content") or ""
        total += len(content) // 4
    return total


class REPLSession:
    """Manages the interactive REPL loop and conversation history."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._llm = LLMClient(config)
        self._history: list[dict] = [
            {"role": "system", "content": config.system_prompt}
        ]
        self._mcp: MCPClient | None = load_mcp_client(config.mcp_config)
        if self._mcp:
            asyncio.get_event_loop().run_until_complete(self._mcp.connect_all())

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self) -> None:
        """Start the interactive REPL loop."""
        formatter.print_welcome()

        try:
            while True:
                try:
                    text = input("\n> ").strip()
                except KeyboardInterrupt:
                    print()
                    formatter.print_error("Interrupted. Goodbye!")
                    return
                except EOFError:
                    print()
                    formatter.print_error("EOF received. Goodbye!")
                    return

                if not text:
                    continue

                if self._handle_command(text):
                    continue

                self._submit_turn(text)
        finally:
            if self._mcp:
                asyncio.get_event_loop().run_until_complete(self._mcp.close())

    # ── Command handling ──────────────────────────────────────────────────────

    def _handle_command(self, text: str) -> bool:
        """Handle built-in REPL commands.

        Returns True if the text was a recognised command (so the caller
        should skip LLM submission), False otherwise.
        """
        lower = text.lower()

        if lower in ("exit", "quit"):
            formatter.print_error("Goodbye!")
            raise SystemExit(0)

        if lower == "/clear":
            self._history = [{"role": "system", "content": self._config.system_prompt}]
            formatter.print_error("History cleared.")
            return True

        if lower == "/history":
            formatter.print_history(self._history)
            return True

        return False

    # ── LLM turn ─────────────────────────────────────────────────────────────

    def _submit_turn(self, user_message: str) -> None:
        """Append the user message, stream the LLM response, and update history."""
        self._history.append({"role": "user", "content": user_message})
        self._truncate_history()

        registry = ToolRegistry()
        tools = registry.get_definitions()
        if self._mcp:
            tools = tools + self._mcp.get_definitions()

        def on_tool_call(name: str, args: dict) -> str:
            formatter.print_tool_call(name, args)
            if self._mcp and self._mcp.owns(name):
                return self._mcp.execute(name, args)
            return registry.execute(name, args)

        try:
            with formatter.print_loading():
                full_response = self._llm.stream_response(
                    messages=self._history,
                    tools=tools,
                    on_token=lambda _: None,
                    on_tool_call=on_tool_call,
                )

            formatter.print_assistant_final(full_response)
            self._history.append({"role": "assistant", "content": full_response})

        except Exception:
            logger.exception("Unhandled error during turn")
            formatter.print_error("An error occurred. Please try again.")

    # ── History management ────────────────────────────────────────────────────

    def _truncate_history(self) -> None:
        """Remove oldest non-system message pairs until within the token limit."""
        while _estimate_tokens(self._history) > _CONTEXT_WINDOW_LIMIT:
            # Find the first non-system message (index 1 onwards)
            non_system = [
                i for i, m in enumerate(self._history) if m["role"] != "system"
            ]
            if len(non_system) < 2:
                # Nothing left to remove safely
                break
            # Remove the oldest pair (user + assistant)
            self._history.pop(non_system[0])
            # After removal indices shift; recalculate
            non_system = [
                i for i, m in enumerate(self._history) if m["role"] != "system"
            ]
            if non_system:
                self._history.pop(non_system[0])
