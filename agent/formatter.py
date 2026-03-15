"""
Rich-based terminal output helpers for the CLI AI Agent.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Generator

from rich.console import Console
from rich.markdown import Markdown
from rich.status import Status

console = Console()

# ── Prefixes ──────────────────────────────────────────────────────────────────
_PREFIX_USER = "[bold cyan][You][/bold cyan]"
_PREFIX_AGENT = "[bold green][Agent][/bold green]"
_PREFIX_TOOL = "[bold yellow][Tool][/bold yellow]"
_PREFIX_ERROR = "[bold red][Error][/bold red]"
_PREFIX_SYSTEM = "[bold magenta][System][/bold magenta]"


def print_welcome() -> None:
    """Print the welcome banner shown at REPL startup."""
    console.print()
    console.print("[bold green]╔══════════════════════════════════╗[/bold green]")
    console.print("[bold green]║      CLI Python AI Agent         ║[/bold green]")
    console.print("[bold green]║  Type 'exit' or 'quit' to leave  ║[/bold green]")
    console.print("[bold green]╚══════════════════════════════════╝[/bold green]")
    console.print()


def print_user(text: str) -> None:
    """Print a user message with a cyan prefix."""
    console.print(f"{_PREFIX_USER} {text}")


def print_assistant_token(text: str) -> None:
    """Print a single streaming token without a trailing newline."""
    console.print(text, end="", highlight=False)


def print_assistant_final(text: str) -> None:
    """Print the complete assistant response rendered as Markdown."""
    console.print(f"{_PREFIX_AGENT}")
    console.print(Markdown(text))


def print_tool_call(name: str, args: dict) -> None:
    """Print a tool call with its name and arguments."""
    args_str = json.dumps(args, indent=2)
    console.print(f"{_PREFIX_TOOL} [bold]{name}[/bold]")
    console.print(f"[yellow]{args_str}[/yellow]")


def print_error(message: str) -> None:
    """Print an error message with a red prefix."""
    console.print(f"{_PREFIX_ERROR} {message}")


def print_history(messages: list[dict]) -> None:
    """Print all messages in the session history with appropriate formatting."""
    console.print()
    console.print("[bold]── Conversation History ──[/bold]")
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content") or ""

        if role == "user":
            console.print(f"{_PREFIX_USER} {content}")
        elif role == "assistant":
            if content:
                console.print(f"{_PREFIX_AGENT}")
                console.print(Markdown(content))
            # tool_calls on assistant messages have no printable content here
        elif role == "tool":
            tool_name = msg.get("name", "unknown")
            console.print(f"{_PREFIX_TOOL} [bold]{tool_name}[/bold] result: {content}")
        elif role == "system":
            console.print(f"{_PREFIX_SYSTEM} {content}")
    console.print()


@contextmanager
def print_loading(message: str = "Thinking…") -> Generator[Status, None, None]:
    """Context manager that displays a spinner while the LLM is generating."""
    with console.status(f"[bold green]{message}[/bold green]", spinner="dots") as status:
        yield status
