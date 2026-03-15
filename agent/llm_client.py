"""LLM client for streaming chat completions with tool support."""

from __future__ import annotations

import json
from typing import Callable

import openai

from agent.config import Config


class LLMError(Exception):
    """Raised when the LLM API returns an error or a network failure occurs."""


class LLMClient:
    """Wraps the OpenAI SDK for streaming chat completions with tool support."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._client = openai.OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )

    def stream_response(
        self,
        messages: list[dict],
        tools: list[dict],
        on_token: Callable[[str], None],
        on_tool_call: Callable[[str, dict], str],
    ) -> str:
        """Stream a chat completion, handling tool calls recursively.

        Args:
            messages: Conversation history including system prompt.
            tools: OpenAI function-calling tool definitions.
            on_token: Called with each streamed text token.
            on_tool_call: Called with (tool_name, arguments_dict); returns result string.

        Returns:
            The full accumulated assistant message text.

        Raises:
            LLMError: On any OpenAI API or network error.
        """
        accumulated_text = ""
        current_messages = list(messages)

        while True:
            # Accumulate text tokens and tool call deltas from one streaming pass
            text_chunks: list[str] = []
            # tool_calls_map: index -> {"id": str, "name": str, "arguments": str}
            tool_calls_map: dict[int, dict] = {}
            # assistant message with tool_calls field (built from deltas)
            assistant_tool_calls: list[dict] = []

            try:
                kwargs: dict = {
                    "model": self._config.model,
                    "messages": current_messages,
                    "stream": True,
                }
                if tools:
                    kwargs["tools"] = tools

                stream = self._client.chat.completions.create(**kwargs)

                for chunk in stream:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if delta is None:
                        continue

                    # Accumulate text content
                    if delta.content:
                        text_chunks.append(delta.content)
                        accumulated_text += delta.content
                        on_token(delta.content)

                    # Accumulate tool call deltas
                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index
                            if idx not in tool_calls_map:
                                tool_calls_map[idx] = {
                                    "id": "",
                                    "name": "",
                                    "arguments": "",
                                }
                            entry = tool_calls_map[idx]
                            if tc_delta.id:
                                entry["id"] += tc_delta.id
                            if tc_delta.function and tc_delta.function.name:
                                entry["name"] += tc_delta.function.name
                            if tc_delta.function and tc_delta.function.arguments:
                                entry["arguments"] += tc_delta.function.arguments

            except openai.APIConnectionError as exc:
                raise LLMError(f"Connection error: {exc}") from exc
            except openai.APITimeoutError as exc:
                raise LLMError(f"Request timed out: {exc}") from exc
            except openai.APIError as exc:
                raise LLMError(f"API error: {exc}") from exc

            # If no tool calls were requested, we're done
            if not tool_calls_map:
                break

            # Build the assistant message that includes tool_calls
            for idx in sorted(tool_calls_map):
                entry = tool_calls_map[idx]
                assistant_tool_calls.append(
                    {
                        "id": entry["id"],
                        "type": "function",
                        "function": {
                            "name": entry["name"],
                            "arguments": entry["arguments"],
                        },
                    }
                )

            assistant_msg: dict = {
                "role": "assistant",
                "content": "".join(text_chunks) or None,
                "tool_calls": assistant_tool_calls,
            }
            current_messages.append(assistant_msg)

            # Execute each tool call and append results
            for tc in assistant_tool_calls:
                tool_name = tc["function"]["name"]
                raw_args = tc["function"]["arguments"]
                try:
                    arguments = json.loads(raw_args) if raw_args else {}
                except json.JSONDecodeError:
                    arguments = {}

                result = on_tool_call(tool_name, arguments)

                current_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "name": tool_name,
                        "content": result,
                    }
                )

            # Reset for the next streaming pass (after tool results)
            tool_calls_map = {}
            assistant_tool_calls = []

        return accumulated_text
