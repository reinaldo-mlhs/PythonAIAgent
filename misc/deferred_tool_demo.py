"""
Deferred Tool Calls Demo — Chat with Local + Deferred Tools
============================================================

Two tools:
  - get_weather(city)      → local, executes immediately (fake data)
  - send_notification(...) → deferred, pauses the run for web UI approval

Chat loop flow per turn:
  1. Run the agent with the user's message.
  2. If the output is a plain string → print it, continue chatting.
  3. If the output is DeferredToolRequests → show the "web UI" approval
     dialog, collect the decision, then resume the agent (no new user
     prompt) to let it finish the turn.
  4. The resumed result becomes the history base for the next turn.
"""

import os
from pydantic_ai import (
    Agent,
    CallDeferred,
    DeferredToolRequests,
    DeferredToolResults,
    RunContext,
    ToolDenied,
)
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

# ── Setup ──────────────────────────────────────────────────────────────────────

api_key = ""
model = OpenAIChatModel('gpt-5.2', provider=OpenAIProvider(api_key=api_key))

agent = Agent(
    model=model,
    instructions=(
        "You are a helpful assistant. "
        "You can look up weather for a city and send notifications to users. "
        "Notifications always require web UI approval before being sent."
    ),
    output_type=[str, DeferredToolRequests],
)


# ── Local tool — runs immediately ──────────────────────────────────────────────

@agent.tool_plain
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # Fake data — swap in a real API call if you like.
    fake_data = {
        "london":   "12°C, overcast",
        "new york": "24°C, sunny",
        "tokyo":    "18°C, light rain",
    }
    result = fake_data.get(city.lower(), "Weather data unavailable for that city.")
    print(f"\n  [LOCAL TOOL] get_weather({city!r}) → {result}")
    return result


# ── Deferred tool — pauses the run for external approval ──────────────────────

@agent.tool
def send_notification(ctx: RunContext, recipient: str, message: str) -> str:
    """
    Send a notification to a user.
    Always defers — requires web UI approval before the message is sent.
    """
    print(f"\n  [DEFERRED TOOL] send_notification called — pausing for UI approval")
    print(f"                  tool_call_id: {ctx.tool_call_id}")
    raise CallDeferred(metadata={"recipient": recipient, "message": message})


# ── Simulated web UI ───────────────────────────────────────────────────────────

def simulate_web_ui(requests: DeferredToolRequests) -> DeferredToolResults:
    """Simulate a frontend approval dialog for each pending notification."""
    results = DeferredToolResults()

    for call in requests.calls:
        meta = requests.metadata.get(call.tool_call_id, {})

        recipient = meta.get("recipient", "unknown")
        message   = meta.get("message",   "")

        print()
        print("  ┌─────────────────────────────────────────┐")
        print("  │        WEB UI — Approve Notification?   │")
        print("  ├─────────────────────────────────────────┤")
        print(f"  │  To:      {recipient:<31}│")
        # wrap long messages across lines
        words, line = message.split(), ""
        lines = []
        for w in words:
            if len(line) + len(w) + 1 > 31:
                lines.append(line)
                line = w
            else:
                line = (line + " " + w).strip()
        if line:
            lines.append(line)
        for l in lines:
            print(f"  │  Message: {l:<31}│")
        print("  └─────────────────────────────────────────┘")

        decision = input("  Approve? [y/n]: ").strip().lower()

        if decision == "y":
            results.calls[call.tool_call_id] = (
                f"Notification sent to {recipient!r}: {message!r}"
            )
        else:
            results.calls[call.tool_call_id] = ToolDenied(
                "User rejected the notification via web UI."
            )

    return results


# ── Chat loop ──────────────────────────────────────────────────────────────────

def run_turn(user_input: str, history: list) -> list:
    """
    Run one chat turn. Handles the deferred-tool resume internally so the
    caller always gets back a clean message history to pass to the next turn.
    """
    kwargs = {"message_history": history} if history else {}
    result = agent.run_sync(user_input, **kwargs)

    # ── Deferred tool path ─────────────────────────────────────────────────────
    if isinstance(result.output, DeferredToolRequests):
        messages = result.all_messages()
        print(f"\n  [AGENT PAUSED] {len(result.output.calls)} deferred call(s) pending...")

        deferred_results = simulate_web_ui(result.output)

        print("\n  [RESUMING AGENT...]\n")
        result = agent.run_sync(
            message_history=messages,
            deferred_tool_results=deferred_results,
        )

    # ── Normal path (or after resume) ─────────────────────────────────────────
    print(f"\nAssistant: {result.output}\n")
    return result.all_messages()


def main():
    print("Chat started. Type 'exit' to quit.")
    print("Try: 'What's the weather in Tokyo?'")
    print("     'Send a notification to bob@example.com saying the build passed.'\n")

    history = []
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("Goodbye.")
            break

        history = run_turn(user_input, history)


if __name__ == "__main__":
    main()
