# CLI Python AI Agent

A terminal-based conversational AI agent with a REPL interface. Chat with an LLM directly from your terminal, with support for multi-turn conversations, tool use (file reading, shell commands), and rich formatted output.

## Features

- Interactive REPL with multi-turn conversation history
- Streaming LLM responses with a loading indicator
- Tool use: read files and run shell commands on behalf of the model
- Rich terminal output with Markdown rendering and syntax highlighting
- Automatic context window management (truncates oldest turns when needed)
- OpenAI-compatible API support (works with OpenAI, Ollama, and other compatible providers)

## Requirements

- Python 3.11+
- An OpenAI-compatible API key

## Installation

```bash
pip install -e .
```

This installs the `kiro-agent` CLI entry point.

## Configuration

Configuration is loaded from environment variables or a `.env` file in the working directory.

| Variable | Required | Default | Description |
|---|---|---|---|
| `LLM_API_KEY` | yes | — | Your LLM API key |
| `LLM_MODEL` | no | `gpt-4o` | Model name to use |
| `LLM_BASE_URL` | no | — | Base URL for OpenAI-compatible providers (e.g. Ollama) |
| `MCP_CONFIG` | no | — | Path to a JSON file defining MCP servers |

Example `.env`:

```env
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4o
LLM_BASE_URL=http://localhost:11434/v1
MCP_CONFIG=./mcp_servers.json
```

## Usage

```bash
# Start a session
kiro-agent

# Or via Python module
python -m agent

# Override model or system prompt
kiro-agent --model gpt-4o-mini --system-prompt "You are a concise assistant."

# Load MCP servers from a config file
kiro-agent --mcp-config ./mcp_servers.json

# Show help
kiro-agent --help
```

## MCP Servers

The agent supports the [Model Context Protocol](https://modelcontextprotocol.io) (MCP), letting you connect external tool servers that the LLM can call just like built-in tools.

### Config file format

Create a JSON file containing an array of server objects. Each server needs a `name` and either a `command` (stdio transport) or a `url` (SSE transport).

```json
[
  {
    "name": "filesystem",
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
  },
  {
    "name": "my-sse-server",
    "url": "http://localhost:8080/sse"
  }
]
```

Stdio server fields:

| Field | Required | Description |
|---|---|---|
| `name` | yes | Unique label for this server |
| `command` | yes | Executable to launch (e.g. `npx`, `python`, `uvx`) |
| `args` | no | Array of arguments passed to the command |
| `env` | no | Object of extra environment variables for the process |

SSE server fields:

| Field | Required | Description |
|---|---|---|
| `name` | yes | Unique label for this server |
| `url` | yes | Full SSE endpoint URL |

### Pointing the agent at your config

Either pass it on the command line:

```bash
kiro-agent --mcp-config ./mcp_servers.json
```

Or set it in your `.env` / environment:

```env
MCP_CONFIG=./mcp_servers.json
```

### Example: filesystem server

```json
[
  {
    "name": "filesystem",
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/projects"]
  }
]
```

```bash
kiro-agent --mcp-config ./mcp_servers.json
```

The agent will connect to the server on startup, discover its tools, and make them available to the LLM automatically. Tool calls from MCP servers are displayed in the terminal the same way as built-in tools.

## REPL Commands

| Command | Description |
|---|---|
| `exit` / `quit` | End the session |
| `/clear` | Reset conversation history (keeps system prompt) |
| `/history` | Print all messages in the current session |
| `Ctrl+C` / `Ctrl+D` | Graceful exit |

## Tools

The agent can invoke these tools during a conversation:

- `read_file(path)` — reads and returns the contents of a file
- `run_shell(command, timeout?)` — executes a shell command and returns stdout + stderr (default timeout: 30s)

Tool calls are displayed in the terminal before execution so you can see what the agent is doing.

## Project Structure

```
agent/
  cli.py          # click CLI entry point, kiro-agent command
  config.py       # Config dataclass, env/dotenv loading
  repl.py         # REPL loop and conversation history management
  llm_client.py   # OpenAI-compatible streaming client with tool support
  tools.py        # Tool registry: read_file and run_shell
  formatter.py    # Rich-based terminal output helpers
  __main__.py     # python -m agent entry point
tests/            # Unit and property-based tests (hypothesis)
pyproject.toml
```

## Running Tests

```bash
pip install pytest hypothesis
pytest
```

## Using with Ollama

Set `LLM_BASE_URL` to your Ollama endpoint and `LLM_MODEL` to the model you have pulled:

```env
LLM_API_KEY=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=llama3
```
