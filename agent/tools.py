"""Tool registry and tool implementations for the CLI AI agent."""

import subprocess


def read_file(path: str) -> str:
    """Read the contents of a file at the given path.

    Returns the file contents as a string, or an error message string on failure.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file '{path}': {e}"


def run_shell(command: str, timeout: int = 30) -> str:
    """Execute a shell command and return its stdout + stderr output.

    Returns the combined output as a string, or an error message string on failure.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout
        if result.stderr:
            output += result.stderr
        return output
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout} seconds"
    except Exception as e:
        return f"Error running shell command: {e}"


class ToolRegistry:
    """Registry that holds tool definitions and dispatches tool calls."""

    _DEFINITIONS = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the contents of a file at the given path.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute or relative file path",
                        }
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_shell",
                "description": "Execute a shell command and return its stdout and stderr output.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The shell command to execute",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Timeout in seconds (default: 30)",
                        },
                    },
                    "required": ["command"],
                },
            },
        },
    ]

    def get_definitions(self) -> list[dict]:
        """Return the OpenAI function-calling schema for all registered tools."""
        return self._DEFINITIONS

    def execute(self, name: str, arguments: dict) -> str:
        """Dispatch a tool call by name with the given arguments.

        Returns the tool result as a string, or an error message string on failure.
        """
        try:
            if name == "read_file":
                return read_file(**arguments)
            elif name == "run_shell":
                return run_shell(**arguments)
            else:
                return f"Error: unknown tool '{name}'"
        except Exception as e:
            return f"Error executing tool '{name}': {e}"
