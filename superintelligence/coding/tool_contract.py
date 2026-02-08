"""
GeniusPro Superintelligence — Cursor coding tool contract

Defines a stable set of client-executed tools that a Cursor-integrated
client can provide to the coding endpoint.

These tools are NOT executed by the Superintelligence server. They are
described to the model so it can request actions via tool calls; the client
is responsible for running them and returning results back as tool messages.
"""

from __future__ import annotations


def get_cursor_coding_tool_definitions() -> list[dict]:
    """
    OpenAI-format tool definitions.

    Keep these names stable; Cursor clients can implement them however they want
    (terminal, filesystem APIs, MCP servers, etc).
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file from the workspace. Use this before proposing edits when file contents matter.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path"},
                        "start_line": {"type": "integer", "description": "1-indexed start line (optional)"},
                        "end_line": {"type": "integer", "description": "1-indexed end line (optional)"},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_directory",
                "description": "List files/directories for a path. Use to understand repo structure.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Workspace-relative directory path. Use '.' for workspace root."},
                        "max_depth": {"type": "integer", "description": "Max recursion depth (optional)"},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_code",
                "description": "Search workspace code. Prefer this over guessing symbol names.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query or regex"},
                        "path": {"type": "string", "description": "Optional path scope"},
                        "is_regex": {"type": "boolean", "description": "Treat query as regex (default false)"},
                        "glob": {"type": "string", "description": "Optional glob filter, e.g. **/*.py"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "apply_patch",
                "description": "Apply a unified diff/patch to files in the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "patch": {"type": "string", "description": "Unified diff patch"},
                    },
                    "required": ["patch"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_terminal",
                "description": "Run a terminal command. Use this for builds, tests, formatting, git, and gh.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Command to run"},
                        "cwd": {"type": "string", "description": "Working directory (optional)"},
                    },
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "gh",
                "description": "Run GitHub CLI commands (preferred for GitHub interactions). Provide args exactly as you would after `gh`.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "args": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "CLI args, e.g. ['pr','create','--title','...']",
                        }
                    },
                    "required": ["args"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "open_windows_credential_manager_popup",
                "description": "On Windows: clear cached git GitHub credentials, set git to use Windows Credential Manager, and trigger an auth-required git operation to open the account popup.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "remote": {"type": "string", "description": "Remote name, default 'origin'"},
                        "branch": {"type": "string", "description": "Branch to push, default current"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "save_memory_snippet",
                "description": "Save a user-approved snippet to long-term memory. Only call this when the user explicitly approves saving.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "project_slug": {"type": "string", "description": "Project identifier slug"},
                        "language": {"type": "string", "description": "Language (optional)"},
                        "content": {"type": "string", "description": "The approved snippet text"},
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional tags",
                        },
                    },
                    "required": ["project_slug", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "load_memory_snippets",
                "description": "Load previously saved user-approved snippets for a project.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "project_slug": {"type": "string", "description": "Project identifier slug"},
                        "limit": {"type": "integer", "description": "Max snippets to return (default 20)"},
                    },
                    "required": ["project_slug"],
                },
            },
        },
    ]


def merge_client_tools(*, request_tools: list[dict], injected_tools: list[dict]) -> list[dict]:
    """
    Merge request-provided tools with injected tools, deduping by function name.
    Request tools win on conflicts (client may override descriptions/schema).
    """
    merged: dict[str, dict] = {}

    def _name(tool: dict) -> str:
        if tool.get("type") != "function":
            return ""
        return tool.get("function", {}).get("name", "") or ""

    for t in injected_tools:
        name = _name(t)
        if name:
            merged[name] = t

    for t in request_tools or []:
        name = _name(t)
        if name:
            merged[name] = t

    return list(merged.values())

