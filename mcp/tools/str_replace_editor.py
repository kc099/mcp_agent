"""File and directory manipulation tool with sandbox support."""

from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, List, Literal, Optional, get_args

from .base import BaseTool, CLIResult, ToolResult, ToolError

Command = Literal[
    "view",
    "create",
    "str_replace",
    "insert",
    "undo_edit",
]

# Constants
SNIPPET_LINES: int = 4
MAX_RESPONSE_LEN: int = 16000
TRUNCATED_MESSAGE: str = (
    "<response clipped><NOTE>To save on context only part of this file has been shown to you. "
    "You should retry this tool after you have searched inside the file with `grep -n` "
    "in order to find the line numbers of what you are looking for.</NOTE>"
)

# Tool description
_STR_REPLACE_EDITOR_DESCRIPTION = """Custom editing tool for viewing, creating and editing files
* State is persistent across command calls and discussions with the user
* If `path` is a file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
* The `create` command cannot be used if the specified `path` already exists as a file
* If a `command` generates a long output, it will be truncated and marked with `<response clipped>`
* The `undo_edit` command will revert the last edit made to the file at `path`

Notes for using the `str_replace` command:
* The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
* If the `old_str` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `old_str` to make it unique
* The `new_str` parameter should contain the edited lines that should replace the `old_str`
"""


def maybe_truncate(
    content: str, truncate_after: Optional[int] = MAX_RESPONSE_LEN
) -> str:
    """Truncate content and append a notice if content exceeds the specified length."""
    if not truncate_after or len(content) <= truncate_after:
        return content
    return content[:truncate_after] + TRUNCATED_MESSAGE


class StrReplaceEditor(BaseTool):
    """A tool for viewing, creating, and editing files with sandbox support."""

    name: str = "str_replace_editor"
    description: str = _STR_REPLACE_EDITOR_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "command": {
                "description": "The commands to run. Allowed options are: `view`, `create`, `str_replace`, `insert`, `undo_edit`.",
                "enum": ["view", "create", "str_replace", "insert", "undo_edit"],
                "type": "string",
            },
            "path": {
                "description": "Absolute path to file or directory.",
                "type": "string",
            },
            "file_text": {
                "description": "Required parameter of `create` command, with the content of the file to be created.",
                "type": "string",
            },
            "old_str": {
                "description": "Required parameter of `str_replace` command containing the string in `path` to replace.",
                "type": "string",
            },
            "new_str": {
                "description": "Optional parameter of `str_replace` command containing the new string (if not given, no string will be added). Required parameter of `insert` command containing the string to insert.",
                "type": "string",
            },
            "insert_line": {
                "description": "Required parameter of `insert` command. The `new_str` will be inserted AFTER the line `insert_line` of `path`.",
                "type": "integer",
            },
            "view_range": {
                "description": "Optional parameter of `view` command when `path` points to a file. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file.",
                "items": {"type": "integer"},
                "type": "array",
            },
        },
        "required": ["command", "path"],
    }
    _file_history: DefaultDict[str, List[str]] = defaultdict(list)

    async def execute(
        self,
        *,
        command: Command,
        path: str,
        file_text: str | None = None,
        view_range: list[int] | None = None,
        old_str: str | None = None,
        new_str: str | None = None,
        insert_line: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Execute a file operation command."""
        # Validate path and command combination
        await self.validate_path(command, Path(path))

        # Execute the appropriate command
        if command == "view":
            result = await self.view(path, view_range)
        elif command == "create":
            if file_text is None:
                raise ToolError("Parameter `file_text` is required for command: create")
            await self.create_file(path, file_text)
            result = ToolResult(output=f"File created successfully at: {path}")
        elif command == "str_replace":
            if old_str is None:
                raise ToolError(
                    "Parameter `old_str` is required for command: str_replace"
                )
            result = await self.str_replace(path, old_str, new_str)
        elif command == "insert":
            if insert_line is None:
                raise ToolError(
                    "Parameter `insert_line` is required for command: insert"
                )
            if new_str is None:
                raise ToolError("Parameter `new_str` is required for command: insert")
            result = await self.insert(path, insert_line, new_str)
        elif command == "undo_edit":
            result = await self.undo_edit(path)
        else:
            # This should be caught by type checking, but we include it for safety
            raise ToolError(
                f'Unrecognized command {command}. The allowed commands for the {self.name} tool are: {", ".join(get_args(Command))}'
            )

        return str(result)

    async def validate_path(
        self, command: str, path: Path
    ) -> None:
        """Validate path and command combination based on execution environment."""
        # Check if path is absolute
        if not path.is_absolute():
            raise ToolError(f"The path {path} is not an absolute path")

        # Only check if path exists for non-create commands
        if command != "create":
            if not path.exists():
                raise ToolError(
                    f"The path {path} does not exist. Please provide a valid path."
                )

            # Check if path is a directory
            is_dir = path.is_dir()
            if is_dir and command != "view":
                raise ToolError(
                    f"The path {path} is a directory and only the `view` command can be used on directories"
                )

        # Check if file exists for create command
        elif command == "create":
            if path.exists():
                raise ToolError(
                    f"File already exists at: {path}. Cannot overwrite files using command `create`."
                )

    async def view(
        self,
        path: str,
        view_range: Optional[List[int]] = None,
    ) -> CLIResult:
        """Display file or directory content."""
        path_obj = Path(path)
        # Determine if path is a directory
        is_dir = path_obj.is_dir()

        if is_dir:
            # Directory handling
            if view_range:
                raise ToolError(
                    "The `view_range` parameter is not allowed when `path` points to a directory."
                )

            return await self._view_directory(path_obj)
        else:
            # File handling
            return await self._view_file(path_obj, view_range)

    @staticmethod
    async def _view_directory(path: Path) -> CLIResult:
        """Display directory contents."""
        try:
            # List files and directories up to 2 levels deep
            files = []
            for root, dirs, filenames in path.walk():
                rel_path = root.relative_to(path)
                if str(rel_path) == '.':
                    continue
                if len(rel_path.parts) > 2:
                    continue
                for d in dirs:
                    if not d.startswith('.'):
                        files.append(str(rel_path / d))
                for f in filenames:
                    if not f.startswith('.'):
                        files.append(str(rel_path / f))
            
            output = f"Here's the files and directories up to 2 levels deep in {path}, excluding hidden items:\n"
            output += "\n".join(sorted(files))
            return CLIResult(output=output)
        except Exception as e:
            return CLIResult(error=str(e))

    async def _view_file(
        self,
        path: Path,
        view_range: Optional[List[int]] = None,
    ) -> CLIResult:
        """Display file content, optionally within a specified line range."""
        try:
            # Read file content
            file_content = path.read_text()
            lines = file_content.splitlines()
            
            # Apply view range if specified
            if view_range:
                if len(view_range) != 2 or not all(isinstance(i, int) for i in view_range):
                    raise ToolError(
                        "Invalid `view_range`. It should be a list of two integers."
                    )
                start, end = view_range
                if start < 1 or end > len(lines):
                    raise ToolError(
                        f"Invalid line range. File has {len(lines)} lines."
                    )
                if end == -1:
                    end = len(lines)
                lines = lines[start - 1:end]
            
            # Format output with line numbers
            output = []
            for i, line in enumerate(lines, start=1):
                output.append(f"{i:6d}  {line}")
            
            return CLIResult(output="\n".join(output))
        except Exception as e:
            return CLIResult(error=str(e))

    async def create_file(self, path: str, content: str) -> None:
        """Create a new file with the given content."""
        try:
            Path(path).write_text(content)
            self._file_history[path].append(content)
        except Exception as e:
            raise ToolError(f"Failed to create file: {e}")

    async def str_replace(
        self,
        path: str,
        old_str: str,
        new_str: Optional[str] = None,
    ) -> CLIResult:
        """Replace a string in a file."""
        try:
            path_obj = Path(path)
            content = path_obj.read_text()
            
            if old_str not in content:
                return CLIResult(error=f"String '{old_str}' not found in file.")
            
            new_content = content.replace(old_str, new_str or "")
            path_obj.write_text(new_content)
            self._file_history[path].append(new_content)
            
            return CLIResult(output=f"Successfully replaced string in {path}")
        except Exception as e:
            return CLIResult(error=str(e))

    async def insert(
        self,
        path: str,
        insert_line: int,
        new_str: str,
    ) -> CLIResult:
        """Insert text at a specific line in a file."""
        try:
            path_obj = Path(path)
            content = path_obj.read_text()
            lines = content.splitlines()
            
            if insert_line < 1 or insert_line > len(lines) + 1:
                raise ToolError(
                    f"Invalid line number. File has {len(lines)} lines."
                )
            
            lines.insert(insert_line - 1, new_str)
            new_content = "\n".join(lines)
            path_obj.write_text(new_content)
            self._file_history[path].append(new_content)
            
            return CLIResult(output=f"Successfully inserted text at line {insert_line}")
        except Exception as e:
            return CLIResult(error=str(e))

    async def undo_edit(
        self, path: str
    ) -> CLIResult:
        """Undo the last edit made to a file."""
        try:
            if path not in self._file_history or len(self._file_history[path]) < 2:
                return CLIResult(error="No previous version to restore.")
            
            # Remove the current version and restore the previous one
            self._file_history[path].pop()
            previous_content = self._file_history[path][-1]
            
            Path(path).write_text(previous_content)
            return CLIResult(output=f"Successfully restored previous version of {path}")
        except Exception as e:
            return CLIResult(error=str(e)) 