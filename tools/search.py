import os
import subprocess
import time
from typing import Optional
from tools.base import BaseTool
from core.types import ToolResult


class SearchCodeTool(BaseTool):
    name = "search_code"
    description = (
        "Fast codebase search using ripgrep (rg). Searches file contents for regex or text patterns, "
        "returning matching lines with file paths and line numbers."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Regex or string pattern to search for."},
            "path": {"type": "string", "description": "Directory or file path to search within.", "default": "."},
            "file_pattern": {"type": "string", "description": "Optional glob filter, e.g. '*.py' or '*.ts'."}
        },
        "required": ["query"]
    }

    async def execute(self, query: str, path: str = ".", file_pattern: Optional[str] = None, **kwargs) -> ToolResult:
        start_time = time.time()
        search_path = os.path.abspath(os.path.expanduser(path))

        cmd = ["rg", "--line-number", "--no-heading", "--color=never", "--max-count=50"]
        if file_pattern:
            cmd.extend(["-g", file_pattern])
        cmd.extend([query, search_path])

        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            output = res.stdout.strip()
            if not output and res.returncode == 1:
                return ToolResult(
                    tool_name=self.name,
                    content=f"No matches found for query '{query}' in {path}.",
                    execution_time=time.time() - start_time
                )
            if res.returncode not in (0, 1) and res.stderr:
                return ToolResult(
                    tool_name=self.name,
                    content=f"ripgrep error: {res.stderr.strip()}",
                    is_error=True,
                    execution_time=time.time() - start_time
                )

            # Limit output lines if too verbose
            lines = output.splitlines()
            if len(lines) > 60:
                output = "\n".join(lines[:60]) + f"\n\n... [{len(lines) - 60} additional matches omitted]"

            return ToolResult(
                tool_name=self.name,
                content=output or "No matches found.",
                execution_time=time.time() - start_time
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                content=f"Search failed: {str(e)}",
                is_error=True,
                execution_time=time.time() - start_time
            )


class FindFilesTool(BaseTool):
    name = "find_files"
    description = "Locate files and folders matching a filename pattern or extension using fd/find."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob or name pattern to search for (e.g. '*.json', 'config*')."},
            "path": {"type": "string", "description": "Root path to search from.", "default": "."}
        },
        "required": ["pattern"]
    }

    async def execute(self, pattern: str, path: str = ".", **kwargs) -> ToolResult:
        start_time = time.time()
        search_path = os.path.abspath(os.path.expanduser(path))

        cmd = ["fd", "--glob", "--hidden", "--exclude", ".git", "--max-results", "60", pattern, search_path]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            output = res.stdout.strip()
            if not output:
                return ToolResult(
                    tool_name=self.name,
                    content=f"No files matching '{pattern}' found in {path}.",
                    execution_time=time.time() - start_time
                )
            return ToolResult(
                tool_name=self.name,
                content=output,
                execution_time=time.time() - start_time
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                content=f"find_files failed: {str(e)}",
                is_error=True,
                execution_time=time.time() - start_time
            )


class ListDirTool(BaseTool):
    name = "list_dir"
    description = "List files and directories in a directory with file types and sizes."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory path to list.", "default": "."}
        }
    }

    async def execute(self, path: str = ".", **kwargs) -> ToolResult:
        start_time = time.time()
        target_path = os.path.abspath(os.path.expanduser(path))

        if not os.path.exists(target_path):
            return ToolResult(
                tool_name=self.name,
                content=f"Error: Directory '{path}' does not exist.",
                is_error=True,
                execution_time=time.time() - start_time
            )

        if not os.path.isdir(target_path):
            return ToolResult(
                tool_name=self.name,
                content=f"Error: '{path}' is a file, not a directory.",
                is_error=True,
                execution_time=time.time() - start_time
            )

        try:
            entries = sorted(os.listdir(target_path))
            dirs = []
            files = []
            for entry in entries:
                if entry.startswith(".") and entry not in (".env", ".gitignore", ".cursorrules"):
                    continue
                full = os.path.join(target_path, entry)
                if os.path.isdir(full):
                    dirs.append(f"📁 {entry}/")
                else:
                    size = os.path.getsize(full)
                    size_str = f"{size} B" if size < 1024 else f"{size / 1024:.1f} KB"
                    files.append(f"📄 {entry} ({size_str})")

            summary = [f"Directory: {path}"]
            summary.extend(dirs)
            summary.extend(files)
            return ToolResult(
                tool_name=self.name,
                content="\n".join(summary),
                execution_time=time.time() - start_time
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                content=f"list_dir error: {str(e)}",
                is_error=True,
                execution_time=time.time() - start_time
            )
