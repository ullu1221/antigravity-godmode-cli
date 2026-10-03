import os
import difflib
import time
from typing import Optional
from tools.base import BaseTool
from core.types import ToolResult


class ViewFileTool(BaseTool):
    name = "view_file"
    description = (
        "View the contents of a file on the local filesystem with 1-indexed line numbers. "
        "Supports line slicing via start_line and end_line."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute or relative file path."},
            "start_line": {"type": "integer", "description": "1-indexed starting line number (optional)."},
            "end_line": {"type": "integer", "description": "1-indexed ending line number (optional)."}
        },
        "required": ["path"]
    }

    async def execute(self, path: Optional[str] = None, start_line: Optional[int] = None, end_line: Optional[int] = None, **kwargs) -> ToolResult:
        start_time = time.time()
        path = path or kwargs.get("file") or kwargs.get("filename") or kwargs.get("target_file")
        if not path:
            return ToolResult(tool_name=self.name, content="Error: 'path' argument is required.", is_error=True, execution_time=time.time() - start_time)
        expanded_path = os.path.abspath(os.path.expanduser(path))

        if not os.path.exists(expanded_path):
            return ToolResult(
                tool_name=self.name,
                content=f"Error: File '{path}' does not exist.",
                is_error=True,
                execution_time=time.time() - start_time
            )

        if os.path.isdir(expanded_path):
            return ToolResult(
                tool_name=self.name,
                content=f"Error: '{path}' is a directory, not a regular file.",
                is_error=True,
                execution_time=time.time() - start_time
            )

        try:
            with open(expanded_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            total_lines = len(lines)
            s_line = max(1, start_line or 1)
            e_line = min(total_lines, end_line or total_lines)

            if s_line > total_lines:
                return ToolResult(
                    tool_name=self.name,
                    content=f"File '{path}' has {total_lines} lines; start_line {s_line} is out of range.",
                    is_error=True,
                    execution_time=time.time() - start_time
                )

            selected_lines = lines[s_line - 1 : e_line]
            formatted = [f"{s_line + i}: {line.rstrip()}" for i, line in enumerate(selected_lines)]
            header = f"File: {path} (Showing lines {s_line} to {e_line} of {total_lines})\n"
            content = header + "\n".join(formatted)

            return ToolResult(
                tool_name=self.name,
                content=content,
                execution_time=time.time() - start_time
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                content=f"Error reading file '{path}': {str(e)}",
                is_error=True,
                execution_time=time.time() - start_time
            )


class WriteFileTool(BaseTool):
    name = "write_file"
    description = "Create a new file or completely overwrite an existing file with the provided content."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Target file path."},
            "content": {"type": "string", "description": "The exact content to write to the file."},
            "overwrite": {"type": "boolean", "description": "Overwrite existing file if true.", "default": True}
        },
        "required": ["path", "content"]
    }

    async def execute(self, path: Optional[str] = None, content: Optional[str] = None, overwrite: bool = True, **kwargs) -> ToolResult:
        start_time = time.time()
        path = path or kwargs.get("file") or kwargs.get("filename") or kwargs.get("target_file")
        content = content if content is not None else kwargs.get("text") or kwargs.get("code") or kwargs.get("body") or ""
        if not path:
            return ToolResult(tool_name=self.name, content="Error: 'path' argument is required.", is_error=True, execution_time=time.time() - start_time)
        expanded_path = os.path.abspath(os.path.expanduser(path))

        if os.path.exists(expanded_path) and not overwrite:
            return ToolResult(
                tool_name=self.name,
                content=f"Error: File '{path}' already exists and overwrite is set to False.",
                is_error=True,
                execution_time=time.time() - start_time
            )

        try:
            os.makedirs(os.path.dirname(expanded_path), exist_ok=True)
            with open(expanded_path, "w", encoding="utf-8") as f:
                f.write(content)

            return ToolResult(
                tool_name=self.name,
                content=f"Successfully written {len(content)} characters ({len(content.splitlines())} lines) to '{path}'.",
                execution_time=time.time() - start_time
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                content=f"Error writing to file '{path}': {str(e)}",
                is_error=True,
                execution_time=time.time() - start_time
            )


def _find_normalized_match(content: str, target: str) -> tuple[Optional[str], int]:
    """Find target chunk in content ignoring differences in leading/trailing indentation or empty lines."""
    if target in content:
        return target, content.count(target)
    
    file_lines = content.splitlines(keepends=True)
    target_lines = [l.strip() for l in target.splitlines() if l.strip()]
    if not target_lines:
        return None, 0
    
    matches = []
    for i in range(len(file_lines)):
        matched_orig = []
        t_idx = 0
        for j in range(i, len(file_lines)):
            s = file_lines[j].strip()
            if not s:
                matched_orig.append(file_lines[j])
                continue
            if s == target_lines[t_idx]:
                matched_orig.append(file_lines[j])
                t_idx += 1
                if t_idx == len(target_lines):
                    matches.append("".join(matched_orig))
                    break
            else:
                break
    return (matches[0], len(matches)) if len(matches) == 1 else (None, len(matches))


def _adjust_indentation(orig_chunk: str, repl_chunk: str) -> str:
    """Adjust repl_chunk indentation to match orig_chunk base indentation."""
    orig_lines = orig_chunk.splitlines(keepends=True)
    if not orig_lines:
        return repl_chunk
    orig_indent = len(orig_lines[0]) - len(orig_lines[0].lstrip())
    
    repl_lines = repl_chunk.splitlines(keepends=True)
    if not repl_lines:
        return repl_chunk
    repl_first_indent = len(repl_lines[0]) - len(repl_lines[0].lstrip())
    
    diff_indent = orig_indent - repl_first_indent
    if diff_indent == 0:
        return repl_chunk
    
    adjusted = []
    for line in repl_lines:
        if not line.strip():
            adjusted.append(line)
        elif diff_indent > 0:
            adjusted.append(" " * diff_indent + line)
        else:
            to_remove = min(-diff_indent, len(line) - len(line.lstrip()))
            adjusted.append(line[to_remove:])
    return "".join(adjusted)


class ReplaceFileContentTool(BaseTool):
    name = "replace_file_content"
    description = (
        "Precision file editor: Replaces an exact target chunk of text within a specified line range "
        "with new replacement text. Prevents wiping or truncating large files."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Target file path to modify."},
            "target_content": {"type": "string", "description": "The exact string chunk in the file to be replaced."},
            "replacement_content": {"type": "string", "description": "The new replacement string."},
            "start_line": {"type": "integer", "description": "Optional 1-indexed starting line of search range."},
            "end_line": {"type": "integer", "description": "Optional 1-indexed ending line of search range."}
        },
        "required": ["path", "target_content", "replacement_content"]
    }

    async def execute(
        self,
        path: Optional[str] = None,
        target_content: Optional[str] = None,
        replacement_content: Optional[str] = None,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        **kwargs
    ) -> ToolResult:
        start_time = time.time()
        path = path or kwargs.get("file") or kwargs.get("filename") or kwargs.get("target_file")
        target_content = target_content if target_content is not None else (kwargs.get("target") or kwargs.get("old_content") or kwargs.get("search"))
        replacement_content = replacement_content if replacement_content is not None else (kwargs.get("replacement") or kwargs.get("new_content") or kwargs.get("replace") or kwargs.get("content"))

        if not path:
            return ToolResult(tool_name=self.name, content="Error: 'path' argument is required.", is_error=True, execution_time=time.time() - start_time)
        if target_content is None:
            return ToolResult(tool_name=self.name, content="Error: 'target_content' argument is required.", is_error=True, execution_time=time.time() - start_time)
        if replacement_content is None:
            return ToolResult(tool_name=self.name, content="Error: 'replacement_content' argument is required.", is_error=True, execution_time=time.time() - start_time)

        expanded_path = os.path.abspath(os.path.expanduser(path))

        if not os.path.exists(expanded_path):
            return ToolResult(
                tool_name=self.name,
                content=f"Error: File '{path}' does not exist.",
                is_error=True,
                execution_time=time.time() - start_time
            )

        try:
            with open(expanded_path, "r", encoding="utf-8") as f:
                original_text = f.read()

            lines = original_text.splitlines(keepends=True)
            total_lines = len(lines)

            # Determine search slice
            s_idx = max(0, (start_line - 1) if start_line else 0)
            e_idx = min(total_lines, end_line if end_line else total_lines)

            # Reconstruct the search window
            search_window = "".join(lines[s_idx:e_idx])
            actual_target = target_content
            actual_replacement = replacement_content

            if target_content not in search_window:
                # If bounded search failed, check the whole file for convenience
                if target_content in original_text:
                    s_idx = 0
                    e_idx = total_lines
                    search_window = original_text
                else:
                    # Try normalized line matching on search window, then full file
                    matched_chunk, count = _find_normalized_match(search_window, target_content)
                    if not matched_chunk:
                        matched_chunk, count = _find_normalized_match(original_text, target_content)
                        if matched_chunk:
                            s_idx = 0
                            e_idx = total_lines
                            search_window = original_text

                    if matched_chunk and count == 1:
                        actual_target = matched_chunk
                        actual_replacement = _adjust_indentation(matched_chunk, replacement_content)
                    else:
                        return ToolResult(
                            tool_name=self.name,
                            content=(
                                f"Error: target_content not found in '{path}'. "
                                f"Make sure exact indentation and whitespace match."
                            ),
                            is_error=True,
                            execution_time=time.time() - start_time
                        )

            occurrences = search_window.count(actual_target)
            if occurrences > 1:
                return ToolResult(
                    tool_name=self.name,
                    content=(
                        f"Error: target_content appears {occurrences} times in the search window. "
                        f"Specify narrower start_line / end_line or provide more surrounding context."
                    ),
                    is_error=True,
                    execution_time=time.time() - start_time
                )

            # Perform single replacement in the window
            modified_window = search_window.replace(actual_target, actual_replacement, 1)

            # Reassemble full file
            new_full_text = "".join(lines[:s_idx]) + modified_window + "".join(lines[e_idx:])

            # Generate diff preview
            orig_split = original_text.splitlines(keepends=True)
            new_split = new_full_text.splitlines(keepends=True)
            diff = "".join(difflib.unified_diff(orig_split, new_split, fromfile=f"a/{path}", tofile=f"b/{path}", n=2))

            # Write file back
            with open(expanded_path, "w", encoding="utf-8") as f:
                f.write(new_full_text)

            diff_summary = diff[:1500] + ("\n... [diff truncated]" if len(diff) > 1500 else "")
            return ToolResult(
                tool_name=self.name,
                content=f"Successfully patched '{path}':\n\n```diff\n{diff_summary}\n```",
                execution_time=time.time() - start_time
            )

        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                content=f"Error patching file '{path}': {str(e)}",
                is_error=True,
                execution_time=time.time() - start_time
            )
