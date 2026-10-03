import asyncio
import os
import time
from typing import Dict, Any
from tools.base import BaseTool
from core.types import ToolResult


class BashTool(BaseTool):
    name = "run_command"
    description = (
        "Execute a bash shell command. Supports cd (tracks current working directory across calls), "
        "process execution, git operations, testing, package management, and system commands."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The command line string to execute in bash."
            },
            "timeout": {
                "type": "integer",
                "description": "Optional timeout in seconds (default: 120).",
                "default": 120
            }
        },
        "required": ["command"]
    }

    def __init__(self, default_cwd: Optional[str] = None):
        self.cwd = os.path.abspath(default_cwd or os.getcwd())

    async def execute(self, command: str, timeout: int = 120, **kwargs) -> ToolResult:
        start_time = time.time()
        command = command.strip()

        # Handle simple standalone 'cd' commands directly to update working directory
        if command.startswith("cd ") and not any(op in command for op in ["&&", "||", ";", "|", "\n"]):
            target = command[3:].strip().strip('"').strip("'")
            new_path = os.path.abspath(os.path.join(self.cwd, os.path.expanduser(target)))
            if os.path.isdir(new_path):
                self.cwd = new_path
                return ToolResult(
                    tool_name=self.name,
                    content=f"Directory changed to: {self.cwd}",
                    execution_time=time.time() - start_time
                )
            else:
                return ToolResult(
                    tool_name=self.name,
                    content=f"cd error: No such directory: {target}",
                    is_error=True,
                    execution_time=time.time() - start_time
                )

        # Wrap command to preserve exit code while capturing directory changes
        wrapped_command = f"{command}\n__AGY_CODE=$?\npwd\nexit $__AGY_CODE"

        env = os.environ.copy()
        venv_bin = os.path.expanduser("~/agent-env/bin")
        if os.path.isdir(venv_bin):
            env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"

        try:
            process = await asyncio.create_subprocess_shell(
                wrapped_command,
                cwd=self.cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )

            try:
                stdout_data, stderr_data = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                return ToolResult(
                    tool_name=self.name,
                    content=f"Command timed out after {timeout} seconds.",
                    is_error=True,
                    execution_time=time.time() - start_time
                )

            stdout_str = stdout_data.decode("utf-8", errors="replace")
            stderr_str = stderr_data.decode("utf-8", errors="replace")

            # Extract the last line for pwd if present
            lines = stdout_str.rstrip("\r\n").splitlines()
            if lines:
                potential_pwd = lines[-1].strip()
                if os.path.isdir(potential_pwd):
                    if process.returncode == 0:
                        self.cwd = potential_pwd
                    stdout_str = "\n".join(lines[:-1]) + ("\n" if len(lines) > 1 else "")

            output_parts = []
            if stdout_str:
                output_parts.append(f"STDOUT:\n{stdout_str.rstrip()}")
            if stderr_str:
                output_parts.append(f"STDERR:\n{stderr_str.rstrip()}")
            
            output_parts.append(f"[Exit code: {process.returncode}] [CWD: {self.cwd}]")
            final_output = "\n".join(output_parts)

            return ToolResult(
                tool_name=self.name,
                content=final_output,
                is_error=(process.returncode != 0),
                execution_time=time.time() - start_time
            )

        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                content=f"Error running command '{command}': {str(e)}",
                is_error=True,
                execution_time=time.time() - start_time
            )
