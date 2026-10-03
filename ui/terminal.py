import sys
from typing import Any, Dict, Optional
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.theme import Theme
from rich.table import Table

custom_theme = Theme({
    "info": "dim cyan",
    "warning": "bold yellow",
    "danger": "bold red",
    "success": "bold green",
    "agent": "bold bright_blue",
    "tool": "bold magenta",
    "diff_add": "green",
    "diff_sub": "red",
})

console = Console(theme=custom_theme)


def print_banner(model_name: str, cwd: str, backend_info: str) -> None:
    banner_text = (
        f"[bold bright_cyan]⚡ ANTIGRAVITY GODMODE CLI[/bold bright_cyan] [dim](Autonomous AI Coding Agent)[/dim]\n\n"
        f"• [bold white]Active Model:[/bold white] [bright_green]{model_name}[/bright_green]\n"
        f"• [bold white]Hardware/Backend:[/bold white] [yellow]{backend_info}[/yellow]\n"
        f"• [bold white]Working Directory:[/bold white] [dim]{cwd}[/dim]\n"
        f"• [bold white]Capabilities:[/bold white] [magenta]Persistent Bash | AST Code Patching | Ripgrep | Live Web | Liquid Self-Refinement[/magenta]\n\n"
        f"[dim]Type your request, or use [bold]/help[/bold], [bold]/model[/bold], [bold]/status[/bold], [bold]/clear[/bold], [bold]/exit[/bold][/dim]"
    )
    console.print(Panel(banner_text, border_style="bright_blue", expand=False))


def print_user_query(query: str) -> None:
    console.print(f"\n[bold bright_green]❯ User:[/bold bright_green] {query}")


def print_agent_thought(content: str) -> None:
    if content.strip():
        console.print(Panel(Markdown(content), title="[bold bright_blue]🤖 Agent[/bold bright_blue]", border_style="blue", padding=(0, 1)))


def print_tool_call(tool_name: str, arguments: Dict[str, Any]) -> None:
    arg_summary = []
    for k, v in arguments.items():
        v_str = str(v)
        if len(v_str) > 80:
            v_str = v_str[:80] + "..."
        arg_summary.append(f"[cyan]{k}[/cyan]=[white]{v_str}[/white]")
    
    args_display = ", ".join(arg_summary) if arg_summary else "no arguments"
    console.print(f"  [tool]⚙ Tool Call:[/tool] [bold bright_yellow]{tool_name}[/bold bright_yellow]({args_display})")


def print_tool_result(tool_name: str, content: str, is_error: bool, execution_time: float) -> None:
    color = "danger" if is_error else "dim white"
    status_icon = "❌" if is_error else "✓"
    time_str = f"({execution_time:.2f}s)"

    # If the tool is replace_file_content and output contains diff, render syntax highlighted diff
    if "```diff" in content:
        parts = content.split("```diff")
        prefix = parts[0].strip()
        diff_body = parts[1].split("```")[0].strip()
        if prefix:
            console.print(f"    {status_icon} [{color}]{prefix}[/{color}] [dim]{time_str}[/dim]")
        console.print(Syntax(diff_body, "diff", theme="monokai", line_numbers=False))
        return

    # Normal tool output
    lines = content.strip().splitlines()
    if len(lines) > 8:
        preview = "\n".join(lines[:6]) + f"\n... [+{len(lines) - 6} more lines]"
    else:
        preview = "\n".join(lines)

    console.print(f"    {status_icon} [{color}]{preview}[/{color}] [dim]{time_str}[/dim]")


def print_status(msg: str) -> None:
    console.print(f"[dim]• {msg}[/dim]")


def print_error(msg: str) -> None:
    console.print(f"[danger]Error: {msg}[/danger]")


def print_success(msg: str) -> None:
    console.print(f"[success]✓ {msg}[/success]")
