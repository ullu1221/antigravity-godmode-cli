#!/usr/bin/env python3
import sys
import os
import asyncio
import yaml
from pathlib import Path
from typing import Dict, Any

from core.engine import LLMEngine
from core.agent import AutonomousAgent
from tools import build_tool_registry
from ui.terminal import (
    console,
    print_banner,
    print_user_query,
    print_status,
    print_error,
    print_success
)
from ui.prompt import AgentPrompt
from rich.table import Table
from rich.markdown import Markdown


def load_config() -> Dict[str, Any]:
    config_path = Path(__file__).parent / "config" / "default_config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {
        "active_model": "qwen2.5-coder:7b-instruct-q5_K_M",
        "endpoints": {"ollama": {"base_url": "http://localhost:11434"}},
        "agent": {"max_iterations": 30, "max_history_turns": 40},
        "tools": {"enable_bash": True, "enable_files": True, "enable_search": True, "enable_web": True}
    }


def show_help():
    help_md = r"""
### ⚡ Godmode CLI Commands & Shortcuts

| Command | Description |
| :--- | :--- |
| **`/help`** | Display this command overview |
| **`/model [name]`** | Show available models or switch the active model |
| **`/coder`** | Quick-switch to Qwen2.5-Coder-7B (100% GPU, ~38 tok/s) |
| **`/uncensored`** or **`/agent`** | Quick-switch to Hermes-3-8B (100% Uncensored agent, ~35 tok/s) |
| **`/reasoner`** | Quick-switch to Qwen2.5-14B (Deep hybrid reasoning, ~9.5 tok/s) |
| **`/fast`** | Quick-switch to Gemma-4-e4B (Ultra-low latency, ~70 tok/s) |
| **`/tools`** | List all registered tools and their signatures |
| **`/status`** | View local LLM backend health, VRAM, and context usage |
| **`/desktop [on\|off]`** | Toggle Wayland desktop control (screenshots & ydotool) |
| **`/clear`** | Clear conversation context window and reset state |
| **`/cwd [path]`** | Print or change current working directory |
| **`/exit`** or **`/quit`** | Exit the Godmode CLI session |

**Autonomous Capabilities:**
- Type any task in plain English (e.g. *"Inspect the weather app in Projects/04-weather-app, run tests, and fix any failing edge cases"*).
- The agent reads files, searches code with ripgrep, runs bash commands, edits files, and self-heals until done.
"""
    console.print(Markdown(help_md))


def show_tools(registry):
    table = Table(title="🛠 Registered Autonomous Tools", border_style="bright_blue")
    table.add_column("Tool Name", style="bold cyan")
    table.add_column("Description", style="white")
    for name, tool in registry._tools.items():
        table.add_row(name, tool.description[:110] + ("..." if len(tool.description) > 110 else ""))
    console.print(table)


import argparse

async def main():
    parser = argparse.ArgumentParser(
        description="⚡ Antigravity Godmode CLI: Autonomous AI Coding Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("prompt", nargs="*", help="Optional one-shot task or query to execute autonomously")
    parser.add_argument("-m", "--model", help="Override the active model (e.g. hermes3:8b, qwen2.5-coder:7b-instruct-q5_K_M)")
    parser.add_argument("-d", "--cwd", help="Set the initial working directory")
    parser.add_argument("--endpoint", "--base-url", dest="endpoint", help="Custom LLM API base URL (e.g. https://openrouter.ai/api/v1)")
    parser.add_argument("--api-key", help="API key for custom/cloud LLM endpoint (or set OPENROUTER_API_KEY / DEEPSEEK_API_KEY)")
    parser.add_argument("--tools", action="store_true", help="Print available tools and exit")
    parser.add_argument("--status", action="store_true", help="Check backend status and exit")

    parsed_args = parser.parse_args()

    config = load_config()
    endpoint_url = parsed_args.endpoint or config.get("endpoints", {}).get("ollama", {}).get("base_url", "http://localhost:11434")
    active_model = parsed_args.model or config.get("active_model", "qwen2.5-coder:7b-instruct-q5_K_M")
    initial_cwd = os.path.abspath(parsed_args.cwd) if parsed_args.cwd else os.getcwd()
    if os.path.isdir(initial_cwd):
        os.chdir(initial_cwd)

    # Initialize Engine & Tools
    engine = LLMEngine(base_url=endpoint_url, default_model=active_model, api_key=parsed_args.api_key)
    registry = build_tool_registry(config, initial_cwd=initial_cwd)
    agent = AutonomousAgent(config, registry, engine, initial_cwd=initial_cwd)
    agent.set_model(active_model)

    if parsed_args.tools:
        show_tools(registry)
        return

    # Health check
    healthy, info = await engine.check_health()
    if not healthy:
        # Fallback check for llama-server
        hybrid_url = config.get("endpoints", {}).get("llama_cpp", {}).get("base_url", "http://localhost:8080/v1")
        hybrid_engine = LLMEngine(base_url=hybrid_url, default_model="Qwen2.5-14B-Instruct-Q4_K_M")
        h_ok, h_info = await hybrid_engine.check_health()
        if h_ok:
            engine = hybrid_engine
            agent.engine = hybrid_engine
            agent.set_model("Qwen2.5-14B-Instruct-Q4_K_M")
            active_model = "Qwen2.5-14B-Instruct-Q4_K_M"
            info = f"Hybrid llama-server ({h_info})"
        else:
            print_error(f"Cannot connect to local LLM backend at {endpoint_url} or {hybrid_url}.\nEnsure Ollama is running (`ollama serve`) or run `~/run_hybrid.sh`.")
            sys.exit(1)

    if parsed_args.status:
        console.print(f"• Backend Status: [green]{info}[/green]")
        console.print(f"• Active Model: [cyan]{active_model}[/cyan]")
        console.print(f"• CWD: [yellow]{initial_cwd}[/yellow]")
        return

    # Non-interactive one-shot mode
    if parsed_args.prompt:
        one_shot_prompt = " ".join(parsed_args.prompt)
        console.print(f"[bold bright_cyan]Running one-shot command:[/bold bright_cyan] {one_shot_prompt}")
        await agent.run_turn(one_shot_prompt)
        return

    # Interactive CLI Mode
    print_banner(active_model, agent.cwd, info)
    prompt_ui = AgentPrompt()

    while True:
        try:
            user_input = await prompt_ui.get_input(agent.cwd)
        except Exception:
            break

        if not user_input:
            continue

        # Handle Slash Commands
        if user_input.startswith("/"):
            parts = user_input.split()
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd in ("/exit", "/quit", "/q"):
                console.print("[dim]👋 Exiting Godmode CLI. Happy hacking![/dim]")
                break
            elif cmd == "/help":
                show_help()
            elif cmd == "/tools":
                show_tools(agent.tool_registry)
            elif cmd == "/clear":
                agent.clear_context()
                print_success("Context history cleared.")
            elif cmd in ("/model", "/models"):
                if arg:
                    agent.set_model(arg)
                    print_success(f"Switched active model to: [bold]{arg}[/bold]")
                else:
                    table = Table(title="🤖 Available Local AI Models", border_style="bright_blue")
                    table.add_column("Model Name", style="bold cyan")
                    table.add_column("Type", style="green")
                    table.add_column("Size", style="yellow")
                    table.add_column("Status", style="magenta")

                    # Query Ollama models via API
                    try:
                        import httpx
                        async with httpx.AsyncClient(timeout=3.0) as client:
                            resp = await client.get(f"{agent.engine.base_url}/api/tags")
                            if resp.status_code == 200:
                                for m in resp.json().get("models", []):
                                    m_name = m.get("name", "")
                                    m_size_bytes = m.get("size", 0)
                                    m_size_gb = f"{m_size_bytes / (1024**3):.1f} GB" if m_size_bytes else "N/A"
                                    is_active = (m_name == agent.active_model)
                                    status = "● Active" if is_active else "Available"
                                    table.add_row(m_name, "Ollama", m_size_gb, status)
                    except Exception:
                        pass

                    # Query ~/models directory for local GGUF models
                    models_dir = Path.home() / "models"
                    if models_dir.exists():
                        for gguf in sorted(models_dir.glob("*.gguf")):
                            size_gb = f"{gguf.stat().st_size / (1024**3):.1f} GB"
                            is_active = (gguf.stem == agent.active_model)
                            status = "● Active" if is_active else "Available (llama.cpp / g15)"
                            table.add_row(gguf.name, "GGUF", size_gb, status)

                    console.print(table)
                    console.print(f"Current active model: [bold green]{agent.active_model}[/bold green]")
                    console.print("[dim]Switch active model with: /model <name>[/dim]")
            elif cmd == "/coder":
                agent.set_model("qwen2.5-coder:7b-instruct-q5_K_M")
                print_success("Switched to [bold cyan]Coder Preset[/bold cyan] (Qwen2.5-Coder-7B: 100% GPU, ~38 tok/s)")
            elif cmd in ("/agent", "/uncensored"):
                agent.set_model("hermes3:8b")
                print_success("Switched to [bold magenta]Uncensored Agent Preset[/bold magenta] (Hermes-3-8B: 100% Uncensored, ~35 tok/s)")
            elif cmd == "/reasoner":
                agent.set_model("qwen2.5-14b:latest")
                print_success("Switched to [bold yellow]Reasoner Preset[/bold yellow] (Qwen2.5-14B: Hybrid 14B Reasoning, ~9.5 tok/s)")
            elif cmd == "/fast":
                agent.set_model("gemma4:e4b")
                print_success("Switched to [bold green]Fast Preset[/bold green] (Gemma-4-e4B: Ultra-Low Latency, ~70 tok/s)")
            elif cmd == "/desktop":
                if arg.lower() in ("on", "true", "enable"):
                    config["tools"]["enable_desktop"] = True
                    agent.tool_registry = build_tool_registry(config, initial_cwd=agent.cwd)
                    print_success("Desktop tools enabled (screenshot, mouse, keyboard).")
                elif arg.lower() in ("off", "false", "disable"):
                    config["tools"]["enable_desktop"] = False
                    agent.tool_registry = build_tool_registry(config, initial_cwd=agent.cwd)
                    print_success("Desktop tools disabled.")
                else:
                    console.print("Usage: /desktop on | /desktop off")
            elif cmd == "/status":
                h, msg = await agent.engine.check_health()
                console.print(f"• Backend Status: [green]{msg}[/green]")
                console.print(f"• Active Model: [cyan]{agent.active_model}[/cyan]")
                console.print(f"• Working Directory: [yellow]{agent.cwd}[/yellow]")
                console.print(f"• Context Messages: [magenta]{len(agent.context.messages)}[/magenta]")
            elif cmd == "/cwd":
                if arg:
                    new_cwd = os.path.abspath(os.path.expanduser(arg))
                    if os.path.isdir(new_cwd):
                        agent.cwd = new_cwd
                        os.chdir(new_cwd)
                        # Update bash tool cwd as well
                        bash = agent.tool_registry.get("run_command")
                        if bash:
                            bash.cwd = new_cwd
                        print_success(f"CWD set to: {agent.cwd}")
                    else:
                        print_error(f"Not a valid directory: {arg}")
                else:
                    console.print(f"CWD: {agent.cwd}")
            else:
                print_error(f"Unknown command '{cmd}'. Type /help for available commands.")
            continue

        # Execute autonomous turn
        print_user_query(user_input)
        await agent.run_turn(user_input)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[dim]Session terminated by user.[/dim]")
