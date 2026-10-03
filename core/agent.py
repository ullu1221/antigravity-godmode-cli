import os
import asyncio
from typing import Dict, Any, Optional, List
from core.types import Message, ToolCall, ToolResult, AgentResponse
from core.engine import LLMEngine
from core.context import ContextManager
from core.speculative import SpeculativeRacingEngine, LiquidRefinementEngine
from tools.base import ToolRegistry
from ui.terminal import (
    print_agent_thought,
    print_tool_call,
    print_tool_result,
    print_status,
    print_error,
    console
)

GODMODE_SYSTEM_PROMPT = """You are ANTIGRAVITY GODMODE — an elite, fully autonomous AI software engineering agent with direct, unrestricted execution capabilities on Linux.

YOUR RUNTIME ENVIRONMENT:
- OS: EndeavourOS Linux (Zen Kernel, Wayland)
- Hardware: Intel i5-13450HX (16 threads), NVIDIA RTX 3050 Laptop GPU (6GB VRAM), 16GB DDR5
- Shell: Bash with persistent working directory tracking
- Fast Search: ripgrep (rg) and fd pre-installed
- Toolset: run_command, view_file, write_file, replace_file_content, search_code, find_files, list_dir, web_search, read_url

YOUR CORE PRINCIPLES:
1. FULL AUTONOMY & ACTION MANDATE:
   - You do not just discuss, outline, or explain hypothetical steps. You EXECUTE them via tools.
   - Do NOT emit long bulleted plans repeating what you will do. Immediately invoke the tool.
   - STEP-BY-STEP EXECUTION:
     * When investigating a failure, run the test first: run_command(command="pytest ...")
     * To inspect source code, invoke: view_file(path="...")
     * NEVER guess code sight unseen or call replace_file_content in the same turn as view_file.
     * Wait for the tool output of view_file to inspect the exact lines and indentation.
     * In the next turn, invoke replace_file_content with the exact code snippet.
     * Then invoke run_command to verify tests pass 100%.
2. NO SHORTCUTS OR FAKE DATA:
   - When inspecting hardware, system metrics, configs, or source code, read genuine system files and execute real tools. Never invent dummy data.
3. PRECISION EDITING MANDATE:
   - Implementation files (e.g. priority_queue.py, circuit_breaker.py, metrics.py) contain algorithms, NOT unit test assertions.
   - Inspect the actual function in the implementation file with view_file, diagnose the algorithm/logic error, and patch that function.
4. AUTONOMOUS VERIFICATION & SELF-HEALING:
   - When debugging failing tests, run the test, read the traceback, view the failing module with view_file, fix the logic bug, and re-run the tests until 100% pass.
5. REAL-TIME WEB & CURRENT EVENTS:
   - When asked about news, current events, or documentation, invoke `web_search`.
   - `web_search` returns rich article summaries with dates and sources. Synthesize your answer directly from these summaries.
   - If a URL encounters CDN or bot protection, do not keep repeating failed fetches; use the facts from `web_search` to immediately formulate a comprehensive, informative response.
"""


class AutonomousAgent:
    def __init__(
        self,
        config: Dict[str, Any],
        tool_registry: ToolRegistry,
        engine: LLMEngine,
        initial_cwd: str = "."
    ):
        self.config = config
        self.tool_registry = tool_registry
        self.engine = engine
        self.cwd = os.path.abspath(initial_cwd)
        
        self.active_model = config.get("active_model", "qwen2.5-coder:7b-instruct-q5_K_M")
        self.max_iterations = config.get("agent", {}).get("max_iterations", 30)
        self.speculative_enabled = config.get("agent", {}).get("speculative_racing", True)

        self.context = ContextManager(
            system_prompt=GODMODE_SYSTEM_PROMPT,
            max_turns=config.get("agent", {}).get("max_history_turns", 40)
        )
        self.speculative = SpeculativeRacingEngine(self.engine)

    def set_model(self, model_name: str) -> None:
        self.active_model = model_name

    def clear_context(self) -> None:
        self.context.clear_history()

    async def run_turn(self, user_input: str) -> str:
        """Run an autonomous agent turn until completion or iteration limit."""
        self.context.add_user_message(user_input)
        tools_schema = self.tool_registry.get_all_schemas()

        iteration = 0
        final_answer = ""
        consecutive_empty_directives = 0

        while iteration < self.max_iterations:
            iteration += 1

            # Obtain completion from engine
            try:
                with console.status(f"[cyan]Agent thinking (Iteration {iteration}/{self.max_iterations})...[/cyan]", spinner="dots"):
                    response: AgentResponse = await self.engine.chat(
                        messages=self.context.get_messages(),
                        tools=tools_schema,
                        model=self.active_model,
                        temperature=0.2,
                    )
            except Exception as e:
                print_error(f"Inference error: {e}")
                return f"Inference failure: {e}"

            # If model produced text content, print and store it
            if response.content:
                print_agent_thought(response.content)
                final_answer = response.content

            # Record assistant turn in context
            self.context.add_assistant_message(
                Message(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls if response.tool_calls else None
                )
            )

            # Check for incomplete intent before breaking
            if not response.tool_calls:
                content_lower = (response.content or "").strip().lower()
                incomplete_triggers = [
                    "let's execute", "let us execute", "let's inspect", "let us inspect",
                    "let's run", "let's re-run", "re-run", "rerun", "re-test", "let's try",
                    "let's fix", "let's view", "let's do", "trying another", "i will try",
                    "i will check", "i will search", "execute these", "execute this",
                    "run these commands", "view the contents", "steps to follow",
                    "steps using the", "verify it passes", "test to verify", "verify that",
                    "proceed with", "let's proceed", "will replace", "we will replace",
                    "replace the line", "replace the content", "let's correct", "correct the bug"
                ]
                has_pending_plan = any(trig in content_lower for trig in incomplete_triggers)
                if iteration < self.max_iterations and has_pending_plan and consecutive_empty_directives < 2 and len(content_lower) < 1500:
                    consecutive_empty_directives += 1
                    # Model announced an action or outlined steps but did not execute a tool
                    self.context.add_user_message(
                        "[System Autonomous Directive]: You proposed an action or fix above, but did not execute any tool. "
                        "Do NOT write conversational filler. Immediately invoke the appropriate tool (e.g. replace_file_content, run_command, view_file) to apply the change."
                    )
                    continue
                break

            consecutive_empty_directives = 0

            # Execute tool calls
            for tc_idx, tc in enumerate(response.tool_calls):
                print_tool_call(tc.name, tc.arguments)
                
                # Execute tool
                result: ToolResult = await self.tool_registry.execute(tc.name, tc.arguments)
                
                # Print result
                print_tool_result(result.tool_name, result.content, result.is_error, result.execution_time)

                # Liquid refinement feedback if tool errored
                feedback_content = LiquidRefinementEngine.inspect_and_refine(
                    result, iteration, self.max_iterations
                )

                # Feed back to context
                self.context.add_tool_result(
                    tool_name=tc.name,
                    content=feedback_content,
                    tool_call_id=tc.id or f"call_{iteration}_{tc_idx}"
                )

                # If bash changed directory, update agent cwd
                bash_tool = self.tool_registry.get("run_command")
                if bash_tool and hasattr(bash_tool, "cwd") and bash_tool.cwd != self.cwd:
                    self.cwd = bash_tool.cwd
                    try:
                        os.chdir(self.cwd)
                    except Exception:
                        pass

        if iteration >= self.max_iterations:
            print_status(f"[warning]Reached iteration safety limit ({self.max_iterations}).[/warning]")

        return final_answer
