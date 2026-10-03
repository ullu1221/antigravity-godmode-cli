import asyncio
from typing import List, Dict, Any, Optional, Tuple, Callable
from core.types import Message, AgentResponse, ToolResult
from core.engine import LLMEngine


class SpeculativeRacingEngine:
    """
    Godmode Parallel Racing & Speculative Orchestration Engine.
    Executes parallel candidate generation across prompt variations or models,
    racing them asynchronously and picking the fastest, highest-fidelity code/tool candidate.
    """

    def __init__(self, engine: LLMEngine):
        self.engine = engine

    async def race_prompts(
        self,
        messages: List[Message],
        tools: List[Dict[str, Any]],
        model: str,
        prompt_variations: List[str],
        temperature: float = 0.3
    ) -> AgentResponse:
        """
        Races multiple prompt variants asynchronously.
        Returns the first completion that contains valid executable tool calls or well-formed code.
        """
        async def _eval_variant(variant_prompt: str) -> AgentResponse:
            # Clone messages and augment user prompt
            cloned = [m.model_copy() for m in messages]
            if cloned and cloned[-1].role == "user":
                cloned[-1] = Message(
                    role="user",
                    content=f"{cloned[-1].content}\n\n[Instruction Focus: {variant_prompt}]"
                )
            return await self.engine.chat(
                messages=cloned,
                tools=tools,
                model=model,
                temperature=temperature
            )

        tasks = [asyncio.create_task(_eval_variant(v)) for v in prompt_variations]
        
        # As tasks complete, inspect for viable tool calls or code
        for completed_task in asyncio.as_completed(tasks):
            try:
                res: AgentResponse = await completed_task
                if res.tool_calls or (res.content and len(res.content.strip()) > 20):
                    # Cancel remaining tasks to conserve GPU compute
                    for t in tasks:
                        if not t.done():
                            t.cancel()
                    return res
            except Exception:
                continue

        # Fallback to standard chat if racing didn't yield early winner
        return await self.engine.chat(messages=messages, tools=tools, model=model)


class LiquidRefinementEngine:
    """
    Liquid Refinement & Autonomous Self-Healing.
    When a tool returns an error or failure, generates targeted self-correcting feedback
    to enable zero-friction autonomous problem resolution.
    """

    @staticmethod
    def inspect_and_refine(tool_result: ToolResult, current_iteration: int, max_iterations: int) -> str:
        """Format targeted self-healing guidance for the agent upon tool failure."""
        if not tool_result.is_error:
            return tool_result.content

        guidance = [
            f"[TOOL FAILURE DIAGNOSTIC: '{tool_result.tool_name}']",
            f"Output / Error Details:",
            tool_result.content,
            "",
            "AUTONOMOUS SELF-HEALING DIRECTIVE:",
            "1. Analyze the exact error message, missing module, syntax mistake, or missing file above.",
            "2. Do NOT apologize and do NOT ask the user what to do.",
        ]
        if tool_result.tool_name == "replace_file_content" and "not found" in tool_result.content:
            guidance.extend([
                "3. CRITICAL: You attempted to edit a file without matching its exact contents. CALL 'view_file' ON THE TARGET FILE NOW to see its genuine lines before calling replace_file_content again.",
                "4. Do NOT copy assertions from test files into implementation files."
            ])
        else:
            guidance.append("3. Formulate an immediate corrective action (e.g. inspect with search_code / view_file, fix syntax with replace_file_content, or adjust the bash command).")
            
        guidance.append(f"4. You have {max_iterations - current_iteration} iterations remaining to succeed.")
        return "\n".join(guidance)
