import os
import sys
import tempfile
import asyncio
import pytest
from core.types import Message, ToolCall, ToolResult, AgentResponse
from core.context import ContextManager
from core.engine import LLMEngine
from core.speculative import LiquidRefinementEngine, SpeculativeRacingEngine
from core.agent import AutonomousAgent
from tools.base import ToolRegistry
from tools.files import ViewFileTool, WriteFileTool, ReplaceFileContentTool
from tools.bash import BashTool


# ==============================================================================
# 1. CONTEXT MANAGER STRESS & INVARIANT TESTS
# ==============================================================================

def test_context_manager_system_prompt_preservation():
    system_prompt = "GODMODE V3 SYSTEM PROMPT"
    ctx = ContextManager(system_prompt=system_prompt, max_turns=5)
    
    # Add 20 user/assistant turns
    for i in range(20):
        ctx.add_user_message(f"User turn {i}")
        ctx.add_assistant_message(Message(role="assistant", content=f"Assistant turn {i}"))
        
    msgs = ctx.get_messages()
    # Invariant 1: System prompt is always preserved at index 0
    assert msgs[0].role == "system"
    assert msgs[0].content == system_prompt
    # Invariant 2: Total messages bounded by max_turns + 1
    assert len(msgs) <= 5 + 1


def test_context_manager_orphaned_tool_cleanup():
    ctx = ContextManager(system_prompt="SYS", max_turns=3)
    
    # Simulate an assistant call with multiple tool results
    ctx.add_user_message("Do task")
    ctx.add_assistant_message(Message(
        role="assistant",
        content="",
        tool_calls=[ToolCall(name="view_file", arguments={"path": "a.txt"})]
    ))
    ctx.add_tool_result("view_file", "file contents a")
    ctx.add_tool_result("view_file", "file contents b")
    ctx.add_user_message("Next step")
    ctx.add_assistant_message(Message(role="assistant", content="Done"))

    msgs = ctx.get_messages()
    assert msgs[0].role == "system"
    # Ensure no leading message in history is a 'tool' message
    if len(msgs) > 1:
        assert msgs[1].role != "tool"


def test_context_manager_large_output_truncation():
    ctx = ContextManager(system_prompt="SYS", max_tool_output_chars=200)
    large_payload = "A" * 1000 + "ERROR_CRITICAL_MARKER" + "Z" * 1000
    ctx.add_tool_result("run_command", large_payload)

    msgs = ctx.get_messages()
    tool_msg = msgs[-1]
    assert tool_msg.role == "tool"
    assert "Truncated" in tool_msg.content
    assert tool_msg.content.startswith("A" * 50)
    assert tool_msg.content.endswith("Z" * 50)
    assert len(tool_msg.content) < 500


def test_message_to_dict_ollama_vs_openai():
    tc = ToolCall(id="call_1", name="run_command", arguments={"command": "ls -la"})
    msg = Message(role="assistant", content="Running", tool_calls=[tc])

    # Ollama native: arguments is a dict
    ollama_dict = msg.to_dict(is_ollama_native=True)
    assert isinstance(ollama_dict["tool_calls"][0]["function"]["arguments"], dict)
    assert ollama_dict["tool_calls"][0]["function"]["arguments"]["command"] == "ls -la"

    # OpenAI compatible: arguments is a serialized JSON string
    openai_dict = msg.to_dict(is_ollama_native=False)
    assert isinstance(openai_dict["tool_calls"][0]["function"]["arguments"], str)
    assert '"command": "ls -la"' in openai_dict["tool_calls"][0]["function"]["arguments"]


# ==============================================================================
# 2. TOOL CALL EXTRACTION ROBUSTNESS TESTS
# ==============================================================================

def test_extract_tag_format():
    engine = LLMEngine()
    raw = (
        "I will now inspect the test file.\n"
        "<tool_call>\n"
        '{"name": "view_file", "arguments": {"path": "tests/test_demo.py", "start_line": 10}}\n'
        "</tool_call>\n"
        "Let me know if you need more."
    )
    calls, cleaned = engine._extract_embedded_tool_calls(raw)
    assert len(calls) == 1
    assert calls[0].name == "view_file"
    assert calls[0].arguments["path"] == "tests/test_demo.py"
    assert calls[0].arguments["start_line"] == 10
    assert "<tool_call>" not in cleaned


def test_extract_markdown_json_format():
    engine = LLMEngine()
    raw = (
        "Here is the command to run:\n"
        "```json\n"
        '{\n  "name": "run_command",\n  "arguments": {\n    "command": "pytest -k test_feature"\n  }\n}\n'
        "```\n"
    )
    calls, cleaned = engine._extract_embedded_tool_calls(raw)
    assert len(calls) == 1
    assert calls[0].name == "run_command"
    assert calls[0].arguments["command"] == "pytest -k test_feature"


def test_extract_pythonic_function_syntax():
    engine = LLMEngine()
    raw = (
        "Executing:\n"
        'replace_file_content(path="main.py", target_content="old_var = 1", replacement_content="new_var = 2")\n'
    )
    calls, cleaned = engine._extract_embedded_tool_calls(raw)
    assert len(calls) == 1
    assert calls[0].name == "replace_file_content"
    assert calls[0].arguments["path"] == "main.py"
    assert calls[0].arguments["target_content"] == "old_var = 1"
    assert calls[0].arguments["replacement_content"] == "new_var = 2"


def test_extract_dirty_json_with_literal_newlines():
    engine = LLMEngine()
    # Model emits JSON with literal unescaped newlines inside the command
    raw = (
        '<tool_call>\n'
        '{"name": "run_command", "arguments": {"command": "echo line1\necho line2"}}\n'
        '</tool_call>'
    )
    calls, cleaned = engine._extract_embedded_tool_calls(raw)
    assert len(calls) == 1
    assert calls[0].name == "run_command"
    assert "echo line1" in calls[0].arguments["command"]


def test_extract_cli_style():
    engine = LLMEngine()
    raw = "Let's check git status:\nrun_command git status --short"
    calls, cleaned = engine._extract_embedded_tool_calls(raw)
    assert len(calls) == 1
    assert calls[0].name == "run_command"
    assert calls[0].arguments["command"] == "git status --short"


# ==============================================================================
# 3. PRECISION FILE TOOLS DEEP TESTS
# ==============================================================================

@pytest.mark.asyncio
async def test_view_file_edge_cases():
    viewer = ViewFileTool()
    with tempfile.NamedTemporaryFile("w+", delete=False, encoding="utf-8") as tmp:
        tmp.write("line 1: Alpha\nline 2: Beta\nline 3: Gamma\nline 4: Delta\n")
        tmp_path = tmp.name

    try:
        # Slicing lines 2 to 3
        res = await viewer.execute(path=tmp_path, start_line=2, end_line=3)
        assert not res.is_error
        assert "2: line 2: Beta" in res.content
        assert "3: line 3: Gamma" in res.content
        assert "Alpha" not in res.content
        assert "Delta" not in res.content

        # Out of bounds start line
        err_res = await viewer.execute(path=tmp_path, start_line=999)
        assert err_res.is_error
        assert "out of range" in err_res.content

        # Nonexistent file
        missing_res = await viewer.execute(path="/nonexistent/path/xyz.txt")
        assert missing_res.is_error
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@pytest.mark.asyncio
async def test_replace_file_content_bounded_duplicate_targeting():
    """Verify that when a chunk appears multiple times, start_line/end_line isolates the desired one."""
    patcher = ReplaceFileContentTool()
    with tempfile.NamedTemporaryFile("w+", delete=False, encoding="utf-8") as tmp:
        content = (
            "def worker_one():\n"
            "    timeout = 10\n"
            "    return timeout\n"
            "\n"
            "def worker_two():\n"
            "    timeout = 10\n"
            "    return timeout\n"
        )
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Without line bounding, timeout = 10 is ambiguous (appears twice)
        ambig_res = await patcher.execute(
            path=tmp_path,
            target_content="    timeout = 10",
            replacement_content="    timeout = 60"
        )
        assert ambig_res.is_error
        assert "appears 2 times" in ambig_res.content

        # With line bounding around worker_two (lines 5 to 7), targeting succeeds
        bounded_res = await patcher.execute(
            path=tmp_path,
            target_content="    timeout = 10",
            replacement_content="    timeout = 60",
            start_line=5,
            end_line=7
        )
        assert not bounded_res.is_error

        # Verify worker_one is still 10, worker_two is now 60
        with open(tmp_path, "r") as f:
            updated = f.read()
        assert "def worker_one():\n    timeout = 10" in updated
        assert "def worker_two():\n    timeout = 60" in updated

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@pytest.mark.asyncio
async def test_replace_file_content_indentation_adaptation():
    """Verify that AST whitespace normalization adapts if model drifts 2-spaces vs 4-spaces."""
    patcher = ReplaceFileContentTool()
    with tempfile.NamedTemporaryFile("w+", delete=False, encoding="utf-8") as tmp:
        # File has 8 spaces
        content = (
            "class Service:\n"
            "    def execute(self):\n"
            "        x = 100\n"
            "        return x\n"
        )
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Model provides target with 4 spaces instead of 8
        res = await patcher.execute(
            path=tmp_path,
            target_content="    x = 100\n    return x",
            replacement_content="    x = 200\n    return x * 2"
        )
        assert not res.is_error

        with open(tmp_path, "r") as f:
            updated = f.read()
        assert "        x = 200\n        return x * 2" in updated

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ==============================================================================
# 4. BASH TOOL & PERSISTENT WORKING DIRECTORY TESTS
# ==============================================================================

@pytest.mark.asyncio
async def test_bash_persistent_cd_and_exit_codes():
    bash = BashTool()
    initial_dir = bash.cwd

    # 1. Successful command
    res1 = await bash.execute("echo 'GODMODE_RUNNING'")
    assert not res1.is_error
    assert "GODMODE_RUNNING" in res1.content
    assert "[Exit code: 0]" in res1.content

    # 2. Directory change via cd
    res2 = await bash.execute("cd /tmp")
    assert not res2.is_error
    assert bash.cwd == "/tmp"

    # 3. Follow-up command inherits /tmp
    res3 = await bash.execute("pwd")
    assert not res3.is_error
    assert "/tmp" in res3.content

    # 4. Failing command captures non-zero code
    res4 = await bash.execute("ls /nonexistent_directory_for_test_12345")
    assert res4.is_error
    assert "[Exit code:" in res4.content
    assert "[Exit code: 0]" not in res4.content

    # Restore directory
    await bash.execute(f"cd {initial_dir}")


@pytest.mark.asyncio
async def test_bash_timeout_handling():
    bash = BashTool()
    res = await bash.execute("sleep 5", timeout=1)
    assert res.is_error
    assert "timed out after 1 seconds" in res.content


# ==============================================================================
# 5. LIQUID REFINEMENT & AGENT SELF-HEALING DIAGNOSTICS
# ==============================================================================

def test_liquid_refinement_replace_error_guidance():
    fail_res = ToolResult(
        tool_name="replace_file_content",
        content="Error: target_content not found in 'core/types.py'.",
        is_error=True
    )
    guidance = LiquidRefinementEngine.inspect_and_refine(fail_res, current_iteration=2, max_iterations=10)
    assert "TOOL FAILURE DIAGNOSTIC" in guidance
    assert "CALL 'view_file' ON THE TARGET FILE NOW" in guidance
    assert "8 iterations remaining" in guidance


# ==============================================================================
# 6. END-TO-END AUTONOMOUS AGENT SIMULATION
# ==============================================================================

class MockScriptedLLMEngine(LLMEngine):
    """Simulates an LLM providing a sequence of responses to solve a task."""
    def __init__(self, responses: list):
        super().__init__()
        self.responses = responses
        self.turn = 0

    async def chat(self, messages, tools=None, model=None, **kwargs):
        if self.turn < len(self.responses):
            resp = self.responses[self.turn]
            self.turn += 1
            return resp
        return AgentResponse(content="Final solution complete.")


@pytest.mark.asyncio
async def test_autonomous_agent_full_turn_loop():
    # Setup temporary environment
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "math_mod.py")
        with open(test_file, "w") as f:
            f.write("def add(a, b):\n    return a - b  # Bug: minus instead of plus\n")

        registry = ToolRegistry()
        registry.register(ViewFileTool())
        registry.register(ReplaceFileContentTool())
        registry.register(BashTool(default_cwd=tmpdir))

        # Scripted model steps:
        # Step 1: Model inspects file
        step1 = AgentResponse(
            content="I will view math_mod.py",
            tool_calls=[ToolCall(name="view_file", arguments={"path": test_file})]
        )
        # Step 2: Model fixes the bug
        step2 = AgentResponse(
            content="Fixing subtraction to addition",
            tool_calls=[ToolCall(
                name="replace_file_content",
                arguments={
                    "path": test_file,
                    "target_content": "    return a - b  # Bug: minus instead of plus",
                    "replacement_content": "    return a + b"
                }
            )]
        )
        # Step 3: Model concludes
        step3 = AgentResponse(
            content="The bug in math_mod.py has been resolved and verified."
        )

        mock_engine = MockScriptedLLMEngine([step1, step2, step3])
        agent = AutonomousAgent(
            config={"active_model": "mock", "agent": {"max_iterations": 10}},
            tool_registry=registry,
            engine=mock_engine,
            initial_cwd=tmpdir
        )

        final_answer = await agent.run_turn("Fix the math function")
        assert "resolved and verified" in final_answer

        # Verify file on disk was genuinely patched
        with open(test_file, "r") as f:
            content = f.read()
        assert "return a + b" in content
