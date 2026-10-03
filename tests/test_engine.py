import pytest
from core.types import Message, ToolCall
from core.engine import LLMEngine
from core.context import ContextManager


def test_message_serialization():
    # Test tool call with arguments as dict
    tc = ToolCall(id="call_1", name="run_command", arguments={"command": "ls -la"})
    msg = Message(role="assistant", content="Running command", tool_calls=[tc])

    # Ollama native: arguments should be dict
    ollama_dict = msg.to_dict(is_ollama_native=True)
    assert isinstance(ollama_dict["tool_calls"][0]["function"]["arguments"], dict)
    assert ollama_dict["tool_calls"][0]["function"]["arguments"]["command"] == "ls -la"

    # OpenAI format: arguments should be json string
    openai_dict = msg.to_dict(is_ollama_native=False)
    assert isinstance(openai_dict["tool_calls"][0]["function"]["arguments"], str)
    assert '"command": "ls -la"' in openai_dict["tool_calls"][0]["function"]["arguments"]


def test_embedded_json_tool_call_extraction():
    engine = LLMEngine()
    
    # Markdown json block
    raw_markdown = """Here is the plan.
```json
{
  "name": "run_command",
  "arguments": {"command": "git status"}
}
```
Let me know if this works.
"""
    calls, cleaned = engine._extract_embedded_tool_calls(raw_markdown)
    assert len(calls) == 1
    assert calls[0].name == "run_command"
    assert calls[0].arguments == {"command": "git status"}
    assert "Here is the plan." in cleaned

    # Single-line JSON format
    single_line = '{"name": "view_file", "parameters": {"path": "main.py"}}'
    calls2, _ = engine._extract_embedded_tool_calls(single_line)
    assert len(calls2) == 1
    assert calls2[0].name == "view_file"
    assert calls2[0].arguments == {"path": "main.py"}


def test_context_manager_pruning():
    ctx = ContextManager(system_prompt="System Prompt", max_turns=4, max_tool_output_chars=100)
    
    # Add messages exceeding max_turns
    for i in range(10):
        ctx.add_user_message(f"User message {i}")
        ctx.add_assistant_message(Message(role="assistant", content=f"Assistant reply {i}"))

    msgs = ctx.get_messages()
    # System prompt must remain at index 0
    assert msgs[0].role == "system"
    assert msgs[0].content == "System Prompt"
    # Total messages kept should be bounded
    assert len(msgs) <= 5

    # Test tool output truncation
    large_output = "X" * 500
    ctx.add_tool_result(tool_name="bash", content=large_output)
    last_msg = ctx.get_messages()[-1]
    assert "Truncated" in last_msg.content
    assert len(last_msg.content) < 500


def test_engine_api_key_and_headers():
    engine = LLMEngine(
        base_url="https://openrouter.ai/api/v1",
        default_model="qwen/qwen-2.5-coder-32b-instruct",
        api_key="sk-test-key-12345"
    )
    assert engine.api_key == "sk-test-key-12345"
    assert not engine.is_ollama_native

