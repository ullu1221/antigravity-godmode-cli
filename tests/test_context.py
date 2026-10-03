import pytest
from core.context import ContextManager
from core.types import Message


def test_context_manager_pruning():
    ctx = ContextManager(system_prompt="system prompt", max_turns=4)
    ctx.add_user_message("u1")
    ctx.add_assistant_message(Message(role="assistant", content="a1"))
    ctx.add_user_message("u2")
    ctx.add_assistant_message(Message(role="assistant", content="a2"))
    ctx.add_user_message("u3")

    msgs = ctx.get_messages()
    # System message must always be preserved at index 0
    assert msgs[0].role == "system"
    assert msgs[0].content == "system prompt"
    # Total messages should be bounded by max_turns + 1
    assert len(msgs) <= 5


def test_context_manager_orphaned_tool_pruning():
    ctx = ContextManager(system_prompt="sys", max_turns=3)
    # Adding turns such that after sliding window, a tool message would be at recent_msgs[0]
    ctx.add_user_message("u1")
    ctx.add_assistant_message(Message(role="assistant", content="call", tool_calls=[]))
    ctx.add_tool_result(tool_name="view_file", content="file data")
    ctx.add_user_message("u2")
    ctx.add_assistant_message(Message(role="assistant", content="done"))

    msgs = ctx.get_messages()
    assert msgs[0].role == "system"
    # The message immediately following system prompt must NOT be an orphaned tool message
    if len(msgs) > 1:
        assert msgs[1].role != "tool"


def test_context_truncation():
    ctx = ContextManager(system_prompt="sys", max_tool_output_chars=100)
    huge_text = "x" * 500
    ctx.add_tool_result("bash", huge_text)
    msgs = ctx.get_messages()
    assert len(msgs[1].content) < 200
    assert "Truncated" in msgs[1].content
