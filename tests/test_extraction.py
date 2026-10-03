import pytest
from core.engine import LLMEngine


def test_tag_extraction():
    engine = LLMEngine()
    content = '<tool_call>\n{"name": "view_file", "arguments": {"path": "main.py"}}\n</tool_call>'
    calls, cleaned = engine._extract_embedded_tool_calls(content)
    assert len(calls) == 1
    assert calls[0].name == "view_file"
    assert calls[0].arguments == {"path": "main.py"}


def test_json_block_extraction():
    engine = LLMEngine()
    content = 'Here is the call:\n```json\n{"name": "run_command", "arguments": {"command": "pytest"}}\n```'
    calls, cleaned = engine._extract_embedded_tool_calls(content)
    assert len(calls) == 1
    assert calls[0].name == "run_command"
    assert calls[0].arguments == {"command": "pytest"}


def test_python_call_extraction():
    engine = LLMEngine()
    content = 'Calling now: run_command(command="pytest test_suite.py")'
    calls, cleaned = engine._extract_embedded_tool_calls(content)
    assert len(calls) == 1
    assert calls[0].name == "run_command"
    assert calls[0].arguments.get("command") == "pytest test_suite.py"


def test_cli_style_extraction():
    engine = LLMEngine()
    content = 'Let us execute:\nrun_command "pytest -v"'
    calls, cleaned = engine._extract_embedded_tool_calls(content)
    assert len(calls) == 1
    assert calls[0].name == "run_command"
    assert "pytest -v" in calls[0].arguments.get("command")
