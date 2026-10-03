import pytest
import os
import tempfile
from tools.files import ReplaceFileContentTool, _find_normalized_match, _adjust_indentation


@pytest.mark.asyncio
async def test_normalized_matching():
    sample = """class MyClass:
    def execute(self) -> bool:
        if self.val < 10:
            return False
        return True
"""
    # Target without leading indentation
    target = """def execute(self) -> bool:
    if self.val < 10:"""
    
    match, count = _find_normalized_match(sample, target)
    assert count == 1
    assert "def execute(self) -> bool:" in match


def test_adjust_indentation():
    orig_chunk = "    def allow_request(self):\n        return False\n"
    repl_chunk = "def allow_request(self):\n    return True\n"
    adjusted = _adjust_indentation(orig_chunk, repl_chunk)
    assert adjusted.startswith("    def allow_request(self):")
    assert "        return True" in adjusted


@pytest.mark.asyncio
async def test_replace_with_old_str_alias():
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".py") as tf:
        tf.write("line1\nline2_old\nline3\n")
        temp_path = tf.name

    try:
        tool = ReplaceFileContentTool()
        res = await tool.execute(
            path=temp_path,
            old_str="line2_old",
            new_str="line2_new"
        )
        assert not res.is_error
        with open(temp_path, "r") as f:
            content = f.read()
        assert "line2_new" in content
        assert "line2_old" not in content
    finally:
        os.remove(temp_path)
