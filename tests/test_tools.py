import os
import tempfile
import pytest
import asyncio
from tools.bash import BashTool
from tools.files import ViewFileTool, WriteFileTool, ReplaceFileContentTool
from tools.search import SearchCodeTool, FindFilesTool, ListDirTool
from tools.web import WebSearchTool, ReadUrlTool


@pytest.mark.asyncio
async def test_bash_tool_basic():
    tool = BashTool()
    res = await tool.execute(command="echo 'Antigravity Godmode'")
    assert not res.is_error
    assert "Antigravity Godmode" in res.content
    assert "[Exit code: 0]" in res.content


@pytest.mark.asyncio
async def test_bash_tool_exit_code():
    tool = BashTool()
    res = await tool.execute(command="bash -c 'exit 42'")
    assert res.is_error
    assert "[Exit code: 42]" in res.content


@pytest.mark.asyncio
async def test_bash_tool_cwd_persistence():
    tool = BashTool()
    res1 = await tool.execute(command="cd /tmp")
    assert not res1.is_error
    assert tool.cwd == "/tmp"

    res2 = await tool.execute(command="pwd")
    assert not res2.is_error
    assert "/tmp" in res2.content


@pytest.mark.asyncio
async def test_bash_tool_timeout():
    tool = BashTool()
    res = await tool.execute(command="sleep 3", timeout=1)
    assert res.is_error
    assert "timed out" in res.content.lower()


@pytest.mark.asyncio
async def test_file_tools_lifecycle():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "subdir", "sample.py")

        # 1. Write file
        writer = WriteFileTool()
        w_res = await writer.execute(
            path=test_file,
            content="line 1: import os\nline 2: def greet():\nline 3:     return 'hello'\nline 4: print(greet())\n"
        )
        assert not w_res.is_error
        assert os.path.exists(test_file)

        # 2. View file with slicing
        viewer = ViewFileTool()
        v_res = await viewer.execute(path=test_file, start_line=2, end_line=3)
        assert not v_res.is_error
        assert "2: line 2: def greet():" in v_res.content
        assert "3: line 3:     return 'hello'" in v_res.content
        assert "line 4" not in v_res.content

        # 3. Surgical replace_file_content
        patcher = ReplaceFileContentTool()
        p_res = await patcher.execute(
            path=test_file,
            target_content="    return 'hello'",
            replacement_content="    return 'world'",
            start_line=2,
            end_line=4
        )
        assert not p_res.is_error
        assert "Successfully patched" in p_res.content
        assert "-line 3:     return 'hello'" in p_res.content
        assert "+line 3:     return 'world'" in p_res.content

        # 4. Verify modified content
        with open(test_file, "r") as f:
            content = f.read()
        assert "return 'world'" in content
        assert "return 'hello'" not in content


@pytest.mark.asyncio
async def test_replace_file_content_ambiguity():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "duplicate.txt")
        with open(test_file, "w") as f:
            f.write("target\nsome text\ntarget\n")

        patcher = ReplaceFileContentTool()
        # Searching the whole file where "target" appears twice should return error
        res = await patcher.execute(
            path=test_file,
            target_content="target",
            replacement_content="replaced"
        )
        assert res.is_error
        assert "appears 2 times" in res.content


@pytest.mark.asyncio
async def test_search_and_find_tools():
    searcher = SearchCodeTool()
    res = await searcher.execute(query="Antigravity", path="config")
    assert not res.is_error
    assert "default_config.yaml" in res.content

    finder = FindFilesTool()
    f_res = await finder.execute(pattern="*.yaml", path="config")
    assert not f_res.is_error
    assert "default_config.yaml" in f_res.content

    lister = ListDirTool()
    l_res = await lister.execute(path="config")
    assert not l_res.is_error
    assert "default_config.yaml" in l_res.content


@pytest.mark.asyncio
async def test_web_search_news():
    web = WebSearchTool()
    res = await web.execute(query="latest technology news 2026", max_results=2)
    assert not res.is_error
    assert len(res.content) > 50
    assert "[1]" in res.content
