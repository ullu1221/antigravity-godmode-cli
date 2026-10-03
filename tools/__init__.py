from typing import Dict, Any
from tools.base import ToolRegistry
from tools.bash import BashTool
from tools.files import ViewFileTool, WriteFileTool, ReplaceFileContentTool
from tools.search import SearchCodeTool, FindFilesTool, ListDirTool
from tools.web import WebSearchTool, ReadUrlTool
from tools.desktop import ScreenshotTool, MouseMoveTool, MouseClickTool, KeyPressTool


def build_tool_registry(config: Dict[str, Any], initial_cwd: str = ".") -> ToolRegistry:
    registry = ToolRegistry()
    tools_cfg = config.get("tools", {})

    # File tools
    if tools_cfg.get("enable_files", True):
        registry.register(ViewFileTool())
        registry.register(WriteFileTool())
        registry.register(ReplaceFileContentTool())

    # Shell tools
    if tools_cfg.get("enable_bash", True):
        registry.register(BashTool(default_cwd=initial_cwd))

    # Search tools
    if tools_cfg.get("enable_search", True):
        registry.register(SearchCodeTool())
        registry.register(FindFilesTool())
        registry.register(ListDirTool())

    # Web tools
    if tools_cfg.get("enable_web", True):
        registry.register(WebSearchTool())
        registry.register(ReadUrlTool())

    # Desktop tools (optional)
    if tools_cfg.get("enable_desktop", False):
        registry.register(ScreenshotTool())
        registry.register(MouseMoveTool())
        registry.register(MouseClickTool())
        registry.register(KeyPressTool())

    return registry
