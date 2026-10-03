import subprocess
import os
import tempfile
import base64
import time
from tools.base import BaseTool
from core.types import ToolResult


class ScreenshotTool(BaseTool):
    name = "screenshot"
    description = "Capture a screenshot of the current active desktop display to inspect GUI applications or browser state."
    parameters = {"type": "object", "properties": {}}

    async def execute(self, **kwargs) -> ToolResult:
        start_time = time.time()
        tmp = tempfile.mktemp(suffix=".png")
        shot_taken = False

        for cmd in [["spectacle", "-b", "-n", "-o", tmp], ["grim", tmp]]:
            try:
                res = subprocess.run(cmd, capture_output=True, timeout=5)
                if res.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 0:
                    shot_taken = True
                    break
            except Exception:
                continue

        if not shot_taken:
            try:
                import pyautogui
                pyautogui.screenshot(tmp)
                shot_taken = os.path.exists(tmp) and os.path.getsize(tmp) > 0
            except Exception:
                pass

        if not shot_taken:
            return ToolResult(
                tool_name=self.name,
                content="Error: Could not capture screenshot via spectacle, grim, or pyautogui.",
                is_error=True,
                execution_time=time.time() - start_time
            )

        size_kb = os.path.getsize(tmp) / 1024
        return ToolResult(
            tool_name=self.name,
            content=f"Screenshot captured successfully ({size_kb:.1f} KB saved to {tmp}). Screen state observed.",
            execution_time=time.time() - start_time
        )


class MouseMoveTool(BaseTool):
    name = "mouse_move"
    description = "Move mouse cursor to specific coordinates (x, y) via ydotool."
    parameters = {
        "type": "object",
        "properties": {
            "x": {"type": "integer", "description": "X coordinate"},
            "y": {"type": "integer", "description": "Y coordinate"}
        },
        "required": ["x", "y"]
    }

    async def execute(self, x: int, y: int, **kwargs) -> ToolResult:
        start_time = time.time()
        res = subprocess.run(
            ["ydotool", "mousemove", "--absolute", "-x", str(x), "-y", str(y)],
            capture_output=True, text=True
        )
        return ToolResult(
            tool_name=self.name,
            content=f"Mouse moved to ({x}, {y})",
            is_error=(res.returncode != 0),
            execution_time=time.time() - start_time
        )


class MouseClickTool(BaseTool):
    name = "mouse_click"
    description = "Click mouse button (left, right, or middle) via ydotool."
    parameters = {
        "type": "object",
        "properties": {
            "button": {"type": "string", "enum": ["left", "right", "middle"], "default": "left"}
        }
    }

    async def execute(self, button: str = "left", **kwargs) -> ToolResult:
        start_time = time.time()
        btn_map = {"left": "0xC0", "right": "0xC1", "middle": "0xC2"}
        btn = btn_map.get(button, "0xC0")
        res = subprocess.run(["ydotool", "click", btn], capture_output=True)
        return ToolResult(
            tool_name=self.name,
            content=f"Clicked mouse {button} button",
            is_error=(res.returncode != 0),
            execution_time=time.time() - start_time
        )


class KeyPressTool(BaseTool):
    name = "key_press"
    description = "Press key combinations (e.g. 'enter', 'ctrl+c', 'alt+tab', 'super') or type strings."
    parameters = {
        "type": "object",
        "properties": {
            "keys": {"type": "string", "description": "Key or combination to press."},
            "type_text": {"type": "string", "description": "Optional text to type directly."}
        }
    }

    async def execute(self, keys: Optional[str] = None, type_text: Optional[str] = None, **kwargs) -> ToolResult:
        start_time = time.time()
        if type_text:
            res = subprocess.run(["ydotool", "type", "--", type_text], capture_output=True)
            return ToolResult(
                tool_name=self.name,
                content=f"Typed text: {type_text[:50]}...",
                is_error=(res.returncode != 0),
                execution_time=time.time() - start_time
            )
        elif keys:
            res = subprocess.run(["ydotool", "key", keys], capture_output=True)
            return ToolResult(
                tool_name=self.name,
                content=f"Pressed key: {keys}",
                is_error=(res.returncode != 0),
                execution_time=time.time() - start_time
            )
        return ToolResult(
            tool_name=self.name,
            content="Error: Neither keys nor type_text provided.",
            is_error=True,
            execution_time=time.time() - start_time
        )
