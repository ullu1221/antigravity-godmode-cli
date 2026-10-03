from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from core.types import ToolResult


class BaseTool(ABC):
    name: str
    description: str
    parameters: Dict[str, Any]

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        pass

    def to_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def get_all_schemas(self) -> List[Dict[str, Any]]:
        return [tool.to_schema() for tool in self._tools.values()]

    async def execute(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        tool = self.get(name)
        if not tool:
            return ToolResult(
                tool_name=name,
                content=f"Error: Unknown tool '{name}'. Available tools: {list(self._tools.keys())}",
                is_error=True
            )
        try:
            return await tool.execute(**arguments)
        except Exception as e:
            return ToolResult(
                tool_name=name,
                content=f"Tool execution exception in '{name}': {str(e)}",
                is_error=True
            )
