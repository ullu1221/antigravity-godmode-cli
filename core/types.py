from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
import json


class ToolCall(BaseModel):
    id: str = ""
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "ToolCall":
        func = data.get("function", {})
        args = func.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args, strict=False)
            except Exception:
                args = {"raw": args}
        return cls(
            id=data.get("id", ""),
            name=func.get("name", data.get("name", "")),
            arguments=args if isinstance(args, dict) else {}
        )


class Message(BaseModel):
    role: str  # system, user, assistant, tool
    content: Optional[str] = ""
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None

    def to_dict(self, is_ollama_native: bool = False) -> Dict[str, Any]:
        d: Dict[str, Any] = {"role": self.role}
        if self.content is not None:
            d["content"] = self.content
        if self.name:
            d["name"] = self.name
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            calls = []
            for i, tc in enumerate(self.tool_calls):
                args = tc.arguments
                if is_ollama_native:
                    # Ollama native API expects arguments to be a dictionary
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {}
                else:
                    # OpenAI API expects arguments to be a JSON string
                    if isinstance(args, dict):
                        args = json.dumps(args)

                calls.append({
                    "id": tc.id or f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": args
                    }
                })
            d["tool_calls"] = calls
        return d


class ToolResult(BaseModel):
    tool_name: str
    content: str
    is_error: bool = False
    execution_time: float = 0.0
    tool_call_id: Optional[str] = None

    @property
    def success(self) -> bool:
        return not self.is_error


class AgentResponse(BaseModel):
    content: Optional[str] = ""
    tool_calls: List[ToolCall] = Field(default_factory=list)
    model: str = ""
    finish_reason: Optional[str] = None
