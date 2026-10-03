import json
import re
import shlex
import ast
import httpx
from typing import List, Dict, Any, Optional, Tuple
from core.types import Message, ToolCall, AgentResponse


class LLMEngine:
    def __init__(self, base_url: str = "http://localhost:11434", default_model: str = "qwen2.5-coder:7b-instruct-q5_K_M"):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        # Detect whether endpoint is Ollama native or OpenAI-compatible
        self.is_ollama_native = "/v1" not in self.base_url and "11434" in self.base_url

    async def check_health(self) -> Tuple[bool, str]:
        """Check if LLM backend is accessible and return status info."""
        try:
            async with httpx.AsyncClient(timeout=4.0) as client:
                if self.is_ollama_native:
                    resp = await client.get(f"{self.base_url}/api/tags")
                    if resp.status_code == 200:
                        models = [m.get("name", "") for m in resp.json().get("models", [])]
                        return True, f"Ollama online ({len(models)} models available: {', '.join(models[:3])}...)"
                else:
                    resp = await client.get(f"{self.base_url}/models")
                    if resp.status_code == 200:
                        return True, "OpenAI-compatible server online"
        except Exception as e:
            return False, f"Backend connection failed: {e}"
        return False, "Backend unreachable"

    async def chat(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_tokens: int = 4096,
    ) -> AgentResponse:
        """Execute chat completion with tool calling support."""
        target_model = model or self.default_model
        
        # Prepare messages in dict form matching the backend API
        formatted_msgs = [m.to_dict(is_ollama_native=self.is_ollama_native) for m in messages]

        # Call appropriate backend
        if self.is_ollama_native:
            return await self._call_ollama_native(
                formatted_msgs, tools, target_model, temperature, top_p, max_tokens
            )
        else:
            return await self._call_openai_compat(
                formatted_msgs, tools, target_model, temperature, top_p, max_tokens
            )

    async def _call_ollama_native(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> AgentResponse:
        url = f"{self.base_url}/api/chat"
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
            }
        }
        if tools:
            payload["tools"] = tools

        async with httpx.AsyncClient(timeout=180.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        msg = data.get("message", {})
        raw_content = msg.get("content", "") or ""
        tool_calls_raw = msg.get("tool_calls") or []

        tool_calls: List[ToolCall] = []
        for i, tc in enumerate(tool_calls_raw):
            parsed = ToolCall.from_dict(tc)
            if not parsed.id:
                parsed.id = f"call_{i}"
            tool_calls.append(parsed)

        # Robust Fallback: Extract JSON tool calls embedded in content if none caught natively
        if not tool_calls and raw_content:
            extracted_calls, cleaned_content = self._extract_embedded_tool_calls(raw_content)
            if extracted_calls:
                tool_calls = extracted_calls
                raw_content = cleaned_content

        return AgentResponse(
            content=raw_content,
            tool_calls=tool_calls,
            model=model,
            finish_reason=data.get("done_reason")
        )

    async def _call_openai_compat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> AgentResponse:
        url = f"{self.base_url}/chat/completions"
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        async with httpx.AsyncClient(timeout=180.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        choice = data["choices"][0]
        msg = choice.get("message", {})
        raw_content = msg.get("content", "") or ""
        tool_calls_raw = msg.get("tool_calls") or []

        tool_calls: List[ToolCall] = []
        for i, tc in enumerate(tool_calls_raw):
            parsed = ToolCall.from_dict(tc)
            if not parsed.id:
                parsed.id = f"call_{i}"
            tool_calls.append(parsed)

        if not tool_calls and raw_content:
            extracted_calls, cleaned_content = self._extract_embedded_tool_calls(raw_content)
            if extracted_calls:
                tool_calls = extracted_calls
                raw_content = cleaned_content

        return AgentResponse(
            content=raw_content,
            tool_calls=tool_calls,
            model=model,
            finish_reason=choice.get("finish_reason")
        )

    def _extract_embedded_tool_calls(self, content: str) -> Tuple[List[ToolCall], str]:
        """Extract tool calls from tags, JSON code blocks, embedded objects, Pythonic calls, or CLI invocations."""
        calls: List[ToolCall] = []
        cleaned_text = content

        known_tools = {
            "run_command": ["command"],
            "view_file": ["path"],
            "write_file": ["path", "content"],
            "replace_file_content": ["path", "target_content", "replacement_content"],
            "search_code": ["query"],
            "find_files": ["pattern"],
            "list_dir": ["path"],
            "web_search": ["query"],
            "read_url": ["url"],
        }

        # 1. <tool_call> tags (Qwen / DeepSeek style)
        tag_pattern = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
        for m in tag_pattern.findall(content):
            try:
                obj = json.loads(m.strip())
                if isinstance(obj, dict) and "name" in obj:
                    calls.append(ToolCall(
                        name=obj["name"],
                        arguments=obj.get("arguments") or obj.get("parameters") or {}
                    ))
            except Exception:
                pass
        if calls:
            cleaned_text = tag_pattern.sub("", content).strip()
            return calls, cleaned_text

        # 2. ```json ... ``` blocks
        code_block_pattern = re.compile(r"```(?:json)?\s*(\{[^`]+\})\s*```", re.DOTALL)
        for m in code_block_pattern.findall(content):
            try:
                obj = json.loads(m.strip())
                if isinstance(obj, dict) and "name" in obj and ("arguments" in obj or "parameters" in obj):
                    calls.append(ToolCall(
                        name=obj["name"],
                        arguments=obj.get("arguments") or obj.get("parameters") or {}
                    ))
            except Exception:
                pass
        if calls:
            cleaned_text = code_block_pattern.sub("", content).strip()
            return calls, cleaned_text

        # 3. Balanced brace scanner to extract any embedded JSON object with "name"
        i = 0
        while i < len(content):
            if content[i] == '{':
                depth = 0
                for j in range(i, len(content)):
                    if content[j] == '{':
                        depth += 1
                    elif content[j] == '}':
                        depth -= 1
                        if depth == 0:
                            candidate = content[i:j+1]
                            try:
                                obj = json.loads(candidate)
                                if isinstance(obj, dict) and "name" in obj and ("arguments" in obj or "parameters" in obj):
                                    calls.append(ToolCall(
                                        name=obj["name"],
                                        arguments=obj.get("arguments") or obj.get("parameters") or {}
                                    ))
                                    cleaned_text = cleaned_text.replace(candidate, "")
                                    i = j
                            except Exception:
                                pass
                            break
            i += 1
        if calls:
            return calls, cleaned_text.strip()

        # 4. Python function call syntax e.g. tool_name(param="val", ...)
        for tool_name, params in known_tools.items():
            fn_pattern = re.compile(rf"\b({tool_name})\s*\((.*?)\)", re.DOTALL)
            for m in fn_pattern.finditer(content):
                call_str = m.group(0)
                try:
                    tree = ast.parse(call_str, mode="eval")
                    if isinstance(tree.body, ast.Call):
                        kwargs = {}
                        for kw in tree.body.keywords:
                            kwargs[kw.arg] = ast.literal_eval(kw.value)
                        for idx, arg in enumerate(tree.body.args):
                            if idx < len(params):
                                kwargs[params[idx]] = ast.literal_eval(arg)
                        calls.append(ToolCall(name=tool_name, arguments=kwargs))
                        cleaned_text = cleaned_text.replace(call_str, "")
                except Exception:
                    pass
        if calls:
            return calls, cleaned_text.strip()

        # 5. CLI style invocations on dedicated lines e.g. run_command "pytest ..."
        for line in content.splitlines():
            sline = line.strip().strip("`")
            for tool_name, params in known_tools.items():
                if sline.startswith(tool_name + " "):
                    try:
                        parts = shlex.split(sline)
                        if parts and parts[0] == tool_name:
                            args = {}
                            if tool_name == "run_command":
                                args["command"] = " ".join(parts[1:])
                            elif tool_name in ("view_file", "list_dir"):
                                args["path"] = parts[1]
                            elif tool_name in ("search_code", "web_search"):
                                args["query"] = " ".join(parts[1:])
                            else:
                                for idx, val in enumerate(parts[1:]):
                                    if idx < len(params):
                                        args[params[idx]] = val
                            calls.append(ToolCall(name=tool_name, arguments=args))
                            cleaned_text = cleaned_text.replace(line, "")
                    except Exception:
                        pass

        return calls, cleaned_text.strip()
