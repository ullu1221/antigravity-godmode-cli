from typing import List, Optional
from core.types import Message


class ContextManager:
    def __init__(self, system_prompt: str, max_turns: int = 40, max_tool_output_chars: int = 4000):
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.max_tool_output_chars = max_tool_output_chars
        self.messages: List[Message] = [Message(role="system", content=system_prompt)]

    def add_user_message(self, content: str) -> None:
        self.messages.append(Message(role="user", content=content))
        self._prune_if_needed()

    def add_assistant_message(self, message: Message) -> None:
        self.messages.append(message)
        self._prune_if_needed()

    def add_tool_result(self, tool_name: str, content: str, tool_call_id: Optional[str] = None) -> None:
        # Sanitize and bound content length
        if len(content) > self.max_tool_output_chars:
            head = content[: self.max_tool_output_chars // 2]
            tail = content[-self.max_tool_output_chars // 2 :]
            content = f"{head}\n\n[... Truncated {len(content) - self.max_tool_output_chars} characters ...]\n\n{tail}"

        self.messages.append(Message(
            role="tool",
            name=tool_name,
            content=content,
            tool_call_id=tool_call_id
        ))
        self._prune_if_needed()

    def get_messages(self) -> List[Message]:
        return list(self.messages)

    def update_system_prompt(self, new_prompt: str) -> None:
        self.system_prompt = new_prompt
        if self.messages and self.messages[0].role == "system":
            self.messages[0].content = new_prompt
        else:
            self.messages.insert(0, Message(role="system", content=new_prompt))

    def clear_history(self) -> None:
        self.messages = [Message(role="system", content=self.system_prompt)]

    def _prune_if_needed(self) -> None:
        """Sliding window pruning: keep system prompt + recent turns."""
        if len(self.messages) <= self.max_turns + 1:
            return

        # Always preserve system prompt (index 0)
        sys_msg = self.messages[0]
        # Keep the most recent messages
        recent_msgs = self.messages[-self.max_turns:]
        
        # Ensure we don't start with an orphaned tool response without assistant
        while recent_msgs and recent_msgs[0].role == "tool":
            recent_msgs.pop(0)

        self.messages = [sys_msg] + recent_msgs
