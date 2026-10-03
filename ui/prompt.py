import os
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML

style = Style.from_dict({
    "prompt": "#00d7ff bold",
    "cwd": "#808080",
    "symbol": "#00ff87 bold",
})


class AgentPrompt:
    def __init__(self, history_file: str = "~/.godmode_history"):
        expanded_history = os.path.expanduser(history_file)
        os.makedirs(os.path.dirname(expanded_history) or ".", exist_ok=True)
        self.session: PromptSession = PromptSession(
            history=FileHistory(expanded_history),
            auto_suggest=AutoSuggestFromHistory(),
        )

    async def get_input(self, cwd: str) -> str:
        # Shorten cwd if in user's home
        home = os.path.expanduser("~")
        display_cwd = cwd.replace(home, "~") if cwd.startswith(home) else cwd
        if len(display_cwd) > 28:
            display_cwd = "..." + display_cwd[-25:]

        prompt_text = HTML(f"<cwd>[{display_cwd}]</cwd> <prompt>godmode</prompt> <symbol>❯</symbol> ")
        try:
            res = await self.session.prompt_async(prompt_text, style=style)
            return res.strip()
        except (KeyboardInterrupt, EOFError):
            return "/exit"
