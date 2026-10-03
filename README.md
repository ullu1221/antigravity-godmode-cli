# ⚡ Antigravity Godmode CLI (`godmode`)

An autonomous, multi-agent AI coding CLI engineered for **full autonomy**, **zero refusals**, and **surgical code manipulation** — combining the best developer experience paradigms of **Claude Code CLI**, **Google Antigravity CLI**, **OpenClaw**, and **LangGraph**, accelerated with **speculative racing** and **liquid self-refinement**.

Directly accelerated on your **Dell G15 5530** (NVIDIA RTX 3050 6GB + Intel i5-13450HX 16-thread P-cores).

---

## 🚀 Quickstart

Run directly from any terminal or directory on your system:

```bash
# 1. Start the interactive terminal UI
godmode

# 2. Or pass a one-shot autonomous task directly
godmode "Inspect Projects/04-weather-app, run tests, and fix any failing edge cases"
```

---

## 🛠 Feature & Tool Matrix

| Capability | Tool Name | Description |
| :--- | :--- | :--- |
| **Persistent Bash** | `run_command` | Stateful shell session that tracks working directory (`cd`), streams stdout/stderr, and returns exit codes. |
| **Precision Patcher** | `replace_file_content` | Surgically edits exact target code chunks within line boundaries with automated unified diff previews. Eliminates accidental file truncation. |
| **File Inspector** | `view_file` | Reads files with 1-indexed line numbers and slice boundaries (`start_line`, `end_line`). |
| **File Creator** | `write_file` | Creates new files or overwrites existing targets with automatic parent directory creation. |
| **Ripgrep Search** | `search_code` | Native `rg` regex and string search across the codebase with file patterns and line numbers. |
| **Fast Finder** | `find_files` | Native `fd` discovery for locating files and paths matching glob patterns. |
| **Directory Explorer** | `list_dir` | Clean listing of folder structures, directories, and file sizes. |
| **Live Web Search** | `web_search` | Real-time DuckDuckGo search queries for up-to-date documentation and solutions. |
| **URL Reader** | `read_url` | Webpage content extractor that converts HTML to clean markdown/text. |
| **Desktop Control** | `screenshot`, `mouse_move`, `mouse_click`, `key_press` | Wayland-native screen capture and `ydotool` input automation (toggleable via `/desktop on`). |

---

## 🧠 Autonomous Architecture

### 1. Speculative Parallel Racing Engine
When generating complex implementations or solving tricky bugs, the agent can race multiple prompt strategies and candidate solutions concurrently via `asyncio`, picking the fastest, syntactically valid result.

### 2. Liquid Self-Refinement Loop
If a bash command or unit test fails:
- The error traceback and exit code are automatically trapped.
- A targeted self-healing diagnostic is injected into the agent's context.
- The agent immediately inspects the source, identifies the root cause, patches the file, and re-executes tests without human intervention.

### 3. Multi-Model Hot-Swapping
Easily switch between specialized models:
- **`qwen2.5-coder:7b-instruct-q5_K_M`** (Default): 100% GPU offloaded, ~38 tok/s, high AST and tool-calling precision.
- **`hermes3:8b`**: Uncensored generalist, strategic planning, adversarial steering, zero refusals.
- **`Qwen2.5-14B-Instruct-Q4_K_M`**: 24 GPU layers + 24 CPU P-core layers via native `llama-server` hybrid inference (~9 tok/s, deep reasoning).

---

## ⌨️ Interactive Slash Commands

Inside the `godmode` interactive shell:

| Command | Action |
| :--- | :--- |
| **`/help`** | Show commands and shortcut reference. |
| **`/model [name]`** | Show active model or hot-swap (e.g. `/model hermes3:8b`). |
| **`/tools`** | List all registered tools and their JSON schemas. |
| **`/status`** | View backend health, active model, and conversation context depth. |
| **`/desktop [on\|off]`** | Enable or disable Wayland desktop screenshot and mouse/keyboard automation. |
| **`/cwd [path]`** | Show or change current working directory. |
| **`/clear`** | Reset conversation history. |
| **`/exit`** or **`/quit`** | Exit the CLI. |

---

## 📂 Project Structure

```
Projects/05-antigravity-godmode-cli/
├── bin/
│   └── godmode               # Executable launcher (symlinked to ~/.local/bin/godmode)
├── config/
│   └── default_config.yaml   # Endpoints, model sampling, and tool configurations
├── core/
│   ├── agent.py              # Autonomous ReAct loop with multi-agent orchestration
│   ├── context.py            # Sliding window token and message history manager
│   ├── engine.py             # Resilient Ollama / llama.cpp / OpenAI client
│   ├── speculative.py        # Speculative racing and liquid refinement engine
│   └── types.py              # Pydantic schemas for messages, tools, and responses
├── tools/
│   ├── base.py               # Tool registry and JSON schema generator
│   ├── bash.py               # Persistent shell with working directory tracking
│   ├── files.py              # view_file, replace_file_content, write_file
│   ├── search.py             # ripgrep and fd search tools
│   ├── web.py                # DuckDuckGo and webpage reader
│   └── desktop.py            # Wayland screenshot and ydotool input control
├── ui/
│   ├── terminal.py           # Rich terminal output (diffs, banners, spinners)
│   └── prompt.py             # prompt_toolkit interactive shell with history
├── main.py                   # CLI entrypoint
└── README.md
```
