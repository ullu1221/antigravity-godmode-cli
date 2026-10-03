import time
import re
import requests
from typing import Optional, List, Dict, Any
from tools.base import BaseTool
from core.types import ToolResult


class WebSearchTool(BaseTool):
    name = "web_search"
    description = (
        "Perform a live web or news search via DuckDuckGo to retrieve real-time documentation, "
        "breaking news, current events, software release info, or solutions."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search keywords or question."},
            "max_results": {"type": "integer", "description": "Number of results to return (default: 5).", "default": 5}
        },
        "required": ["query"]
    }

    async def execute(self, query: str, max_results: int = 5, **kwargs) -> ToolResult:
        start_time = time.time()
        try:
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS

            query_lower = query.lower()
            is_news = any(kw in query_lower for kw in [
                "news", "today", "latest", "breaking", "current", "update", "yesterday", "headline", "2026"
            ])

            results: List[Dict[str, Any]] = []
            ddgs_client = DDGS()

            # For news/current event queries, fetch news directly first
            if is_news:
                try:
                    news_results = list(ddgs_client.news(query, max_results=max_results))
                    if news_results:
                        formatted = []
                        for i, r in enumerate(news_results, 1):
                            title = r.get("title", "Untitled")
                            link = r.get("url") or r.get("href", "")
                            source = r.get("source", "Unknown Source")
                            date = r.get("date", "")
                            snippet = r.get("body", "")
                            formatted.append(
                                f"[{i}] {title}\n"
                                f"    Source: {source} | Date: {date}\n"
                                f"    Summary: {snippet}\n"
                                f"    URL: {link}"
                            )
                        return ToolResult(
                            tool_name=self.name,
                            content="\n\n".join(formatted),
                            execution_time=time.time() - start_time
                        )
                except Exception:
                    pass

            # Standard web text search
            text_results = list(ddgs_client.text(query, max_results=max_results))
            if text_results:
                formatted = []
                for i, r in enumerate(text_results, 1):
                    title = r.get("title", "Untitled")
                    link = r.get("href", "")
                    snippet = r.get("body", "")
                    formatted.append(f"[{i}] {title}\n    URL: {link}\n    Snippet: {snippet}")

                return ToolResult(
                    tool_name=self.name,
                    content="\n\n".join(formatted),
                    execution_time=time.time() - start_time
                )

            # Fallback to news if standard search returned nothing
            try:
                news_fallback = list(ddgs_client.news(query, max_results=max_results))
                if news_fallback:
                    formatted = []
                    for i, r in enumerate(news_fallback, 1):
                        title = r.get("title", "Untitled")
                        link = r.get("url", "")
                        source = r.get("source", "")
                        snippet = r.get("body", "")
                        formatted.append(f"[{i}] {title} ({source})\n    URL: {link}\n    Summary: {snippet}")
                    return ToolResult(
                        tool_name=self.name,
                        content="\n\n".join(formatted),
                        execution_time=time.time() - start_time
                    )
            except Exception:
                pass

            return ToolResult(
                tool_name=self.name,
                content=f"No results found for query: '{query}'. Try broader keywords.",
                execution_time=time.time() - start_time
            )

        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                content=f"Web search error: {str(e)}",
                is_error=True,
                execution_time=time.time() - start_time
            )


class ReadUrlTool(BaseTool):
    name = "read_url"
    description = "Fetch and parse webpage, article, or documentation content from a URL into clean readable text."
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The URL to fetch and read."}
        },
        "required": ["url"]
    }

    async def execute(self, url: str, **kwargs) -> ToolResult:
        start_time = time.time()
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:130.0) Gecko/20100101 Firefox/130.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Upgrade-Insecure-Requests": "1"
        }

        try:
            resp = requests.get(url, headers=headers, timeout=12, allow_redirects=True)
            
            # Detect CDN / Anti-Bot blockages
            if resp.status_code in (403, 410, 429, 503) or "errors.edgesuite.net" in resp.text:
                return ToolResult(
                    tool_name=self.name,
                    content=(
                        f"[Notice]: Access to '{url}' returned HTTP {resp.status_code} (CDN/Bot Protection). "
                        f"You do not need to fetch this URL again; synthesize your answer from the search summaries "
                        f"already returned by web_search or search an alternative open source."
                    ),
                    is_error=False,  # Mark as false so agent does not panic
                    execution_time=time.time() - start_time
                )

            html_content = resp.text
            cleaned_text = ""

            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html_content, "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "aside", "svg", "form"]):
                    tag.decompose()

                # Extract main content elements
                elements = soup.find_all(["h1", "h2", "h3", "h4", "p", "article", "pre", "code", "li"])
                blocks = []
                for el in elements:
                    txt = el.get_text(separator=" ", strip=True)
                    if len(txt) > 25:
                        blocks.append(txt)
                cleaned_text = "\n\n".join(blocks)
            except Exception:
                pass

            if not cleaned_text:
                txt = re.sub(r"<script[^>]*>.*?</script>", "", html_content, flags=re.DOTALL)
                txt = re.sub(r"<style[^>]*>.*?</style>", "", txt, flags=re.DOTALL)
                txt = re.sub(r"<[^>]+>", " ", txt)
                cleaned_text = re.sub(r"\s+", " ", txt).strip()

            if len(cleaned_text) > 4500:
                cleaned_text = cleaned_text[:4500] + "\n\n... [content truncated at 4500 characters]"

            return ToolResult(
                tool_name=self.name,
                content=cleaned_text or f"No readable text extracted from {url}.",
                execution_time=time.time() - start_time
            )

        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                content=(
                    f"Could not connect to URL '{url}': {str(e)}. "
                    f"Use the news/web search snippets already available to answer."
                ),
                is_error=False,
                execution_time=time.time() - start_time
            )
