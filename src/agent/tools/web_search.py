"""Web search tool for external real-time information.

Implementation note — **best-effort only**
------------------------------------------
This tool scrapes the DuckDuckGo HTML endpoint which is public but not an
official API.  It may break if DuckDuckGo changes its HTML structure.
Use it only when the knowledge base does not contain the required information.
For production deployments that require reliable web search, replace
``_search_duckduckgo`` with a proper API call (e.g. Serper, Brave Search,
Google Custom Search).

Configuration (via ``configs/default.yaml``):

.. code-block:: yaml

    agent:
      tools:
        - knowledge_search
        - web_search            # remove this line to disable web search entirely

Environment variables:

    WEB_SEARCH_TIMEOUT   seconds (default: 10)
"""

from __future__ import annotations

import re

from src.agent.base_tool import BaseTool
from src.core.logging import get_logger

logger = get_logger(__name__)

_SNIPPET_RE = re.compile(
    r'class="result__snippet"[^>]*>(.*?)</a>',
    re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")
_DEFAULT_TIMEOUT = 10


class WebSearchTool(BaseTool):
    """Search the web for real-time / external information.

    **Best-effort**: falls back gracefully if the search service is
    unavailable, returns an HTTP error, or times out.  Do not rely on this
    tool for production-critical data paths.
    """

    def __init__(self, timeout: float = _DEFAULT_TIMEOUT):
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "搜索互联网获取实时信息（尽力而为，非保证可用）。"
            "当知识库中无法找到相关信息或需要最新数据时使用。"
            "返回搜索结果摘要（最多 num_results 条）。"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索查询关键词",
                },
                "num_results": {
                    "type": "integer",
                    "description": "返回结果数量，默认 5，最大 10",
                },
            },
            "required": ["query"],
        }

    async def execute(
        self, query: str = "", num_results: int = 5, **kwargs
    ) -> str:
        num_results = max(1, min(int(num_results), 10))
        try:
            return await self._search_duckduckgo(query, num_results)
        except Exception as e:
            logger.warning("web_search_error", query=query, error=str(e))
            return f"网页搜索暂时不可用，请稍后重试或仅使用知识库回答。原因: {e}"

    async def _search_duckduckgo(self, query: str, num_results: int) -> str:
        """Scrape DuckDuckGo HTML endpoint (best-effort, not an official API)."""
        import httpx

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (compatible; RAG-Pipeline/0.1; "
                        "+https://github.com/example/rag-pipeline)"
                    )
                },
                follow_redirects=True,
            )

        if resp.status_code != 200:
            return f"网页搜索失败: HTTP {resp.status_code}"

        snippets = _SNIPPET_RE.findall(resp.text)
        if not snippets:
            return "未找到相关网页结果。"

        lines: list[str] = [f"搜索 '{query}' 的结果：\n"]
        for i, s in enumerate(snippets[:num_results], start=1):
            text = _TAG_RE.sub("", s).strip()
            if text:
                lines.append(f"[{i}] {text}")

        return "\n\n".join(lines) if len(lines) > 1 else "未找到相关网页结果。"
