"""用于外部实时信息的网页搜索工具。

实现说明：**仅尽力而为**
------------------------------------------
本工具抓取 DuckDuckGo HTML endpoint，该端点公开但不是官方 API。
如果 DuckDuckGo 修改 HTML 结构，本工具可能失效。
仅在知识库不包含所需信息时使用。
若生产部署需要可靠网页搜索，请将 ``_search_duckduckgo`` 替换为正式 API
调用（例如 Serper、Brave Search、Google Custom Search）。

配置（通过 ``configs/default.yaml``）：

.. code-block:: yaml

    agent:
      tools:
        - knowledge_search
        - web_search            # 移除此行即可完全禁用网页搜索

环境变量：

    WEB_SEARCH_TIMEOUT   秒数（默认 10）
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
    """搜索网页以获取实时 / 外部信息。

    **尽力而为**：当搜索服务不可用、返回 HTTP 错误或超时时优雅降级。
    不要在生产关键数据链路中依赖此工具。
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
        """抓取 DuckDuckGo HTML endpoint（尽力而为，非官方 API）。"""
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
