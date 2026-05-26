"""Web search tool for external real-time information."""

from src.agent.base_tool import BaseTool


class WebSearchTool(BaseTool):
    """Search the web for real-time / external information.

    Falls back gracefully if the search service is unavailable or no API key is configured.
    """

    def __init__(self, api_key: str = "", engine: str = "duckduckgo"):
        self._api_key = api_key
        self._engine = engine

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "搜索互联网获取实时信息。"
            "当知识库中无法找到相关信息或需要最新数据时使用。"
            "返回搜索结果摘要。"
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
                    "description": "返回结果数量，默认 5",
                },
            },
            "required": ["query"],
        }

    async def execute(self, query: str = "", num_results: int = 5, **kwargs) -> str:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    "https://html.duckduckgo.com/html/",
                    params={"q": query},
                    headers={"User-Agent": "RAG-Pipeline/0.1"},
                )
                if resp.status_code != 200:
                    return f"网页搜索失败: HTTP {resp.status_code}"

                # Extract snippets from DuckDuckGo HTML results
                import re
                snippets = re.findall(
                    r'class="result__snippet"[^>]*>(.*?)</a>',
                    resp.text,
                    re.DOTALL,
                )
                if not snippets:
                    return "未找到相关网页。"

                lines = [f"搜索 '{query}' 的结果：\n"]
                for i, s in enumerate(snippets[:num_results], start=1):
                    text = re.sub(r"<[^>]+>", "", s).strip()
                    if text:
                        lines.append(f"[{i}] {text}")
                return "\n\n".join(lines) if len(lines) > 1 else "未找到相关网页。"

        except Exception as e:
            return f"网页搜索暂时不可用: {e}"
