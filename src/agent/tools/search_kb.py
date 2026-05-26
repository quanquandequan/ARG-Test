"""Knowledge base search tool — wraps Generator for RAG retrieval."""

from src.agent.base_tool import BaseTool
from src.generation.generator import Generator


class KnowledgeBaseTool(BaseTool):
    """Search the enterprise knowledge base for relevant documents.

    The Agent should call this tool when the user's question requires
    information from the knowledge base. Returns formatted document
    chunks with source citations.
    """

    def __init__(self, generator: Generator):
        self._generator = generator

    @property
    def name(self) -> str:
        return "search_kb"

    @property
    def description(self) -> str:
        return (
            "在知识库中搜索与查询相关的文档内容。"
            "当用户的问题需要从知识库获取信息时使用此工具。"
            "返回带有来源编号的文档片段。"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索查询，应使用与知识库语言一致的精确关键词",
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回的文档片段数量，默认为 5",
                },
                "filters": {
                    "type": "object",
                    "description": "可选的元数据过滤条件，例如按文档来源筛选",
                },
            },
            "required": ["query"],
        }

    async def execute(self, query: str = "", top_k: int = 5, filters: dict | None = None, **kwargs) -> str:
        result = await self._generator.query(query=query, final_k=top_k, filters=filters)
        if not result.citations:
            return "未找到相关文档。"

        lines: list[str] = [f"找到 {len(result.citations)} 个相关文档片段：\n"]
        for i, c in enumerate(result.citations, start=1):
            source = c.source_path or "unknown"
            lines.append(f"[{i}] (来源: {source}, 相关度: {c.relevance_score:.2f})\n{c.text}")
        return "\n\n".join(lines)
