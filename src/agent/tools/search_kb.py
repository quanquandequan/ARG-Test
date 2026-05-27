"""Knowledge base search tool — wraps RetrievalEngine for RAG retrieval."""

from src.agent.base_tool import BaseTool
from src.retriever.retrieval_engine import RetrievalEngine


class KnowledgeBaseTool(BaseTool):
    """Search the enterprise knowledge base for relevant documents.

    The Agent should call this tool when the user's question requires
    information from the knowledge base. Returns formatted document
    chunks with source citations.
    """

    def __init__(self, retrieval_engine: RetrievalEngine):
        self._retrieval_engine = retrieval_engine

    @property
    def name(self) -> str:
        return "knowledge_search"

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

    async def execute(
        self,
        query: str = "",
        top_k: int = 5,
        filters: dict | None = None,
        **kwargs,
    ) -> str:
        results = await self._retrieval_engine.search(
            query=query,
            top_k=20,
            final_k=top_k,
            filters=filters,
        )
        if not results:
            return "未找到相关文档。"

        lines: list[str] = [f"找到 {len(results)} 个相关文档片段：\n"]
        for i, r in enumerate(results, start=1):
            source = r.document_id or "unknown"
            score = r.score
            lines.append(f"[{i}] (来源: {source}, 相关度: {score:.2f})\n{r.content}")
        return "\n\n".join(lines)
