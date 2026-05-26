"""Reranker tool — lets the Agent explicitly re-rank retrieval results."""

from src.agent.base_tool import BaseTool
from src.core.logging import get_logger
from src.retriever.reranker_base import BaseReranker
from src.vectordb.base import SearchResult

logger = get_logger(__name__)


class RerankerTool(BaseTool):
    """Re-rank a set of retrieved chunks for better precision.

    The Agent can call this after search_kb if it wants to tighten
    the result quality before generating an answer.
    """

    def __init__(self, reranker: BaseReranker):
        self._reranker = reranker

    @property
    def name(self) -> str:
        return "reranker"

    @property
    def description(self) -> str:
        return (
            "对已检索的文档片段进行重排序，选出最相关的片段。"
            "当 search_kb 返回较多结果、需要精选最相关的片段时调用。"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "原始用户查询",
                },
                "chunks_text": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "需要重排序的文档片段文本列表",
                },
                "top_k": {
                    "type": "integer",
                    "description": "重排序后保留的数量，默认 3",
                },
            },
            "required": ["query", "chunks_text"],
        }

    async def execute(
        self,
        query: str = "",
        chunks_text: list[str] | None = None,
        top_k: int = 3,
        **kwargs,
    ) -> str:
        if not chunks_text:
            return "没有需要重排序的片段。"

        # Build lightweight SearchResult objects for the reranker
        candidates = [
            SearchResult(
                id=f"r_{i}",
                document_id="",
                content=text,
                score=0.0,
            )
            for i, text in enumerate(chunks_text)
        ]

        try:
            ranked = await self._reranker.rerank(query, candidates, top_k=top_k)
        except Exception as e:
            logger.warning("reranker_tool_failed", error=str(e))
            # Fallback: return first top_k as-is
            ranked = candidates[:top_k]

        lines = ["重排序后的结果：\n"]
        for i, r in enumerate(ranked, start=1):
            lines.append(f"[{i}] (相关度: {r.score:.2f})\n{r.content}")
        return "\n\n".join(lines)
