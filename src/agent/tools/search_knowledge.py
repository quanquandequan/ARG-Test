"""统一的知识查询工具：优先知识库，按需补充网页搜索。"""

from __future__ import annotations

from src.agent.base_tool import BaseTool
from src.agent.tools.search_kb import KnowledgeBaseTool
from src.agent.tools.web_search import WebSearchTool
from src.core.logging import get_logger
from src.retriever.retrieval_engine import RetrievalEngine

logger = get_logger(__name__)


def _try_create_rewriter():
    """创建 QueryRewriter；disabled、LLM 未配置或 prompt 缺失时静默返回 None。"""
    try:
        from src.core.config import get_config
        cfg = get_config().get("query_rewriter", {})
        if not cfg.get("enabled", True):
            return None
        from src.llm.factory import get_llm
        from src.retriever.query_rewriter import QueryRewriter
        timeout = float(cfg.get("timeout_seconds", 8.0))
        return QueryRewriter(get_llm(), timeout_seconds=timeout)
    except Exception as exc:
        logger.warning("query_rewriter_unavailable", error=str(exc))
        return None


class SearchKnowledgeTool(BaseTool):
    """对 Agent 暴露单一的知识查询入口。"""

    def __init__(
        self,
        retrieval_engine: RetrievalEngine,
        web_tool: WebSearchTool | None = None,
    ):
        rewriter = _try_create_rewriter()
        self._kb_tool = KnowledgeBaseTool(retrieval_engine, query_rewriter=rewriter)
        self._web_tool = web_tool or WebSearchTool()

    @property
    def name(self) -> str:
        return "search_knowledge"

    @property
    def description(self) -> str:
        return (
            "统一查询知识信息：优先搜索知识库；只有知识库无命中时才补充网页搜索。"
            "知识库与网页结果会明确分区，不能混作同一事实来源。"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "要查询的问题或关键词",
                },
                "need_fresh_info": {
                    "type": "boolean",
                    "description": "是否明确需要实时互联网信息；为 true 时无论知识库是否命中都会补充网页结果",
                },
                "top_k": {
                    "type": "integer",
                    "description": "知识库返回片段数量，默认 8",
                },
                "filters": {
                    "type": "object",
                    "description": "知识库检索过滤条件（可选）",
                },
                "num_web_results": {
                    "type": "integer",
                    "description": "网页搜索返回条数，默认 5",
                },
            },
            "required": ["query"],
        }

    async def execute(
        self,
        query: str = "",
        need_fresh_info: bool = False,
        top_k: int = 8,
        filters: dict | None = None,
        num_web_results: int = 5,
        debug_queries: bool = False,
        **kwargs,
    ) -> str:
        if not query.strip():
            return "错误：请提供 query 参数。"

        kb_result = await self._kb_tool.search_typed(
            query=query,
            top_k=top_k,
            filters=filters,
            debug_queries=debug_queries,
        )
        should_use_web = kb_result.hit_count == 0 or need_fresh_info

        lines = ["【知识库结果】", kb_result.content]
        if should_use_web:
            web_result = await self._web_tool.execute(
                query=query,
                num_results=num_web_results,
            )
            reason = "需要实时信息" if need_fresh_info else "知识库无命中"
            lines += ["", f"【网页结果（{reason}）】", web_result]

        return "\n".join(lines)
