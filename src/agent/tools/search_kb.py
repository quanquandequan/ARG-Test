"""知识库搜索工具：封装 RetrievalEngine 执行 RAG 检索。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.agent.base_tool import BaseTool
from src.retriever.retrieval_engine import RetrievalEngine
from src.vectordb.base import SearchResult


@dataclass(slots=True, frozen=True)
class KnowledgeSearchResult:
    """知识库检索的结构化结果。"""

    content: str
    hit_count: int
    results: list[SearchResult]


@dataclass(slots=True, frozen=True)
class KnowledgeSourceCategory:
    """知识库来源分类，用于控制事实优先级。"""

    key: str
    title: str
    priority: int


_SOURCE_EXCEL_CASE = KnowledgeSourceCategory(
    key="excel_case",
    title="Excel测试用例（事实优先）",
    priority=0,
)
_SOURCE_BUG = KnowledgeSourceCategory(
    key="bug",
    title="Bug记录（辅助）",
    priority=1,
)
_SOURCE_XMIND = KnowledgeSourceCategory(
    key="xmind",
    title="XMind（辅助）",
    priority=2,
)
_SOURCE_OTHER = KnowledgeSourceCategory(
    key="other",
    title="其他知识库文档（背景）",
    priority=3,
)
_BUG_MARKERS = ("bug", "buglist", "缺陷", "acn_buglist")
_MIN_CANDIDATE_K = 120


class KnowledgeBaseTool(BaseTool):
    """在企业知识库中搜索相关文档。

    当用户问题需要知识库信息时，Agent 应调用此工具。
    返回带来源引用的格式化文档片段。
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
        result = await self.search_typed(query=query, top_k=top_k, filters=filters)
        return result.content

    async def search_typed(
        self,
        query: str = "",
        top_k: int = 5,
        filters: dict | None = None,
    ) -> KnowledgeSearchResult:
        """返回带命中数量的结构化检索结果，供复合工具判断 fallback。"""
        candidate_k = max(_MIN_CANDIDATE_K, top_k * 12)
        candidates = await self._retrieval_engine.retrieve_candidates(
            query=query,
            top_k=candidate_k,
            filters=filters,
        )
        if not candidates:
            return KnowledgeSearchResult(
                content="未找到相关文档。",
                hit_count=0,
                results=[],
            )

        rerank_pool = _build_source_aware_rerank_pool(candidates, top_k)
        results = await self._retrieval_engine.rerank_candidates(
            query=query,
            candidates=rerank_pool,
            top_k=len(rerank_pool),
        )
        results = _select_source_aware_results(results, top_k)
        if not results:
            return KnowledgeSearchResult(
                content="未找到相关文档。",
                hit_count=0,
                results=[],
            )

        results = _sort_by_source_priority(results)
        lines = _format_grouped_results(results)
        return KnowledgeSearchResult(
            content="\n\n".join(lines),
            hit_count=len(results),
            results=list(results),
        )


def classify_knowledge_source(result: SearchResult) -> KnowledgeSourceCategory:
    """识别知识库片段来源，兼容旧库的 source_path 元数据。"""
    metadata = dict(result.metadata or {})
    source_format = str(
        metadata.get("source_format")
        or metadata.get("format")
        or ""
    ).lower()
    source_path = str(metadata.get("source_path") or "").strip()
    source_name = str(metadata.get("source_name") or "").strip()
    source_ext = str(metadata.get("source_ext") or "").lower().strip()

    path = Path(source_path) if source_path else None
    filename = (
        source_name
        or (path.name if path is not None else "")
        or str(result.document_id or "")
    )
    filename_lower = filename.lower()
    ext = source_ext or (path.suffix.lower() if path is not None else "")
    content_lower = result.content.lower()

    if _is_bug_source(filename_lower, content_lower):
        return _SOURCE_BUG
    if source_format == "xmind" or ext == ".xmind":
        return _SOURCE_XMIND
    if source_format in {"xlsx", "xlsm"} or ext in {".xlsx", ".xlsm"}:
        return _SOURCE_EXCEL_CASE
    return _SOURCE_OTHER


def _is_bug_source(filename_lower: str, content_lower: str) -> bool:
    return (
        any(marker in filename_lower for marker in _BUG_MARKERS)
        or "bug key:" in content_lower
    )


def _sort_by_source_priority(results: list[SearchResult]) -> list[SearchResult]:
    return sorted(
        results,
        key=lambda result: (
            classify_knowledge_source(result).priority,
            -float(result.score or 0.0),
        ),
    )


def _build_source_aware_rerank_pool(
    candidates: list[SearchResult],
    top_k: int,
) -> list[SearchResult]:
    """构造来源感知候选池，确保 Excel 候选不会在 rerank 前丢失。"""
    excel = _by_category(candidates, "excel_case")
    bug = _by_category(candidates, "bug")
    xmind = _by_category(candidates, "xmind")
    other = _by_category(candidates, "other")

    pool = []
    pool.extend(excel)
    pool.extend(bug[: max(top_k, 10)])
    pool.extend(xmind[: max(top_k, 10)])
    pool.extend(other[: max(top_k, 10)])
    return _dedupe_results(pool)


def _select_source_aware_results(
    ranked: list[SearchResult],
    top_k: int,
) -> list[SearchResult]:
    """从 rerank 结果中选择最终展示项，保留 Excel 并限制辅助来源数量。"""
    excel = _by_category(ranked, "excel_case")
    bug = _by_category(ranked, "bug")
    xmind = _by_category(ranked, "xmind")
    other = _by_category(ranked, "other")

    selected: list[SearchResult] = []
    selected.extend(excel[: max(top_k, 5)])
    remaining = max(top_k - len(selected), 0)
    if remaining:
        selected.extend(bug[:remaining])
    remaining = max(top_k - len(selected), 0)
    if remaining:
        selected.extend(xmind[:remaining])
    remaining = max(top_k - len(selected), 0)
    if remaining:
        selected.extend(other[:remaining])

    if not selected:
        selected.extend(ranked[:top_k])
    return _dedupe_results(selected)


def _by_category(results: list[SearchResult], key: str) -> list[SearchResult]:
    return [
        result
        for result in results
        if classify_knowledge_source(result).key == key
    ]


def _dedupe_results(results: list[SearchResult]) -> list[SearchResult]:
    seen: set[str] = set()
    deduped: list[SearchResult] = []
    for result in results:
        identity = result.id or f"{result.document_id}:{hash(result.content)}"
        if identity in seen:
            continue
        seen.add(identity)
        deduped.append(result)
    return deduped


def _format_grouped_results(results: list[SearchResult]) -> list[str]:
    lines: list[str] = [f"找到 {len(results)} 个相关文档片段。"]
    if any(classify_knowledge_source(result).key == "excel_case" for result in results):
        lines.append("【来源说明】已命中 Excel 测试用例，功能结论应优先依据该分区。")
    elif any(
        classify_knowledge_source(result).key in {"bug", "xmind"}
        for result in results
    ):
        lines.append(
            "【来源说明】未命中 Excel 测试用例，以下 Bug/XMind 仅为辅助参考，"
            "不能作为当前功能完整事实。"
        )
    current_key = ""
    item_index = 1
    for result in results:
        category = classify_knowledge_source(result)
        if category.key != current_key:
            lines.extend(["", f"【知识库结果：{category.title}】"])
            current_key = category.key

        lines.append(_format_result_item(item_index, result))
        item_index += 1
    return lines


def _format_result_item(index: int, result: SearchResult) -> str:
    metadata = dict(result.metadata or {})
    source_path = str(metadata.get("source_path") or "").strip()
    source = str(metadata.get("source_name") or "").strip()
    if not source and source_path:
        source = Path(source_path).name
    if not source:
        source = result.document_id
    return (
        f"[{index}] (来源: {source or 'unknown'}, 相关度: {result.score:.2f})\n"
        f"{result.content}"
    )
