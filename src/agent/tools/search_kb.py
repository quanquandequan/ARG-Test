"""知识库搜索工具：封装 RetrievalEngine 执行 RAG 检索。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.agent.base_tool import BaseTool
from src.agent.tools.kb_compressor import compress_results
from src.retriever.query_expansion import expand_query as _expand_query
from src.retriever.retrieval_engine import RetrievalEngine
from src.vectordb.base import SearchResult

if TYPE_CHECKING:
    from src.retriever.query_rewriter import QueryRewriter


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

# 多 query 召回配置
_PER_QUERY_CANDIDATE_K = 40    # 每个 query variant 的向量召回数
_MAX_MERGED_CANDIDATES = 240   # 合并去重后进入 rerank 前处理的上限

# Excel boost：Bug 列表数量庞大，全量向量搜索时容易把测试用例全部挤出 top-k，
# 因此额外做一次仅限 Excel 的带过滤向量搜索，强制保留足量测试用例候选。
# 注：Bug 列表也是 xlsx，通过 content not like "%Bug Key:%" 排除。
_EXCEL_BOOST_K = 80
_EXCEL_BOOST_EXPR = (
    '(metadata["source_format"] == "xlsx" or metadata["source_format"] == "xlsm")'
    ' and not (content like "%Bug Key:%")'
)
_EXCEL_SOURCE_FILTER: dict = {"_raw": _EXCEL_BOOST_EXPR}


class KnowledgeBaseTool(BaseTool):
    """在企业知识库中搜索相关文档。

    当用户问题需要知识库信息时，Agent 应调用此工具。
    返回带来源引用的格式化文档片段。
    """

    def __init__(
        self,
        retrieval_engine: RetrievalEngine,
        query_rewriter: "QueryRewriter | None" = None,
    ):
        self._retrieval_engine = retrieval_engine
        # LLM 改写器：有则用 LLM 扩展，无则回退规则扩展（UI 别名）
        self._query_rewriter = query_rewriter

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
        debug_queries: bool = False,
        **kwargs,
    ) -> str:
        result = await self.search_typed(
            query=query,
            top_k=top_k,
            filters=filters,
            debug_queries=debug_queries,
        )
        return result.content

    async def search_typed(
        self,
        query: str = "",
        top_k: int = 5,
        filters: dict | None = None,
        expand_query: bool = True,
        max_query_variants: int = 5,
        debug_queries: bool = False,
    ) -> KnowledgeSearchResult:
        """返回带命中数量的结构化检索结果，供复合工具判断 fallback。

        expand_query=True 时自动扩展检索 query，提高召回面；
        rerank 始终使用原始 query，避免扩展词影响相关度排序。
        """
        per_query_k = max(_PER_QUERY_CANDIDATE_K, top_k * 8)

        # 生成 query variants：优先用 LLM 改写，回退到规则扩展（UI 别名）
        if expand_query and max_query_variants > 1:
            if self._query_rewriter is not None:
                variants = await self._query_rewriter.rewrite(
                    query, max_variants=max_query_variants
                )
                # LLM 超时/失败时 variants == [query]，叠加规则扩展补充召回面
                if len(variants) <= 1:
                    variants = _expand_query(query, max_variants=max_query_variants)
            else:
                variants = _expand_query(query, max_variants=max_query_variants)
        else:
            variants = [query]

        # 1. 并发召回所有 variants 的候选（全量，无来源过滤）
        raw_candidates = await _multi_query_candidates(
            self._retrieval_engine, variants, per_query_k, filters
        )
        # 2. Excel 专项召回：Bug 列表数量庞大，全量搜索时常把测试用例挤出 top-k；
        #    用 source_format 过滤做额外一次向量搜索，强制把 Excel 候选纳入池中。
        excel_filter = dict(_EXCEL_SOURCE_FILTER)
        if filters:
            excel_filter.update(filters)
        excel_boost = await self._retrieval_engine.retrieve_candidates(
            query=query,  # 用原始 query 做 Excel 专项召回，避免扩展词引入噪声
            top_k=_EXCEL_BOOST_K,
            filters=excel_filter,
        )
        raw_candidates = raw_candidates + excel_boost

        raw_count = len(raw_candidates)
        candidates = _stable_dedup(raw_candidates)
        candidates = candidates[:_MAX_MERGED_CANDIDATES]
        dedup_count = len(candidates)

        if not candidates:
            return KnowledgeSearchResult(
                content="未找到相关文档。",
                hit_count=0,
                results=[],
            )

        rerank_pool = _build_source_aware_rerank_pool(candidates, top_k)
        # rerank 始终用原始 query，不受扩展词影响
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
        compressed = compress_results(results, query)
        lines = _format_grouped_results(results, compressed)

        if debug_queries:
            lines = [
                _build_debug_header(query, variants, raw_count, dedup_count, len(rerank_pool))
            ] + lines

        return KnowledgeSearchResult(
            content="\n\n".join(lines),
            hit_count=len(results),
            results=list(results),
        )


# ── 多 query 并发召回 ─────────────────────────────────────────────────────────

async def _multi_query_candidates(
    engine: RetrievalEngine,
    variants: list[str],
    per_query_k: int,
    filters: dict | None,
) -> list[SearchResult]:
    """并发为每个 query variant 召回候选，合并返回（未去重）。"""
    tasks = [
        engine.retrieve_candidates(query=v, top_k=per_query_k, filters=filters)
        for v in variants
    ]
    batches = await asyncio.gather(*tasks)
    merged: list[SearchResult] = []
    for batch in batches:
        merged.extend(batch)
    return merged


# ── 稳定去重（多 query 合并用） ───────────────────────────────────────────────

def _stable_dedup(results: list[SearchResult]) -> list[SearchResult]:
    """多 query 合并后去重：按稳定元数据 key > UUID > 内容哈希三层兜底。

    稳定 key 使 Excel 同一行通过不同 query 命中时只保留一条，
    比单纯依赖 UUID 更健壮（抵御极端情况下的 ID 漂移）。
    """
    seen_stable: set[str] = set()
    seen_ids: set[str] = set()
    seen_content: set[int] = set()
    deduped: list[SearchResult] = []

    for result in results:
        meta = dict(result.metadata or {})
        stable_key = _build_stable_key(result, meta)
        row_id = result.id or ""
        content_key = hash(result.content)

        if stable_key and stable_key in seen_stable:
            continue
        if row_id and row_id in seen_ids:
            continue
        if content_key in seen_content:
            continue

        if stable_key:
            seen_stable.add(stable_key)
        if row_id:
            seen_ids.add(row_id)
        seen_content.add(content_key)
        deduped.append(result)

    return deduped


def _build_stable_key(result: SearchResult, meta: dict) -> str:
    """构建稳定去重 key：Excel 行用 source+sheet+row；其他用 doc+chunk。"""
    source_name = str(meta.get("source_name") or "").strip()
    sheet_name = str(meta.get("sheet_name") or "").strip()
    row_index = meta.get("row_index")

    if source_name and sheet_name and row_index is not None:
        return f"xlsx:{source_name}:{sheet_name}:{row_index}"

    doc_id = str(result.document_id or "").strip()
    chunk_index = meta.get("chunk_index")
    if doc_id and chunk_index is not None:
        return f"doc:{doc_id}:{chunk_index}"

    return ""  # 兜底由内容哈希去重


# ── Debug 输出 ────────────────────────────────────────────────────────────────

def _build_debug_header(
    original_query: str,
    variants: list[str],
    raw_count: int,
    dedup_count: int,
    pool_size: int,
) -> str:
    lines = ["【检索调试】", f"原始 query: {original_query}"]
    if len(variants) > 1:
        lines.append("扩展 query:")
        for i, v in enumerate(variants, 1):
            lines.append(f"  {i}. {v}")
    else:
        lines.append("（未启用 query 扩展）")
    lines.append(f"候选统计: raw={raw_count}, dedup={dedup_count}, rerank_pool={pool_size}")
    return "\n".join(lines)


# ── 来源分类 ──────────────────────────────────────────────────────────────────

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

    pool: list[SearchResult] = []
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
    """候选池内部去重：UUID + 内容哈希（池内调用，不跨 query）。"""
    seen_ids: set[str] = set()
    seen_content: set[int] = set()
    deduped: list[SearchResult] = []
    for result in results:
        content_key = hash(result.content)
        row_id = result.id or ""
        if (row_id and row_id in seen_ids) or content_key in seen_content:
            continue
        if row_id:
            seen_ids.add(row_id)
        seen_content.add(content_key)
        deduped.append(result)
    return deduped


# ── 格式化输出 ────────────────────────────────────────────────────────────────

def _format_grouped_results(
    results: list[SearchResult],
    compressed: dict[str, str],
) -> list[str]:
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

        lines.append(_format_result_item(item_index, result, compressed))
        item_index += 1
    return lines


def _format_result_item(
    index: int,
    result: SearchResult,
    compressed: dict[str, str],
) -> str:
    metadata = dict(result.metadata or {})
    source_path = str(metadata.get("source_path") or "").strip()
    source = str(metadata.get("source_name") or "").strip()
    if not source and source_path:
        source = Path(source_path).name
    if not source:
        source = result.document_id
    # 优先使用压缩内容（规则抽取相关字段）；压缩缺失时回退展示原始内容
    display_content = compressed.get(result.id or "", result.content)
    return (
        f"[{index}] (来源: {source or 'unknown'}, 相关度: {result.score:.2f})\n"
        f"{display_content}"
    )
