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

# Bug boost：普通多 query 召回路径上，bug 记录每条内容仅一行摘要，
# 向量相似度天然低于多字段的测试用例行（实测最高分约 0.26，低于 0.3 阈值），
# 导致 bug 查询时 bug 行被测试用例行完全挤出。
# 与 Excel boost 对称，额外做一次只搜 bug 记录的专属向量召回，
# 并在候选池和最终选择阶段给 bug 记录优先名额。
_BUG_BOOST_K = 80
_BUG_BOOST_EXPR = 'content like "%Bug Key:%"'
_BUG_SOURCE_FILTER: dict = {"_raw": _BUG_BOOST_EXPR}
# 用户 query 里含这些词时视为 bug 查询，激活 bug boost 路径
_BUG_TRIGGER_WORDS = frozenset(["bug", "缺陷", "严重", "异常", "崩溃", "闪退", "故障"])

# XMind boost：XMind 覆盖的是叭嗒 App 之外的小程序/插件等场景（如百度/抖音小程序、
# 漫画插件阅读器），这类内容 Excel 测试用例基本不覆盖或已过时，数量也少
# （每文件 2-43 chunk），普通召回中会被 Excel 测试用例行（3000+）完全挤出 top-k。
# 仅当用户明确在问小程序/插件这类 XMind 独有场景时才激活，额外做一次只搜 xmind
# 的专属向量召回，并在候选池和最终选择阶段给 XMind 记录优先名额；
# 叭嗒 App 主体功能查询一律默认优先走 Excel（见下方普通分支），因为 Excel 是当前
# 版本最全、逻辑最新的测试用例，其余来源可能是历史版本迭代遗留、已过时。
_XMIND_BOOST_K = 60
_XMIND_BOOST_EXPR = 'metadata["source_format"] == "xmind"'
_XMIND_SOURCE_FILTER: dict = {"_raw": _XMIND_BOOST_EXPR}
# 用户 query 里含这些词时视为小程序/插件类查询，激活 xmind boost 路径
_XMIND_TRIGGER_WORDS = frozenset(["小程序", "插件"])

# 对比类查询：需要同时从 XMind（规格）和 Excel（测试用例）各取一部分，不能独占全部槽位
_COMPARISON_TRIGGER_WORDS = frozenset(["差异", "区别", "对比", "不同", "比较", "相比", "vs"])

# Excel 最低保底名额：xmind_query_mode / bug_query_mode 下，若 boost 来源候选数
# ≥ top_k，会把全部名额占满，导致"事实优先"（priority=0）的 Excel 测试用例
# 即使命中也完全挤空（例如 query 里带"小程序""插件"会触发 xmind_query_mode，
# 但同一功能在 Excel 里也可能有对应记录）。保底名额确保 Excel 命中时
# 至少能展示一部分，同时 XMind/Bug 仍拿大多数名额，不影响其原有优先级。
_MIN_EXCEL_FLOOR_RATIO = 4


def _excel_slot_floor(top_k: int) -> int:
    """xmind/bug boost 模式下给 Excel 预留的最低名额。"""
    return max(2, top_k // _MIN_EXCEL_FLOOR_RATIO)


def _is_bug_query(query: str) -> bool:
    """判断是否为 bug 相关查询。"""
    q = query.lower()
    return any(w in q for w in _BUG_TRIGGER_WORDS)


def _is_xmind_query(query: str) -> bool:
    """判断是否为小程序/插件类查询（这类场景只有 XMind 覆盖，应优先返回 XMind）。"""
    return any(w in query for w in _XMIND_TRIGGER_WORDS)


def _is_comparison_query(query: str) -> bool:
    """判断是否为对比类查询（需要 XMind + Excel 混合结果，不能独占槽位）。"""
    return any(w in query for w in _COMPARISON_TRIGGER_WORDS)


def _strip_bug_noise(query: str) -> str:
    """去掉 query 里的 bug 类噪声词，保留功能/模块词用于 bug boost 向量搜索。

    bug 记录本身已通过 filter 锁定，boost query 里不需要"bug/缺陷/严重"这类词，
    去掉后向量更贴近 bug 摘要里描述的具体症状，相似度会显著提升。
    """
    noise_words = ["bug", "缺陷", "严重", "异常", "崩溃", "闪退", "故障",
                   "都有什么", "有哪些", "有什么", "哪些"]
    result = query
    for w in noise_words:
        result = result.replace(w, "")
    return result.strip() or query


class KnowledgeBaseTool(BaseTool):
    """在企业知识库中搜索相关文档。

    当用户问题需要知识库信息时，Agent 应调用此工具。
    返回带来源引用的格式化文档片段。
    """

    def __init__(
        self,
        retrieval_engine: RetrievalEngine,
        query_rewriter: QueryRewriter | None = None,
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
                    "description": "返回的文档片段数量，默认为 8",
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
        top_k: int = 8,
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
        top_k: int = 8,
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

        # 生成 query variants + sub_queries：
        #   variants    = 原始 query + LLM/规则子查询，用于向量搜索（保证召回）
        #   sub_queries = 仅 LLM 子查询（去掉原始），用于 Rerank（过滤跨 App 噪声）
        sub_queries: list[str] = []
        if expand_query and max_query_variants > 1:
            if self._query_rewriter is not None:
                variants = await self._query_rewriter.rewrite(
                    query, max_variants=max_query_variants
                )
                if len(variants) <= 1:
                    # LLM 超时/失败 → 规则扩展兜底
                    variants = _expand_query(query, max_variants=max_query_variants)
                else:
                    # LLM 成功：子查询不含原始 query，用于 Rerank 时去掉干扰词
                    sub_queries = variants[1:]
            else:
                variants = _expand_query(query, max_variants=max_query_variants)
        else:
            variants = [query]

        # 1. 并发召回所有 variants 的候选（含原始 query，保证广召回）
        raw_candidates = await _multi_query_candidates(
            self._retrieval_engine, variants, per_query_k, filters
        )
        # 2. 专项 boost 召回（Excel + 按需 Bug），两路并发执行：
        #    · Excel boost：LLM 子查询聚焦，不含 App 名，过滤掉 bug 行，
        #      防止 bug 列表把测试用例挤出候选池。
        #    · Bug boost：仅在 bug 查询时激活；去掉 query 中的 bug 类噪声词后
        #      向量搜索，使 query 更贴近 bug 摘要里的具体症状描述。
        bug_query_mode = _is_bug_query(query)
        xmind_query_mode = _is_xmind_query(query)
        comparison_mode = _is_comparison_query(query)
        focus_query = sub_queries[0] if sub_queries else query

        excel_filter = dict(_EXCEL_SOURCE_FILTER)
        if filters:
            excel_filter.update(filters)
        boost_coros = [
            self._retrieval_engine.retrieve_candidates(
                query=focus_query,
                top_k=_EXCEL_BOOST_K,
                filters=excel_filter,
            )
        ]
        if bug_query_mode:
            bug_filter = dict(_BUG_SOURCE_FILTER)
            if filters:
                bug_filter.update(filters)
            bug_boost_query = _strip_bug_noise(focus_query)
            boost_coros.append(
                self._retrieval_engine.retrieve_candidates(
                    query=bug_boost_query,
                    top_k=_BUG_BOOST_K,
                    filters=bug_filter,
                )
            )
        if xmind_query_mode:
            xmind_filter = dict(_XMIND_SOURCE_FILTER)
            if filters:
                xmind_filter.update(filters)
            boost_coros.append(
                self._retrieval_engine.retrieve_candidates(
                    query=focus_query,
                    top_k=_XMIND_BOOST_K,
                    filters=xmind_filter,
                )
            )
        for boost in await asyncio.gather(*boost_coros):
            raw_candidates = raw_candidates + boost

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

        rerank_pool = _build_source_aware_rerank_pool(
            candidates,
            top_k,
            bug_query_mode,
            xmind_query_mode,
            comparison_mode,
        )
        # 子查询多路 rerank：对每个子查询独立打分，各候选取最高分；
        # 子查询不含原始 query 中的 App 名称等干扰词，能更准确地把跨 App 用例排在后面
        if sub_queries:
            results = await _multi_query_rerank(
                self._retrieval_engine, sub_queries, rerank_pool
            )
        else:
            results = await self._retrieval_engine.rerank_candidates(
                query=query,
                candidates=rerank_pool,
                top_k=len(rerank_pool),
            )
        results = _select_source_aware_results(
            results,
            top_k,
            bug_query_mode,
            xmind_query_mode,
            comparison_mode,
        )
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
                _build_debug_header(
                    query, variants, sub_queries, raw_count, dedup_count, len(rerank_pool)
                )
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


async def _multi_query_rerank(
    engine: RetrievalEngine,
    queries: list[str],
    candidates: list[SearchResult],
) -> list[SearchResult]:
    """多子查询 rerank：对每个子查询独立打分，按子查询轮询交叉排序。

    解决原始长 query 会把跨 App 用例排在正确 Sheet 内容前面的问题：
    每个短子查询更聚焦，能更准确地识别对应 Sheet 内容的相关度。

    子查询往往对应不同内容模块（如"动画/漫画/帖子"）。若直接按全局最高分
    排序，候选行数多或打分普遍偏高的模块会挤占其余模块的展示名额，
    导致"页面有哪些内容"类枚举问题只能看到 1-2 个模块。因此按子查询轮询
    交叉取候选（每轮各子查询贡献一个未出现过的候选），保证不同模块都有
    机会进入最终 top_k，而不仅是分数最高的单一模块。
    """
    if not candidates or not queries:
        return candidates

    tasks = [
        engine.rerank_candidates(query=q, candidates=candidates, top_k=len(candidates))
        for q in queries
    ]
    all_ranked: list[list[SearchResult]] = await asyncio.gather(*tasks)

    # 每个 candidate 取所有子查询打分中的最高分，用于展示「相关度」
    score_map: dict[str, float] = {}
    best_result: dict[str, SearchResult] = {}
    for ranked_list in all_ranked:
        for result in ranked_list:
            key = result.id or str(hash(result.content))
            score = float(result.score or 0.0)
            if score > score_map.get(key, 0.0):
                score_map[key] = score
                best_result[key] = result

    # 按子查询轮询交叉排序，保证模块多样性；同一轮内部仍按最高分排序
    seen: set[str] = set()
    interleaved: list[SearchResult] = []
    max_len = max((len(lst) for lst in all_ranked), default=0)
    for i in range(max_len):
        round_keys: list[str] = []
        for ranked_list in all_ranked:
            if i >= len(ranked_list):
                continue
            result = ranked_list[i]
            key = result.id or str(hash(result.content))
            if key in seen:
                continue
            seen.add(key)
            round_keys.append(key)
        round_keys.sort(key=lambda k: score_map.get(k, 0.0), reverse=True)
        interleaved.extend(best_result[k] for k in round_keys)

    return interleaved


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
    sub_queries: list[str],
    raw_count: int,
    dedup_count: int,
    pool_size: int,
) -> str:
    lines = ["【检索调试】", f"原始 query: {original_query}"]
    if len(variants) > 1:
        lines.append("向量搜索 query（全部 variants）:")
        for i, v in enumerate(variants, 1):
            lines.append(f"  {i}. {v}")
    else:
        lines.append("（未启用 query 扩展）")
    if sub_queries:
        lines.append("Rerank query（LLM 子查询，去原始）:")
        for i, v in enumerate(sub_queries, 1):
            lines.append(f"  {i}. {v}")
    else:
        lines.append("Rerank query: 原始 query（无 LLM 子查询）")
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
    bug_query_mode: bool = False,
    xmind_query_mode: bool = False,
    comparison_mode: bool = False,
) -> list[SearchResult]:
    """构造来源感知候选池，确保主力来源候选不会在 rerank 前丢失。

    bug_query_mode=True 时 bug 记录全量保留；
    xmind_query_mode=True 时 XMind 记录全量保留（与 bug 模式对称）；
    comparison_mode=True 时 XMind 和 Excel 各全量保留（对比查询需要双来源）；
    普通模式下 Excel 全量保留。
    """
    excel = _by_category(candidates, "excel_case")
    bug = _by_category(candidates, "bug")
    xmind = _by_category(candidates, "xmind")
    other = _by_category(candidates, "other")

    pool: list[SearchResult] = []
    if bug_query_mode:
        pool.extend(bug)                      # bug 查询：bug 记录全量保留
        pool.extend(excel[: max(top_k, 10)])
        pool.extend(xmind[: max(top_k, 10)])
    elif xmind_query_mode and comparison_mode:
        pool.extend(xmind)                    # 对比查询：XMind 和 Excel 都全量保留
        pool.extend(excel)
        pool.extend(bug[: max(top_k, 10)])
    elif xmind_query_mode:
        pool.extend(xmind)                    # 纯功能查询：XMind 全量保留
        pool.extend(excel[: max(top_k, 10)])
        pool.extend(bug[: max(top_k, 10)])
    else:
        pool.extend(excel)                    # 普通查询：Excel 全量保留
        pool.extend(bug[: max(top_k, 10)])
        pool.extend(xmind[: max(top_k, 10)])
    pool.extend(other[: max(top_k, 10)])
    return _dedupe_results(pool)


def _select_source_aware_results(
    ranked: list[SearchResult],
    top_k: int,
    bug_query_mode: bool = False,
    xmind_query_mode: bool = False,
    comparison_mode: bool = False,
) -> list[SearchResult]:
    """从 rerank 结果中选择最终展示项。

    bug_query_mode=True  → bug 记录优先占槽，但为 Excel 保留最低名额，剩余给 Excel。
    xmind_query_mode=True AND comparison_mode=True
                         → XMind / Excel 各占一半槽位（对比查询双来源）。
    xmind_query_mode=True → XMind 优先占槽，但为 Excel 保留最低名额，剩余给 Excel。
    普通模式           → Excel 优先，剩余给 bug/xmind。

    xmind/bug 优先占槽时会先扣除 Excel 的保底名额（见 `_excel_slot_floor`），
    避免 boost 来源命中数量 ≥ top_k 时把"事实优先"的 Excel 完全挤空。
    """
    excel = _by_category(ranked, "excel_case")
    bug = _by_category(ranked, "bug")
    xmind = _by_category(ranked, "xmind")
    other = _by_category(ranked, "other")

    selected: list[SearchResult] = []
    if bug_query_mode:
        excel_floor = min(len(excel), _excel_slot_floor(top_k))
        selected.extend(bug[: max(top_k - excel_floor, 0)])
        remaining = max(top_k - len(selected), 0)
        if remaining:
            selected.extend(excel[:remaining])
    elif xmind_query_mode and comparison_mode:
        # 对比查询：XMind 和 Excel 各占一半，保证双来源都有代表
        half = max(top_k // 2, 2)
        selected.extend(xmind[:half])
        selected.extend(excel[:half])
    elif xmind_query_mode:
        excel_floor = min(len(excel), _excel_slot_floor(top_k))
        selected.extend(xmind[: max(top_k - excel_floor, 0)])
        remaining = max(top_k - len(selected), 0)
        if remaining:
            selected.extend(excel[:remaining])
    else:
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
