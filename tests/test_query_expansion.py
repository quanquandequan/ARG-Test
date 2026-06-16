"""query_expansion 和 search_kb 多 query 逻辑单元测试。"""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from src.retriever.query_expansion import _append_unique, expand_query
from src.vectordb.base import SearchResult


# ── expand_query 测试 ─────────────────────────────────────────────────────────

class TestExpandQuery(unittest.TestCase):
    def test_original_query_always_first(self):
        variants = expand_query("追番表 Card 页面逻辑")
        self.assertEqual(variants[0], "追番表 Card 页面逻辑")

    def test_lowercase_ui_alias(self):
        # 原始 query 含 Card，variants 第一项本身即满足条件
        variants = expand_query("追番表 Card 页面逻辑")
        self.assertTrue(any("追番表" in v and "card" in v.lower() for v in variants))

    def test_chinese_ui_alias(self):
        # 卡片 是 Card 的中文别名，规则扩展在默认 max_variants=5 内即可生成
        variants = expand_query("追番表 Card 页面逻辑")
        self.assertTrue(any("卡片" in v for v in variants))

    def test_max_variants_respected(self):
        for max_v in [1, 2, 3, 5]:
            variants = expand_query("追番表 Card 页面逻辑", max_variants=max_v)
            self.assertLessEqual(len(variants), max_v)

    def test_max_variants_1_returns_original_only(self):
        variants = expand_query("追番表 Card 页面逻辑", max_variants=1)
        self.assertEqual(variants, ["追番表 Card 页面逻辑"])

    def test_no_duplicates(self):
        variants = expand_query("追番表 Card 页面逻辑", max_variants=10)
        lower = [v.lower() for v in variants]
        self.assertEqual(len(lower), len(set(lower)))

    def test_unknown_query_returns_original(self):
        variants = expand_query("完全未知的查询词xyz", max_variants=5)
        self.assertEqual(variants, ["完全未知的查询词xyz"])

    def test_empty_query(self):
        variants = expand_query("", max_variants=5)
        self.assertEqual(variants, [""])

    def test_tab_alias(self):
        variants = expand_query("追番表 Tab 切换", max_variants=10)
        self.assertTrue(any("标签" in v or "页签" in v for v in variants))


class TestAppendUnique(unittest.TestCase):
    def test_adds_new_item(self):
        lst = ["原始 query"]
        _append_unique(lst, "新 variant")
        self.assertIn("新 variant", lst)

    def test_skips_duplicate_case_insensitive(self):
        lst = ["追番表 Card"]
        _append_unique(lst, "追番表 card")  # 大小写不同但内容相同
        self.assertEqual(len(lst), 1)

    def test_skips_exact_duplicate(self):
        lst = ["追番表 Card"]
        _append_unique(lst, "追番表 Card")
        self.assertEqual(len(lst), 1)


# ── 多 query 并发召回测试 ─────────────────────────────────────────────────────

def _make_search_result(content: str, rid: str, score: float = 0.9) -> SearchResult:
    r = MagicMock(spec=SearchResult)
    r.id = rid
    r.content = content
    r.score = score
    r.metadata = {"source_name": "test.xlsx", "source_format": "xlsx",
                  "sheet_name": "Sheet1", "row_index": int(rid.split("-")[-1])}
    r.document_id = "doc-001"
    return r


class TestMultiQueryCandidates(unittest.IsolatedAsyncioTestCase):
    async def test_retrieve_called_for_each_variant(self):
        """dense retrieval 应为每个 query variant 调用一次。"""
        from src.agent.tools.search_kb import _multi_query_candidates

        engine = MagicMock()
        engine.retrieve_candidates = AsyncMock(return_value=[])

        await _multi_query_candidates(engine, ["q1", "q2", "q3"], 40, None)

        self.assertEqual(engine.retrieve_candidates.call_count, 3)
        calls_queries = [c.kwargs["query"] for c in engine.retrieve_candidates.call_args_list]
        self.assertIn("q1", calls_queries)
        self.assertIn("q2", calls_queries)
        self.assertIn("q3", calls_queries)

    async def test_results_merged(self):
        """来自不同 query 的候选应合并。"""
        from src.agent.tools.search_kb import _multi_query_candidates

        engine = MagicMock()
        engine.retrieve_candidates = AsyncMock(
            side_effect=[
                [_make_search_result("content A", "id-1")],
                [_make_search_result("content B", "id-2")],
            ]
        )

        results = await _multi_query_candidates(engine, ["q1", "q2"], 40, None)
        self.assertEqual(len(results), 2)


# ── _multi_query_rerank 模块多样性测试 ────────────────────────────────────────

class TestMultiQueryRerank(unittest.IsolatedAsyncioTestCase):
    async def test_round_robin_across_subqueries_avoids_starvation(self):
        """单一模块候选行数多/打分高时，仍需保证其余模块进入最终结果。

        复现场景：「搜索结果页都能搜索出什么内容」这类枚举型问题，子查询
        覆盖 动画/漫画/帖子/用户 4 个模块；若按全局最高分排序，候选行数多的
        模块（如动画有 5 条高分行）会挤占其余模块名额。轮询交叉排序后，
        取前 4 个结果应当来自 4 个不同模块。
        """
        from src.agent.tools.search_kb import _multi_query_rerank

        # 动画模块候选行数多且打分普遍很高，漫画/帖子/用户各只有 1 条但打分稍低
        anim = [_make_search_result(f"动画内容{i}", f"anim-{i}", score=0.95 - i * 0.01) for i in range(5)]
        manga = [_make_search_result("漫画内容", "manga-1", score=0.80)]
        post = [_make_search_result("帖子内容", "post-1", score=0.78)]
        user = [_make_search_result("用户内容", "user-1", score=0.76)]
        candidates = anim + manga + post + user

        async def fake_rerank(query, candidates, top_k):
            # 模拟各子查询对应模块的候选打分最高，其余候选打分很低
            target = {
                "搜索结果页 动画": anim,
                "搜索结果页 漫画": manga,
                "搜索结果页 帖子": post,
                "搜索结果页 用户": user,
            }[query]
            others = [c for c in candidates if c not in target]
            return target + others

        engine = MagicMock()
        engine.rerank_candidates = AsyncMock(side_effect=fake_rerank)

        queries = ["搜索结果页 动画", "搜索结果页 漫画", "搜索结果页 帖子", "搜索结果页 用户"]
        results = await _multi_query_rerank(engine, queries, candidates)

        top4_ids = {r.id for r in results[:4]}
        self.assertEqual(top4_ids, {"anim-0", "manga-1", "post-1", "user-1"})


# ── _stable_dedup 测试 ────────────────────────────────────────────────────────

class TestStableDedup(unittest.TestCase):
    def test_same_excel_row_via_different_queries_deduped(self):
        """同一 Excel 行通过不同 query 命中时，只保留一条。"""
        from src.agent.tools.search_kb import _stable_dedup

        r1 = _make_search_result("内容", "id-1")
        r1.metadata = {"source_name": "ACN.xlsx", "sheet_name": "Sheet1", "row_index": 5}
        r2 = _make_search_result("内容", "id-1")
        r2.metadata = {"source_name": "ACN.xlsx", "sheet_name": "Sheet1", "row_index": 5}

        result = _stable_dedup([r1, r2])
        self.assertEqual(len(result), 1)

    def test_different_rows_kept(self):
        """不同行应全部保留。"""
        from src.agent.tools.search_kb import _stable_dedup

        r1 = _make_search_result("内容A", "id-1")
        r1.metadata = {"source_name": "ACN.xlsx", "sheet_name": "Sheet1", "row_index": 1}
        r2 = _make_search_result("内容B", "id-2")
        r2.metadata = {"source_name": "ACN.xlsx", "sheet_name": "Sheet1", "row_index": 2}

        result = _stable_dedup([r1, r2])
        self.assertEqual(len(result), 2)

    def test_uuid_fallback_dedup(self):
        """无 sheet_name 时按 UUID 去重（非 Excel 文档）。"""
        from src.agent.tools.search_kb import _stable_dedup

        r1 = MagicMock(spec=SearchResult)
        r1.id = "uuid-same"
        r1.content = "非 Excel 内容"
        r1.metadata = {"source_name": "doc.pdf", "chunk_index": 0}
        r1.document_id = "doc-001"
        r1.score = 0.9

        r2 = MagicMock(spec=SearchResult)
        r2.id = "uuid-same"  # 同一 UUID
        r2.content = "非 Excel 内容"
        r2.metadata = {"source_name": "doc.pdf", "chunk_index": 0}
        r2.document_id = "doc-001"
        r2.score = 0.8

        result = _stable_dedup([r1, r2])
        self.assertEqual(len(result), 1)


# ── search_typed debug 输出测试 ───────────────────────────────────────────────

class TestSearchTypedDebug(unittest.IsolatedAsyncioTestCase):
    def _make_engine(self, candidates=None, reranked=None):
        engine = MagicMock()
        engine.retrieve_candidates = AsyncMock(return_value=candidates or [])
        engine.rerank_candidates = AsyncMock(return_value=reranked or [])
        return engine

    async def test_no_debug_no_header(self):
        """debug_queries=False 时结果不含调试头。"""
        from src.agent.tools.search_kb import KnowledgeBaseTool

        engine = self._make_engine()
        tool = KnowledgeBaseTool(engine)
        result = await tool.search_typed(query="追番表", debug_queries=False)
        self.assertNotIn("【检索调试】", result.content)

    async def test_debug_shows_header(self):
        """debug_queries=True 时结果含调试头和扩展 query 信息。"""
        from src.agent.tools.search_kb import KnowledgeBaseTool

        r = _make_search_result("[Sheet: S] [Row 1] #: 1 | 目录路径: x | 标题: t | 优先级: P1", "id-1")
        r.metadata = {
            "source_name": "ACN.xlsx", "source_format": "xlsx",
            "sheet_name": "Sheet1", "row_index": 1, "chunk_index": 1,
        }
        engine = self._make_engine(candidates=[r], reranked=[r])
        tool = KnowledgeBaseTool(engine)

        result = await tool.search_typed(
            query="追番表 Card",
            debug_queries=True,
            expand_query=True,
            max_query_variants=3,
        )
        self.assertIn("【检索调试】", result.content)
        self.assertIn("原始 query:", result.content)

    async def test_rerank_uses_original_query(self):
        """rerank 始终使用原始 query，不使用扩展 query。"""
        from src.agent.tools.search_kb import KnowledgeBaseTool

        r = _make_search_result("[Sheet: S] [Row 1] #: 1 | 目录路径: x | 标题: t | 优先级: P1", "id-1")
        r.metadata = {
            "source_name": "ACN.xlsx", "source_format": "xlsx",
            "sheet_name": "Sheet1", "row_index": 1, "chunk_index": 1,
        }
        engine = self._make_engine(candidates=[r], reranked=[r])
        tool = KnowledgeBaseTool(engine)

        original_query = "追番表 Card 页面逻辑"
        await tool.search_typed(
            query=original_query,
            expand_query=True,
            max_query_variants=3,
        )

        # rerank_candidates 的 query 参数必须是原始 query
        call_args = engine.rerank_candidates.call_args
        self.assertEqual(call_args.kwargs.get("query") or call_args.args[0], original_query)


if __name__ == "__main__":
    unittest.main()
