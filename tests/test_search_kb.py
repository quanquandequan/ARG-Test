"""search_kb 知识来源路由单元测试。

覆盖点：叭嗒 App 主体功能查询应默认优先 Excel，只有小程序/插件类查询才
应该触发 XMind 优先；对比查询命中小程序/插件时仍需双来源各占一半。
"""

import unittest

from src.agent.tools.search_kb import (
    _is_bug_query,
    _is_comparison_query,
    _is_xmind_query,
    _select_source_aware_results,
)
from src.vectordb.base import SearchResult


def _make_result(content: str, rid: str, source_format: str, score: float = 0.5) -> SearchResult:
    """创建带来源元数据的最小 SearchResult 用于测试。"""
    return SearchResult(
        id=rid,
        document_id=rid,
        content=content,
        score=score,
        metadata={"source_format": source_format, "source_name": f"{rid}.{source_format}"},
    )


# ── 触发词判断 ────────────────────────────────────────────────────────────────

class TestXmindTriggerWords(unittest.TestCase):
    def test_app_core_function_query_is_not_xmind_mode(self):
        # 叭嗒 App 主体功能查询（不含小程序/插件）不应触发 XMind 优先，
        # 这类内容应默认优先 Excel（Excel 是当前版本最全、最新的测试用例）。
        self.assertFalse(_is_xmind_query("查询叭嗒的核心功能都有哪些"))
        self.assertFalse(_is_xmind_query("动画播放器都有什么严重的bug"))

    def test_mini_program_and_plugin_query_is_xmind_mode(self):
        # 小程序/插件场景 Excel 基本不覆盖或已过时，只有 XMind 有记录。
        self.assertTrue(_is_xmind_query("小程序有哪些功能"))
        self.assertTrue(_is_xmind_query("漫画插件的阅读器有什么功能"))

    def test_bug_and_comparison_triggers_unchanged(self):
        self.assertTrue(_is_bug_query("动画播放器都有什么严重的bug"))
        self.assertTrue(_is_comparison_query("漫画插件阅读器与叭嗒阅读器有哪些差异"))


# ── 来源选取：叭嗒 App 主体查询默认 Excel 优先 ─────────────────────────────────

class TestSelectSourceAwareResultsDefaultsToExcel(unittest.TestCase):
    def test_normal_mode_prioritizes_excel_over_xmind(self):
        ranked = [
            _make_result("xmind 功能点", "x1", "xmind", score=0.9),
            _make_result("excel 用例行", "e1", "xlsx", score=0.4),
        ]
        selected = _select_source_aware_results(
            ranked, top_k=1, bug_query_mode=False, xmind_query_mode=False, comparison_mode=False,
        )
        self.assertEqual([r.id for r in selected], ["e1"])

    def test_xmind_mode_reserves_excel_floor(self):
        excel = [_make_result(f"excel 用例 {i}", f"e{i}", "xlsx") for i in range(3)]
        xmind = [_make_result(f"xmind 条目 {i}", f"x{i}", "xmind") for i in range(10)]
        selected = _select_source_aware_results(
            xmind + excel, top_k=8,
            bug_query_mode=False, xmind_query_mode=True, comparison_mode=False,
        )
        selected_ids = {r.id for r in selected}
        # xmind 拿大多数名额，但 excel 仍应保留至少一部分（保底名额）
        self.assertTrue(any(rid.startswith("x") for rid in selected_ids))
        self.assertTrue(any(rid.startswith("e") for rid in selected_ids))

    def test_comparison_mode_splits_evenly(self):
        excel = [_make_result(f"excel 用例 {i}", f"e{i}", "xlsx") for i in range(5)]
        xmind = [_make_result(f"xmind 条目 {i}", f"x{i}", "xmind") for i in range(5)]
        selected = _select_source_aware_results(
            xmind + excel, top_k=4,
            bug_query_mode=False, xmind_query_mode=True, comparison_mode=True,
        )
        excel_count = sum(1 for r in selected if r.id.startswith("e"))
        xmind_count = sum(1 for r in selected if r.id.startswith("x"))
        self.assertEqual(excel_count, 2)
        self.assertEqual(xmind_count, 2)


if __name__ == "__main__":
    unittest.main()
