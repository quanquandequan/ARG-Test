"""kb_compressor 单元测试。"""

import unittest
from unittest.mock import MagicMock, patch

from src.agent.tools.kb_compressor import (
    _compress_structured,
    _compress_text,
    _is_structured_row,
    _parse_kv_pairs,
    compress_results,
    extract_keywords,
)
from src.vectordb.base import SearchResult


def _make_result(content: str, rid: str = "id-001") -> SearchResult:
    """创建最小 SearchResult 用于测试。"""
    r = MagicMock(spec=SearchResult)
    r.id = rid
    r.content = content
    r.metadata = {}
    r.score = 0.9
    r.document_id = "doc-001"
    return r


# ── _parse_kv_pairs ──────────────────────────────────────────────────────────

class TestParseKvPairs(unittest.TestCase):
    def test_basic(self):
        pairs = _parse_kv_pairs("#: 1 | 标题: 登录功能 | 优先级: P1")
        self.assertEqual(pairs, [("#", "1"), ("标题", "登录功能"), ("优先级", "P1")])

    def test_value_with_colon(self):
        # value 中含冒号时只在第一个冒号处切分
        pairs = _parse_kv_pairs("操作步骤: 打开 URL: http://example.com | 预期结果: 正常")
        self.assertEqual(pairs[0], ("操作步骤", "打开 URL: http://example.com"))
        self.assertEqual(pairs[1], ("预期结果", "正常"))

    def test_no_colon_segment(self):
        pairs = _parse_kv_pairs("standalone")
        self.assertEqual(pairs, [("standalone", "")])


# ── _is_structured_row ───────────────────────────────────────────────────────

class TestIsStructuredRow(unittest.TestCase):
    def test_detects_excel_row(self):
        self.assertTrue(
            _is_structured_row("[Sheet: ACN用例] [Row 5] #: 1 | 标题: xxx")
        )

    def test_rejects_plain_text(self):
        self.assertFalse(_is_structured_row("这是普通文本，没有 Sheet 前缀。"))


# ── _compress_structured ─────────────────────────────────────────────────────

_HEADER = "[Sheet: ACN用例] [Row 10] "
_CASE_BODY = (
    "#: 1 | "
    "目录路径: 追番表/Card | "
    "标题: 追番Card展示正确 | "
    "优先级: P1 | "
    "操作步骤: 1.打开追番表 2.查看Card区域 | "
    "预期结果: Card正确显示封面和标题"
)
_CASE_CONTENT = _HEADER + _CASE_BODY


class TestCompressStructured(unittest.TestCase):
    def test_anchor_fields_always_present(self):
        """无论关键词如何，锚定字段（#/目录路径/标题/优先级）应始终保留。"""
        kw = {"completely_unrelated_keyword_xyz"}
        # 无关键词时应 fallback 返回原文前 500 字符（因为 kept 为空）
        result = _compress_structured(_CASE_CONTENT, kw)
        # fallback: 原文前 500 字符，包含锚定字段
        self.assertIn("目录路径", result)

    def test_keyword_match_keeps_field(self):
        """关键词命中字段时，该字段应出现在压缩结果中。"""
        kw = {"追番"}
        result = _compress_structured(_CASE_CONTENT, kw)
        self.assertIn("目录路径", result)
        self.assertIn("追番", result)

    def test_title_hit_expands_steps(self):
        """标题/路径命中时，操作步骤和预期结果应被展开到压缩结果中。"""
        kw = {"追番card展示正确"}  # 命中标题字段 value
        result = _compress_structured(_CASE_CONTENT, kw)
        self.assertIn("操作步骤", result)
        self.assertIn("预期结果", result)

    def test_hit_field_summary_line_added(self):
        """命中时应在结果首行添加 [命中字段: ...] 摘要行。"""
        kw = {"追番"}
        result = _compress_structured(_CASE_CONTENT, kw)
        self.assertIn("[命中字段:", result)

    def test_bug_row_returned_as_is(self):
        """字段数 ≤ 3 的 Bug 行应直接返回原文，不压缩。"""
        bug = "[Sheet: Bug列表] [Row 3] Bug Key: ACNBUG-001 | 摘要: 追番Card封面不显示"
        kw = {"追番"}
        result = _compress_structured(bug, kw)
        self.assertEqual(result, bug)

    def test_no_keyword_match_keeps_only_anchors(self):
        """关键词完全不命中时，只保留锚定字段，不展开步骤/预期。"""
        kw = {"completely_unrelated_xyz"}
        result = _compress_structured(_CASE_CONTENT, kw)
        # 锚定字段仍然保留
        self.assertIn("目录路径", result)
        self.assertIn("标题", result)
        self.assertIn("优先级", result)
        # 步骤/预期未展开（标题未命中）
        self.assertNotIn("操作步骤", result)
        self.assertNotIn("预期结果", result)
        # 无命中字段摘要行
        self.assertNotIn("[命中字段:", result)

    def test_expand_max_chars_truncation(self):
        """展开字段超过 300 字符时应被截断。"""
        long_steps = "步骤" * 200  # 400 字符
        long_case = _HEADER + f"#: 1 | 目录路径: 追番表 | 标题: T | 优先级: P1 | 操作步骤: {long_steps} | 预期结果: 正常"
        kw = {"追番"}
        result = _compress_structured(long_case, kw)
        # 操作步骤被展开但截断到 300 字符
        self.assertIn("操作步骤", result)
        self.assertNotIn(long_steps, result)  # 原完整 400 字符字符串不应出现


# ── _compress_text ────────────────────────────────────────────────────────────

class TestCompressText(unittest.TestCase):
    _TEXT = (
        "这是关于追番功能的说明。\n"
        "追番表用于用户收藏动漫，支持多种视图模式。\n"
        "播放器支持倍速播放。\n"
        "追番Card在列表视图中展示封面和标题。\n"
        "系统支持搜索功能。"
    )

    def test_returns_matching_sentences(self):
        kw = {"追番"}
        result = _compress_text(self._TEXT, kw)
        self.assertIn("追番", result)
        # 不含无关句子
        self.assertNotIn("播放器", result)

    def test_top_n_limit(self):
        kw = {"追番"}
        result = _compress_text(self._TEXT, kw)
        # 最多 3 条，用 " | " 分隔，所以 " | " 最多出现 2 次
        self.assertLessEqual(result.count(" | "), 2)

    def test_no_match_fallback(self):
        kw = {"completely_unrelated_xyz"}
        result = _compress_text(self._TEXT, kw)
        self.assertEqual(result, self._TEXT[:500])

    def test_order_preserved(self):
        """命中片段按原文顺序输出，不按分数倒序。"""
        kw = {"追番"}
        result = _compress_text(self._TEXT, kw)
        parts = result.split(" | ")
        # 找到每个片段在原文中的位置，确认单调递增
        positions = [self._TEXT.find(p) for p in parts]
        self.assertEqual(positions, sorted(positions))


# ── extract_keywords ──────────────────────────────────────────────────────────

class TestExtractKeywords(unittest.TestCase):
    def test_stopwords_filtered(self):
        with patch("jieba.lcut", return_value=["追番", "的", "Card", "展示"]):
            kw = extract_keywords("追番的Card展示")
        self.assertNotIn("的", kw)
        self.assertIn("追番", kw)
        self.assertIn("card", kw)  # 转小写

    def test_short_tokens_filtered(self):
        with patch("jieba.lcut", return_value=["我", "a", "追番"]):
            kw = extract_keywords("追番")
        self.assertNotIn("我", kw)
        self.assertNotIn("a", kw)

    def test_original_phrase_kept(self):
        with patch("jieba.lcut", return_value=["追番"]):
            kw = extract_keywords("追番card展示")
        # 原 query 转小写后应作为短语保留
        self.assertIn("追番card展示", kw)


# ── compress_results ──────────────────────────────────────────────────────────

class TestCompressResults(unittest.TestCase):
    def test_structured_row_dispatched(self):
        content = "[Sheet: ACN用例] [Row 5] #: 1 | 目录路径: 追番 | 标题: T | 优先级: P1 | 操作步骤: step | 预期结果: ok"
        result = _make_result(content, "uid-1")
        with patch("jieba.lcut", return_value=["追番"]):
            compressed = compress_results([result], "追番")
        self.assertIn("uid-1", compressed)
        self.assertIn("追番", compressed["uid-1"])

    def test_text_row_dispatched(self):
        content = "这是追番功能说明。\n追番表支持多种视图。\n系统支持搜索。"
        result = _make_result(content, "uid-2")
        with patch("jieba.lcut", return_value=["追番"]):
            compressed = compress_results([result], "追番")
        self.assertIn("uid-2", compressed)
        self.assertIn("追番", compressed["uid-2"])

    def test_missing_id_skipped(self):
        result = _make_result("some content", rid="")
        result.id = None
        with patch("jieba.lcut", return_value=["追番"]):
            compressed = compress_results([result], "追番")
        self.assertEqual(compressed, {})

    def test_exception_silently_skipped(self):
        """压缩过程抛异常时，该 result 不写入 compressed 字典，其他正常处理。"""
        bad = _make_result("good content", "uid-good")
        with (
            patch("jieba.lcut", return_value=["追番"]),
            patch(
                "src.agent.tools.kb_compressor._compress_text",
                side_effect=RuntimeError("压缩异常"),
            ),
        ):
            compressed = compress_results([bad], "追番")
        # 出错时跳过，字典为空（不崩溃）
        self.assertEqual(compressed, {})


if __name__ == "__main__":
    unittest.main()
