"""QueryRewriter 单元测试（使用 Mock LLM，不依赖网络或 API key）。"""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from src.llm.types import ChatResponse


def _make_llm(response_text: str) -> MagicMock:
    """创建返回固定文本的 mock LLM。"""
    llm = MagicMock()
    llm.generate_chat = AsyncMock(return_value=ChatResponse(
        content=response_text,
        model="mock-model",
    ))
    return llm


def _make_rewriter(response_text: str, timeout: float = 5.0):
    """创建带 mock LLM 的 QueryRewriter（绕过 prompt 文件系统依赖）。"""
    from src.retriever.query_rewriter import QueryRewriter

    with patch(
        "src.retriever.query_rewriter.require_prompt_fields",
        return_value={"query_rewriter_prompt": "测试 prompt，请生成 {max_variants} 个变体"},
    ):
        return QueryRewriter(_make_llm(response_text), timeout_seconds=timeout)


class TestQueryRewriter(unittest.IsolatedAsyncioTestCase):

    async def test_original_query_always_first(self):
        """原始 query 始终是返回列表的第一个元素。"""
        rewriter = _make_rewriter("搜索结果页 动画\n搜索结果页 漫画\n搜索结果页 帖子")
        variants = await rewriter.rewrite("搜索结果页", max_variants=4)
        self.assertEqual(variants[0], "搜索结果页")

    async def test_llm_variants_appended(self):
        """LLM 输出的变体被追加到原始 query 之后。"""
        rewriter = _make_rewriter("搜索结果页 动画\n搜索结果页 漫画")
        variants = await rewriter.rewrite("搜索结果页", max_variants=5)
        self.assertIn("搜索结果页 动画", variants)
        self.assertIn("搜索结果页 漫画", variants)

    async def test_max_variants_respected(self):
        """输出总数不超过 max_variants。"""
        rewriter = _make_rewriter("A\nB\nC\nD\nE\nF\nG")
        for max_v in [1, 2, 3, 5]:
            variants = await rewriter.rewrite("query", max_variants=max_v)
            self.assertLessEqual(len(variants), max_v, f"max_variants={max_v}")

    async def test_max_variants_1_returns_original_only(self):
        """max_variants=1 只返回原始 query，不调用 LLM。"""
        rewriter = _make_rewriter("A\nB\nC")
        variants = await rewriter.rewrite("追番表", max_variants=1)
        self.assertEqual(variants, ["追番表"])

    async def test_no_duplicates(self):
        """输出中无重复条目（忽略大小写）。"""
        rewriter = _make_rewriter("追番表 动画\n追番表 动画\n追番表 漫画")
        variants = await rewriter.rewrite("追番表", max_variants=5)
        lower = [v.lower() for v in variants]
        self.assertEqual(len(lower), len(set(lower)))

    async def test_original_not_duplicated_by_llm_output(self):
        """LLM 重复输出原始 query 时，结果中原始 query 只出现一次。"""
        rewriter = _make_rewriter("搜索结果页\n搜索结果页 动画")
        variants = await rewriter.rewrite("搜索结果页", max_variants=4)
        self.assertEqual(variants.count("搜索结果页"), 1)

    async def test_timeout_returns_original_only(self):
        """LLM 超时时静默返回 [query]。"""
        from src.retriever.query_rewriter import QueryRewriter

        async def _slow(*args, **kwargs):
            await asyncio.sleep(100)

        llm = MagicMock()
        llm.generate_chat = _slow

        with patch(
            "src.retriever.query_rewriter.require_prompt_fields",
            return_value={"query_rewriter_prompt": "test {max_variants}"},
        ):
            rewriter = QueryRewriter(llm, timeout_seconds=0.01)

        variants = await rewriter.rewrite("超时测试", max_variants=5)
        self.assertEqual(variants, ["超时测试"])

    async def test_llm_error_returns_original_only(self):
        """LLM 抛异常时静默返回 [query]。"""
        from src.retriever.query_rewriter import QueryRewriter

        llm = MagicMock()
        llm.generate_chat = AsyncMock(side_effect=Exception("API 调用失败"))

        with patch(
            "src.retriever.query_rewriter.require_prompt_fields",
            return_value={"query_rewriter_prompt": "test {max_variants}"},
        ):
            rewriter = QueryRewriter(llm)

        variants = await rewriter.rewrite("错误测试", max_variants=5)
        self.assertEqual(variants, ["错误测试"])

    async def test_transient_error_retried_then_succeeds(self):
        """首次报错/超时为偶发故障，重试一次后成功则使用 LLM 变体，而非直接回退。"""
        from src.retriever.query_rewriter import QueryRewriter

        llm = MagicMock()
        llm.generate_chat = AsyncMock(
            side_effect=[
                Exception("网络抖动"),
                ChatResponse(content="搜索结果页 动画\n搜索结果页 漫画", model="mock"),
            ]
        )

        with patch(
            "src.retriever.query_rewriter.require_prompt_fields",
            return_value={"query_rewriter_prompt": "test {max_variants}"},
        ):
            rewriter = QueryRewriter(llm)

        variants = await rewriter.rewrite("搜索结果页", max_variants=4)
        self.assertIn("搜索结果页 动画", variants)
        self.assertEqual(llm.generate_chat.call_count, 2)

    async def test_empty_llm_response_returns_original(self):
        """LLM 返回空字符串时，只返回原始 query。"""
        rewriter = _make_rewriter("   \n\n  ")
        variants = await rewriter.rewrite("追番表", max_variants=5)
        self.assertEqual(variants, ["追番表"])

    async def test_empty_query_returns_empty(self):
        """空 query 原样返回。"""
        rewriter = _make_rewriter("some output")
        variants = await rewriter.rewrite("", max_variants=5)
        self.assertEqual(variants, [""])

    async def test_numbered_prefixes_stripped(self):
        """LLM 输出中的序号、破折号等前缀被去除。"""
        rewriter = _make_rewriter("1. 搜索结果页 动画\n2. 搜索结果页 漫画\n- 搜索结果页 帖子")
        variants = await rewriter.rewrite("搜索结果页", max_variants=5)
        self.assertIn("搜索结果页 动画", variants)
        self.assertFalse(any(v.startswith("1.") or v.startswith("-") for v in variants))

    async def test_llm_receives_original_query_as_user_message(self):
        """LLM 调用时，user message 的 content 为原始 query。"""
        from src.retriever.query_rewriter import QueryRewriter

        llm = MagicMock()
        llm.generate_chat = AsyncMock(return_value=ChatResponse(content="A\nB", model="mock"))

        with patch(
            "src.retriever.query_rewriter.require_prompt_fields",
            return_value={"query_rewriter_prompt": "test {max_variants}"},
        ):
            rewriter = QueryRewriter(llm)

        await rewriter.rewrite("追番表 Card", max_variants=3)
        call_messages = llm.generate_chat.call_args.kwargs["messages"]
        user_msg = next(m for m in call_messages if m.role == "user")
        self.assertEqual(user_msg.content, "追番表 Card")

    async def test_llm_temperature_is_zero(self):
        """枚举型任务使用 temperature=0 保证确定性。"""
        from src.retriever.query_rewriter import QueryRewriter

        llm = MagicMock()
        llm.generate_chat = AsyncMock(return_value=ChatResponse(content="A", model="mock"))

        with patch(
            "src.retriever.query_rewriter.require_prompt_fields",
            return_value={"query_rewriter_prompt": "test {max_variants}"},
        ):
            rewriter = QueryRewriter(llm)

        await rewriter.rewrite("query", max_variants=3)
        kwargs = llm.generate_chat.call_args.kwargs
        self.assertEqual(kwargs.get("temperature"), 0.0)


class TestParseVariants(unittest.TestCase):

    def test_plain_lines(self):
        from src.retriever.query_rewriter import _parse_variants

        result = _parse_variants("搜索结果页 动画\n搜索结果页 漫画\n搜索结果页 帖子", 5)
        self.assertEqual(result, ["搜索结果页 动画", "搜索结果页 漫画", "搜索结果页 帖子"])

    def test_numbered_lines(self):
        from src.retriever.query_rewriter import _parse_variants

        result = _parse_variants("1. 动画\n2. 漫画\n3. 帖子", 5)
        self.assertEqual(result, ["动画", "漫画", "帖子"])

    def test_dash_prefix(self):
        from src.retriever.query_rewriter import _parse_variants

        result = _parse_variants("- 动画\n- 漫画", 5)
        self.assertEqual(result, ["动画", "漫画"])

    def test_max_count_respected(self):
        from src.retriever.query_rewriter import _parse_variants

        result = _parse_variants("A\nB\nC\nD\nE", 3)
        self.assertEqual(len(result), 3)

    def test_blank_lines_skipped(self):
        from src.retriever.query_rewriter import _parse_variants

        result = _parse_variants("A\n\n\nB\n   \nC", 5)
        self.assertEqual(result, ["A", "B", "C"])


if __name__ == "__main__":
    unittest.main()
