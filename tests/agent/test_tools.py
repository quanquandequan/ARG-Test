"""Tests for individual agent tools."""

import pytest

from src.agent.base_tool import BaseTool
from src.agent.tool_registry import ToolRegistry
from src.agent.tools.calculator import CalculatorTool
from src.agent.tools.reranker import RerankerTool
from src.agent.tools.search_kb import KnowledgeBaseTool
from src.agent.tools.web_search import WebSearchTool
from src.vectordb.base import SearchResult
from tests.fakes import FakeReranker


class TestToolRegistry:
    def test_register_and_get(self):
        class ATool(BaseTool):
            @property
            def name(self): return "a"
            @property
            def description(self): return "desc"
            @property
            def parameters(self): return {}
            async def execute(self, **kwargs): return "ok"

        reg = ToolRegistry([ATool()])
        assert "a" in reg.names()
        assert reg.get("a") is not None

    def test_duplicate_rejects(self):
        reg = ToolRegistry([])

        class ATool(BaseTool):
            @property
            def name(self): return "x"
            @property
            def description(self): return ""
            @property
            def parameters(self): return {}
            async def execute(self, **kwargs): return ""

        reg.register(ATool())
        with pytest.raises(ValueError):
            reg.register(ATool())

    def test_get_missing_raises(self):
        reg = ToolRegistry([])
        with pytest.raises(KeyError):
            reg.get("nonexistent")

    def test_definitions_format(self):
        reg = ToolRegistry([CalculatorTool()])
        defs = reg.definitions()
        assert len(defs) == 1
        assert defs[0]["name"] == "calculator"
        assert "parameters" in defs[0]


class TestCalculatorTool:
    async def test_basic_arithmetic(self):
        tool = CalculatorTool()
        result = await tool.execute(expression="2 + 3 * 4")
        assert "14" in result

    async def test_math_functions(self):
        tool = CalculatorTool()
        result = await tool.execute(expression="sqrt(144)")
        assert "12" in result

    async def test_blocked_keyword(self):
        tool = CalculatorTool()
        result = await tool.execute(expression="__import__('os')")
        assert "错误" in result

    async def test_invalid_expression(self):
        tool = CalculatorTool()
        result = await tool.execute(expression="foo()")
        assert "错误" in result


class TestRerankerTool:
    async def test_rerank_chunks(self):
        tool = RerankerTool(FakeReranker())
        result = await tool.execute(
            query="测试查询",
            chunks_text=["片段A", "片段B", "片段C"],
            top_k=2,
        )
        assert "重排序" in result or "片段" in result

    async def test_empty_chunks(self):
        tool = RerankerTool(FakeReranker())
        result = await tool.execute(query="测试", chunks_text=[], top_k=3)
        assert "没有" in result


class TestWebSearchTool:
    async def test_web_search_returns_fallback(self):
        """Web search should gracefully handle being called without network."""
        tool = WebSearchTool()
        result = await tool.execute(query="Python")
        assert isinstance(result, str)
        assert len(result) > 0
