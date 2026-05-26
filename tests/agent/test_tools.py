"""Tests for individual agent tools."""

import pytest

from src.agent.base_tool import BaseTool
from src.agent.tool_registry import ToolRegistry
from src.agent.tools.web_search import WebSearchTool


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
        class ATool(BaseTool):
            @property
            def name(self): return "test_tool"
            @property
            def description(self): return "test description"
            @property
            def parameters(self): return {}
            async def execute(self, **kwargs): return "ok"

        reg = ToolRegistry([ATool()])
        defs = reg.definitions()
        assert len(defs) == 1
        assert defs[0]["name"] == "test_tool"
        assert "parameters" in defs[0]



class TestWebSearchTool:
    async def test_web_search_returns_fallback(self):
        """Web search should gracefully handle being called without network."""
        tool = WebSearchTool()
        result = await tool.execute(query="Python")
        assert isinstance(result, str)
        assert len(result) > 0
