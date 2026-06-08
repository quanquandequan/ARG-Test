"""工具注册表：注册、查找并列出工具 schema。"""

from src.agent.base_tool import BaseTool


class ToolRegistry:
    def __init__(self, tools: list[BaseTool] | None = None):
        self._tools: dict[str, BaseTool] = {}
        for t in (tools or []):
            self.register(t)

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"Tool '{name}' not found. Available: {list(self._tools.keys())}")
        return tool

    def definitions(self) -> list[dict]:
        """返回供 LLM function-calling 使用的工具 schema。"""
        return [t.to_tool_schema() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools.keys())
