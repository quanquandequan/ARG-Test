"""Agent 可调用工具的抽象基类。"""

from abc import ABC, abstractmethod

from src.agent.tool_result import ToolExecutionResult

FINAL_ANSWER_LLM = "llm_summarize"
FINAL_ANSWER_PASSTHROUGH = "passthrough_final_answer"


class BaseTool(ABC):
    """Agent 可通过 function calling 调用的工具。

    子类必须实现 ``name``、``description``、``parameters`` 和 ``execute``。
    暴露给 LLM 的 ``description`` 与 ``name`` 可在运行时覆盖
    （例如来自 YAML 配置），无需修改工具源码；构造后调用
    ``override_description()`` 即可。
    """

    # 运行时覆盖项：当 YAML 配置提供该值时由 tool_factory 设置。
    _description_override: str | None = None
    final_answer_mode: str = FINAL_ANSWER_LLM

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """展示给 LLM 的默认描述。

        可通过 ``override_description(text)`` 在运行时覆盖，
        使用 YAML 配置值替代硬编码默认值。
        """

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """工具参数的 JSON Schema。"""

    @abstractmethod
    async def execute(self, **kwargs) -> str:
        """执行工具并返回字符串结果。"""
        ...

    async def execute_typed(self, **kwargs) -> ToolExecutionResult:
        """应用服务使用的结构化执行钩子。"""
        return ToolExecutionResult(content=await self.execute(**kwargs))

    # ── 运行时覆盖辅助方法 ───────────────────────────────────────────────

    def override_description(self, text: str) -> None:
        """用 YAML 配置值替换默认描述。

        LLM 会在 ``to_tool_schema()`` 中收到覆盖后的文本，
        因而可在不修改 Python 源码的情况下调优提示词。
        """
        self._description_override = text.strip() or None

    def effective_description(self) -> str:
        """返回 LLM 将看到的描述（优先使用覆盖值）。"""
        return self._description_override or self.description

    def to_tool_schema(self) -> dict:
        """以 function-calling schema 形式返回工具定义。"""
        return {
            "name": self.name,
            "description": self.effective_description(),
            "parameters": self.parameters,
        }
