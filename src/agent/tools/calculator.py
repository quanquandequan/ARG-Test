"""Calculator tool — safe arithmetic evaluation."""

import math
import re

from src.agent.base_tool import BaseTool


class CalculatorTool(BaseTool):
    """Evaluate mathematical expressions safely.

    Only allows numbers, basic operators, and a whitelist of math functions.
    """

    _ALLOWED = re.compile(r"^[\d\s+\-*/%().,eEa-zA-Z_]+$")
    _ALLOWED_START = re.compile(r"^[\d+\-(.a-zA-Z]")

    def __init__(self):
        self._safe_globals = {
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "pow": pow,
            "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
            "tan": math.tan, "log": math.log, "log10": math.log10,
            "pi": math.pi, "e": math.e,
        }

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return (
            "执行数值计算。支持基本算术运算和常用数学函数 "
            "(abs, round, sqrt, sin, cos, log, pi, e 等)。"
            "当需要精确数值结果时使用此工具。"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "要计算的数学表达式，例如: '2 + 3 * 4' 或 'sqrt(144)'",
                },
            },
            "required": ["expression"],
        }

    async def execute(self, expression: str = "", **kwargs) -> str:
        expr = expression.strip()
        if not expr:
            return "错误: 表达式为空"

        if not self._ALLOWED_START.match(expr):
            return f"错误: 不允许的表达式开头: {expr[:20]}"

        if not self._ALLOWED.match(expr):
            return f"错误: 表达式包含不允许的字符: {expr[:50]}"

        # Block known-dangerous constructs
        dangerous = ("__", "import", "exec", "eval", "compile", "open", "file")
        for kw in dangerous:
            if kw in expr.lower():
                return f"错误: 表达式包含不允许的关键字: {kw}"

        try:
            result = eval(expr, {"__builtins__": {}}, self._safe_globals)
            return f"计算结果: {result}"
        except Exception as e:
            return f"计算错误: {e}"
