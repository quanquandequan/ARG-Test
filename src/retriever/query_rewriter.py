"""基于 LLM 的 Query 改写：把用户 query 扩展为多个检索变体。

替代硬编码的 _COMPOUND_EXPANSIONS / _SCENE_CONTEXT 业务词表，
通过 LLM 理解语义自动生成适合知识库检索的搜索变体。
超时或 LLM 报错时静默回退到 [query]（原始 query），不影响正常检索流程。
"""
from __future__ import annotations

import asyncio
import re

from src.core.logging import get_logger
from src.core.prompt_loader import require_prompt_fields
from src.llm.base import BaseLLM
from src.llm.types import Message

logger = get_logger(__name__)

# 枚举型任务，温度=0 保证确定性输出
_TEMPERATURE = 0.0
_MAX_TOKENS = 256


class QueryRewriter:
    """调用 LLM 把单个 query 改写为多个检索变体。"""

    def __init__(self, llm: BaseLLM, timeout_seconds: float = 5.0):
        self._llm = llm
        self._timeout = timeout_seconds
        # 预加载 prompt；启动时即可发现配置缺失
        prompt_data = require_prompt_fields("search_knowledge", ["query_rewriter_prompt"])
        self._system_template: str = prompt_data["query_rewriter_prompt"]

    async def rewrite(self, query: str, max_variants: int = 5) -> list[str]:
        """返回 query 变体列表，第一项始终为原始 query。

        LLM 生成 max_variants-1 个变体，再加上原始 query 共 max_variants 项。
        超时或报错时静默返回 [query]，由调用方决定是否叠加规则扩展。
        """
        query = query.strip()
        if not query or max_variants <= 1:
            return [query]
        try:
            llm_variants = await asyncio.wait_for(
                self._call_llm(query, max_variants - 1),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("query_rewriter_timeout", query=query, timeout=self._timeout)
            return [query]
        except Exception as exc:
            logger.warning("query_rewriter_error", query=query, error=str(exc))
            return [query]

        # 原始 query 始终排第一，LLM 变体去重后依次追加
        result: list[str] = [query]
        seen = {query.lower()}
        for v in llm_variants:
            vl = v.lower()
            if vl and vl not in seen and len(result) < max_variants:
                seen.add(vl)
                result.append(v)
        return result

    async def _call_llm(self, query: str, num_variants: int) -> list[str]:
        """调用 LLM 并将逐行输出解析为变体列表。"""
        system_prompt = self._system_template.format(max_variants=num_variants)
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=query),
        ]
        response = await self._llm.generate_chat(
            messages=messages,
            temperature=_TEMPERATURE,
            max_tokens=_MAX_TOKENS,
        )
        return _parse_variants(response.content, num_variants)


def _parse_variants(text: str, max_count: int) -> list[str]:
    """将 LLM 输出的逐行文本解析为干净的变体列表。

    去除序号、列表符号等前缀（"1. " "- " "* " 等）。
    """
    variants: list[str] = []
    for line in text.strip().split("\n"):
        # 去除常见前缀：序号、点、破折号、星号、前导空白
        clean = re.sub(r"^[\d\.\-\*\s]+", "", line).strip()
        if clean and len(variants) < max_count:
            variants.append(clean)
    return variants
