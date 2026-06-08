"""需求分析共享上下文构建工具。"""

from __future__ import annotations

from src.agent.tools.search_kb import KnowledgeBaseTool
from src.retriever.retrieval_engine import RetrievalEngine


async def build_requirement_kb_context(
    retrieval_engine: RetrievalEngine,
    module: str,
    requirement_text: str,
    *,
    top_k: int = 5,
    final_k: int = 5,
) -> str:
    """根据模块和需求文本构建需求分析可复用的 KB 辅助上下文。"""
    query = f"{module} 测试用例".strip() if module.strip() else requirement_text[:60]
    display_k = final_k if final_k > 0 else top_k
    kb_result = await KnowledgeBaseTool(retrieval_engine).search_typed(
        query=query,
        top_k=display_k,
    )
    if kb_result.hit_count == 0:
        return ""

    return (
        "【历史知识库参考（辅助）】\n"
        "以下内容只用于识别历史功能、历史差异、回归风险和回测范围；"
        "不得作为当前需求事实来源，不得修改或补写 PRD 中没有描述的功能。\n\n"
        f"{kb_result.content}"
    )
