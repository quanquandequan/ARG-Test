"""需求解析工作流节点。"""

from __future__ import annotations

from src.application.workflow_nodes.base import WorkflowContext, WorkflowNode
from src.core.logging import get_logger
from src.llm.base import BaseLLM
from src.services.requirement_ir_builder import RequirementIRBuilder

logger = get_logger(__name__)


class RequirementParserNode(WorkflowNode):
    """构建供下游测试设计节点消费的 RequirementIR。"""

    def __init__(
        self,
        llm: BaseLLM,
        builder: RequirementIRBuilder | None = None,
    ):
        self._builder = builder or RequirementIRBuilder(llm=llm)

    async def execute(self, context: WorkflowContext) -> WorkflowContext:
        requirement_ir = await self._builder.build(
            requirement=context.requirement_text,
            module=context.module,
            kb_context=context.kb_samples,
        )
        if requirement_ir is None:
            logger.warning("workflow_requirement_ir_parse_failed", module=context.module)
            raise ValueError("LLM 未能生成有效的 RequirementIR，请检查需求文档内容后重试。")
        context.requirement_ir = requirement_ir
        return context
