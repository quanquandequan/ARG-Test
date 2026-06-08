"""Artifact builder workflow node."""

from __future__ import annotations

from src.application.workflow_nodes.base import WorkflowContext, WorkflowNode
from src.domain.artifacts.test_design_artifact import TestDesignArtifact


class ArtifactBuilderNode(WorkflowNode):
    """Build the unified test design artifact consumed by exporters."""

    async def execute(self, context: WorkflowContext) -> WorkflowContext:
        if context.requirement_ir is None:
            raise ValueError("RequirementIR is required before building artifact.")
        context.artifact = TestDesignArtifact(
            module=context.module,
            generation_mode=context.generation_mode,
            requirement_ir=context.requirement_ir,
            test_points=context.test_points,
            scenarios=context.scenarios,
            test_cases=context.test_cases,
            metadata={
                "kb_samples_used": bool(context.kb_samples.strip()),
                "source_length": len(context.requirement_text),
            },
        )
        return context
