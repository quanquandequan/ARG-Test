"""执行阶段的请求与结果模型。"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.domain.artifacts import ArtifactRecord


@dataclass(slots=True)
class ScenarioExecutionRequest:
    """执行单条自动化场景所需的输入。"""

    automation_json_path: str
    case_id: str = ""
    case_title: str = ""
    app_package: str = ""
    output_dir: str = ""
    request_id: str = ""


@dataclass(slots=True)
class ExecutionStepResult:
    """执行过程中的单步结果。"""

    stage: str
    name: str
    success: bool
    detail: str


@dataclass(slots=True)
class ScenarioExecutionResult:
    """单条自动化场景执行结果。"""

    case_id: str
    title: str
    module: str
    status: str
    steps: list[ExecutionStepResult] = field(default_factory=list)
    report_artifact: ArtifactRecord | None = None
    screenshot_artifact: ArtifactRecord | None = None
    failure_reason: str = ""
