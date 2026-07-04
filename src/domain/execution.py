"""执行阶段的请求与结果模型。"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.domain.artifacts import ArtifactRecord


@dataclass(slots=True)
class ScenarioExecutionRequest:
    """执行自动化场景所需的输入。

    ``case_id``/``case_title`` 用于单条执行（历史行为：都不填时默认执行
    ``cases[0]``）。``case_ids``/``max_cases``/``exclude_types`` 用于批量执行
    （见 ``ExecutionWorkflow.execute_batch``），三者与 ``case_id``/``case_title``
    互斥，由调用方按场景二选一。
    """

    automation_json_path: str
    case_id: str = ""
    case_title: str = ""
    app_package: str = ""
    output_dir: str = ""
    request_id: str = ""
    case_ids: list[str] = field(default_factory=list)
    max_cases: int | None = None
    exclude_types: list[str] = field(default_factory=list)


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


@dataclass(slots=True)
class ScenarioBatchExecutionResult:
    """批量执行多条自动化场景的汇总结果。

    ``skipped_case_ids`` 记录因 ``exclude_types`` 过滤掉、或显式 ``case_ids``
    里未命中的用例 id，便于在结果里如实告知用户哪些用例没有被执行。
    """

    module: str
    results: list[ScenarioExecutionResult] = field(default_factory=list)
    skipped_case_ids: list[str] = field(default_factory=list)

    @property
    def pass_count(self) -> int:
        return sum(1 for r in self.results if r.status == "PASS")

    @property
    def fail_count(self) -> int:
        return sum(1 for r in self.results if r.status == "FAIL")
