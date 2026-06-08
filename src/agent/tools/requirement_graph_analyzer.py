"""RequirementGraph 分析器：生成结构化需求图谱。

工作流（由 Agent 编排）：
  1. Agent 调用 ``knowledge_search`` 检索背景信息。
     - 叭嗒 app 功能  → 搜索 "功能名 测试用例"
     - 插件 / 基线    → 搜索 "功能名 xmind"
  2. Agent 携带需求文本和 KB 搜索结果（``kb_context``）
     调用兼容工具名 ``analyze_requirements``。
  3. 本工具调用注入的 LLM 生成 RequirementGraph JSON，
     然后保存 <module>_<ts>_req_graph.json。
  4. 向 Agent 返回 JSON 文件路径和关键洞察。

RequirementGraph JSON schema：
  {
    "summary": str,
    "actors": [str],
    "features": [{
      "id": str, "name": str, "description": str,
      "priority": "P0"|"P1"|"P2",
      "risk_level": "high"|"medium"|"low",
      "risk_reason": str,
      "boundaries": [str],
      "test_focus": [str],
      "dependencies": [str],
      "evidence": [{"field": str, "source": "prd"|"confirmation", "quote": str}]
    }],
    "state_transitions": [{
      "entity": str, "states": [str],
      "transitions": [{"from": str, "to": str, "trigger": str, "condition": str}],
      "evidence": [{"field": str, "source": "prd"|"confirmation", "quote": str}]
    }],
    "risks": [{"area": str, "level": str, "description": str, "suggestion": str}],
    "clarifications": [{"id": str, "question": str, "context": str, "impact": str}],
    "test_strategy": {"scope": str, "focus_areas": [str], "exclusions": [str], "suggestion": str}
  }
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from src.agent.base_tool import BaseTool
from src.agent.tool_result import ToolExecutionResult
from src.services.artifact_repository import LocalArtifactRepository
from src.core.logging import get_logger
from src.domain.artifacts import ArtifactKind
from src.domain.requirements import RequirementAnalysisData
from src.llm.base import BaseLLM
from src.llm.types import Message

logger = get_logger(__name__)

_DEFAULT_OUTPUT_DIR = "./outputs/requirements"
_EVIDENCE_SOURCES = {"prd", "confirmation"}
_VALIDATION_RETRY_LIMIT = 1

# ── LLM 提示词 ───────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
你是一名资深软件测试工程师，专注于从测试视角分析需求文档。
你的任务是将需求文档解析为结构化的 RequirementGraph（需求图谱），以 JSON 格式输出。

## 信息优先级（非常重要）
- 当前输入的【需求文档内容】是唯一的业务事实来源。
- 知识库背景只用于识别回归风险、历史差异和可能需要澄清的问题。
- 当知识库背景与需求文档冲突时，必须以需求文档为准。
- 不得把知识库中的历史位置、登录态、分页、文案、跳转等行为写入
  features、state_transitions 或测试策略，除非需求文档明确写到。
- 如果知识库内容看起来与需求文档不一致，只能放入 risks 或
  clarifications，并说明“历史逻辑可能冲突”，不能当作新需求。
- 知识库中的旧页面、旧功能、旧登录态、旧分页、旧跳转、旧文案只能用于
  “历史功能 / 历史差异 / 回测范围”分析，不得写入 features。

## 事实证据规则（非常重要）
- features 中的 description、每一条 boundaries，以及 state_transitions 中的
  每一条 transitions，都必须能追溯到需求文档或用户确认答复。
- test_focus、risks、test_strategy 是测试视角推导字段，不要求逐字 evidence；
  但它们不能引入新的产品功能事实。
- 每个 feature 必须提供 evidence 数组。evidence 只能使用以下 source：
  - "prd"：quote 必须逐字摘自【需求文档内容】
  - "confirmation"：quote 必须逐字摘自【用户确认补充】
- evidence.field 必须指向被证明的字段，例如：
  - "description"
  - "boundaries[0]"
- quote 必须是原文短句，不能改写、概括或补写。
- 如果某个功能、边界、状态、文案、默认值、重试、缓存、空态、接口策略等
  在需求文档和用户确认答复中找不到逐字或明确依据，不得写入 features；
  应写入 clarifications 向用户确认。
- 知识库、历史测试用例、Bug、XMind 不能作为 features 的 evidence。

## V1 测试范围收敛
- 本阶段只服务功能测试和 UI 自动化测试。
- 不生成接口测试、埋点测试、配置专项测试。
- 需求文档中的“接口字段建议”“数据字段建议”“配置能力建议”“埋点需求”
  仅作为理解功能展示、页面状态和用户操作的背景，不要拆成独立 feature。
- features 必须聚焦：页面展示、用户点击、状态变化、页面跳转、UI 文案、
  按钮/入口、模块显示隐藏、空态/异常态、UI 自动化可验证点。
- test_focus 应优先表达 UI 自动化可验证点，例如页面入口、按钮文案、
  点击动作、跳转目标、空态文案、模块显示/隐藏。

## 输出格式（严格遵守）
只输出 JSON 对象，不加任何 Markdown 标记或解释文字。

JSON 结构如下：
{
  "summary": "一句话描述本次需求的核心功能（30字以内）",
  "actors": ["参与角色1", "参与角色2"],
  "features": [
    {
      "id": "F001",
      "name": "功能名称",
      "description": "功能描述（面向测试人员，说明做什么、何时触发、对谁生效）",
      "priority": "P0",
      "risk_level": "high",
      "risk_reason": "高风险时必填，说明原因",
      "boundaries": ["边界条件1", "边界条件2"],
      "test_focus": ["测试重点1", "测试重点2"],
      "dependencies": ["F002"],
      "evidence": [
        {
          "field": "description",
          "source": "prd",
          "quote": "需求文档中的逐字原文短句"
        },
        {
          "field": "boundaries[0]",
          "source": "confirmation",
          "quote": "用户确认答复中的逐字原文短句"
        }
      ]
    }
  ],
  "state_transitions": [
    {
      "entity": "数据实体名（如：订单、账户、内容）",
      "states": ["状态A", "状态B", "状态C"],
      "transitions": [
        {
          "from": "状态A",
          "to": "状态B",
          "trigger": "触发事件",
          "condition": "前提条件（无则填空字符串）"
        }
      ],
      "evidence": [
        {
          "field": "transitions[0]",
          "source": "prd",
          "quote": "需求文档中的逐字原文短句"
        }
      ]
    }
  ],
  "risks": [
    {
      "area": "风险区域",
      "level": "high",
      "description": "风险详细描述",
      "suggestion": "测试应对建议"
    }
  ],
  "clarifications": [
    {
      "id": "Q001",
      "question": "具体问题（以问号结尾）",
      "context": "问题来源的需求上下文",
      "impact": "若不澄清对测试的影响",
      "blocking": true
    }
  ],
  "test_strategy": {
    "scope": "测试范围说明",
    "focus_areas": ["重点测试区域1"],
    "exclusions": ["明确排除的内容（无则为空数组）"],
    "suggestion": "整体测试策略建议（100字以内）"
  }
}

## 分析原则
- features：将需求中每个独立的用户操作或系统行为拆为一个 feature
  - priority：P0=核心主流程，P1=重要功能，P2=边缘/辅助功能
  - risk_level：high=复杂逻辑/并发/权限/安全/金融，medium=一般功能，low=纯展示
- state_transitions：识别有明显状态变化的业务实体（订单、账户、内容审核状态等）
  - 每条 transitions 都必须有 evidence，且 evidence quote 必须来自需求文档或用户确认答复
- risks：至少包含2个，按 level 降序排列
- clarifications：需求模糊、缺失、矛盾之处，列出具体可回答的问题
  - blocking=true：会影响当前功能事实确认，confirmed 阶段必须先向用户确认
  - blocking=false：不影响当前功能事实确认，仅作为后续优化或测试关注点
- 结合知识库背景信息，对比现有功能逻辑，重点标注变更点和潜在回归风险；
  但所有功能点必须能在需求文档中找到依据
- confirmed 阶段如果仍有关键逻辑不清楚，必须输出 blocking=true 的
  clarifications，不要把推测写入 features
- test_strategy.focus_areas 必须包含“UI 自动化优先覆盖主流程”；
  无法稳定自动化的视觉/设计验收点只作为人工检查建议
- 不要把接口字段、配置能力、埋点字段写成独立测试范围
"""

_USER_TEMPLATE = """\
需求文档内容：
{requirement}

模块名称：{module}

{kb_section}
请对上述需求进行完整分析，输出 RequirementGraph JSON。
"""

_KB_SECTION_TEMPLATE = """\
【历史知识库参考（辅助）】
以下是知识库中的历史功能逻辑 / 测试用例 / 缺陷记录，请仅用于识别历史差异、\
回归风险和回测范围。
注意：知识库不是本次需求的事实来源；若与需求文档冲突，必须以需求文档为准。
不得把知识库中的功能写入 features，除非需求文档明确描述了同一功能。

{context}

---
"""


class RequirementGraphAnalyzerTool(BaseTool):
    """分析需求文档并生成结构化 Requirement Graph。

    本工具由 LLM 驱动：调用注入的语言模型抽取功能点、状态转换、
    风险和待澄清问题，然后写入 JSON 文件（RequirementGraph：机器可读）。

    所有生成参数（prompts、output dir、temperature）都可通过
    ``configs/default.yaml`` → ``req_analyzer`` 配置。
    优先级为构造函数参数最高，其次 YAML，最后代码默认值。
    """

    def __init__(
        self,
        llm: BaseLLM,
        system_prompt: str | None = None,
        output_dir: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        from src.core.config import get_config

        cfg = get_config().get("req_analyzer", {})
        self._llm = llm
        self._default_output_dir = Path(
            output_dir or cfg.get("output_dir", _DEFAULT_OUTPUT_DIR)
        )
        self._temperature = (
            temperature if temperature is not None
            else float(cfg.get("temperature", 0.3))
        )
        self._max_tokens = (
            max_tokens if max_tokens is not None
            else int(cfg.get("max_tokens", 8192))
        )
        self._system_prompt = system_prompt or cfg.get("system_prompt", "") or _SYSTEM_PROMPT
        self._artifacts = LocalArtifactRepository(base_dir=str(self._default_output_dir))

    @property
    def name(self) -> str:
        return "analyze_requirements"

    @property
    def description(self) -> str:
        return (
            "对需求文档进行测试视角分析，生成结构化 RequirementGraph（需求图谱）。\n"
            "包含：功能点拆解、状态转换图、风险点、待澄清问题、测试策略建议。\n"
            "仅输出 JSON（机器可读）文件，返回文件路径和核心摘要。\n\n"
            "调用规范：当前需求文档是唯一需求事实来源；kb_context 仅可作为"
            "历史功能、历史差异和回测范围参考。"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "requirement": {
                    "type": "string",
                    "description": "需求文档的完整文本内容",
                },
                "kb_context": {
                    "type": "string",
                    "description": (
                        "来自 knowledge_search 的背景信息（现有功能逻辑/测试用例）。"
                        "建议提供，用于识别变更点和回归风险。"
                    ),
                },
                "module": {
                    "type": "string",
                    "description": "功能模块名称，用于文件命名（可选）",
                },
                "output_dir": {
                    "type": "string",
                    "description": f"输出目录（可选，默认 {_DEFAULT_OUTPUT_DIR}）",
                },
            },
            "required": ["requirement"],
        }

    async def execute(
        self,
        requirement: str = "",
        kb_context: str = "",
        module: str = "",
        output_dir: str = "",
        **kwargs,
    ) -> str:
        result = await self.execute_typed(
            requirement=requirement,
            kb_context=kb_context,
            module=module,
            output_dir=output_dir,
            **kwargs,
        )
        return result.content

    async def execute_typed(
        self,
        requirement: str = "",
        kb_context: str = "",
        module: str = "",
        output_dir: str = "",
        request_id: str = "",
        persist: bool = True,
        analysis_status: str = "",
        requirement_source_path: str = "",
        clarification_answers: str = "",
        **kwargs,
    ) -> ToolExecutionResult:
        if not requirement or not requirement.strip():
            return ToolExecutionResult(content="错误：请提供需求文档内容。")

        module = module.strip() or "需求分析"

        # 1. 构建 LLM prompt
        kb_section = (
            _KB_SECTION_TEMPLATE.format(context=kb_context.strip())
            if kb_context.strip()
            else ""
        )
        user_content = _USER_TEMPLATE.format(
            kb_section=kb_section,
            requirement=requirement.strip(),
            module=module,
        )
        if clarification_answers.strip():
            user_content += (
                "\n\n【用户确认补充】\n"
                "以下内容是用户针对待澄清问题给出的确认答案，属于本次需求事实补充：\n\n"
                f"{clarification_answers.strip()}\n"
            )
        messages = [
            Message(role="system", content=self._system_prompt),
            Message(role="user", content=user_content),
        ]

        # 2. 调用 LLM，并在 confirmed 落盘前执行证据校验。
        graph: dict | None = None
        raw = ""
        validation_errors: list[str] = []
        should_validate = _should_validate_confirmed(persist, analysis_status)
        for attempt in range(_VALIDATION_RETRY_LIMIT + 1):
            response = await self._llm.generate_chat(
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            raw = response.content.strip()

            # 3. 解析 RequirementGraph JSON
            graph = self._parse_graph(raw, module)
            if graph is None:
                logger.warning(
                    "analyze_requirements_parse_failed",
                    module=module,
                    raw=raw[:200],
                )
                return ToolExecutionResult(
                    content=(
                        "LLM 未能生成有效的 RequirementGraph，"
                        "请检查需求文档内容后重试。"
                    )
                )

            validation_errors = (
                _validate_confirmed_graph(
                    graph,
                    requirement=requirement,
                    clarification_answers=clarification_answers,
                )
                if should_validate
                else []
            )
            if not validation_errors:
                break
            if attempt < _VALIDATION_RETRY_LIMIT:
                logger.info(
                    "analyze_requirements_evidence_retry",
                    module=module,
                    error_count=len(validation_errors),
                )
                messages = [
                    *messages,
                    Message(role="assistant", content=raw),
                    Message(
                        role="user",
                        content=_build_validation_retry_prompt(validation_errors),
                    ),
                ]

        if graph is None:
            return ToolExecutionResult(
                content="LLM 未能生成有效的 RequirementGraph，请检查需求文档内容后重试。"
            )
        if validation_errors:
            logger.warning(
                "analyze_requirements_evidence_validation_failed",
                module=module,
                error_count=len(validation_errors),
            )
            return ToolExecutionResult(
                content=_render_validation_failure(validation_errors),
                metadata={
                    "request_id": request_id,
                    "validation_error_count": len(validation_errors),
                },
            )

        # 4. 添加 meta 区块
        graph["_meta"] = {
            "module": module,
            "generated_at": datetime.now().isoformat(),
            "source_length": len(requirement),
            "has_kb_context": bool(kb_context.strip()),
            "analysis_status": analysis_status.strip() or (
                "confirmed" if persist else "draft"
            ),
            "clarification_answers_used": bool(clarification_answers.strip()),
            "evidence_validated": should_validate,
        }
        if requirement_source_path.strip():
            graph["_meta"]["requirement_source_path"] = requirement_source_path.strip()
        if clarification_answers.strip():
            graph["_meta"]["clarification_answers"] = clarification_answers.strip()
        if persist:
            graph["_meta"]["confirmed_at"] = datetime.now().isoformat()

        features = graph.get("features", [])
        risks = graph.get("risks", [])
        clarifications = graph.get("clarifications", [])
        logger.info(
            "analyze_requirements_done",
            module=module,
            features=len(features),
            risks=len(risks),
            clarifications=len(clarifications),
        )
        data = RequirementAnalysisData(
            module=module,
            summary=graph.get("summary", "—"),
            graph=graph,
            feature_count=len(features),
            risk_count=len(risks),
            clarification_count=len(clarifications),
            kb_context=kb_context,
        )
        if not persist:
            return ToolExecutionResult(
                content=_render_analysis_summary(data, json_path=""),
                data=data,
                metadata={
                    "request_id": request_id,
                    "analysis_status": "draft",
                    "persisted": False,
                },
            )

        out_dir = (
            Path(output_dir.strip())
            if output_dir.strip()
            else self._default_output_dir
        )
        metadata = _build_request_metadata(request_id)
        json_artifact = self._artifacts.save_json(
            ArtifactKind.REQUIREMENT_ANALYSIS_JSON,
            module,
            graph,
            metadata=metadata,
            suffix="req_graph",
            directory=out_dir,
        )
        return ToolExecutionResult(
            content=_render_analysis_summary(data, json_path=str(json_artifact.path)),
            data=data,
            artifacts=[json_artifact],
            metadata={
                "request_id": request_id,
                "output_dir": str(out_dir.resolve()),
            },
        )

    # ── 解析 ─────────────────────────────────────────────────────────────────

    def _parse_graph(self, raw: str, module: str) -> dict | None:
        """从 LLM 输出中提取 RequirementGraph 字典；失败时返回 None。"""
        text = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE).strip()

        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return _normalise_graph(data, module)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                data = json.loads(match.group())
                if isinstance(data, dict):
                    return _normalise_graph(data, module)
            except json.JSONDecodeError:
                pass

        return None


# ── 文件输出辅助方法（模块级，更便于测试）──────────────────────────────────

def _normalise_graph(data: dict, module: str) -> dict:
    """用安全默认值填充缺失的顶层键。"""
    return {
        "summary": str(data.get("summary", "")).strip() or f"{module} 需求分析",
        "actors": list(data.get("actors", [])),
        "features": list(data.get("features", [])),
        "state_transitions": list(data.get("state_transitions", [])),
        "risks": list(data.get("risks", [])),
        "clarifications": list(data.get("clarifications", [])),
        "test_strategy": data.get(
            "test_strategy",
            {"scope": "", "focus_areas": [], "exclusions": [], "suggestion": ""},
        ),
    }


def _should_validate_confirmed(persist: bool, analysis_status: str) -> bool:
    """判断当前调用是否需要执行 confirmed 产物事实校验。"""
    status = analysis_status.strip().lower()
    return status == "confirmed" or (persist and status != "draft")


def _validate_confirmed_graph(
    graph: dict,
    *,
    requirement: str,
    clarification_answers: str,
) -> list[str]:
    """校验 confirmed RequirementGraph 中的需求事实都有原文证据。"""
    errors: list[str] = []
    source_texts = {
        "prd": _normalise_evidence_text(requirement),
        "confirmation": _normalise_evidence_text(clarification_answers),
    }
    _downgrade_answered_blocking_clarifications(
        graph.get("clarifications", []),
        clarification_answers,
    )

    blocking_clarifications = _blocking_clarifications(graph.get("clarifications", []))
    if blocking_clarifications:
        errors.append(
            "confirmed JSON 中仍包含 blocking=true 的待澄清问题。"
            "请先让用户确认这些关键问题，不要把未确认需求落为最终 JSON。"
        )

    features = graph.get("features", [])
    if not isinstance(features, list) or not features:
        errors.append("confirmed JSON 必须包含至少一个 features 功能点。")
        return errors

    for feature_idx, feature in enumerate(features, start=1):
        if not isinstance(feature, dict):
            errors.append(f"features[{feature_idx - 1}] 不是对象，无法校验证据。")
            continue
        feature_label = str(feature.get("id") or f"F{feature_idx:03d}")
        evidence = _normalise_evidence_items(feature.get("evidence"))
        if not evidence:
            errors.append(f"{feature_label} 缺少 evidence，不能写入 confirmed JSON。")
            continue

        _validate_required_field_evidence(
            errors,
            item_label=feature_label,
            field="description",
            evidence=evidence,
            source_texts=source_texts,
        )
        boundaries = feature.get("boundaries", [])
        if isinstance(boundaries, list):
            for idx, value in enumerate(boundaries):
                if str(value).strip():
                    _validate_required_field_evidence(
                        errors,
                        item_label=feature_label,
                        field=f"boundaries[{idx}]",
                        evidence=evidence,
                        source_texts=source_texts,
                    )
    _validate_state_transition_evidence(
        errors,
        graph.get("state_transitions", []),
        source_texts,
    )
    return errors


def _validate_state_transition_evidence(
    errors: list[str],
    state_transitions,
    source_texts: dict[str, str],
) -> None:
    """校验状态转换条目是否具备 PRD 或用户确认答复证据。"""
    if not isinstance(state_transitions, list):
        return
    for entity_idx, entity in enumerate(state_transitions):
        if not isinstance(entity, dict):
            continue
        transitions = entity.get("transitions", [])
        if not isinstance(transitions, list):
            continue
        evidence = _normalise_evidence_items(entity.get("evidence"))
        entity_name = str(entity.get("entity") or f"state_transitions[{entity_idx}]")
        for idx, transition in enumerate(transitions):
            if not isinstance(transition, dict) or not transition:
                continue
            _validate_required_field_evidence(
                errors,
                item_label=entity_name,
                field=f"transitions[{idx}]",
                evidence=evidence,
                source_texts=source_texts,
            )


def _validate_required_field_evidence(
    errors: list[str],
    *,
    item_label: str,
    field: str,
    evidence: list[dict[str, str]],
    source_texts: dict[str, str],
) -> None:
    """校验指定字段是否有可在事实源中找到的 evidence quote。"""
    candidates = [
        item for item in evidence
        if _evidence_field_matches(item.get("field", ""), field)
    ]
    if not candidates:
        errors.append(f"{item_label}.{field} 缺少 evidence。")
        return
    if not any(_evidence_quote_is_valid(item, source_texts) for item in candidates):
        errors.append(
            f"{item_label}.{field} 的 evidence quote 未在 PRD 或用户确认答复中找到。"
        )


def _normalise_evidence_items(raw) -> list[dict[str, str]]:
    """把 LLM 输出的 evidence 规整为字典列表。"""
    if not isinstance(raw, list):
        return []
    items: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        items.append({
            "field": str(item.get("field") or "").strip(),
            "source": str(item.get("source") or "").strip().lower(),
            "quote": str(item.get("quote") or "").strip(),
        })
    return items


def _evidence_field_matches(actual: str, expected: str) -> bool:
    """判断 evidence.field 是否指向目标字段。"""
    actual = actual.strip()
    if actual == expected:
        return True
    if expected.startswith("transitions["):
        return actual in {"transitions", "state_transitions"} or actual == expected
    return False


def _evidence_quote_is_valid(
    item: dict[str, str],
    source_texts: dict[str, str],
) -> bool:
    """判断 evidence quote 是否能在声明的事实源中找到。"""
    source = item.get("source", "")
    quote = _normalise_evidence_text(item.get("quote", ""))
    if source not in _EVIDENCE_SOURCES or not quote:
        return False
    source_text = source_texts.get(source, "")
    if not source_text:
        return False
    if quote in source_text:
        return True
    return _quote_tokens_in_source(quote, source_text)


def _normalise_evidence_text(text: str) -> str:
    """规整空白和中英文引号，让短句包含判断更稳定。"""
    normalised = str(text or "")
    replacements = {
        "“": '"',
        "”": '"',
        "，": ",",
        "。": ".",
        "：": ":",
        "；": ";",
        "、": ",",
        "！": "!",
        "？": "?",
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
        "《": "<",
        "》": ">",
    }
    for src, dst in replacements.items():
        normalised = normalised.replace(src, dst)
    normalised = re.sub(r"[‘’'`]", '"', normalised)
    normalised = re.sub(r"[-–—_]+", "", normalised)
    return re.sub(r"\s+", "", normalised)


def _quote_tokens_in_source(quote: str, source_text: str) -> bool:
    """允许证据短句跨 Markdown 列表或被轻微合并表达。"""
    tokens = _evidence_tokens(quote)
    if not tokens:
        return False
    if len(tokens) == 1:
        return tokens[0] in source_text

    cursor = 0
    matched = 0
    for token in tokens:
        found = source_text.find(token, cursor)
        if found < 0:
            continue
        matched += 1
        cursor = found + len(token)
    if matched == len(tokens):
        return True
    return _quote_token_coverage_is_enough(tokens, source_text)


def _quote_token_coverage_is_enough(tokens: list[str], source_text: str) -> bool:
    """当 quote 由列表项合并而来时，允许较高覆盖率命中。"""
    unique_tokens = list(dict.fromkeys(tokens))
    if len(unique_tokens) < 4:
        return False
    matched = sum(1 for token in unique_tokens if token in source_text)
    return matched >= 4 and matched / len(unique_tokens) >= 0.55


def _evidence_tokens(text: str) -> list[str]:
    """提取用于顺序包含匹配的稳定 token。"""
    tokens: list[str] = []
    for part in re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9]+", text):
        if re.fullmatch(r"[\u4e00-\u9fff]+", part):
            if len(part) <= 6:
                tokens.append(part)
            else:
                tokens.extend(part[idx:idx + 2] for idx in range(len(part) - 1))
            continue
        if len(part) >= 2:
            tokens.append(part)
    return tokens


def _downgrade_answered_blocking_clarifications(
    clarifications,
    clarification_answers: str,
) -> None:
    """把用户确认答复中已覆盖的问题从阻塞项降级为非阻塞项。"""
    if not isinstance(clarifications, list) or not clarification_answers.strip():
        return
    answer_tokens = set(_evidence_tokens(_normalise_evidence_text(clarification_answers)))
    if not answer_tokens:
        return
    for item in clarifications:
        if not isinstance(item, dict) or not _is_blocking_clarification(item):
            continue
        question_text = " ".join(
            str(item.get(key) or "")
            for key in ("question", "context", "impact")
        )
        question_tokens = set(_evidence_tokens(_normalise_evidence_text(question_text)))
        if _has_answer_overlap(question_tokens, answer_tokens):
            item["blocking"] = False
            item["blocking_downgraded_reason"] = "用户确认答复中已有相关信息"


def _has_answer_overlap(question_tokens: set[str], answer_tokens: set[str]) -> bool:
    """判断澄清问题是否已被用户答复覆盖。"""
    if not question_tokens or not answer_tokens:
        return False
    matched = 0
    for token in question_tokens:
        if token in answer_tokens:
            matched += 1
            continue
        if any(token in answer or answer in token for answer in answer_tokens):
            matched += 1
    return matched >= 2 or bool(question_tokens & answer_tokens)


def _blocking_clarifications(value) -> list[dict]:
    """返回会阻塞 confirmed 落盘的关键澄清项。"""
    if not isinstance(value, list):
        return []
    blocking_items: list[dict] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        if not any(str(val).strip() for val in item.values()):
            continue
        if _is_blocking_clarification(item):
            blocking_items.append(item)
    return blocking_items


def _is_blocking_clarification(item: dict) -> bool:
    """判断单条澄清问题是否阻塞 confirmed 产物。"""
    raw = item.get("blocking", False)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in {"true", "yes", "y", "1", "是", "阻塞"}
    return False


def _build_validation_retry_prompt(errors: list[str]) -> str:
    """构造一次自动修正提示，要求 LLM 删除无依据事实或补充合法 evidence。"""
    rendered_errors = "\n".join(f"- {error}" for error in errors[:20])
    return (
        "上一次 RequirementGraph 未通过 confirmed 事实证据校验：\n"
        f"{rendered_errors}\n\n"
        "请重新输出完整 JSON，并严格遵守：\n"
        "1. features.description、每条 boundaries、"
        "每条 state_transitions.transitions 都必须提供 evidence。\n"
        "2. 字段可以用测试人员可读的规整表达，但 evidence.quote 必须来自"
        "【需求文档内容】或【用户确认补充】。\n"
        "3. 找不到来源证据的产品事实必须删除，或改为 clarifications 向用户确认。\n"
        "4. 如果仍有影响功能事实确认的待澄清问题，请标记 blocking=true；"
        "非阻塞测试关注点可标记 blocking=false。\n"
        "只输出修正后的 JSON 对象。"
    )


def _render_validation_failure(errors: list[str]) -> str:
    """渲染 confirmed 产物事实校验失败信息。"""
    lines = [
        "需求分析未生成 confirmed JSON：最终产物未通过事实依据校验。",
        "我没有保存 JSON，因为以下内容缺少 PRD 或用户确认答复中的逐字依据：",
        "",
    ]
    for idx, error in enumerate(errors[:20], start=1):
        lines.append(f"{idx}. {error}")
    lines += [
        "",
        "请补充或确认上述问题后，再生成确认版需求分析 JSON。",
    ]
    return "\n".join(lines)


def _save_json(graph: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)


def _render_analysis_summary(
    data: RequirementAnalysisData,
    json_path: str = "",
) -> str:
    graph = data.graph
    features = graph.get("features", [])
    risks = graph.get("risks", [])
    clarifications = graph.get("clarifications", [])
    high_risks = [r for r in risks if r.get("level") == "high"]

    lines = [f"需求分析完成：{data.module}", ""]
    if json_path:
        lines.append(f"JSON 文件：{json_path}")
    if json_path:
        lines.append("")
    lines += [
        f"摘要：{data.summary}",
        (
            f"功能点：{len(features)} 个"
            f"（P0: {sum(1 for f in features if f.get('priority') == 'P0')} 个，"
            f"P1: {sum(1 for f in features if f.get('priority') == 'P1')} 个）"
        ),
        f"状态转换实体：{len(graph.get('state_transitions', []))} 个",
        f"风险点：{len(risks)} 个（高风险: {len(high_risks)} 个）",
        f"待澄清问题：{len(clarifications)} 个",
    ]

    if high_risks:
        lines += ["", "高风险区域："]
        for risk in high_risks[:3]:
            desc = risk.get("description", "")
            desc_short = desc[:60] + "..." if len(desc) > 60 else desc
            lines.append(f"  [{risk.get('area', '')}] {desc_short}")

    if clarifications:
        lines += ["", "待澄清问题（前3条）："]
        for question in clarifications[:3]:
            lines.append(f"  {question.get('id', '')}: {question.get('question', '')}")

    strategy = graph.get("test_strategy", {})
    if strategy.get("suggestion"):
        lines += ["", f"测试策略建议：{strategy['suggestion']}"]

    return "\n".join(lines)


def _build_request_metadata(request_id: str) -> dict:
    metadata: dict[str, str] = {}
    if request_id:
        metadata["request_id"] = request_id
    return metadata
