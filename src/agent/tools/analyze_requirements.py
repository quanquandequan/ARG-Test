"""Requirements analysis tool — produces a structured Requirement Graph.

Workflow (orchestrated by the Agent):
  1. Agent calls ``knowledge_search`` to retrieve background information.
     - 叭嗒 app features  → search "功能名 测试用例"
     - Plugin / baseline   → search "功能名 xmind"
  2. Agent calls ``analyze_requirements`` with the requirement text and the
     KB search results as ``kb_context``.
  3. This tool calls the injected LLM to produce a RequirementGraph JSON,
     then saves two files:
       <module>_<ts>_req_graph.json  — machine-readable, for downstream tools
       <module>_<ts>_analysis.md     — human-readable report
  4. Returns file paths + key insights to the Agent.

RequirementGraph JSON schema:
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
      "dependencies": [str]
    }],
    "state_transitions": [{
      "entity": str, "states": [str],
      "transitions": [{"from": str, "to": str, "trigger": str, "condition": str}]
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
from src.core.logging import get_logger
from src.domain.requirements import RequirementAnalysisData
from src.llm.base import BaseLLM
from src.llm.types import Message

logger = get_logger(__name__)

_DEFAULT_OUTPUT_DIR = "./outputs/requirements"

# ── LLM prompts ───────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
你是一名资深软件测试工程师，专注于从测试视角分析需求文档。
你的任务是将需求文档解析为结构化的 RequirementGraph（需求图谱），以 JSON 格式输出。

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
      "dependencies": ["F002"]
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
      "impact": "若不澄清对测试的影响"
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
- risks：至少包含2个，按 level 降序排列
- clarifications：需求模糊、缺失、矛盾之处，列出具体可回答的问题
- 结合知识库背景信息，对比现有功能逻辑，重点标注变更点和潜在回归风险
"""

_USER_TEMPLATE = """\
{kb_section}需求文档内容：
{requirement}

模块名称：{module}

请对上述需求进行完整分析，输出 RequirementGraph JSON。
"""

_KB_SECTION_TEMPLATE = """\
以下是知识库中的相关背景信息（现有功能逻辑 / 测试用例），\
请用于识别变更点、回归风险和需求完整性：

{context}

---
"""


class AnalyzeRequirementsTool(BaseTool):
    """Analyze a requirements document and produce a structured Requirement Graph.

    The tool is LLM-powered: it calls the injected language model to extract
    features, state transitions, risks, and clarification questions, then writes:
      - A JSON file  (RequirementGraph — machine-readable, used by downstream tools)
      - A Markdown file (human-readable analysis report)

    All generation parameters (prompts, output dir, temperature) are configurable
    via ``configs/default.yaml`` → ``req_analyzer``.  Constructor arguments take
    the highest priority, then YAML, then code defaults.
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

    @property
    def name(self) -> str:
        return "analyze_requirements"

    @property
    def description(self) -> str:
        return (
            "对需求文档进行测试视角分析，生成结构化 RequirementGraph（需求图谱）。\n"
            "包含：功能点拆解、状态转换图、风险点、待澄清问题、测试策略建议。\n"
            "输出 JSON（机器可读）+ Markdown（可读报告）两个文件，返回文件路径和核心摘要。\n\n"
            "调用规范：调用前先使用 knowledge_search 获取背景信息\n"
            "- 叭嗒 app 功能：搜索 '功能名 测试用例'（了解现有功能逻辑）\n"
            "- 插件/基线/小程序功能：搜索 '功能名 xmind'\n"
            "将搜索结果传入 kb_context 参数。"
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
        if result.data is not None:
            out_dir = (
                Path(output_dir.strip())
                if output_dir.strip()
                else self._default_output_dir
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            graph = result.data.graph
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_module = re.sub(r'[\\/:*?"<>|]', "_", result.data.module)
            json_path = out_dir / f"{safe_module}_{timestamp}_req_graph.json"
            md_path = out_dir / f"{safe_module}_{timestamp}_analysis.md"
            _save_json(graph, json_path)
            _save_markdown(graph, md_path, result.data.module)
            result.content = _render_analysis_summary(
                result.data,
                json_path=str(json_path.resolve()),
                markdown_path=str(md_path.resolve()),
            )
        return result.content

    async def execute_typed(
        self,
        requirement: str = "",
        kb_context: str = "",
        module: str = "",
        output_dir: str = "",
        **kwargs,
    ) -> ToolExecutionResult:
        if not requirement or not requirement.strip():
            return ToolExecutionResult(content="错误：请提供需求文档内容。")

        module = module.strip() or "需求分析"

        # 1. Build LLM prompt
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
        messages = [
            Message(role="system", content=self._system_prompt),
            Message(role="user", content=user_content),
        ]

        # 2. Call LLM
        response = await self._llm.generate_chat(
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        raw = response.content.strip()

        # 3. Parse RequirementGraph JSON
        graph = self._parse_graph(raw, module)
        if graph is None:
            logger.warning(
                "analyze_requirements_parse_failed",
                module=module,
                raw=raw[:200],
            )
            return ToolExecutionResult(
                content="LLM 未能生成有效的 RequirementGraph，请检查需求文档内容后重试。"
            )

        # 4. Add meta section
        graph["_meta"] = {
            "module": module,
            "generated_at": datetime.now().isoformat(),
            "source_length": len(requirement),
            "has_kb_context": bool(kb_context.strip()),
        }

        features = graph.get("features", [])
        risks = graph.get("risks", [])
        clarifications = graph.get("clarifications", [])
        high_risks = [r for r in risks if r.get("level") == "high"]

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
        return ToolExecutionResult(
            content=_render_analysis_summary(data),
            data=data,
            metadata={"output_dir": output_dir.strip() or str(self._default_output_dir)},
        )

    def render_markdown(self, graph: dict, module: str) -> str:
        return _render_markdown_text(graph, module)

    # ── Parsing ──────────────────────────────────────────────────────────────

    def _parse_graph(self, raw: str, module: str) -> dict | None:
        """Extract RequirementGraph dict from LLM output; return None on failure."""
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


# ── File output helpers (module-level, easier to test) ────────────────────────

def _normalise_graph(data: dict, module: str) -> dict:
    """Fill missing top-level keys with safe defaults."""
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


def _save_json(graph: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)


def _render_markdown_text(graph: dict, module: str) -> str:
    """Render the RequirementGraph as a human-readable Markdown report."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = [
        f"# 需求分析报告：{module}",
        "",
        f"> 生成时间：{ts}",
        "",
        "## 摘要",
        "",
        graph.get("summary", ""),
        "",
    ]

    # Actors
    actors = graph.get("actors", [])
    if actors:
        lines += ["## 参与角色", ""]
        lines += [f"- {a}" for a in actors]
        lines.append("")

    # Feature table
    features = graph.get("features", [])
    if features:
        _risk_label = {"high": "🔴 高", "medium": "🟡 中", "low": "🟢 低"}
        lines += [
            "## 功能点清单",
            "",
            "| ID | 功能名称 | 优先级 | 风险等级 | 测试重点 |",
            "|---|---|---|---|---|",
        ]
        for f in features:
            focus = "、".join(f.get("test_focus", [])[:2])
            risk = _risk_label.get(f.get("risk_level", "medium"), f.get("risk_level", ""))
            lines.append(
                f"| {f.get('id','')} | {f.get('name','')} |"
                f" {f.get('priority','')} | {risk} | {focus} |"
            )
        lines.append("")

        # Feature details
        lines += ["## 功能详情", ""]
        for f in features:
            lines += [f"### {f.get('id','')}：{f.get('name','')}", ""]
            if f.get("description"):
                lines += [f"**描述**：{f['description']}", ""]
            if f.get("boundaries"):
                lines += ["**边界条件**：", ""]
                lines += [f"- {b}" for b in f["boundaries"]]
                lines.append("")
            if f.get("test_focus"):
                lines += ["**测试重点**：", ""]
                lines += [f"- {t}" for t in f["test_focus"]]
                lines.append("")
            if f.get("risk_reason"):
                lines += [f"**风险原因**：{f['risk_reason']}", ""]
            if f.get("dependencies"):
                lines += [f"**依赖**：{', '.join(f['dependencies'])}", ""]

    # State transitions
    for st in graph.get("state_transitions", []):
        lines += [
            f"## 状态转换：{st.get('entity','')}",
            "",
            f"**状态**：{'、'.join(st.get('states', []))}",
            "",
            "| 当前状态 | 触发事件 | 目标状态 | 前提条件 |",
            "|---|---|---|---|",
        ]
        for t in st.get("transitions", []):
            lines.append(
                f"| {t.get('from','')} | {t.get('trigger','')} |"
                f" {t.get('to','')} | {t.get('condition','')} |"
            )
        lines.append("")

    # Risks
    risks = graph.get("risks", [])
    if risks:
        _risk_label = {"high": "🔴 高", "medium": "🟡 中", "low": "🟢 低"}
        lines += [
            "## 风险分析",
            "",
            "| 风险区域 | 等级 | 描述 | 测试建议 |",
            "|---|---|---|---|",
        ]
        for r in risks:
            label = _risk_label.get(r.get("level", "medium"), r.get("level", ""))
            lines.append(
                f"| {r.get('area','')} | {label} |"
                f" {r.get('description','')} | {r.get('suggestion','')} |"
            )
        lines.append("")

    # Clarifications
    clars = graph.get("clarifications", [])
    if clars:
        lines += [
            "## 待澄清问题",
            "",
            "| ID | 问题 | 背景 | 影响范围 |",
            "|---|---|---|---|",
        ]
        for q in clars:
            lines.append(
                f"| {q.get('id','')} | {q.get('question','')} |"
                f" {q.get('context','')} | {q.get('impact','')} |"
            )
        lines.append("")

    # Test strategy
    strategy = graph.get("test_strategy", {})
    if any(strategy.get(k) for k in ("scope", "focus_areas", "suggestion")):
        lines += ["## 测试策略建议", ""]
        if strategy.get("scope"):
            lines += [f"**范围**：{strategy['scope']}", ""]
        if strategy.get("focus_areas"):
            lines += ["**重点区域**：", ""]
            lines += [f"- {a}" for a in strategy["focus_areas"]]
            lines.append("")
        if strategy.get("exclusions"):
            lines += ["**排除项**：", ""]
            lines += [f"- {e}" for e in strategy["exclusions"]]
            lines.append("")
        if strategy.get("suggestion"):
            lines += [f"**策略建议**：{strategy['suggestion']}", ""]

    return "\n".join(lines)


def _save_markdown(graph: dict, path: Path, module: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(_render_markdown_text(graph, module))


def _render_analysis_summary(
    data: RequirementAnalysisData,
    json_path: str = "",
    markdown_path: str = "",
) -> str:
    graph = data.graph
    features = graph.get("features", [])
    risks = graph.get("risks", [])
    clarifications = graph.get("clarifications", [])
    high_risks = [r for r in risks if r.get("level") == "high"]

    lines = [f"需求分析完成：{data.module}", ""]
    if json_path:
        lines.append(f"JSON 文件：{json_path}")
    if markdown_path:
        lines.append(f"Markdown 报告：{markdown_path}")
    if json_path or markdown_path:
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
