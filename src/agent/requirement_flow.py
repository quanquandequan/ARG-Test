"""需求分析 CLI 交互辅助：意图识别、澄清问题解析与答案组装。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable

# CLI 与 ReAct 直達路由使用的内部标记（用户正常输入不会命中）
CLI_FINAL_MARKER = "[RAG:CLI_FINAL]"
CLI_DESIGN_CASES_MARKER = "[RAG:CLI_DESIGN_CASES]"

_DRAFT_HEADER = "需求分析草稿（待确认）"
_QUESTIONS_HEADER = "需求确认问题："
_NO_QUESTIONS_HINT = "无必须确认的问题"

# 从 draft 回答中解析编号问题：  1. 问题文本（来源：A001）
_QUESTION_LINE = re.compile(
    r"^\s*(\d+)\.\s+(.+?)(?:\s*（来源：[^）]+）)?\s*$"
)

# 从 final 回答中提取 confirmed JSON 路径
_ANALYSIS_JSON_PATH = re.compile(
    r"(?:分析结果|JSON 文件)[：:]\s*([^\s\n]+\.json)",
    re.IGNORECASE,
)

# 用户意图关键词
_ANALYSIS_KEYWORDS = ("需求分析", "分析需求", "知识图谱", "需求图谱", "req_graph")
_MANUAL_CASE_KEYWORDS = ("测试用例", "用例设计", "编写用例", "设计用例", "写用例")
_AUTOMATION_KEYWORDS = ("自动化", "automation", "自动化用例", "自动化case", "ui自动化")


class RequirementGoal(StrEnum):
    """需求流程目标：只分析 / 人工用例 / 自动化 case。"""

    ANALYSIS_ONLY = "analysis_only"
    MANUAL_CASES = "manual_cases"
    AUTOMATION_CASES = "automation_cases"


@dataclass(slots=True)
class ClarificationQuestion:
    """一条待澄清问题。"""

    index: int
    text: str
    source_id: str = ""


@dataclass(slots=True)
class RequirementFlowSession:
    """CLI 澄清会话状态：保存 draft 参数与原始目标。"""

    goal: RequirementGoal
    draft_args: dict[str, Any] = field(default_factory=dict)
    questions: list[ClarificationQuestion] = field(default_factory=list)
    original_query: str = ""


def detect_requirement_goal(query: str) -> RequirementGoal | None:
    """从用户输入识别需求相关目标；非需求场景返回 None。"""
    text = (query or "").strip()
    if not text:
        return None

    # 已有 confirmed JSON 路径时由 react_loop 直达，不走澄清流程
    if re.search(r"\.json\b", text) and any(
        kw in text for kw in ("生成", "设计", "执行", "运行")
    ):
        return None

    has_requirement_signal = any(
        signal in text.lower() or signal in text
        for signal in (
            "需求",
            "prd",
            ".md",
            ".pdf",
            ".xlsx",
            ".xmind",
            "requirement",
        )
    ) or bool(re.search(r"(?:/|\./)[^\s]+\.(md|pdf|xlsx|xmind|txt)\b", text, re.I))

    if not has_requirement_signal:
        return None

    lowered = text.lower()
    if any(kw in text or kw in lowered for kw in _AUTOMATION_KEYWORDS):
        return RequirementGoal.AUTOMATION_CASES
    if any(kw in text for kw in _MANUAL_CASE_KEYWORDS):
        return RequirementGoal.MANUAL_CASES
    if any(kw in text for kw in _ANALYSIS_KEYWORDS):
        return RequirementGoal.ANALYSIS_ONLY

    # 给了需求文件但未说明目标，默认只做分析
    if re.search(r"(?:/|\./)[^\s]+\.(md|pdf|xlsx|xmind|txt)\b", text, re.I):
        return RequirementGoal.ANALYSIS_ONLY
    if len(text) > 80 and "需求" in text:
        return RequirementGoal.ANALYSIS_ONLY
    return None


def is_draft_pending_confirmation(answer: str) -> bool:
    """判断 Agent 回答是否为待澄清的 draft。"""
    return _DRAFT_HEADER in (answer or "")


def parse_clarification_questions(answer: str) -> list[ClarificationQuestion]:
    """从 draft 回答的「需求确认问题」区块提取编号问题列表。"""
    text = answer or ""
    if _QUESTIONS_HEADER not in text:
        return []

    section = text.split(_QUESTIONS_HEADER, 1)[1]
    # 截到下一个大段标题或文件尾
    for stop in ("\n\n【", "\n\n回测范围", "\n\n草稿功能点"):
        if stop in section:
            section = section.split(stop, 1)[0]

    questions: list[ClarificationQuestion] = []
    for line in section.splitlines():
        line = line.strip()
        if not line or line.startswith("请按编号"):
            continue
        if _NO_QUESTIONS_HINT in line:
            return []
        match = _QUESTION_LINE.match(line)
        if match:
            idx = int(match.group(1))
            body = match.group(2).strip()
            source_match = re.search(r"（来源：([^）]+)）", line)
            source_id = source_match.group(1) if source_match else ""
            questions.append(
                ClarificationQuestion(index=idx, text=body, source_id=source_id)
            )
    return questions


def format_clarification_answers(
    answers: list[tuple[int, str, str]],
) -> str:
    """将逐题收集的答案组装为 analyze_requirement final 所需的文本。"""
    if not answers:
        return "确认：无补充说明，按草稿内容继续。"
    lines: list[str] = []
    for idx, question, answer in answers:
        lines.append(f"{idx}. {question}")
        lines.append(f"答：{answer}")
        lines.append("")
    return "\n".join(lines).strip()


def collect_clarification_answers_interactive(
    questions: list[ClarificationQuestion],
    *,
    input_fn: Callable[[str], str] = input,
) -> str | None:
    """在 CLI 中逐条询问澄清问题；返回 None 表示用户取消。"""
    if not questions:
        return "确认"

    print("\n--- 需求澄清（逐条回答；空行或 /skip 跳过；/done 提前完成；/cancel 取消）---\n")
    collected: list[tuple[int, str, str]] = []

    for question in questions:
        print(f"Q{question.index}. {question.text}")
        while True:
            try:
                raw = input_fn(f"   A{question.index}: ")
            except (EOFError, KeyboardInterrupt):
                print("\n   (已取消澄清)\n")
                return None
            answer = raw.strip()
            if answer == "/cancel":
                print("   (已取消澄清)\n")
                return None
            if answer == "/done":
                return format_clarification_answers(collected)
            if answer in ("", "/skip"):
                break
            collected.append((question.index, question.text, answer))
            break

    return format_clarification_answers(collected)


def build_cli_final_query(
    session: RequirementFlowSession,
    clarification_answers: str,
) -> str:
    """构造 ReAct 直達 final 分析的内部查询字符串。"""
    payload = {
        "requirement": session.draft_args.get("requirement", ""),
        "requirement_file": session.draft_args.get("requirement_file", ""),
        "module": session.draft_args.get("module", ""),
        "output_dir": session.draft_args.get("output_dir", ""),
        "clarification_answers": clarification_answers,
        "goal": session.goal.value,
    }
    return f"{CLI_FINAL_MARKER}\n{json.dumps(payload, ensure_ascii=False)}"


def build_cli_design_cases_query(
    analysis_json_path: str,
    generation_mode: str,
    module: str = "",
) -> str:
    """构造 ReAct 直達 design_test_cases 的内部查询字符串。"""
    payload = {
        "analysis_json_path": analysis_json_path,
        "generation_mode": generation_mode,
        "module": module,
    }
    return f"{CLI_DESIGN_CASES_MARKER}\n{json.dumps(payload, ensure_ascii=False)}"


def parse_cli_final_payload(query: str) -> dict[str, Any] | None:
    """解析 CLI final 标记负载。"""
    text = (query or "").strip()
    if not text.startswith(CLI_FINAL_MARKER):
        return None
    body = text[len(CLI_FINAL_MARKER) :].strip()
    try:
        data = json.loads(body)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def parse_cli_design_cases_payload(query: str) -> dict[str, Any] | None:
    """解析 CLI design_test_cases 标记负载。"""
    text = (query or "").strip()
    if not text.startswith(CLI_DESIGN_CASES_MARKER):
        return None
    body = text[len(CLI_DESIGN_CASES_MARKER) :].strip()
    try:
        data = json.loads(body)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def extract_analysis_json_path(answer: str) -> str:
    """从 analyze_requirement final 工具输出中提取 JSON 路径。"""
    match = _ANALYSIS_JSON_PATH.search(answer or "")
    return match.group(1).strip() if match else ""


def extract_draft_tool_args(steps: list) -> dict[str, Any]:
    """从 AgentResult.steps 中提取最近一次 analyze_requirement draft 调用参数。"""
    for step in reversed(steps or []):
        tc = getattr(step, "tool_call", None)
        if tc is None or tc.name != "analyze_requirement":
            continue
        args = dict(tc.arguments or {})
        mode = str(args.get("analysis_mode", "draft")).lower()
        if mode == "draft" or "analysis_mode" not in args:
            return args
    return {}


def goal_to_generation_mode(goal: RequirementGoal) -> str:
    """将流程目标映射为 design_test_cases 的 generation_mode。"""
    if goal == RequirementGoal.AUTOMATION_CASES:
        return "automation"
    return "manual"
