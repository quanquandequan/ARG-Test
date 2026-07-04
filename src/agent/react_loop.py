"""ReAct Agent：带工具调用的 Think-Act-Observe 循环。

``run()``（非流式）和 ``run_stream()``（SSE 流式）共享同一个
``_react_core()`` 异步生成器，由它产出类型化事件 dataclass。
这样消除了过去约 80% 的代码重复，也避免了流式路径里的二次 LLM 调用
（最终答案现在从缓存的 ``generate_chat`` 响应切分，而不是再次调用
``generate_chat_stream``）。
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path

from src.agent.base_tool import FINAL_ANSWER_PASSTHROUGH, BaseTool
from src.agent.history import truncate_history
from src.agent.requirement_flow import (
    parse_cli_design_cases_payload,
    parse_cli_final_payload,
)
from src.agent.tool_registry import ToolRegistry
from src.agent.types import AgentResult, AgentStep, Citation
from src.core.logging import get_logger
from src.llm.base import BaseLLM
from src.llm.types import Message, ToolCall

logger = get_logger(__name__)

# ── _react_core 产出的内部事件类型 ───────────────────────────────────────


@dataclass
class _ToolCallEvent:
    tool_names: list[str]
    iteration: int


@dataclass
class _ToolResultEvent:
    tool_name: str
    result: str
    duration_ms: float
    tool_call: ToolCall
    final_answer_mode: str



@dataclass
class _FinalAnswer:
    text: str
    iteration: int
    citations: list[Citation]
    processing_stages: dict[str, float]
    steps: list[AgentStep]


@dataclass
class _ForcedAnswer:
    text: str
    processing_stages: dict[str, float]
    steps: list[AgentStep]



_Event = _ToolCallEvent | _ToolResultEvent | _FinalAnswer | _ForcedAnswer

# 每个 SSE token 分片的大致字符数。
_TOKEN_CHUNK_SIZE = 20

# 未调用工具却出现 [来源：片段N] 引用标注 —— 系统提示词明确禁止的伪造信号，
# 命中即说明本轮模型大概率跳过了应有的 search_knowledge 调用（云端 LLM
# 在 temperature=0 下仍非完全确定性，prompt 约束无法 100% 杜绝）。
_FABRICATED_CITATION_PATTERN = re.compile(r"\[来源[:：]\s*片段\d+")


class ReActAgent:
    """ReAct 模式 Agent：Think -> Act -> Observe -> Repeat -> Answer。"""

    def __init__(
        self,
        llm: BaseLLM,
        tools: list[BaseTool] | None = None,
        system_prompt: str = "",
        max_iterations: int = 10,
        max_history_tokens: int = 4000,
    ):
        if not system_prompt:
            raise ValueError(
                "system_prompt is required — configure it in "
                "configs/default.yaml (agent.system_prompt)"
            )
        self._llm = llm
        self._registry = ToolRegistry(tools or [])
        self._system_prompt = system_prompt
        self._max_iterations = max_iterations
        self._max_history_tokens = max_history_tokens

    @property
    def tool_names(self) -> list[str]:
        return self._registry.names()

    # ── 消息构建 ────────────────────────────────────────────────────────

    def _build_messages(self, history: list[Message] | None, query: str) -> list[Message]:
        """组装 system、截断后的历史记录和当前用户问题。"""
        from datetime import date

        truncated = truncate_history(
            history or [], max_tokens=self._max_history_tokens
        )
        today = date.today().strftime("%Y-%m-%d")
        route_hint = self._build_route_hint(query)
        dated_prompt = f"{self._system_prompt}\n\n当前日期: {today}"
        if route_hint:
            dated_prompt = f"{dated_prompt}\n\n{route_hint}"
        return (
            [Message(role="system", content=dated_prompt)]
            + truncated
            + [Message(role="user", content=query)]
        )

    def _build_route_hint(self, query: str) -> str:
        """针对高频输入模式补充硬提示，减少 LLM 选错工具。"""
        route = _detect_json_route(query)
        if route == "design_test_cases":
            return (
                "【工具路由提示】当前用户输入的是 confirmed 需求分析 JSON 路径，"
                "且目标是生成测试用例。此场景必须直接调用 design_test_cases，"
                "不要先调用 search_knowledge，也不要调用 analyze_requirement。"
            )
        if route == "execute_scenario":
            hint = (
                "【工具路由提示】当前用户输入的是自动化用例 JSON 路径，"
                "且目标是执行已有用例。此场景必须直接调用 execute_scenario，"
                "不要调用 design_test_cases。"
            )
            if _has_batch_qualifier(query):
                hint += (
                    "用户指令里包含数量/类型过滤等限定词（如「前N条」「跳过回归」），"
                    "必须换算成 execute_scenario 的 max_cases（按 JSON 中用例声明顺序"
                    "只跑前 N 条）和/或 exclude_types（要跳过的用例 type，例如"
                    "['回归测试']）参数，在同一次调用里一起传入；如果用户明确列出了"
                    "具体的用例 id，改用 case_ids 参数。不要为了跑多条用例而多次调用"
                    "本工具——execute_scenario 一次调用后不会再进入下一轮。"
                )
            return hint
        return ""

    def _build_direct_tool_call(self, query: str) -> ToolCall | None:
        """对可明确识别的高频请求做硬路由，避免 LLM 选错工具。"""
        cli_final = _build_cli_final_tool_call(query)
        if cli_final is not None:
            return cli_final

        cli_design = _build_cli_design_cases_tool_call(query)
        if cli_design is not None:
            return cli_design

        route = _detect_json_route(query)
        path_match = re.search(r"(?:/|\./|\.\./)[^\s]+\.json", (query or "").strip())
        if route is None or path_match is None:
            return None

        if route == "execute_scenario":
            # 带数量/类型过滤限定词（如"前6条""跳过回归"）时，硬路由只会透传
            # 路径、无法表达这些约束，交给 LLM 通过 max_cases/exclude_types/
            # case_ids 参数精确表达（route hint 已引导它这么做）。
            if _has_batch_qualifier(query):
                return None
            return ToolCall(
                id=f"direct-execute-{uuid.uuid4()}",
                name="execute_scenario",
                arguments={
                    "automation_json_path": path_match.group(0),
                },
            )

        generation_mode = "automation" if _is_automation_case_request(query) else "manual"
        return ToolCall(
            id=f"direct-design-{uuid.uuid4()}",
            name="design_test_cases",
            arguments={
                "analysis_json_path": path_match.group(0),
                "generation_mode": generation_mode,
            },
        )

    # ── 工具执行辅助方法 ────────────────────────────────────────────────

    async def _safe_execute(
        self, tc: ToolCall, trace_id: str
    ) -> tuple[ToolCall, str, float, str]:
        """执行一次工具调用，捕获异常，并返回工具结果与终态回答策略。"""
        t0 = time.perf_counter()
        final_answer_mode = ""
        try:
            tool = self._registry.get(tc.name)
            final_answer_mode = getattr(tool, "final_answer_mode", "")
            result = await tool.execute(
                **tc.arguments,
                request_id=trace_id,
            )
        except Exception as e:
            result = f"工具执行错误: {e}"
            logger.warning(
                "tool_execution_error",
                trace_id=trace_id,
                tool=tc.name,
                error=str(e),
            )
        dur = (time.perf_counter() - t0) * 1000
        return tc, result, dur, final_answer_mode

    # ── 核心 ReAct 循环（run 与 run_stream 共用）────────────────────────

    async def _react_core(
        self,
        query: str,
        history: list[Message] | None = None,
        temperature: float | None = None,
        trace_id: str = "",
    ) -> AsyncGenerator[_Event]:
        """产出类型化事件的统一 ReAct 循环。

        ``run()`` 与 ``run_stream()`` 都消费这些事件，从而消除重复循环逻辑，
        并避免流式路径中的二次 LLM 调用。
        """
        messages = self._build_messages(history, query)
        tool_defs = self._registry.definitions()
        processing_stages: dict[str, float] = {}
        steps: list[AgentStep] = []
        run_t0 = time.perf_counter()
        use_tools = bool(tool_defs)

        direct_tool_call = self._build_direct_tool_call(query)
        if direct_tool_call is not None:
            yield _ToolCallEvent(tool_names=[direct_tool_call.name], iteration=0)
            tc, result, dur_ms, final_answer_mode = await self._safe_execute(
                direct_tool_call,
                trace_id,
            )
            processing_stages[f"iter0.{tc.name}"] = round(dur_ms, 1)
            steps.append(AgentStep(
                step_index=0,
                tool_call=tc,
                tool_result=result,
                duration_ms=round(dur_ms, 1),
            ))
            yield _ToolResultEvent(
                tool_name=tc.name,
                result=result,
                duration_ms=round(dur_ms, 1),
                tool_call=tc,
                final_answer_mode=final_answer_mode,
            )
            total_ms = (time.perf_counter() - run_t0) * 1000
            processing_stages["total"] = round(total_ms, 1)
            yield _FinalAnswer(
                text=result,
                iteration=0,
                citations=self._extract_citations(result),
                processing_stages=processing_stages,
                steps=steps,
            )
            return

        for i in range(self._max_iterations):
            iter_t0 = time.perf_counter()

            response = await self._llm.generate_chat(
                messages=messages,
                tools=tool_defs if use_tools else None,
                tool_choice="auto" if use_tools else None,
                temperature=temperature,
            )

            if (
                i == 0
                and use_tools
                and response.stop_reason != "tool_use"
                and "search_knowledge" in self._registry.names()
                and _FABRICATED_CITATION_PATTERN.search(response.content or "")
            ):
                # 检测到已知的伪造信号：没调用工具就编出了 [来源：片段N] 引用，
                # 强制本轮重试一次 search_knowledge，而不是直接把幻觉答案放出去。
                logger.warning(
                    "agent_forced_retool",
                    trace_id=trace_id,
                    reason="citation_without_tool_call",
                )
                response = await self._llm.generate_chat(
                    messages=messages,
                    tools=tool_defs,
                    tool_choice="search_knowledge",
                    temperature=temperature,
                )

            if response.stop_reason == "tool_use" and response.tool_calls:
                # ── 工具调用轮次 ──
                llm_ms = (time.perf_counter() - iter_t0) * 1000
                logger.info(
                    "agent_tool_calls",
                    trace_id=trace_id,
                    iteration=i,
                    tools=[tc.name for tc in response.tool_calls],
                    llm_ms=round(llm_ms, 1),
                )

                yield _ToolCallEvent(
                    tool_names=[tc.name for tc in response.tool_calls],
                    iteration=i,
                )

                messages.append(Message(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                ))

                # 并发执行所有工具调用
                tc_results = list(await asyncio.gather(
                    *[self._safe_execute(tc, trace_id) for tc in response.tool_calls]
                ))

                passthrough_results: list[str] = []
                for tc, result, dur_ms, final_answer_mode in tc_results:
                    processing_stages[f"iter{i}.{tc.name}"] = round(dur_ms, 1)
                    steps.append(AgentStep(
                        step_index=i,
                        tool_call=tc,
                        tool_result=result,
                        duration_ms=round(dur_ms, 1),
                    ))
                    messages.append(Message(
                        role="tool",
                        content=result,
                        tool_call_id=tc.id,
                        name=tc.name,
                    ))
                    yield _ToolResultEvent(
                        tool_name=tc.name,
                        result=result,
                        duration_ms=round(dur_ms, 1),
                        tool_call=tc,
                        final_answer_mode=final_answer_mode,
                    )
                    if final_answer_mode == FINAL_ANSWER_PASSTHROUGH:
                        passthrough_results.append(result)

                if passthrough_results:
                    total_ms = (time.perf_counter() - run_t0) * 1000
                    processing_stages["total"] = round(total_ms, 1)
                    text = "\n\n".join(passthrough_results)
                    citations = self._extract_citations(text)
                    logger.info(
                        "agent_passthrough_answer",
                        trace_id=trace_id,
                        iteration=i,
                        citations=len(citations),
                        total_ms=round(total_ms, 1),
                    )
                    yield _FinalAnswer(
                        text=text,
                        iteration=i,
                        citations=citations,
                        processing_stages=processing_stages,
                        steps=steps,
                    )
                    return

            else:
                # ── 最终答案 ──
                iter_ms = (time.perf_counter() - iter_t0) * 1000
                steps.append(AgentStep(
                    step_index=i,
                    thinking=response.content,
                    duration_ms=round(iter_ms, 1),
                ))

                citations = self._extract_citations(response.content)
                total_ms = (time.perf_counter() - run_t0) * 1000
                processing_stages["total"] = round(total_ms, 1)

                logger.info(
                    "agent_answer",
                    trace_id=trace_id,
                    iterations=i + 1,
                    citations=len(citations),
                    total_ms=round(total_ms, 1),
                )

                yield _FinalAnswer(
                    text=response.content,
                    iteration=i,
                    citations=citations,
                    processing_stages=processing_stages,
                    steps=steps,
                )
                return

        # 超过最大轮次后，强制不使用工具生成最终答案
        logger.warning(
            "agent_max_iterations",
            trace_id=trace_id,
            max=self._max_iterations,
        )
        messages.append(Message(
            role="user",
            content="请综合以上所有工具的结果，给出最终答案。如果信息不足，请说明。",
        ))
        response = await self._llm.generate_chat(
            messages=messages, tools=None, temperature=temperature
        )
        total_ms = (time.perf_counter() - run_t0) * 1000
        processing_stages["total"] = round(total_ms, 1)

        yield _ForcedAnswer(
            text=response.content,
            processing_stages=processing_stages,
            steps=steps,
        )

    # ── Token 流式辅助方法 ──────────────────────────────────────────────

    @staticmethod
    async def _yield_tokens(text: str) -> AsyncGenerator[str]:
        """将已缓存的 LLM 响应作为 SSE token 事件产出。

        **模拟流式**：此方法不会再次调用 LLM。
        它会把已收到的 ``generate_chat`` 响应拆成固定大小的分片
        （``_TOKEN_CHUNK_SIZE`` 字符），并在每个分片之间通过
        ``asyncio.sleep(0)`` 让出事件循环。

        相比真正的 ``generate_chat_stream``，这里的取舍是：
        - 每轮只调用一次 LLM，避免重复成本和延迟。
        - 工具调用与最终答案的判断更可靠（基于完整响应）。
        - 分片以网络速度到达，而不是按 LLM 生成 token 的速度到达。
          客户端会感知为短暂停顿后快速输出，而非稳定逐 token 输出。
          对大多数聊天 UI 这是可接受的；若必须使用真实 token 级流式，
          可替换为 ``generate_chat_stream``，并在流式解析器中处理工具调用检测。
        """
        for start in range(0, len(text), _TOKEN_CHUNK_SIZE):
            chunk = text[start : start + _TOKEN_CHUNK_SIZE]
            token_payload = json.dumps({"text": chunk}, ensure_ascii=False)
            yield f"event: token\ndata: {token_payload}\n\n"
            await asyncio.sleep(0)

        answer_payload = json.dumps({"text": text}, ensure_ascii=False)
        yield f"event: answer\ndata: {answer_payload}\n\n"

    # ── 对外 API ────────────────────────────────────────────────────────

    async def run(
        self,
        query: str,
        history: list[Message] | None = None,
        temperature: float | None = None,
        trace_id: str | None = None,
    ) -> AgentResult:
        trace_id = trace_id or str(uuid.uuid4())
        answer = ""
        iterations = 0
        citations: list[Citation] = []
        processing_stages: dict[str, float] = {}
        steps: list[AgentStep] = []

        async for event in self._react_core(query, history, temperature, trace_id):
            if isinstance(event, _FinalAnswer):
                answer = event.text
                iterations = event.iteration + 1
                citations = event.citations
                processing_stages = event.processing_stages
                steps = event.steps
            elif isinstance(event, _ForcedAnswer):
                answer = event.text
                iterations = self._max_iterations
                processing_stages = event.processing_stages
                steps = event.steps

        return AgentResult(
            answer=answer,
            steps=steps,
            iterations=iterations,
            citations=citations,
            processing_stages=processing_stages,
            trace_id=trace_id,
        )

    async def run_stream(
        self,
        query: str,
        history: list[Message] | None = None,
        temperature: float | None = None,
        trace_id: str | None = None,
    ) -> AsyncGenerator[str]:
        """以 SSE 格式事件流式输出 Agent 活动。

        使用 ``generate_chat``（非流式）进行工具调用检测，并切分缓存响应
        来输出逐 token 效果，避免旧实现所需的额外 LLM 调用。
        """
        trace_id = trace_id or str(uuid.uuid4())

        yield (
            f"event: start\n"
            f"data: {json.dumps({'trace_id': trace_id}, ensure_ascii=False)}\n\n"
        )

        async for event in self._react_core(query, history, temperature, trace_id):
            if isinstance(event, _ToolCallEvent):
                tc_payload = {"tools": event.tool_names, "iteration": event.iteration}
                yield (
                    f"event: tool_call\n"
                    f"data: {json.dumps(tc_payload, ensure_ascii=False)}\n\n"
                )

            elif isinstance(event, _ToolResultEvent):
                payload = {
                    "tool": event.tool_name,
                    "result_len": len(event.result),
                    "duration_ms": event.duration_ms,
                }
                yield (
                    f"event: tool_result\n"
                    f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                )

            elif isinstance(event, (_FinalAnswer, _ForcedAnswer)):
                async for token_event in self._yield_tokens(event.text):
                    yield token_event

    # ── 引用提取 ────────────────────────────────────────────────────────

    @staticmethod
    def _extract_citations(text: str) -> list[Citation]:
        """从最终答案中提取引用标记 [N]。"""
        found: set[int] = set()
        for match in re.finditer(r"\[(\d+)\]", text):
            found.add(int(match.group(1)))
        return [Citation(index=idx) for idx in sorted(found)]


_GENERATE_CASE_KEYWORDS = ("生成", "设计", "产出")
_EXECUTE_CASE_KEYWORDS = ("执行", "运行", "回放", "跑")
_AUTOMATION_CASE_KEYWORDS = ("自动化", "automation", "case")
# 数量/类型过滤限定词：命中时说明用户想批量执行（前 N 条 / 跳过某类用例），
# 这类约束无法用硬路由的"只透传路径"表达，需要交给 LLM 换算成
# execute_scenario 的 max_cases/exclude_types/case_ids 参数。
_BATCH_QUALIFIER_PATTERN = re.compile(r"\d+\s*条|回归|跳过|只(跑|执行|运行|测)")


def _has_batch_qualifier(query: str) -> bool:
    """检测用户指令中是否包含批量执行的数量/类型限定词。"""
    return bool(_BATCH_QUALIFIER_PATTERN.search(query or ""))


def _detect_json_route(text: str) -> str | None:
    """根据用户意图和 JSON 文件类型判断应直达哪个工具。"""
    query = (text or "").strip()
    if not query:
        return None

    path_match = re.search(r"(?:/|\./|\.\./)[^\s]+\.json", query)
    if path_match is None:
        return None

    path = Path(path_match.group(0))
    filename = path.name.lower()
    has_generate_intent = any(keyword in query for keyword in _GENERATE_CASE_KEYWORDS)
    has_execute_intent = any(keyword in query for keyword in _EXECUTE_CASE_KEYWORDS)

    if _looks_like_automation_json(filename) and has_execute_intent:
        return "execute_scenario"
    if _looks_like_analysis_json(filename) and has_generate_intent:
        return "design_test_cases"
    return None


def _looks_like_analysis_json(filename: str) -> bool:
    """req_graph/需求分析 JSON 用于 design_test_cases。"""
    return filename.endswith("_req_graph.json") or "需求分析" in filename


def _looks_like_automation_json(filename: str) -> bool:
    """automation JSON 用于 execute_scenario。"""
    return "automation" in filename or filename.endswith("_scenario.json")


def _is_automation_case_request(text: str) -> bool:
    """判断当前请求是否明确指向自动化用例。"""
    lowered = text.lower()
    return any(keyword in text or keyword in lowered for keyword in _AUTOMATION_CASE_KEYWORDS)


def _build_cli_final_tool_call(query: str) -> ToolCall | None:
    """CLI 澄清完成后直達 analyze_requirement final，不经过 LLM 选工具。"""
    payload = parse_cli_final_payload(query)
    if payload is None:
        return None
    answers = str(payload.get("clarification_answers", "")).strip()
    if not answers:
        return None
    arguments: dict[str, str] = {
        "analysis_mode": "final",
        "clarification_answers": answers,
    }
    for key in ("requirement", "requirement_file", "module", "output_dir"):
        value = str(payload.get(key, "") or "").strip()
        if value:
            arguments[key] = value
    return ToolCall(
        id=f"direct-cli-final-{uuid.uuid4()}",
        name="analyze_requirement",
        arguments=arguments,
    )


def _build_cli_design_cases_tool_call(query: str) -> ToolCall | None:
    """CLI 在 confirmed JSON 后直達 design_test_cases，不经过 LLM 选工具。"""
    payload = parse_cli_design_cases_payload(query)
    if payload is None:
        return None
    path = str(payload.get("analysis_json_path", "")).strip()
    if not path:
        return None
    mode = str(payload.get("generation_mode", "manual")).strip().lower()
    if mode not in {"manual", "automation"}:
        mode = "manual"
    arguments: dict[str, str] = {
        "analysis_json_path": path,
        "generation_mode": mode,
    }
    module = str(payload.get("module", "") or "").strip()
    if module:
        arguments["module"] = module
    return ToolCall(
        id=f"direct-cli-design-{uuid.uuid4()}",
        name="design_test_cases",
        arguments=arguments,
    )
