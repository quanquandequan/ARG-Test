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

from src.agent.base_tool import FINAL_ANSWER_PASSTHROUGH, BaseTool
from src.agent.history import truncate_history
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
        dated_prompt = f"{self._system_prompt}\n\n当前日期: {today}"
        return (
            [Message(role="system", content=dated_prompt)]
            + truncated
            + [Message(role="user", content=query)]
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

        for i in range(self._max_iterations):
            iter_t0 = time.perf_counter()

            response = await self._llm.generate_chat(
                messages=messages,
                tools=tool_defs if use_tools else None,
                tool_choice="auto" if use_tools else None,
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
