"""ReAct Agent — Think-Act-Observe loop with tool calling.

Both ``run()`` (non-streaming) and ``run_stream()`` (SSE streaming) share a
single ``_react_core()`` async generator that yields typed event dataclasses.
This eliminates the previous ~80% code duplication and the double-LLM-call
in the streaming path (the final answer is now chunked from the cached
``generate_chat`` response instead of calling ``generate_chat_stream`` again).
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass

from src.agent.base_tool import BaseTool
from src.agent.history import truncate_history
from src.agent.tool_registry import ToolRegistry
from src.agent.types import AgentResult, AgentStep, Citation
from src.core.logging import get_logger
from src.llm.base import BaseLLM
from src.llm.types import Message, ToolCall

logger = get_logger(__name__)

# ── Internal event types yielded by _react_core ───────────────────────────


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

# Approximate character count per SSE token chunk.
_TOKEN_CHUNK_SIZE = 20


class ReActAgent:
    """ReAct-pattern Agent: Think -> Act -> Observe -> Repeat -> Answer."""

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

    # ── Message building ─────────────────────────────────────────────────

    def _build_messages(self, history: list[Message] | None, query: str) -> list[Message]:
        """Assemble system + truncated history + current user query."""
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

    # ── Tool execution helper ────────────────────────────────────────────

    async def _safe_execute(
        self, tc: ToolCall, trace_id: str
    ) -> tuple[ToolCall, str, float]:
        """Execute one tool call, catch any exception, return (tc, result, ms)."""
        t0 = time.perf_counter()
        try:
            tool = self._registry.get(tc.name)
            result = await tool.execute(**tc.arguments)
        except Exception as e:
            result = f"工具执行错误: {e}"
            logger.warning(
                "tool_execution_error",
                trace_id=trace_id,
                tool=tc.name,
                error=str(e),
            )
        dur = (time.perf_counter() - t0) * 1000
        return tc, result, dur

    # ── Core ReAct loop (shared by run and run_stream) ───────────────────

    async def _react_core(
        self,
        query: str,
        history: list[Message] | None = None,
        temperature: float | None = None,
        trace_id: str = "",
    ) -> AsyncGenerator[_Event]:
        """Unified ReAct loop that yields typed events.

        Both ``run()`` and ``run_stream()`` consume these events, eliminating
        duplicated loop logic and the double-LLM-call in the streaming path.
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
                # ── Tool call iteration ──
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

                # Execute all tool calls concurrently
                tc_results = list(await asyncio.gather(
                    *[self._safe_execute(tc, trace_id) for tc in response.tool_calls]
                ))

                for tc, result, dur_ms in tc_results:
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
                    )

            else:
                # ── Final answer ──
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

        # Max iterations exceeded — force a final answer without tools
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

    # ── Token streaming helper ───────────────────────────────────────────

    @staticmethod
    async def _yield_tokens(text: str) -> AsyncGenerator[str]:
        """Yield the buffered LLM response as SSE token events.

        **Simulated streaming**: this method does NOT call the LLM again.
        Instead it splits the already-received ``generate_chat`` response into
        fixed-size chunks (``_TOKEN_CHUNK_SIZE`` chars) and yields them with an
        ``asyncio.sleep(0)`` between each chunk to let the event loop breathe.

        Trade-offs vs. true ``generate_chat_stream``:
        - ✅ Single LLM call per iteration — no duplicate cost or latency.
        - ✅ Tool-call vs. final-answer detection is reliable (full response).
        - ⚠️  Chunks arrive at network speed, not at LLM token-generation speed.
          The client perceives a short pause then a burst, rather than a steady
          per-token drip.  For most chat UIs this is acceptable; if true
          token-level streaming is required, swap this for ``generate_chat_stream``
          and handle tool-call detection in the streaming parser.
        """
        for start in range(0, len(text), _TOKEN_CHUNK_SIZE):
            chunk = text[start : start + _TOKEN_CHUNK_SIZE]
            token_payload = json.dumps({"text": chunk}, ensure_ascii=False)
            yield f"event: token\ndata: {token_payload}\n\n"
            await asyncio.sleep(0)

        answer_payload = json.dumps({"text": text}, ensure_ascii=False)
        yield f"event: answer\ndata: {answer_payload}\n\n"

    # ── Public API ───────────────────────────────────────────────────────

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
        """Stream agent activity as SSE-format events.

        Uses ``generate_chat`` (non-streaming) for tool-call detection and
        chunks the cached response for token-by-token output, avoiding the
        extra LLM call that the previous implementation required.
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

    # ── Citation extraction ──────────────────────────────────────────────

    @staticmethod
    def _extract_citations(text: str) -> list[Citation]:
        """Extract citation markers [N] from the final answer."""
        found: set[int] = set()
        for match in re.finditer(r"\[(\d+)\]", text):
            found.add(int(match.group(1)))
        return [Citation(index=idx) for idx in sorted(found)]
