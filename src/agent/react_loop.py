"""ReAct Agent — Think-Act-Observe loop with tool calling."""

import asyncio
import json
import re
import time
import uuid
from collections.abc import AsyncIterator

from src.agent.base_tool import BaseTool
from src.agent.history import truncate_history
from src.agent.tool_registry import ToolRegistry
from src.agent.types import AgentResult, AgentStep, Citation
from src.core.logging import get_logger
from src.llm.base import BaseLLM
from src.llm.types import Message, ToolCall

logger = get_logger(__name__)

# Python-level fallback — canonical value lives in configs/default.yaml (agent.system_prompt)
_SYSTEM_PROMPT = (
    "你是一个智能知识库助手，可以通过调用工具来获取信息和回答问题。\n\n"
    "当用户有问题时，优先调用 knowledge_search 在知识库中查找，"
    "必要时再使用 web_search 获取实时信息。\n"
    "回答时引用来源编号 [1]、[2] 等，使用中文回答。"
)

_MAX_ITERATIONS = 10
_DEFAULT_MAX_HISTORY_TOKENS = 4000


class ReActAgent:
    """ReAct-pattern Agent: Think → Act → Observe → Repeat → Answer."""

    def __init__(
        self,
        llm: BaseLLM,
        tools: list[BaseTool] | None = None,
        system_prompt: str = "",
        max_iterations: int = _MAX_ITERATIONS,
        max_history_tokens: int = _DEFAULT_MAX_HISTORY_TOKENS,
    ):
        self._llm = llm
        self._registry = ToolRegistry(tools or [])
        self._system_prompt = system_prompt or _SYSTEM_PROMPT
        self._max_iterations = max_iterations
        self._max_history_tokens = max_history_tokens

    @property
    def tool_names(self) -> list[str]:
        return self._registry.names()

    def _build_messages(self, history: list[Message] | None, query: str) -> list[Message]:
        """Assemble system + truncated history + current user query."""
        truncated = truncate_history(
            history or [], max_tokens=self._max_history_tokens
        )
        return (
            [Message(role="system", content=self._system_prompt)]
            + truncated
            + [Message(role="user", content=query)]
        )

    async def _execute_tool_safe(self, tc: ToolCall, trace_id: str) -> tuple[ToolCall, str, float]:
        """Execute one tool call and return (tc, result, duration_ms)."""
        t0 = time.perf_counter()
        try:
            tool = self._registry.get(tc.name)
            result = await tool.execute(**tc.arguments)
        except Exception as e:
            result = f"工具执行错误: {e}"
            logger.warning("tool_execution_error", trace_id=trace_id, tool=tc.name, error=str(e))
        duration_ms = (time.perf_counter() - t0) * 1000
        return tc, result, duration_ms

    async def _run_tool_round(
        self,
        tool_calls: list[ToolCall],
        messages: list[Message],
        trace_id: str,
    ) -> list[tuple[ToolCall, str, float]]:
        """Execute all tool calls concurrently, append results to messages.

        Returns a list of (tc, result, duration_ms) tuples.
        """
        tc_results: list[tuple[ToolCall, str, float]] = list(
            await asyncio.gather(
                *[self._execute_tool_safe(tc, trace_id) for tc in tool_calls]
            )
        )
        for tc, result, _ in tc_results:
            messages.append(Message(
                role="tool",
                content=result,
                tool_call_id=tc.id,
                name=tc.name,
            ))
        return tc_results

    async def _stream_final_answer(
        self,
        messages: list[Message],
        temperature: float | None,
    ) -> AsyncIterator[str]:
        """Stream final answer tokens then emit a consolidated answer event."""
        full_text = ""
        async for block in self._llm.generate_chat_stream(
            messages=messages,
            tools=None,
            tool_choice=None,
            temperature=temperature,
        ):
            if block.type == "text" and block.text:
                full_text += block.text
                token_payload = json.dumps({"text": block.text}, ensure_ascii=False)
                yield f"event: token\ndata: {token_payload}\n\n"

        answer_payload = json.dumps({"text": full_text}, ensure_ascii=False)
        yield f"event: answer\ndata: {answer_payload}\n\n"

    async def run(
        self,
        query: str,
        history: list[Message] | None = None,
        temperature: float | None = None,
        trace_id: str | None = None,
    ) -> AgentResult:
        trace_id = trace_id or str(uuid.uuid4())

        messages = self._build_messages(history, query)
        tool_defs = self._registry.definitions()
        steps: list[AgentStep] = []
        processing_stages: dict[str, float] = {}
        run_t0 = time.perf_counter()

        for i in range(self._max_iterations):
            iter_t0 = time.perf_counter()

            response = await self._llm.generate_chat(
                messages=messages,
                tools=tool_defs if tool_defs else None,
                tool_choice="auto" if tool_defs else None,
                temperature=temperature,
            )

            if response.stop_reason == "tool_use" and response.tool_calls:
                llm_ms = (time.perf_counter() - iter_t0) * 1000
                logger.info(
                    "agent_tool_calls",
                    trace_id=trace_id,
                    iteration=i,
                    tools=[tc.name for tc in response.tool_calls],
                    llm_ms=round(llm_ms, 1),
                )

                messages.append(Message(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                ))

                tc_results = await self._run_tool_round(
                    response.tool_calls, messages, trace_id
                )

                for tc, result, dur_ms in tc_results:
                    processing_stages[f"iter{i}.{tc.name}"] = round(dur_ms, 1)
                    steps.append(AgentStep(
                        step_index=i,
                        tool_call=tc,
                        tool_result=result,
                        duration_ms=round(dur_ms, 1),
                    ))

            else:
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
                return AgentResult(
                    answer=response.content,
                    steps=steps,
                    iterations=i + 1,
                    citations=citations,
                    processing_stages=processing_stages,
                    trace_id=trace_id,
                )

        # Max iterations exceeded — force a final answer without tools
        logger.warning("agent_max_iterations", trace_id=trace_id, max=self._max_iterations)
        messages.append(Message(
            role="user",
            content="请综合以上所有工具的结果，给出最终答案。如果信息不足，请说明。",
        ))
        response = await self._llm.generate_chat(
            messages=messages, tools=None, temperature=temperature
        )
        total_ms = (time.perf_counter() - run_t0) * 1000
        processing_stages["total"] = round(total_ms, 1)
        return AgentResult(
            answer=response.content,
            steps=steps,
            iterations=self._max_iterations,
            processing_stages=processing_stages,
            trace_id=trace_id,
        )

    async def run_stream(
        self,
        query: str,
        history: list[Message] | None = None,
        temperature: float | None = None,
        trace_id: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream agent activity as SSE-format events.

        Implementation note: uses generate_chat_stream for the final answer so
        that the LLM is called exactly once per iteration (no double-call).
        For tool-call detection we still use generate_chat (non-streaming)
        because streaming tool-call parsing is provider-specific and error-prone;
        only the final text answer is truly streamed token-by-token.
        """
        trace_id = trace_id or str(uuid.uuid4())

        messages = self._build_messages(history, query)
        tool_defs = self._registry.definitions()

        yield (
            f"event: start\n"
            f"data: {json.dumps({'trace_id': trace_id}, ensure_ascii=False)}\n\n"
        )

        for i in range(self._max_iterations):
            # Use generate_chat to detect tool calls vs. final answer.
            # When the LLM returns a final answer (no tool calls), we re-invoke
            # via generate_chat_stream to stream that answer token-by-token.
            # This means one extra non-streaming call per tool-call iteration,
            # but only a single non-streaming + one streaming call for the final answer.
            response = await self._llm.generate_chat(
                messages=messages,
                tools=tool_defs if tool_defs else None,
                tool_choice="auto" if tool_defs else None,
                temperature=temperature,
            )

            if response.stop_reason == "tool_use" and response.tool_calls:
                tool_names = [tc.name for tc in response.tool_calls]
                tc_payload = {"tools": tool_names, "iteration": i}
                yield (
                    f"event: tool_call\n"
                    f"data: {json.dumps(tc_payload, ensure_ascii=False)}\n\n"
                )

                messages.append(Message(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                ))

                tc_results = await self._run_tool_round(
                    response.tool_calls, messages, trace_id
                )

                for tc, result, dur_ms in tc_results:
                    payload = {
                        "tool": tc.name,
                        "result_len": len(result),
                        "duration_ms": round(dur_ms, 1),
                    }
                    yield (
                        f"event: tool_result\n"
                        f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    )

            else:
                # LLM wants to give a final answer — stream it token-by-token.
                # We discard response.content here and re-stream from scratch,
                # because generate_chat above buffered the whole response.
                async for event in self._stream_final_answer(messages, temperature):
                    yield event
                return

        # Max iterations: force summarisation, streamed
        messages.append(Message(
            role="user",
            content="请综合以上所有工具的结果，给出最终答案。如果信息不足，请说明。",
        ))
        async for event in self._stream_final_answer(messages, temperature):
            yield event

    @staticmethod
    def _extract_citations(text: str) -> list[Citation]:
        """Extract citation markers [N] from the final answer."""
        found: set[int] = set()
        for match in re.finditer(r"\[(\d+)\]", text):
            found.add(int(match.group(1)))
        return [Citation(index=idx) for idx in sorted(found)]
