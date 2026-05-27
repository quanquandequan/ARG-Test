"""ReAct Agent — Think-Act-Observe loop with tool calling."""

import asyncio
import json
import re
from collections.abc import AsyncIterator

from src.agent.base_tool import BaseTool
from src.agent.tool_registry import ToolRegistry
from src.agent.types import AgentResult, AgentStep
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


class ReActAgent:
    """ReAct-pattern Agent: Think → Act → Observe → Repeat → Answer."""

    def __init__(
        self,
        llm: BaseLLM,
        tools: list[BaseTool] | None = None,
        system_prompt: str = "",
        max_iterations: int = _MAX_ITERATIONS,
    ):
        self._llm = llm
        self._registry = ToolRegistry(tools or [])
        self._system_prompt = system_prompt or _SYSTEM_PROMPT
        self._max_iterations = max_iterations

    @property
    def tool_names(self) -> list[str]:
        return self._registry.names()

    async def _execute_tool_safe(self, tc: ToolCall) -> tuple[ToolCall, str]:
        """Execute one tool call, capturing any exception as a result string."""
        try:
            tool = self._registry.get(tc.name)
            result = await tool.execute(**tc.arguments)
        except Exception as e:
            result = f"工具执行错误: {e}"
            logger.warning("tool_execution_error", tool=tc.name, error=str(e))
        return tc, result

    async def run(
        self,
        query: str,
        history: list[Message] | None = None,
        temperature: float | None = None,
    ) -> AgentResult:
        messages: list[Message] = [Message(role="system", content=self._system_prompt)]
        if history:
            messages.extend(history)
        messages.append(Message(role="user", content=query))

        tool_defs = self._registry.definitions()
        steps: list[AgentStep] = []

        for i in range(self._max_iterations):
            response = await self._llm.generate_chat(
                messages=messages,
                tools=tool_defs if tool_defs else None,
                tool_choice="auto" if tool_defs else None,
                temperature=temperature,
            )

            if response.stop_reason == "tool_use" and response.tool_calls:
                logger.info(
                    "agent_tool_calls",
                    iteration=i,
                    tools=[tc.name for tc in response.tool_calls],
                )

                messages.append(Message(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                ))

                # Execute all tool calls concurrently; preserve request order in results
                tc_results: list[tuple[ToolCall, str]] = list(
                    await asyncio.gather(
                        *[self._execute_tool_safe(tc) for tc in response.tool_calls]
                    )
                )

                for tc, result in tc_results:
                    steps.append(AgentStep(step_index=i, tool_call=tc, tool_result=result))
                    messages.append(Message(
                        role="tool",
                        content=result,
                        tool_call_id=tc.id,
                        name=tc.name,
                    ))

            else:
                # LLM produced a final answer
                steps.append(AgentStep(step_index=i, thinking=response.content))
                citations = self._extract_citations(response.content)
                logger.info("agent_answer", iterations=i + 1, citations=len(citations))
                return AgentResult(
                    answer=response.content,
                    steps=steps,
                    iterations=i + 1,
                    citations=citations,
                )

        # Max iterations exceeded — force a final answer without tools
        logger.warning("agent_max_iterations", max=self._max_iterations)
        messages.append(Message(
            role="user",
            content="请综合以上所有工具的结果，给出最终答案。如果信息不足，请说明。",
        ))
        response = await self._llm.generate_chat(
            messages=messages, tools=None, temperature=temperature
        )
        return AgentResult(
            answer=response.content,
            steps=steps,
            iterations=self._max_iterations,
        )

    async def run_stream(
        self,
        query: str,
        history: list[Message] | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """Stream agent activity as SSE-format events."""
        messages: list[Message] = [Message(role="system", content=self._system_prompt)]
        if history:
            messages.extend(history)
        messages.append(Message(role="user", content=query))

        tool_defs = self._registry.definitions()

        for i in range(self._max_iterations):
            response = await self._llm.generate_chat(
                messages=messages,
                tools=tool_defs if tool_defs else None,
                tool_choice="auto" if tool_defs else None,
                temperature=temperature,
            )

            if response.stop_reason == "tool_use" and response.tool_calls:
                tool_names = [tc.name for tc in response.tool_calls]
                yield (
                    f"event: tool_call\n"
                    f"data: {json.dumps({'tools': tool_names}, ensure_ascii=False)}\n\n"
                )

                messages.append(Message(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                ))

                tc_results: list[tuple[ToolCall, str]] = list(
                    await asyncio.gather(
                        *[self._execute_tool_safe(tc) for tc in response.tool_calls]
                    )
                )

                for tc, result in tc_results:
                    payload = {"tool": tc.name, "result_len": len(result)}
                    yield (
                        f"event: tool_result\n"
                        f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    )
                    messages.append(Message(
                        role="tool",
                        content=result,
                        tool_call_id=tc.id,
                        name=tc.name,
                    ))

            else:
                answer_payload = json.dumps({"text": response.content}, ensure_ascii=False)
                yield f"event: answer\ndata: {answer_payload}\n\n"
                return

        error_payload = json.dumps({"message": "超过最大迭代次数"}, ensure_ascii=False)
        yield f"event: error\ndata: {error_payload}\n\n"

    @staticmethod
    def _extract_citations(text: str) -> list[dict]:
        """Extract citation markers [N] from the final answer."""
        found: set[int] = set()
        for match in re.finditer(r"\[(\d+)\]", text):
            found.add(int(match.group(1)))
        return [{"index": idx} for idx in sorted(found)]
