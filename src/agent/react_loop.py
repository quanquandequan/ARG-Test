"""ReAct Agent — Think-Act-Observe loop with tool calling."""

import json
import re
from typing import AsyncIterator

from src.agent.base_tool import BaseTool
from src.agent.tool_registry import ToolRegistry
from src.agent.types import AgentResult, AgentStep
from src.core.logging import get_logger
from src.llm.base import BaseLLM
from src.llm.types import ChatResponse, ContentBlock, Message, ToolCall

logger = get_logger(__name__)

_SYSTEM_PROMPT = """你是一个智能知识库助手，可以通过调用工具来获取信息和回答问题。为叭嗒、漫画、小程序、插件等产品功能提供技术支持，根据提供的稳定内容回答问题。

## 可用工具
- **knowledge_search**: 在知识库中搜索文档，自动完成检索和精排，返回最相关的片段及来源编号。当用户问题需要专业知识库信息时，这是首选工具。
- **web_search**: 搜索互联网获取实时信息。仅当知识库无法回答时使用。

## 信息来源优先级
每个文档片段会标注来源文件名，请据此判断权威性：
- **ACN_cases.xlsx**：主要来源，包含测试用例和现有应用逻辑，是产品行为的权威定义。
- **ACN_buglist.xlsx**：缺陷记录，反映已知问题和历史 Bug，用于辅助判断。
- **以"叭嗒"、"基线"、"后端"等开头的 .xmind 文件**：产品需求脑图，用于补充背景和上下文。
- **其他 .xmind 文件**：辅助参考。
当不同来源信息冲突时，按以上优先级采纳。

## 工作流程
1. 分析用户问题，确定需要哪些信息
2. 如有必要，调用 knowledge_search 搜索知识库
3. 如果知识库无法回答，尝试 web_search
4. 综合所有信息，给出完整答案

## 回答规则
- 必须引用信息来源，使用 [1]、[2] 等编号
- 如果所有工具都无法找到答案，明确告知"根据目前的信息无法回答"
- 答案应简洁、准确、结构化
- 使用中文回答"""

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

    async def run(
        self,
        query: str,
        history: list[Message] | None = None,
        temperature: float | None = None,
    ) -> AgentResult:
        messages: list[Message] = [
            Message(role="system", content=self._system_prompt),
        ]
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
                # ── Act ──
                tool_count = len(response.tool_calls)
                logger.info(
                    "agent_tool_calls",
                    iteration=i,
                    tools=[tc.name for tc in response.tool_calls],
                )

                # Add assistant message with tool_calls
                assistant_msg = Message(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
                messages.append(assistant_msg)

                # Execute tools in parallel
                for tc in response.tool_calls:
                    try:
                        tool = self._registry.get(tc.name)
                        result = await tool.execute(**tc.arguments)
                    except Exception as e:
                        result = f"工具执行错误: {e}"
                        logger.warning(
                            "tool_execution_error",
                            tool=tc.name,
                            error=str(e),
                        )

                    steps.append(AgentStep(
                        step_index=i,
                        tool_call=tc,
                        tool_result=result,
                    ))

                    # Add tool result message
                    messages.append(Message(
                        role="tool",
                        content=result,
                        tool_call_id=tc.id,
                        name=tc.name,
                    ))

            else:
                # ── Answer ──
                steps.append(AgentStep(step_index=i, thinking=response.content))
                citations = self._extract_citations(response.content)
                logger.info("agent_answer", iterations=i + 1, citations=len(citations))
                return AgentResult(
                    answer=response.content,
                    steps=steps,
                    iterations=i + 1,
                    citations=citations,
                )

        # Max iterations exceeded — force answer
        logger.warning("agent_max_iterations", max=self._max_iterations)
        messages.append(Message(
            role="user",
            content="请综合以上所有工具的结果，给出最终答案。如果信息不足，请说明。",
        ))
        response = await self._llm.generate_chat(
            messages=messages,
            tools=None,
            temperature=temperature,
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
        """Stream agent activity as SSE-like events."""
        messages: list[Message] = [
            Message(role="system", content=self._system_prompt),
        ]
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
                yield f"event: tool_call\ndata: {json.dumps({'tools': [tc.name for tc in response.tool_calls]}, ensure_ascii=False)}\n\n"

                messages.append(Message(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                ))

                for tc in response.tool_calls:
                    try:
                        tool = self._registry.get(tc.name)
                        result = await tool.execute(**tc.arguments)
                    except Exception as e:
                        result = f"工具执行错误: {e}"

                    yield f"event: tool_result\ndata: {json.dumps({'tool': tc.name, 'result_len': len(result)}, ensure_ascii=False)}\n\n"

                    messages.append(Message(
                        role="tool",
                        content=result,
                        tool_call_id=tc.id,
                        name=tc.name,
                    ))

            else:
                yield f"event: answer\ndata: {json.dumps({'text': response.content}, ensure_ascii=False)}\n\n"
                return

        yield f"event: error\ndata: {json.dumps({'message': '超过最大迭代次数'}, ensure_ascii=False)}\n\n"

    @staticmethod
    def _extract_citations(text: str) -> list[dict]:
        """Extract citation markers [N] from the answer."""
        found: set[int] = set()
        for match in re.finditer(r"\[(\d+)\]", text):
            found.add(int(match.group(1)))
        return [{"index": idx} for idx in sorted(found)]
