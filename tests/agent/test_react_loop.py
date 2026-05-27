"""Tests for ReActAgent loop logic."""

import pytest

from src.agent.react_loop import ReActAgent
from src.llm.types import ChatResponse, ToolCall
from tests.fakes import FakeLLM


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def fake_tool():
    """A minimal tool that echoes back its argument."""
    from src.agent.base_tool import BaseTool

    class EchoTool(BaseTool):
        @property
        def name(self) -> str:
            return "echo"

        @property
        def description(self) -> str:
            return "Echo back the message."

        @property
        def parameters(self) -> dict:
            return {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "The message to echo"},
                },
                "required": ["message"],
            }

        async def execute(self, message: str = "", **kwargs) -> str:
            return f"ECHO: {message}"

    return EchoTool()


class TestReActAgent:
    async def test_direct_answer_no_tools(self, fake_llm):
        """Agent answers directly when LLM returns text."""
        fake_llm.response_text = "你是ReAct Agent吗？不，我是普通回答。"
        agent = ReActAgent(llm=fake_llm, tools=[])

        result = await agent.run(query="你好")

        assert result.iterations == 1
        assert "回答" in result.answer
        assert len(result.steps) == 1
        assert result.steps[0].thinking != ""

    async def test_single_tool_call_then_answer(self, fake_llm, fake_tool):
        """Agent calls a tool once, then answers."""
        fake_llm._responses = [
            ChatResponse(
                content="",
                model="fake",
                stop_reason="tool_use",
                tool_calls=[ToolCall(id="t1", name="echo", arguments={"message": "hello"})],
            ),
            ChatResponse(
                content="工具返回了 'ECHO: hello'，所以答案是 hello。",
                model="fake",
                stop_reason="end_turn",
                tool_calls=[],
            ),
        ]
        agent = ReActAgent(llm=fake_llm, tools=[fake_tool])

        result = await agent.run(query="echo hello")

        assert result.iterations == 2
        assert len(result.steps) == 2
        assert result.steps[0].tool_call is not None
        assert result.steps[0].tool_call.name == "echo"
        assert result.steps[0].tool_result == "ECHO: hello"
        assert result.steps[1].thinking

    async def test_max_iterations_force_answer(self, fake_llm, fake_tool):
        """Agent forces an answer after max_iterations."""
        responses = [
            ChatResponse(
                content="",
                model="fake",
                stop_reason="tool_use",
                tool_calls=[ToolCall(id=f"t{i}", name="echo", arguments={"message": "x"})],
            )
            for i in range(2)
        ]
        responses.append(ChatResponse(
            content="已达到最大迭代次数，强制总结。",
            model="fake",
            stop_reason="end_turn",
        ))
        fake_llm._responses = responses
        fake_llm._response_idx = 0
        agent = ReActAgent(llm=fake_llm, tools=[fake_tool], max_iterations=3)

        result = await agent.run(query="loop")

        assert result.iterations == 3
        assert "总结" in result.answer

    async def test_citation_extraction(self, fake_llm):
        """Agent extracts [N] markers from the answer."""
        fake_llm.response_text = "根据 [1] 和 [3] 的资料，答案是 X。"
        agent = ReActAgent(llm=fake_llm, tools=[])

        result = await agent.run(query="测试")

        assert len(result.citations) == 2
        indices = [c["index"] for c in result.citations]
        assert indices == [1, 3]

    async def test_stream_events_direct_answer(self, fake_llm):
        """Streaming emits an answer event when LLM answers directly."""
        fake_llm.response_text = "流式答案"
        agent = ReActAgent(llm=fake_llm, tools=[])

        events = []
        async for event in agent.run_stream(query="test"):
            events.append(event)

        assert len(events) == 1
        assert "event: answer" in events[0]
        assert "流式答案" in events[0]

    async def test_tool_execution_error_handled(self, fake_llm):
        """Agent handles tool execution errors gracefully."""
        from src.agent.base_tool import BaseTool

        class FailingTool(BaseTool):
            @property
            def name(self): return "failing"
            @property
            def description(self): return "Always fails."
            @property
            def parameters(self): return {"type": "object", "properties": {}}
            async def execute(self, **kwargs):
                raise RuntimeError("工具内部错误")

        fake_llm._responses = [
            ChatResponse(
                content="",
                model="fake",
                stop_reason="tool_use",
                tool_calls=[ToolCall(id="t1", name="failing", arguments={})],
            ),
            ChatResponse(
                content="工具调用失败了，但我会尽力回答。",
                model="fake",
                stop_reason="end_turn",
            ),
        ]
        agent = ReActAgent(llm=fake_llm, tools=[FailingTool()])

        result = await agent.run(query="test")

        assert result.iterations == 2
        assert "工具内部错误" in result.steps[0].tool_result

    # ── A2: parallel tool calls ──────────────────────────────────────────────

    async def test_parallel_tool_calls_both_execute(self, fake_llm, fake_tool):
        """When LLM returns multiple tool_calls in one response, all execute."""
        fake_llm._responses = [
            ChatResponse(
                content="",
                model="fake",
                stop_reason="tool_use",
                tool_calls=[
                    ToolCall(id="t1", name="echo", arguments={"message": "first"}),
                    ToolCall(id="t2", name="echo", arguments={"message": "second"}),
                ],
            ),
            ChatResponse(
                content="两次工具调用的综合答案",
                model="fake",
                stop_reason="end_turn",
            ),
        ]
        agent = ReActAgent(llm=fake_llm, tools=[fake_tool])

        result = await agent.run(query="call two tools")

        assert result.iterations == 2
        tool_steps = [s for s in result.steps if s.tool_call is not None]
        assert len(tool_steps) == 2
        results = {s.tool_result for s in tool_steps}
        assert "ECHO: first" in results
        assert "ECHO: second" in results
        assert "综合" in result.answer

    # ── A3: multi-turn multi-tool ────────────────────────────────────────────

    async def test_multi_turn_multi_tool_sequence(self, fake_llm, fake_tool):
        """Agent correctly chains two separate tool-call rounds then answers."""
        fake_llm._responses = [
            ChatResponse(
                content="",
                model="fake",
                stop_reason="tool_use",
                tool_calls=[ToolCall(id="t1", name="echo", arguments={"message": "round1"})],
            ),
            ChatResponse(
                content="",
                model="fake",
                stop_reason="tool_use",
                tool_calls=[ToolCall(id="t2", name="echo", arguments={"message": "round2"})],
            ),
            ChatResponse(
                content="综合两轮工具结果的最终答案",
                model="fake",
                stop_reason="end_turn",
            ),
        ]
        agent = ReActAgent(llm=fake_llm, tools=[fake_tool])

        result = await agent.run(query="multi-round")

        assert result.iterations == 3
        tool_steps = [s for s in result.steps if s.tool_call is not None]
        assert len(tool_steps) == 2
        assert tool_steps[0].tool_result == "ECHO: round1"
        assert tool_steps[1].tool_result == "ECHO: round2"
        assert "综合" in result.answer

    # ── A4: streaming with tool events ──────────────────────────────────────

    async def test_stream_emits_tool_call_and_result_events(self, fake_llm, fake_tool):
        """Streaming emits tool_call, tool_result, and answer events in correct order."""
        fake_llm._responses = [
            ChatResponse(
                content="",
                model="fake",
                stop_reason="tool_use",
                tool_calls=[ToolCall(id="t1", name="echo", arguments={"message": "streaming"})],
            ),
            ChatResponse(
                content="流式最终答案",
                model="fake",
                stop_reason="end_turn",
            ),
        ]
        agent = ReActAgent(llm=fake_llm, tools=[fake_tool])

        events: list[str] = []
        async for event in agent.run_stream(query="stream test"):
            events.append(event)

        event_types = [e.split("\n")[0] for e in events]
        assert "event: tool_call" in event_types
        assert "event: tool_result" in event_types
        assert "event: answer" in event_types
        # Ordering: tool_call → tool_result → answer
        assert event_types.index("event: tool_call") < event_types.index("event: answer")
        assert event_types.index("event: tool_result") < event_types.index("event: answer")

    # ── A5: unknown tool name ────────────────────────────────────────────────

    async def test_unknown_tool_name_captured_in_step_result(self, fake_llm):
        """If LLM calls a non-existent tool, the error is captured without raising."""
        fake_llm._responses = [
            ChatResponse(
                content="",
                model="fake",
                stop_reason="tool_use",
                tool_calls=[ToolCall(id="t1", name="ghost_tool", arguments={})],
            ),
            ChatResponse(
                content="工具不存在，直接回答。",
                model="fake",
                stop_reason="end_turn",
            ),
        ]
        agent = ReActAgent(llm=fake_llm, tools=[])  # no tools registered

        result = await agent.run(query="call nonexistent")

        assert result.iterations == 2
        assert result.steps[0].tool_call is not None
        assert "工具执行错误" in result.steps[0].tool_result
        assert result.steps[1].thinking == "工具不存在，直接回答。"
