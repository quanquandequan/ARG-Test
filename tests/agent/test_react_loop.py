"""Tests for ReActAgent loop logic."""

import pytest

from src.agent.react_loop import ReActAgent
from src.llm.types import ChatResponse, ToolCall
from tests.fakes import FakeLLM


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


@pytest.fixture
def fake_llm():
    return FakeLLM()


class TestReActAgent:
    async def test_direct_answer_no_tools(self, fake_llm):
        """Agent answers directly when LLM returns text."""
        fake_llm.response_text = "你是ReAct Agent吗？不，我是普通回答。"
        agent = ReActAgent(llm=fake_llm, tools=[])

        result = await agent.run(query="你好")

        assert result.iterations == 1
        assert "回答" in result.answer
        assert len(result.steps) == 1
        assert result.steps[0].thinking != ""  # Verify thinking was captured

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
        # Step 0: tool call
        assert result.steps[0].tool_call is not None
        assert result.steps[0].tool_call.name == "echo"
        assert result.steps[0].tool_result == "ECHO: hello"
        # Step 1: answer
        assert result.steps[1].thinking

    async def test_max_iterations_force_answer(self, fake_llm, fake_tool):
        """Agent forces an answer after max_iterations."""
        # Always return tool_calls to force max_iterations
        responses = [
            ChatResponse(
                content="",
                model="fake",
                stop_reason="tool_use",
                tool_calls=[ToolCall(id=f"t{i}", name="echo", arguments={"message": "x"})],
            )
            for i in range(2)
        ]
        # Final forced answer
        responses.append(ChatResponse(
            content="已达到最大迭代次数，强制总结。",
            model="fake",
            stop_reason="end_turn",
        ))
        fake_llm._responses = responses
        agent = ReActAgent(llm=fake_llm, tools=[fake_tool], max_iterations=3)

        # Reset to make the final response work
        fake_llm._response_idx = 0
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

    async def test_stream_events(self, fake_llm):
        """Streaming emits proper SSE events."""
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
