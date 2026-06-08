"""ReActAgent 循环逻辑测试。"""

import pytest

from src.agent.base_tool import FINAL_ANSWER_PASSTHROUGH, BaseTool
from src.agent.react_loop import ReActAgent
from src.llm.types import ChatResponse, ToolCall
from tests.fakes import FakeLLM


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def fake_tool():
    """回显参数的最小工具。"""
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


class _StaticTool(BaseTool):
    """测试用的静态工具，可模拟普通工具和 passthrough 工具。"""

    def __init__(
        self,
        name: str,
        result: str,
        final_answer_mode: str = "",
    ):
        self._name = name
        self._result = result
        self.final_answer_mode = final_answer_mode

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"{self._name} 测试工具"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs) -> str:
        return self._result


class TestReActAgent:
    async def test_direct_answer_no_tools(self, fake_llm):
        """LLM 返回文本时，Agent 直接回答。"""
        fake_llm.response_text = "你是ReAct Agent吗？不，我是普通回答。"
        agent = ReActAgent(llm=fake_llm, tools=[], system_prompt="test")

        result = await agent.run(query="你好")

        assert result.iterations == 1
        assert "回答" in result.answer
        assert len(result.steps) == 1
        assert result.steps[0].thinking != ""

    async def test_single_tool_call_then_answer(self, fake_llm, fake_tool):
        """Agent 调用一次工具后回答。"""
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
        agent = ReActAgent(llm=fake_llm, tools=[fake_tool], system_prompt="test")

        result = await agent.run(query="echo hello")

        assert result.iterations == 2
        assert len(result.steps) == 2
        assert result.steps[0].tool_call is not None
        assert result.steps[0].tool_call.name == "echo"
        assert result.steps[0].tool_result == "ECHO: hello"
        assert result.steps[1].thinking

    async def test_max_iterations_force_answer(self, fake_llm, fake_tool):
        """达到 max_iterations 后，Agent 强制给出答案。"""
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
        agent = ReActAgent(llm=fake_llm, tools=[fake_tool], system_prompt="test", max_iterations=3)

        result = await agent.run(query="loop")

        assert result.iterations == 3
        assert "总结" in result.answer

    async def test_citation_extraction(self, fake_llm):
        """Agent 从答案中提取 [N] 标记。"""
        fake_llm.response_text = "根据 [1] 和 [3] 的资料，答案是 X。"
        agent = ReActAgent(llm=fake_llm, tools=[], system_prompt="test")

        result = await agent.run(query="测试")

        assert len(result.citations) == 2
        indices = [c.index for c in result.citations]
        assert indices == [1, 3]

    async def test_stream_events_direct_answer(self, fake_llm):
        """流式输出先发 token 事件，再发汇总 answer 事件。"""
        fake_llm.response_text = "流式答案"
        agent = ReActAgent(llm=fake_llm, tools=[], system_prompt="test")

        events = []
        async for event in agent.run_stream(query="test"):
            events.append(event)

        event_types = [e.split("\n")[0] for e in events]
        assert "event: token" in event_types
        assert "event: answer" in event_types
        assert event_types.index("event: token") < event_types.index("event: answer")
        assert any("流式答案" in e for e in events)

    async def test_tool_execution_error_handled(self, fake_llm):
        """Agent 可优雅处理工具执行错误。"""
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
        agent = ReActAgent(llm=fake_llm, tools=[FailingTool()], system_prompt="test")

        result = await agent.run(query="test")

        assert result.iterations == 2
        assert "工具内部错误" in result.steps[0].tool_result

    # ── A2: parallel tool calls ──────────────────────────────────────────────

    async def test_parallel_tool_calls_both_execute(self, fake_llm, fake_tool):
        """LLM 在一次响应中返回多个 tool_calls 时，会全部执行。"""
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
        agent = ReActAgent(llm=fake_llm, tools=[fake_tool], system_prompt="test")

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
        """Agent 能正确串联两轮独立工具调用后再回答。"""
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
        agent = ReActAgent(llm=fake_llm, tools=[fake_tool], system_prompt="test")

        result = await agent.run(query="multi-round")

        assert result.iterations == 3
        tool_steps = [s for s in result.steps if s.tool_call is not None]
        assert len(tool_steps) == 2
        assert tool_steps[0].tool_result == "ECHO: round1"
        assert tool_steps[1].tool_result == "ECHO: round2"
        assert "综合" in result.answer

    # ── A4: streaming with tool events ──────────────────────────────────────

    async def test_stream_emits_tool_call_and_result_events(self, fake_llm, fake_tool):
        """流式模式按正确顺序发出 tool_call、tool_result 和 answer 事件。"""
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
        agent = ReActAgent(llm=fake_llm, tools=[fake_tool], system_prompt="test")

        events: list[str] = []
        async for event in agent.run_stream(query="stream test"):
            events.append(event)

        event_types = [e.split("\n")[0] for e in events]
        assert "event: tool_call" in event_types
        assert "event: tool_result" in event_types
        assert "event: answer" in event_types
        # 顺序：tool_call → tool_result → answer
        assert event_types.index("event: tool_call") < event_types.index("event: answer")
        assert event_types.index("event: tool_result") < event_types.index("event: answer")

    # ── A5: unknown tool name ────────────────────────────────────────────────

    async def test_unknown_tool_name_captured_in_step_result(self, fake_llm):
        """LLM 调用不存在的工具时，错误会被捕获且不抛出。"""
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
        agent = ReActAgent(llm=fake_llm, tools=[], system_prompt="test")  # 未注册工具

        result = await agent.run(query="call nonexistent")

        assert result.iterations == 2
        assert result.steps[0].tool_call is not None
        assert "工具执行错误" in result.steps[0].tool_result
        assert result.steps[1].thinking == "工具不存在，直接回答。"

    async def test_passthrough_tool_result_becomes_final_answer(self, fake_llm):
        """产物型工具返回后，Agent 不再调用 LLM 二次改写。"""
        tool_result = (
            "确认版需求分析完成：动画频道-追番表Card\n"
            "F006 追番表入口跳转：模块右上角'追番表'入口，"
            "点击后跳转至完整追番表页面。"
        )
        fake_llm._responses = [
            ChatResponse(
                content="",
                model="fake",
                stop_reason="tool_use",
                tool_calls=[
                    ToolCall(
                        id="t1",
                        name="analyze_requirement",
                        arguments={},
                    )
                ],
            ),
            ChatResponse(
                content="错误总结：点击'查看全部追番表'跳转。",
                model="fake",
                stop_reason="end_turn",
            ),
        ]
        agent = ReActAgent(
            llm=fake_llm,
            tools=[
                _StaticTool(
                    "analyze_requirement",
                    tool_result,
                    FINAL_ANSWER_PASSTHROUGH,
                )
            ],
            system_prompt="test",
        )

        result = await agent.run(query="分析需求")

        assert result.iterations == 1
        assert result.answer == tool_result
        assert "查看全部追番表" not in result.answer
        assert fake_llm._response_idx == 1

    async def test_design_and_execution_tools_passthrough(self, fake_llm):
        """测试用例设计与执行工具也应原样返回结果。"""
        fake_llm._responses = [
            ChatResponse(
                content="",
                model="fake",
                stop_reason="tool_use",
                tool_calls=[
                    ToolCall(id="t1", name="design_test_cases", arguments={}),
                    ToolCall(id="t2", name="execute_scenario", arguments={}),
                ],
            ),
            ChatResponse(
                content="不应出现的二次总结",
                model="fake",
                stop_reason="end_turn",
            ),
        ]
        agent = ReActAgent(
            llm=fake_llm,
            tools=[
                _StaticTool(
                    "design_test_cases",
                    "已生成测试用例 Excel 文件：/tmp/cases.xlsx",
                    FINAL_ANSWER_PASSTHROUGH,
                ),
                _StaticTool(
                    "execute_scenario",
                    "执行结果：PASS\n用例：CASE-001 - 登录成功",
                    FINAL_ANSWER_PASSTHROUGH,
                ),
            ],
            system_prompt="test",
        )

        result = await agent.run(query="生成并执行")

        assert result.answer == (
            "已生成测试用例 Excel 文件：/tmp/cases.xlsx\n\n"
            "执行结果：PASS\n用例：CASE-001 - 登录成功"
        )
        assert fake_llm._response_idx == 1

    async def test_search_knowledge_still_uses_llm_summary(self, fake_llm):
        """知识查询工具保持普通总结模式，不启用 passthrough。"""
        fake_llm._responses = [
            ChatResponse(
                content="",
                model="fake",
                stop_reason="tool_use",
                tool_calls=[
                    ToolCall(id="t1", name="search_knowledge", arguments={})
                ],
            ),
            ChatResponse(
                content="基于知识库结果的最终回答",
                model="fake",
                stop_reason="end_turn",
            ),
        ]
        agent = ReActAgent(
            llm=fake_llm,
            tools=[
                _StaticTool(
                    "search_knowledge",
                    "【知识库结果】模块右上角'追番表'入口",
                )
            ],
            system_prompt="test",
        )

        result = await agent.run(query="查知识库")

        assert result.iterations == 2
        assert result.answer == "基于知识库结果的最终回答"
        assert fake_llm._response_idx == 2
