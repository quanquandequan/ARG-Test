"""A8: Provider message / tool format conversion unit tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.llm.claude_provider import ClaudeProvider
from src.llm.openai_provider import OpenAIProvider
from src.llm.types import Message, ToolCall


@pytest.fixture
def claude() -> ClaudeProvider:
    return ClaudeProvider(api_key="test-key", model="claude-test")


@pytest.fixture
def openai_provider() -> OpenAIProvider:
    return OpenAIProvider(api_key="test-key", model="gpt-test")


class TestClaudeMessageConversion:
    def test_system_extracted_and_omitted_from_messages(self, claude):
        messages = [
            Message(role="system", content="你是助手"),
            Message(role="user", content="你好"),
        ]
        converted = claude._messages_to_anthropic(messages)
        assert converted == [{"role": "user", "content": "你好"}]
        assert claude._extract_system_prompt(messages) == "你是助手"

    def test_tool_result_becomes_user_tool_result_block(self, claude):
        messages = [
            Message(
                role="tool",
                content="检索结果",
                tool_call_id="call_1",
                name="knowledge_search",
            ),
        ]
        converted = claude._messages_to_anthropic(messages)
        assert converted[0]["role"] == "user"
        block = converted[0]["content"][0]
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == "call_1"
        assert block["content"] == "检索结果"

    def test_assistant_tool_calls_become_tool_use_blocks(self, claude):
        messages = [
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCall(id="t1", name="echo", arguments={"message": "hi"}),
                ],
            ),
        ]
        converted = claude._messages_to_anthropic(messages)
        assert converted[0]["role"] == "assistant"
        blocks = converted[0]["content"]
        assert blocks[0]["type"] == "tool_use"
        assert blocks[0]["name"] == "echo"
        assert blocks[0]["input"] == {"message": "hi"}

    def test_tools_to_anthropic_input_schema(self, claude):
        tools = [{
            "name": "knowledge_search",
            "description": "搜索知识库",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        }]
        converted = claude._tools_to_anthropic(tools)
        assert converted is not None
        assert converted[0]["name"] == "knowledge_search"
        assert converted[0]["input_schema"]["required"] == ["query"]

    def test_parse_anthropic_response_text_and_tool_use(self, claude):
        text_block = SimpleNamespace(type="text", text="最终答案")
        tool_block = SimpleNamespace(
            type="tool_use",
            id="toolu_1",
            name="knowledge_search",
            input={"query": "RAG"},
        )
        response = SimpleNamespace(
            content=[text_block, tool_block],
            stop_reason="tool_use",
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        )
        parsed = claude._parse_anthropic_response(response)
        assert parsed.content == "最终答案"
        assert parsed.stop_reason == "tool_use"
        assert len(parsed.tool_calls) == 1
        assert parsed.tool_calls[0].name == "knowledge_search"
        assert parsed.tool_calls[0].arguments == {"query": "RAG"}


class TestOpenAIMessageConversion:
    def test_tool_message_includes_tool_call_id(self, openai_provider):
        messages = [
            Message(
                role="tool",
                content="结果",
                tool_call_id="call_abc",
                name="knowledge_search",
            ),
        ]
        converted = openai_provider._messages_to_openai(messages)
        assert converted[0]["role"] == "tool"
        assert converted[0]["tool_call_id"] == "call_abc"
        assert converted[0]["content"] == "结果"

    def test_assistant_tool_calls_serialize_arguments_as_json(self, openai_provider):
        messages = [
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCall(id="call_1", name="echo", arguments={"msg": "x"}),
                ],
            ),
        ]
        converted = openai_provider._messages_to_openai(messages)
        tc = converted[0]["tool_calls"][0]
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "echo"
        assert json.loads(tc["function"]["arguments"]) == {"msg": "x"}

    def test_tools_to_openai_function_wrappers(self, openai_provider):
        tools = [{"name": "web_search", "description": "搜索", "parameters": {}}]
        converted = openai_provider._tools_to_openai(tools)
        assert converted[0]["type"] == "function"
        assert converted[0]["function"]["name"] == "web_search"

    def test_parse_openai_response_maps_tool_calls_finish_reason(self, openai_provider):
        fn = SimpleNamespace(name="echo", arguments='{"message":"hi"}')
        tc = SimpleNamespace(id="call_1", function=fn)
        msg = SimpleNamespace(content="思考中", tool_calls=[tc])
        choice = SimpleNamespace(message=msg, finish_reason="tool_calls")
        response = SimpleNamespace(
            choices=[choice],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=7),
        )
        parsed = openai_provider._parse_openai_response(response)
        assert parsed.stop_reason == "tool_use"
        assert parsed.tool_calls[0].arguments == {"message": "hi"}
        assert parsed.content == "思考中"

    def test_parse_openai_response_invalid_json_arguments_becomes_empty(self, openai_provider):
        fn = SimpleNamespace(name="echo", arguments="not-json")
        tc = SimpleNamespace(id="call_1", function=fn)
        msg = SimpleNamespace(content="", tool_calls=[tc])
        choice = SimpleNamespace(message=msg, finish_reason="tool_calls")
        response = SimpleNamespace(choices=[choice], usage=None)
        parsed = openai_provider._parse_openai_response(response)
        assert parsed.tool_calls[0].arguments == {}
