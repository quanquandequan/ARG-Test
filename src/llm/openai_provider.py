"""OpenAI-compatible API provider — works with OpenAI, DeepSeek, DashScope, etc."""

import json
import os
from typing import AsyncIterator

from src.core.config import get_config
from src.core.exceptions import LLMError
from src.core.logging import get_logger
from src.llm.base import BaseLLM
from src.llm.types import ChatResponse, ContentBlock, Message, ToolCall

logger = get_logger(__name__)


class OpenAIProvider(BaseLLM):
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ):
        cfg = get_config().get("llm", {})
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._model = model or cfg.get("model", "gpt-4o")
        self._base_url = base_url or cfg.get("base_url")

    def _build_client(self):
        from openai import AsyncOpenAI

        kwargs = {"api_key": self._api_key}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        return AsyncOpenAI(**kwargs)

    # ── Message / tool format conversion ──

    def _messages_to_openai(self, messages: list[Message]) -> list[dict]:
        """Convert internal Message list to OpenAI API format."""
        converted: list[dict] = []
        for msg in messages:
            entry: dict = {"role": msg.role}
            if msg.role == "assistant" and msg.tool_calls:
                entry["content"] = msg.content if isinstance(msg.content, str) else str(msg.content)
                entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            elif msg.role == "tool":
                entry["tool_call_id"] = msg.tool_call_id or ""
                entry["content"] = msg.content if isinstance(msg.content, str) else str(msg.content)
            else:
                entry["content"] = msg.content if isinstance(msg.content, str) else str(msg.content)
            converted.append(entry)
        return converted

    def _tools_to_openai(self, tools: list[dict] | None) -> list[dict] | None:
        if not tools:
            return None
        return [
            {"type": "function", "function": t}
            for t in tools
        ]

    def _parse_openai_response(self, response) -> ChatResponse:
        tool_calls: list[ToolCall] = []
        text = ""
        choice = response.choices[0]
        msg = choice.message

        if msg.content:
            text = msg.content

        if msg.tool_calls:
            for tc in msg.tool_calls:
                args = {}
                if tc.function.arguments:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))

        stop_reason = "end_turn"
        if choice.finish_reason == "tool_calls":
            stop_reason = "tool_use"
        elif choice.finish_reason == "stop":
            stop_reason = "end_turn"
        elif choice.finish_reason == "length":
            stop_reason = "max_tokens"

        return ChatResponse(
            content=text,
            model=self._model,
            stop_reason=stop_reason,
            tool_calls=tool_calls,
            usage={
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens if response.usage else 0,
            },
        )

    # ── Agent interface ──

    async def generate_chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        cfg = get_config().get("llm", {})
        temperature = temperature if temperature is not None else cfg.get("temperature", 0.3)
        max_tokens = max_tokens or cfg.get("max_tokens", 4096)

        if not self._api_key:
            raise LLMError("LLM API key not set")

        client = self._build_client()
        converted = self._messages_to_openai(messages)
        converted_tools = self._tools_to_openai(tools)

        kwargs = {
            "model": self._model,
            "messages": converted,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if converted_tools:
            kwargs["tools"] = converted_tools
            if tool_choice:
                if tool_choice == "auto":
                    kwargs["tool_choice"] = "auto"
                elif tool_choice == "any":
                    kwargs["tool_choice"] = "required"
                else:
                    kwargs["tool_choice"] = {"type": "function", "function": {"name": tool_choice}}

        try:
            response = await client.chat.completions.create(**kwargs)
        except Exception as e:
            raise LLMError(f"LLM API error: {e}") from e

        return self._parse_openai_response(response)

    async def generate_chat_stream(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[ContentBlock]:
        cfg = get_config().get("llm", {})
        temperature = temperature if temperature is not None else cfg.get("temperature", 0.3)
        max_tokens = max_tokens or cfg.get("max_tokens", 4096)

        if not self._api_key:
            raise LLMError("LLM API key not set")

        client = self._build_client()
        converted = self._messages_to_openai(messages)
        converted_tools = self._tools_to_openai(tools)

        kwargs = {
            "model": self._model,
            "messages": converted,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if converted_tools:
            kwargs["tools"] = converted_tools

        try:
            stream = await client.chat.completions.create(**kwargs)
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield ContentBlock(
                        type="text",
                        text=chunk.choices[0].delta.content,
                    )
        except Exception as e:
            raise LLMError(f"LLM API streaming error: {e}") from e
