"""Claude API provider via Anthropic SDK."""

import json
import os
from typing import AsyncIterator

from src.core.config import get_config
from src.core.exceptions import LLMError
from src.core.logging import get_logger
from src.llm.base import BaseLLM
from src.llm.types import ChatResponse, ContentBlock, Message, ToolCall

logger = get_logger(__name__)


class ClaudeProvider(BaseLLM):
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        cfg = get_config().get("llm", {})
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._model = model or cfg.get("model", "claude-sonnet-4-6")

    # ── Message / tool format conversion ──

    def _messages_to_anthropic(self, messages: list[Message]) -> list[dict]:
        """Convert internal Message list to Anthropic API format."""
        converted: list[dict] = []
        for msg in messages:
            if msg.role == "system":
                continue
            elif msg.role == "tool":
                converted.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": msg.content if isinstance(msg.content, str) else "",
                    }],
                })
            elif msg.role == "assistant" and msg.tool_calls:
                blocks: list[dict] = []
                if isinstance(msg.content, str) and msg.content:
                    blocks.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    blocks.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    })
                converted.append({"role": "assistant", "content": blocks})
            elif isinstance(msg.content, str):
                converted.append({"role": msg.role, "content": msg.content})
            else:
                converted.append({"role": msg.role, "content": str(msg.content)})
        return converted

    def _extract_system_prompt(self, messages: list[Message]) -> str:
        for msg in messages:
            if msg.role == "system":
                return msg.content if isinstance(msg.content, str) else ""
        return ""

    def _tools_to_anthropic(self, tools: list[dict] | None) -> list[dict] | None:
        if not tools:
            return None
        converted = []
        for t in tools:
            converted.append({
                "name": t["name"],
                "description": t.get("description", ""),
                "input_schema": {
                    "type": "object",
                    "properties": t.get("parameters", {}).get("properties", {}),
                    "required": t.get("parameters", {}).get("required", []),
                },
            })
        return converted

    def _parse_anthropic_response(self, response) -> ChatResponse:
        tool_calls: list[ToolCall] = []
        text = ""
        stop_reason = getattr(response, "stop_reason", "end_turn")

        for block in response.content:
            if block.type == "text":
                text += block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else {},
                ))

        return ChatResponse(
            content=text,
            model=self._model,
            stop_reason=stop_reason,
            tool_calls=tool_calls,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
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
        import anthropic

        cfg = get_config().get("llm", {})
        temperature = temperature if temperature is not None else cfg.get("temperature", 0.3)
        max_tokens = max_tokens or cfg.get("max_tokens", 4096)

        if not self._api_key:
            raise LLMError("ANTHROPIC_API_KEY not set")

        client = anthropic.AsyncAnthropic(api_key=self._api_key)
        system = self._extract_system_prompt(messages)
        converted = self._messages_to_anthropic(messages)
        converted_tools = self._tools_to_anthropic(tools)

        kwargs = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": converted,
        }
        if system:
            kwargs["system"] = system
        if converted_tools:
            kwargs["tools"] = converted_tools
            if tool_choice:
                if tool_choice == "auto":
                    tc = {"type": "auto"}
                elif tool_choice == "any":
                    tc = {"type": "any"}
                else:
                    tc = {"type": "tool", "name": tool_choice}
                kwargs["tool_choice"] = tc

        try:
            response = await client.messages.create(**kwargs)
        except anthropic.APIError as e:
            raise LLMError(f"Claude API error: {e}") from e

        return self._parse_anthropic_response(response)

    async def generate_chat_stream(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[ContentBlock]:
        import anthropic

        cfg = get_config().get("llm", {})
        temperature = temperature if temperature is not None else cfg.get("temperature", 0.3)
        max_tokens = max_tokens or cfg.get("max_tokens", 4096)

        if not self._api_key:
            raise LLMError("ANTHROPIC_API_KEY not set")

        client = anthropic.AsyncAnthropic(api_key=self._api_key)
        system = self._extract_system_prompt(messages)
        converted = self._messages_to_anthropic(messages)
        converted_tools = self._tools_to_anthropic(tools)

        kwargs = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": converted,
        }
        if system:
            kwargs["system"] = system
        if converted_tools:
            kwargs["tools"] = converted_tools

        try:
            async with client.messages.stream(**kwargs) as stream:
                async for event in stream:
                    if event.type == "content_block_start":
                        if event.content_block.type == "tool_use":
                            pass
                    elif event.type == "content_block_delta":
                        if event.delta.type == "text_delta":
                            yield ContentBlock(type="text", text=event.delta.text)
                        elif event.delta.type == "input_json_delta":
                            pass
                    elif event.type == "content_block_stop":
                        pass
        except anthropic.APIError as e:
            raise LLMError(f"Claude API streaming error: {e}") from e
