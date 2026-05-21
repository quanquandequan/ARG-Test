"""Claude API provider via Anthropic SDK."""

import os
from typing import AsyncIterator

from src.core.config import get_config
from src.core.exceptions import LLMError
from src.core.logging import get_logger
from src.llm.base import BaseLLM, LLMResponse

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

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        import anthropic

        cfg = get_config().get("llm", {})
        temperature = temperature if temperature is not None else cfg.get("temperature", 0.3)
        max_tokens = max_tokens or cfg.get("max_tokens", 2048)

        if not self._api_key:
            raise LLMError("ANTHROPIC_API_KEY not set")

        client = anthropic.AsyncAnthropic(api_key=self._api_key)

        try:
            message = await client.messages.create(
                model=self._model,
                system=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.APIError as e:
            raise LLMError(f"Claude API error: {e}") from e

        content = message.content[0].text if message.content else ""
        return LLMResponse(
            content=content,
            model=self._model,
            usage={
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
            },
        )

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        import anthropic

        cfg = get_config().get("llm", {})
        temperature = temperature if temperature is not None else cfg.get("temperature", 0.3)
        max_tokens = max_tokens or cfg.get("max_tokens", 2048)

        if not self._api_key:
            raise LLMError("ANTHROPIC_API_KEY not set")

        client = anthropic.AsyncAnthropic(api_key=self._api_key)

        try:
            async with client.messages.stream(
                model=self._model,
                system=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                async for event in stream:
                    if event.type == "content_block_delta" and event.delta.type == "text_delta":
                        yield event.delta.text
        except anthropic.APIError as e:
            raise LLMError(f"Claude API streaming error: {e}") from e
