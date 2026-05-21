"""OpenAI-compatible API provider — works with OpenAI, DeepSeek, DashScope, etc."""

import os
from typing import AsyncIterator

from src.core.config import get_config
from src.core.exceptions import LLMError
from src.core.logging import get_logger
from src.llm.base import BaseLLM, LLMResponse

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

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        cfg = get_config().get("llm", {})
        temperature = temperature if temperature is not None else cfg.get("temperature", 0.3)
        max_tokens = max_tokens or cfg.get("max_tokens", 2048)

        if not self._api_key:
            raise LLMError("LLM API key not set")

        client = self._build_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            raise LLMError(f"LLM API error: {e}") from e

        choice = response.choices[0]
        return LLMResponse(
            content=choice.message.content or "",
            model=self._model,
            usage={
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens if response.usage else 0,
            },
        )

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        cfg = get_config().get("llm", {})
        temperature = temperature if temperature is not None else cfg.get("temperature", 0.3)
        max_tokens = max_tokens or cfg.get("max_tokens", 2048)

        if not self._api_key:
            raise LLMError("LLM API key not set")

        client = self._build_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            stream = await client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            raise LLMError(f"LLM API streaming error: {e}") from e
