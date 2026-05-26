"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator

from src.llm.types import ChatResponse, ContentBlock, Message


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: dict = field(default_factory=dict)  # {"input_tokens": N, "output_tokens": M}


class BaseLLM(ABC):
    # ── 单轮接口 (保留，Generator 仍在使用) ──

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        ...

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        ...

    # ── 多轮 / Agent 接口 ──

    @abstractmethod
    async def generate_chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        ...

    @abstractmethod
    async def generate_chat_stream(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[ContentBlock]:
        ...
