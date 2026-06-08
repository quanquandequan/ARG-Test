"""LLM provider 抽象基类。"""

from abc import ABC, abstractmethod
from typing import AsyncIterator

from src.llm.types import ChatResponse, ContentBlock, Message


class BaseLLM(ABC):
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
