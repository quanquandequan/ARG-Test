"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: dict = field(default_factory=dict)  # {"input_tokens": N, "output_tokens": M}


class BaseLLM(ABC):
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
