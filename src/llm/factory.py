"""LLM factory — config-driven instantiation."""

import os

from src.core.config import get_config
from src.llm.base import BaseLLM
from src.llm.claude_provider import ClaudeProvider
from src.llm.openai_provider import OpenAIProvider

_DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def get_llm() -> BaseLLM:
    cfg = get_config().get("llm", {})
    provider = os.environ.get("LLM_PROVIDER") or cfg.get("provider", "claude")

    if provider == "claude":
        return ClaudeProvider()
    elif provider == "openai":
        return OpenAIProvider()
    elif provider == "deepseek":
        return OpenAIProvider(
            api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
            model=cfg.get("model", "deepseek-chat"),
            base_url=_DEEPSEEK_BASE_URL,
        )

    raise ValueError(f"Unknown LLM provider: {provider}")
