"""Build prompts with context and citation markers."""

from pathlib import Path

from src.core.config import get_config
from src.vectordb.base import SearchResult


class PromptBuilder:
    """Assemble prompts for RAG generation with citation support."""

    def __init__(self, system_prompt_template: str | None = None):
        cfg_system = get_config().get("llm", {}).get("system_prompt", "")
        self._system_template = system_prompt_template or cfg_system

    def build_system_prompt(self, context_chunks: list[SearchResult]) -> str:
        """Build system prompt with numbered context chunks for citation."""
        parts: list[str] = []
        for i, chunk in enumerate(context_chunks, start=1):
            source_name = Path(chunk.metadata.get("source_path", "")).name or "unknown"
            parts.append(f"[{i}] <source={source_name}>\n{chunk.content}")

        context_text = "\n\n".join(parts)
        return self._system_template.format(context=context_text)

    def build_user_prompt(self, query: str) -> str:
        return f"用户问题：{query}\n\n请回答："
