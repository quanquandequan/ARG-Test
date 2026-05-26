"""Unit tests for PromptBuilder."""

from src.generation.prompt_builder import PromptBuilder
from src.vectordb.base import SearchResult


def _chunks(n: int) -> list[SearchResult]:
    return [
        SearchResult(
            id=f"c{i}",
            document_id=f"doc{i}",
            content=f"片段{i}内容",
            score=1.0 - i * 0.1,
            metadata={"source_path": f"/srv/{i}.md", "chunk_index": i},
        )
        for i in range(n)
    ]


def test_user_prompt_contains_query():
    builder = PromptBuilder()
    p = builder.build_user_prompt("如何使用 RAG？")
    assert "如何使用 RAG？" in p
    assert "用户问题" in p


def test_system_prompt_numbers_chunks_and_includes_source():
    builder = PromptBuilder()
    sp = builder.build_system_prompt(_chunks(3))
    assert "[1]" in sp and "[2]" in sp and "[3]" in sp
    assert "0.md" in sp and "1.md" in sp
    assert "片段0内容" in sp


def test_empty_context_does_not_crash():
    builder = PromptBuilder()
    sp = builder.build_system_prompt([])
    # `{context}` placeholder should still be substituted (with empty string)
    assert "{context}" not in sp


def test_custom_template_overrides_config():
    builder = PromptBuilder(system_prompt_template="模板:{context}:结束")
    sp = builder.build_system_prompt(_chunks(1))
    assert sp.startswith("模板:")
    assert sp.endswith(":结束")
    assert "片段0内容" in sp
