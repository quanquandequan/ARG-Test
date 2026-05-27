"""A12: CLI unit tests — exercise output formatting and SSE parsing helpers."""

from __future__ import annotations

from argparse import Namespace
from unittest.mock import AsyncMock, patch

from src.agent.cli import _format_answer, _parse_sse_event
from src.agent.types import AgentResult, AgentStep, Citation

# ── _parse_sse_event ────────────────────────────────────────────────────────

class TestParseSseEvent:
    def test_answer_event(self):
        raw = 'event: answer\ndata: {"text": "你好"}\n\n'
        etype, data = _parse_sse_event(raw)
        assert etype == "answer"
        assert isinstance(data, dict)
        assert data["text"] == "你好"

    def test_token_event(self):
        raw = 'event: token\ndata: {"text": "H"}\n\n'
        etype, data = _parse_sse_event(raw)
        assert etype == "token"
        assert data["text"] == "H"

    def test_tool_call_event(self):
        raw = (
            'event: tool_call\n'
            'data: {"tools": ["knowledge_search"], "iteration": 0}\n\n'
        )
        etype, data = _parse_sse_event(raw)
        assert etype == "tool_call"
        assert data["tools"] == ["knowledge_search"]
        assert data["iteration"] == 0

    def test_start_event(self):
        raw = 'event: start\ndata: {"trace_id": "abc-123"}\n\n'
        etype, data = _parse_sse_event(raw)
        assert etype == "start"
        assert data["trace_id"] == "abc-123"

    def test_invalid_json_falls_back_to_string(self):
        raw = "event: unknown\ndata: not-json\n\n"
        etype, data = _parse_sse_event(raw)
        assert etype == "unknown"
        assert data == "not-json"


# ── _format_answer ──────────────────────────────────────────────────────────

class TestFormatAnswer:
    def test_plain_text_unchanged(self):
        text = "RAG 是检索增强生成技术。"
        assert _format_answer(text) == text

    def test_markdown_table_re_aligned(self):
        table = "| 名称 | 分数 |\n| --- | --- |\n| RAG | 95 |"
        result = _format_answer(table)
        # Output must still be a table with pipe characters
        assert "|" in result
        assert "RAG" in result

    def test_multiline_mixed(self):
        text = "说明如下：\n| A | B |\n| 1 | 2 |\n继续。"
        result = _format_answer(text)
        assert "说明如下" in result
        assert "继续" in result


# ── _run_query integration (mocked agent) ──────────────────────────────────

class TestRunQuery:
    async def test_non_stream_prints_answer(self, capsys):
        fake_result = AgentResult(
            answer="这是回答 [1]。",
            iterations=1,
            steps=[AgentStep(step_index=0, thinking="这是回答 [1]。", duration_ms=42.0)],
            citations=[Citation(index=1)],
            processing_stages={"total": 123.0},
            trace_id="test-trace",
        )

        async def fake_run(*args, **kwargs):
            return fake_result

        with (
            patch("src.api.dependencies.get_agent") as mock_get_agent,
            patch("src.core.config.load_config"),
            patch("src.core.logging.setup_logging"),
        ):
            mock_agent = AsyncMock()
            mock_agent.run = AsyncMock(return_value=fake_result)
            mock_get_agent.return_value = mock_agent

            from src.agent.cli import _run_query
            args = Namespace(
                query="测试问题",
                stream=False,
                verbose=False,
                env="test",
            )
            await _run_query(args)

        captured = capsys.readouterr()
        assert "这是回答" in captured.out

    async def test_verbose_shows_trace_and_timing(self, capsys):
        fake_result = AgentResult(
            answer="答案",
            iterations=1,
            steps=[AgentStep(step_index=0, thinking="答案", duration_ms=55.0)],
            citations=[],
            processing_stages={"total": 88.0},
            trace_id="verbose-trace-id",
        )

        with (
            patch("src.api.dependencies.get_agent") as mock_get_agent,
            patch("src.core.config.load_config"),
            patch("src.core.logging.setup_logging"),
        ):
            mock_agent = AsyncMock()
            mock_agent.run = AsyncMock(return_value=fake_result)
            mock_get_agent.return_value = mock_agent

            from src.agent.cli import _run_query
            args = Namespace(
                query="测试",
                stream=False,
                verbose=True,
                env="test",
            )
            await _run_query(args)

        captured = capsys.readouterr()
        assert "verbose-trace-id" in captured.out
        assert "88" in captured.out  # total timing

    async def test_stream_collects_tokens(self, capsys):
        """Streaming mode should print token text as it arrives."""
        events = [
            'event: start\ndata: {"trace_id": "t1"}\n\n',
            'event: token\ndata: {"text": "Hello"}\n\n',
            'event: token\ndata: {"text": " World"}\n\n',
            'event: answer\ndata: {"text": "Hello World"}\n\n',
        ]

        async def fake_run_stream(*args, **kwargs):
            for e in events:
                yield e

        with (
            patch("src.api.dependencies.get_agent") as mock_get_agent,
            patch("src.core.config.load_config"),
            patch("src.core.logging.setup_logging"),
        ):
            mock_agent = AsyncMock()
            mock_agent.run_stream = fake_run_stream
            mock_get_agent.return_value = mock_agent

            from src.agent.cli import _run_query
            args = Namespace(
                query="stream test",
                stream=True,
                verbose=False,
                env="test",
            )
            await _run_query(args)

        captured = capsys.readouterr()
        assert "Hello" in captured.out
        assert "World" in captured.out
