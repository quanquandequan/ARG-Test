#!/usr/bin/env python3
"""RAG Agent CLI — single queries and interactive chat.

Usage:
    rag ask "你的问题"              # Single query
    rag ask -s "你的问题"           # Single query + streaming
    rag ask -v "你的问题"           # Verbose: show tool calls + timing
    rag chat                        # Interactive chat mode
    rag chat -s                     # Interactive + streaming
    rag chat -v                     # Interactive + verbose
    rag --env production ask "..."  # Use production config
"""

from __future__ import annotations  # noqa: I001

# -- Suppress C++ gRPC stderr noise via fd-level redirect ----------
# gRPC C++ core writes GOAWAY / too_many_pings directly to stderr,
# bypassing Python logging & env vars.  We dup2 stderr through a
# pipe and spawn a daemon thread to filter the noise.
import os as _os  # noqa: I001
import threading as _threading
import re as _re

_orig_stderr_fd = _os.dup(2)
_rfd, _wfd = _os.pipe()
_os.dup2(_wfd, 2)
_os.close(_wfd)

_GRPC_NOISE = _re.compile(
    rb"(?:GOAWAY|too_many_pings|chttp2_transport\.cc|grpc_init)"
)


def _stderr_filter():
    try:
        while True:
            chunk = _os.read(_rfd, 65536)  # noqa: F821
            if not chunk:
                break
            clean = b"\n".join(
                line
                for line in chunk.split(b"\n")
                if line.strip() and not _GRPC_NOISE.search(line)
            )
            if clean:
                _os.write(_orig_stderr_fd, clean + b"\n")
    except OSError:
        pass


_threading.Thread(target=_stderr_filter, daemon=True).start()
del _re, _threading, _rfd, _wfd
# -----------------------------------------------------------------

import argparse  # noqa: E402
import asyncio  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import readline  # noqa: E402, F401 — init readline/libedit for CJK backspace
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# Ensure project root is on sys.path for both `python -m src.agent.cli` and `pip install -e`
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _display_width(s: str) -> int:
    """Return display width: CJK chars = 2, ASCII = 1."""
    import unicodedata
    w = 0
    for c in s:
        w += 2 if unicodedata.east_asian_width(c) in ("W", "F") else 1
    return w


def _format_answer(text: str) -> str:
    """Re-align markdown tables for terminal display (CJK-aware)."""
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith("|") and stripped.count("|") >= 2:
            rows: list[list[str]] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().split("|")]
                if cells and cells[0] == "":
                    cells = cells[1:]
                if cells and cells[-1] == "":
                    cells = cells[:-1]
                rows.append(cells)
                i += 1

            if not rows:
                continue

            ncols = max(len(r) for r in rows)
            widths = [0] * ncols
            for row in rows:
                for ci, cell in enumerate(row):
                    if ci < ncols:
                        widths[ci] = max(widths[ci], _display_width(cell))

            for row in rows:
                padded = []
                for ci in range(ncols):
                    cell = row[ci] if ci < len(row) else ""
                    pad = widths[ci] - _display_width(cell)
                    padded.append(cell + " " * pad)
                out.append("| " + " | ".join(padded) + " |")
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out)


def _parse_sse_event(raw: str) -> tuple[str, dict | str]:
    """Return (event_type, data) from a raw SSE block."""
    event_type = "message"
    data_str = ""
    for line in raw.splitlines():
        if line.startswith("event:"):
            event_type = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_str = line[len("data:"):].strip()
    try:
        return event_type, json.loads(data_str)
    except (json.JSONDecodeError, ValueError):
        return event_type, data_str


def _load_dotenv():
    """Load .env file from project root without python-dotenv dependency."""
    env_path = _PROJECT_ROOT / ".env"
    if not env_path.is_file():
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if value and value[0] in ('"', "'") and value[-1] == value[0]:
                value = value[1:-1]
            if key not in os.environ:
                os.environ[key] = value


async def _run_query(args):
    from src.api.dependencies import get_agent  # noqa: PLC0415
    from src.core.config import load_config  # noqa: PLC0415
    from src.core.logging import setup_logging  # noqa: PLC0415

    load_config(args.env)
    setup_logging()

    agent = get_agent()
    print()  # blank line before answer

    if args.stream:
        async for event in agent.run_stream(query=args.query):
            event_type, data = _parse_sse_event(event)
            if event_type == "token" and isinstance(data, dict):
                print(data.get("text", ""), end="", flush=True)
            elif event_type == "tool_call" and args.verbose and isinstance(data, dict):
                tools = ", ".join(data.get("tools", []))
                iteration = data.get("iteration", "?")
                print(
                    f"\n  [TOOL iter={iteration}] {tools}",
                    file=sys.stderr,
                    flush=True,
                )
            elif event_type == "tool_result" and args.verbose and isinstance(data, dict):
                print(
                    f"\n  [RESULT] {data.get('tool')} "
                    f"({data.get('duration_ms', 0):.0f}ms, "
                    f"{data.get('result_len', 0)} chars)",
                    file=sys.stderr,
                    flush=True,
                )
        print()
    else:
        result = await agent.run(query=args.query)

        if args.verbose:
            print(f"  trace_id={result.trace_id}  ({result.iterations} 次迭代)\n")

        print(_format_answer(result.answer))

        if args.verbose and result.steps:
            print()
            for s in result.steps:
                if s.tool_call:
                    print(
                        f"  [CALL] [{s.step_index}] "
                        f"{s.tool_call.name}({s.tool_call.arguments}) "
                        f"→ {s.duration_ms:.0f}ms"
                    )
                else:
                    print(
                        f"  [THINK] [{s.step_index}] think → answer "
                        f"({s.duration_ms:.0f}ms)"
                    )

        if result.processing_stages:
            total = result.processing_stages.get("total", 0)
            print(f"\n  [TIMING] total={total:.0f}ms")

        if result.citations:
            print(f"  [CITE] {[c.index for c in result.citations]}")


async def _run_chat(args):
    from src.api.dependencies import get_agent  # noqa: PLC0415
    from src.core.config import load_config  # noqa: PLC0415
    from src.core.logging import setup_logging  # noqa: PLC0415
    from src.llm.types import Message  # noqa: PLC0415

    load_config(args.env)
    setup_logging()

    agent = get_agent()
    history: list[Message] = []
    last_answer: str = ""

    print("\nRAG Agent Chat — 输入 /quit 退出, /clear 清除历史\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query:
            continue
        if query == "/quit":
            print("Bye!")
            break
        if query == "/clear":
            history.clear()
            print("   (历史已清除)\n")
            continue

        if args.stream:
            print("Agent: ", end="", flush=True)
            last_answer = ""
            async for event in agent.run_stream(query=query, history=history):
                event_type, data = _parse_sse_event(event)
                if event_type == "token" and isinstance(data, dict):
                    tok = data.get("text", "")
                    last_answer += tok
                    print(tok, end="", flush=True)
                elif event_type == "answer" and isinstance(data, dict):
                    last_answer = data.get("text", last_answer)
                elif event_type == "tool_call" and args.verbose and isinstance(data, dict):
                    tools = ", ".join(data.get("tools", []))
                    print(f"\n   [TOOL] {tools}", file=sys.stderr, flush=True)
            print()
        else:
            result = await agent.run(query=query, history=history)
            last_answer = result.answer
            if args.verbose and result.steps:
                for s in result.steps:
                    if s.tool_call:
                        print(
                            f"   [TOOL] {s.tool_call.name}({s.tool_call.arguments}) "
                            f"→ {s.duration_ms:.0f}ms"
                        )
            formatted = _format_answer(result.answer)
            if formatted.startswith("|"):
                print(f"Agent:\n{formatted}")
            else:
                print(f"Agent: {formatted}")

        # Append to history (append both sides so multi-turn works)
        history.append(Message(role="user", content=query))
        history.append(Message(role="assistant", content=last_answer))
        print()


def main():
    _load_dotenv()

    parser = argparse.ArgumentParser(
        description="RAG Agent CLI — 知识库智能问答",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  rag ask "我们的产品有哪些功能？"
  rag ask -s "1+1等于几？"
  rag ask -v "如何配置Milvus？"
  rag chat
  rag chat -v
        """,
    )
    parser.add_argument("--env", default="development", help="配置环境 (default: development)")
    sub = parser.add_subparsers(dest="command")

    # ask
    ask = sub.add_parser("ask", help="单次查询")
    ask.add_argument("query", help="你的问题")
    ask.add_argument("-s", "--stream", action="store_true", help="流式输出")
    ask.add_argument("-v", "--verbose", action="store_true", help="显示工具调用、计时和步骤")

    # chat
    chat = sub.add_parser("chat", help="交互式对话")
    chat.add_argument("-s", "--stream", action="store_true", help="流式输出")
    chat.add_argument("-v", "--verbose", action="store_true", help="显示工具调用")

    args = parser.parse_args()

    if args.command == "ask":
        asyncio.run(_run_query(args))
    elif args.command == "chat":
        asyncio.run(_run_chat(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
