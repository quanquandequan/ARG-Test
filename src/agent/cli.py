#!/usr/bin/env python3
"""RAG Agent CLI — 交互式聊天。

用法：
    rag chat           # 精简模式（无日志输出）
    rag chat -d        # 调试模式（显示工具调用与详细日志）
    rag chat -s        # 流式输出
    rag --env production chat
"""

from __future__ import annotations  # noqa: I001

# -- 通过 fd 级重定向抑制 C++ gRPC stderr 噪声 ---------------------------
from src.core.utils import suppress_grpc_stderr

suppress_grpc_stderr()
# --------------------------------------------------------------------------

import warnings  # noqa: E402
warnings.filterwarnings("ignore", category=FutureWarning, module="pymilvus")

import argparse  # noqa: E402
import asyncio  # noqa: E402
import logging  # noqa: E402
import readline  # noqa: E402, F401 — 初始化 readline/libedit 以支持 CJK 退格
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

from src.core.utils import format_answer, load_dotenv, parse_sse_event  # noqa: E402

# 为向后兼容重新导出（测试会从 cli 导入）
_format_answer = format_answer
_parse_sse_event = parse_sse_event

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# 确保 `python -m src.agent.cli` 和 `pip install -e` 时项目根目录都在 sys.path 中
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


async def _run_chat(args):
    from src.core.container import get_agent  # noqa: PLC0415
    from src.core.config import load_config  # noqa: PLC0415
    from src.core.logging import setup_logging  # noqa: PLC0415
    from src.llm.types import Message  # noqa: PLC0415

    load_config(args.env)
    setup_logging()
    if not args.debug:
        # 精简模式：将根日志级别提升到 WARNING，屏蔽所有 INFO/DEBUG 输出
        logging.root.setLevel(logging.WARNING)

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
                event_type, data = parse_sse_event(event)
                if event_type == "token" and isinstance(data, dict):
                    tok = data.get("text", "")
                    last_answer += tok
                    print(tok, end="", flush=True)
                elif event_type == "answer" and isinstance(data, dict):
                    last_answer = data.get("text", last_answer)
                elif event_type == "tool_call" and args.debug and isinstance(data, dict):
                    tools = ", ".join(data.get("tools", []))
                    print(f"\n   [TOOL] {tools}", file=sys.stderr, flush=True)
            print()
        else:
            result = await agent.run(query=query, history=history)
            last_answer = result.answer
            if args.debug and result.steps:
                for s in result.steps:
                    if s.tool_call:
                        print(
                            f"   [TOOL] {s.tool_call.name}({s.tool_call.arguments}) "
                            f"→ {s.duration_ms:.0f}ms"
                        )
            formatted = format_answer(result.answer)
            if formatted.startswith("|"):
                print(f"Agent:\n{formatted}")
            else:
                print(f"Agent: {formatted}")

        # 追加到历史记录（双方消息都追加，保证多轮对话可用）
        history.append(Message(role="user", content=query))
        history.append(Message(role="assistant", content=last_answer))
        print()


def main():
    load_dotenv(_PROJECT_ROOT)

    parser = argparse.ArgumentParser(
        description="RAG Agent CLI — 知识库智能问答",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  rag chat              # 精简模式，无日志
  rag chat -d           # 调试模式，显示工具调用与日志
  rag chat -s           # 流式输出
  rag --env production chat
        """,
    )
    parser.add_argument("--env", default="development", help="配置环境 (default: development)")
    sub = parser.add_subparsers(dest="command")

    # chat 子命令
    chat = sub.add_parser("chat", help="交互式对话")
    chat.add_argument("-s", "--stream", action="store_true", help="流式输出")
    chat.add_argument("-d", "--debug", action="store_true", help="调试模式：显示工具调用与详细日志")

    args = parser.parse_args()

    if args.command == "chat":
        asyncio.run(_run_chat(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
