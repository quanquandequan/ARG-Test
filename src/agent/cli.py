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

# 答案合成是事实抽取/引用任务而非创意生成，用 0.0 保证确定性，
# 避免全局 llm.temperature(0.3) 带来的偶发幻觉（参考 query_rewriter 的同类取舍）
_CHAT_TEMPERATURE = 0.0
# 确保 `python -m src.agent.cli` 和 `pip install -e` 时项目根目录都在 sys.path 中
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _print_agent_answer(answer: str) -> None:
    """统一打印 Agent 最终回答。"""
    formatted = format_answer(answer)
    if formatted.startswith("|"):
        print(f"Agent:\n{formatted}")
    else:
        print(f"Agent: {formatted}")


def _print_debug_steps(steps, debug: bool) -> None:
    """调试模式下打印工具调用明细。"""
    if not debug or not steps:
        return
    for step in steps:
        if step.tool_call:
            print(
                f"   [TOOL] {step.tool_call.name}({step.tool_call.arguments}) "
                f"→ {step.duration_ms:.0f}ms",
                file=sys.stderr,
            )


async def _run_requirement_clarification_chain(
    agent,
    *,
    history: list,
    draft_answer: str,
    draft_steps: list,
    pending_goal,
    original_query: str,
    debug: bool,
) -> list[str]:
    """draft 澄清后自动执行 final，并按原始目标衔接用例生成。返回各阶段回答文本。"""
    from src.agent.requirement_flow import (
        RequirementFlowSession,
        RequirementGoal,
        build_cli_design_cases_query,
        build_cli_final_query,
        collect_clarification_answers_interactive,
        extract_analysis_json_path,
        extract_draft_tool_args,
        goal_to_generation_mode,
        parse_clarification_questions,
    )
    from src.llm.types import Message  # noqa: PLC0415

    goal = pending_goal or RequirementGoal.ANALYSIS_ONLY
    draft_args = extract_draft_tool_args(draft_steps)
    if not draft_args:
        draft_args = {"requirement": original_query, "analysis_mode": "draft"}

    questions = parse_clarification_questions(draft_answer)
    clarification_answers = collect_clarification_answers_interactive(questions)
    if clarification_answers is None:
        print("Agent: 已取消需求澄清，未生成确认版 JSON。\n")
        return [draft_answer]

    session = RequirementFlowSession(
        goal=goal,
        draft_args=draft_args,
        questions=questions,
        original_query=original_query,
    )
    final_query = build_cli_final_query(session, clarification_answers)
    final_result = await agent.run(
        query=final_query,
        history=history,
        temperature=_CHAT_TEMPERATURE,
    )
    _print_debug_steps(final_result.steps, debug)
    _print_agent_answer(final_result.answer)

    history.append(Message(role="user", content="（已提交澄清答复，生成确认版需求分析）"))
    history.append(Message(role="assistant", content=final_result.answer))

    answers = [draft_answer, final_result.answer]

    if goal in {RequirementGoal.MANUAL_CASES, RequirementGoal.AUTOMATION_CASES}:
        json_path = extract_analysis_json_path(final_result.answer)
        if not json_path:
            print(
                "Agent: 未从确认版分析结果中解析到 JSON 路径，"
                "请手动提供 analysis_json_path 后调用 design_test_cases。\n"
            )
            return answers

        design_query = build_cli_design_cases_query(
            json_path,
            goal_to_generation_mode(goal),
            str(draft_args.get("module", "") or ""),
        )
        design_result = await agent.run(
            query=design_query,
            history=history,
            temperature=_CHAT_TEMPERATURE,
        )
        _print_debug_steps(design_result.steps, debug)
        _print_agent_answer(design_result.answer)

        history.append(
            Message(
                role="user",
                content=(
                    "（基于确认版需求分析 JSON 生成"
                    f"{'自动化' if goal == RequirementGoal.AUTOMATION_CASES else '人工'}测试用例）"
                ),
            )
        )
        history.append(Message(role="assistant", content=design_result.answer))
        answers.append(design_result.answer)

    print()
    return answers


async def _run_chat(args):
    from src.agent.requirement_flow import (
        detect_requirement_goal,
        is_draft_pending_confirmation,
    )
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

    print("\nRAG Agent Chat — 输入 /quit 退出, /clear 清除历史\n")
    if not args.stream:
        print(
            "提示：需求分析 draft 后会进入逐条澄清；"
            "澄清中可用 /skip 跳过、/done 提前完成、/cancel 取消。\n"
        )

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

        pending_goal = detect_requirement_goal(query)
        last_answer = ""
        last_steps: list = []

        # 单轮请求容错：LLM/网络的瞬时故障（如 DeepSeek 500）或用户中途 Ctrl+C
        # 取消，都只应中断本轮，不应让整个交互式会话崩溃、丢失历史。
        try:
            if args.stream:
                print("Agent: ", end="", flush=True)
                async for event in agent.run_stream(
                    query=query, history=history, temperature=_CHAT_TEMPERATURE
                ):
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
                result = await agent.run(
                    query=query, history=history, temperature=_CHAT_TEMPERATURE
                )
                last_answer = result.answer
                last_steps = result.steps
                _print_debug_steps(last_steps, args.debug)
                _print_agent_answer(last_answer)
        except KeyboardInterrupt:
            print("\nAgent: 本轮请求已取消。\n")
            continue
        except Exception as e:  # noqa: BLE001 — CLI 顶层兜底，任何异常都不应崩溃会话
            print(f"\nAgent: 调用失败（{type(e).__name__}: {e}），请重试。\n")
            continue

        history.append(Message(role="user", content=query))
        history.append(Message(role="assistant", content=last_answer))

        # draft 澄清交互：非流式下自动进入逐条问答（未识别目标时默认只输出知识图谱）
        if not args.stream and is_draft_pending_confirmation(last_answer):
            try:
                chain_answers = await _run_requirement_clarification_chain(
                    agent,
                    history=history,
                    draft_answer=last_answer,
                    draft_steps=last_steps,
                    pending_goal=pending_goal,
                    original_query=query,
                    debug=args.debug,
                )
            except KeyboardInterrupt:
                print("\nAgent: 已取消后续澄清流程。\n")
                continue
            except Exception as e:  # noqa: BLE001 — 同上，避免澄清链路的瞬时故障拖垮会话
                print(f"\nAgent: 澄清流程调用失败（{type(e).__name__}: {e}），请重试。\n")
                continue
            if len(chain_answers) > 1:
                # 历史已在 chain 内追加 final / design 轮次
                continue

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
