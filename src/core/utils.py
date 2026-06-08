"""共享工具函数：gRPC 噪声抑制、格式化与 SSE 解析。"""

from __future__ import annotations

import json
import os
import re
import threading


# -- gRPC stderr 噪声抑制 ----------------------------------------------------

_GRPC_NOISE = re.compile(
    rb"(?:GOAWAY|too_many_pings|chttp2_transport\.cc|grpc_init)"
)


def suppress_grpc_stderr() -> None:
    """过滤 stderr 中的 gRPC C++ core 噪声（GOAWAY、too_many_pings 等）。

    必须在进程启动早期、导入依赖 gRPC 的库之前调用。
    可安全重复调用（通过模块级守卫保持幂等）。
    """
    _orig_stderr_fd = os.dup(2)
    _rfd, _wfd = os.pipe()
    os.dup2(_wfd, 2)
    os.close(_wfd)

    def _filter():
        try:
            while True:
                chunk = os.read(_rfd, 65536)
                if not chunk:
                    break
                clean = b"\n".join(
                    line
                    for line in chunk.split(b"\n")
                    if line.strip() and not _GRPC_NOISE.search(line)
                )
                if clean:
                    os.write(_orig_stderr_fd, clean + b"\n")
        except OSError:
            pass

    threading.Thread(target=_filter, daemon=True).start()


# -- CJK 感知的显示宽度与表格格式化 ---------------------------------------


def display_width(s: str) -> int:
    """返回显示宽度：CJK 字符 = 2，ASCII = 1。"""
    import unicodedata

    w = 0
    for c in s:
        w += 2 if unicodedata.east_asian_width(c) in ("W", "F") else 1
    return w


def format_answer(text: str) -> str:
    """为终端展示重新对齐 markdown 表格（CJK 感知）。"""
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
                        widths[ci] = max(widths[ci], display_width(cell))

            for row in rows:
                padded = []
                for ci in range(ncols):
                    cell = row[ci] if ci < len(row) else ""
                    pad = widths[ci] - display_width(cell)
                    padded.append(cell + " " * pad)
                out.append("| " + " | ".join(padded) + " |")
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out)


# -- SSE 事件解析 -----------------------------------------------------------


def parse_sse_event(raw: str) -> tuple[str, dict | str]:
    """从原始 SSE 块返回 (event_type, data)。"""
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


# -- dotenv 加载器 ----------------------------------------------------------


def load_dotenv(project_root) -> None:
    """不依赖 python-dotenv，从项目根目录加载 .env 文件。"""
    env_path = project_root / ".env"
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
