"""Shared utility functions — gRPC noise suppression, formatting, SSE parsing."""

from __future__ import annotations

import json
import os
import re
import threading


# -- gRPC stderr noise suppression ------------------------------------------

_GRPC_NOISE = re.compile(
    rb"(?:GOAWAY|too_many_pings|chttp2_transport\.cc|grpc_init)"
)


def suppress_grpc_stderr() -> None:
    """Filter gRPC C++ core noise from stderr (GOAWAY, too_many_pings, etc.).

    Must be called early at process startup, before importing gRPC-dependent
    libraries.  Safe to call multiple times (idempotent via module-level guard).
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


# -- CJK-aware display width and table formatting -------------------------


def display_width(s: str) -> int:
    """Return display width: CJK chars = 2, ASCII = 1."""
    import unicodedata

    w = 0
    for c in s:
        w += 2 if unicodedata.east_asian_width(c) in ("W", "F") else 1
    return w


def format_answer(text: str) -> str:
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


# -- SSE event parsing -----------------------------------------------------


def parse_sse_event(raw: str) -> tuple[str, dict | str]:
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


# -- dotenv loader ----------------------------------------------------------


def load_dotenv(project_root) -> None:
    """Load .env file from project root without python-dotenv dependency."""
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
