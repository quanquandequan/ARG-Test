"""轻量级 Prompt YAML 加载器。"""

from __future__ import annotations

from collections.abc import Iterable
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from src.core.exceptions import ConfigurationError

_PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"


@cache
def load_prompt_file(name: str) -> dict[str, Any]:
    """按文件名读取 ``prompts/<name>.yaml``。"""
    clean_name = name.strip()
    if not clean_name or "/" in clean_name or "\\" in clean_name:
        raise ConfigurationError(f"非法 prompt 文件名：{name}")

    path = _PROMPT_DIR / f"{clean_name}.yaml"
    if not path.exists():
        raise ConfigurationError(f"Prompt 文件不存在：{path}")

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Prompt YAML 解析失败：{path}") from exc

    if not isinstance(data, dict):
        raise ConfigurationError(f"Prompt 文件必须是 YAML 对象：{path}")

    prompt_id = str(data.get("id", "")).strip()
    if prompt_id != clean_name:
        raise ConfigurationError(
            f"Prompt id 不匹配：{path} 中 id={prompt_id!r}，期望 {clean_name!r}"
        )
    return data


def require_prompt_fields(name: str, fields: Iterable[str]) -> dict[str, Any]:
    """读取 prompt 文件并校验必填文本字段。"""
    data = load_prompt_file(name)
    missing = [
        field for field in fields
        if not isinstance(data.get(field), str) or not data.get(field, "").strip()
    ]
    if missing:
        joined = ", ".join(missing)
        raise ConfigurationError(f"Prompt 文件 {name}.yaml 缺少必填字段：{joined}")
    return data
