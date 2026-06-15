"""规则型 Query 扩展：为 query 中的通用 UI 词追加同义别名变体。

只做领域无关的 UI 组件词替换（Card/卡片、Tab/标签 等），
不含任何业务领域硬编码映射（那部分由 QueryRewriter / LLM 负责）。
原始 query 始终作为第一个变体，保证行为可回退。
"""
from __future__ import annotations

import re

# UI/组件词 → 别名列表（首项为规范写法，key 统一小写）
_UI_ALIASES: dict[str, list[str]] = {
    "card":   ["card", "卡片", "卡"],
    "tab":    ["tab", "标签", "页签"],
    "banner": ["banner", "轮播图", "横幅"],
    "list":   ["list", "列表"],
    "grid":   ["grid", "网格"],
    "入口":   ["入口", "点击", "跳转"],
    "展示":   ["展示", "显示"],
    "逻辑":   ["逻辑", "规则", "策略"],
    "页面":   ["页面", "视图", "界面"],
}

# 最长优先排序，避免短词（"卡"）先于长词（"卡片"）匹配产生重叠替换
_UI_KEYS_SORTED = sorted(_UI_ALIASES.keys(), key=len, reverse=True)


def expand_query(query: str, max_variants: int = 5) -> list[str]:
    """返回 query 扩展列表，第一项始终为原始 query。

    对 query 中出现的 UI 组件词逐一替换为其同义别名，生成新变体。
    若 max_variants <= 1 或无 UI 词命中，直接返回 [query]。
    """
    query = query.strip()
    if not query or max_variants <= 1:
        return [query]

    variants: list[str] = [query]
    ql = query.lower()

    for key in _UI_KEYS_SORTED:
        if key not in ql:
            continue
        for alias in _UI_ALIASES[key]:
            if len(variants) >= max_variants:
                break
            if alias.lower() in ql:
                continue  # 别名已在 query 中，跳过重复替换
            # 替换首次出现的 UI 词为别名，保留 query 其余部分不变
            new_v = re.sub(re.escape(key), alias, query, count=1, flags=re.IGNORECASE).strip()
            _append_unique(variants, new_v)
        if len(variants) >= max_variants:
            break

    return variants[:max_variants]


def _append_unique(variants: list[str], candidate: str) -> None:
    """仅当 candidate 不在 variants 中时追加（忽略大小写）。"""
    cl = candidate.lower()
    if not any(v.lower() == cl for v in variants):
        variants.append(candidate)
