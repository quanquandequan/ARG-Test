"""知识库结果二次压缩：从已召回 chunk 中抽取与 query 相关的原文字段/句子。

规则抽取，不调用 LLM，不改写业务含义，所有输出均为原文片段，保留可追溯证据。
压缩在 rerank 选完最终结果之后、格式化之前运行，不影响召回/排序策略。
"""
from __future__ import annotations

import re

from src.vectordb.base import SearchResult

# ── 结构化行检测 ──────────────────────────────────────────────────────────────
# Excel chunk 固定格式：[Sheet: xxx] [Row N] 字段: 值 | 字段: 值
_ROW_HEADER_RE = re.compile(r"^\[Sheet:[^\]]+\]\s*\[Row\s*\d+\]\s*")

# ── Excel 测试用例字段配置 ─────────────────────────────────────────────────────
# 无论关键词是否命中都保留（定位/优先级）
_ANCHOR_FIELDS: frozenset[str] = frozenset({
    "#", "目录路径", "标题", "用例名", "用例标题", "优先级",
})
# 标题/路径命中时额外展开（步骤/预期）
_EXPAND_FIELDS: frozenset[str] = frozenset({
    "操作步骤", "测试步骤", "步骤", "预期结果", "期望结果", "预期",
})
# 命中后触发展开的字段名（包含这些词的字段算"标题命中"）
_TITLE_FIELDS: frozenset[str] = frozenset({
    "目录路径", "标题", "用例名", "用例标题",
})

# ── 关键词提取配置 ────────────────────────────────────────────────────────────
# 中文高频停用词（精简集，不用外部词表）
_STOPWORDS: frozenset[str] = frozenset({
    "的", "了", "是", "在", "与", "及", "等", "或", "和", "对",
    "中", "有", "为", "到", "于", "从", "以", "其", "将", "被",
    "也", "都", "该", "此", "这", "那", "时", "后", "前", "上",
    "下", "内", "外", "不", "无", "可", "应", "需", "要", "如",
    "已", "并", "按", "则", "用", "所", "各", "把", "使", "他",
    "她", "它", "们", "我", "你",
})
_MIN_KW_LEN = 2          # 关键词最短字符数
_EXPAND_MAX_CHARS = 300  # 展开字段（步骤/预期）最多保留字符数
_FALLBACK_MAX_CHARS = 500  # 无命中时回退保留字符数
_TEXT_MAX_EVIDENCE = 3   # 自由文本最多保留命中 evidence 片段数


# ── 公开接口 ──────────────────────────────────────────────────────────────────

def extract_keywords(query: str) -> set[str]:
    """从 query 提取关键词：jieba 切词 + 停用词过滤 + 保留原 query 全文短语。"""
    import jieba
    words: set[str] = set()
    q = query.strip().lower()
    if q:
        words.add(q)  # 全文短语用于精确命中
    for token in jieba.lcut(query):
        t = token.strip().lower()
        if len(t) >= _MIN_KW_LEN and t not in _STOPWORDS:
            words.add(t)
    return words


def compress_results(results: list[SearchResult], query: str) -> dict[str, str]:
    """返回 {result.id: compressed_content}。

    id 缺失或压缩失败时不写入字典，调用方回退展示原 result.content。
    """
    keywords = extract_keywords(query)
    compressed: dict[str, str] = {}
    for result in results:
        rid = result.id or ""
        if not rid:
            continue
        try:
            if _is_structured_row(result.content):
                compressed[rid] = _compress_structured(result.content, keywords)
            else:
                compressed[rid] = _compress_text(result.content, keywords)
        except Exception:
            # 压缩出错时静默跳过，调用方展示原始内容
            pass
    return compressed


# ── 内部实现 ──────────────────────────────────────────────────────────────────

def _is_structured_row(content: str) -> bool:
    """判断是否为 [Sheet: X] [Row N] 结构化行（来自 xlsx reader）。"""
    return bool(_ROW_HEADER_RE.match(content))


def _parse_kv_pairs(body: str) -> list[tuple[str, str]]:
    """将 'field: value | field: value' 解析为有序列表。

    用 ' | '（含空格）分割，避免误切 URL 或中文中的 | 号。
    """
    pairs: list[tuple[str, str]] = []
    for segment in body.split(" | "):
        segment = segment.strip()
        if ": " in segment:
            field, _, value = segment.partition(": ")
            pairs.append((field.strip(), value.strip()))
        elif segment:
            # 无冒号字段（如纯数字行号），以整体作为值
            pairs.append((segment, ""))
    return pairs


def _compress_structured(content: str, keywords: set[str]) -> str:
    """压缩结构化行（Excel 测试用例 / Bug 行）。

    策略：
    - Bug 等字段数 ≤ 3 的短行直接返回（已经很精简，无需压缩）
    - 测试用例行：锚定字段始终保留；关键词命中字段保留；
      标题/路径命中时额外展开步骤/预期（最多 300 字符）
    - 无命中时回退为原行前 500 字符
    """
    m = _ROW_HEADER_RE.match(content)
    header = m.group(0).rstrip() if m else ""
    body = content[m.end():] if m else content

    pairs = _parse_kv_pairs(body)
    if not pairs:
        return content[:_FALLBACK_MAX_CHARS]

    # 字段极少的行（Bug: Bug Key + 摘要）直接返回，避免过度压缩
    if len(pairs) <= 3:
        return content

    kept: list[tuple[str, str]] = []
    matched_fields: list[str] = []
    title_hit = False

    for field, value in pairs:
        is_anchor = any(a in field for a in _ANCHOR_FIELDS)
        hits_kw = any(kw in field.lower() or kw in value.lower() for kw in keywords)

        if is_anchor or hits_kw:
            kept.append((field, value))
        if hits_kw:
            matched_fields.append(field)
            # 标题/路径命中 → 后续展开步骤/预期
            if any(tf in field for tf in _TITLE_FIELDS):
                title_hit = True

    # 标题命中：补充展开步骤/预期字段（若尚未被关键词命中收录）
    if title_hit:
        for field, value in pairs:
            if any(ef in field for ef in _EXPAND_FIELDS):
                if not any(f == field for f, _ in kept):
                    kept.append((field, value[:_EXPAND_MAX_CHARS]))

    if not kept:
        # 无任何命中：回退原行前 500 字符（低置信证据）
        return content[:_FALLBACK_MAX_CHARS]

    lines: list[str] = []
    # 命中字段摘要行（去重保持顺序）
    if matched_fields:
        seen: set[str] = set()
        deduped = [f for f in matched_fields if not (f in seen or seen.add(f))]  # type: ignore[func-returns-value]
        lines.append(f"[命中字段: {' / '.join(deduped[:4])}]")
    if header:
        lines.append(header)
    for field, value in kept:
        lines.append(f"{field}: {value}")
    return "\n".join(lines)


def _compress_text(content: str, keywords: set[str]) -> str:
    """压缩自由文本（XMind / PDF / TXT）：保留最多 3 条命中 evidence 片段。

    按换行和中文句界切分，优先保留包含最多关键词的片段，
    输出按原文顺序排列保持可读性。无命中时回退为前 500 字符。
    """
    sentences = [s.strip() for s in re.split(r"[\n。；;！!？?]+", content) if s.strip()]
    if not sentences:
        return content[:_FALLBACK_MAX_CHARS]

    # 统计每个片段命中的关键词数
    scored: list[tuple[int, int, str]] = []  # (命中数, 原序号, 文本)
    for idx, sent in enumerate(sentences):
        sl = sent.lower()
        score = sum(1 for kw in keywords if kw in sl)
        if score > 0:
            scored.append((score, idx, sent))

    if not scored:
        return content[:_FALLBACK_MAX_CHARS]

    # 按命中数降序取前 N，再按原序排列保证可读性
    top = sorted(scored, key=lambda x: -x[0])[:_TEXT_MAX_EVIDENCE]
    top_ordered = sorted(top, key=lambda x: x[1])
    return " | ".join(s for _, _, s in top_ordered)
