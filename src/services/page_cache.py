"""移动端 screen_tool 的页面状态缓存。

避免对未变化页面重复解析或重复调用 VLM。
缓存条目以 XML 页面源码的“结构哈希”为键
（见 ``screen_parser.compute_structure_hash``），因此纯动态内容
（计数器、时间戳）不会导致缓存失效。

TTL 默认 30 秒，适用于大多数手动节奏的测试流程。
执行任何 UI 操作（点击、滑动、输入）后都应使缓存失效，
确保下一次读取屏幕时反映最新状态。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from src.mobile.screen_parser import ParsedScreen


@dataclass
class CachedPage:
    """一份缓存的屏幕快照。"""

    page_hash: str
    parsed_screen: ParsedScreen
    activity: str
    screenshot_base64: str | None
    captured_at: float = field(default_factory=time.monotonic)

    def is_expired(self, ttl_seconds: float) -> bool:
        return (time.monotonic() - self.captured_at) > ttl_seconds


class PageCache:
    """Android 页面状态的 LRU 风格缓存。

    Args:
        ttl_seconds: 缓存条目保持新鲜的时长。
        max_size:    最多保留的不同页面数量。超过限制时优先淘汰最旧条目。
    """

    def __init__(self, ttl_seconds: float = 30.0, max_size: int = 10) -> None:
        self._ttl = ttl_seconds
        self._max = max_size
        self._store: dict[str, CachedPage] = {}

    # ── 对外 API ─────────────────────────────────────────────────────────────

    def get(self, page_hash: str) -> CachedPage | None:
        """返回 *page_hash* 对应的未过期条目；不存在则返回 None。"""
        entry = self._store.get(page_hash)
        if entry is None:
            return None
        if entry.is_expired(self._ttl):
            del self._store[page_hash]
            return None
        return entry

    def put(
        self,
        page_hash: str,
        parsed_screen: ParsedScreen,
        activity: str = "",
        screenshot_base64: str | None = None,
    ) -> CachedPage:
        """插入或刷新缓存条目。"""
        if len(self._store) >= self._max and page_hash not in self._store:
            self._evict_oldest()

        entry = CachedPage(
            page_hash=page_hash,
            parsed_screen=parsed_screen,
            activity=activity,
            screenshot_base64=screenshot_base64,
        )
        self._store[page_hash] = entry
        return entry

    def invalidate(self, page_hash: str | None = None) -> None:
        """使指定条目失效，或清空整个缓存。"""
        if page_hash is None:
            self._store.clear()
        else:
            self._store.pop(page_hash, None)

    def size(self) -> int:
        return len(self._store)

    # ── 内部实现 ─────────────────────────────────────────────────────────────

    def _evict_oldest(self) -> None:
        if not self._store:
            return
        oldest_key = min(self._store, key=lambda k: self._store[k].captured_at)
        del self._store[oldest_key]
