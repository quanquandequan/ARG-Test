"""Page-state cache for the mobile screen tool.

Avoids re-parsing / re-invoking the VLM for a screen that hasn't changed.
Cache entries are keyed by the *structural* hash of the XML page source
(see ``screen_parser.compute_structure_hash``), so purely dynamic content
(counters, timestamps) does not bust the cache.

TTL defaults to 30 seconds — appropriate for most manual-paced test flows.
After any UI action (tap, swipe, input) the cache should be invalidated so
the next screen read reflects the updated state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from src.mobile.screen_parser import ParsedScreen


@dataclass
class CachedPage:
    """One cached screen snapshot."""

    page_hash: str
    parsed_screen: ParsedScreen
    activity: str
    screenshot_base64: str | None
    captured_at: float = field(default_factory=time.monotonic)

    def is_expired(self, ttl_seconds: float) -> bool:
        return (time.monotonic() - self.captured_at) > ttl_seconds


class PageCache:
    """LRU-style cache for Android page states.

    Args:
        ttl_seconds: How long a cached entry is considered fresh.
        max_size:    Maximum number of distinct pages to keep.  Oldest entries
                     are evicted first when the limit is exceeded.
    """

    def __init__(self, ttl_seconds: float = 30.0, max_size: int = 10) -> None:
        self._ttl = ttl_seconds
        self._max = max_size
        self._store: dict[str, CachedPage] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def get(self, page_hash: str) -> CachedPage | None:
        """Return a non-expired entry for *page_hash*, or None."""
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
        """Insert or refresh a cache entry."""
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
        """Invalidate a specific entry or clear the entire cache."""
        if page_hash is None:
            self._store.clear()
        else:
            self._store.pop(page_hash, None)

    def size(self) -> int:
        return len(self._store)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _evict_oldest(self) -> None:
        if not self._store:
            return
        oldest_key = min(self._store, key=lambda k: self._store[k].captured_at)
        del self._store[oldest_key]
