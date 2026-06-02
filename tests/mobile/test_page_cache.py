"""Unit tests for PageCache — no Appium required."""

import time

from src.mobile.screen_parser import ParsedScreen, parse_page_source
from src.services.page_cache import CachedPage, PageCache

# ── Fixtures ──────────────────────────────────────────────────────────────────

_SAMPLE_XML = """\
<hierarchy>
  <android.widget.Button text="OK" resource-id="com.e:id/ok"
    bounds="[100,200][400,250]" clickable="true"/>
</hierarchy>
"""


def _make_cache(ttl: float = 30.0, max_size: int = 10) -> PageCache:
    return PageCache(ttl_seconds=ttl, max_size=max_size)


def _make_screen() -> ParsedScreen:
    return parse_page_source(_SAMPLE_XML)


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_put_and_get():
    cache = _make_cache()
    screen = _make_screen()
    cache.put("hash1", screen, activity=".MainActivity")
    entry = cache.get("hash1")
    assert entry is not None
    assert isinstance(entry, CachedPage)
    assert entry.activity == ".MainActivity"


def test_get_missing_returns_none():
    cache = _make_cache()
    assert cache.get("nonexistent") is None


def test_expired_entry_returns_none():
    cache = _make_cache(ttl=0.01)  # 10 ms TTL
    screen = _make_screen()
    cache.put("hash1", screen)
    time.sleep(0.05)
    assert cache.get("hash1") is None


def test_valid_entry_not_expired():
    cache = _make_cache(ttl=60.0)
    cache.put("hash1", _make_screen())
    assert cache.get("hash1") is not None


def test_invalidate_specific():
    cache = _make_cache()
    cache.put("h1", _make_screen())
    cache.put("h2", _make_screen())
    cache.invalidate("h1")
    assert cache.get("h1") is None
    assert cache.get("h2") is not None


def test_invalidate_all():
    cache = _make_cache()
    cache.put("h1", _make_screen())
    cache.put("h2", _make_screen())
    cache.invalidate()
    assert cache.size() == 0


def test_size_tracking():
    cache = _make_cache()
    assert cache.size() == 0
    cache.put("h1", _make_screen())
    assert cache.size() == 1
    cache.put("h2", _make_screen())
    assert cache.size() == 2


def test_max_size_evicts_oldest():
    cache = _make_cache(max_size=2)
    screen = _make_screen()
    cache.put("h1", screen)
    time.sleep(0.01)
    cache.put("h2", screen)
    time.sleep(0.01)
    cache.put("h3", screen)  # should evict h1
    assert cache.size() == 2
    assert cache.get("h1") is None  # oldest evicted
    assert cache.get("h2") is not None
    assert cache.get("h3") is not None


def test_put_returns_cached_page():
    cache = _make_cache()
    screen = _make_screen()
    entry = cache.put("h1", screen, activity=".SomeActivity", screenshot_base64="abc")
    assert isinstance(entry, CachedPage)
    assert entry.page_hash == "h1"
    assert entry.activity == ".SomeActivity"
    assert entry.screenshot_base64 == "abc"


def test_put_overwrites_existing():
    cache = _make_cache()
    screen1 = _make_screen()
    screen2 = parse_page_source(_SAMPLE_XML)
    cache.put("h1", screen1, activity=".A")
    cache.put("h1", screen2, activity=".B")
    entry = cache.get("h1")
    assert entry is not None
    assert entry.activity == ".B"
    assert cache.size() == 1  # not duplicated


def test_cached_page_is_not_expired_immediately():
    screen = _make_screen()
    entry = CachedPage(
        page_hash="h1",
        parsed_screen=screen,
        activity=".A",
        screenshot_base64=None,
    )
    assert not entry.is_expired(ttl_seconds=30.0)


def test_cached_page_is_expired_after_ttl():
    screen = _make_screen()
    entry = CachedPage(
        page_hash="h1",
        parsed_screen=screen,
        activity=".A",
        screenshot_base64=None,
        captured_at=time.monotonic() - 60.0,  # 60s ago
    )
    assert entry.is_expired(ttl_seconds=30.0)
