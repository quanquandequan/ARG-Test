"""Unit tests for screen_parser — no Appium required."""


from src.mobile.screen_parser import (
    ParsedScreen,
    UIElement,
    compute_structure_hash,
    parse_page_source,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

LOGIN_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <android.widget.FrameLayout
      resource-id="android:id/content"
      bounds="[0,0][1080,2340]"
      clickable="false">
    <android.widget.LinearLayout bounds="[0,0][1080,2340]" clickable="false">
      <android.widget.TextView
          text="欢迎登录"
          resource-id="com.example:id/title"
          bounds="[280,200][800,270]"
          clickable="false"/>
      <android.widget.EditText
          text=""
          content-desc="用户名输入框"
          resource-id="com.example:id/username"
          bounds="[80,320][1000,400]"
          clickable="true"
          focusable="true"/>
      <android.widget.EditText
          text=""
          content-desc="密码输入框"
          resource-id="com.example:id/password"
          bounds="[80,420][1000,500]"
          clickable="true"
          focusable="true"/>
      <android.widget.Button
          text="登录"
          resource-id="com.example:id/login_btn"
          bounds="[200,560][880,630]"
          clickable="true"
          enabled="true"/>
      <android.widget.TextView
          text="忘记密码？"
          resource-id="com.example:id/forgot"
          bounds="[400,650][680,700]"
          clickable="true"/>
    </android.widget.LinearLayout>
  </android.widget.FrameLayout>
</hierarchy>
"""

SPARSE_XML = """\
<hierarchy>
  <android.widget.FrameLayout bounds="[0,0][1080,2340]">
    <android.widget.View bounds="[0,0][1080,2340]" resource-id="com.example:id/root"/>
  </android.widget.FrameLayout>
</hierarchy>
"""

EMPTY_XML = ""


# ── Tests: parse_page_source ──────────────────────────────────────────────────

def test_parses_login_page_elements():
    screen = parse_page_source(LOGIN_XML)
    assert isinstance(screen, ParsedScreen)
    # 6 elements: FrameLayout (has resource-id), title, username, password, login_btn, forgot
    assert len(screen.elements) == 6


def test_extracts_text_correctly():
    screen = parse_page_source(LOGIN_XML)
    labels = [e.text for e in screen.elements]
    assert "欢迎登录" in labels
    assert "登录" in labels
    assert "忘记密码？" in labels


def test_extracts_content_desc():
    screen = parse_page_source(LOGIN_XML)
    descs = [e.content_desc for e in screen.elements]
    assert "用户名输入框" in descs
    assert "密码输入框" in descs


def test_clickable_elements():
    screen = parse_page_source(LOGIN_XML)
    clickable = screen.clickable_elements()
    clickable_texts = [e.text or e.content_desc for e in clickable]
    assert "登录" in clickable_texts
    assert "用户名输入框" in clickable_texts


def test_find_by_text_fuzzy():
    screen = parse_page_source(LOGIN_XML)
    el = screen.find_by_text("登录")
    assert el is not None


def test_find_by_text_exact():
    screen = parse_page_source(LOGIN_XML)
    el = screen.find_by_text("登录", exact=True)
    assert el is not None
    assert el.text == "登录"


def test_find_by_resource_id_partial():
    screen = parse_page_source(LOGIN_XML)
    el = screen.find_by_resource_id("login_btn")
    assert el is not None
    assert el.text == "登录"


def test_find_by_resource_id_full():
    screen = parse_page_source(LOGIN_XML)
    el = screen.find_by_resource_id("com.example:id/login_btn")
    assert el is not None


def test_missing_element_returns_none():
    screen = parse_page_source(LOGIN_XML)
    assert screen.find_by_text("不存在的元素") is None
    assert screen.find_by_resource_id("no_such_id") is None


def test_bounds_parsed_correctly():
    screen = parse_page_source(LOGIN_XML)
    # Use exact match to avoid matching "欢迎登录" with fuzzy "登录"
    btn = screen.find_by_text("登录", exact=True)
    assert btn is not None
    assert btn.bounds == [200, 560, 880, 630]


def test_center_computed():
    screen = parse_page_source(LOGIN_XML)
    btn = screen.find_by_text("登录", exact=True)
    assert btn is not None
    assert btn.center == (540, 595)


def test_is_visible_nonzero_bounds():
    screen = parse_page_source(LOGIN_XML)
    for el in screen.elements:
        assert el.is_visible


def test_label_prefers_text_over_id():
    el = UIElement(
        text="登录",
        resource_id="com.example:id/btn",
        class_name="Button",
        content_desc="",
        bounds=[0, 0, 100, 50],
        clickable=True,
        enabled=True,
        checkable=False,
        checked=False,
        focusable=False,
    )
    assert el.label == "登录"


def test_label_falls_back_to_content_desc():
    el = UIElement(
        text="",
        resource_id="com.example:id/btn",
        class_name="Button",
        content_desc="提交按钮",
        bounds=[0, 0, 100, 50],
        clickable=True,
        enabled=True,
        checkable=False,
        checked=False,
        focusable=False,
    )
    assert el.label == "提交按钮"


def test_empty_xml_returns_empty_screen():
    screen = parse_page_source(EMPTY_XML)
    assert len(screen.elements) == 0


def test_sparse_xml_no_text_elements():
    screen = parse_page_source(SPARSE_XML)
    # The sparse XML only has nodes with resource-id but no text/content-desc
    # The root FrameLayout has no identifying info, but the View has a resource-id
    assert isinstance(screen, ParsedScreen)


def test_malformed_xml_returns_empty():
    screen = parse_page_source("<hierarchy><not_closed>")
    assert len(screen.elements) == 0


def test_has_meaningful_content_true():
    screen = parse_page_source(LOGIN_XML)
    assert screen.has_meaningful_content(min_elements=3)


def test_has_meaningful_content_false():
    screen = parse_page_source(SPARSE_XML)
    assert not screen.has_meaningful_content(min_elements=3)


def test_to_agent_summary_structure():
    screen = parse_page_source(LOGIN_XML)
    summary = screen.to_agent_summary()
    assert "element_count" in summary
    assert "clickable_count" in summary
    assert "elements" in summary
    assert summary["element_count"] == 6  # incl. FrameLayout with resource-id
    assert summary["clickable_count"] == 4  # username, password, login_btn, forgot


# ── Tests: compute_structure_hash ─────────────────────────────────────────────

def test_same_xml_produces_same_hash():
    h1 = compute_structure_hash(LOGIN_XML)
    h2 = compute_structure_hash(LOGIN_XML)
    assert h1 == h2
    assert len(h1) == 32  # MD5 hex


def test_different_xml_produces_different_hash():
    h1 = compute_structure_hash(LOGIN_XML)
    h2 = compute_structure_hash(SPARSE_XML)
    assert h1 != h2


def test_structural_hash_ignores_text_changes():
    xml_v1 = LOGIN_XML
    xml_v2 = LOGIN_XML.replace("欢迎登录", "欢迎回来")  # text changed, structure same
    assert compute_structure_hash(xml_v1) == compute_structure_hash(xml_v2)


def test_empty_xml_hash_is_empty_string():
    assert compute_structure_hash("") == ""


def test_malformed_xml_hash_is_empty_string():
    assert compute_structure_hash("<bad>") == ""
