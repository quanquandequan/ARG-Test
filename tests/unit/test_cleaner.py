"""Unit tests for TextCleaner."""

from src.ingestion.cleaner import TextCleaner


def test_empty_input_returns_empty():
    assert TextCleaner().clean("") == ""
    assert TextCleaner().clean("   \n  \t  ") == ""


def test_fullwidth_punctuation_to_halfwidth():
    cleaner = TextCleaner()
    out = cleaner.clean("你好，世界！这是（测试）：第１个例子。")
    assert "，" not in out
    assert "！" not in out
    assert "（" not in out and "）" not in out
    assert "：" not in out
    # NFKC converts fullwidth digits to halfwidth
    assert "1" in out


def test_control_characters_removed():
    cleaner = TextCleaner()
    raw = "正文\x00有控制字符\x07这里\x1f结尾"
    out = cleaner.clean(raw)
    assert "\x00" not in out
    assert "\x07" not in out
    assert "\x1f" not in out
    assert "正文" in out and "结尾" in out


def test_excess_whitespace_collapsed_paragraph_preserved():
    cleaner = TextCleaner()
    raw = "段落一    多空格\n\n\n\n段落二"
    out = cleaner.clean(raw)
    assert "    " not in out  # collapsed
    assert "\n\n\n" not in out  # paragraph break clamped to 2
    assert "段落一" in out and "段落二" in out


def test_chinese_quotes_normalized():
    cleaner = TextCleaner()
    raw = "他说‘你好’和「再见」与『结束』"
    out = cleaner.clean(raw)
    # All quote variants are mapped to “”
    assert "‘" not in out and "’" not in out
    assert "「" not in out and "」" not in out
    assert "『" not in out and "』" not in out
