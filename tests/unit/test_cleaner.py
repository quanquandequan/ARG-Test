"""TextCleaner 单元测试。"""

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
    # NFKC 会将全角数字转换为半角
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
    assert "    " not in out  # 已折叠
    assert "\n\n\n" not in out  # 段落换行限制为 2 个
    assert "段落一" in out and "段落二" in out


def test_chinese_quotes_normalized():
    cleaner = TextCleaner()
    raw = "他说‘你好’和「再见」与『结束』"
    out = cleaner.clean(raw)
    # 所有引号变体都会映射为 “”
    assert "‘" not in out and "’" not in out
    assert "「" not in out and "」" not in out
    assert "『" not in out and "』" not in out
