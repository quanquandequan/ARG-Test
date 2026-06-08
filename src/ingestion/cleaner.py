"""中文内容的文本规范化与清洗。"""

import re
import unicodedata


class TextCleaner:
    """规范化并清洗抽取出的文本。"""

    # 全角标点到半角标点的映射
    _FULLWIDTH_MAP: dict[int, int] = {}
    for _fw, _hw in [
        (0x3000, 0x0020),  # 全角空格 → 半角空格
        (0xFF0C, 0x002C),  # ， → ,
        (0xFF0E, 0x002E),  # ． → .
        (0xFF1A, 0x003A),  # ： → :
        (0xFF1B, 0x003B),  # ； → ;
        (0xFF01, 0x0021),  # ！ → !
        (0xFF1F, 0x003F),  # ？ → ?
        (0xFF08, 0x0028),  # （ → (
        (0xFF09, 0x0029),  # ） → )
        (0xFF3C, 0x005C),  # ＼ → \
        (0xFF0F, 0x002F),  # ／ → /
        (0xFF05, 0x0025),  # ％ → %
    ]:
        _FULLWIDTH_MAP[_fw] = _hw

    # 中文引号规范化
    _QUOTE_PAIRS = [
        ("‘", "’"),  # ' '
        ("「", "」"),  # 「 」
        ("『", "』"),  # 『 』
    ]

    def clean(self, text: str) -> str:
        if not text.strip():
            return ""

        text = unicodedata.normalize("NFKC", text)

        # 规范化全角标点
        text = text.translate(self._FULLWIDTH_MAP)

        # 将中文引号规范化为标准形式
        for left, right in self._QUOTE_PAIRS:
            text = text.replace(left, "“").replace(right, "”")

        # 折叠过多空白，同时保留段落换行
        text = re.sub(r"[^\S\n]{2,}", " ", text)
        # 移除纯空白行，但保留换行结构
        text = re.sub(r"\n{3,}", "\n\n", text)
        # 规范化每行首尾空白
        lines = [line.strip() for line in text.splitlines()]
        text = "\n".join(lines)

        # 移除控制字符（保留 \n、\t）
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        return text.strip()
