"""Text normalization and cleaning for Chinese content."""

import re
import unicodedata


class TextCleaner:
    """Normalize and clean extracted text."""

    # Fullwidth punctuation to halfwidth mapping
    _FULLWIDTH_MAP: dict[int, int] = {}
    for _fw, _hw in [
        (0x3000, 0x0020),  # IDEOGRAPHIC SPACE → SPACE
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

    # Chinese quotes normalization
    _QUOTE_PAIRS = [
        ("“", "”"),  # " "
        ("‘", "’"),  # ' '
        ("「", "」"),  # 「 」
        ("『", "』"),  # 『 』
    ]

    def clean(self, text: str) -> str:
        if not text.strip():
            return ""

        text = unicodedata.normalize("NFKC", text)

        # Normalize fullwidth punctuation
        text = text.translate(self._FULLWIDTH_MAP)

        # Normalize Chinese quotes to standard form
        for left, right in self._QUOTE_PAIRS:
            text = text.replace(left, "“").replace(right, "”")

        # Collapse excessive whitespace but preserve paragraph breaks
        text = re.sub(r"[^\S\n]{2,}", " ", text)
        # Remove lines that are purely whitespace, but preserve newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Normalize leading/trailing whitespace per line
        lines = [line.strip() for line in text.splitlines()]
        text = "\n".join(lines)

        # Remove control characters (except \n, \t)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        return text.strip()
