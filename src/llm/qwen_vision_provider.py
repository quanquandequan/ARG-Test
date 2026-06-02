"""Qwen Vision Language Model provider for screen understanding.

Used exclusively by ``screen_tool`` to interpret Android screenshots when
the Appium XML tree lacks sufficient semantic information.

This provider is intentionally minimal — it only supports image description,
not general LLM tool-calling.  The main Agent continues to use DeepSeek/OpenAI.

API: DashScope OpenAI-compatible endpoint
Model: configurable via ``qwen_vision.model`` in default.yaml
       (default: "qwen-vl-plus")
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

from src.core.config import get_config
from src.core.exceptions import LLMError
from src.core.logging import get_logger

logger = get_logger(__name__)

_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

_DEFAULT_SCREEN_PROMPT = """\
请分析这个Android应用截图，识别所有可见的UI元素。

只输出JSON对象，不加任何Markdown标记：
{
  "page_name": "页面名称（如：登录页、首页、商品详情页）",
  "elements": [
    {
      "type": "button|input|text|image|list_item|tab|checkbox|icon",
      "text": "元素上的文字（无则填空字符串）",
      "description": "语义描述（如：登录按钮、用户名输入框）",
      "position": (
        "top_left|top_center|top_right|center_left|center"
        "|center_right|bottom_left|bottom_center|bottom_right"
      )
    }
  ]
}

注意：
- 只列出对测试有意义的元素（可点击的、有文字的、可输入的）
- position 描述元素在屏幕上的大致位置
"""


class QwenVisionProvider:
    """Qwen VL provider for Android screen description.

    Accepts screenshots as file paths or base64-encoded strings and returns
    a structured description of visible UI elements.

    Configuration (``configs/default.yaml`` → ``qwen_vision``):
      model:   model name (default: "qwen-vl-plus")
      api_key: API key (default: env DASHSCOPE_API_KEY)
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        screen_prompt: str | None = None,
        max_tokens: int = 1024,
    ):
        cfg = get_config().get("qwen_vision", {})
        self._model = model or cfg.get("model", "qwen-vl-plus")
        self._api_key = (
            api_key
            or cfg.get("api_key")
            or os.environ.get("DASHSCOPE_API_KEY", "")
            or os.environ.get("QWEN_API_KEY", "")
        )
        self._screen_prompt = (
            screen_prompt
            or cfg.get("screen_description_prompt", "")
            or _DEFAULT_SCREEN_PROMPT
        )
        self._max_tokens = int(cfg.get("max_tokens", max_tokens))
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            if not self._api_key:
                raise LLMError(
                    "Qwen VL API key not set. "
                    "Set DASHSCOPE_API_KEY environment variable or "
                    "qwen_vision.api_key in configs/default.yaml"
                )
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=_DASHSCOPE_BASE_URL,
            )
        return self._client

    def is_available(self) -> bool:
        """Return True if the API key is configured."""
        return bool(self._api_key)

    async def describe_screen(
        self,
        screenshot: str | bytes | Path,
        prompt: str | None = None,
    ) -> str:
        """Describe a screenshot using Qwen VL.

        Args:
            screenshot: Base64 string, raw bytes, or file path to the image.
            prompt: Custom prompt. Defaults to the structured JSON prompt.

        Returns:
            LLM text response (ideally JSON matching the schema above).
        """
        image_base64 = _to_base64(screenshot)
        used_prompt = prompt or self._screen_prompt

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        },
                    },
                    {"type": "text", "text": used_prompt},
                ],
            }
        ]

        try:
            response = await self._get_client().chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=self._max_tokens,
            )
            result = response.choices[0].message.content or ""
            logger.debug("qwen_vlm_response", chars=len(result))
            return result
        except Exception as e:
            raise LLMError(f"Qwen VL API error: {e}") from e

    async def describe_screen_raw(
        self,
        screenshot: str | bytes | Path,
        question: str,
    ) -> str:
        """Ask a free-form question about a screenshot."""
        return await self.describe_screen(screenshot, prompt=question)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_base64(image: str | bytes | Path) -> str:
    """Convert image to base64 string regardless of input format."""
    is_path = isinstance(image, Path) or (
        isinstance(image, str) and len(image) < 500 and "\n" not in image
    )
    if is_path:
        path = Path(image)  # type: ignore[arg-type]
        if path.exists():
            return base64.b64encode(path.read_bytes()).decode()
    if isinstance(image, bytes):
        return base64.b64encode(image).decode()
    # Already a base64 string
    return image
