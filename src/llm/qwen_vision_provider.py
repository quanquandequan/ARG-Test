"""用于屏幕理解的 Qwen Vision Language Model provider。

仅供 ``screen_tool`` 使用：当 Appium XML 树缺少足够语义信息时，
用于解读 Android 截图。

该 provider 有意保持最小化，只支持图像描述，不支持通用 LLM 工具调用。
主 Agent 仍继续使用 DeepSeek/OpenAI。

API: DashScope OpenAI-compatible endpoint
Model: 可通过 default.yaml 中的 ``qwen_vision.model`` 配置
       （默认："qwen-vl-plus"）
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
    """用于 Android 屏幕描述的 Qwen VL provider。

    接收文件路径或 base64 编码字符串形式的截图，返回可见 UI 元素的结构化描述。

    配置（``configs/default.yaml`` → ``qwen_vision``）：
      model:   模型名称（默认："qwen-vl-plus"）
      api_key: API key（默认读取环境变量 DASHSCOPE_API_KEY）
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
        """API key 已配置时返回 True。"""
        return bool(self._api_key)

    async def describe_screen(
        self,
        screenshot: str | bytes | Path,
        prompt: str | None = None,
    ) -> str:
        """使用 Qwen VL 描述截图。

        Args:
            screenshot: 图片的 base64 字符串、原始字节或文件路径。
            prompt: 自定义 prompt；默认使用结构化 JSON prompt。

        Returns:
            LLM 文本响应（理想情况下为匹配上方 schema 的 JSON）。
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
        """针对截图提出自由格式问题。"""
        return await self.describe_screen(screenshot, prompt=question)


# ── 辅助方法 ─────────────────────────────────────────────────────────────────

def _to_base64(image: str | bytes | Path) -> str:
    """无论输入格式如何，都转换为 base64 字符串。"""
    is_path = isinstance(image, Path) or (
        isinstance(image, str) and len(image) < 500 and "\n" not in image
    )
    if is_path:
        path = Path(image)  # type: ignore[arg-type]
        if path.exists():
            return base64.b64encode(path.read_bytes()).decode()
    if isinstance(image, bytes):
        return base64.b64encode(image).decode()
    # 已经是 base64 字符串
    return image
