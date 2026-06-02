"""Build Agent tools from config-driven tool names.

Tool configuration supports two formats (both in ``agent.tools`` YAML list):

  # Simple string — uses code defaults for description and prompts
  - knowledge_search

  # Object — overrides description and/or tool-internal system_prompt
  - name: write_test_cases
    description: |
      根据需求文档生成测试用例...（覆盖代码默认值）
    system_prompt: |
      你是一名资深测试工程师...（工具调用 LLM 时的提示词）
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.agent.base_tool import BaseTool
from src.agent.tools.analyze_requirements import AnalyzeRequirementsTool
from src.agent.tools.mobile.action_tool import ActionTool
from src.agent.tools.mobile.assertion_tool import AssertionTool
from src.agent.tools.mobile.device_tool import DeviceTool
from src.agent.tools.mobile.screen_tool import ScreenTool
from src.agent.tools.requirement_parser import RequirementParserTool
from src.agent.tools.requirement_reviewer import RequirementReviewerTool
from src.agent.tools.search_kb import KnowledgeBaseTool
from src.agent.tools.web_search import WebSearchTool
from src.agent.tools.write_test_cases import WriteTestCasesTool
from src.core.logging import get_logger
from src.llm.base import BaseLLM
from src.mobile.driver import AppiumDriverManager
from src.retriever.retrieval_engine import RetrievalEngine
from src.services.page_cache import PageCache

logger = get_logger(__name__)

_DEFAULT_TOOLS = ("knowledge_search", "web_search")

# Shared mobile singletons — created lazily the first time a mobile tool is built.
# All four mobile tools reference the same driver manager and page cache so that
# DeviceTool.connect() is visible to ScreenTool / ActionTool / AssertionTool.
_shared_driver_manager: AppiumDriverManager | None = None
_shared_page_cache: PageCache | None = None


def _get_mobile_singletons() -> tuple[AppiumDriverManager, PageCache]:
    global _shared_driver_manager, _shared_page_cache
    if _shared_driver_manager is None:
        _shared_driver_manager = AppiumDriverManager()
    if _shared_page_cache is None:
        _shared_page_cache = PageCache()
    return _shared_driver_manager, _shared_page_cache


def _parse_tool_config(entry: Any) -> tuple[str, str | None, str | None]:
    """Return ``(name, description_override, system_prompt)`` from a config entry.

    Accepts either a plain string or a dict/DictConfig with at least a ``name`` key.
    """
    if isinstance(entry, str):
        return entry, None, None

    # dict or OmegaConf DictConfig
    name = str(entry.get("name", "")).strip()
    description = entry.get("description") or None
    system_prompt = entry.get("system_prompt") or None

    if description:
        description = str(description).strip() or None
    if system_prompt:
        system_prompt = str(system_prompt).strip() or None

    return name, description, system_prompt


def build_agent_tools(
    retrieval_engine: RetrievalEngine,
    tool_configs: list[Any] | None = None,
    llm: BaseLLM | None = None,
) -> list[BaseTool]:
    """Instantiate tools listed in ``agent.tools`` config (order preserved).

    Args:
        retrieval_engine: Used by ``knowledge_search``.
        tool_configs: Ordered list from config.  Each entry is either a plain
            string (tool name) or a dict with ``name``, optional
            ``description``, and optional ``system_prompt`` keys.
            Defaults to ``_DEFAULT_TOOLS``.
        llm: Required only for LLM-powered tools (e.g. ``write_test_cases``).
             If *None* and such a tool is requested, it is skipped with a warning.
    """
    configs = tool_configs or list(_DEFAULT_TOOLS)

    def _build_qwen_vlm():
        """Return a QwenVisionProvider if configured, else None."""
        try:
            from src.llm.qwen_vision_provider import QwenVisionProvider

            vlm = QwenVisionProvider()
            if vlm.is_available():
                return vlm
            logger.info("qwen_vlm_not_configured_screen_tool_xml_only")
            return None
        except Exception as exc:
            logger.warning("qwen_vlm_init_failed", error=str(exc))
            return None

    def _require_llm(tool_name: str) -> BaseLLM | None:
        if llm is None:
            logger.warning("llm_required_for_tool_skipped", tool=tool_name)
        return llm

    # Factories receive (description_override, system_prompt) as context
    def _make_knowledge_search(_desc: str | None, _sp: str | None) -> BaseTool:
        return KnowledgeBaseTool(retrieval_engine)

    def _make_web_search(_desc: str | None, _sp: str | None) -> BaseTool:
        return WebSearchTool()

    def _make_write_test_cases(
        _desc: str | None, sys_prompt: str | None
    ) -> BaseTool | None:
        _llm = _require_llm("write_test_cases")
        if _llm is None:
            return None
        return WriteTestCasesTool(_llm, system_prompt=sys_prompt or None)

    def _make_analyze_requirements(
        _desc: str | None, sys_prompt: str | None
    ) -> BaseTool | None:
        _llm = _require_llm("analyze_requirements")
        if _llm is None:
            return None
        return AnalyzeRequirementsTool(_llm, system_prompt=sys_prompt or None)

    def _make_requirement_parser(
        _desc: str | None, sys_prompt: str | None
    ) -> BaseTool | None:
        _llm = _require_llm("requirement_parser")
        if _llm is None:
            return None
        return RequirementParserTool(_llm, system_prompt=sys_prompt or None)

    def _make_requirement_reviewer(
        _desc: str | None, sys_prompt: str | None
    ) -> BaseTool | None:
        _llm = _require_llm("requirement_reviewer")
        if _llm is None:
            return None
        return RequirementReviewerTool(_llm, system_prompt=sys_prompt or None)

    def _make_device_tool(_desc: str | None, _sp: str | None) -> BaseTool:
        driver_mgr, _ = _get_mobile_singletons()
        return DeviceTool(driver_manager=driver_mgr)

    def _make_screen_tool(_desc: str | None, _sp: str | None) -> BaseTool:
        driver_mgr, page_cache = _get_mobile_singletons()
        vlm = _build_qwen_vlm()
        return ScreenTool(driver_manager=driver_mgr, page_cache=page_cache, vlm=vlm)

    def _make_action_tool(_desc: str | None, _sp: str | None) -> BaseTool:
        driver_mgr, page_cache = _get_mobile_singletons()
        return ActionTool(driver_manager=driver_mgr, page_cache=page_cache)

    def _make_assertion_tool(_desc: str | None, _sp: str | None) -> BaseTool:
        driver_mgr, _ = _get_mobile_singletons()
        return AssertionTool(driver_manager=driver_mgr)

    factories: dict[str, Callable[[str | None, str | None], BaseTool | None]] = {
        "knowledge_search": _make_knowledge_search,
        "web_search": _make_web_search,
        "write_test_cases": _make_write_test_cases,
        "analyze_requirements": _make_analyze_requirements,
        "requirement_parser": _make_requirement_parser,
        "requirement_reviewer": _make_requirement_reviewer,
        "device_tool": _make_device_tool,
        "screen_tool": _make_screen_tool,
        "action_tool": _make_action_tool,
        "assertion_tool": _make_assertion_tool,
    }

    tools: list[BaseTool] = []
    for entry in configs:
        name, description, system_prompt = _parse_tool_config(entry)
        if not name:
            logger.warning("tool_config_missing_name", entry=str(entry))
            continue

        factory = factories.get(name)
        if factory is None:
            logger.warning("unknown_agent_tool", tool=name)
            continue

        tool = factory(description, system_prompt)
        if tool is None:
            continue

        # Apply YAML-level description override (affects what the LLM sees)
        if description:
            tool.override_description(description)

        tools.append(tool)

    return tools
