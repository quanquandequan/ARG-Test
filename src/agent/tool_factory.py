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
from src.agent.tools.search_kb import KnowledgeBaseTool
from src.agent.tools.web_search import WebSearchTool
from src.agent.tools.write_test_cases import WriteTestCasesTool
from src.core.logging import get_logger
from src.llm.base import BaseLLM
from src.retriever.retrieval_engine import RetrievalEngine

logger = get_logger(__name__)

_DEFAULT_TOOLS = ("knowledge_search", "web_search")


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

    factories: dict[str, Callable[[str | None, str | None], BaseTool | None]] = {
        "knowledge_search": _make_knowledge_search,
        "web_search": _make_web_search,
        "write_test_cases": _make_write_test_cases,
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
