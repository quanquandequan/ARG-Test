"""Build Agent tools from config-driven tool names."""

from __future__ import annotations

from collections.abc import Callable

from src.agent.base_tool import BaseTool
from src.agent.tools.search_kb import KnowledgeBaseTool
from src.agent.tools.web_search import WebSearchTool
from src.core.logging import get_logger
from src.retriever.retrieval_engine import RetrievalEngine

logger = get_logger(__name__)

_DEFAULT_TOOLS = ("knowledge_search", "web_search")


def build_agent_tools(
    retrieval_engine: RetrievalEngine,
    tool_names: list[str] | None = None,
) -> list[BaseTool]:
    """Instantiate tools listed in ``agent.tools`` config (order preserved)."""
    names = tool_names or list(_DEFAULT_TOOLS)
    factories: dict[str, Callable[[], BaseTool]] = {
        "knowledge_search": lambda: KnowledgeBaseTool(retrieval_engine),
        "web_search": WebSearchTool,
    }

    tools: list[BaseTool] = []
    for name in names:
        factory = factories.get(name)
        if factory is None:
            logger.warning("unknown_agent_tool", tool=name)
            continue
        tools.append(factory())
    return tools
