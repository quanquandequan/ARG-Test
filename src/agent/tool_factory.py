"""根据配置驱动的工具名称构建 Agent 工具。

工具配置支持两种格式（通常来自某个 profile 的 ``tools`` 列表）：

  # 简单字符串：使用代码默认的描述与提示词
  - search_knowledge

  # 对象：覆盖描述和/或工具内部 system_prompt
  - name: design_test_cases
    description: |
      根据需求文档生成测试用例...（覆盖代码默认值）
    system_prompt: |
      你是一名资深测试工程师...（工具调用 LLM 时的提示词）
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.agent.base_tool import BaseTool
from src.agent.tools.analyze_requirement import AnalyzeRequirementTool
from src.agent.tools.design_test_cases import DesignTestCasesTool
from src.agent.tools.execute_scenario import ExecuteScenarioTool
from src.agent.tools.mobile.action_tool import ActionTool
from src.agent.tools.mobile.assertion_tool import AssertionTool
from src.agent.tools.mobile.device_tool import DeviceTool
from src.agent.tools.mobile.screen_tool import ScreenTool
from src.agent.tools.search_knowledge import SearchKnowledgeTool
from src.core.logging import get_logger
from src.ingestion.cleaner import TextCleaner
from src.ingestion.loader import DocumentLoader
from src.llm.base import BaseLLM
from src.mobile.driver import AppiumDriverManager
from src.retriever.retrieval_engine import RetrievalEngine
from src.services.page_cache import PageCache
from src.workflows.execution import ExecutionWorkflow
from src.workflows.testcase_design import TestCaseGenerationWorkflow

logger = get_logger(__name__)

_DEFAULT_TOOLS = (
    "search_knowledge",
    "analyze_requirement",
    "design_test_cases",
    "execute_scenario",
)

# 共享移动端单例：首次构建移动端工具时延迟创建。
# 四个移动端工具引用同一个 driver manager 和 page cache，
# 确保 DeviceTool.connect() 对 ScreenTool / ActionTool / AssertionTool 可见。
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
    """从配置条目返回 ``(name, description_override, system_prompt)``。

    支持纯字符串，或至少包含 ``name`` 键的 dict/DictConfig。
    """
    if isinstance(entry, str):
        return entry, None, None

    # dict 或 OmegaConf DictConfig
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
    test_case_generation_service: TestCaseGenerationWorkflow | None = None,
    mobile_execution_service: ExecutionWorkflow | None = None,
    driver_manager: AppiumDriverManager | None = None,
    page_cache: PageCache | None = None,
    loader: DocumentLoader | None = None,
    cleaner: TextCleaner | None = None,
) -> list[BaseTool]:
    """按当前 profile 的 ``tools`` 配置实例化工具（保留顺序）。

    Args:
        retrieval_engine: 供知识检索类工具使用。
        tool_configs: 来自配置的有序列表。每个条目可以是纯字符串（工具名），
            也可以是包含 ``name``、可选 ``description`` 和可选
            ``system_prompt`` 的 dict。默认使用 ``_DEFAULT_TOOLS``。
        llm: 仅 LLM 驱动工具需要（例如 ``analyze_requirement``）。
             如果为 *None* 且请求了这类工具，则记录警告并跳过。
        test_case_generation_service: 测试用例生成能力使用的业务门面。
        mobile_execution_service: 自动化执行能力使用的业务门面。
        driver_manager: 可选移动端运行时实例，供低阶工具与执行工作流共享。
        page_cache: 可选页面缓存实例，供低阶工具与执行工作流共享。
        loader: 可选文档加载器，供需求分析门面读取本地需求文件。
        cleaner: 可选文本清洗器，供需求分析门面清洗需求文件内容。
    """
    configs = tool_configs or list(_DEFAULT_TOOLS)

    def _require_mobile_runtime(tool_name: str) -> tuple[AppiumDriverManager, PageCache]:
        if driver_manager is None or page_cache is None:
            raise RuntimeError(f"{tool_name} requires driver_manager and page_cache; these are injected by dependencies.")
        return driver_manager, page_cache

    def _build_qwen_vlm():
        """若已配置则返回 QwenVisionProvider，否则返回 None。"""
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

    # 工厂函数接收 (description_override, system_prompt) 作为上下文
    def _make_search_knowledge(_desc: str | None, _sp: str | None) -> BaseTool:
        return SearchKnowledgeTool(retrieval_engine)

    def _make_design_test_cases(
        _desc: str | None, sys_prompt: str | None
    ) -> BaseTool | None:
        if test_case_generation_service is None:
            logger.warning("service_required_for_tool_skipped", tool="design_test_cases")
            return None
        return DesignTestCasesTool(
            test_case_generation_service,
            system_prompt=sys_prompt or None,
        )

    def _make_analyze_requirement(
        _desc: str | None, sys_prompt: str | None
    ) -> BaseTool | None:
        if sys_prompt:
            raise ValueError(
                "analyze_requirement 是复合门面工具，不支持 system_prompt 覆盖；"
                "请配置 requirement_parser / requirement_reviewer / "
                "analyze_requirements 等内部工具的提示词。"
            )
        _llm = _require_llm("analyze_requirement")
        if _llm is None:
            return None
        return AnalyzeRequirementTool(
            llm=_llm,
            retrieval_engine=retrieval_engine,
            loader=loader,
            cleaner=cleaner,
        )

    def _make_device_tool(_desc: str | None, _sp: str | None) -> BaseTool:
        dm, _ = _require_mobile_runtime("device_tool")
        return DeviceTool(driver_manager=dm)

    def _make_screen_tool(_desc: str | None, _sp: str | None) -> BaseTool:
        dm, pc = _require_mobile_runtime("screen_tool")
        vlm = _build_qwen_vlm()
        return ScreenTool(driver_manager=dm, page_cache=pc, vlm=vlm)

    def _make_action_tool(_desc: str | None, _sp: str | None) -> BaseTool:
        dm, pc = _require_mobile_runtime("action_tool")
        return ActionTool(driver_manager=dm, page_cache=pc)

    def _make_assertion_tool(_desc: str | None, _sp: str | None) -> BaseTool:
        dm, _ = _require_mobile_runtime("assertion_tool")
        return AssertionTool(driver_manager=dm)

    def _make_execute_scenario(_desc: str | None, _sp: str | None) -> BaseTool | None:
        if mobile_execution_service is None:
            logger.warning("service_required_for_tool_skipped", tool="execute_scenario")
            return None
        return ExecuteScenarioTool(workflow=mobile_execution_service)

    factories: dict[str, Callable[[str | None, str | None], BaseTool | None]] = {        "search_knowledge": _make_search_knowledge,        "design_test_cases": _make_design_test_cases,        "analyze_requirement": _make_analyze_requirement,        "device_tool": _make_device_tool,
        "screen_tool": _make_screen_tool,
        "action_tool": _make_action_tool,
        "assertion_tool": _make_assertion_tool,
        "execute_scenario": _make_execute_scenario,
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

        # 应用 YAML 级描述覆盖（影响 LLM 看到的内容）
        if description:
            tool.override_description(description)

        tools.append(tool)

    return tools
