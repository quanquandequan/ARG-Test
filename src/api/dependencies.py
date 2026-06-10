"""FastAPI dependency injection — lru_cache singletons."""

import sys
from functools import lru_cache

from src.agent.react_loop import ReActAgent
from src.agent.tool_factory import build_agent_tools
from src.core.config import get_config
from src.core.prompt_loader import require_prompt_fields
from src.embedding.factory import get_embedder
from src.ingestion.chunker import ChineseChunker
from src.ingestion.cleaner import TextCleaner
from src.ingestion.loader import DocumentLoader
from src.ingestion.pipeline import IngestionPipeline
from src.llm.factory import get_llm
from src.mobile.driver import AppiumDriverManager
from src.retriever.dense_retriever import DenseRetriever
from src.retriever.reranker_factory import get_reranker
from src.retriever.retrieval_engine import RetrievalEngine
from src.services.artifact_repository import LocalArtifactRepository
from src.services.ingestion_service import DocumentIngestionService
from src.services.page_cache import PageCache
from src.services.requirement_analysis import RequirementAnalysisService
from src.vectordb.factory import get_vectordb
from src.workflows.execution import ExecutionWorkflow
from src.workflows.testcase_design import TestCaseGenerationWorkflow


@lru_cache(maxsize=1)
def _llm(): return get_llm()

@lru_cache(maxsize=1)
def _embedder(): e = get_embedder(); e.load(); return e

@lru_cache(maxsize=1)
def _vectordb(): return get_vectordb()

@lru_cache(maxsize=1)
def _reranker(): return get_reranker()

@lru_cache(maxsize=1)
def _retrieval_engine(): return RetrievalEngine(DenseRetriever(_embedder(), _vectordb()), _reranker())

@lru_cache(maxsize=1)
def _artifacts():
    cfg = get_config().get("artifacts", {})
    return LocalArtifactRepository(base_dir=cfg.get("base_dir", "./outputs"))

@lru_cache(maxsize=1)
def _loader(): return DocumentLoader()

@lru_cache(maxsize=1)
def _cleaner(): return TextCleaner()

@lru_cache(maxsize=1)
def _driver_manager(): return AppiumDriverManager()

@lru_cache(maxsize=1)
def _page_cache(): return PageCache()

@lru_cache(maxsize=1)
def get_ingestion_service():
    return DocumentIngestionService(IngestionPipeline(_loader(), _cleaner(), ChineseChunker()), _vectordb(), _embedder())

@lru_cache(maxsize=1)
def get_requirement_analysis_service():
    from src.agent.tools.requirement_graph_analyzer import RequirementGraphAnalyzerTool
    return RequirementAnalysisService(_loader(), _cleaner(), _retrieval_engine(), RequirementGraphAnalyzerTool(llm=_llm()), _artifacts())

@lru_cache(maxsize=1)
def get_test_case_generation_service():
    wf = TestCaseGenerationWorkflow.create_default(_loader(), _cleaner(), _retrieval_engine(), _artifacts(), _llm())
    return wf

@lru_cache(maxsize=1)
def get_mobile_workflow():
    return ExecutionWorkflow(driver_manager=_driver_manager(), page_cache=_page_cache(), artifacts=_artifacts())

@lru_cache(maxsize=2)
def get_agent(profile_name=None):
    from omegaconf import OmegaConf
    cfg = get_config().get("agent", {})
    profiles = cfg.get("profiles", {})

    if not profile_name and "qa_agent" in profiles:
        cfg_agent = profiles["qa_agent"]
        if OmegaConf.is_config(cfg_agent):
            cfg_agent = OmegaConf.to_container(cfg_agent, resolve=True)
    elif profile_name and profile_name in profiles:
        cfg_agent = profiles[profile_name]
        if OmegaConf.is_config(cfg_agent):
            cfg_agent = OmegaConf.to_container(cfg_agent, resolve=True)
    elif profile_name:
        raise ValueError(f"Unknown agent profile: {profile_name}")
    else:
        cfg_agent = cfg

    raw_tools = cfg_agent.get("tools", [])
    if OmegaConf.is_config(raw_tools):
        raw_tools = OmegaConf.to_container(raw_tools, resolve=True)

    tools = build_agent_tools(_retrieval_engine(), raw_tools, llm=_llm(),
        test_case_generation_service=get_test_case_generation_service(),
        mobile_execution_service=get_mobile_workflow(),
        driver_manager=_driver_manager(), page_cache=_page_cache(),
        loader=_loader(), cleaner=_cleaner())

    system_prompt_id = cfg_agent.get("system_prompt_id") or cfg_agent.get("prompt_id")
    system_prompt = (
        require_prompt_fields(str(system_prompt_id), ["system_prompt"])["system_prompt"]
        if system_prompt_id
        else cfg_agent.get("system_prompt", "") or ""
    )

    return ReActAgent(llm=_llm(), tools=tools,
        system_prompt=system_prompt,
        max_iterations=int(cfg_agent.get("max_iterations", 10)),
        max_history_tokens=int(cfg_agent.get("max_history_tokens", 4000)))

def clear_all_caches():
    _this = sys.modules[__name__]
    for name in ("_llm", "_embedder", "_vectordb", "_reranker", "_retrieval_engine",
                 "_artifacts", "_loader", "_cleaner", "_driver_manager", "_page_cache",
                 "get_ingestion_service", "get_requirement_analysis_service",
                 "get_test_case_generation_service", "get_mobile_workflow", "get_agent"):
        fn = getattr(_this, name, None)
        if fn and hasattr(fn, "cache_clear"):
            fn.cache_clear()
