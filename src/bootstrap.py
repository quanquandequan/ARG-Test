"""Explicit application container shared by API and CLI entrypoints."""

from __future__ import annotations

from dataclasses import dataclass

from omegaconf import OmegaConf

from src.agent.react_loop import ReActAgent
from src.agent.tool_factory import build_agent_tools
from src.application.artifact_repository import LocalArtifactRepository
from src.application.ingestion_service import DocumentIngestionService
from src.application.requirement_services import (
    RequirementAnalysisService,
    TestCaseGenerationService,
)
from src.core.config import get_config
from src.embedding.base import BaseEmbedder
from src.embedding.factory import get_embedder
from src.ingestion.chunker import ChineseChunker
from src.ingestion.cleaner import TextCleaner
from src.ingestion.loader import DocumentLoader
from src.ingestion.pipeline import IngestionPipeline
from src.llm.base import BaseLLM
from src.llm.factory import get_llm
from src.retriever.dense_retriever import DenseRetriever
from src.retriever.reranker_base import BaseReranker
from src.retriever.reranker_factory import get_reranker
from src.retriever.retrieval_engine import RetrievalEngine
from src.vectordb.base import BaseVectorDB
from src.vectordb.factory import get_vectordb


@dataclass
class AppContainer:
    _embedder: BaseEmbedder | None = None
    _vectordb: BaseVectorDB | None = None
    _llm: BaseLLM | None = None
    _reranker: BaseReranker | None = None
    _retrieval_engine: RetrievalEngine | None = None
    _artifact_repository: LocalArtifactRepository | None = None
    _loader: DocumentLoader | None = None
    _cleaner: TextCleaner | None = None
    _chunker: ChineseChunker | None = None
    _ingestion_pipeline: IngestionPipeline | None = None
    _ingestion_service: DocumentIngestionService | None = None
    _requirement_analysis_service: RequirementAnalysisService | None = None
    _test_case_generation_service: TestCaseGenerationService | None = None
    _agents: dict[str, ReActAgent] | None = None

    def get_embedder(self) -> BaseEmbedder:
        if self._embedder is None:
            self._embedder = get_embedder()
            self._embedder.load()
        return self._embedder

    def get_vectordb(self) -> BaseVectorDB:
        if self._vectordb is None:
            self._vectordb = get_vectordb()
        return self._vectordb

    def get_llm(self) -> BaseLLM:
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    def get_reranker(self) -> BaseReranker:
        if self._reranker is None:
            self._reranker = get_reranker()
        return self._reranker

    def get_retrieval_engine(self) -> RetrievalEngine:
        if self._retrieval_engine is None:
            cfg = get_config().get("retrieval", {})
            dense = DenseRetriever(
                self.get_embedder(),
                self.get_vectordb(),
                top_k=int(cfg.get("top_k", 20)),
                similarity_threshold=float(cfg.get("similarity_threshold", 0.0)),
            )
            self._retrieval_engine = RetrievalEngine(
                dense_retriever=dense,
                reranker=self.get_reranker(),
            )
        return self._retrieval_engine

    def get_loader(self) -> DocumentLoader:
        if self._loader is None:
            self._loader = DocumentLoader()
        return self._loader

    def get_cleaner(self) -> TextCleaner:
        if self._cleaner is None:
            self._cleaner = TextCleaner()
        return self._cleaner

    def get_chunker(self) -> ChineseChunker:
        if self._chunker is None:
            self._chunker = ChineseChunker()
        return self._chunker

    def get_ingestion_pipeline(self) -> IngestionPipeline:
        if self._ingestion_pipeline is None:
            self._ingestion_pipeline = IngestionPipeline(
                loader=self.get_loader(),
                cleaner=self.get_cleaner(),
                chunker=self.get_chunker(),
                embedder=self.get_embedder(),
                vectordb=self.get_vectordb(),
            )
        return self._ingestion_pipeline

    def get_artifact_repository(self) -> LocalArtifactRepository:
        if self._artifact_repository is None:
            base_dir = str(get_config().get("artifacts", {}).get("base_dir", "./outputs"))
            self._artifact_repository = LocalArtifactRepository(base_dir=base_dir)
        return self._artifact_repository

    def get_ingestion_service(self) -> DocumentIngestionService:
        if self._ingestion_service is None:
            self._ingestion_service = DocumentIngestionService(
                pipeline=self.get_ingestion_pipeline(),
                vectordb=self.get_vectordb(),
                embedder=self.get_embedder(),
            )
        return self._ingestion_service

    def get_requirement_analysis_service(self) -> RequirementAnalysisService:
        if self._requirement_analysis_service is None:
            from src.agent.tools.analyze_requirements import AnalyzeRequirementsTool

            self._requirement_analysis_service = RequirementAnalysisService(
                loader=self.get_loader(),
                cleaner=self.get_cleaner(),
                retrieval_engine=self.get_retrieval_engine(),
                analyzer_tool=AnalyzeRequirementsTool(llm=self.get_llm()),
                artifacts=self.get_artifact_repository(),
            )
        return self._requirement_analysis_service

    def get_test_case_generation_service(self) -> TestCaseGenerationService:
        if self._test_case_generation_service is None:
            from src.agent.tools.write_test_cases import WriteTestCasesTool

            self._test_case_generation_service = TestCaseGenerationService(
                loader=self.get_loader(),
                cleaner=self.get_cleaner(),
                retrieval_engine=self.get_retrieval_engine(),
                writer_tool=WriteTestCasesTool(llm=self.get_llm()),
                artifacts=self.get_artifact_repository(),
            )
        return self._test_case_generation_service

    def get_agent(self, profile_name: str = "qa_agent") -> ReActAgent:
        if self._agents is None:
            self._agents = {}
        agent = self._agents.get(profile_name)
        if agent is not None:
            return agent

        cfg_agent = _resolve_agent_profile(profile_name)
        raw_tools = cfg_agent.get("tools", ["knowledge_search", "web_search"])
        if OmegaConf.is_config(raw_tools):
            tool_configs = OmegaConf.to_container(raw_tools, resolve=True)
        else:
            tool_configs = raw_tools
        tools = build_agent_tools(
            self.get_retrieval_engine(),
            tool_configs,
            llm=self.get_llm(),
        )
        agent = ReActAgent(
            llm=self.get_llm(),
            tools=tools,
            system_prompt=cfg_agent.get("system_prompt", "") or "",
            max_iterations=int(cfg_agent.get("max_iterations", 10)),
            max_history_tokens=int(cfg_agent.get("max_history_tokens", 4000)),
        )
        self._agents[profile_name] = agent
        return agent


def _resolve_agent_profile(profile_name: str) -> dict:
    cfg = get_config().get("agent", {})
    profiles = cfg.get("profiles")
    if profiles and profile_name in profiles:
        profile_cfg = profiles[profile_name]
        if OmegaConf.is_config(profile_cfg):
            return OmegaConf.to_container(profile_cfg, resolve=True)
        return profile_cfg
    if OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, resolve=True)
    return cfg
