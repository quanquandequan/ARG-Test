"""Requirement analysis service."""

from __future__ import annotations



class RequirementAnalysisService:
    def __init__(
        self,
        loader: DocumentLoader,
        cleaner: TextCleaner,
        retrieval_engine: RetrievalEngine,
        analyzer_tool: RequirementGraphAnalyzerTool,
        artifacts: LocalArtifactRepository,
    ):
        self._loader = loader
        self._cleaner = cleaner
        self._retrieval_engine = retrieval_engine
        self._analyzer_tool = analyzer_tool
        self._artifacts = artifacts

    async def analyze_upload(
        self,
        filename: str,
        content: bytes,
        module: str = "",
    ) -> RequirementAnalysisResult:
        requirement_text = self._load_requirement_text(filename, content)
        resolved_module = module.strip() or "需求分析"
        kb_context = await build_requirement_kb_context(
            self._retrieval_engine,
            resolved_module,
            requirement_text,
        )

        tool_result = await self._analyzer_tool.execute_typed(
            requirement=requirement_text,
            kb_context=kb_context,
            module=resolved_module,
        )
        analysis = tool_result.data
        if analysis is None:
            raise ValueError(tool_result.content or "需求分析未产出有效结构化结果。")

        json_artifact = next(
            (
                artifact
                for artifact in tool_result.artifacts
                if artifact.kind == ArtifactKind.REQUIREMENT_ANALYSIS_JSON
            ),
            None,
        )
        if json_artifact is None:
            metadata = {
                "module": analysis.module,
                "summary": analysis.summary,
                "feature_count": analysis.feature_count,
                "risk_count": analysis.risk_count,
                "clarification_count": analysis.clarification_count,
            }
            json_artifact = self._artifacts.save_json(
                ArtifactKind.REQUIREMENT_ANALYSIS_JSON,
                analysis.module,
                analysis.graph,
                metadata=metadata,
                suffix="req_graph",
            )
        return RequirementAnalysisResult(
            analysis=analysis,
            json_artifact=json_artifact,
        )

    def _load_requirement_text(self, filename: str, content: bytes) -> str:
        suffix = Path(filename or "upload").suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            doc = self._loader.load(tmp_path)
            return self._cleaner.clean(doc.content).strip()
        finally:
            tmp_path.unlink(missing_ok=True)
