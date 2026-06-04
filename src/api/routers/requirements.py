"""Requirements analysis endpoint — upload a doc, get a Requirement Graph."""

from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from src.api import dependencies as deps
from src.core.logging import get_logger

router = APIRouter(prefix="/requirements", tags=["requirements"])
logger = get_logger(__name__)

_SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf", ".xlsx", ".xlsm", ".xmind"}


class AnalyzeRequirementsResponse(BaseModel):
    """Response from the requirements analysis endpoint."""

    json_path: str
    markdown_path: str
    module: str
    summary: str


@router.post("/analyze", response_model=AnalyzeRequirementsResponse)
async def analyze_requirements(
    file: UploadFile = File(
        ..., description="需求文档（支持 .txt/.md/.pdf/.xlsx/.xmind）"
    ),
    module: str = Form("", description="功能模块名称（可选，影响文件命名）"),
    output_dir: str = Form(
        "", description="输出目录（当前保留字段，默认由 ArtifactRepository 管理）"
    ),
):
    """Upload a requirements document and receive a structured Requirement Graph."""
    suffix = Path(file.filename or "upload").suffix.lower()
    if suffix not in _SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"不支持的文件类型 '{suffix}'。"
                f"支持的格式：{', '.join(sorted(_SUPPORTED_SUFFIXES))}"
            ),
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="上传文件为空")

    try:
        service = deps.get_requirement_analysis_service()
        result = await service.analyze_upload(
            filename=file.filename or "upload",
            content=content,
            module=module,
        )
        logger.info(
            "requirements_analyzed",
            filename=file.filename,
            module=result.analysis.module,
            json_path=str(result.json_artifact.path),
            requested_output_dir=output_dir or None,
        )
        return AnalyzeRequirementsResponse(
            json_path=str(result.json_artifact.path),
            markdown_path=str(result.markdown_artifact.path),
            module=result.analysis.module,
            summary=result.analysis.summary,
        )
    except Exception as e:
        logger.exception("requirements_analyze_failed", filename=file.filename)
        raise HTTPException(status_code=422, detail=f"无法解析需求文档：{e}") from e
