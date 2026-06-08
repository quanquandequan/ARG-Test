"""测试用例生成端点：上传需求文档并获取 Excel 文件。"""

from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from src.api import dependencies as deps
from src.core.logging import get_logger

router = APIRouter(prefix="/test-cases", tags=["test-cases"])
logger = get_logger(__name__)

_SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf", ".xlsx", ".xlsm", ".xmind"}


class GenerateTestCasesResponse(BaseModel):
    file_path: str
    automation_json_path: str = ""
    module: str
    generation_mode: str
    summary: str


@router.post("/generate", response_model=GenerateTestCasesResponse)
async def generate_test_cases(
    file: UploadFile = File(..., description="需求文档（支持 .txt/.md/.pdf/.xlsx/.xmind）"),
    module: str = Form("", description="功能模块名称（可选，影响文件命名和用例分组）"),
    generation_mode: str = Form(
        "manual",
        description="生成模式：manual=人工测试用例，automation=移动端UI自动化用例",
    ),
    output_dir: str = Form(
        "",
        description="输出目录（当前保留字段，默认由 ArtifactRepository 管理）",
    ),
):
    """上传需求文档并生成 Excel 测试用例文件。"""
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
        service = deps.get_test_case_generation_service()
        result = await service.generate_from_upload(
            filename=file.filename or "upload",
            content=content,
            module=module,
            generation_mode=generation_mode,
        )
        logger.info(
            "test_cases_generated",
            filename=file.filename,
            module=result.generation.module,
            file_path=str(result.workbook_artifact.path),
            generation_mode=result.generation.generation_mode,
            requested_output_dir=output_dir or None,
        )
        return GenerateTestCasesResponse(
            file_path=str(result.workbook_artifact.path),
            automation_json_path=(
                str(result.automation_json_artifact.path)
                if result.automation_json_artifact
                else ""
            ),
            module=result.generation.module,
            generation_mode=result.generation.generation_mode,
            summary=(
                f"已生成测试用例，共 {result.generation.case_count} 条，"
                f"artifact_id={result.workbook_artifact.artifact_id}"
            ),
        )
    except Exception as e:
        logger.exception("test_cases_generate_failed", filename=file.filename)
        raise HTTPException(status_code=422, detail=f"无法解析需求文档：{e}") from e
