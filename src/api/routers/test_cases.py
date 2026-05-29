"""Test case generation endpoint — upload a requirements doc, get an Excel file."""

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from src.api import dependencies as deps
from src.core.logging import get_logger
from src.ingestion.cleaner import TextCleaner
from src.ingestion.loader import DocumentLoader
from src.ingestion.pipeline import IngestionPipeline

router = APIRouter(prefix="/test-cases", tags=["test-cases"])
logger = get_logger(__name__)

_SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf", ".xlsx", ".xlsm", ".xmind"}


class GenerateTestCasesResponse(BaseModel):
    file_path: str
    module: str
    summary: str


@router.post("/generate", response_model=GenerateTestCasesResponse)
async def generate_test_cases(
    file: UploadFile = File(..., description="需求文档（支持 .txt/.md/.pdf/.xlsx/.xmind）"),
    module: str = Form("", description="功能模块名称（可选，影响文件命名和用例分组）"),
    output_dir: str = Form("", description="Excel 输出目录（可选，默认 ./outputs/test_cases）"),
):
    """Upload a requirements document and generate test cases as an Excel file.

    **Workflow**:
    1. Parse the uploaded file to plain text.
    2. Run the Agent with a test-case-generation intent.
       The Agent will automatically call ``knowledge_search`` to retrieve
       existing test case samples for format reference, then call
       ``write_test_cases`` to generate and save the Excel.
    3. Return the saved file path and a brief summary.
    """
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

    # Save to a temp file so the existing readers can process it
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        # Parse the requirement document to plain text
        pipeline = IngestionPipeline(loader=DocumentLoader(), cleaner=TextCleaner())
        try:
            doc, _ = pipeline.ingest(tmp_path)
            requirement_text = doc.content.strip()
        except Exception as e:
            logger.exception("test_cases_parse_failed", filename=file.filename)
            raise HTTPException(
                status_code=422,
                detail=f"无法解析需求文档：{e}",
            ) from e

        if not requirement_text:
            raise HTTPException(status_code=400, detail="需求文档内容为空")

        # Build the Agent query — the Agent will orchestrate KB search + Excel generation
        module_hint = f"模块：{module.strip()}\n" if module.strip() else ""
        output_hint = f"输出目录：{output_dir.strip()}\n" if output_dir.strip() else ""
        agent_query = (
            "请根据以下需求文档生成完整的测试用例，并输出为 Excel 文件。\n"
            f"{module_hint}{output_hint}\n"
            f"需求文档：\n{requirement_text}"
        )

        agent = deps.get_agent()
        result = await agent.run(query=agent_query)

        # Extract the file path from the agent's answer
        file_path = _extract_file_path(result.answer)
        if not file_path:
            logger.warning(
                "test_cases_no_path",
                answer=result.answer[:200],
                filename=file.filename,
            )
            raise HTTPException(
                status_code=500,
                detail=f"未能从 Agent 回答中提取文件路径。Agent 回答：{result.answer}",
            )

        logger.info(
            "test_cases_generated",
            filename=file.filename,
            module=module or "通用",
            file_path=file_path,
        )
        return GenerateTestCasesResponse(
            file_path=file_path,
            module=module or "通用",
            summary=result.answer,
        )

    finally:
        if tmp_path:
            tmp_path.unlink(missing_ok=True)


def _extract_file_path(text: str) -> str:
    """Extract the absolute or relative Excel file path from the Agent's answer."""
    import re

    # Match an absolute or relative path ending with .xlsx
    match = re.search(r"([/\\.].+?\.xlsx)", text)
    if match:
        return match.group(1).strip()
    return ""
