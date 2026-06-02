"""Requirements analysis endpoint — upload a doc, get a Requirement Graph."""

import re
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from src.api import dependencies as deps
from src.core.logging import get_logger
from src.ingestion.cleaner import TextCleaner
from src.ingestion.loader import DocumentLoader
from src.ingestion.pipeline import IngestionPipeline

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
        "", description="输出目录（可选，默认 ./outputs/requirements）"
    ),
):
    """Upload a requirements document and receive a structured Requirement Graph.

    **Workflow**
    1. Parse the uploaded file to plain text.
    2. Run the Agent, which will:
       a. Call ``knowledge_search`` to retrieve existing feature background.
          (叭嗒 app functions → test cases; plugin/mini-program → xmind docs)
       b. Call ``analyze_requirements`` with the text + KB context.
    3. Extract the JSON and Markdown file paths from the Agent's response.
    4. Return the paths plus a brief summary.

    **Output files** (saved to ``output_dir``):
    - ``<module>_<timestamp>_req_graph.json`` — RequirementGraph (machine-readable)
    - ``<module>_<timestamp>_analysis.md``    — Analysis report (human-readable)
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

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        # Parse requirement document to plain text
        pipeline = IngestionPipeline(loader=DocumentLoader(), cleaner=TextCleaner())
        try:
            doc, _ = pipeline.ingest(tmp_path)
            requirement_text = doc.content.strip()
        except Exception as e:
            logger.exception("requirements_parse_failed", filename=file.filename)
            raise HTTPException(
                status_code=422,
                detail=f"无法解析需求文档：{e}",
            ) from e

        if not requirement_text:
            raise HTTPException(status_code=400, detail="需求文档内容为空")

        # Build Agent query — Agent orchestrates KB search → analyze_requirements
        module_hint = f"模块：{module.strip()}\n" if module.strip() else ""
        output_hint = f"输出目录：{output_dir.strip()}\n" if output_dir.strip() else ""
        agent_query = (
            "请对以下需求文档进行测试视角分析，生成 Requirement Graph 报告。\n"
            f"{module_hint}{output_hint}\n"
            f"需求文档：\n{requirement_text}"
        )

        agent = deps.get_agent()
        result = await agent.run(query=agent_query)

        # Extract file paths from Agent's answer
        json_path = _extract_path(result.answer, ".json")
        md_path = _extract_path(result.answer, ".md")

        if not json_path or not md_path:
            logger.warning(
                "requirements_no_paths",
                answer=result.answer[:300],
                filename=file.filename,
            )
            raise HTTPException(
                status_code=500,
                detail=(
                    "未能从 Agent 回答中提取文件路径。"
                    f"Agent 回答：{result.answer}"
                ),
            )

        logger.info(
            "requirements_analyzed",
            filename=file.filename,
            module=module or "通用",
            json_path=json_path,
        )
        return AnalyzeRequirementsResponse(
            json_path=json_path,
            markdown_path=md_path,
            module=module or "通用",
            summary=result.answer,
        )

    finally:
        if tmp_path:
            tmp_path.unlink(missing_ok=True)


def _extract_path(text: str, extension: str) -> str:
    """Extract the first absolute or relative path with the given extension."""
    pattern = rf"([/\\.][^\s\n]+\{re.escape(extension)})"
    match = re.search(pattern, text)
    return match.group(1).strip() if match else ""
