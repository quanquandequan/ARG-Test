#!/usr/bin/env python3
"""批量文档摄取 CLI。

用法：
    python scripts/ingest_docs.py --dir ./data/documents --ext pdf,md,txt
    python scripts/ingest_docs.py --dir ./data/documents --collection my_kb
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

# 将项目根目录加入路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import load_config
from src.core.logging import setup_logging
from src.embedding.factory import get_embedder
from src.ingestion.pipeline import IngestionPipeline
from src.vectordb.factory import get_vectordb


def main():
    parser = argparse.ArgumentParser(description="Batch document ingestion for RAG pipeline")
    parser.add_argument("--dir", required=True, help="Directory containing documents")
    parser.add_argument("--ext", default="pdf,md,txt,xmind,xlsx", help="Comma-separated extensions")
    parser.add_argument("--collection", default=None, help="Milvus collection name")
    parser.add_argument("--env", default="development", help="Config environment")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    args = parser.parse_args()

    load_config(args.env)
    setup_logging()

    from src.core.logging import get_logger
    logger = get_logger(__name__)

    extensions = set(f".{e.strip()}" for e in args.ext.split(","))
    doc_dir = Path(args.dir)
    if not doc_dir.is_dir():
        logger.error("directory_not_found", path=str(doc_dir))
        sys.exit(1)

    files = [p for p in doc_dir.rglob("*") if p.suffix.lower() in extensions and p.is_file()]
    if not files:
        logger.warning("no_files_found", dir=str(doc_dir), extensions=list(extensions))
        return

    logger.info("ingestion_start", file_count=len(files), extensions=list(extensions))

    embedder = get_embedder()
    embedder.load()
    vectordb = get_vectordb()

    if args.collection:
        vectordb.create_collection(args.collection, embedder.dim(), drop_existing=False)

    pipeline = IngestionPipeline()

    total_chunks = 0

    def process_file(path: Path):
        doc, chunks = pipeline.ingest(path)
        texts = [c.content for c in chunks]
        vectors = embedder.embed_documents(texts)
        rows = [
            (c.id, c.document_id, c.content, c.chunk_index, vec,
             {"source_path": str(path)})  # 元数据
            for c, vec in zip(chunks, vectors)
        ]
        vectordb.insert(rows)
        return len(chunks)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_file, fp): fp for fp in files}
        with tqdm(total=len(files), desc="Ingesting") as pbar:
            for future in as_completed(futures):
                path = futures[future]
                try:
                    count = future.result()
                    total_chunks += count
                    pbar.set_postfix(chunks=total_chunks)
                except Exception as e:
                    logger.error("file_failed", path=str(path), error=str(e))
                pbar.update(1)

    logger.info("ingestion_complete", files=len(files), total_chunks=total_chunks)


if __name__ == "__main__":
    main()
