#!/usr/bin/env python3
"""查询 Milvus 中已入库的所有来源文件。"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    """打印当前 Milvus collection 中已入库的来源文件及 chunk 数。"""
    warnings.filterwarnings("ignore")

    from dotenv import load_dotenv
    from pymilvus import MilvusClient

    from src.core.config import get_config, load_config

    load_dotenv(PROJECT_ROOT / ".env")
    load_config("development")

    cfg = get_config()
    vdb_cfg = cfg.get("vectordb", {})
    collection = vdb_cfg.get("collection_name", "knowledge_base")
    uri = vdb_cfg.get("uri", "http://localhost:19530")
    print(f"collection={collection}  uri={uri}")

    client = MilvusClient(uri=uri)
    client.load_collection(collection_name=collection)

    # 分批拉取所有 metadata 字段，提取 source_path
    sources = {}  # source_path -> count
    offset = 0
    batch = 1000
    while True:
        rows = client.query(
            collection_name=collection,
            filter="",
            output_fields=["metadata"],
            limit=batch,
            offset=offset,
        )
        if not rows:
            break
        for r in rows:
            meta = r.get("metadata") or {}
            sp = meta.get("source_path", meta.get("source_name", "unknown"))
            sources[sp] = sources.get(sp, 0) + 1
        offset += len(rows)
        if len(rows) < batch:
            break

    print(f"\n已入库文件（共 {len(sources)} 个，总 {offset} 条记录）：")
    for sp, cnt in sorted(sources.items(), key=lambda x: x[0]):
        fname = os.path.basename(sp)
        print(f"  [{cnt:5d} chunks] {fname}")


if __name__ == "__main__":
    main()
