#!/usr/bin/env python3
"""Retrieval quality evaluation script.

Computes MRR, NDCG@K, Recall@K, Precision@K against a labeled test set.

Test set format (JSONL):
    {"query": "什么是RAG？", "relevant_docs": ["doc_id_1", "doc_id_2"]}
    {"query": "如何使用Milvus？", "relevant_docs": ["doc_id_3"]}

Usage:
    python scripts/eval_retrieval.py --test-set ./data/test_queries.jsonl --k 5
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import load_config
from src.embedding.factory import get_embedder
from src.retriever.dense_retriever import DenseRetriever
from src.vectordb.factory import get_vectordb


def load_test_set(path: Path) -> list[dict]:
    queries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def compute_mrr(results: list[list[str]], relevant: list[set[str]]) -> float:
    scores = []
    for ranked, rel in zip(results, relevant):
        for i, doc_id in enumerate(ranked, start=1):
            if doc_id in rel:
                scores.append(1.0 / i)
                break
        else:
            scores.append(0.0)
    return float(np.mean(scores))


def compute_recall_at_k(results: list[list[str]], relevant: list[set[str]], k: int) -> float:
    scores = []
    for ranked, rel in zip(results, relevant):
        if not rel:
            continue
        retrieved = set(ranked[:k])
        scores.append(len(retrieved & rel) / len(rel))
    return float(np.mean(scores)) if scores else 0.0


def compute_precision_at_k(results: list[list[str]], relevant: list[set[str]], k: int) -> float:
    scores = []
    for ranked, rel in zip(results, relevant):
        retrieved = set(ranked[:k])
        scores.append(len(retrieved & rel) / k)
    return float(np.mean(scores)) if scores else 0.0


def compute_ndcg_at_k(results: list[list[str]], relevant: list[set[str]], k: int) -> float:
    scores = []
    for ranked, rel in zip(results, relevant):
        dcg = 0.0
        for i, doc_id in enumerate(ranked[:k], start=1):
            if doc_id in rel:
                dcg += 1.0 / np.log2(i + 1)
        ideal_count = min(len(rel), k)
        idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_count + 1))
        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval quality")
    parser.add_argument("--test-set", required=True, help="JSONL test set file")
    parser.add_argument("--k", type=int, default=5, help="Metrics computed at @K")
    parser.add_argument("--env", default="development", help="Config environment")
    args = parser.parse_args()

    test_set_path = Path(args.test_set)
    if not test_set_path.exists():
        print(f"Test set not found: {args.test_set}")
        sys.exit(1)

    load_config(args.env)

    queries = load_test_set(test_set_path)
    if not queries:
        print("No queries found in test set.")
        return

    embedder = get_embedder()
    embedder.load()
    vectordb = get_vectordb()
    retriever = DenseRetriever(embedder, vectordb, top_k=args.k * 4)

    all_results: list[list[str]] = []
    all_relevant: list[set[str]] = []

    for q in queries:
        hits = retriever.retrieve(q["query"], top_k=args.k)
        ranked_ids = [h.document_id for h in hits]
        all_results.append(ranked_ids)
        all_relevant.append(set(q["relevant_docs"]))

    k = args.k
    mrr = compute_mrr(all_results, all_relevant)
    recall = compute_recall_at_k(all_results, all_relevant, k)
    precision = compute_precision_at_k(all_results, all_relevant, k)
    ndcg = compute_ndcg_at_k(all_results, all_relevant, k)

    print("=" * 50)
    print(f"Retrieval Evaluation Results (N={len(queries)}, K={k})")
    print("=" * 50)
    print(f"MRR        @{k}: {mrr:.4f}")
    print(f"Recall     @{k}: {recall:.4f}")
    print(f"Precision  @{k}: {precision:.4f}")
    print(f"NDCG       @{k}: {ndcg:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
