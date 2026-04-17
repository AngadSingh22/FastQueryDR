from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from fastquerydr.config import AppConfig
from fastquerydr.retrieval.metrics import mean_reciprocal_rank_at_k, recall_at_k
from fastquerydr.retrieval.pipeline import encode_texts, prepare_retrieval_artifacts, rank_documents

try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None


def build_hnsw_ip_index(embeddings: np.ndarray, m: int = 32, ef_construction: int = 80):
    if faiss is None:
        raise ImportError("faiss is required for ANN evaluation.")
    index = faiss.IndexHNSWFlat(embeddings.shape[1], m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.add(embeddings.astype("float32"))
    return index


def build_ivf_ip_index(embeddings: np.ndarray, nlist: int = 256, nprobe: int = 16):
    if faiss is None:
        raise ImportError("faiss is required for ANN evaluation.")
    quantizer = faiss.IndexFlatIP(embeddings.shape[1])
    index = faiss.IndexIVFFlat(quantizer, embeddings.shape[1], nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(embeddings.astype("float32"))
    index.add(embeddings.astype("float32"))
    index.nprobe = nprobe
    return index


def run_ann_comparison(
    *,
    config: AppConfig,
    checkpoint_path: str | None,
    run_dir: str | Path,
    device,
    query_limit: int | None = None,
) -> dict:
    run_dir = Path(run_dir)
    artifacts = prepare_retrieval_artifacts(config=config, checkpoint_path=checkpoint_path, device=device)

    query_ids = artifacts.query_ids[:query_limit] if query_limit is not None else artifacts.query_ids
    query_texts = artifacts.query_texts[:query_limit] if query_limit is not None else artifacts.query_texts

    query_embeddings = encode_texts(
        model=artifacts.model,
        tokenizer=artifacts.tokenizer,
        texts=query_texts,
        prefix=config.data.query_prefix,
        encoder_role="query",
        max_length=config.data.text_max_length,
        batch_size=config.retrieval.batch_size,
        device=device,
    )

    index_specs = {
        "flat": lambda emb: artifacts.index,
        "hnsw": lambda emb: build_hnsw_ip_index(emb),
        "ivf": lambda emb: build_ivf_ip_index(emb),
    }
    results: dict[str, dict] = {}
    for name, builder in index_specs.items():
        build_start = time.perf_counter()
        index = builder(artifacts.corpus_embeddings)
        build_seconds = time.perf_counter() - build_start if name != "flat" else artifacts.index_build_seconds

        search_start = time.perf_counter()
        ranked_doc_ids = rank_documents(index, query_embeddings, artifacts.corpus_ids, config.retrieval.top_k)
        search_seconds = time.perf_counter() - search_start
        results[name] = {
            "mrr_at_10": mean_reciprocal_rank_at_k(ranked_doc_ids, artifacts.qrels, query_ids, k=10),
            "recall_at_100": recall_at_k(ranked_doc_ids, artifacts.qrels, query_ids, k=min(100, config.retrieval.top_k)),
            "index_build_seconds": build_seconds,
            "search_seconds": search_seconds,
            "mean_search_ms": (search_seconds / max(len(query_ids), 1)) * 1000.0,
            "query_count": len(query_ids),
        }

    output = {
        "checkpoint_path": checkpoint_path,
        "query_limit": query_limit,
        "indexes": results,
    }
    with (run_dir / "ann_comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    return output
