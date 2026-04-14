from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from fastquerydr.config import AppConfig
from fastquerydr.data import load_id_text_tsv, load_qrels
from fastquerydr.retrieval.latency import benchmark_latency
from fastquerydr.models import SymmetricBiEncoder
from fastquerydr.retrieval.index import build_flat_ip_index
from fastquerydr.retrieval.metrics import mean_reciprocal_rank_at_k, recall_at_k


@dataclass
class RetrievalArtifacts:
    tokenizer: object
    model: SymmetricBiEncoder
    corpus_ids: list[str]
    query_ids: list[str]
    query_texts: list[str]
    qrels: dict[str, set[str]]
    corpus_embeddings: np.ndarray
    index: object
    corpus_encode_seconds: float
    index_build_seconds: float


@torch.inference_mode()
def encode_texts(
    model: SymmetricBiEncoder,
    tokenizer,
    texts: list[str],
    prefix: str,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    embeddings: list[np.ndarray] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="encoding", leave=False):
        batch_texts = [f"{prefix}{text}" for text in texts[start : start + batch_size]]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        batch_embeddings = model.encode(encoded).detach().cpu().numpy().astype("float32")
        embeddings.append(batch_embeddings)
    return np.concatenate(embeddings, axis=0)


def rank_documents(index, query_embeddings: np.ndarray, corpus_ids: list[str], top_k: int) -> list[list[str]]:
    search_k = min(top_k, len(corpus_ids))
    _, doc_indices = index.search(query_embeddings, search_k)
    return [[corpus_ids[index] for index in row] for row in doc_indices.tolist()]


def _resolve_checkpoint(model: SymmetricBiEncoder, checkpoint_path: str | None, device: torch.device) -> None:
    if checkpoint_path is None:
        return
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)


def prepare_retrieval_artifacts(
    config: AppConfig,
    checkpoint_path: str | None,
    device: torch.device,
) -> RetrievalArtifacts:
    if config.retrieval is None or not config.retrieval.enabled:
        raise ValueError("Retrieval config is missing or disabled")

    tokenizer = AutoTokenizer.from_pretrained(config.model.encoder_name)
    model = SymmetricBiEncoder(
        encoder_name=config.model.encoder_name,
        pooling=config.model.pooling,
        normalize=config.model.normalize,
    ).to(device)
    _resolve_checkpoint(model, checkpoint_path, device)
    model.eval()

    corpus_records = load_id_text_tsv(config.retrieval.corpus_path)
    query_records = load_id_text_tsv(config.retrieval.query_path)
    qrels = load_qrels(config.retrieval.qrels_path)

    corpus_ids = [record.record_id for record in corpus_records]
    corpus_texts = [record.text for record in corpus_records]
    query_ids = [record.record_id for record in query_records]
    query_texts = [record.text for record in query_records]

    corpus_start = time.perf_counter()
    corpus_embeddings = encode_texts(
        model=model,
        tokenizer=tokenizer,
        texts=corpus_texts,
        prefix=config.data.passage_prefix,
        max_length=config.data.text_max_length,
        batch_size=config.retrieval.batch_size,
        device=device,
    )
    corpus_encode_seconds = time.perf_counter() - corpus_start

    index_start = time.perf_counter()
    index = build_flat_ip_index(corpus_embeddings)
    index_build_seconds = time.perf_counter() - index_start

    return RetrievalArtifacts(
        tokenizer=tokenizer,
        model=model,
        corpus_ids=corpus_ids,
        query_ids=query_ids,
        query_texts=query_texts,
        qrels=qrels,
        corpus_embeddings=corpus_embeddings,
        index=index,
        corpus_encode_seconds=corpus_encode_seconds,
        index_build_seconds=index_build_seconds,
    )


def run_retrieval_pipeline(
    config: AppConfig,
    checkpoint_path: str | None,
    run_dir: str | Path,
    device: torch.device,
) -> dict:
    if config.retrieval is None or not config.retrieval.enabled:
        raise ValueError("Retrieval config is missing or disabled")

    run_dir = Path(run_dir)
    artifacts = prepare_retrieval_artifacts(
        config=config,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    query_start = time.perf_counter()
    query_embeddings = encode_texts(
        model=artifacts.model,
        tokenizer=artifacts.tokenizer,
        texts=artifacts.query_texts,
        prefix=config.data.query_prefix,
        max_length=config.data.text_max_length,
        batch_size=config.retrieval.batch_size,
        device=device,
    )
    query_encode_seconds = time.perf_counter() - query_start

    search_k = min(config.retrieval.top_k, len(artifacts.corpus_ids))
    search_start = time.perf_counter()
    ranked_doc_ids = rank_documents(artifacts.index, query_embeddings, artifacts.corpus_ids, search_k)
    search_seconds = time.perf_counter() - search_start

    metrics = {
        "mrr_at_10": mean_reciprocal_rank_at_k(ranked_doc_ids, artifacts.qrels, artifacts.query_ids, k=10),
        "recall_at_100": recall_at_k(ranked_doc_ids, artifacts.qrels, artifacts.query_ids, k=min(100, search_k)),
        "corpus_size": len(artifacts.corpus_ids),
        "query_count": len(artifacts.query_ids),
        "corpus_encode_seconds": artifacts.corpus_encode_seconds,
        "index_build_seconds": artifacts.index_build_seconds,
        "query_encode_seconds": query_encode_seconds,
        "search_seconds": search_seconds,
        "mean_query_encode_ms": (query_encode_seconds / max(len(artifacts.query_ids), 1)) * 1000.0,
        "mean_search_ms": (search_seconds / max(len(artifacts.query_ids), 1)) * 1000.0,
        "top_k": search_k,
        "checkpoint_path": checkpoint_path,
    }

    if config.retrieval.latency is not None and config.retrieval.latency.enabled:
        metrics["latency"] = benchmark_latency(
            config=config,
            model=artifacts.model,
            tokenizer=artifacts.tokenizer,
            query_ids=artifacts.query_ids,
            query_texts=artifacts.query_texts,
            index=artifacts.index,
            run_dir=run_dir,
            device=device,
        )

    with (run_dir / "retrieval_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    if config.retrieval.save_embeddings:
        np.save(run_dir / "corpus_embeddings.npy", artifacts.corpus_embeddings)
        np.save(run_dir / "query_embeddings.npy", query_embeddings)
        with (run_dir / "corpus_ids.json").open("w", encoding="utf-8") as handle:
            json.dump(artifacts.corpus_ids, handle)
        with (run_dir / "query_ids.json").open("w", encoding="utf-8") as handle:
            json.dump(artifacts.query_ids, handle)

    return metrics
