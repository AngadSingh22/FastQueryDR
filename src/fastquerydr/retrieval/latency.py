from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import psutil
import torch

from fastquerydr.config import AppConfig
from fastquerydr.models import SymmetricBiEncoder
from fastquerydr.utils.repro import synchronize_device


def _percentile_ms(samples: list[float], percentile: float) -> float:
    if not samples:
        return 0.0
    return float(np.percentile(np.array(samples, dtype=np.float64), percentile))


@torch.inference_mode()
def _encode_single_query(
    model: SymmetricBiEncoder,
    tokenizer,
    query_text: str,
    prefix: str,
    max_length: int,
    device: torch.device,
) -> np.ndarray:
    encoded = tokenizer(
        [f"{prefix}{query_text}"],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    embedding = model.encode(encoded).detach().cpu().numpy().astype("float32")
    return embedding


def benchmark_latency(
    config: AppConfig,
    model: SymmetricBiEncoder,
    tokenizer,
    query_ids: list[str],
    query_texts: list[str],
    index,
    run_dir: str | Path,
    device: torch.device,
) -> dict:
    if config.retrieval is None or config.retrieval.latency is None or not config.retrieval.latency.enabled:
        raise ValueError("Latency config is missing or disabled")

    latency_config = config.retrieval.latency
    if latency_config.query_batch_size != 1:
        raise ValueError("Phase 3 latency benchmarking currently supports only query_batch_size=1")

    run_dir = Path(run_dir)

    total_requested = latency_config.warmup_queries + latency_config.measured_queries
    total_available = min(total_requested, len(query_texts))
    warmup_count = min(latency_config.warmup_queries, total_available)
    measured_count = max(total_available - warmup_count, 0)

    selected_query_ids = query_ids[:total_available]
    selected_query_texts = query_texts[:total_available]
    measured_query_ids = selected_query_ids[warmup_count:]
    measured_query_texts = selected_query_texts[warmup_count:]

    process = psutil.Process()
    search_k = min(latency_config.search_top_k, config.retrieval.top_k, int(index.ntotal))
    query_encode_ms: list[float] = []
    end_to_end_ms: list[float] = []
    cpu_memory_deltas: list[int] = []
    gpu_memory_peaks: list[int] = []

    for query_text in selected_query_texts[:warmup_count]:
        _ = _encode_single_query(
            model=model,
            tokenizer=tokenizer,
            query_text=query_text,
            prefix=config.data.query_prefix,
            max_length=config.data.text_max_length,
            device=device,
        )
        synchronize_device(device)

    for query_text in measured_query_texts:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        rss_before = process.memory_info().rss
        start_total = time.perf_counter()
        start_encode = time.perf_counter()
        query_embedding = _encode_single_query(
            model=model,
            tokenizer=tokenizer,
            query_text=query_text,
            prefix=config.data.query_prefix,
            max_length=config.data.text_max_length,
            device=device,
        )
        synchronize_device(device)
        encode_elapsed_ms = (time.perf_counter() - start_encode) * 1000.0
        _ = index.search(query_embedding, search_k)
        end_to_end_elapsed_ms = (time.perf_counter() - start_total) * 1000.0
        rss_after = process.memory_info().rss

        query_encode_ms.append(encode_elapsed_ms)
        end_to_end_ms.append(end_to_end_elapsed_ms)
        cpu_memory_deltas.append(max(rss_after - rss_before, 0))

        if device.type == "cuda":
            gpu_memory_peaks.append(int(torch.cuda.max_memory_allocated(device)))

    metrics = {
        "device": str(device),
        "warmup_queries": warmup_count,
        "measured_queries": measured_count,
        "query_batch_size": latency_config.query_batch_size,
        "search_top_k": search_k,
        "query_ids": measured_query_ids,
        "query_encode_latency_ms_p50": _percentile_ms(query_encode_ms, 50),
        "query_encode_latency_ms_p95": _percentile_ms(query_encode_ms, 95),
        "query_encode_latency_ms_mean": float(np.mean(query_encode_ms)) if query_encode_ms else 0.0,
        "end_to_end_latency_ms_p50": _percentile_ms(end_to_end_ms, 50),
        "end_to_end_latency_ms_p95": _percentile_ms(end_to_end_ms, 95),
        "end_to_end_latency_ms_mean": float(np.mean(end_to_end_ms)) if end_to_end_ms else 0.0,
        "query_memory_peak_bytes": max(gpu_memory_peaks) if gpu_memory_peaks else 0,
        "query_memory_rss_delta_bytes": max(cpu_memory_deltas) if cpu_memory_deltas else 0,
        "memory_measurement": "gpu_peak_allocated" if device.type == "cuda" else "process_rss_delta",
    }

    with (run_dir / "latency_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    return metrics
