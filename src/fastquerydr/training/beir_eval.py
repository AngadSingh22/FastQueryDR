from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from transformers import AutoTokenizer

from fastquerydr.config import load_config
from fastquerydr.models import build_bi_encoder
from fastquerydr.retrieval.index import build_flat_ip_index
from fastquerydr.retrieval.metrics import mean_reciprocal_rank_at_k, recall_at_k
from fastquerydr.retrieval.pipeline import _resolve_checkpoint, encode_texts, rank_documents
from fastquerydr.utils.repro import prepare_run_dir, seed_everything, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BEIR transfer evaluation for a FastQueryDR checkpoint")
    parser.add_argument("--config", required=True, help="Path to a YAML config file")
    parser.add_argument("--dataset", default="scifact", help="BEIR dataset name, e.g. scifact or fiqa")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path to load")
    parser.add_argument("--data-dir", default="data/beir", help="Directory for downloaded BEIR datasets")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    seed_everything(config.experiment.seed)
    run_dir = prepare_run_dir(config.experiment.output_dir, f"{config.experiment.name}_beir_{args.dataset}")
    device = select_device(config.training.device)

    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{args.dataset}.zip"
    data_path = util.download_and_unzip(url, args.data_dir)
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    tokenizer = AutoTokenizer.from_pretrained(config.model.encoder_name)
    model = build_bi_encoder(config.model).to(device)
    _resolve_checkpoint(model, args.checkpoint, device)
    model.eval()

    corpus_ids = list(corpus.keys())
    corpus_texts = [f"{item.get('title', '').strip()} {item.get('text', '').strip()}".strip() for item in corpus.values()]
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]
    qrels_sets = {qid: set(doc_scores.keys()) for qid, doc_scores in qrels.items()}

    corpus_start = time.perf_counter()
    corpus_embeddings = encode_texts(
        model=model,
        tokenizer=tokenizer,
        texts=corpus_texts,
        prefix=config.data.passage_prefix,
        encoder_role="passage",
        max_length=config.data.text_max_length,
        batch_size=config.retrieval.batch_size,
        device=device,
    )
    corpus_encode_seconds = time.perf_counter() - corpus_start

    index_start = time.perf_counter()
    index = build_flat_ip_index(corpus_embeddings)
    index_build_seconds = time.perf_counter() - index_start

    query_start = time.perf_counter()
    query_embeddings = encode_texts(
        model=model,
        tokenizer=tokenizer,
        texts=query_texts,
        prefix=config.data.query_prefix,
        encoder_role="query",
        max_length=config.data.text_max_length,
        batch_size=config.retrieval.batch_size,
        device=device,
    )
    query_encode_seconds = time.perf_counter() - query_start

    search_start = time.perf_counter()
    ranked_doc_ids = rank_documents(index, query_embeddings, corpus_ids, config.retrieval.top_k)
    search_seconds = time.perf_counter() - search_start
    metrics = {
        "dataset": args.dataset,
        "checkpoint_path": args.checkpoint,
        "mrr_at_10": mean_reciprocal_rank_at_k(ranked_doc_ids, qrels_sets, query_ids, k=10),
        "recall_at_100": recall_at_k(ranked_doc_ids, qrels_sets, query_ids, k=min(100, config.retrieval.top_k)),
        "corpus_size": len(corpus_ids),
        "query_count": len(query_ids),
        "corpus_encode_seconds": corpus_encode_seconds,
        "index_build_seconds": index_build_seconds,
        "query_encode_seconds": query_encode_seconds,
        "search_seconds": search_seconds,
    }
    with (run_dir / "beir_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
