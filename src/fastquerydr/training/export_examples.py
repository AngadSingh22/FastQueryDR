from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from fastquerydr.config import load_config
from fastquerydr.data import load_id_text_tsv, load_qrels
from fastquerydr.models import build_bi_encoder
from fastquerydr.retrieval.index import build_flat_ip_index
from fastquerydr.retrieval.pipeline import encode_texts, prepare_retrieval_artifacts, rank_documents
from fastquerydr.retrieval.probe import build_retrieval_probe
from fastquerydr.utils.repro import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export qualitative retrieval examples for a model checkpoint")
    parser.add_argument("--config", required=True, help="Path to a YAML config file")
    parser.add_argument("--output", required=True, help="Path to write qualitative examples JSON")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path. Omit for zero-shot pretrained evaluation.")
    parser.add_argument("--num-queries", type=int, default=20, help="Number of queries to export")
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved documents to include per query")
    parser.add_argument("--probe-queries", type=int, default=None, help="Optional retrieval probe query limit for faster qualitative exports")
    parser.add_argument("--probe-corpus-size", type=int, default=None, help="Optional retrieval probe corpus size for faster qualitative exports")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = select_device(config.training.device)

    probe_metadata = None
    if args.probe_queries is not None or args.probe_corpus_size is not None:
        probe = build_retrieval_probe(
            corpus_path=config.retrieval.corpus_path,
            query_path=config.retrieval.query_path,
            qrels_path=config.retrieval.qrels_path,
            query_limit=args.probe_queries or args.num_queries,
            corpus_size=args.probe_corpus_size or 5000,
            top_k=max(args.top_k, 10),
            seed=config.experiment.seed,
        )
        tokenizer = AutoTokenizer.from_pretrained(config.model.encoder_name)
        model = build_bi_encoder(config.model).to(device)
        if args.checkpoint is not None:
            state_dict = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(state_dict)
        model.eval()
        corpus_embeddings = encode_texts(
            model=model,
            tokenizer=tokenizer,
            texts=probe.corpus_texts,
            prefix=config.data.passage_prefix,
            encoder_role="passage",
            max_length=config.data.text_max_length,
            batch_size=min(config.retrieval.batch_size, 64),
            device=device,
        )
        index = build_flat_ip_index(corpus_embeddings)
        corpus_lookup = dict(zip(probe.corpus_ids, probe.corpus_texts))
        qrels = probe.qrels
        query_ids = probe.query_ids[: args.num_queries]
        query_texts = probe.query_texts[: args.num_queries]
        probe_metadata = {
            "query_limit": probe.requested_query_limit,
            "corpus_size": len(probe.corpus_ids),
        }
    else:
        artifacts = prepare_retrieval_artifacts(config=config, checkpoint_path=args.checkpoint, device=device)
        corpus_lookup = {record.record_id: record.text for record in load_id_text_tsv(config.retrieval.corpus_path)}
        query_records = load_id_text_tsv(config.retrieval.query_path)
        qrels = load_qrels(config.retrieval.qrels_path)
        selected_queries = [record for record in query_records if record.record_id in qrels][: args.num_queries]
        query_texts = [record.text for record in selected_queries]
        query_ids = [record.record_id for record in selected_queries]
        model = artifacts.model
        tokenizer = artifacts.tokenizer
        index = artifacts.index

    query_embeddings = encode_texts(
        model=model,
        tokenizer=tokenizer,
        texts=query_texts,
        prefix=config.data.query_prefix,
        encoder_role="query",
        max_length=config.data.text_max_length,
        batch_size=min(64, max(1, len(query_texts))),
        device=device,
    )
    ranked_doc_ids = rank_documents(
        index=index,
        query_embeddings=query_embeddings,
        corpus_ids=probe.corpus_ids if probe_metadata is not None else artifacts.corpus_ids,
        top_k=args.top_k,
    )

    examples = []
    for query_id, query_text, ranked_ids in zip(query_ids, query_texts, ranked_doc_ids):
        examples.append(
            {
                "query_id": query_id,
                "query_text": query_text,
                "relevant_doc_ids": sorted(qrels.get(query_id, [])),
                "top_docs": [
                    {
                        "rank": rank,
                        "doc_id": doc_id,
                        "is_relevant": doc_id in qrels.get(query_id, set()),
                        "text": corpus_lookup.get(doc_id, ""),
                    }
                    for rank, doc_id in enumerate(ranked_ids, start=1)
                ],
            }
        )

    payload = {
        "checkpoint": args.checkpoint,
        "model_name": config.model.encoder_name,
        "pooling": config.model.pooling,
        "num_queries": len(examples),
        "top_k": args.top_k,
        "probe": probe_metadata,
        "examples": examples,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(json.dumps({"output": str(output_path), "queries": len(examples), "top_k": args.top_k}, indent=2))


if __name__ == "__main__":
    main()
