from __future__ import annotations

import random
from dataclasses import dataclass

from fastquerydr.data import load_id_text_tsv, load_qrels


@dataclass
class RetrievalProbe:
    corpus_ids: list[str]
    corpus_texts: list[str]
    query_ids: list[str]
    query_texts: list[str]
    qrels: dict[str, set[str]]
    top_k: int
    requested_query_limit: int
    requested_corpus_size: int


def build_retrieval_probe(
    *,
    corpus_path: str,
    query_path: str,
    qrels_path: str,
    query_limit: int,
    corpus_size: int,
    top_k: int,
    seed: int,
) -> RetrievalProbe:
    qrels = load_qrels(qrels_path)
    selected_queries = [record for record in load_id_text_tsv(query_path) if record.record_id in qrels][:query_limit]
    selected_qrels = {record.record_id: qrels[record.record_id] for record in selected_queries}
    relevant_doc_ids = {doc_id for doc_ids in selected_qrels.values() for doc_id in doc_ids}

    relevant_records = []
    distractor_records = []
    for record in load_id_text_tsv(corpus_path):
        if record.record_id in relevant_doc_ids:
            relevant_records.append(record)
        else:
            distractor_records.append(record)

    extra_needed = max(corpus_size - len(relevant_records), 0)
    if extra_needed >= len(distractor_records):
        sampled_distractors = distractor_records
    else:
        sampled_distractors = random.Random(seed).sample(distractor_records, extra_needed)

    selected_corpus = relevant_records + sampled_distractors
    return RetrievalProbe(
        corpus_ids=[record.record_id for record in selected_corpus],
        corpus_texts=[record.text for record in selected_corpus],
        query_ids=[record.record_id for record in selected_queries],
        query_texts=[record.text for record in selected_queries],
        qrels=selected_qrels,
        top_k=min(top_k, len(selected_corpus)),
        requested_query_limit=query_limit,
        requested_corpus_size=corpus_size,
    )
