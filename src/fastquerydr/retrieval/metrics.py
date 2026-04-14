from __future__ import annotations


def mean_reciprocal_rank_at_k(results: list[list[str]], qrels: dict[str, set[str]], query_ids: list[str], k: int) -> float:
    scores: list[float] = []
    for query_id, ranked_doc_ids in zip(query_ids, results):
        relevant = qrels.get(query_id, set())
        reciprocal_rank = 0.0
        for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
            if doc_id in relevant:
                reciprocal_rank = 1.0 / rank
                break
        scores.append(reciprocal_rank)
    return sum(scores) / len(scores) if scores else 0.0


def recall_at_k(results: list[list[str]], qrels: dict[str, set[str]], query_ids: list[str], k: int) -> float:
    recalls: list[float] = []
    for query_id, ranked_doc_ids in zip(query_ids, results):
        relevant = qrels.get(query_id, set())
        if not relevant:
            continue
        hits = sum(1 for doc_id in ranked_doc_ids[:k] if doc_id in relevant)
        recalls.append(hits / len(relevant))
    return sum(recalls) / len(recalls) if recalls else 0.0
