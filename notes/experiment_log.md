# Experiment Log

## Phase 1

- Teacher encoder: `BAAI/bge-base-en-v1.5`
- Initial dataset slice: `100,000` MS MARCO triples, with `5,000` validation triples
- Initial training objective: symmetric bi-encoder with in-batch negatives

## Phase 2

- Retrieval backend: `faiss.IndexFlatIP`
- Held-out retrieval inputs: `corpus.tsv`, `dev_queries.tsv`, `dev_qrels.tsv`
- Logged metrics: `MRR@10`, `Recall@100`, corpus encoding time, query encoding time, and search time

## Phase 3

- Latency protocol: fixed warmup plus measured query subset, batch-size-1 encoding
- Reported latency statistics: `p50`, `p95`, and mean for query encoding and end-to-end retrieval
- Reported memory statistic: GPU peak allocation or CPU RSS delta during single-query encoding
