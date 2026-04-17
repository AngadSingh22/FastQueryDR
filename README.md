# FastQueryDR

FastQueryDR is an experimental repository for studying online-efficient dense retrieval through asymmetric query compression.

The project focuses on a simple but important systems question: if document encoding is amortized offline and only query encoding is on the online critical path, how much retrieval quality survives when we compress only the query encoder and leave the document encoder strong?

## Project Goal

The repository studies dense retrieval under an explicitly asymmetric deployment model:

- document embeddings are computed offline and indexed once
- query embeddings are computed online at request time
- query latency therefore matters more than document-side cost
- student compression is applied only to the query tower

The core research question is:

> Can shallow query-side compression produce meaningful latency gains without destroying dense retrieval quality?

## What The Study Actually Tested

The experimental progression in this repository is:

1. Establish a strong teacher reference with frozen zero-shot `BAAI/bge-base-en-v1.5`
2. Validate the training, retrieval, and latency pipeline on an MS MARCO-derived benchmark slice
3. Build asymmetric shallow students with 4-layer and 2-layer query encoders while keeping the document tower full-depth and frozen
4. Add one intervention at a time on the student side:
   - query-side mean pooling
   - query-only projection head
   - lightweight teacher-student distillation
5. Compare exact FAISS retrieval against ANN search on the strongest shallow student
6. Package the results into final tables, figures, and a paper draft

This is not a leaderboard-oriented repository. It is designed to isolate failure modes and first-order tradeoffs.

## Final Study Status

- Teacher reference: frozen zero-shot `BAAI/bge-base-en-v1.5`
- Teacher benchmark on the prepared MS MARCO slice:
  `MRR@10 = 0.7670`, `Recall@100 = 0.9882`
- Best shallow student: 4-layer asymmetric query encoder with query-side mean pooling
  `MRR@10 = 0.0479`, `Recall@100 = 0.2492`
- ANN result: `IVF` is the only approximate index that provides a plausible speed-quality tradeoff on the best student

## Main Findings

The repository now supports a clear empirical story:

- Zero-shot BGE is the strongest teacher under the current setup.
- Naive 4-layer and 2-layer query truncation causes catastrophic retrieval collapse.
- Query-side mean pooling is the first clear positive intervention for the shallow 4-layer student.
- The tested projection head hurts retrieval quality.
- The tested lightweight distillation objective is near-neutral.
- On the systems side, `IVF` is the only ANN variant worth discussing as a practical compromise; `HNSW` is faster but too lossy.

## Final Deliverables

- Paper source: [paper/main.tex](/home/cis-lab/Angad%20Singh%20Ahuja/Cloned%20Repositeries/Private/FastQueryDR/paper/main.tex)
- Paper PDF: [paper/main.pdf](/home/cis-lab/Angad%20Singh%20Ahuja/Cloned%20Repositeries/Private/FastQueryDR/paper/main.pdf)
- Paper archive: [FastQueryDR_paper.zip](/home/cis-lab/Angad%20Singh%20Ahuja/Cloned%20Repositeries/Private/FastQueryDR/FastQueryDR_paper.zip)
- Main results: [results/main_table.csv](/home/cis-lab/Angad%20Singh%20Ahuja/Cloned%20Repositeries/Private/FastQueryDR/results/main_table.csv)
- Latency results: [results/latency_table.csv](/home/cis-lab/Angad%20Singh%20Ahuja/Cloned%20Repositeries/Private/FastQueryDR/results/latency_table.csv)
- ANN results: [results/ann_table.csv](/home/cis-lab/Angad%20Singh%20Ahuja/Cloned%20Repositeries/Private/FastQueryDR/results/ann_table.csv)

## Key Configs

Teacher-side:

- Zero-shot teacher reference:
  [configs/teacher_bge_zero_shot_reference.yaml](/home/cis-lab/Angad%20Singh%20Ahuja/Cloned%20Repositeries/Private/FastQueryDR/configs/teacher_bge_zero_shot_reference.yaml)
- Conservative teacher fine-tune ablation:
  [configs/teacher_msmarco_finetune_conservative.yaml](/home/cis-lab/Angad%20Singh%20Ahuja/Cloned%20Repositeries/Private/FastQueryDR/configs/teacher_msmarco_finetune_conservative.yaml)

Student-side:

- Raw 4-layer student:
  [configs/student_query4_msmarco.yaml](/home/cis-lab/Angad%20Singh%20Ahuja/Cloned%20Repositeries/Private/FastQueryDR/configs/student_query4_msmarco.yaml)
- Raw 2-layer student:
  [configs/student_query2_msmarco.yaml](/home/cis-lab/Angad%20Singh%20Ahuja/Cloned%20Repositeries/Private/FastQueryDR/configs/student_query2_msmarco.yaml)
- Best student:
  [configs/student_query4_pool_mean_msmarco.yaml](/home/cis-lab/Angad%20Singh%20Ahuja/Cloned%20Repositeries/Private/FastQueryDR/configs/student_query4_pool_mean_msmarco.yaml)
- Projection-head ablation:
  [configs/student_query4_pool_mean_proj256_msmarco.yaml](/home/cis-lab/Angad%20Singh%20Ahuja/Cloned%20Repositeries/Private/FastQueryDR/configs/student_query4_pool_mean_proj256_msmarco.yaml)
- Distillation ablation:
  [configs/student_query4_pool_mean_distill_msmarco.yaml](/home/cis-lab/Angad%20Singh%20Ahuja/Cloned%20Repositeries/Private/FastQueryDR/configs/student_query4_pool_mean_distill_msmarco.yaml)

## Repository Layout

```text
.
├── configs/         Experiment configurations
├── data/            Local dataset directory (ignored)
├── documentation/   Original proposal and implementation-plan docs
├── paper/           Final paper source, figures, tables, and PDF
├── results/         Final CSV summaries
├── scripts/         Training and evaluation entrypoints
├── src/             Core library code
└── pyproject.toml
```

## Data Format

Expected MS MARCO-style files:

```text
data/msmarco/train_triples.tsv
data/msmarco/corpus.tsv
data/msmarco/dev_queries.tsv
data/msmarco/dev_qrels.tsv
```

Formats:

```text
train_triples.tsv: query<TAB>positive_passage<TAB>negative_passage
corpus.tsv:        passage_id<TAB>passage_text
dev_queries.tsv:   query_id<TAB>query_text
dev_qrels.tsv:     query_id<TAB>passage_id
```

`dev_qrels.tsv` also accepts TREC-style rows:
`query_id<TAB>0<TAB>passage_id<TAB>1`

## Minimal Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
.venv/bin/pip install -e .
```

## Common Commands

Zero-shot teacher evaluation:

```bash
.venv/bin/python scripts/run_retrieval_eval.py --config configs/teacher_bge_zero_shot_reference.yaml
```

Best student training:

```bash
.venv/bin/python scripts/train_teacher.py --config configs/student_query4_pool_mean_msmarco.yaml
```

Latency benchmark:

```bash
.venv/bin/python scripts/run_latency_benchmark.py --config configs/teacher_msmarco_phase3.yaml --checkpoint /path/to/best_model.pt
```

ANN comparison on the best student:

```bash
.venv/bin/python scripts/run_ann_comparison.py --config configs/student_query4_pool_mean_msmarco.yaml --checkpoint /path/to/best_model.pt
```

Paper build:

```bash
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

## Implementation Notes

- Query texts use the BGE retrieval instruction prefix:
  `Represent this sentence for searching relevant passages: `
- The strongest teacher in this repository is the frozen zero-shot model, not a fine-tuned checkpoint.
- The document tower stays full-depth and frozen in the asymmetric student experiments.
- Latency measurement uses batch-size-1 query encoding and separates query encoding cost from search cost.
- The paper is the main final artifact; the CSV files provide the compact quantitative summaries used by the paper.

## Scope And Limitations

This repository should be read as a controlled study, not a full production retriever:

- the benchmark is an MS MARCO-derived slice, not the full passage-ranking benchmark
- only one teacher backbone is used
- the shallow student intervention space is intentionally small
- the transfer evaluation is limited
- the ANN comparison is secondary to the encoder-side compression story

That narrowness is deliberate: the project is built to make the failure modes visible and reproducible.
