# FastQueryDR

Initial scaffold for studying online-efficient dense retrieval via asymmetric query encoders.

## Current Teacher Reference

- Frozen teacher reference: zero-shot `BAAI/bge-base-en-v1.5`
- Reference benchmark on the prepared MS MARCO slice:
  `MRR@10 = 0.7670`, `Recall@100 = 0.9882`
- Fine-tuning is optional and currently treated as an ablation path, not the default teacher definition

## Asymmetric Students

- Phase 4 starts from the frozen zero-shot BGE teacher as the reference model.
- Student experiments truncate only the query encoder and keep the document encoder full-depth and frozen.
- First student configs:
  [configs/student_query4_msmarco.yaml](/home/cis-lab/Angad%20Singh%20Ahuja/Cloned%20Repositeries/Private/FastQueryDR/configs/student_query4_msmarco.yaml)
  [configs/student_query2_msmarco.yaml](/home/cis-lab/Angad%20Singh%20Ahuja/Cloned%20Repositeries/Private/FastQueryDR/configs/student_query2_msmarco.yaml)

## Phase 1 Decisions

- Teacher encoder: `BAAI/bge-base-en-v1.5`
- Initial dataset slice: first `100,000` MS MARCO training triples, with `5,000` held out for validation
- First objective: symmetric bi-encoder training with pure query-positive in-batch contrastive loss
- Query encoding uses the standard BGE retrieval instruction prefix: `Represent this sentence for searching relevant passages: `
- BGE configs use `CLS` pooling and a conservative `1e-5` learning rate to avoid degrading pretrained retrieval behavior during fine-tuning
- The strongest teacher baseline in this repo is the frozen zero-shot BGE checkpoint, not a fine-tuned checkpoint

## Project Layout

```text
.
├── configs/
├── data/
├── documentation/
├── figures/
├── notes/
├── results/
├── scripts/
├── src/
└── pyproject.toml
```

## Expected Data Format

Place a tab-separated file at `data/msmarco/train_triples.tsv` with one triple per line:

```text
query<TAB>positive_passage<TAB>negative_passage
```

The training script will take the configured slice from this file and split off a validation set.

For retrieval evaluation, add:

```text
data/msmarco/corpus.tsv
data/msmarco/dev_queries.tsv
data/msmarco/dev_qrels.tsv
```

Expected formats:

```text
corpus.tsv:      passage_id<TAB>passage_text
dev_queries.tsv: query_id<TAB>query_text
dev_qrels.tsv:   query_id<TAB>passage_id
```

`dev_qrels.tsv` also accepts TREC-style rows such as `query_id<TAB>0<TAB>passage_id<TAB>1`.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python scripts/train_teacher.py --config configs/teacher_msmarco_mini.yaml
python scripts/run_retrieval_eval.py --config configs/teacher_bge_zero_shot_reference.yaml
python scripts/run_retrieval_eval.py --config configs/teacher_msmarco_phase2.yaml --checkpoint /path/to/best_model.pt
python scripts/run_latency_benchmark.py --config configs/teacher_msmarco_phase3.yaml --checkpoint /path/to/best_model.pt
python scripts/export_retrieval_examples.py --config configs/teacher_bge_zero_shot_reference.yaml --output notes/zero_shot_examples_probe.json --probe-queries 20 --probe-corpus-size 5000
python scripts/export_retrieval_examples.py --config configs/teacher_msmarco_phase3.yaml --checkpoint results/runs/<run_name>/best_model.pt --output notes/trained_examples_probe.json --probe-queries 20 --probe-corpus-size 5000
python scripts/train_teacher.py --config configs/student_query4_msmarco.yaml
python scripts/train_teacher.py --config configs/student_query2_msmarco.yaml
```

The first training or evaluation run will also need to download `BAAI/bge-base-en-v1.5` from Hugging Face unless that model is already cached locally.

## Phase 3 Latency Protocol

Latency is measured on a fixed held-out subset of dev queries with:

- batch-size-1 query encoding
- warmup queries excluded from measurement
- end-to-end retrieval timing against a prebuilt FAISS flat index
- query memory footprint reported as GPU peak allocation when on CUDA, otherwise process RSS delta on CPU

## Conservative Fine-Tuning

Use [configs/teacher_msmarco_finetune_conservative.yaml](/home/cis-lab/Angad%20Singh%20Ahuja/Cloned%20Repositeries/Private/FastQueryDR/configs/teacher_msmarco_finetune_conservative.yaml) if you still want a fine-tuned teacher ablation. It tightens the recipe by:

- lowering the learning rate to `5e-6`
- capping training with `max_steps`
- selecting checkpoints by retrieval probe `MRR@10` rather than validation loss
- enabling patience-based early stopping on a fixed probe subset of dev queries and corpus passages

Student configs reuse the same conservative recipe, but switch to `architecture: asymmetric`, set `query_num_hidden_layers` to `4` or `2`, and freeze the full-depth document encoder.
