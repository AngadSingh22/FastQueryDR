# FastQueryDR

Initial scaffold for studying online-efficient dense retrieval via asymmetric query encoders.

## Phase 1 Decisions

- Teacher encoder: `BAAI/bge-base-en-v1.5`
- Initial dataset slice: first `100,000` MS MARCO training triples, with `5,000` held out for validation
- First objective: symmetric bi-encoder training with in-batch negatives

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

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python scripts/train_teacher.py --config configs/teacher_msmarco_mini.yaml
```
