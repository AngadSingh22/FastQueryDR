# Online-Efficient Dense Retrieval via Asymmetric Query Encoders

## What This Project Is
This project studies dense retrieval under online efficiency constraints. The main idea is simple: keep the document encoder strong, aggressively shrink only the query encoder, and measure how much retrieval quality can be preserved under latency and memory limits.

## What We Are Trying to Learn
We want to know whether shallow or otherwise cheaper query encoders can remain useful in a realistic dense retrieval pipeline, and which internal choices matter most when the query side is compressed.

## Main Setup
We compare a strong symmetric teacher against asymmetric student retrievers where the document encoder stays full-depth but the query encoder is truncated. We then test whether simple architectural or training changes improve the tradeoff.

## Candidate Changes
- change pooling
- add a small projection head
- add lightweight distillation

## Datasets
The default plan is to use a manageable MS MARCO setup for the main result and optionally test transfer on a small number of BEIR datasets.

## What We Report
- MRR@10
- Recall@100
- p50 and p95 query latency
- query memory footprint
- full online retrieval latency

## Why This Is Interesting
Dense retrieval is often limited not only by model quality but by production constraints. Query encoding is on the online path, so making it cheaper without collapsing retrieval quality is a meaningful systems and modeling problem.

## Repository Intent
This repository is meant to stay narrow and readable. The goal is not to collect every possible experiment. The goal is to produce one clean study with a defensible claim about the latency-quality tradeoff for asymmetric query encoders.

## Suggested Folder Layout
```text
.
├── README.md
├── 1_mini_proposal.md
├── 2_implementation_plan.md
├── paper/
│   ├── main.tex
│   └── refs.bib
├── src/
├── scripts/
├── configs/
├── results/
├── figures/
└── notes/
```

## Status
Documentation scaffold only. Experimental implementation and results are to be added.
