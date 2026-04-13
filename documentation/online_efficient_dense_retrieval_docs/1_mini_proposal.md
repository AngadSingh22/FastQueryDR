# Mini-Proposal

## Project Title
Online-Efficient Dense Retrieval via Asymmetric Query Encoders

## Problem Statement
Dense retrieval systems often perform well only when both the query encoder and document encoder are relatively large. In production settings, this is often undesirable because query encoding sits directly on the online path and must satisfy strict latency and memory limits. This project studies whether retrieval quality can be preserved when only the query encoder is aggressively compressed while the document encoder remains strong.

## Core Question
Under a fixed online latency budget, how much retrieval quality can be preserved by shrinking only the query encoder, and which internal architectural choices matter most for preserving quality?

## Objectives
1. Build a clean dense retrieval benchmark with a strong symmetric teacher bi-encoder.
2. Construct asymmetric students by truncating only the query encoder.
3. Evaluate whether pooling, projection heads, or lightweight distillation recover quality in shallow query encoders.
4. Measure the tradeoff between retrieval quality and online efficiency.
5. Produce a concise experimental note with one or two defensible claims.

## Scope
This project is intentionally narrow. It is not meant to solve dense retrieval in general. It is meant to isolate the online query-side efficiency problem in a controlled setting.

## Main Hypotheses
H1. A 4-layer query encoder preserves a large fraction of full-model retrieval quality under a meaningful latency reduction.

H2. For shallow query encoders, internal design choices such as pooling matter enough to partially offset depth reduction.

H3. A lightweight teacher-student distillation term improves the quality of shallow query encoders at low additional training cost.

## Experimental Systems
### System A: Teacher Baseline
Full dual encoder with the same strong encoder on both query and document sides.

### System B: Asymmetric Student, 4 Layers
Full document encoder retained. Query encoder truncated to 4 layers.

### System C: Asymmetric Student, 2 Layers
Full document encoder retained. Query encoder truncated to 2 layers.

### System D: Modified Low-Cost Variant
One research-oriented variant built on the shallow query encoder. Candidate directions:
- alternative pooling
- small projection head
- lightweight distillation objective
- a combination of one architecture and one training change

## Architecture Summary
- Query encoder: truncated or full transformer-based text encoder
- Document encoder: strong full-depth encoder
- Similarity: dot product or cosine similarity
- Retrieval backend: FAISS
- Training loss: in-batch contrastive objective, optionally with distillation

## Datasets
### Primary Training / Evaluation
MS MARCO passage ranking, or a manageable subset if compute is limited.

### Optional Transfer Evaluation
One or two BEIR datasets such as:
- SciFact
- FiQA

## Metrics
### Retrieval Quality
- MRR@10
- Recall@100
- nDCG@10 (optional)

### Online Systems Metrics
- p50 query encoding latency
- p95 query encoding latency
- full online retrieval latency
- query-side memory footprint

### Secondary Metrics
- document embedding throughput
- FAISS search latency
- index size

## Key Ablations
- full depth vs 4-layer vs 2-layer query encoder
- mean pooling vs CLS pooling
- with vs without projection head
- with vs without distillation
- exact FAISS vs approximate FAISS in a later-stage systems comparison

## Success Criteria
The project is successful if it yields at least one clear and defensible statement of the following form:
- shallow asymmetric query encoders preserve most retrieval quality under a given latency budget
- pooling choice matters significantly once query depth is reduced
- lightweight distillation materially improves the latency-quality tradeoff

## Expected Deliverables
- training and evaluation code
- benchmark tables
- one latency-quality tradeoff plot
- short manuscript-style note

## Open Decisions
- final teacher backbone
- exact MS MARCO subset size
- whether to use mined hard negatives
- exact distillation formulation
- which ANN index to compare after the clean baseline phase
