# Phase 5 Implementation Plan

## Source-Grounded Constraints

- Official BGE usage keeps retrieval queries on the documented retrieval instruction and uses first-token CLS pooling for embeddings.
  Source: [BAAI/bge-base-en-v1.5 model card](https://huggingface.co/BAAI/bge-base-en-v1.5)
- Official FlagEmbedding code documents the default pooling method as `cls`.
  Source: [FlagEmbedding encoder-only base docs](https://bge-model.com/_modules/FlagEmbedding/inference/embedder/encoder_only/base.html)
- Sentence-Transformers documents pooling as a modular low-cost change with supported modes including `cls` and `mean`.
  Source: [Sentence-Transformers pooling module docs](https://www.sbert.net/docs/package_reference/sentence_transformer/modules.html)
- SimCLR reports that a learnable nonlinear projection between representation and contrastive loss can help, but that is a separate ablation family.
  Source: [SimCLR paper page](https://huggingface.co/papers/2002.05709)
- RocketQA emphasizes stronger negatives and training changes, but those are training-family interventions, not the first clean architecture-family ablation.
  Source: [RocketQA paper page](https://huggingface.co/papers/2010.08191)

## Integrated Decision

Inference from the sources above:

- The frozen teacher and document tower should stay on official BGE behavior: retrieval instruction on queries only, no passage instruction, and `cls` passage pooling.
- The first Phase 5 ablation should therefore be query-side pooling only, not a global pooling rewrite.
- Projection heads and harder-negative / distillation changes should be deferred until the pooling family is evaluated in isolation.

## First Phase 5 Experiment

Target model:

- 4-layer asymmetric student

Change:

- query pooling: `mean`
- passage pooling: `cls`

Rationale:

- It is the smallest architecture-side change that can plausibly help shallow query encoders recover token-level information without changing the frozen passage tower or mixing in a second ablation family.

Config:

- [configs/student_query4_pool_mean_msmarco.yaml](/home/cis-lab/Angad%20Singh%20Ahuja/Cloned%20Repositeries/Private/FastQueryDR/configs/student_query4_pool_mean_msmarco.yaml)

## Second Phase 5 Experiment

Condition for moving on:

- only after the query-side pooling ablation shows a real gain over the raw 4-layer student

Target model:

- 4-layer asymmetric student with the improved query-side mean pooling

Change:

- add a lightweight query-only projection head
- keep the passage tower unchanged
- keep the loss and training recipe unchanged

Rationale:

- SimCLR motivates a small projection head as a low-cost representation-space adjustment.
- Applying it only on the query side preserves the clean asymmetric focus and avoids changing the frozen passage tower.

Config:

- [configs/student_query4_pool_mean_proj256_msmarco.yaml](/home/cis-lab/Angad%20Singh%20Ahuja/Cloned%20Repositeries/Private/FastQueryDR/configs/student_query4_pool_mean_proj256_msmarco.yaml)
