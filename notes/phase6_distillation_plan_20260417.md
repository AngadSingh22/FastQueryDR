# Phase 6 Distillation Plan

## Source-Grounded Direction

- RocketQA shows that dense retrieval training is sensitive to training strategy, especially negatives and training/inference mismatch.
  Source: [RocketQA paper page](https://huggingface.co/papers/2010.08191)
- Recent DPR distillation work shows that teacher-student knowledge transfer is effective, but increasingly elaborate methods often combine extra assistants, harder data, or iterative schedules.
  Source: [MTA4DPR ACL page](https://aclanthology.org/2024.emnlp-main.336/)

## Minimal Phase 6 Choice

Inference from the sources above:

- The first distillation variant in this repo should stay lightweight.
- Do not change negatives, data, or iterative curriculum at the same time.
- Keep the current best student architecture fixed and add only one teacher-student term.

## Implemented Variant

Student base:

- 4-layer asymmetric student
- query pooling: `mean`
- passage pooling: `cls`
- frozen full-depth passage tower

Teacher:

- frozen zero-shot `BAAI/bge-base-en-v1.5`
- query pooling: `cls`
- passage pooling: `cls`

Loss:

- existing in-batch contrastive loss
- plus KL divergence between student and teacher in-batch similarity distributions

Config:

- [configs/student_query4_pool_mean_distill_msmarco.yaml](/home/cis-lab/Angad%20Singh%20Ahuja/Cloned%20Repositeries/Private/FastQueryDR/configs/student_query4_pool_mean_distill_msmarco.yaml)
