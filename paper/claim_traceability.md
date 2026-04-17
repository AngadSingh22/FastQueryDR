# Claim Traceability

This file maps every paper-level claim to a concrete repo artifact or note.

## Core Narrative

1. `BAAI/bge-base-en-v1.5` zero-shot is the strongest teacher reference.
   - Source: `notes/teacher_reference.md`
   - Artifact: `results/runs/teacher_bge_base_zero_shot_reference_retrieval_20260414_220138/retrieval_metrics.json`

2. Naive query truncation to 4 and 2 layers causes catastrophic quality collapse.
   - Source: `notes/experiment_log.md`
   - Artifacts:
     - `results/runs/student_bge_query4_msmarco_20260415_195330/retrieval_metrics.json`
     - `results/runs/student_bge_query2_msmarco_20260416_170428/retrieval_metrics.json`

3. Query-side mean pooling is the first clear positive intervention on the shallow 4-layer student.
   - Sources:
     - `notes/best_student_reference.md`
     - `notes/phase5_implementation_plan_20260416.md`
   - Artifact: `results/runs/student_bge_query4_pool_mean_msmarco_20260416_230108/retrieval_metrics.json`

4. The tested projection head hurts retrieval quality.
   - Source: `notes/experiment_log.md`
   - Artifact: `results/runs/student_bge_query4_pool_mean_proj256_msmarco_20260416_232922/retrieval_metrics.json`

5. The tested lightweight distillation is near-neutral.
   - Sources:
     - `notes/phase6_distillation_plan_20260417.md`
     - `notes/experiment_log.md`
   - Artifact: `results/runs/student_bge_query4_pool_mean_distill_msmarco_20260417_003541/retrieval_metrics.json`

6. IVF is the only ANN variant worth discussing as a plausible systems tradeoff.
   - Sources:
     - `notes/experiment_log.md`
     - `results/ann_table.csv`
   - Artifact: `results/runs/student_bge_query4_pool_mean_msmarco_ann_20260417_150805/ann_comparison.json`

7. SciFact is a secondary transfer check, not the main result.
   - Source: `documentation/online_efficient_dense_retrieval_docs/1_mini_proposal.md`
   - Artifact: `results/runs/student_bge_query4_pool_mean_msmarco_beir_scifact_20260417_011110/beir_metrics.json`
