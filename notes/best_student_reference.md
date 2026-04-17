# Best Student Reference

- Current best student run: `student_bge_query4_pool_mean_msmarco_20260416_230108`
- Retrieval metrics: `MRR@10 = 0.0479215`, `Recall@100 = 0.2492314`
- Reason: query-side mean pooling is the first clear positive ablation on the 4-layer student.

Use this checkpoint as the student reference for Phase 7 transfer evaluation and any exact-vs-ANN comparison:

- [best_model.pt](/home/cis-lab/Angad%20Singh%20Ahuja/Cloned%20Repositeries/Private/FastQueryDR/results/runs/student_bge_query4_pool_mean_msmarco_20260416_230108/best_model.pt)
