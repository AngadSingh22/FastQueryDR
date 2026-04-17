# Teacher Reference

Current frozen teacher reference:

- Model: `BAAI/bge-base-en-v1.5`
- Pooling: `cls`
- Mode: zero-shot, no fine-tuning
- Benchmark config: [configs/teacher_bge_zero_shot_reference.yaml](/home/cis-lab/Angad%20Singh%20Ahuja/Cloned%20Repositeries/Private/FastQueryDR/configs/teacher_bge_zero_shot_reference.yaml)

Measured on the prepared MS MARCO slice:

- `MRR@10 = 0.7669974473169981`
- `Recall@100 = 0.9882369522712824`

Source:

- [results/runs/teacher_bge_base_msmarco_phase3_retrieval_20260414_165738/retrieval_metrics.json](/home/cis-lab/Angad%20Singh%20Ahuja/Cloned%20Repositeries/Private/FastQueryDR/results/runs/teacher_bge_base_msmarco_phase3_retrieval_20260414_165738/retrieval_metrics.json)

Fine-tuned reference status:

- The patched in-batch contrastive run recovered usable retrieval but still underperformed zero-shot:
  `MRR@10 = 0.5320705435004258`, `Recall@100 = 0.8619696642556408`
- Fine-tuning remains an ablation path, not the default teacher definition.

Recommended commands:

```bash
cd "/home/cis-lab/Angad Singh Ahuja/Cloned Repositeries/Private/FastQueryDR" && .venv/bin/python scripts/run_retrieval_eval.py --config configs/teacher_bge_zero_shot_reference.yaml
```

```bash
cd "/home/cis-lab/Angad Singh Ahuja/Cloned Repositeries/Private/FastQueryDR" && .venv/bin/python scripts/train_teacher.py --config configs/teacher_msmarco_finetune_conservative.yaml
```
