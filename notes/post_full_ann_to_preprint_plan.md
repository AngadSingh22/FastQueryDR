# Post-Full-ANN To Pre-Print Plan

## Full ANN Run

Run the full exact-vs-ANN comparison on the frozen best student:

```bash
cd "/home/cis-lab/Angad Singh Ahuja/Cloned Repositeries/Private/FastQueryDR" && .venv/bin/python scripts/run_ann_comparison.py --config configs/student_query4_pool_mean_msmarco.yaml --checkpoint results/runs/student_bge_query4_pool_mean_msmarco_20260416_230108/best_model.pt
```

Expected output:

- A new run directory under `results/runs/`
- `ann_comparison.json` inside that run directory

Target comparison:

- `flat` exact FAISS inner-product index
- `hnsw`
- `ivf`

Metrics to capture:

- `MRR@10`
- `Recall@100`
- `index_build_seconds`
- `search_seconds`
- `mean_search_ms`
- `query_count`

## Immediate Post-Run Tasks

1. Open the produced `ann_comparison.json`.
2. Verify all three index families are present: `flat`, `hnsw`, `ivf`.
3. Check for obvious failures:
   - ANN quality collapsing far below `flat`
   - IVF training failure or suspiciously low recall
   - search latency not improving relative to `flat`
4. Extract the numbers into a compact summary table.

## Repo Updates After Full ANN

1. Create `results/ann_table.csv` with one row each for `flat`, `hnsw`, and `ivf`.
2. Append an `ANN Comparison` section to `notes/experiment_log.md`.
3. Add a short interpretation:
   - whether ANN gives a meaningful speedup
   - whether the quality drop is acceptable
   - which ANN index, if any, is worth mentioning in the final write-up
4. If ANN is not useful, state that explicitly and keep `flat` as the main reported system.

Suggested `results/ann_table.csv` columns:

- `index_type`
- `mrr_at_10`
- `recall_at_100`
- `index_build_seconds`
- `search_seconds`
- `mean_search_ms`
- `query_count`
- `notes`

## Result Packaging After ANN

1. Re-check `results/main_table.csv`.
2. Re-check `results/latency_table.csv`.
3. Decide whether ANN belongs in:
   - the main table
   - a separate systems table
4. If useful, add a second figure:
   - `figures/ann_search_latency_vs_mrr.png`

Do not replace the main Pareto figure unless ANN clearly changes the story.

## Claims To Lock After ANN

After ANN results are in, finalize the project claims:

1. Zero-shot BGE remains the strongest teacher reference under the current setup.
2. Naive shallow query truncation causes catastrophic retrieval collapse.
3. Query-side mean pooling substantially improves the shallow 4-layer student.
4. The tested projection head hurts retrieval quality.
5. The tested lightweight distillation is near-neutral under the current setup.
6. ANN either:
   - provides a useful search-speed tradeoff at acceptable quality cost, or
   - does not justify replacing exact search for this study.

## Pre-Print Writing Checklist

Once ANN results are integrated, move to paper writing.

### Abstract

Write one short abstract with:

- problem: online-efficient dense retrieval
- method: asymmetric query compression with a frozen full-depth document encoder
- main finding: shallow query compression hurts strongly without care
- best positive result: query-side pooling helps
- systems result: include ANN only if it materially helps

### Introduction

Cover:

- why query-time cost matters more than offline document encoding cost
- why asymmetric encoders are the right target
- what this project tested
- what the main empirical conclusion is

### Method

Describe:

- frozen zero-shot teacher reference
- asymmetric student design
- 4-layer and 2-layer truncation
- pooling, projection, and distillation ablations
- retrieval and latency measurement protocol
- ANN comparison setup if retained

### Experimental Setup

Include:

- MS MARCO-derived training/retrieval slice
- corpus size and query count used in the benchmark
- teacher and student model variants
- metrics: `MRR@10`, `Recall@100`, query latency, end-to-end latency
- SciFact transfer result as optional robustness check

### Results

Use:

- `results/main_table.csv`
- `results/latency_table.csv`
- `results/ann_table.csv` if ANN is kept
- `figures/pareto_latency_vs_mrr.png`

Key comparisons:

- teacher zero-shot vs teacher fine-tune
- raw 4-layer and 2-layer students
- 4-layer mean-pooling student
- projection-head ablation
- distillation ablation
- ANN comparison, if meaningful

### Discussion

Address:

- why shallow compression fails so hard under naive settings
- why pooling is the only clearly positive intervention so far
- why the best student is still far behind the teacher
- what this implies for future student design

### Limitations

State clearly:

- reduced benchmark rather than full MS MARCO passage retrieval
- only one teacher backbone
- limited student intervention space
- limited transfer evaluation
- ANN evaluation is optional and secondary to the main retrieval-quality story

### Conclusion

End with:

- the strongest validated result
- the best current student
- the remaining performance gap
- the next research direction: stronger distillation or better student architecture

## Final Cleanup Before Posting A Pre-Print

1. Ensure every metric quoted in the draft matches repo artifacts.
2. Ensure every table is reproducible from saved run outputs.
3. Remove stale notes or contradictory experimental claims.
4. Confirm the frozen best-student reference is consistent everywhere.
5. Keep one concise narrative:
   - teacher is strong
   - naive compression fails
   - pooling helps
   - other tested interventions do not yet close the gap
