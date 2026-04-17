# Experiment Log

Frozen best student reference:

- Run: `student_bge_query4_pool_mean_msmarco_20260416_230108`
- Rationale: clearest positive ablation result among shallow students

Defensible claims:

- Shrinking to 4 layers causes a large quality drop under the current setup.
- Query-side pooling matters substantially for shallow query encoders.
- The tested projection head hurt retrieval quality.
- The tested lightweight distillation did not materially improve the tradeoff.

Run Summary:

## Teacher Zero-Shot
- Run dir: `teacher_bge_base_zero_shot_reference_retrieval_20260414_220138`
- Family: `teacher`
- MRR@10: `0.766997`
- Recall@100: `0.988237`
- Query p50 latency (ms): `3.0894`
- End-to-end p50 latency (ms): `23.5675`
- Notes: Frozen reference teacher

## Teacher Fine-Tune
- Run dir: `teacher_bge_base_msmarco_finetune_conservative_20260414_221912`
- Family: `teacher_ablation`
- MRR@10: `0.740860`
- Recall@100: `0.983169`
- Query p50 latency (ms): `3.0752`
- End-to-end p50 latency (ms): `23.8368`
- Notes: Optional fine-tuned teacher ablation

## Student Q4 CLS
- Run dir: `student_bge_query4_msmarco_20260415_195330`
- Family: `student`
- MRR@10: `0.000776`
- Recall@100: `0.016888`
- Query p50 latency (ms): `7.1715`
- End-to-end p50 latency (ms): `31.2392`
- Notes: Raw 4-layer student

## Student Q2 CLS
- Run dir: `student_bge_query2_msmarco_20260416_170428`
- Family: `student`
- MRR@10: `0.000465`
- Recall@100: `0.010698`
- Query p50 latency (ms): `4.1066`
- End-to-end p50 latency (ms): `24.8004`
- Notes: Raw 2-layer student

## Student Q4 Mean
- Run dir: `student_bge_query4_pool_mean_msmarco_20260416_230108`
- Family: `student_best`
- MRR@10: `0.047922`
- Recall@100: `0.249231`
- Query p50 latency (ms): `1.8190`
- End-to-end p50 latency (ms): `25.2433`
- Notes: Current best student

## Student Q4 Mean + Proj
- Run dir: `student_bge_query4_pool_mean_proj256_msmarco_20260416_232922`
- Family: `student_ablation`
- MRR@10: `0.000208`
- Recall@100: `0.004516`
- Query p50 latency (ms): `1.8481`
- End-to-end p50 latency (ms): `24.8220`
- Notes: Projection head regression

## Student Q4 Mean + Distill
- Run dir: `student_bge_query4_pool_mean_distill_msmarco_20260417_003541`
- Family: `student_ablation`
- MRR@10: `0.048725`
- Recall@100: `0.246582`
- Query p50 latency (ms): `1.8236`
- End-to-end p50 latency (ms): `23.1720`
- Notes: Lightweight distillation
