# Qualitative Comparison

Compared files:

- Zero-shot: [notes/zero_shot_examples_probe.json](/home/cis-lab/Angad%20Singh%20Ahuja/Cloned%20Repositeries/Private/FastQueryDR/notes/zero_shot_examples_probe.json)
- Patched fine-tuned: [notes/trained_examples_20260414_190617_probe.json](/home/cis-lab/Angad%20Singh%20Ahuja/Cloned%20Repositeries/Private/FastQueryDR/notes/trained_examples_20260414_190617_probe.json)

Probe setup:

- `20` qrel-backed dev queries
- `5000`-document probe corpus containing all relevant passages plus sampled distractors

High-level result:

- Zero-shot top-1 hits: `18/20`
- Fine-tuned top-1 hits: `14/20`
- Zero-shot top-5 hits: `19/20`
- Fine-tuned top-5 hits: `17/20`
- Fine-tuned model produced no top-1 recovery that zero-shot missed on this probe.

Observed failure modes in the fine-tuned model:

- It drifts from direct definitional matches toward broader nearby topics.
  Query `+is biology a social science`
  Zero-shot ranked the biology definition first; fine-tuned moved a sociology page above it.

- It overweights local token overlap and weak clinical context over the exact abbreviation expansion.
  Query `+what does ca cells mean urine test`
  Zero-shot ranked the urine calcium explanation first; fine-tuned preferred a hematuria page.

- It loses rare acronym grounding.
  Query `+what is cchaps`
  Zero-shot found the exact `CCHAPS` expansion; fine-tuned missed the relevant passage entirely in top-5.

- It confuses nearby acronym families.
  Query `.vbs what it mean`
  Zero-shot returned the exact `VBS` sense; fine-tuned switched to `VBA`.

Interpretation:

- The patched fine-tuning recipe no longer collapses retrieval.
- The remaining degradation is mostly a precision problem on exact-definition and acronym-heavy queries.
- Zero-shot BGE should remain the frozen teacher reference for student compression work.
