"""In-training evaluation harness for v2.

- `preflight` — leakage gate that runs ONCE at training start. Refuses
  to begin if the tokenized CPT corpus contains any 50-token overlap
  with the locked eval set, or any eval-row id appears in the
  per-source manifests.
- `pulse` — HF `TrainerCallback` that runs a 200-Q dev probe every
  500 steps, an MMLU general-capability probe every 1000 steps, and
  a Hindi no-regression probe every 1000 steps. Writes one JSONL line
  per pulse to `runs/<run_id>/pulse.jsonl`; can hard-stop training
  via `TrainerControl.should_training_stop`.
"""
