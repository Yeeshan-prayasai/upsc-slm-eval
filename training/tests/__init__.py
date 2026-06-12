"""Unit tests for training-side correctness — production-grade asserts:

- `test_leakage`: synthetic eval-row injection → gate must fail loud
- `test_length_penalty`: 3-row batch with known lengths → loss = CE + 0.05·penalty exact
- `test_wsd_scheduler`: 1000-step shape (warmup → stable → cosine decay)

Run via:
    pytest training/tests/ -v
"""
