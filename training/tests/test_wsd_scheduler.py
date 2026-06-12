"""WSD scheduler correctness tests.

Verifies the math from Wen et al. 2024 §3.2 against the
`training.trainers.schedulers.wsd_lambda` implementation. Catches:
- Off-by-one at phase boundaries
- Cosine decay floor (must hit min_lr_ratio at the last step)
- Monotonicity in warmup + decay phases
- Constant value across the stable phase
"""
from __future__ import annotations

import math

import pytest

from training.trainers.schedulers import WSDConfig, wsd_lambda


def test_wsd_phase_boundaries():
    """Verify the three-phase shape: warmup → stable → decay."""
    cfg = WSDConfig(total_steps=1000, warmup_frac=0.01, stable_frac=0.80,
                    min_lr_ratio=0.1)
    assert cfg.warmup_steps == 10
    assert cfg.stable_end_step == 810

    # Step 0 is the first warmup step — multiplier should be exactly 1/warmup
    assert math.isclose(wsd_lambda(0, cfg), 1.0 / 10, rel_tol=1e-9)
    # End of warmup
    assert math.isclose(wsd_lambda(9, cfg), 1.0, rel_tol=1e-9)
    # Middle of stable
    assert math.isclose(wsd_lambda(500, cfg), 1.0, rel_tol=1e-9)
    # Last step of stable
    assert math.isclose(wsd_lambda(809, cfg), 1.0, rel_tol=1e-9)
    # First step of decay still ~1.0 (cosine just barely off 1)
    assert wsd_lambda(810, cfg) <= 1.0
    # Final step should hit min_lr_ratio (within float tolerance)
    final = wsd_lambda(999, cfg)
    assert math.isclose(final, cfg.min_lr_ratio, abs_tol=1e-3), (
        f"final lr_mul {final} should be ~{cfg.min_lr_ratio}")


def test_wsd_decay_is_monotonic():
    """Cosine decay must be strictly monotonically decreasing."""
    cfg = WSDConfig(total_steps=1000, warmup_frac=0.01, stable_frac=0.80)
    decay_steps = list(range(cfg.stable_end_step, cfg.total_steps))
    values = [wsd_lambda(s, cfg) for s in decay_steps]
    assert all(values[i] >= values[i + 1] for i in range(len(values) - 1)), (
        "decay phase must be monotonically non-increasing")


def test_wsd_warmup_is_monotonic():
    """Warmup phase must be strictly monotonically increasing, reaching 1.0
    at the last warmup step (step warmup_steps-1)."""
    cfg = WSDConfig(total_steps=1000)
    # Iterate ONLY across the warmup phase (0..warmup_steps-1).
    # Step `warmup_steps` is the first stable-phase step, also 1.0 —
    # including it would break strict monotonicity at the boundary.
    warmup_values = [wsd_lambda(s, cfg) for s in range(cfg.warmup_steps)]
    assert all(warmup_values[i] < warmup_values[i + 1]
               for i in range(len(warmup_values) - 1)), (
        "warmup must be strictly increasing")
    assert math.isclose(warmup_values[-1], 1.0, rel_tol=1e-9)


def test_wsd_stable_phase_constant():
    """Every step in the stable phase must return exactly 1.0."""
    cfg = WSDConfig(total_steps=1000, warmup_frac=0.01, stable_frac=0.80)
    stable_steps = list(range(cfg.warmup_steps, cfg.stable_end_step))
    assert all(wsd_lambda(s, cfg) == 1.0 for s in stable_steps), (
        "stable phase must be exactly constant 1.0")


def test_wsd_config_validation():
    """Bad WSDConfig must error at construction time."""
    with pytest.raises(ValueError, match="warmup_frac"):
        WSDConfig(total_steps=100, warmup_frac=1.5)
    with pytest.raises(ValueError, match="stable_frac"):
        WSDConfig(total_steps=100, stable_frac=1.1)
    with pytest.raises(ValueError, match="must leave room for decay"):
        WSDConfig(total_steps=100, warmup_frac=0.5, stable_frac=0.5)
    with pytest.raises(ValueError, match="min_lr_ratio"):
        WSDConfig(total_steps=100, min_lr_ratio=2.0)
    with pytest.raises(ValueError, match="total_steps"):
        WSDConfig(total_steps=0)


def test_wsd_with_torch_lambdalr():
    """Verify end-to-end integration with torch's LambdaLR + AdamW."""
    import torch
    from training.trainers.schedulers import build_wsd_scheduler

    model = torch.nn.Linear(8, 8)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0e-5)
    cfg = WSDConfig(total_steps=1000)
    sched = build_wsd_scheduler(opt, cfg)

    for step in range(1000):
        # Take a no-op optimizer step (no actual gradient) just to drive
        # the scheduler. opt.step() before sched.step() per torch >= 1.1.
        opt.step()
        sched.step()
    # After all steps: realized LR should be at the min-ratio floor.
    final_lr = opt.param_groups[0]["lr"]
    assert math.isclose(final_lr, 1.0e-5 * cfg.min_lr_ratio,
                        abs_tol=2e-7), f"final realized LR {final_lr}"
