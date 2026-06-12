"""Warmup-Stable-Decay (WSD) learning-rate scheduler.

Implementation follows **Wen et al. 2024** ("Understanding Warmup-Stable-
Decay Learning Rates", arXiv 2410.05192) and the variant adopted for
continued pretraining in **Ibrahim et al. 2024** ("Simple and Scalable
Strategies to Continually Pre-train Large Language Models",
arXiv 2403.08763).

Three phases over `total_steps`:

1. **Warmup** — linear 0 → peak_lr over `warmup_frac · total_steps` steps
2. **Stable** — constant peak_lr over `stable_frac · total_steps` steps
3. **Decay**  — cosine peak_lr → `min_lr_ratio · peak_lr` over the
   remaining `1 − warmup_frac − stable_frac` fraction

The defaults match the methodology doc §4.3:
- warmup_frac=0.01, stable_frac=0.80, min_lr_ratio=0.1 → decay_frac=0.19

WSD beats cosine for CPT because the stable phase preserves a constant
LR floor over most of training (where domain knowledge accumulates),
and the decay tail anneals just enough at the end to settle into a
sharper minimum without the gradient-norm collapse cosine exhibits on
distribution-intensification regimes.

The scheduler is implemented as `torch.optim.lr_scheduler.LambdaLR` so
it composes with HF Trainer's gradient-accumulation and resumption
logic without modification.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


@dataclass(frozen=True)
class WSDConfig:
    """Configuration for the WSD scheduler.

    `min_lr_ratio` is the floor LR as a fraction of peak (default 0.1
    → end of decay reaches 10 % of peak). Wen et al. §3.2 finds the
    final loss is relatively insensitive to this value in the 0.05-0.2
    band; 0.1 is the most commonly cited default.
    """
    total_steps: int
    warmup_frac: float = 0.01
    stable_frac: float = 0.80
    min_lr_ratio: float = 0.1

    def __post_init__(self) -> None:
        if not (0 <= self.warmup_frac < 1):
            raise ValueError(f"warmup_frac out of [0, 1): {self.warmup_frac}")
        if not (0 <= self.stable_frac <= 1):
            raise ValueError(f"stable_frac out of [0, 1]: {self.stable_frac}")
        if self.warmup_frac + self.stable_frac >= 1:
            raise ValueError(
                f"warmup_frac + stable_frac must leave room for decay "
                f"(got {self.warmup_frac} + {self.stable_frac} >= 1)"
            )
        if not (0 < self.min_lr_ratio <= 1):
            raise ValueError(f"min_lr_ratio must be in (0, 1]: {self.min_lr_ratio}")
        if self.total_steps <= 0:
            raise ValueError(f"total_steps must be positive: {self.total_steps}")

    @property
    def warmup_steps(self) -> int:
        return max(1, int(round(self.total_steps * self.warmup_frac)))

    @property
    def stable_end_step(self) -> int:
        return self.warmup_steps + int(round(self.total_steps * self.stable_frac))


def wsd_lambda(step: int, cfg: WSDConfig) -> float:
    """Return the LR multiplier at integer training step `step`.
    Multiplier is in [min_lr_ratio, 1.0]; the optimizer's base LR is
    `peak_lr`, so the realized LR is `peak_lr * wsd_lambda(step)`."""
    # Phase 1: linear warmup
    if step < cfg.warmup_steps:
        return float(step + 1) / float(cfg.warmup_steps)
    # Phase 2: stable
    if step < cfg.stable_end_step:
        return 1.0
    # Phase 3: cosine decay from 1.0 to min_lr_ratio
    decay_steps = cfg.total_steps - cfg.stable_end_step
    if decay_steps <= 0:
        return cfg.min_lr_ratio
    progress = (step - cfg.stable_end_step) / float(decay_steps)
    progress = min(max(progress, 0.0), 1.0)
    # Half-cosine that runs from 1.0 → min_lr_ratio:
    #   lr_mul = min + (1 - min) * 0.5 * (1 + cos(pi * progress))
    cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cos_factor


def build_wsd_scheduler(optimizer: Optimizer, cfg: WSDConfig) -> LambdaLR:
    """Build a `LambdaLR` driving `optimizer` with the WSD curve."""
    return LambdaLR(optimizer, lr_lambda=lambda step: wsd_lambda(step, cfg))
