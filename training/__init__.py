"""UPSC SLM training pipeline.

Modules:
    training.data.acquire — per-source raw corpus acquirers
    training.data         — OCR + clean + dedupe + leakage-gate + tokenize + pack
    training.trainers     — CPT (HF Trainer) + SFT (trl SFTTrainer w/ length-penalty)
    training.eval         — pre-flight leakage gate + in-training pulse callbacks
    training.orchestration — ablation grid driver + CLI entrypoints
"""

__version__ = "1.0.0"
