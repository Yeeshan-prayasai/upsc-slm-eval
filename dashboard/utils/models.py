"""Lazy model loaders for the Playground.

Loading strategy targets a 16 GB Mac:
- MLX models are loaded on demand and cached via `@st.cache_resource`.
- Only ONE MLX model is kept in memory at a time. Switching evicts the
  previous one via `st.cache_resource.clear()` on the loader.
- Gemini API runners are cheap to construct (no local memory).

Resident-memory expectations (per `experiment-report.md` §3.2):
  Gemma-4-E4B-it 4-bit MLX → ~5.0 GB
  Qwen-3.5-4B    4-bit MLX → ~3.0 GB
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

REPO = Path(__file__).resolve().parent.parent.parent

# Make scripts/ importable so we reuse the existing Runner classes verbatim.
# Same trick scripts/run_inference.py uses.
sys.path.insert(0, str(REPO / "scripts"))

MLX_PATHS = {
    "C1a": REPO / "adapters" / "gemma4-e4b-upsc-v1-mlx",
    "C1b": REPO / "adapters" / "qwen35-4b-upsc-v1-mlx",
}

FT_CORPUS = REPO / "data" / "ft_corpus.parquet"


def mlx_path_for(condition: str) -> Path:
    """Resolve MLX adapter dir for the given condition (C1a or C1b)."""
    if condition not in MLX_PATHS:
        raise ValueError(f"no MLX path for condition {condition!r}; expected C1a or C1b")
    path = MLX_PATHS[condition]
    if not path.exists():
        raise FileNotFoundError(
            f"MLX adapter not found at {path}. "
            f"v1 adapters should have been produced via scripts/run_ft.py."
        )
    return path


@st.cache_resource(max_entries=1, show_spinner="Loading MLX model (one-time, ~10-15s) …")
def load_mlx_runner(condition: str):
    """Lazy-load an MLXLoRARunner for C1a or C1b.

    `max_entries=1` keeps only the most-recently-used model resident; switching
    from C1a to C1b evicts C1a from memory automatically. This is the key
    behavior for 16 GB Mac compatibility.
    """
    from runners import MLXLoRARunner  # noqa: E402  scripts/ on sys.path

    return MLXLoRARunner(base=str(mlx_path_for(condition)), adapter=None)


@st.cache_resource(show_spinner="Connecting to Gemini API …")
def load_gemini_zs(model_name: str):
    """Cached Gemini zero-shot runner (no local memory cost)."""
    from runners import GeminiZeroShotRunner  # noqa: E402

    return GeminiZeroShotRunner(model=model_name)


@st.cache_resource(show_spinner="Connecting to Gemini API + loading FT-corpus exemplars …")
def load_gemini_fs(model_name: str):
    """Cached Gemini few-shot runner (reads ft_corpus.parquet for exemplars)."""
    from runners import GeminiFewShotRunner  # noqa: E402

    return GeminiFewShotRunner(ft_corpus_path=FT_CORPUS, model=model_name)


def evict_mlx() -> None:
    """Clear the MLX runner cache; releases ~3-5 GB of Apple Metal memory.

    Called from the Playground when the user explicitly chooses to free RAM
    (e.g. before switching to a memory-heavy operation). Cache will re-load on
    the next call to `load_mlx_runner`.
    """
    load_mlx_runner.clear()
