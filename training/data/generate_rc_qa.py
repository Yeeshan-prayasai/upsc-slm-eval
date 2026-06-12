"""Generate synthetic reading-comprehension Q&A over the high-yield
corpus core (NCERT + reference books) for the CPT mix.

Why: factual recall is learned when a model sees MANY phrasings of each
fact (Ovadia et al. 2024, arXiv 2312.05934); raw-text CPT alone injects
little new knowledge at LoRA scale (Biderman et al. 2024). AdaptLLM
(Cheng et al. 2023, arXiv 2309.09530) converts raw domain text into
reading-comprehension tasks and shows it both injects knowledge AND
preserves prompting ability. This script implements that conversion:
for each ~1,500-word chunk of NCERT/reference-book text, an LLM
generates 4-6 Q&A pairs grounded ONLY in that chunk.

Output: `data/cpt_raw/rc_qa/<source>/<doc>.md` — passage excerpt +
Q&A pairs as plain markdown, flowing through the normal clean → dedup
→ leakage gate → mix (add an `rc_qa` entry to data_mix_cpt.yaml when
enabling).

COST: ~paid API generation. The script refuses to run without
`--confirm-cost`; `--dry-run` (default behavior when the flag is
absent) prints the chunk count and an output-token cost estimate.

CLI:
    python -m training.data.generate_rc_qa --dry-run
    python -m training.data.generate_rc_qa --confirm-cost --limit 50
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from .acquire._base import RepoPaths

REPO = RepoPaths.root()
SOURCES = ["ncert", "reference_books"]   # the high-yield Prelims core
CPT_CLEAN_DEDUP = REPO / "data" / "cpt_clean_dedup"
OUT_ROOT = REPO / "data" / "cpt_raw" / "rc_qa"

CHUNK_WORDS = 1500
GEN_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
# Rough planning numbers for the cost estimate (per chunk):
EST_OUT_TOKENS_PER_CHUNK = 600
EST_USD_PER_M_OUT_TOKENS = 0.40   # flash-tier output pricing ballpark

PROMPT_TEMPLATE = """\
You are creating exam-preparation study material. Read the passage and
write 5 question-answer pairs that test the FACTS stated in it. Rules:
- Every answer must be verifiable from the passage alone.
- Vary the question forms: direct factual, "which of the following",
  match-the-pairs, chronology, and cause/effect.
- Answers: 1-3 sentences, stating the fact plainly.
- Output format: "Q: ...\\nA: ..." pairs separated by blank lines.
  No preamble, no numbering, no markdown headers.

PASSAGE:
{passage}
"""


def _chunks(text: str, n_words: int = CHUNK_WORDS) -> "list[str]":
    words = text.split()
    return [" ".join(words[i:i + n_words])
            for i in range(0, len(words), n_words)
            if len(words[i:i + n_words]) >= 200]   # skip tiny tails


def discover_chunks() -> "list[tuple[Path, int, str]]":
    """(source_file, chunk_index, chunk_text) for every chunk in scope."""
    out = []
    for source in SOURCES:
        root = CPT_CLEAN_DEDUP / source
        if not root.exists():
            print(f"  (skip {source} — not under cpt_clean_dedup)", file=sys.stderr)
            continue
        for f in sorted(root.rglob("*.md")):
            text = f.read_text(encoding="utf-8", errors="replace")
            for i, ch in enumerate(_chunks(text)):
                out.append((f, i, ch))
    return out


def generate(chunks, limit: int | None, rate_s: float = 1.0) -> int:
    from google import genai   # lazy: needs GEMINI_API_KEY

    client = genai.Client()    # reads GEMINI_API_KEY from env
    todo = chunks[:limit] if limit else chunks
    n_done = 0
    for f, idx, chunk in todo:
        rel = f.relative_to(CPT_CLEAN_DEDUP)
        out_path = OUT_ROOT / rel.parent / f"{rel.stem}__rc{idx:03d}.md"
        if out_path.exists():
            continue   # resume-safe
        # Transient-error retry (503s are routine on flash-tier endpoints).
        qa = ""
        for attempt in range(5):
            try:
                resp = client.models.generate_content(
                    model=GEN_MODEL,
                    contents=PROMPT_TEMPLATE.format(passage=chunk),
                )
                qa = (resp.text or "").strip()
                break
            except Exception as e:
                wait = 2 ** (attempt + 2)   # 4..64s
                print(f"  retry {attempt+1}/5 after {type(e).__name__} "
                      f"(sleep {wait}s)")
                time.sleep(wait)
        else:
            print(f"  giving up on {out_path.name} after 5 attempts")
            continue
        if not qa.startswith("Q:"):
            print(f"  skip {out_path.name} — unexpected output shape")
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            f"Study Q&A based on {rel}:\n\n{qa}\n", encoding="utf-8")
        n_done += 1
        if n_done % 25 == 0:
            print(f"  [{n_done}/{len(todo)}] generated")
        time.sleep(rate_s)
    print(f"Generated {n_done} RC/QA docs → {OUT_ROOT.relative_to(REPO)}")
    print("Next: re-run clean+dedup+leakage over rc_qa, add an `rc_qa` "
          "entry to data_mix_cpt.yaml (repeat: 2), and re-tokenize.")
    return 0


def main(argv: "list[str] | None" = None) -> int:
    p = argparse.ArgumentParser(description="Generate RC/QA data (AdaptLLM-style).")
    p.add_argument("--confirm-cost", action="store_true",
                   help="Actually call the generation API (paid).")
    p.add_argument("--dry-run", action="store_true",
                   help="Only print chunk count + cost estimate.")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap the number of chunks (smoke runs).")
    args = p.parse_args(argv)

    chunks = discover_chunks()
    n = len(chunks[:args.limit] if args.limit else chunks)
    est_usd = n * EST_OUT_TOKENS_PER_CHUNK / 1e6 * EST_USD_PER_M_OUT_TOKENS
    print(f"Chunks in scope: {n:,} (sources: {SOURCES}; {CHUNK_WORDS}-word chunks)")
    print(f"Estimated output: ~{n * EST_OUT_TOKENS_PER_CHUNK / 1e6:.1f} M tokens, "
          f"~${est_usd:.2f} at flash-tier pricing")

    if args.dry_run or not args.confirm_cost:
        print("\nDry run only. Re-run with --confirm-cost to generate.")
        return 0
    return generate(chunks, args.limit)


if __name__ == "__main__":
    sys.exit(main())
