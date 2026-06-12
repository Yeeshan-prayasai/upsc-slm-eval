"""Pre-flight gate that runs ONCE at training start.

Refuses to begin training if:
1. The locked eval set's `question_id` set intersects any ID recorded
   in `data/cpt_raw/local_db/.../rows.index.jsonl` (the only acquired
   source that carries per-row IDs).
2. Any 50-token contiguous overlap exists between an eval gold-text
   passage and the deduplicated CPT corpus (`data/cpt_clean_dedup/`).
3. Either of the two tokenized Parquet outputs is missing or empty.

The leakage check at this stage is REDUNDANT with the corpus-build
gate in `build_cpt_corpus.py` — but redundancy is the point. The
corpus-build gate runs against `data/cpt_clean_dedup/`; this gate runs
against the tokenized Parquet that the trainer actually reads, so a
packaging error (wrong tokenization, accidental re-introduction of
eval rows during the .arrow build) would still get caught here.

This module is invoked automatically at the start of `run_cpt.py`
and `run_sft.py`. It can also be called as a standalone CLI for
ad-hoc verification:

    python -m training.eval.preflight                        # check current state
    python -m training.eval.preflight --tokenizer gemma      # check only gemma corpus
    python -m training.eval.preflight --no-ngram             # skip 50-gram check (fast)

Exits with code 0 on clean, 2 on any leakage finding, 3 on missing
required files.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from ..data.acquire._base import RepoPaths
from ..data import leakage as leakage_mod

REPO = RepoPaths.root()
EVAL_SET = REPO / "data" / "eval_set.parquet"
CPT_CLEAN_DEDUP = REPO / "data" / "cpt_clean_dedup"
TOKENIZED = {
    "gemma": REPO / "data" / "cpt_corpus_gemma.parquet",
    "qwen":  REPO / "data" / "cpt_corpus_qwen.parquet",
}


@dataclass
class PreflightReport:
    """Result of running the gate."""
    eval_set_path: str
    eval_set_rows: int
    cpt_corpus_paragraphs: int = 0
    tokenized_files_ok: dict = field(default_factory=dict)   # tokenizer → bool
    tokenized_sequences: dict = field(default_factory=dict)  # tokenizer → int
    leakage_report: leakage_mod.LeakageReport | None = None
    fatal_errors: list[str] = field(default_factory=list)

    def is_clean(self) -> bool:
        return (not self.fatal_errors
                and (self.leakage_report is None or self.leakage_report.is_clean()))


def check_tokenized(tokenizer_key: str) -> tuple[bool, int]:
    """Returns (exists_and_nonempty, sequence_count). Sequence count
    via Parquet row count without loading the full file."""
    path = TOKENIZED.get(tokenizer_key)
    if not path or not path.exists():
        return False, 0
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(str(path))
        return True, pf.metadata.num_rows
    except Exception:
        return False, 0


def check_id_intersection(eval_ids: set[str], cpt_raw_root: Path) -> set[str]:
    """Walk every `*.index.jsonl` file under `cpt_raw_root` (only
    `local_db/` produces these) and look for any eval-row id."""
    hits: set[str] = set()
    if not cpt_raw_root.exists():
        return hits
    for idx_path in cpt_raw_root.rglob("*.index.jsonl"):
        with idx_path.open(encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rid = str(rec.get("id", ""))
                if rid and rid in eval_ids:
                    hits.add(rid)
    return hits


def run_preflight(
    tokenizer_keys: list[str],
    skip_ngram: bool = False,
    require_tokenized: bool = True,
) -> PreflightReport:
    """Run the full gate. Returns a `PreflightReport`; caller should
    check `.is_clean()` and decide whether to abort.

    `require_tokenized=False` lets you preflight before tokenization
    (corpus build phase); set to True when called from the actual
    trainer-start path.
    """
    rep = PreflightReport(eval_set_path=str(EVAL_SET), eval_set_rows=0)

    # 0. Eval set exists
    if not EVAL_SET.exists():
        rep.fatal_errors.append(f"eval_set.parquet not found at {EVAL_SET}")
        return rep

    # 1. Build eval indices
    print(f"[preflight] loading eval set: {EVAL_SET}")
    eval_ids, hash_to_qid, gram_to_qids = leakage_mod.build_eval_index(EVAL_SET)
    rep.eval_set_rows = len(eval_ids)
    print(f"  {len(eval_ids)} eval rows, "
          f"{len(hash_to_qid)} text-hashes, "
          f"{len(gram_to_qids):,} distinct 50-grams")

    # 2. ID-level intersection across local_db row indices
    cpt_raw_root = REPO / "data" / "cpt_raw"
    print(f"[preflight] checking id-level intersection vs {cpt_raw_root}")
    id_hits = check_id_intersection(eval_ids, cpt_raw_root)
    if id_hits:
        rep.fatal_errors.append(
            f"id-level overlap: {len(id_hits)} eval question_ids found "
            f"in per-source row indices. Examples: {sorted(id_hits)[:5]}"
        )

    # 3. Tokenized parquet existence + non-emptiness
    for key in tokenizer_keys:
        ok, n_seq = check_tokenized(key)
        rep.tokenized_files_ok[key] = ok
        rep.tokenized_sequences[key] = n_seq
        if not ok and require_tokenized:
            rep.fatal_errors.append(
                f"tokenized corpus for '{key}' missing or empty at "
                f"{TOKENIZED[key]}. Run `make build-cpt-corpus` first."
            )
        if ok:
            print(f"  ✓ {key}: {TOKENIZED[key].name} — {n_seq:,} sequences")

    # 4. 50-gram contiguous overlap on the deduplicated CPT corpus.
    # This is the *content-level* check, complementing the ID check.
    if not skip_ngram and CPT_CLEAN_DEDUP.exists():
        print(f"[preflight] 50-gram leakage scan vs {CPT_CLEAN_DEDUP}")
        paragraphs = (
            list(leakage_mod.iter_text_files([CPT_CLEAN_DEDUP]))
            + list(leakage_mod.iter_jsonl_text([CPT_CLEAN_DEDUP]))
        )
        rep.cpt_corpus_paragraphs = len(paragraphs)
        print(f"  {len(paragraphs):,} paragraphs to scan")
        rep.leakage_report = leakage_mod.check_corpus_text(
            paragraphs=paragraphs,
            eval_ids=eval_ids,
            hash_to_qid=hash_to_qid,
            gram_to_qids=gram_to_qids,
            item_ids=None,
        )
    elif skip_ngram:
        print(f"[preflight] --no-ngram: skipping 50-gram scan")
    else:
        rep.fatal_errors.append(
            f"cleaned/deduped corpus missing at {CPT_CLEAN_DEDUP}. "
            f"Run `make build-cpt-corpus` first."
        )

    return rep


def render(rep: PreflightReport) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("PRE-FLIGHT LEAKAGE GATE")
    lines.append("=" * 60)
    lines.append(f"  eval_set: {rep.eval_set_path}")
    lines.append(f"  eval rows: {rep.eval_set_rows}")
    lines.append(f"  cpt paragraphs scanned: {rep.cpt_corpus_paragraphs}")
    for key, ok in rep.tokenized_files_ok.items():
        n = rep.tokenized_sequences.get(key, 0)
        lines.append(f"  tokenized[{key}]: {'OK' if ok else 'MISSING'} "
                     f"({n:,} sequences)")
    lines.append("")
    if rep.fatal_errors:
        lines.append("✗ FATAL ERRORS:")
        for e in rep.fatal_errors:
            lines.append(f"    - {e}")
    if rep.leakage_report:
        lines.append(rep.leakage_report.render())
    if rep.is_clean():
        lines.append("✓ PRE-FLIGHT CLEAN — safe to begin training")
    else:
        lines.append("✗ PRE-FLIGHT FAILED — DO NOT TRAIN until cleared")
    lines.append("=" * 60)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Pre-flight leakage gate for training.")
    p.add_argument("--tokenizer", choices=("gemma", "qwen", "both"), default="both",
                   help="Which tokenized corpus to verify")
    p.add_argument("--no-ngram", action="store_true",
                   help="Skip the 50-token contiguous overlap scan (much faster)")
    p.add_argument("--no-tokenized", action="store_true",
                   help="Don't require tokenized parquet to exist "
                        "(useful for pre-corpus-build verification)")
    args = p.parse_args(argv)

    keys = (["gemma", "qwen"] if args.tokenizer == "both" else [args.tokenizer])
    rep = run_preflight(keys, skip_ngram=args.no_ngram,
                        require_tokenized=not args.no_tokenized)
    print(render(rep))
    return 0 if rep.is_clean() else 2


if __name__ == "__main__":
    sys.exit(main())
