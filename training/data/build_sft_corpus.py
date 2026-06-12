"""Build the SFT corpus from v1's `data/ft_corpus.parquet`.

Deltas over v1's SFT corpus:
1. **EN-only filter** — Hindi is deferred to v2-hindi-strategy.md
2. **Length control in the prompt** — Task B rows with a known
   `pyqs.word_count` get "Answer in approximately N words." appended
   to the prompt; plain CE learns the association. (Replaces the
   length-penalty loss term, which was computed from label lengths —
   a constant w.r.t. parameters — and trained nothing.)
3. **Conversational prompt-completion format** — rows are
   `{"prompt": [{"role": "user", ...}], "completion": [{"role":
   "assistant", ...}]}` so trl renders them through the tokenizer's
   chat template (matching `scripts/runners.py` at inference) and
   applies completion-only loss. The previous raw-`text` concat
   trained on a framing inference never uses, with full-sequence loss.
4. **Over-length filter** — rows whose estimated token length exceeds
   the 4096 training window are dropped (trl truncates from the END,
   which would cut the gold answer's tail + EOS and teach
   non-stopping). Counts logged per task.
5. **Leakage gate** — every row's prompt+completion text is checked
   against the locked eval set (ID + exact-text + 50-token contiguous,
   same gate as the CPT corpus). Build fails loud on any hit.
6. **Train/valid split** — 95/5 stratified by task (deterministic seed)

Each output row carries:
    {
      "pair_id": str,
      "task": "A" | "B" | "C" | "E",
      "target_word_count": int | None,    # metadata; hint lives in prompt
      "prompt": [{"role": "user", "content": str}],
      "completion": [{"role": "assistant", "content": str}],
    }

The v1 `ft_corpus.parquet` is locked (SHA-pinned); this script does
NOT modify it. We only read it and write fresh JSONL artifacts under
`data/sft_v2/`.

CLI:
    python -m training.data.build_sft_corpus
    python -m training.data.build_sft_corpus --include-hindi   # rare; for debug
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .acquire._base import RepoPaths


REPO = RepoPaths.root()
FT_CORPUS = REPO / "data" / "ft_corpus.parquet"
DB_PATH = RepoPaths.db_snapshot()
OUT_DIR = REPO / "data" / "sft_v2"
DEFAULT_SEED = 20260605
DEFAULT_VALID_FRAC = 0.05

# Over-length filter: estimated tokens = words × this ratio + template
# overhead; rows above the budget are dropped (trl truncates from the
# END — losing the answer tail + EOS — which trains non-stopping).
TOKENS_PER_WORD_EST = 1.4
TEMPLATE_OVERHEAD_TOKENS = 64
MAX_TRAIN_TOKENS = 4096


@dataclass
class BuildReport:
    n_in: int = 0
    n_filtered_hindi: int = 0
    n_task_b_with_target: int = 0
    n_task_b_without_target: int = 0
    n_overlength_dropped: int = 0
    overlength_by_task: dict | None = None
    n_eval_siblings_dropped: int = 0
    n_leakage_rows_dropped: int = 0
    n_train: int = 0
    n_valid: int = 0
    train_path: str = ""
    valid_path: str = ""

    def render(self) -> str:
        return (
            f"SFT corpus build\n"
            f"  input rows (ft_corpus): {self.n_in:,}\n"
            f"  filtered out hindi:     {self.n_filtered_hindi:,}\n"
            f"  task B w/ target_word_count: {self.n_task_b_with_target:,}\n"
            f"  task B w/o target_word_count: {self.n_task_b_without_target:,}\n"
            f"  over-length rows dropped: {self.n_overlength_dropped:,} "
            f"(by task: {self.overlength_by_task})\n"
            f"  eval-sibling rows dropped (cross-language/namespace): "
            f"{self.n_eval_siblings_dropped:,}\n"
            f"  50-token-overlap rows dropped: {self.n_leakage_rows_dropped:,}\n"
            f"  train rows: {self.n_train:,} -> {self.train_path}\n"
            f"  valid rows: {self.n_valid:,} -> {self.valid_path}\n"
        )


def _load_pyqs_word_counts() -> "dict[str, int]":
    """Read `question_id → word_count` from `pyqs` table."""
    with sqlite3.connect(DB_PATH) as con:
        df = pd.read_sql_query(
            "SELECT question_id, word_count FROM pyqs "
            "WHERE word_count IS NOT NULL AND word_count > 0",
            con,
        )
    return dict(zip(df["question_id"], df["word_count"].astype(int)))


def _qid_from_pair_id(pair_id: str) -> "str | None":
    """`mains:<uuid>:<lang>` → `<uuid>`; None for other formats."""
    parts = pair_id.split(":")
    return parts[1] if len(parts) >= 2 and parts[0] == "mains" else None


def _build_prompt(row: "pd.Series", target_word_count: "int | None") -> str:
    """User-turn content: `instruction\\n\\ninput` — byte-identical to the
    prompt `scripts/runners.py` wraps in the chat template at inference,
    plus the length hint for rows with a known answer-word target."""
    parts = [str(row["instruction"]).strip()]
    inp = str(row.get("input") or "").strip()
    if inp:
        parts.append(inp)
    if target_word_count:
        parts.append(f"Answer in approximately {int(target_word_count)} words.")
    return "\n\n".join(parts)


def _estimate_tokens(prompt: str, output: str) -> int:
    n_words = len(prompt.split()) + len(output.split())
    return int(n_words * TOKENS_PER_WORD_EST) + TEMPLATE_OVERHEAD_TOKENS


def _split_stratified(
    df: "pd.DataFrame", valid_frac: float, seed: int
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Per-task stratified split — guarantees minority tasks (B, E) get
    representation in both train and valid splits."""
    rng = random.Random(seed)
    trains, valids = [], []
    for task, grp in df.groupby("task", sort=True):
        idx = list(grp.index)
        rng.shuffle(idx)
        n_valid = max(1, int(len(idx) * valid_frac))
        valids.append(grp.loc[idx[:n_valid]])
        trains.append(grp.loc[idx[n_valid:]])
    train_df = pd.concat(trains).sample(frac=1.0, random_state=seed)
    valid_df = pd.concat(valids).sample(frac=1.0, random_state=seed)
    return train_df, valid_df


def build(
    out_dir: Path = OUT_DIR,
    include_hindi: bool = False,
    valid_frac: float = DEFAULT_VALID_FRAC,
    seed: int = DEFAULT_SEED,
) -> BuildReport:
    """Read v1 ft_corpus, filter, join word_count, split, write JSONL."""
    if not FT_CORPUS.exists():
        raise FileNotFoundError(
            f"v1 ft_corpus not found at {FT_CORPUS}; "
            f"run scripts/build_ft_corpus.py first or restore from snapshot."
        )
    df = pd.read_parquet(FT_CORPUS)
    report = BuildReport(n_in=len(df))

    if not include_hindi:
        n_before = len(df)
        df = df[df["language"] == "en"].reset_index(drop=True)
        report.n_filtered_hindi = n_before - len(df)

    word_counts = _load_pyqs_word_counts()

    # Compute target_word_count per row (Task B only; everything else = None).
    targets: list[int | None] = []
    for _, row in df.iterrows():
        if row["task"] != "B":
            targets.append(None)
            continue
        qid = _qid_from_pair_id(str(row["pair_id"]))
        wc = word_counts.get(qid) if qid else None
        targets.append(int(wc) if wc else None)
        if wc:
            report.n_task_b_with_target += 1
        else:
            report.n_task_b_without_target += 1
    df = df.copy()
    df["target_word_count"] = targets
    df["prompt_text"] = [
        _build_prompt(row, tgt)
        for (_, row), tgt in zip(df.iterrows(), targets)
    ]
    df["output_text"] = df["output"].astype(str).str.strip()

    # Over-length filter: trl truncates from the END at max_length, which
    # would cut the gold answer's tail + EOS — drop instead, with counts.
    est = [
        _estimate_tokens(p, o)
        for p, o in zip(df["prompt_text"], df["output_text"])
    ]
    over = pd.Series(est, index=df.index) > MAX_TRAIN_TOKENS
    report.n_overlength_dropped = int(over.sum())
    report.overlength_by_task = (
        df.loc[over, "task"].value_counts().to_dict() if over.any() else {}
    )
    df = df[~over].reset_index(drop=True)

    # Cross-language sibling exclusion: eval ids are `<ns>:<uuid>:<lang>`
    # and the FT corpus carries the SAME question under the other
    # language (eval `ai:X:hi` ↔ corpus `ai:X:en` / `prod_mcq:X:en`).
    # v1's ID check compared full ids (language suffix included) and
    # missed these — training on the English version of a question
    # evaluated in Hindi contaminates the Hindi stratum. Exclude by
    # base UUID, any namespace, any language.
    from .leakage import EVAL_SET, build_eval_index, check_corpus_text
    HOLDOUT = REPO / "data" / "eval_set_holdout.parquet"
    idx_paths = [EVAL_SET] + ([HOLDOUT] if HOLDOUT.exists() else [])
    eval_ids, hash_to_qid, gram_to_qids, gram_lengths = build_eval_index(idx_paths)
    eval_uuids = {parts[1] for qid in eval_ids
                  if len(parts := str(qid).split(":")) >= 2}
    row_uuid = df["pair_id"].astype(str).str.split(":").str[1]
    sibling = row_uuid.isin(eval_uuids)
    report.n_eval_siblings_dropped = int(sibling.sum())
    df = df[~sibling].reset_index(drop=True)
    def _gate(frame: "pd.DataFrame"):
        return check_corpus_text(
            paragraphs=(
                (str(row.pair_id), f"{row.prompt_text} {row.output_text}")
                for row in frame.itertuples()
            ),
            eval_ids=eval_ids,
            hash_to_qid=hash_to_qid,
            gram_to_qids=gram_to_qids,
            gram_lengths=gram_lengths,
        )

    # First pass flags rows sharing any 50-token window with eval gold.
    # Two real patterns observed: identical canned explanation passages
    # reused across different questions (genuine memorization risk) and
    # UPSC Statement-I/II option boilerplate (harmless format text).
    # Policy: drop every flagged row — conservative, costs ~1% of rows,
    # and leaves zero eval-overlapping text in the corpus either way.
    leak = _gate(df)
    if leak.ngram_hits or leak.hash_overlaps:
        flagged = {src for _, src in leak.ngram_hits}
        report.n_leakage_rows_dropped = int(
            df["pair_id"].astype(str).isin(flagged).sum())
        df = df[~df["pair_id"].astype(str).isin(flagged)].reset_index(drop=True)
        leak = _gate(df)   # re-verify after the drop
    if not leak.is_clean():
        raise RuntimeError(
            "SFT corpus failed the eval-leakage gate after row drops:\n"
            + leak.render()
        )
    print(f"[sft-corpus] leakage gate: CLEAN "
          f"({len(df):,} rows; {report.n_leakage_rows_dropped} flagged rows "
          f"dropped)")

    train_df, valid_df = _split_stratified(df, valid_frac, seed)
    report.n_train = len(train_df)
    report.n_valid = len(valid_df)

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    valid_path = out_dir / "valid.jsonl"
    _write_jsonl(train_df, train_path)
    _write_jsonl(valid_df, valid_path)
    report.train_path = str(train_path.relative_to(REPO))
    report.valid_path = str(valid_path.relative_to(REPO))
    return report


def _write_jsonl(df: "pd.DataFrame", path: Path) -> None:
    """Emit conversational prompt-completion JSONL (trl 1.5 renders these
    through the tokenizer's chat template with completion-only loss)."""
    cols = ["pair_id", "task", "target_word_count", "prompt_text", "output_text"]
    with path.open("w", encoding="utf-8") as fp:
        for _, row in df[cols].iterrows():
            obj = {
                "pair_id": str(row["pair_id"]),
                "task": str(row["task"]),
                "target_word_count": (int(row["target_word_count"])
                                      if pd.notna(row["target_word_count"])
                                      else None),
                "prompt": [{"role": "user", "content": str(row["prompt_text"])}],
                "completion": [{"role": "assistant", "content": str(row["output_text"])}],
            }
            fp.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main(argv: "list[str] | None" = None) -> int:
    p = argparse.ArgumentParser(description="Build the SFT corpus (JSONL).")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--include-hindi", action="store_true",
                   help="Keep Hindi rows (debug; production trains on EN only).")
    p.add_argument("--valid-frac", type=float, default=DEFAULT_VALID_FRAC)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = p.parse_args(argv)

    report = build(
        out_dir=args.out_dir,
        include_hindi=args.include_hindi,
        valid_frac=args.valid_frac,
        seed=args.seed,
    )
    print(report.render())
    return 0


if __name__ == "__main__":
    sys.exit(main())
