"""Tokenize the cleaned/deduped CPT corpus and pack into fixed-length
sequences for the trainer.

Two outputs per run — one per base model vocab — because Gemma-4-E4B
and Qwen-3.5-4B use different tokenizers:

    data/cpt_corpus_gemma.parquet   ← tokenized with google/gemma-4-E4B-it
    data/cpt_corpus_qwen.parquet    ← tokenized with Qwen/Qwen3.5-4B

Packing follows the standard CPT recipe (Llama 3 / Gemma 3 / Qwen 3
tech reports), with one model-specific addition:
- Each document becomes `[BOS] + ids + [EOS]` when the tokenizer has a
  BOS token (Gemma — its pretraining format is BOS-prefixed documents
  and the model is BOS-sensitive), or `ids + [EOS]` when it doesn't
  (Qwen). Pre-templated documents that already start with BOS are not
  double-prefixed.
- Slices into fixed-length `seq_len` sequences (default 4096); the
  trailing partial sequence is dropped (no padding).
- No attention-mask resets at document boundaries — naive concatenation
  (GPT-3/Pythia-style). Documented in v2-methodology §4.4.

**Mix weighting** (`training/configs/data_mix_cpt.yaml`) is enforced
here — per-source `repeat` factors (epochs over that source) and
`cap_tokens` ceilings, applied at document granularity. The expanded
document list is shuffled with a fixed seed before packing so packs
mix sources. Without this stage the batch share of every source would
equal its accidental disk volume.

Source formats:
    .md / .txt → one document per file; `.txt` files containing the
                 `<<<END-RECORD>>>` delimiter (local-DB extracts) are
                 split into one document per record
    .jsonl     → one document per row's `text` field (replay buffer),
                 or chat-rendered `{prompt, completion}` rows
                 (instruction data — rendered with the model's own
                 chat template so CPT preserves the -it formatting)

Output schema: {input_ids: list[int32]}. Per-source token counts go to
`data/cpt_corpus_manifest.json`.

CLI:
    python -m training.data.tokenize_pack --tokenizer gemma
    python -m training.data.tokenize_pack --tokenizer both
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .acquire._base import RepoPaths
from .clean import END_RECORD_DELIM

REPO = RepoPaths.root()
CPT_CLEAN_DEDUP = REPO / "data" / "cpt_clean_dedup"
DEFAULT_MIX_CONFIG = REPO / "training" / "configs" / "data_mix_cpt.yaml"

TOKENIZER_NAMES = {
    "gemma": "google/gemma-4-E4B-it",
    "qwen": "Qwen/Qwen3.5-4B",
}

DEFAULT_SEQ_LEN = 4096
SHUFFLE_SEED = 20260514


@dataclass
class PackStats:
    files_seen: int = 0
    docs_tokenized: int = 0
    tokens_total: int = 0
    sequences_emitted: int = 0
    tokens_dropped_trailing: int = 0
    bytes_written: int = 0
    per_source_tokens: dict[str, int] = field(default_factory=dict)
    per_source_unique_tokens: dict[str, int] = field(default_factory=dict)
    per_source_docs_capped: dict[str, int] = field(default_factory=dict)


# ----------- Mix config -----------

@dataclass(frozen=True)
class SourceMix:
    repeat: float = 1.0          # epochs over this source (fractional OK)
    cap_tokens: int | None = None  # hard ceiling on UNIQUE tokens consumed


def load_mix_config(path: Path) -> dict[str, SourceMix]:
    """Read per-source mix weights. Fails loud if the file is missing —
    an unweighted corpus is exactly the silent failure this stage
    exists to prevent."""
    import yaml

    if not path.exists():
        raise FileNotFoundError(
            f"Mix config not found: {path}. The CPT corpus must be "
            f"weighted (methodology §4.5) — author the file or pass "
            f"--mix-config explicitly."
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    sources = raw.get("sources") or {}
    out: dict[str, SourceMix] = {}
    for name, spec in sources.items():
        spec = spec or {}
        sm = SourceMix(
            repeat=float(spec.get("repeat", 1.0)),
            cap_tokens=int(spec["cap_tokens"]) if spec.get("cap_tokens") else None,
        )
        # Cap + repeat>1 would let cap-skipped docs slip back in on
        # later repeats; caps are for downsampling, repeats for
        # upsampling — pick one per source.
        if sm.cap_tokens is not None and sm.repeat > 1.0:
            raise ValueError(
                f"mix source '{name}': cap_tokens and repeat>1 are "
                f"mutually exclusive (cap={sm.cap_tokens}, repeat={sm.repeat})"
            )
        out[name] = sm
    return out


def _frac_keep(path: Path, rep_idx: int, frac: float) -> bool:
    """Deterministic per-(doc, repeat-index) inclusion for the
    fractional part of a repeat factor — stable across runs/machines
    (no RNG state dependence)."""
    h = hashlib.sha256(f"{path}#{rep_idx}".encode()).digest()
    return (int.from_bytes(h[:4], "big") / 0xFFFFFFFF) < frac


# ----------- Document discovery + reading -----------

def _iter_documents(corpus_root: Path) -> list[tuple[str, Path]]:
    """Discover every (source, file) pair under the cleaned corpus root.
    `source` = the top-level dir name directly under `corpus_root`."""
    pairs: list[tuple[str, Path]] = []
    for pattern in ("*.md", "*.txt", "*.jsonl"):
        for path in sorted(corpus_root.rglob(pattern)):
            rel = path.relative_to(corpus_root)
            source = rel.parts[0] if rel.parts else "unknown"
            pairs.append((source, path))
    return pairs


def _read_documents(path: Path) -> list[str | dict]:
    """One file → list of documents.

    `.txt`/`.md`: one document per file, EXCEPT `.txt` files containing
    the `<<<END-RECORD>>>` delimiter (local-DB extracts) which split
    into one document per record — each record then gets its own
    EOS boundary instead of thousands of rows packing as one "document".
    `.jsonl`: `{"text": ...}` rows → str docs;
              `{"prompt": ..., "completion": ...}` rows (instruction
              data) → dict docs, chat-rendered at tokenization.
    """
    if path.suffix == ".jsonl":
        docs: list[str | dict] = []
        with path.open(encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("prompt") and d.get("completion"):
                    docs.append({"prompt": d["prompt"], "completion": d["completion"]})
                    continue
                t = (d.get("text") or "").strip()
                if t:
                    docs.append(t)
        return docs
    text = path.read_text(encoding="utf-8", errors="replace")
    if END_RECORD_DELIM in text:
        return [rec.strip() for rec in text.split(END_RECORD_DELIM) if rec.strip()]
    return [text.strip()] if text.strip() else []


def _encode_doc(tok, doc: "str | dict", bos_id: int | None) -> list[int]:
    """Document → token ids with correct special-token framing.

    Plain text: `[BOS] + encode(text)` when the tokenizer has BOS
    (Gemma), else `encode(text)` (Qwen). EOS is appended by the pack
    loop as the document separator.

    Instruction rows: rendered through the tokenizer's own chat
    template (this is the point — CPT on an -it model should see its
    chat format); the template already emits BOS where the model
    expects it, so no extra prefix. A defensive check strips nothing
    but avoids double-BOS.
    """
    if isinstance(doc, dict):
        # return_dict=False: transformers 5.x returns a BatchEncoding by
        # default when tokenize=True; buf.extend() on a dict would ingest
        # the string "input_ids" as if it were tokens.
        ids = tok.apply_chat_template(
            [{"role": "user", "content": doc["prompt"]},
             {"role": "assistant", "content": doc["completion"]}],
            tokenize=True, add_generation_prompt=False, return_dict=False,
        )
        if isinstance(ids, dict):
            ids = ids["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        if bos_id is not None and len(ids) >= 2 and ids[0] == bos_id and ids[1] == bos_id:
            ids = ids[1:]
        return ids
    ids = tok.encode(doc, add_special_tokens=False)
    if not ids:
        return []
    if bos_id is not None and ids[0] != bos_id:
        ids = [bos_id] + ids
    return ids


def pack_token_streams(
    token_streams: "list[list[int]]",
    eos_id: int,
    seq_len: int,
    bos_id: int | None = None,
) -> "tuple[list[list[int]], int]":
    """Pure pack-loop helper (no tokenizer, no I/O) — exposed for unit
    tests of the boundary + dropped-trailing-tail invariants.

    Concatenates `[bos_id] + doc_ids + [eos_id]` (BOS only when
    provided and not already present) for every doc and slices the
    stream into fixed-`seq_len` sequences. The trailing partial chunk
    is DROPPED. Returns `(sequences, n_trailing_dropped)`.
    """
    buf: list[int] = []
    out: list[list[int]] = []
    for ids in token_streams:
        if not ids:
            continue
        if bos_id is not None and ids[0] != bos_id:
            buf.append(bos_id)
        buf.extend(ids)
        buf.append(eos_id)
        while len(buf) >= seq_len:
            out.append(buf[:seq_len])
            del buf[:seq_len]
    return out, len(buf)


def tokenize_and_pack(
    tokenizer_key: str,
    corpus_root: Path,
    out_path: Path,
    mix: dict[str, SourceMix],
    seq_len: int = DEFAULT_SEQ_LEN,
    write_batch_rows: int = 1000,
) -> PackStats:
    """Tokenize every document (with per-source mix weighting) and pack
    into `seq_len`-token sequences. Writes Parquet at `out_path`."""
    from transformers import AutoTokenizer

    tok_name = TOKENIZER_NAMES[tokenizer_key]
    print(f"Loading tokenizer {tok_name} ...")
    tok = AutoTokenizer.from_pretrained(tok_name)
    eos_id = tok.eos_token_id
    bos_id = tok.bos_token_id   # None for Qwen — handled per-doc
    if eos_id is None:
        raise RuntimeError(f"Tokenizer {tok_name} has no EOS — can't pack.")

    pairs = _iter_documents(corpus_root)
    if not pairs:
        raise FileNotFoundError(f"No documents under {corpus_root}")

    # ---- Mix expansion: (source, path, rep_idx) work items ----
    unknown = sorted({s for s, _ in pairs} - set(mix))
    if unknown:
        print(f"  WARNING: sources without mix entry (default repeat=1.0): {unknown}")
    work: list[tuple[str, Path, int]] = []
    for source, path in pairs:
        sm = mix.get(source, SourceMix())
        n_full = int(sm.repeat)
        frac = sm.repeat - n_full
        for rep_idx in range(n_full):
            work.append((source, path, rep_idx))
        if frac > 0 and _frac_keep(path, n_full, frac):
            work.append((source, path, n_full))

    # Document-level shuffle so packs mix sources (deterministic seed).
    rng = random.Random(SHUFFLE_SEED)
    rng.shuffle(work)
    print(f"Found {len(pairs)} files → {len(work)} weighted work items.")

    stats = PackStats(files_seen=len(pairs))
    schema = pa.schema([("input_ids", pa.list_(pa.int32()))])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(out_path, schema=schema, compression="zstd")

    buf: list[int] = []
    row_batch: list[list[int]] = []
    # Token cache: NCERT at repeat=4 shouldn't re-tokenize 4×. Keyed by
    # path; only populated for repeated sources to bound memory.
    repeat_paths = {p for s, p, _ in work if mix.get(s, SourceMix()).repeat > 1}
    tok_cache: dict[Path, list[list[int]]] = {}
    capped_unique: dict[str, int] = {}   # unique tokens consumed per capped source

    def _flush_batch() -> None:
        if not row_batch:
            return
        tbl = pa.table({"input_ids": row_batch}, schema=schema)
        writer.write_table(tbl)
        row_batch.clear()

    def _drain_to_sequences() -> None:
        while len(buf) >= seq_len:
            row_batch.append(buf[:seq_len])
            del buf[:seq_len]
            stats.sequences_emitted += 1
            if len(row_batch) >= write_batch_rows:
                _flush_batch()

    for i, (source, path, rep_idx) in enumerate(work):
        if i % 200 == 0:
            print(f"  [{i:>6}/{len(work)}] {source}/{path.name}  "
                  f"(tokens so far: {stats.tokens_total:,})")
        sm = mix.get(source, SourceMix())

        if path in tok_cache:
            doc_ids_list = tok_cache[path]
        else:
            doc_ids_list = [
                _encode_doc(tok, doc, bos_id)
                for doc in _read_documents(path)
            ]
            doc_ids_list = [ids for ids in doc_ids_list if ids]
            if path in repeat_paths:
                tok_cache[path] = doc_ids_list

        for ids in doc_ids_list:
            doc_tokens = len(ids) + 1   # +1 for EOS separator
            if sm.cap_tokens is not None and rep_idx == 0:
                used = capped_unique.get(source, 0)
                if used >= sm.cap_tokens:
                    stats.per_source_docs_capped[source] = (
                        stats.per_source_docs_capped.get(source, 0) + 1)
                    continue
                capped_unique[source] = used + doc_tokens
            stats.docs_tokenized += 1
            stats.tokens_total += doc_tokens
            stats.per_source_tokens[source] = (
                stats.per_source_tokens.get(source, 0) + doc_tokens)
            if rep_idx == 0:
                stats.per_source_unique_tokens[source] = (
                    stats.per_source_unique_tokens.get(source, 0) + doc_tokens)
            buf.extend(ids)
            buf.append(eos_id)
            _drain_to_sequences()

    stats.tokens_dropped_trailing = len(buf)
    buf.clear()
    _flush_batch()
    writer.close()

    stats.bytes_written = out_path.stat().st_size
    return stats


def write_manifest(out_template: Path, stats_by_tok: dict[str, PackStats]) -> Path:
    """Combined manifest covering each tokenizer run."""
    manifest_path = REPO / "data" / "cpt_corpus_manifest.json"
    payload: dict = {"runs": {}}
    for key, s in stats_by_tok.items():
        payload["runs"][key] = {
            "tokenizer": TOKENIZER_NAMES[key],
            "output_path": str(out_template).replace("<TOK>", key),
            "files_seen": s.files_seen,
            "docs_tokenized": s.docs_tokenized,
            "tokens_total": s.tokens_total,
            "sequences_emitted": s.sequences_emitted,
            "tokens_dropped_trailing": s.tokens_dropped_trailing,
            "bytes_written": s.bytes_written,
            "per_source_tokens": s.per_source_tokens,
            "per_source_unique_tokens": s.per_source_unique_tokens,
            "per_source_docs_capped": s.per_source_docs_capped,
        }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Tokenize + pack cleaned CPT corpus.")
    p.add_argument("--tokenizer", choices=("gemma", "qwen", "both"), default="both")
    p.add_argument("--corpus-root", default=str(CPT_CLEAN_DEDUP),
                   help="Cleaned/deduped corpus root (default data/cpt_clean_dedup)")
    p.add_argument("--mix-config", default=str(DEFAULT_MIX_CONFIG),
                   help="Per-source repeat/cap weights "
                        "(default training/configs/data_mix_cpt.yaml)")
    p.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    args = p.parse_args(argv)

    corpus_root = Path(args.corpus_root)
    if not corpus_root.exists():
        print(f"ERROR: corpus root not found at {corpus_root}", file=sys.stderr)
        print("Run the corpus clean+dedupe pass first (`make build-cpt-corpus`).", file=sys.stderr)
        return 1

    mix = load_mix_config(Path(args.mix_config))
    print(f"Mix config: {len(mix)} sources weighted "
          f"({Path(args.mix_config).name})")

    keys = ("gemma", "qwen") if args.tokenizer == "both" else (args.tokenizer,)
    stats: dict[str, PackStats] = {}
    for key in keys:
        out_path = REPO / "data" / f"cpt_corpus_{key}.parquet"
        print(f"\n=== Tokenize+pack for {key} → {out_path.relative_to(REPO)} ===")
        s = tokenize_and_pack(key, corpus_root, out_path, mix=mix,
                              seq_len=args.seq_len)
        stats[key] = s
        print(f"  docs={s.docs_tokenized:,}  tokens={s.tokens_total:,}  "
              f"sequences={s.sequences_emitted:,}  bytes={s.bytes_written:,}")
        print(f"  trailing tokens dropped (partial sequence): {s.tokens_dropped_trailing:,}")
        print(f"  per-source tokens (weighted): {s.per_source_tokens}")

        # Replay must be present — its absence was a silent failure mode
        # (the .jsonl replay never reached the corpus at all).
        replay = sum(s.per_source_tokens.get(k, 0)
                     for k in ("slimpajama", "wikipedia"))
        if replay == 0:
            print("ERROR: replay buffer (slimpajama/wikipedia) contributed 0 "
                  "tokens — the anti-forgetting design requires it. Check "
                  "that the .jsonl sources survived the clean pass.",
                  file=sys.stderr)
            return 2
        share = replay / max(1, s.tokens_total)
        print(f"  replay share: {share:.1%}")

    if len(stats) > 1:
        gemma_docs = stats["gemma"].docs_tokenized
        qwen_docs = stats["qwen"].docs_tokenized
        if gemma_docs != qwen_docs:
            print(f"WARNING: doc counts differ — gemma={gemma_docs} qwen={qwen_docs}. "
                  f"Input corpus should be identical across tokenizers.",
                  file=sys.stderr)

    manifest_out = write_manifest(REPO / "data" / "cpt_corpus_<TOK>.parquet", stats)
    print(f"\nManifest: {manifest_out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
