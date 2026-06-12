"""Stage 1.3 — build the multi-task FT corpus, excluding every eval-set ID.

Reads from the local SQLite snapshot at data/db_snapshot.sqlite. Does NOT touch
any remote DB. CI-asserts eval ∩ ft = ∅. Hard stop on violation.

Output (Parquet) — one row per training pair:
    pair_id     str   -- '<task>:<source_id>:<lang>'
    task        str
    language    str   -- 'en' | 'hi'
    source_db   str
    source_table str
    instruction str
    input       str   -- JSON-encoded structured input
    output      str   -- JSON-encoded for structured tasks; raw text for B
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from local_db import read_table

# Path A2: unified FT instruction format — same string used at train AND inference time,
# so the model sees the same prompt distribution it was trained on.
# These will be overridden with the production prompts when those are provided.
TASK_INSTRUCTIONS = {
    "A": (
        '[TASK=A] You are taking the UPSC Prelims (Indian Civil Services examination). '
        'Read the question and the four options. Return ONLY a JSON object: '
        '{"answer": "<A|B|C|D>", "explanation": "<step-by-step reasoning citing specific '
        'Article numbers / dates / scheme names; explain why the correct option is right '
        'and why each wrong option is wrong>"}'
    ),
    "B": (
        '[TASK=B] You are answering a UPSC Mains question. Write a complete answer at '
        'approximately the given word count, following UPSC structure (introduction, '
        'body with multi-dimensional analysis, conclusion). Cite specific Article numbers, '
        'dates, schemes, court cases where applicable. Return ONLY a JSON object: '
        '{"answer": "<full Mains answer text>"}'
    ),
    "C": (
        '[TASK=C] You are a UPSC Mains evaluator. Grade the student answer against the '
        'maximum marks. Return ONLY a JSON object: {"score": <float 0..max>, '
        '"strengths": [<2-4 specific strength bullets>], '
        '"improvements": {"intro": [<...>], "body": [<...>], "conclusion": [<...>]}}'
    ),
    "E": (
        '[TASK=E] You are creating UPSC study material from a news article. Produce a '
        'synthesis suitable for an aspirant. Return ONLY a JSON object: '
        '{"prelims_info": "<2-4 paragraphs of Prelims-relevant facts: scheme names, dates, '
        'key figures, definitions>", "mains_info": "<3-6 paragraphs of Mains-relevant '
        'analysis: causes, impacts, multi-dimensional framing, way forward>"}'
    ),
}

CUTOFF_DATE = "2026-04-30"


def _normalize_question_text(text: str) -> str:
    """Whitespace-normalize a question stem for content-equality comparison.

    We collapse all runs of whitespace to single spaces and strip — this
    catches the case where the same UPSC question appears in two source
    tables with minor formatting variation (extra newlines, trailing space).
    """
    import re as _re
    return _re.sub(r"\s+", " ", (text or "")).strip()


def _question_hash(text: str) -> str:
    import hashlib as _hl
    return _hl.sha256(_normalize_question_text(text).encode("utf-8")).hexdigest()[:16]


def assert_no_leakage(
    eval_ids: set[str],
    ft_ids: set[str],
    eval_question_hashes: set[str] | None = None,
    ft_question_hashes: set[str] | None = None,
) -> None:
    """Hard-fail the build on either form of leakage:

    1. ID-level — any eval pair_id appears in the FT corpus. Catches direct
       duplication when the same source row goes to both sets.
    2. Content-level — any eval question's normalized text hash appears in
       an FT pair's input. Catches the more subtle leak where the SAME
       UPSC question exists under multiple source tables (e.g., once in
       `prelims_pyq_questions`, once in `prod.mcqs`) with different
       pair_ids, so the ID check passes but the model would still be
       trained on what we test it on.
    """
    id_overlap = eval_ids & ft_ids
    if id_overlap:
        raise AssertionError(
            f"ID-LEVEL LEAKAGE: {len(id_overlap)} eval IDs in FT corpus: "
            f"{sorted(id_overlap)[:5]}"
        )
    if eval_question_hashes is not None and ft_question_hashes is not None:
        content_overlap = eval_question_hashes & ft_question_hashes
        # Drop the empty-string hash (defensive — '' should never reach here
        # but if a row had no question text it would all collapse to one hash).
        content_overlap.discard(_question_hash(""))
        if content_overlap:
            raise AssertionError(
                f"CONTENT-LEVEL LEAKAGE: {len(content_overlap)} eval question "
                f"stems appear verbatim in FT inputs. Same UPSC question may "
                f"exist in multiple source tables with different pair_ids; "
                f"normalize-and-hash check caught it. Examples: "
                f"{list(content_overlap)[:5]}"
            )


def _explanation_from_jsonb(expl) -> str:
    """prod.mcqs.explanation is a jsonb list of {content, ...} blocks; flatten.

    Defensively coerces non-string content (a handful of rows nest a dict where
    a string is expected) so a single malformed row does not abort the build.
    """
    if not expl:
        return ""
    if isinstance(expl, str):
        return expl
    if isinstance(expl, list):
        parts: list[str] = []
        for b in expl:
            if not isinstance(b, dict):
                continue
            c = b.get("content")
            if isinstance(c, str) and c:
                parts.append(c)
            elif c:
                parts.append(json.dumps(c, ensure_ascii=False))
        return "\n\n".join(parts)
    if isinstance(expl, dict):
        c = expl.get("content")
        return c if isinstance(c, str) else (json.dumps(c, ensure_ascii=False) if c else "")
    return ""


def build_A(exclude: set[str]) -> list[dict]:
    out: list[dict] = []

    df = read_table("prelims_pyq_questions").sort_values("question_id")
    df = df[df["question"].apply(lambda x: isinstance(x, dict))
            & df["options"].apply(lambda x: isinstance(x, dict))
            & df["correct_option"].notna()
            & ~df["is_dropped"].fillna(False).astype(bool)]
    for r in df.to_dict("records"):
        q = r["question"]; opts = r["options"]; expl = r["explanation"] or {}
        for lang in ("english", "hindi"):
            pid = f"pyq:{r['question_id']}:{lang[:2]}"
            if pid in exclude or not q.get(lang) or not opts.get(lang):
                continue
            out.append(dict(
                pair_id=pid, task="A", language=lang[:2],
                source_db="upscdev", source_table="prelims_pyq_questions",
                instruction=TASK_INSTRUCTIONS["A"],
                input=json.dumps({"question": q[lang], "options": opts[lang], "paper": "GS1"},
                                 ensure_ascii=False),
                output=json.dumps({"answer": r["correct_option"],
                                   "explanation": expl.get(lang) or ""}, ensure_ascii=False),
            ))

    df = read_table("upsc_prelims_ai_generated_que").sort_values("id")
    df = df[df["quality_pass_flag"].fillna(False).astype(bool)
            & df["answer"].notna()
            & df["options_english"].apply(lambda x: isinstance(x, dict))]
    for r in df.to_dict("records"):
        for lang in ("english", "hindi"):
            qcol = f"question_{lang}"; ocol = f"options_{lang}"
            q = r.get(qcol); opts = r.get(ocol)
            pid = f"ai:{r['id']}:{lang[:2]}"
            if pid in exclude or not q or not isinstance(opts, dict):
                continue
            out.append(dict(
                pair_id=pid, task="A", language=lang[:2],
                source_db="upscdev", source_table="upsc_prelims_ai_generated_que",
                instruction=TASK_INSTRUCTIONS["A"],
                input=json.dumps({"question": q, "options": opts, "paper": "GS1"},
                                 ensure_ascii=False),
                output=json.dumps({"answer": r["answer"],
                                   "explanation": r.get("explanation") or ""}, ensure_ascii=False),
            ))

    # CSAT (and additional GS1) MCQs from prod.mcqs, tagged via learning_items.paper.
    df = read_table("mcqs").sort_values("id")
    df = df[df["questionText"].notna()
            & df["options"].apply(lambda x: isinstance(x, list))
            & df["correctOptionIds"].apply(lambda x: isinstance(x, list) and len(x) > 0)
            & ~df["isMultiSelect"].fillna(False).astype(bool)
            & df["paper"].isin(["gs1", "csat"])]
    for r in df.to_dict("records"):
        pid = f"prod_mcq:{r['id']}:en"
        if pid in exclude:
            continue
        expl_text = _explanation_from_jsonb(r["explanation"])
        if not expl_text:
            continue  # mcqs without explanations are useless for FT
        opts_map = {o["id"].upper(): o["text"] for o in r["options"] if isinstance(o, dict)}
        out.append(dict(
            pair_id=pid, task="A", language="en",
            source_db="prod-db", source_table="mcqs",
            instruction=TASK_INSTRUCTIONS["A"],
            input=json.dumps({"question": r["questionText"], "options": opts_map,
                              "paper": r["paper"].upper()}, ensure_ascii=False),
            output=json.dumps({"answer": r["correctOptionIds"][0].upper(),
                               "explanation": expl_text}, ensure_ascii=False),
        ))

    return out


def build_B(exclude: set[str]) -> list[dict]:
    out: list[dict] = []
    df = read_table("pyqs").sort_values("question_id")
    df = df[df["question"].apply(lambda x: isinstance(x, dict))
            & df["model_answer"].apply(lambda x: isinstance(x, dict))]
    for r in df.to_dict("records"):
        q = r["question"]; ma = r["model_answer"]
        for lang in ("english", "hindi"):
            pid = f"mains:{r['question_id']}:{lang[:2]}"
            if pid in exclude or not q.get(lang) or not ma.get(lang):
                continue
            out.append(dict(
                pair_id=pid, task="B", language=lang[:2],
                source_db="upscdev", source_table="pyqs",
                instruction=TASK_INSTRUCTIONS["B"],
                input=json.dumps({
                    "question": q[lang], "paper": r["paper"], "subject": r["subject"],
                    "word_count": int(r["word_count"] or 250),
                    "max_score": float(r["max_score"] or 15),
                }, ensure_ascii=False),
                output=json.dumps({"answer": ma[lang]}, ensure_ascii=False),
            ))
    return out


def build_C(exclude: set[str]) -> list[dict]:
    out: list[dict] = []
    df = read_table("evaluation_questions").sort_values(["question_id", "evaluation_id"])
    df = df[df["answer_text"].notna()
            & df["strengths"].apply(lambda x: x is not None)
            & df["score"].notna()
            & df["max_score"].notna()
            & df["question_text"].notna()]
    for r in df.to_dict("records"):
        pid = f"eval:{r['question_id']}:{r['evaluation_id']}"
        if pid in exclude:
            continue
        out.append(dict(
            pair_id=pid, task="C", language="en",
            source_db="upscdev", source_table="evaluation_questions",
            instruction=TASK_INSTRUCTIONS["C"],
            input=json.dumps({
                "question_text": r["question_text"], "answer_text": r["answer_text"],
                "max_score": float(r["max_score"]),
            }, ensure_ascii=False),
            output=json.dumps({
                "score": float(r["score"]),
                "strengths": r["strengths"],
                "improvements": r["improvements"],
            }, ensure_ascii=False, default=str),
        ))
    return out


def build_E(exclude: set[str]) -> list[dict]:
    out: list[dict] = []
    df = read_table("news_articles").sort_values("id")
    df = df[df["date"].notna() & df["text"].notna() & df["mainsInfo"].notna()]
    df = df[df["date"] < CUTOFF_DATE]
    for r in df.to_dict("records"):
        pid = f"news:{r['id']}"
        if pid in exclude:
            continue
        out.append(dict(
            pair_id=pid, task="E", language="en",
            source_db="prod-db", source_table="news_articles",
            instruction=TASK_INSTRUCTIONS["E"],
            input=json.dumps({
                "date": str(r["date"]), "title": r["title"], "article": r["text"],
            }, ensure_ascii=False),
            output=json.dumps({
                "prelims_info": r["prelimsInfo"] or "",
                "mains_info": r["mainsInfo"] or "",
            }, ensure_ascii=False),
        ))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=Path, default=Path("data/eval_set.parquet"))
    ap.add_argument("--out", type=Path, default=Path("data/ft_corpus.parquet"))
    args = ap.parse_args()

    eval_df = pd.read_parquet(args.eval)
    eval_ids = set(eval_df["question_id"].tolist())

    # --- Compute the set of eval question-text hashes. We extract the question
    # text from gold_payload (JSON-serialized full record). This is the key
    # used by the content-level leakage check below.
    def _eval_question_text(payload: str) -> str:
        try:
            d = json.loads(payload)
            return (d.get("question") or d.get("question_text") or "")
        except Exception:
            return ""

    eval_question_hashes = {
        _question_hash(_eval_question_text(p))
        for p in eval_df["gold_payload"]
    }
    eval_question_hashes.discard(_question_hash(""))   # defensive
    print(f"[load] eval-set IDs: {len(eval_ids):,}, "
          f"unique question hashes: {len(eval_question_hashes):,}")

    rows: list[dict] = []
    for task, builder in (("A", build_A), ("B", build_B), ("C", build_C), ("E", build_E)):
        n0 = len(rows)
        rows.extend(builder(eval_ids))
        print(f"  task {task}: +{len(rows) - n0:,} (pre-content-filter)")

    # --- Content-level dedup: drop any FT pair whose question text matches an
    # eval question's normalized hash. This catches the case where the same
    # UPSC question appears under multiple source tables with different
    # pair_ids — pair-id-level check would miss it.
    def _ft_question_text(input_str: str) -> str:
        try:
            d = json.loads(input_str)
            return (d.get("question") or d.get("question_text") or "")
        except Exception:
            return ""

    pre_dedup = len(rows)
    filtered: list[dict] = []
    dropped_by_task: dict[str, int] = {}
    for r in rows:
        qh = _question_hash(_ft_question_text(r["input"]))
        if qh in eval_question_hashes:
            dropped_by_task[r["task"]] = dropped_by_task.get(r["task"], 0) + 1
            continue
        filtered.append(r)
    rows = filtered
    if dropped_by_task:
        n_dropped = pre_dedup - len(rows)
        print(f"\n[content-dedup] dropped {n_dropped:,} FT pairs whose question "
              f"text matched an eval question (mode: cross-source duplication):")
        for t, n in sorted(dropped_by_task.items()):
            print(f"    task {t}: -{n:,}")

    ft_ids = {r["pair_id"] for r in rows}
    ft_question_hashes = {
        _question_hash(_ft_question_text(r["input"])) for r in rows
    }
    ft_question_hashes.discard(_question_hash(""))

    # Both checks run; either failure aborts.
    assert_no_leakage(eval_ids, ft_ids, eval_question_hashes, ft_question_hashes)

    df = pd.DataFrame(rows).sort_values(["task", "pair_id"]).reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False, compression="snappy")

    sha = hashlib.sha256(args.out.read_bytes()).hexdigest()
    args.out.with_suffix(".sha256").write_text(sha + "\n")

    print(f"\n[OK] wrote {len(df):,} pairs → {args.out}")
    print(f"     SHA-256: {sha}")
    print(f"     by task: {df.groupby('task').size().to_dict()}")
    print(f"     leakage check: PASS (ID-level + content-level both empty)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
