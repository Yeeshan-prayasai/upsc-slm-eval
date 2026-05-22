"""Stage 1.2 — deterministically sample 2,000 stratified eval items across tasks A/B/C/E.

Reads from the local SQLite snapshot at data/prayas_local.sqlite. Does NOT touch
any remote DB. Run `make snapshot` first if the snapshot file is missing.

Output schema (Parquet):
    question_id   str   -- source PK as text
    task          str   -- 'A' | 'B' | 'C' | 'E'
    source_db     str
    source_table  str
    paper         str   -- nullable
    subject       str
    language      str   -- 'en' | 'hi'
    gold_payload  str   -- JSON-serialized full record for scoring
    stratum_key   str
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from local_db import read_table

TARGETS = {"A": 800, "B": 400, "C": 500, "E": 300}
CUTOFF_DATE = "2026-04-30"


def pull_task_A() -> list[dict]:
    rows: list[dict] = []

    df = read_table("prelims_pyq_questions").sort_values("question_id")
    df = df[df["question"].apply(lambda x: isinstance(x, dict))
            & df["options"].apply(lambda x: isinstance(x, dict))
            & df["correct_option"].notna()
            & ~df["is_dropped"].fillna(False).astype(bool)]
    for r in df.to_dict("records"):
        q = r["question"]; opts = r["options"]; expl = r["explanation"] or {}
        for lang in ("english", "hindi"):
            if not q.get(lang) or not opts.get(lang):
                continue
            rows.append(dict(
                question_id=f"pyq:{r['question_id']}:{lang[:2]}",
                source_db="upscdev", source_table="prelims_pyq_questions",
                paper=r["paper"] or "UNTAGGED", subject=r["subject"] or "UNTAGGED",
                language=lang[:2],
                gold_payload=dict(
                    question=q[lang], options=opts[lang],
                    correct_option=r["correct_option"], explanation=expl.get(lang) or "",
                    silly_mistake_prone=bool(r.get("silly_mistake_prone")),
                    year=r.get("year"),
                ),
                silly=bool(r.get("silly_mistake_prone")),
            ))

    df = read_table("upsc_prelims_ai_generated_que").sort_values("id")
    df = df[df["quality_pass_flag"].fillna(False).astype(bool)
            & df["answer"].notna()
            & df["options_english"].apply(lambda x: isinstance(x, dict))]
    for r in df.to_dict("records"):
        for lang in ("english", "hindi"):
            qcol = f"question_{lang}"; ocol = f"options_{lang}"
            q = r.get(qcol); opts = r.get(ocol)
            if not q or not isinstance(opts, dict):
                continue
            rows.append(dict(
                question_id=f"ai:{r['id']}:{lang[:2]}",
                source_db="upscdev", source_table="upsc_prelims_ai_generated_que",
                paper="GS1", subject=r["subject"] or "UNTAGGED",
                language=lang[:2],
                gold_payload=dict(
                    question=q, options=opts, correct_option=r["answer"],
                    explanation=r.get("explanation") or "",
                    silly_mistake_prone=bool(r.get("prone_to_silly_mistakes")),
                    topic=r.get("topic"), difficulty=r.get("difficulty"),
                ),
                silly=bool(r.get("prone_to_silly_mistakes")),
            ))

    df = read_table("mcqs").sort_values("id")
    df = df[df["questionText"].notna()
            & df["options"].apply(lambda x: isinstance(x, list))
            & df["correctOptionIds"].apply(lambda x: isinstance(x, list) and len(x) > 0)
            & ~df["isMultiSelect"].fillna(False).astype(bool)
            & df["paper"].isin(["gs1", "csat"])]
    for r in df.to_dict("records"):
        correct = r["correctOptionIds"][0]
        paper = r["paper"].upper()
        rows.append(dict(
            question_id=f"prod_mcq:{r['id']}:en",
            source_db="prod-prayas-db", source_table="mcqs",
            paper=paper, subject="UNTAGGED", language="en",
            gold_payload=dict(
                question=r["questionText"], options=r["options"],
                correct_option=correct, explanation=r.get("explanation") or "",
                silly_mistake_prone=bool(r.get("silly_mistake_prone")),
                paper=paper,
            ),
            silly=bool(r.get("silly_mistake_prone")),
        ))

    return rows


def pull_task_B() -> list[dict]:
    out: list[dict] = []
    df = read_table("pyqs").sort_values("question_id")
    df = df[df["question"].apply(lambda x: isinstance(x, dict))
            & df["model_answer"].apply(lambda x: isinstance(x, dict))]
    for r in df.to_dict("records"):
        q = r["question"]; ma = r["model_answer"]
        for lang in ("english", "hindi"):
            if not q.get(lang) or not ma.get(lang):
                continue
            out.append(dict(
                question_id=f"mains:{r['question_id']}:{lang[:2]}",
                source_db="upscdev", source_table="pyqs",
                paper=r["paper"] or "UNTAGGED", subject=r["subject"] or "UNTAGGED",
                language=lang[:2],
                gold_payload=dict(
                    question=q[lang], model_answer=ma[lang],
                    word_count=int(r["word_count"] or 0),
                    max_score=float(r["max_score"] or 0),
                    section=r.get("section"), year=r.get("year"),
                ),
                wc_bin=_wc_bin(int(r["word_count"] or 0)),
            ))
    return out


def pull_task_C() -> list[dict]:
    out: list[dict] = []
    df = read_table("evaluation_questions").sort_values(["question_id", "evaluation_id"])
    df = df[df["answer_text"].notna()
            & df["strengths"].apply(lambda x: x is not None)
            & df["score"].notna()
            & df["max_score"].notna()
            & df["question_text"].notna()]
    for r in df.to_dict("records"):
        score = float(r["score"]); maxs = float(r["max_score"] or 1)
        band = "low" if score / maxs <= 0.30 else "mid" if score / maxs <= 0.60 else "high"
        out.append(dict(
            question_id=f"eval:{r['question_id']}:{r['evaluation_id']}",
            source_db="upscdev", source_table="evaluation_questions",
            paper="UNTAGGED", subject=r["subject"] or "UNTAGGED", language="en",
            gold_payload=dict(
                question_text=r["question_text"], answer_text=r["answer_text"],
                score=score, max_score=maxs, word_count=r.get("word_count"),
                strengths=r["strengths"], improvements=r["improvements"],
                model_answer=r.get("model_answer"),
            ),
            band=band,
        ))
    return out


def pull_task_E() -> list[dict]:
    out: list[dict] = []
    df = read_table("news_articles").sort_values("id")
    df = df[df["date"].notna() & df["text"].notna() & df["mainsInfo"].notna()]
    df = df[df["date"] <= CUTOFF_DATE]
    for r in df.to_dict("records"):
        theme = str(r.get("newsThemeId") or "UNTAGGED")
        month = r["date"][:7] if isinstance(r["date"], str) else r["date"].strftime("%Y-%m")
        out.append(dict(
            question_id=f"news:{r['id']}",
            source_db="prod-prayas-db", source_table="news_articles",
            paper="UNTAGGED", subject="UNTAGGED", language="en",
            gold_payload=dict(
                date=str(r["date"]),
                title=r["title"], source_text=r["text"],
                prelims_info=r["prelimsInfo"] or "", mains_info=r["mainsInfo"] or "",
                source=r.get("source"),
            ),
            theme=theme, month=month,
        ))
    return out


def _wc_bin(wc: int) -> str:
    if wc <= 175: return "150w"
    if wc <= 400: return "250w"
    return "essay"


def _stratify_sample(rows: list[dict], n: int, key_fn, rng: random.Random) -> list[dict]:
    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_stratum[key_fn(r)].append(r)
    strata = sorted(by_stratum.keys())
    per = max(1, n // max(1, len(strata)))
    picked: list[dict] = []
    picked_ids: set[int] = set()
    for k in strata:
        bucket = by_stratum[k]
        rng.shuffle(bucket)
        for r in bucket[:per]:
            picked.append(r); picked_ids.add(id(r))
    if len(picked) > n:
        rng.shuffle(picked)
        picked = picked[:n]
    elif len(picked) < n:
        remaining = [r for r in rows if id(r) not in picked_ids]
        rng.shuffle(remaining)
        picked.extend(remaining[: n - len(picked)])
    for r in picked:
        r["stratum_key"] = key_fn(r)
    return picked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260514)
    ap.add_argument("--out", type=Path, default=Path("data/eval_set.parquet"))
    args = ap.parse_args()

    rng = random.Random(args.seed)

    print(f"[{TARGETS['A']:>4d}/A] pulling Prelims MCQ pools …")
    a_all = pull_task_A()
    print(f"        candidates: {len(a_all):,}")
    a_sample = _stratify_sample(a_all, TARGETS["A"],
        lambda r: f"{r['paper']}|{r['subject']}|silly={int(r['silly'])}|{r['language']}", rng)

    print(f"[{TARGETS['B']:>4d}/B] pulling Mains generation pool …")
    b_all = pull_task_B()
    print(f"        candidates: {len(b_all):,}")
    b_sample = _stratify_sample(b_all, TARGETS["B"],
        lambda r: f"{r['paper']}|{r['subject']}|{r['wc_bin']}|{r['language']}", rng)

    print(f"[{TARGETS['C']:>4d}/C] pulling rubric-graded answers …")
    c_all = pull_task_C()
    print(f"        candidates: {len(c_all):,}")
    c_sample = _stratify_sample(c_all, TARGETS["C"],
        lambda r: f"{r['subject']}|{r['band']}", rng)

    print(f"[{TARGETS['E']:>4d}/E] pulling current-affairs articles (date ≤ {CUTOFF_DATE}) …")
    e_all = pull_task_E()
    print(f"        candidates: {len(e_all):,}")
    e_sample = _stratify_sample(e_all, TARGETS["E"],
        lambda r: f"{r['theme']}|{r['month']}", rng)

    final = []
    for r in a_sample: final.append(("A", r))
    for r in b_sample: final.append(("B", r))
    for r in c_sample: final.append(("C", r))
    for r in e_sample: final.append(("E", r))

    df = pd.DataFrame([
        dict(question_id=r["question_id"], task=t,
             source_db=r["source_db"], source_table=r["source_table"],
             paper=r["paper"], subject=r["subject"], language=r["language"],
             gold_payload=json.dumps(r["gold_payload"], default=str, ensure_ascii=False),
             stratum_key=r["stratum_key"])
        for t, r in final
    ])
    df = df.sort_values(["task", "question_id"]).reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False, compression="snappy")

    sha = hashlib.sha256(args.out.read_bytes()).hexdigest()
    args.out.with_suffix(".sha256").write_text(sha + "\n")

    counts = df.groupby("task").size().to_dict()
    print(f"\n[OK] wrote {len(df):,} rows → {args.out}")
    print(f"     SHA-256: {sha}")
    print(f"     by task: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
