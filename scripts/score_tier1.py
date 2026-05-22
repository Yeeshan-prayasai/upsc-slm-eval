"""Stage 6.1 — Tier-1 deterministic per-row scoring.

Reads `results/predictions.parquet` (Phase-5 output), emits
`results/scores_tier1.parquet` with all per-row metrics for tasks A/B/C/E.

Aggregation-level metrics (Brier Skill Score, ECE, QWK, Spearman/Pearson,
χ² position bias, confusion matrices, bootstrap CIs) live in
`scripts/aggregate.py` — they need the full result set, not a single row.

Deferred for v1 (Path C, git-only deps, or heavy compute):
- BLEURT-20 (bleurt-pytorch is git-only)
- SummaC-ZS / AlignScore / FactScore (Task E faithfulness — git-only)
- Generation perplexity (Task B — requires loading the base model into
  transformers and is dominated by Tier-1 BERTScore signal)
- Glossary term recall (Task E — needs `prod.glossary` in the local snapshot)
- METEOR (Task B — wordnet corpus dep adds friction; BERTScore + ROUGE-L
  cover the same lexical/semantic axis)
"""
from __future__ import annotations
import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
PREDICTIONS = REPO / "results" / "predictions.parquet"
OUT = REPO / "results" / "scores_tier1.parquet"
UPSC_FACTS = REPO / "data" / "upsc_facts.json"


# ---------- shared scorers (lazy-loaded once) ----------

_nlp = None
_rouge = None
_facts: dict | None = None


def _spacy():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm", disable=["parser"])
    return _nlp


def _rouge_scorer():
    global _rouge
    if _rouge is None:
        from rouge_score import rouge_scorer
        _rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return _rouge


def _load_facts() -> dict:
    global _facts
    if _facts is None:
        _facts = json.loads(UPSC_FACTS.read_text())
    return _facts


# ---------- generic helpers ----------

def _parse_json(s: Any) -> dict:
    if isinstance(s, dict):
        return s
    if not isinstance(s, str) or not s.strip():
        return {}
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return {}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-zऀ-ॿ]+", text or "")


def _sentences(text: str) -> list[str]:
    if not text:
        return []
    # nltk's sent_tokenize requires a download; punkt-light regex split is
    # adequate for the readability + sentence-variance metrics we need.
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Zऀ-ॿ])", text.strip())
    return [p for p in parts if p]


def _entities_en(text: str) -> set[str]:
    if not text:
        return set()
    doc = _spacy()(text[:5000])  # cap to avoid pathological inputs
    return {ent.text.lower().strip() for ent in doc.ents if ent.text.strip()}


def _ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _extract_articles(text: str) -> list[int]:
    """Pull 'Article N' references from text. Returns the integer Ns."""
    return [int(m.group(1)) for m in re.finditer(r"\bArticle\s+(\d+)", text or "")]


def _extract_dates(text: str) -> set[str]:
    """Extract year-class dates (1900..2099) and full dates of various formats."""
    return set(re.findall(r"\b(?:19|20|21)\d{2}\b", text or ""))


def _extract_numbers(text: str) -> set[str]:
    """Numeric tokens with optional unit. Normalize to numeric token."""
    return set(re.findall(r"\b\d+(?:\.\d+)?\s?%?\b", text or ""))


def _word_count(text: str) -> int:
    return len(_tokens(text))


# ---------- BATCHED BERTScore + ROUGE-L ----------

def _bertscore_batch(cands: list[str], refs: list[str], lang: str) -> list[float]:
    """Return per-row F1 for matched (cand, ref) pairs. Empty inputs → 0.0."""
    if not cands:
        return []
    import bert_score
    # bert_score crashes on empty strings; replace with a single space.
    cands_clean = [c if c.strip() else " " for c in cands]
    refs_clean = [r if r.strip() else " " for r in refs]
    _, _, f1 = bert_score.score(
        cands_clean, refs_clean, lang=lang, rescale_with_baseline=False,
        verbose=False, batch_size=32,
    )
    return [float(x) for x in f1.tolist()]


def _rouge_l_batch(cands: list[str], refs: list[str]) -> list[float]:
    rs = _rouge_scorer()
    out = []
    for c, r in zip(cands, refs):
        if not c.strip() or not r.strip():
            out.append(0.0)
            continue
        out.append(rs.score(r, c)["rougeL"].fmeasure)
    return out


def _chrf_pair(cand: str, ref: str) -> float:
    if not cand.strip() or not ref.strip():
        return 0.0
    import sacrebleu
    return sacrebleu.sentence_chrf(cand, [ref], word_order=2).score / 100.0


# ---------- Task A ----------

REASONING_MARKERS = (
    r"\b(because|therefore|hence|thus|since|so that|as a result|"
    r"however|but|although|though|whereas|"
    r"first(?:ly)?|second(?:ly)?|third(?:ly)?|finally|"
    r"if|then|else|otherwise|"
    r"in fact|specifically|namely|for example|for instance|e\.g\.|"
    r"क्योंकि|इसलिए|परंतु|लेकिन|यदि|तब)\b"
)


def _distractor_coverage(explanation: str, options: dict | list, correct: str) -> float:
    """Fraction of wrong options addressed in the explanation."""
    if isinstance(options, list):
        opts_map = {o.get("id", "").upper(): o.get("text", "") for o in options if isinstance(o, dict)}
    elif isinstance(options, dict):
        opts_map = {k.upper(): v for k, v in options.items()}
    else:
        return 0.0
    correct = (correct or "").upper()
    wrong_letters = [k for k in ("A", "B", "C", "D") if k in opts_map and k != correct]
    if not wrong_letters:
        return 0.0
    expl_lower = (explanation or "").lower()
    hits = 0
    for letter in wrong_letters:
        opt_text = opts_map.get(letter, "")
        # tokenize option text; require letter mention AND ≥1 distinctive token
        opt_tokens = [t for t in _tokens(opt_text) if len(t) > 3]
        letter_mentioned = re.search(rf"\boption\s+{letter}\b|\b{letter}\)\B", expl_lower) is not None
        token_hit = any(t.lower() in expl_lower for t in opt_tokens[:5])
        if letter_mentioned and token_hit:
            hits += 1
    return hits / len(wrong_letters)


def _citation_accuracy(text: str) -> float:
    """For each Article N mentioned, check N exists in upsc_facts.articles."""
    articles = _extract_articles(text)
    if not articles:
        return 1.0  # vacuous — no citations to be wrong
    facts = _load_facts()
    known = {int(a["number"]) for a in facts.get("articles", []) if "number" in a}
    if not known:
        return 1.0
    hits = sum(1 for n in articles if n in known)
    return hits / len(articles)


def score_task_A(row: dict, gold: dict, pred: dict) -> dict:
    """Per-row scalars for Task A. Batched metrics (BERTScore) populated later."""
    correct_letter = (gold.get("correct_option") or "").strip().upper()
    pred_letter = (pred.get("answer") or "").strip().upper()
    pred_letter = pred_letter[:1] if pred_letter else ""
    format_fail = pred_letter not in {"A", "B", "C", "D"}
    is_correct = (not format_fail) and (pred_letter == correct_letter)

    paper = (row.get("paper") or "").lower()
    if paper == "csat":
        neg = 2.5 if is_correct else (0.0 if format_fail else -2.5 / 3)
    else:
        neg = 2.0 if is_correct else (0.0 if format_fail else -2.0 / 3)

    raw_conf = pred.get("confidence")
    try:
        conf = float(raw_conf) / 100.0 if raw_conf is not None else None
        if conf is not None:
            conf = max(0.0, min(1.0, conf))
    except (TypeError, ValueError):
        conf = None
    brier = (conf - (1.0 if is_correct else 0.0)) ** 2 if conf is not None else np.nan

    expl_pred = (pred.get("explanation") or "").strip()
    expl_gold = (gold.get("explanation") or "").strip()

    entity_f1 = 0.0
    if row.get("language") == "en" and expl_pred and expl_gold:
        ents_p = _entities_en(expl_pred)
        ents_g = _entities_en(expl_gold)
        if ents_p or ents_g:
            inter = ents_p & ents_g
            p = len(inter) / len(ents_p) if ents_p else 0.0
            r = len(inter) / len(ents_g) if ents_g else 0.0
            entity_f1 = (2 * p * r / (p + r)) if (p + r) else 0.0

    markers = re.findall(REASONING_MARKERS, expl_pred, flags=re.IGNORECASE)
    wc = max(_word_count(expl_pred), 1)
    reasoning_density = 100.0 * len(markers) / wc

    sent_lens = [_word_count(s) for s in _sentences(expl_pred)]
    sent_var = float(np.var(sent_lens)) if len(sent_lens) >= 2 else 0.0

    return {
        "is_correct": int(is_correct),
        "format_fail": int(format_fail),
        "predicted_letter": pred_letter if not format_fail else "",
        "correct_letter": correct_letter,
        "upsc_neg_marking_score": neg,
        "confidence_prob": conf if conf is not None else np.nan,
        "brier_loss": brier,
        "silly_mistake_prone": int(bool(gold.get("silly_mistake_prone"))),
        "explanation_entity_f1": entity_f1,
        "distractor_coverage": _distractor_coverage(
            expl_pred, gold.get("options"), correct_letter
        ),
        "reasoning_step_density_per100w": reasoning_density,
        "citation_accuracy": _citation_accuracy(expl_pred),
        "sentence_length_variance": sent_var,
    }


# ---------- Task B ----------

def _mattr(tokens: list[str], window: int = 100) -> float:
    """Moving-Average Type-Token Ratio. Returns 0.0 if text too short."""
    if len(tokens) < window:
        return len(set(tokens)) / max(1, len(tokens))
    ratios = [len(set(tokens[i:i + window])) / window for i in range(len(tokens) - window + 1)]
    return float(np.mean(ratios))


def _paragraph_target(word_target: int) -> tuple[int, int]:
    if word_target <= 175:
        return (1, 2)
    if word_target <= 400:
        return (3, 5)
    return (8, 12)


def _fact_lookup_precision(text: str) -> float:
    """Articles, schemes, acts mentioned in `text` → fraction recognized in upsc_facts."""
    facts = _load_facts()
    known_articles = {int(a["number"]) for a in facts.get("articles", []) if "number" in a}
    known_acts = {a["name"].lower() for a in facts.get("acts", []) if "name" in a}
    known_schemes = {s["name"].lower() for s in facts.get("schemes", []) if "name" in s}

    hits = total = 0
    for n in _extract_articles(text):
        total += 1
        if n in known_articles:
            hits += 1
    text_lower = (text or "").lower()
    for name in known_acts:
        if name in text_lower:
            total += 1
            hits += 1  # mentioning a known act counts as known
    for name in known_schemes:
        if name in text_lower:
            total += 1
            hits += 1
    return (hits / total) if total else 1.0


def _hindi_code_mixing(text: str) -> float:
    """For a Hindi-prompted answer: fraction of letter chars NOT in Devanagari."""
    letters = [c for c in (text or "") if c.isalpha()]
    if not letters:
        return 0.0
    devanagari = sum(1 for c in letters if "DEVANAGARI" in unicodedata.name(c, ""))
    return 1.0 - (devanagari / len(letters))


def score_task_B(row: dict, gold: dict, pred: dict) -> dict:
    cand = (pred.get("answer") or "").strip()
    ref = (gold.get("model_answer") or "").strip()

    target_wc = int(gold.get("word_count") or 250)
    wc_cand = _word_count(cand)
    wc_adherence = max(0.0, 1.0 - abs(wc_cand - target_wc) / max(1, target_wc))

    sents_cand = _sentences(cand)
    sents_ref = _sentences(ref)
    sc_adherence = max(0.0, 1.0 - abs(len(sents_cand) - len(sents_ref)) / max(1, len(sents_ref)))

    paragraphs = [p for p in re.split(r"\n\s*\n+", cand) if p.strip()]
    lo, hi = _paragraph_target(target_wc)
    if lo <= len(paragraphs) <= hi:
        para_adherence = 1.0
    else:
        dist = lo - len(paragraphs) if len(paragraphs) < lo else len(paragraphs) - hi
        para_adherence = max(0.0, 1.0 - dist / max(1, hi))

    entity_f1 = 0.0
    if row.get("language") == "en" and cand and ref:
        ents_c = _entities_en(cand)
        ents_r = _entities_en(ref)
        if ents_c or ents_r:
            inter = ents_c & ents_r
            p = len(inter) / len(ents_c) if ents_c else 0.0
            r = len(inter) / len(ents_r) if ents_r else 0.0
            entity_f1 = (2 * p * r / (p + r)) if (p + r) else 0.0

    dates_c, dates_r = _extract_dates(cand), _extract_dates(ref)
    if dates_c or dates_r:
        inter = dates_c & dates_r
        p = len(inter) / len(dates_c) if dates_c else 0.0
        r = len(inter) / len(dates_r) if dates_r else 0.0
        date_f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    else:
        date_f1 = 1.0  # vacuous

    nums_c, nums_r = _extract_numbers(cand), _extract_numbers(ref)
    if nums_c or nums_r:
        inter = nums_c & nums_r
        p = len(inter) / len(nums_c) if nums_c else 0.0
        r = len(inter) / len(nums_r) if nums_r else 0.0
        num_f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    else:
        num_f1 = 1.0

    toks = _tokens(cand)
    quad = _ngrams(toks, 4)
    if quad:
        cnt = Counter(quad)
        repeated = sum(c for c in cnt.values() if c > 1)
        ngram_rep = repeated / len(quad)
    else:
        ngram_rep = 0.0

    import textstat
    fk = float(textstat.flesch_kincaid_grade(cand)) if cand else 0.0

    return {
        "word_count_adherence": wc_adherence,
        "sentence_count_adherence": sc_adherence,
        "paragraph_count_adherence": para_adherence,
        "entity_f1": entity_f1,
        "date_exact_f1": date_f1,
        "numeric_exact_f1": num_f1,
        "hindi_code_mixing_rate": _hindi_code_mixing(cand) if row.get("language") == "hi" else np.nan,
        "mattr_100": _mattr(toks),
        "flesch_kincaid_grade": fk,
        "ngram4_repetition_rate": ngram_rep,
        "fact_lookup_precision": _fact_lookup_precision(cand),
        "output_word_count": wc_cand,
    }


# ---------- Task C ----------

C_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number"},
        "strengths": {"type": "array", "items": {"type": "string"}},
        "improvements": {
            "type": "object",
            "properties": {
                "intro": {"type": "array", "items": {"type": "string"}},
                "body": {"type": "array", "items": {"type": "string"}},
                "conclusion": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["intro", "body", "conclusion"],
        },
    },
    "required": ["score", "strengths", "improvements"],
}


def _lemma_set_en(text: str) -> set[str]:
    if not text:
        return set()
    doc = _spacy()(text[:5000])
    return {t.lemma_.lower() for t in doc if t.is_alpha and not t.is_stop and len(t) > 2}


def _bullet_list_text(x) -> str:
    if isinstance(x, list):
        return "\n".join(str(s) for s in x)
    if isinstance(x, str):
        return x
    return ""


def _flatten_improvements(imp) -> list[str]:
    """Improvements is nested {intro: [], body: [], conclusion: []} OR a flat list."""
    if isinstance(imp, list):
        return [str(s) for s in imp]
    if isinstance(imp, dict):
        out = []
        for k in ("intro", "body", "conclusion"):
            v = imp.get(k)
            if isinstance(v, list):
                out.extend(str(s) for s in v)
        return out
    return []


def score_task_C(row: dict, gold: dict, pred: dict) -> dict:
    pred_score = pred.get("score")
    try:
        pred_score = float(pred_score)
    except (TypeError, ValueError):
        pred_score = np.nan
    gold_score = float(gold.get("score") or 0)
    max_score = float(gold.get("max_score") or 0) or 1.0

    score_abs_err = abs(pred_score - gold_score) if not np.isnan(pred_score) else np.nan

    import jsonschema
    try:
        jsonschema.validate(pred, C_SCHEMA)
        schema_valid = 1
    except jsonschema.ValidationError:
        schema_valid = 0

    s_pred = _bullet_list_text(pred.get("strengths"))
    s_gold = _bullet_list_text(gold.get("strengths"))
    i_pred_items = _flatten_improvements(pred.get("improvements"))
    i_gold_items = _flatten_improvements(gold.get("improvements"))
    i_pred = "\n".join(i_pred_items)
    i_gold = "\n".join(i_gold_items)

    def _f1(a: set, b: set) -> float:
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        inter = a & b
        p = len(inter) / len(a)
        r = len(inter) / len(b)
        return (2 * p * r / (p + r)) if (p + r) else 0.0

    s_f1 = _f1(_lemma_set_en(s_pred), _lemma_set_en(s_gold))
    i_f1 = _f1(_lemma_set_en(i_pred), _lemma_set_en(i_gold))

    s_pred_n = len(pred.get("strengths") or []) if isinstance(pred.get("strengths"), list) else 0
    s_gold_n = len(gold.get("strengths") or []) if isinstance(gold.get("strengths"), list) else 0
    s_count_adh = max(0.0, 1.0 - abs(s_pred_n - s_gold_n) / max(1, s_gold_n))

    i_pred_n = len(i_pred_items)
    i_gold_n = len(i_gold_items)
    i_count_adh = max(0.0, 1.0 - abs(i_pred_n - i_gold_n) / max(1, i_gold_n))

    return {
        "pred_score": pred_score,
        "gold_score": gold_score,
        "max_score": max_score,
        "score_abs_err": score_abs_err,
        "schema_valid": schema_valid,
        "strengths_token_f1": s_f1,
        "improvements_token_f1": i_f1,
        "strengths_count_adherence": s_count_adh,
        "improvements_count_adherence": i_count_adh,
    }


# ---------- Task E ----------

def _compression_score(cand_tokens: int, src_tokens: int) -> float:
    if src_tokens == 0:
        return 0.0
    ratio = cand_tokens / src_tokens
    if 0.20 <= ratio <= 0.50:
        return 1.0
    if ratio < 0.20:
        return max(0.0, ratio / 0.20)
    return max(0.0, 1.0 - (ratio - 0.50) / 0.50)


def score_task_E(row: dict, gold: dict, pred: dict) -> dict:
    p_pred = (pred.get("prelims_info") or "").strip()
    m_pred = (pred.get("mains_info") or "").strip()
    p_gold = (gold.get("prelims_info") or "").strip()
    m_gold = (gold.get("mains_info") or "").strip()
    source = (gold.get("source_text") or "").strip()

    # Entity-F1 vs gold mains_info; hallucination = ents in m_pred not in source.
    ents_pred = _entities_en(m_pred)
    ents_gold = _entities_en(m_gold)
    ents_src = _entities_en(source)
    if ents_pred or ents_gold:
        inter = ents_pred & ents_gold
        p = len(inter) / len(ents_pred) if ents_pred else 0.0
        r = len(inter) / len(ents_gold) if ents_gold else 0.0
        entity_f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    else:
        entity_f1 = 0.0
    hallucinated = ents_pred - ents_src
    hallucination_rate = (len(hallucinated) / len(ents_pred)) if ents_pred else 0.0
    coverage = (len(ents_src & ents_pred) / len(ents_src)) if ents_src else 0.0

    src_dates, src_nums = _extract_dates(source), _extract_numbers(source)
    pred_dates, pred_nums = _extract_dates(m_pred), _extract_numbers(m_pred)

    def _set_f1(a: set, b: set) -> float:
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        inter = a & b
        p = len(inter) / len(a)
        r = len(inter) / len(b)
        return (2 * p * r / (p + r)) if (p + r) else 0.0

    date_f1 = _set_f1(pred_dates, src_dates)
    num_f1 = _set_f1(pred_nums, src_nums)

    cand_toks = _word_count(m_pred)
    src_toks = _word_count(source)
    compression = _compression_score(cand_toks, src_toks)

    citation_count = len(ents_pred) + len(pred_dates) + len(pred_nums)
    citation_density = 100.0 * citation_count / max(1, cand_toks)

    lead100 = " ".join(m_pred.split()[:100])
    headline_ents = _entities_en(" ".join(source.split()[:120]))
    lead_ents = _entities_en(lead100)
    lead_recall = (len(headline_ents & lead_ents) / len(headline_ents)) if headline_ents else 0.0

    return {
        "entity_f1_vs_gold": entity_f1,
        "hallucination_rate": hallucination_rate,
        "coverage_of_source_entities": coverage,
        "date_f1_vs_source": date_f1,
        "numeric_f1_vs_source": num_f1,
        "compression_ratio_score": compression,
        "citation_density_per100w": citation_density,
        "lead100_entity_recall": lead_recall,
        "fact_lookup_precision": _fact_lookup_precision(m_pred + "\n" + p_pred),
        "prelims_word_count": _word_count(p_pred),
        "mains_word_count": cand_toks,
    }


# ---------- batched cross-row metrics (BERTScore, ROUGE-L, chrF) ----------

def _batch_text_pairs(df: pd.DataFrame) -> dict[str, list[float]]:
    """Compute BERTScore-F1 and ROUGE-L F1 per row, batched by (task, language).

    Returns a dict of column-name → per-row value lists aligned to df.index.
    """
    n = len(df)
    out = {
        "explanation_bertscore_f1": [np.nan] * n,
        "explanation_rouge_l_f1": [np.nan] * n,
        "answer_bertscore_f1": [np.nan] * n,
        "answer_rouge_l_f1": [np.nan] * n,
        "answer_chrf": [np.nan] * n,
        "mains_bertscore_f1": [np.nan] * n,
        "mains_rouge_l_f1": [np.nan] * n,
        "prelims_bertscore_f1": [np.nan] * n,
        "prelims_rouge_l_f1": [np.nan] * n,
        "mains_chrf": [np.nan] * n,
        "strengths_bertscore_f1": [np.nan] * n,
    }
    df = df.reset_index(drop=True)

    def _fill(idxs: list[int], cands: list[str], refs: list[str], lang: str,
              col_bert: str, col_rouge: str, col_chrf: str | None = None) -> None:
        if not idxs:
            return
        bs = _bertscore_batch(cands, refs, lang)
        rs = _rouge_l_batch(cands, refs)
        for i, idx in enumerate(idxs):
            out[col_bert][idx] = bs[i] if i < len(bs) else np.nan
            out[col_rouge][idx] = rs[i] if i < len(rs) else np.nan
        if col_chrf:
            for i, idx in enumerate(idxs):
                out[col_chrf][idx] = _chrf_pair(cands[i], refs[i])

    # Task A — explanation pred vs gold, per language.
    for lang in ("en", "hi"):
        idxs, cands, refs = [], [], []
        for i, row in df.iterrows():
            if row["task"] != "A" or row["language"] != lang:
                continue
            gold = _parse_json(row["gold_payload"])
            pred = _parse_json(row["prediction"])
            expl_p = (pred.get("explanation") or "").strip()
            expl_g = (gold.get("explanation") or "").strip()
            if not expl_p or not expl_g:
                continue
            idxs.append(i)
            cands.append(expl_p)
            refs.append(expl_g)
        _fill(idxs, cands, refs, lang,
              "explanation_bertscore_f1", "explanation_rouge_l_f1")

    # Task B — answer pred vs gold model_answer, per language.
    for lang in ("en", "hi"):
        idxs, cands, refs = [], [], []
        for i, row in df.iterrows():
            if row["task"] != "B" or row["language"] != lang:
                continue
            gold = _parse_json(row["gold_payload"])
            pred = _parse_json(row["prediction"])
            cand = (pred.get("answer") or "").strip()
            ref = (gold.get("model_answer") or "").strip()
            if not cand or not ref:
                continue
            idxs.append(i)
            cands.append(cand)
            refs.append(ref)
        _fill(idxs, cands, refs, lang,
              "answer_bertscore_f1", "answer_rouge_l_f1", "answer_chrf")

    # Task C — strengths concat pred vs gold (English only).
    idxs, cands, refs = [], [], []
    for i, row in df.iterrows():
        if row["task"] != "C":
            continue
        gold = _parse_json(row["gold_payload"])
        pred = _parse_json(row["prediction"])
        cand = _bullet_list_text(pred.get("strengths"))
        ref = _bullet_list_text(gold.get("strengths"))
        if not cand.strip() or not ref.strip():
            continue
        idxs.append(i)
        cands.append(cand)
        refs.append(ref)
    if idxs:
        bs = _bertscore_batch(cands, refs, "en")
        for j, idx in enumerate(idxs):
            out["strengths_bertscore_f1"][idx] = bs[j]

    # Task E — prelims + mains separately, English.
    for kind, col_b, col_r, col_c in (
        ("prelims", "prelims_bertscore_f1", "prelims_rouge_l_f1", None),
        ("mains", "mains_bertscore_f1", "mains_rouge_l_f1", "mains_chrf"),
    ):
        idxs, cands, refs = [], [], []
        for i, row in df.iterrows():
            if row["task"] != "E":
                continue
            gold = _parse_json(row["gold_payload"])
            pred = _parse_json(row["prediction"])
            cand = (pred.get(f"{kind}_info") or "").strip()
            ref = (gold.get(f"{kind}_info") or "").strip()
            if not cand or not ref:
                continue
            idxs.append(i)
            cands.append(cand)
            refs.append(ref)
        _fill(idxs, cands, refs, "en", col_b, col_r, col_c)

    return out


# ---------- main ----------

SCORERS = {"A": score_task_A, "B": score_task_B, "C": score_task_C, "E": score_task_E}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, default=PREDICTIONS)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap rows scored (smoke runs)")
    args = ap.parse_args()

    if not args.predictions.exists():
        print(f"[FAIL] {args.predictions} not found; run inference first")
        return 1

    df = pd.read_parquet(args.predictions)
    if args.limit:
        df = df.head(args.limit)
    print(f"[load] {len(df):,} predictions from {args.predictions}")

    # Per-row metrics (regex/spaCy/textstat — cheap, sequential is fine).
    rows = []
    for r in df.to_dict("records"):
        task = r["task"]
        scorer = SCORERS.get(task)
        if scorer is None:
            continue
        gold = _parse_json(r.get("gold_payload"))
        pred = _parse_json(r.get("prediction"))
        m = scorer(r, gold, pred)
        rows.append({
            "run_id": r["run_id"],
            "condition": r["condition"],
            "question_id": r["question_id"],
            "task": task,
            "language": r.get("language"),
            "paper": r.get("paper"),
            "subject": r.get("subject"),
            "stratum_key": r.get("stratum_key"),
            **m,
        })
    scores = pd.DataFrame(rows)
    print(f"[per-row] {len(scores):,} rows scored")

    # Batched text-pair metrics — BERTScore is the slow one (loads model once).
    print("[batched] computing BERTScore / ROUGE-L / chrF …")
    batched = _batch_text_pairs(df)
    for col, vals in batched.items():
        if any(not (isinstance(v, float) and np.isnan(v)) for v in vals):
            scores[col] = vals[:len(scores)]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(args.out, index=False, compression="snappy")
    print(f"\n[OK] wrote {len(scores):,} score rows → {args.out}")
    print(f"     by task: {scores.groupby('task').size().to_dict()}")
    print(f"     by condition: {scores.groupby('condition').size().to_dict()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
