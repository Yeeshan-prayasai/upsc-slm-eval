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
DIMENSION_KEYWORDS = REPO / "data" / "dimension_keywords.json"

# Per-1M-token Gemini-3-Flash list pricing (USD). Same constants as
# scripts/runners.estimate_gemini_cost; replicated here to avoid an import cycle.
GEMINI_FLASH_IN_USD_PER_M = 0.50
GEMINI_FLASH_OUT_USD_PER_M = 3.00


def _cost_usd(condition: str, in_tokens: int, out_tokens: int) -> float:
    """Marginal $-cost per query. Local FT-SLMs (C1a/C1b) are $0; Gemini
    conditions are billed at the published per-token rate."""
    if condition in ("C1a", "C1b"):
        return 0.0
    return ((in_tokens or 0) / 1_000_000) * GEMINI_FLASH_IN_USD_PER_M \
         + ((out_tokens or 0) / 1_000_000) * GEMINI_FLASH_OUT_USD_PER_M


def _format_valid(task: str, parsed: dict, metrics: dict) -> int:
    """Per-row format-validity flag — did the parser extract a usable shape?
    A is valid iff a letter parsed; C/F/G expose explicit `schema_valid`; B/E
    are valid iff the JSON parsed at all (no _parse_error)."""
    if task == "A":
        return int(not metrics.get("format_fail", 0))
    if task in ("C", "F", "G"):
        return int(metrics.get("schema_valid", 0))
    # B/E: implicit — parsed must not carry the _parse_error sentinel and at
    # least one of the expected keys must be non-empty.
    if parsed.get("_parse_error"):
        return 0
    if task == "B":
        return int(bool((parsed.get("answer") or "").strip()))
    if task == "E":
        return int(bool((parsed.get("prelims_info") or "").strip())
                   or bool((parsed.get("mains_info") or "").strip()))
    return 1


# ---------- shared scorers (lazy-loaded once) ----------

_nlp = None
_rouge = None
_facts: dict | None = None
_dimensions: dict[str, list[str]] | None = None


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


def _load_dimensions() -> dict[str, list[str]]:
    """Load the PESEE dimension lexicon (Task G dimension coverage)."""
    global _dimensions
    if _dimensions is None:
        raw = json.loads(DIMENSION_KEYWORDS.read_text())
        _dimensions = {k: [w.lower() for w in v] for k, v in raw.items()
                       if not k.startswith("_") and isinstance(v, list)}
    return _dimensions


def _dimensions_touched(text: str) -> set[str]:
    """Return the subset of PESEE dimensions whose lexicon hits at least one
    whole-word match in the lowercased text."""
    if not text:
        return set()
    lex = _load_dimensions()
    low = text.lower()
    touched = set()
    for dim, words in lex.items():
        for w in words:
            # Whole-word match; allow multi-word phrases (no word-boundary lib needed).
            if re.search(rf"(?<![\w]){re.escape(w)}(?![\w])", low):
                touched.add(dim)
                break
    return touched


# UPSC GS subject → PESEE dimension proxy. Used as a weak Task-E
# "subject-tag accuracy" surrogate (eval-design §6.3): does the generated
# mains_info mention keywords associated with the gold subject's dimension?
# Engineered metric; flagged as exploratory in the report.
_SUBJECT_TO_DIMENSION = {
    "polity": "political", "governance": "political", "constitution": "political",
    "ir": "international", "international relations": "international",
    "economy": "economic", "economics": "economic",
    "society": "social", "social justice": "social",
    "environment": "environmental", "ecology": "environmental",
    "ethics": "ethical",
}


def _subject_tag_acc(pred_text: str, gold_subject: str,
                     gold_text: str = "") -> float:
    """Weak Task E 'subject-tag accuracy' proxy via PESEE dimension overlap.

    Two-tier resolution of the "gold subject":
    1. If the eval row has an explicit `subject` (e.g. Polity, Economy)
       and it maps to a PESEE dimension, that's the gold dimension.
    2. Otherwise (e.g. snapshot rows with `subject = 'UNTAGGED'`), infer
       the gold dimension from the gold mains_info itself: pick the single
       dominant PESEE dimension by lexicon hit count.

    Returns 1.0 if pred mains_info touches the gold dimension, 0.0 if not,
    NaN only when neither path yields a dimension (no gold text + no mapping).
    """
    dim: str | None = None
    if gold_subject:
        key = gold_subject.lower().strip()
        if key != "untagged":
            dim = _SUBJECT_TO_DIMENSION.get(key)
            if dim is None:
                for k, v in _SUBJECT_TO_DIMENSION.items():
                    if k in key:
                        dim = v
                        break
    if dim is None and gold_text:
        # Dominant-dimension fallback. Count whole-word hits per dimension;
        # pick the argmax. Tie-break is dict-insertion order (deterministic).
        lex = _load_dimensions()
        low = gold_text.lower()
        best, best_n = None, 0
        for d, words in lex.items():
            n = sum(1 for w in words
                    if re.search(rf"(?<![\w]){re.escape(w)}(?![\w])", low))
            if n > best_n:
                best, best_n = d, n
        dim = best
    if dim is None:
        return float("nan")
    return 1.0 if dim in _dimensions_touched(pred_text) else 0.0


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
    """Punkt-light sentence splitter — splits on `.!?` followed by whitespace
    and then ANY of: uppercase letter, Devanagari letter, digit, or an opening
    quote/paren ('"`([). Earlier version required uppercase or Devanagari only,
    which dropped sentences starting with a quoted phrase (`"He said..."`) or
    a number ("2024 marks the..."), inflating the per-sentence variance metric.
    """
    if not text:
        return []
    parts = re.split(
        r"(?<=[.!?])\s+(?=[A-Zऀ-ॿ0-9'\"`\(\[])",
        text.strip(),
    )
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

# Bilingual abstention markers — used to distinguish Task A abstention
# ("model said it cannot answer") from format failure ("model emitted garbage").
# Refusal counts as "abstain" (neg-mark = 0) at the rubric level; format failure
# is a broken prediction. We detect both then keep them in separate columns.
#
# English patterns wrap in `\b`-bounded alternation; Hindi patterns can't rely
# on ASCII `\b` semantics around Devanagari so are listed as bare substrings
# (the surrounding non-capturing alternation handles segmentation). re.UNICODE
# is default in Python 3 — included explicitly for clarity.
_REFUSAL_EN = (
    r"\b(?:cannot answer|can'?t answer|"
    r"i\s+(?:don'?t|do not)\s+know|"
    r"insufficient\s+(?:information|context)|"
    r"unable\s+to\s+(?:answer|determine)|"
    r"no\s+answer|none\s+of\s+the\s+above|"
    r"skip(?:ping)?)\b"
)
_REFUSAL_HI = (
    # मुझे नहीं पता / मुझे ज्ञात नहीं — "I don't know"
    r"मुझे\s+(?:नहीं\s+पता|ज्ञात\s+नहीं|पता\s+नहीं)|"
    # उत्तर नहीं मालूम — "answer not known"
    r"उत्तर\s+(?:नहीं\s+मालूम|ज्ञात\s+नहीं)|"
    # जानकारी अनुपलब्ध / जानकारी नहीं — "information unavailable / no info"
    r"जानकारी\s+(?:अनुपलब्ध|नहीं)|"
    # स्किप / असमर्थ — "skip / unable"
    r"असमर्थ|स्किप"
)
REFUSAL_MARKERS = re.compile(
    rf"(?:{_REFUSAL_EN})|(?:{_REFUSAL_HI})",
    flags=re.IGNORECASE | re.UNICODE,
)


def _is_refusal(raw: str, parsed: dict) -> int:
    """Per-row refusal flag. Distinct from format_fail: refusal is a model
    that *spoke* (parsed JSON or said something) but declined to answer; format
    failure is the parser couldn't extract a usable shape."""
    text = (raw or "") + " " + json.dumps(parsed, ensure_ascii=False)
    return int(bool(REFUSAL_MARKERS.search(text)))


# Common short words that would false-positive a distractor "token hit" if left
# in the option vocabulary (e.g. "this", "with", "from", "have"). Manually
# curated stopword-style filter — only applied to the per-distractor token-hit
# check below.
_DISTRACTOR_TOKEN_STOPWORDS = {
    "this", "that", "with", "from", "have", "been", "were", "they",
    "their", "there", "which", "what", "when", "where", "while", "than",
    "then", "some", "such", "only", "into", "also", "these", "those",
    "would", "could", "should", "shall", "will", "must", "many", "more",
    "most", "much", "very", "even", "each", "every", "both", "above",
    "below", "between", "during", "after", "before", "about",
}


def _distractor_coverage(explanation: str, options: dict | list, correct: str) -> float:
    """Fraction of wrong options the explanation explicitly addresses.

    Per eval-design §4.1: "the option letter is mentioned AND ≥ 1 distinctive
    token from that option's text appears in the explanation". Distinctive =
    >3 chars AND not in `_DISTRACTOR_TOKEN_STOPWORDS`. Letter matching is
    case-insensitive (Option A / option a / A) / a) all count).
    """
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
    expl_text = explanation or ""
    if not expl_text:
        return 0.0
    hits = 0
    for letter in wrong_letters:
        opt_text = opts_map.get(letter, "")
        # Filter to distinctive tokens (>3 chars, not common stopwords).
        opt_tokens = [
            t for t in _tokens(opt_text)
            if len(t) > 3 and t.lower() not in _DISTRACTOR_TOKEN_STOPWORDS
        ]
        # Case-insensitive letter mention. Accepts "Option A", "option A",
        # "(A)", "A)" — all standard rubric formats.
        letter_mentioned = re.search(
            rf"\boption\s+{letter}\b|\(\s*{letter}\s*\)|\b{letter}\)",
            expl_text,
            flags=re.IGNORECASE,
        ) is not None
        if not letter_mentioned:
            continue
        expl_lower = expl_text.lower()
        token_hit = any(t.lower() in expl_lower for t in opt_tokens[:5])
        if token_hit:
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
    refusal = _is_refusal(row.get("raw_output") or "", pred)

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
        "refusal": int(refusal),
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


def _word_count_adherence(actual: int, target: int) -> float:
    """Asymmetric word-count adherence — UPSC graders penalize undershoot
    (missing breadth/depth) more harshly than overshoot (still has content,
    just over time). 1.0 at exact target; linear decay outside; undershoot
    scaled 1.5× steeper than overshoot. Clipped to [0, 1].
    """
    target = max(1, int(target))
    delta = actual - target
    # 1.0 within ±10% (UPSC graders' typical leniency band).
    if abs(delta) <= 0.10 * target:
        return 1.0
    if delta < 0:
        # undershoot: 1 - 1.5 * (target - actual) / target
        return max(0.0, 1.0 - 1.5 * abs(delta) / target)
    # overshoot: 1 - (actual - target) / target
    return max(0.0, 1.0 - delta / target)


def score_task_B(row: dict, gold: dict, pred: dict) -> dict:
    cand = (pred.get("answer") or "").strip()
    ref = (gold.get("model_answer") or "").strip()

    target_wc = int(gold.get("word_count") or 250)
    wc_cand = _word_count(cand)
    wc_adherence = _word_count_adherence(wc_cand, target_wc)

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
        "schema_valid": int(bool(cand) and not pred.get("_parse_error")),
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

    # Per-section improvements lemma-F1 (intro / body / conclusion). Task C
    # rubric organizes improvement bullets by Mains section; per-section F1
    # captures whether the model identifies the SAME section needing work,
    # not just the set of words.
    def _section_lemmas(imp, key: str) -> set[str]:
        if not isinstance(imp, dict):
            return set()
        v = imp.get(key)
        text = "\n".join(str(s) for s in v) if isinstance(v, list) else (str(v) if v else "")
        return _lemma_set_en(text)
    i_pred_imp = pred.get("improvements")
    i_gold_imp = gold.get("improvements")
    intro_f1 = _f1(_section_lemmas(i_pred_imp, "intro"),
                   _section_lemmas(i_gold_imp, "intro"))
    body_f1 = _f1(_section_lemmas(i_pred_imp, "body"),
                  _section_lemmas(i_gold_imp, "body"))
    conc_f1 = _f1(_section_lemmas(i_pred_imp, "conclusion"),
                  _section_lemmas(i_gold_imp, "conclusion"))

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
        "improvements_intro_token_f1": intro_f1,
        "improvements_body_token_f1": body_f1,
        "improvements_conclusion_token_f1": conc_f1,
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
        "schema_valid": int(
            bool(p_pred or m_pred) and not pred.get("_parse_error")
        ),
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
        # Weak subject-tag proxy via PESEE dimension lookup. Falls back to
        # gold-mains_info dominant-dimension when row.subject = 'UNTAGGED'.
        "subject_tag_acc": _subject_tag_acc(
            m_pred, row.get("subject") or "", m_gold,
        ),
    }


# ---------- Task F (production-prompt Prelims explanation) ----------

DIRECTIVE_VERBS = {
    # high-density (causal/contrastive) — "analyze", "evaluate", "critically", "examine"
    "high": ("analyze", "analyse", "evaluate", "critically", "examine", "assess"),
    # low-density (descriptive) — "describe", "list", "outline", "discuss"
    "low":  ("describe", "list", "outline", "discuss", "comment", "explain"),
}


def _devanagari_share(text: str) -> float:
    letters = [c for c in (text or "") if c.isalpha()]
    if not letters:
        return 0.0
    dev = sum(1 for c in letters if "DEVANAGARI" in unicodedata.name(c, ""))
    return dev / len(letters)


def _pick_F_branch(pred: dict, language: str) -> str:
    """Pick the Task F branch to score against the (English-only) gold
    explanation.

    Task F output is `{"english": ..., "hindi": ...}` (bilingual production
    prompt). The Task-A gold `explanation` field in our eval set is **English
    only** (data audit confirmed this even for `language=hi` rows). So:

    - Score the ENGLISH branch against gold for both en + hi rows.
    - The Hindi branch is evaluated separately via the
      `hindi_branch_devanagari_share` + `hindi_branch_code_mixing_rate` columns
      below — those measure bilingual *format compliance*, not faithfulness vs
      gold.

    The `language` argument is retained in the signature for symmetry with the
    batched scorers but does not affect the branch choice.
    """
    en = (pred.get("english") or "").strip()
    hi = (pred.get("hindi") or "").strip()
    # Prefer English (gold is English). Fall back to Hindi only if the
    # model truly produced no English — better than scoring an empty string,
    # though this is an off-spec output and will show up in schema_valid=0.
    return en or hi


def score_task_F(row: dict, gold: dict, pred: dict) -> dict:
    """Task F — production-prompt Prelims Explanation Generation.

    Gold reference is the Task-A `explanation` field (always English in our
    eval set). We score the English branch against gold for both en + hi rows,
    and additionally report Devanagari-purity of the Hindi branch (no gold
    Hindi reference exists; this is format compliance only).
    """
    language = row.get("language") or "en"
    en_pred = (pred.get("english") or "").strip()
    hi_pred = (pred.get("hindi") or "").strip()
    expl_gold = (gold.get("explanation") or "").strip()
    correct_letter = (gold.get("correct_option") or "").strip().upper()

    schema_valid = int(bool(en_pred) and bool(hi_pred) and not pred.get("_parse_error"))

    # Score the language-matched branch vs gold (gold is English-only in this
    # eval set — see eval_set inspection notes). For Hindi rows we keep the
    # English branch for cross-row BERTScore but report hi_devanagari_share
    # separately as a bilingual format-compliance signal.
    branch = _pick_F_branch(pred, language)

    entity_f1 = 0.0
    if branch and expl_gold:
        ents_p = _entities_en(branch)
        ents_g = _entities_en(expl_gold)
        if ents_p or ents_g:
            inter = ents_p & ents_g
            p = len(inter) / len(ents_p) if ents_p else 0.0
            r = len(inter) / len(ents_g) if ents_g else 0.0
            entity_f1 = (2 * p * r / (p + r)) if (p + r) else 0.0

    markers = re.findall(REASONING_MARKERS, branch, flags=re.IGNORECASE)
    wc_pred = max(_word_count(branch), 1)
    reasoning_density = 100.0 * len(markers) / wc_pred

    target_wc = max(_word_count(expl_gold), 1)
    wc_adherence = max(0.0, 1.0 - abs(_word_count(branch) - target_wc) / target_wc)

    return {
        "schema_valid": schema_valid,
        "format_fail": int(not schema_valid),
        "explanation_entity_f1": entity_f1,
        "distractor_coverage": _distractor_coverage(
            branch, gold.get("options"), correct_letter
        ),
        "reasoning_step_density_per100w": reasoning_density,
        "citation_accuracy": _citation_accuracy(branch),
        "fact_lookup_precision": _fact_lookup_precision(branch),
        "word_count_adherence": wc_adherence,
        # NB: renamed from `hindi_code_mixing_rate` to disambiguate from Task B
        # — Task B measures Hindi code-mixing on the SINGLE answer string of a
        # Hindi-language eval row; Task F measures it on the model's Hindi
        # OUTPUT BRANCH regardless of input language (production prompt is
        # bilingual by design).
        "hindi_branch_code_mixing_rate": (
            (1.0 - _devanagari_share(hi_pred)) if hi_pred else np.nan
        ),
        "hindi_branch_devanagari_share": _devanagari_share(hi_pred) if hi_pred else np.nan,
        "english_word_count": _word_count(en_pred),
        "hindi_word_count": _word_count(hi_pred),
    }


# ---------- Task G (production-prompt Mains model-answer) ----------

def _directive_class(question: str) -> str:
    """Classify the directive verb in the Mains question. Returns 'high',
    'low', or 'unknown' for the directive-density expectation."""
    q = (question or "").lower()
    for cls in ("high", "low"):
        for v in DIRECTIVE_VERBS[cls]:
            # re.escape v defensively — current verbs have no metachars but
            # protects against future additions like "evaluate (critically)".
            if re.search(rf"\b{re.escape(v)}\b", q):
                return cls
    return "unknown"


def _causal_marker_density(text: str) -> float:
    """Causal/contrastive marker count per 100 words. Subset of REASONING_MARKERS
    that signals analysis depth (per eval-design §4.7)."""
    if not text:
        return 0.0
    pat = r"\b(because|therefore|hence|thus|since|however|whereas|although|" \
          r"क्योंकि|इसलिए|परंतु|लेकिन)\b"
    hits = re.findall(pat, text, flags=re.IGNORECASE | re.UNICODE)
    return 100.0 * len(hits) / max(1, _word_count(text))


def _directive_density_score(d_pred: float, d_gold: float) -> float:
    """Symmetric "how close is pred density to gold density" score in [0, 1].

    Better than `min(2.0, d_pred / d_gold)` which clipped asymmetrically and
    biased the mean upward when models over-marked. New formula:
        score = 1 - |log(d_pred / d_gold)| / log(3),  clipped to [0, 1]
    so a ratio of 1.0 → 1.0; ratio of 3.0 or 1/3 → 0.0; ratio of 2.0 → ≈0.37.
    Treats over- and under-marking symmetrically in log space.
    """
    if d_gold <= 0 or d_pred <= 0:
        # No causal markers in either generated OR gold → undefined.
        # Returning NaN here lets the aggregator skip these rows cleanly.
        return float("nan")
    ratio = d_pred / d_gold
    return max(0.0, 1.0 - abs(np.log(ratio)) / np.log(3.0))


def score_task_G(row: dict, gold: dict, pred: dict) -> dict:
    """Task G — production-prompt Mains model-answer.

    Carries over all Task-B Tier-1 metrics (word/sentence/paragraph adherence,
    Entity-F1, date/number F1, MATTR, FK grade, n-gram repetition, fact lookup,
    Hindi code mixing) plus two engineered metrics from eval-design §4.7:
    dimension-keyword coverage and directive-conditioned discourse density.
    """
    # Reuse Task B's metrics directly — gold field shapes are identical.
    base = score_task_B(row, gold, pred)

    cand = (pred.get("answer") or "").strip()
    ref = (gold.get("model_answer") or "").strip()
    question = (gold.get("question") or "").strip()

    # Dimension-keyword coverage (PESEE lexicon)
    touched_pred = _dimensions_touched(cand)
    touched_ref = _dimensions_touched(ref)
    if touched_ref:
        dim_coverage = len(touched_pred & touched_ref) / len(touched_ref)
    else:
        # Gold covers no PESEE dimensions; vacuously 1.0 if pred also covers none,
        # else 0.0 (pred over-extends beyond what gold required).
        dim_coverage = 1.0 if not touched_pred else 0.0

    # Directive-conditioned discourse density (exploratory). Symmetric score
    # in [0, 1]: 1.0 = pred density matches gold; 0.0 = off by ≥3× either way.
    d_pred = _causal_marker_density(cand)
    d_gold = _causal_marker_density(ref)
    directive_density_score = _directive_density_score(d_pred, d_gold)

    return {
        **base,
        "schema_valid": int(bool(cand) and not pred.get("_parse_error")),
        "dimension_keyword_coverage": dim_coverage,
        "dimensions_touched_pred": len(touched_pred),
        "dimensions_touched_gold": len(touched_ref),
        "directive_class": _directive_class(question),
        "directive_density_score": directive_density_score,
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
        "explanation_chrf": [np.nan] * n,
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

    # Task F — language-matched branch ({"english", "hindi"}) vs gold A
    # explanation (gold is English-only in this eval set; score the matched
    # branch — see _pick_F_branch + score_task_F docstring).
    for lang in ("en", "hi"):
        idxs, cands, refs = [], [], []
        for i, row in df.iterrows():
            if row["task"] != "F" or row["language"] != lang:
                continue
            gold = _parse_json(row["gold_payload"])
            pred = _parse_json(row["prediction"])
            branch = _pick_F_branch(pred, lang)
            expl_g = (gold.get("explanation") or "").strip()
            if not branch.strip() or not expl_g:
                continue
            idxs.append(i)
            cands.append(branch)
            refs.append(expl_g)
        # chrF is essential for Devanagari/Hindi per eval-design §4.6.
        _fill(idxs, cands, refs, lang,
              "explanation_bertscore_f1", "explanation_rouge_l_f1",
              "explanation_chrf")

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

    # Task G — same answer-vs-gold-model-answer shape as Task B, but for
    # production-prompt outputs (raw markdown body under pred["answer"]).
    for lang in ("en", "hi"):
        idxs, cands, refs = [], [], []
        for i, row in df.iterrows():
            if row["task"] != "G" or row["language"] != lang:
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

SCORERS = {
    "A": score_task_A, "B": score_task_B, "C": score_task_C, "E": score_task_E,
    "F": score_task_F, "G": score_task_G,
}


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
    # We track the SOURCE df-index of each scored row so the batched metrics
    # (computed against the full df with df.iterrows()) can be reattached
    # by index — not by position. Position-based alignment is unsafe if any
    # predictions are skipped (unknown task without scorer).
    rows = []
    source_indices: list[int] = []
    df_reset = df.reset_index(drop=True)
    for i, r in enumerate(df_reset.to_dict("records")):
        task = r["task"]
        scorer = SCORERS.get(task)
        if scorer is None:
            continue
        gold = _parse_json(r.get("gold_payload"))
        pred = _parse_json(r.get("prediction"))
        m = scorer(r, gold, pred)
        # Universal §6.4 metrics propagated from predictions.parquet so the
        # aggregator can compute latency percentiles / cost / validity in a
        # single pass without rejoining the predictions table.
        lat_ms = r.get("latency_ms") or 0
        out_tok = r.get("output_tokens") or 0
        in_tok = r.get("input_tokens") or 0
        tps = (out_tok / (lat_ms / 1000.0)) if lat_ms > 0 else np.nan
        rows.append({
            "run_id": r["run_id"],
            "condition": r["condition"],
            "question_id": r["question_id"],
            "task": task,
            "language": r.get("language"),
            "paper": r.get("paper"),
            "subject": r.get("subject"),
            "stratum_key": r.get("stratum_key"),
            "latency_ms": float(lat_ms),
            "ttft_ms": float(r.get("ttft_ms") or 0),
            "input_tokens": int(in_tok),
            "output_tokens": int(out_tok),
            "tokens_per_sec": float(tps) if tps == tps else np.nan,
            "cost_usd": _cost_usd(r["condition"], int(in_tok), int(out_tok)),
            "format_valid": _format_valid(task, pred, m),
            **m,
        })
        source_indices.append(i)
    scores = pd.DataFrame(rows)
    print(f"[per-row] {len(scores):,} rows scored")

    # Batched text-pair metrics — BERTScore is the slow one (loads model once).
    # Reattach by SOURCE df-index so any skipped (no-scorer) rows don't shift
    # the alignment.
    print("[batched] computing BERTScore / ROUGE-L / chrF …")
    batched = _batch_text_pairs(df_reset)
    for col, vals in batched.items():
        # Pull values at the *source* indices we kept in `rows`.
        aligned = [vals[i] for i in source_indices]
        if any(not (isinstance(v, float) and np.isnan(v)) for v in aligned):
            scores[col] = aligned

    args.out.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(args.out, index=False, compression="snappy")
    print(f"\n[OK] wrote {len(scores):,} score rows → {args.out}")
    print(f"     by task: {scores.groupby('task').size().to_dict()}")
    print(f"     by condition: {scores.groupby('condition').size().to_dict()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
