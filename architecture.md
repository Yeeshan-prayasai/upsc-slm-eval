# Testing architecture — UPSC SLM eval pipeline

| Field | Value |
|---|---|
| Owner | Yeeshan — Data Scientist, prayas.ai |
| Companions | [`experiment-report.md`](experiment-report.md), [`eval-design.md`](eval-design.md), [`project-context.md`](project-context.md), [`project-brief.md`](project-brief.md) |
| Status | Design draft 2026-05-14 |

Specifies (a) the four-plane pipeline, (b) failure modes and recovery, (c) UPSC + ed-tech-specific design choices.

---

## 1. Pipeline

```
┌──────────┐   ┌────────────┐   ┌──────────┐   ┌────────────┐
│   DATA   │ → │ INFERENCE  │ → │  SCORING │ → │ DASHBOARD  │
└────┬─────┘   └─────┬──────┘   └─────┬────┘   └─────┬──────┘
     │               │                │              │
     ▼               ▼                ▼              ▼
   ┌─────────────────────────────────────────────────────────┐
   │ Append-only result store: results/runs/<id>/*.parquet   │
   │ One row per (run_id, condition, question_id)            │
   └─────────────────────────────────────────────────────────┘
     ▲
     │
   ┌─────────────────────────────────────────────────────────┐
   │ Manifest per run: SHAs of code, eval-set, FT corpus,    │
   │ adapter, scorer-models, prompts, vendor model_ids       │
   └─────────────────────────────────────────────────────────┘
```

The result store is the only source of truth. The dashboard, the statistical tests, and the auto-populated [`experiment-report.md`](experiment-report.md) tables all read from it.

---

## 2. Data plane

### 2.1 Eval-set freezing (`scripts/freeze_eval_set.py`)

1. Open Postgres connections (read-only role, `REPEATABLE READ`).
2. Pull source tables with **pinned column projections** — never `SELECT *` — and `ORDER BY question_id` for deterministic ordering.
3. Stratify per [`experiment-report.md §3.3.2`](experiment-report.md).
4. Sample uniformly with `random.Random(20260514)`.
5. Write `data/eval_set.parquet` + `data/eval_set.sha256`.

Determinism invariant tested in `tests/test_freeze_determinism.py`: same seed → same SHA, byte-for-byte.

### 2.2 Anti-leakage CI assertion (`scripts/build_ft_corpus.py`)

```python
eval_ids = set(pd.read_parquet("data/eval_set.parquet")["question_id"])
ft_ids   = set(pd.read_parquet("data/ft_corpus.parquet")["question_id"])
assert eval_ids.isdisjoint(ft_ids), \
    f"LEAKAGE: {len(eval_ids & ft_ids)} eval IDs in FT corpus."
```

Hard stop. FT scripts refuse to start if the assertion fails.

### 2.3 Bilingual handling

`question`, `options`, `explanation`, `model_answer` columns are JSONB shaped `{"english": ..., "hindi": ...}`. Both keys are read; one row per language is emitted with a `language ∈ {'en', 'hi'}` column. Tasks A and B evaluate per language; the bilingual delta is reported as a first-class metric.

### 2.4 Student-PII handling (UPSC-specific)

| Risk | Mitigation |
|---|---|
| `userId` leaking into eval set / FT corpus | All FK columns dropped before writing parquet; `question_id` is the only join key kept |
| Student writing reaching the LLM judge | Tier-2 judge prompts in Task C use *predicted* score/strengths/improvements, not the original student answer |
| `chat_messages` containing PII | Excluded from v1; referenced only for query-pattern analysis |
| Identity-revealing DAF content | Task D out of scope for v1 |

---

## 3. Inference plane

### 3.1 Unified entry (`scripts/run_inference.py --condition {C1a,C1b,C2,C3}`)

```python
class ConditionRunner(Protocol):
    def predict(self, item, language) -> Prediction: ...
    def confidence(self, item, predicted_letter) -> int: ...
    def cost_usd(self, prediction) -> float: ...

class GemmaFTRunner(ConditionRunner):        # C1a — MLX-LM + Gemma LoRA adapter
class QwenFTRunner(ConditionRunner):         # C1b — MLX-LM + Qwen LoRA adapter
class GeminiZeroShotRunner(ConditionRunner): # C2 — google-genai
class GeminiFewShotRunner(ConditionRunner):  # C3 — google-genai + 3 exemplars
```

Same prompt files across runners — only the model invocation differs. C1a and C1b share the `MLXLoRARunner` parent class; the only difference is which (base model, adapter) pair is loaded.

### 3.2 Resumability

`predictions.parquet` is append-only keyed by `(run_id, condition, question_id)`. On crash: read the existing file, compute the set-difference of eval × conditions, resume. A 1,200-question completed prefix is not redone.

### 3.3 Retry + cost ceiling

```python
@retry(stop=stop_after_attempt(5),
       wait=wait_exponential(min=1, max=16),
       retry=retry_if_exception_type((APITimeoutError, APIConnectionError)))
```

Rate-limit (HTTP 429) handled with a per-provider token bucket — sleep until refill, do not fail.

Before C2/C3 launch, the runner estimates `expected_cost`. If above `BUDGET_USD` (default $25 per condition for 2,000 items), the runner refuses to start without `--confirm-cost`.

### 3.4 Adapter quality gate (`scripts/validate_adapter.py`)

Run once per adapter (Gemma and Qwen) before C1a/C1b full inference: 50 held-out (not in eval, not in FT) samples per task. Checks non-empty output, parseable structured output, no NaN/Inf in logits. Failure on either adapter halts the pipeline before the expensive full inference.

### 3.5 Latency measurement

- Application-layer timing (`time.perf_counter()`).
- C1 model load is one-shot warm-up, not per-call.
- C2/C3 reuse the HTTPS connection (TLS handshake out of the timed window).
- All four conditions use the same `max_output_tokens` per task.
- `latency_ms`, `ttft_ms`, `input_tokens`, `output_tokens` written per row.

---

## 4. Scoring plane

### 4.1 Tier 1 — deterministic (`scripts/score_tier1.py`)

```python
def score_row(row) -> dict[str, float]:
    return {**task_A(row), **task_B(row), **task_C(row), **task_E(row), **universal(row)}
```

Idempotent — re-running matches existing values. BERTScore and BLEURT batched per task to amortize scorer-model load.

### 4.2 Tier 2 — LLM-judge (`scripts/score_tier2.py`)

`claude-sonnet-4-6` with Anthropic prompt caching (rubric + task instructions are stable across all 400 items in a task; cache absorbs the input tokens).

Per-row disk cache (`results/runs/<id>/tier2_cache/<question_id>.json`) — accidental re-runs do not double-spend.

Tier-2 results land in *separate* columns (`tier2_*`). The dashboard renders them in a clearly labeled "Diagnostic — not headline" panel.

### 4.3 Aggregation determinism

`pandas.groupby(..., observed=True)`. Bootstrap CIs via `scipy.stats.bootstrap(..., random_state=20260514, n_resamples=10000)`. Same `scored.parquet` → bit-identical `aggregate.parquet`.

### 4.4 Statistical tests (`scripts/test_hypotheses.py`)

- **Continuous metrics:** paired bootstrap on question IDs (preserves pairing), 10K resamples, 95% percentile CI.
- **Binary outcomes:** McNemar exact.
- **Multiple comparisons:** BH-FDR (`statsmodels.stats.multitest.multipletests(method='fdr_bh', alpha=0.05)`) across all (task × metric × comparison) p-values.
- **Effect size:** Cohen's d (continuous) / h (proportions); QWK is its own effect-size.

---

## 5. Result store

One row per `(run_id, condition, question_id)`. Schema in [`eval-design.md §5.3`](eval-design.md). Append-only; old runs never modified. Each run is a separate Parquet partition under `results/runs/<id>/`.

Local primary; private S3 mirror (server-side encrypted) for archive. `scripts/archive_run.py` writes a `manifest.json` with SHA-256 of every artifact — disaster recovery = re-download + verify hashes.

---

## 6. Dashboard plane (Streamlit)

| Page | Purpose |
|---|---|
| Aggregate metrics | Per-(task × stratum) heatmaps of condition deltas; headline tables; significance flags |
| Side-by-side query | Live ad-hoc query: user types, four conditions answer in parallel. For exploration, not for metric computation. |
| Per-question drilldown | Click any `question_id` → gold + 3 predictions + all metrics + judge rationale |
| Calibration plots | Reliability diagrams for Task A confidence-vs-accuracy |
| Failure modes | Worst predictions per task — surfaces patterns |
| Run comparison | Diff two `run_id`s on every aggregate metric — for measuring continual-FT improvements |

C1's local MLX-LM is loaded once into the Streamlit process and stays warm.

`@st.cache_data` on every parquet read (TTL: infinite for past runs, 5 min for the `latest` symlink).

---

## 7. Reliability

### 7.1 Reproducibility mechanisms

| Mechanism | Guarantees |
|---|---|
| Seed `20260514` everywhere | Same data + code → same hashes |
| `requirements.txt` exact-pinned with `pip-compile --generate-hashes` | Library versions stable |
| `models/lockfile.json` with HF model SHAs | Scorer-model checkpoints stable |
| `manifest.json` per run | Every input hash recorded |
| Vendor `model_id` from response headers logged | Detects mid-run model rotation |
| Both LoRA adapter SHAs recorded (`gemma_adapter_sha256`, `qwen_adapter_sha256`) | Same trained models |

### 7.2 Failure-mode catalog

| Failure | Detection | Recovery |
|---|---|---|
| Postgres timeout during freeze | `psycopg2` exception | Retry w/ backoff; surface on persistent fail |
| MLX OOM during FT | `mlx.core.OutOfMemoryError` | Reduce batch/seq_len in `configs/lora.yaml`; note in `runs/<id>/notes.md` |
| Adapter NaN | `validate_adapter.py` flags | Resume from last good checkpoint |
| Vendor API outage | `APIError` after backoff exhausted | Pause; partial `predictions.parquet` preserved |
| Vendor model rotated mid-run | `model_id` header changes | Log warning; either complete + note or abort |
| Scorer checkpoint changed on HF | `models/lockfile.json` SHA mismatch | Pipeline refuses to start; manual re-pin |
| Disk full | `OSError` | Halts; checkpoint intact for resume |

### 7.3 Monitoring (single-machine, v1)

| Signal | Source | Alert threshold |
|---|---|---|
| FT step loss | `training.jsonl` | Divergence (loss × 1.5 over 100 steps) → kill |
| FT throughput | Same | < 50% baseline → investigate thermal |
| Inference latency p95 | `predictions.parquet` | C1 > 5s, C2/C3 > 10s |
| API error rate | Inference logs | > 1% |
| Cost burn rate | Running tally | > 20% above estimate → pause |
| Metric NaN rate | `score_tier1.py` | > 0.1% |

---

## 8. UPSC + ed-tech-specific design choices

- Bilingual stratification (`language ∈ {'en','hi'}`) — first-class because UPSC issues every paper in both.
- `silly_mistake_prone` flag from `prelims_pyq_questions` is its own stratum.
- UPSC negative-marking score (+2 / −0.66 / 0) alongside accuracy — captures the strategic dimension Prelims rewards.
- Word-count adherence as a primary metric for Task B (UPSC explicitly penalizes word-count violations).
- 5-axis G-Eval rubric (Content / Contextual / Analytical / Structural / Directive) — matches `evaluation_questions.strengths` JSONB schema from prayas's evaluator playbook.
- QWK for Task C — ASAP-standard ordinal-essay-scoring metric.
- Hindi code-mixing rate — common failure of generalist models on Indic; jarring for Hindi-medium aspirants.
- Current-Affairs cutoff 2026-04-30 — Mains is September; April is the knowledge frontier just before consolidation.
- Article text passed as input (not assumed in model memory) — UPSC current-affairs prep is synthesis-from-source, not recall.
- Entity-F1 over named entities + dates + numbers — UPSC graders are unforgiving about wrong dates / scheme names / years.

---

## 9. CI / CD

Pre-commit: `ruff`, `mypy --strict`, fast tests, `requirements.txt` lock check.

PR: `tests/test_leakage_assertion.py`, `tests/test_inference_idempotence.py` (5-question mock at temp 0 → bitwise identical), `tests/test_dashboard_smoke.py` (headless Streamlit render).

Merge-to-main: A2 Hindi probe re-run against current base model (drift detection); 100-item abbreviated eval; diff aggregate metrics vs last main-CI run, flag > 2-pp regressions.

---

## 10. Acceptance criteria

Implementation-ready when:

- [ ] `scripts/` has stub implementations passing their precondition checks
- [ ] `requirements.txt` + `models/lockfile.json` committed; `pip install -r requirements.txt` succeeds on M5
- [ ] `tests/test_freeze_determinism.py` and `tests/test_leakage_assertion.py` pass with synthetic fixtures
- [ ] One end-to-end dry run (100-item eval, 5-step FT) completes with all stages green and a populated dashboard

Past this gate, run the full experiment defined in [`experiment-report.md`](experiment-report.md).
