# Test strategy — Phases 1, 2, 3

| Field | Value |
|---|---|
| Owner | Yeeshan |
| Scope | Phase 1 (repo bootstrap) + Phase 2 (data plane: eval-set freezer, FT-corpus builder, UPSC-facts JSON, leakage assertion) + Phase 3 (A2 Hindi-capability probe + gate) |
| Out of scope | Phases 4-8 (FT, inference, scoring, dashboard) — separate strategies when those land |

Six layers, run in this order. Each step has a concrete command and a binary pass/fail.

> **Prod-DB safety:** the only script in this codebase that touches remote Postgres is `scripts/snapshot_to_local.py`. Every other script reads from `data/prayas_local.sqlite`. The snapshot script uses `conn.set_session(readonly=True, isolation_level="REPEATABLE READ")` — Postgres will reject any non-SELECT statement. `grep -rnE '\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE)\b' scripts/` must return empty.

---

## Layer 0 — Snapshot the prod-DB subset (one-shot, ~1 min)

```
make snapshot
```

Pulls 8 tables from upscdev + prod into `data/prayas_local.sqlite`. **This is the only step that touches prod.**

| Check | Pass criterion |
|---|---|
| Exit 0 | `echo $?` |
| SQLite file exists | `test -f data/prayas_local.sqlite` |
| Sidecar SHA exists | `test -f data/prayas_local.sha256` |
| Expected tables present | `sqlite3 data/prayas_local.sqlite '.tables'` shows: `prelims_pyq_questions`, `upsc_prelims_ai_generated_que`, `pyqs`, `evaluation_questions`, `mcqs`, `news_articles`, `current_affairs`, `glossary`, `_snapshot_meta` |
| Row counts plausible | output lists per-table counts; sum ≈ 44K |
| File size | ~50–200 MB |

---

## Layer 1 — Pre-flight (one-shot, ~3 min)

| Check | Command | Pass criterion |
|---|---|---|
| 1.1 Python ≥ 3.12 | `python3 --version` | reports 3.12+ |
| 1.2 Install deps | `pip install -r requirements.txt` | exits 0; no resolver errors |
| 1.3 API keys set | `echo "${ANTHROPIC_API_KEY:?missing}" "${GEMINI_API_KEY:?missing}" \| wc -c` | both set; wc > 1 |
| 1.4 Full env gate | `make verify-env` | prints `[OK] env verified`; exits 0 |

If 1.4 fails, fix the reported issue(s) before proceeding. Common: missing API key, db-creds.txt parse error, missing package. (Network reachability is no longer tested here — that's part of `make snapshot`.)

---

## Layer 2 — Smoke tests (each script runs end-to-end)

Each script is run once and inspected for non-error completion + expected output.

### 2.1 UPSC facts validation

```
make build-facts
```

| Check | How |
|---|---|
| Exit 0 | `echo $?` |
| Schema validates | The script raises `jsonschema.ValidationError` otherwise |
| SHA-256 sidecar written | `test -f data/upsc_facts.json.sha256 && cat data/upsc_facts.json.sha256` |
| Summary counts printed | `articles: 49 / schedules: 12 / acts: 23 / plans: 12 / schemes: 20` |

### 2.2 Eval-set freeze

```
make freeze
```

Expected log lines:
```
[ 800/A] pulling Prelims MCQ pools …
        candidates: N (some 5-figure number)
[ 400/B] pulling Mains generation pool …
        candidates: ~3K
[ 500/C] pulling rubric-graded answers …
        candidates: ~10K
[ 300/E] pulling current-affairs articles (date ≤ 2026-04-30) …
        candidates: ~3K

[OK] wrote 2,000 rows → data/eval_set.parquet
     SHA-256: <64 hex chars>
     by task: {'A': 800, 'B': 400, 'C': 500, 'E': 300}
```

| Check | Pass criterion |
|---|---|
| Total rows == 2,000 | The `by task` sum equals 2,000 |
| Per-task counts == targets | A=800, B=400, C=500, E=300 |
| Candidate pool > target | Otherwise the stratifier is starved — flag it |
| Parquet file exists | `test -f data/eval_set.parquet` |
| Sidecar exists | `test -f data/eval_set.sha256` |

### 2.3 FT corpus build + leakage assertion

```
make build-ft-corpus
```

Expected:
```
[load] eval-set IDs: 2,000
  task A: +N_A      (N_A in 10K-15K range)
  task B: +N_B      (~1K-1.5K)
  task C: +N_C      (~10K)
  task E: +N_E      (~3K)
...
[OK] wrote N pairs → data/ft_corpus.parquet
     SHA-256: <64 hex chars>
     by task: {...}
     leakage check: PASS (eval ∩ ft = ∅)
```

| Check | Pass criterion |
|---|---|
| Exit 0 | `echo $?` |
| Last line says `PASS` | grep |
| FT corpus is large | total ≫ 2,000 (≥ 20K expected) |
| Each task represented | all 4 task counts > 0 |

If the leakage assertion trips, **do not proceed**. Investigate. Either a duplicate-ID is sneaking in or the eval-set freezer changed semantics.

### 2.4 A2 Hindi-capability probe (Phase 3)

```
make probe-hindi    # downloads + runs both base models on 50 Hindi MCQs each
make gate-hindi     # one-sided binomial test (H0: p=0.25) at alpha=0.05
```

Expected log per model:
```
Loading <hf_id> …
  [ 20/50] running acc: 0.X
  [ 40/50] running acc: 0.X
[OK] <hf_id>: NN/50 = 0.XXX → results/pre_ft_hindi_probe.parquet
```

Gate output:
```
A2 Hindi-capability gate
  H0: accuracy = 0.25  vs  H1: accuracy > 0.25  (one-sided binomial)
  alpha = 0.05

  PASS  deadbydawn101/gemma-4-E4B-mlx-4bit          NN/50 = 0.XXX   p = 0.XXXX
  PASS  mlx-community/Qwen3.5-4B-MLX-4bit           NN/50 = 0.XXX   p = 0.XXXX

[A2 gate PASS] all models reject H0 at alpha=0.05.
```

| Check | Pass criterion |
|---|---|
| Probe exit 0 | `echo $?` |
| Results file exists | `test -f results/pre_ft_hindi_probe.parquet` |
| Both models present | the parquet contains rows for both HF ids |
| Gate exit 0 | both models reject H0 (p < 0.05; at n=50 needs ≥ 18/50 correct) → proceed to Phase 4 |
| Gate exit 1 | one or more models fails to reject H0 → flag for separate-stratum reporting; do not block Phase 4 |

**Note:** the gate failing is *not* a project failure — it tells us to report Hindi-stratum results separately for that model. Phase 4 (FT) proceeds either way.

First-run latency: each model downloads ~5 GB on first load, then runs ~1-2 sec per question on M5. Total wall-clock for both models: ~3-5 min on a warm cache; ~15-25 min including model downloads.

---

## Layer 3 — Property tests (`pytest`)

```
make test
```

Two tests, both must pass:

| Test | What it proves |
|---|---|
| `test_eval_set_freeze_is_deterministic` (integration) | Running the freezer twice with the same seed produces a byte-identical Parquet. Catches non-determinism from unstable Postgres row ordering, set iteration order, dict ordering, etc. |
| `test_leakage_assertion.py::test_*` (4 cases) | The real `assert_no_leakage` function in `build_ft_corpus.py` (a) raises on any overlap, (b) passes on disjoint sets, (c) handles empty inputs. Uses the actual production code path — not a shadow. |

Pass criterion: `pytest` prints `5 passed`.

---

## Layer 4 — Data-quality spot-checks (~5 min, paste into a Python REPL)

These verify the *content* of the Parquet files, not just their existence. Run after Layer 2 succeeds.

### 4.1 Eval-set inspection

```python
import json, pandas as pd
df = pd.read_parquet('data/eval_set.parquet')

# 4.1a Schema
assert set(df.columns) == {'question_id','task','source_db','source_table',
                            'paper','subject','language','gold_payload','stratum_key'}, df.columns

# 4.1b Counts
print(df.groupby('task').size())                   # A=800, B=400, C=500, E=300
print(df.groupby(['task','language']).size())      # A bilingual; C, E mostly en

# 4.1c Language values constrained
assert set(df['language'].unique()) <= {'en', 'hi'}, df['language'].unique()

# 4.1d Stratum diversity per task
for t in ['A','B','C','E']:
    n = df[df['task']==t]['stratum_key'].nunique()
    print(f"task {t}: {n} distinct strata")        # expect ≥10 for A, ≥8 for B/C/E

# 4.1e gold_payload parses for every row
df['gold_payload'].apply(json.loads).head()        # must not raise

# 4.1f No null question_id
assert df['question_id'].notna().all()
assert df['question_id'].is_unique                 # no duplicate IDs

# 4.1g Sample one row per task, eyeball it
for t in ['A','B','C','E']:
    row = df[df['task']==t].iloc[0]
    print(f"\n--- task {t} ({row['language']}) ---")
    print(f"id: {row['question_id']}")
    print(f"stratum: {row['stratum_key']}")
    print(f"gold keys: {list(json.loads(row['gold_payload']).keys())}")
```

| Check | Pass criterion |
|---|---|
| 4.1a Schema | passes |
| 4.1b Counts | A=800, B=400, C=500, E=300 |
| 4.1c Languages | only `{en, hi}` |
| 4.1d Diversity | each task ≥ 8 distinct strata (no single stratum dominating) |
| 4.1e JSON parses | all 2,000 rows parse without exception |
| 4.1f IDs unique | no duplicates |
| 4.1g Gold-payload keys | Task A has `question, options, correct_option, explanation`; Task B has `question, model_answer, word_count, max_score`; Task C has `question_text, answer_text, score, max_score, strengths, improvements`; Task E has `date, title, source_text, prelims_info, mains_info` |

### 4.2 FT-corpus inspection

```python
import pandas as pd, json
ft = pd.read_parquet('data/ft_corpus.parquet')

# 4.2a Schema
assert set(ft.columns) == {'pair_id','task','language','source_db','source_table',
                            'instruction','input','output'}

# 4.2b Counts per task (rough)
print(ft.groupby('task').size())                   # all > 0

# 4.2c Leakage verification (defense in depth)
ev = set(pd.read_parquet('data/eval_set.parquet')['question_id'])
ft_ids = set(ft['pair_id'])
assert ev.isdisjoint(ft_ids), f"LEAKAGE: {len(ev & ft_ids)} IDs!"

# 4.2d Instructions valid
assert ft['instruction'].str.startswith('[TASK=').all()

# 4.2e input/output parse where structured
for t in ['A','B','C','E']:
    sample = ft[ft['task']==t].iloc[0]
    print(f"\n--- task {t} ---")
    print(f"instruction: {sample['instruction']}")
    print(f"input head:  {sample['input'][:120]}")
    print(f"output head: {sample['output'][:120]}")
```

| Check | Pass criterion |
|---|---|
| 4.2a Schema | passes |
| 4.2b Counts | all four tasks > 0 |
| 4.2c Leakage | empty intersection (belt-and-suspenders alongside the production assertion) |
| 4.2d Instructions | every row starts with `[TASK=` |
| 4.2e Sample rows | input is meaningful UPSC content; output matches expected shape (letter+explanation for A, prose for B, JSON for C, JSON for E) |

### 4.3 UPSC facts inspection

```python
import json
facts = json.loads(open('data/upsc_facts.json').read())
assert facts['_meta']['version'] == '1.0'
print(f"articles: {len(facts['articles'])}")      # 49
# Spot-check the most-tested article
print(facts['articles']['21'])                    # title 'Protection of life and personal liberty'
print(facts['articles']['368'])                   # 'Power to amend Constitution'
print(facts['acts']['RTI 2005']['year'])          # 2005
print(facts['schemes']['MGNREGA']['start_year'])  # 2005
```

| Check | Pass criterion |
|---|---|
| Article 21 title | `Protection of life and personal liberty` |
| Article 368 | `Power to amend Constitution` |
| RTI 2005 year | 2005 |
| MGNREGA start | 2005 |

---

## Layer 5 — Negative tests (~5 min, build confidence in the failure modes)

Each demonstrates the system fails loudly when it should.

### 5.1 Determinism is real, not coincidence

```
make freeze                       # produces SHA-A
mv data/eval_set.parquet /tmp/a.parquet; mv data/eval_set.sha256 /tmp/a.sha
make freeze                       # produces SHA-B
diff /tmp/a.sha data/eval_set.sha256
```
**Pass:** `diff` shows no difference.

### 5.2 Different seed → different output

```
python scripts/freeze_eval_set.py --seed 1 --out /tmp/seed1.parquet
python scripts/freeze_eval_set.py --seed 2 --out /tmp/seed2.parquet
diff <(sha256sum /tmp/seed1.parquet | cut -d' ' -f1) \
     <(sha256sum /tmp/seed2.parquet | cut -d' ' -f1)
```
**Pass:** `diff` reports a difference (different seeds must yield different samples).

### 5.3 Leakage assertion actually trips

Hand-inject a leak and run the corpus builder:
```python
import pandas as pd
ev = pd.read_parquet('data/eval_set.parquet')
ft = pd.read_parquet('data/ft_corpus.parquet')
# Force one eval ID into the FT corpus
ft.loc[len(ft)] = {
    'pair_id': ev.iloc[0]['question_id'], 'task': 'A', 'language': 'en',
    'source_db': 'upscdev', 'source_table': 'prelims_pyq_questions',
    'instruction': '[TASK=A] ...', 'input': '{}', 'output': '{}'
}
ft.to_parquet('/tmp/contaminated_ft.parquet')

# Now invoke the assertion directly
import sys; sys.path.insert(0, 'scripts')
from build_ft_corpus import assert_no_leakage
eval_ids = set(ev['question_id'])
ft_ids = set(pd.read_parquet('/tmp/contaminated_ft.parquet')['pair_id'])
try:
    assert_no_leakage(eval_ids, ft_ids)
    print("FAIL: should have raised")
except AssertionError as e:
    print(f"PASS: {e}")
```
**Pass:** prints `PASS: LEAKAGE: 1 eval IDs in FT corpus: [...]`.

### 5.4 verify-env fails loudly when API keys missing

```
unset ANTHROPIC_API_KEY GEMINI_API_KEY GOOGLE_API_KEY
make verify-env
echo "exit: $?"
```
**Pass:** exits 1; prints `ANTHROPIC_API_KEY not set` and `GEMINI_API_KEY (or GOOGLE_API_KEY) not set`.

### 5.5 build-ft-corpus refuses to run without eval_set.parquet

```
mv data/eval_set.parquet /tmp/eval_backup.parquet
python scripts/build_ft_corpus.py
echo "exit: $?"
mv /tmp/eval_backup.parquet data/eval_set.parquet
```
**Pass:** non-zero exit; clear `FileNotFoundError` for `data/eval_set.parquet`.

---

## Acceptance criteria (full Phase 1+2 sign-off)

Phases 1, 2, 3 are **ready to hand off to Phase 4 (FT)** when all of the following hold:

- [ ] Layer 0: `make snapshot` clean; 8 tables + `_snapshot_meta` in `data/prayas_local.sqlite`; SHA sidecar present
- [ ] Layer 1: `make verify-env` clean
- [ ] Layer 2.1: `make build-facts` clean, sidecar present
- [ ] Layer 2.2: `make freeze` produces 2,000 rows in the expected per-task split
- [ ] Layer 2.3: `make build-ft-corpus` produces ≥ 20K pairs with `leakage check: PASS`
- [ ] Layer 2.4: `make probe-hindi` clean for both base models; `make gate-hindi` emits a verdict (PASS or FAIL — either is acceptable; FAIL just changes reporting downstream)
- [ ] Layer 3: `pytest` reports `5 passed`
- [ ] Layer 4.1: Eval-set schema, counts, languages, stratum diversity, JSON parses, unique IDs — all green
- [ ] Layer 4.2: FT-corpus schema, leakage defense-in-depth, instruction format, sample-row eyeball — all green
- [ ] Layer 4.3: Facts spot-checks return expected canonical values
- [ ] Layer 5.1: Determinism re-confirmed
- [ ] Layer 5.2: Seed sensitivity confirmed
- [ ] Layer 5.3: Leakage assertion confirmed to trip on injected overlap
- [ ] Layer 5.4-5.5: verify-env and build-ft-corpus fail loudly on missing prerequisites

Tracking artifact: append the run hashes to `data/eval_set.sha256` history (a git commit per Phase-2 run is sufficient — `.gitignore` keeps the parquet out, only the sidecar should be considered for tracking).

---

## What this strategy does NOT cover (deferred)

- **Inference latency / cost / format-validity** — Phase 5 strategy.
- **FT convergence + adapter-quality gate** — Phase 4 strategy.
- **Metric reproducibility across scorer-model versions** — Phase 6 strategy.
- **Dashboard rendering against fixture data** — Phase 7 strategy.
- **Streamlit live-query latency under load** — Phase 7 strategy.
- **Cross-condition statistical-test rigor** — auto-handled by `test_hypotheses.py` once it exists (Phase 6).

Each downstream phase gets its own narrow test strategy when its code lands; they all share the contract that the data-plane outputs (`eval_set.parquet`, `ft_corpus.parquet`, `upsc_facts.json`) are immutable and verified before they're consumed.
