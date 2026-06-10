# Deploy to Streamlit Cloud

## Prerequisites

- Repo pushed to GitHub (private OK — Streamlit Cloud can read private repos via your GitHub auth).
- The `results/*.parquet` files committed to `main` (they are — verified
  via `git status`, total ~53 MB across 6 parquets).

## One-time setup

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with the
   GitHub account that owns this repo (or has read access).
2. Click **New app** → pick the repo (`prayas-ai/SLM` or wherever it
   lives) → branch `main`.
3. Fill in:
   - **Main file path**: `dashboard/app.py`
   - **App URL**: `prayas-upsc-slm-v1` (or whatever subdomain you want)
   - **Python version**: 3.12
   - **Requirements file**: `dashboard/requirements.txt` (Streamlit Cloud
     auto-detects this when it's adjacent to the main script).
4. Click **Deploy**. First build ≈ 3-5 min (installs streamlit + pandas +
   pyarrow + numpy, then mounts the repo).

## Per-deploy refresh

- **Code changes**: push to `main`. Streamlit Cloud watches the branch
  and auto-redeploys within ~30 s of the push.
- **Data refresh** (when you regenerate `results/*.parquet`): commit the
  new parquets and push. Same auto-redeploy. `@st.cache_data(ttl=3600)`
  on the loaders means stale cache clears within an hour even without a
  restart, but a fresh deploy is cleanest.

## What works on Cloud vs locally

| Page | Cloud | Local | Notes |
|---|---|---|---|
| Home (`app.py`) | ✅ | ✅ | Headline + significance column |
| Results | ✅ | ✅ | Full Tier-1 metric tables |
| Significance | ✅ | ✅ | BH-FDR tests + 230-cell heatmap |
| Per-Row Drill | ✅ | ✅ | Side-by-side prediction comparison |
| Playground | ❌ graceful-stop | ✅ | Needs MLX + adapters + Gemini key — only runs locally on Apple Silicon |

## Secrets (none needed for v1 read-only deploy)

The deployed dashboard reads only `results/*.parquet` and `data/eval_set.parquet`
— no API keys, no DB, no model files. No secrets to configure on Streamlit
Cloud for the deploy.

If you later want to enable the **Playground** on Cloud (currently
deferred — would need a CPU-only inference path, since Streamlit Cloud
has no GPU), you'd add `GEMINI_API_KEY` under **App settings → Secrets**
as:

```toml
GEMINI_API_KEY = "..."
```

## Known gotchas

1. **Streamlit Cloud file-size limit is 1 GB per app**. The repo's
   `data/eval_set.parquet` is ~1 MB and the 6 `results/*.parquet` total
   ~53 MB, so we're well under. If `predictions.parquet` (52 MB) ever
   crosses 100 MB, GitHub will warn and we should consider Git LFS.
2. **`data/cpt_raw/`, `data/cpt_text/`, `data/cpt_clean*/` are gitignored**
   (not shipped to Cloud, not needed by the dashboard). Good.
3. **Adapter files (`adapters/*-v1-*/`) are gitignored too** — the
   Playground page never runs on Cloud, so it doesn't need them.
4. **`requirements-aws.txt` is for the training stack** (transformers,
   peft, bitsandbytes, trl). Streamlit Cloud auto-picks
   `dashboard/requirements.txt` because it's adjacent to the main script
   path — the AWS one is correctly ignored.

## Verification after deploy

After the first deploy completes:

1. Open the public URL.
2. Confirm the home-page **Significance column** shows 4 ✓ rows (Tasks
   B, E, F, G), one significant loss (A: −0.264), and one borderline
   (C: +0.213, p=0.0596, ·).
3. Click **Results** → confirm per-language stratum tables render.
4. Click **Significance** → confirm the 230-cell heatmap loads. (Note:
   the delta signs on this page are currently flipped — separate bug in
   `scripts/test_hypotheses.py`, not introduced by this deploy.)
5. Click **Per-Row Drill** → pick any `question_id` and verify the
   four-condition comparison renders.
6. Click **Playground** → should show the friendly "Playground
   unavailable in this deployment" notice, no crash.

## Rollback

`git revert <bad-commit> && git push` — Streamlit Cloud redeploys on
the next push automatically.
