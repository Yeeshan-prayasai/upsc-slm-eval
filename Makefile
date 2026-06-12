.PHONY: verify-env snapshot build-facts freeze build-ft-corpus export-corpus \
        probe-hindi gate-hindi \
        ft-gemma ft-qwen materialize-ft-split verify-aws-env ft-gemma-aws ft-qwen-aws \
        validate-gemma validate-qwen \
        infer infer-c1a infer-c1b infer-c2 infer-c3 \
        score-tier1 aggregate test-hypotheses render-report \
        dashboard \
        acquire-ncert acquire-local-db acquire-slimpajama acquire-ipcc acquire-arc2 \
        acquire-prs acquire-pmf-ias acquire-mrunal acquire-curated acquire-cpt-data \
        cpt-ocr cpt-clean cpt-leakage cpt-tokenize build-cpt-corpus build-sft-corpus \
        cpt-gemma cpt-qwen sft-gemma-v2 sft-qwen-v2 \
        smoke-cpt-gemma smoke-cpt-qwen \
        preflight-v2 ablation-gemma ablation-qwen \
        test clean

RUN_ID ?= $(shell date +%Y%m%d)
PYTHON ?= .venv/bin/python

verify-env:
	$(PYTHON) scripts/verify_env.py

snapshot:
	$(PYTHON) scripts/snapshot_to_local.py

build-facts:
	$(PYTHON) scripts/build_upsc_facts.py

freeze: snapshot
	$(PYTHON) scripts/freeze_eval_set.py --seed 20260514 --out data/eval_set.parquet

build-ft-corpus: freeze
	$(PYTHON) scripts/build_ft_corpus.py --eval data/eval_set.parquet --out data/ft_corpus.parquet

export-corpus:
	$(PYTHON) scripts/export_corpus.py --in data/ft_corpus.parquet
	$(PYTHON) scripts/export_corpus.py --in data/eval_set.parquet

probe-hindi: snapshot
	$(PYTHON) scripts/run_hindi_probe.py --model mlx-community/gemma-4-e4b-it-4bit
	$(PYTHON) scripts/run_hindi_probe.py --model mlx-community/Qwen3.5-4B-MLX-4bit

gate-hindi: probe-hindi
	$(PYTHON) scripts/gate_hindi.py

ft-gemma: build-ft-corpus
	$(PYTHON) scripts/run_ft.py --base mlx-community/gemma-4-e4b-it-4bit \
	                         --adapter-out adapters/gemma4-e4b-upsc-v1

ft-qwen: build-ft-corpus
	$(PYTHON) scripts/run_ft.py --base mlx-community/Qwen3.5-4B-MLX-4bit \
	                         --adapter-out adapters/qwen35-4b-upsc-v1

# --- AWS path (NVIDIA GPU; PyTorch + peft + bnb) ---------------------------
# Run only on a CUDA host (e.g. g6e.xlarge). Expects data/ft_split/{train,valid}.jsonl
# already on disk — produced locally via `make materialize-ft-split` then SCP'd up.
materialize-ft-split: build-ft-corpus
	$(PYTHON) scripts/run_ft.py --materialize-only

# Pre-flight environment verifier — hard-fails before any training starts if
# the GPU isn't visible, bnb doesn't load, HF isn't authenticated, training
# data is missing, or the disk is too tight. Idempotent.
verify-aws-env:
	$(PYTHON) scripts/verify_aws_env.py

# -u: unbuffered stdout/stderr so loss/grad_norm prints flush per-line when run
# under nohup (no TTY). Without it, Python block-buffers stdout, batching loss
# output into 8 KB chunks — hides real-time training progress.
ft-qwen-aws: verify-aws-env
	$(PYTHON) -u scripts/run_ft_aws.py --base Qwen/Qwen3.5-4B \
	                                   --adapter-out adapters/qwen35-4b-upsc-v1

ft-gemma-aws: verify-aws-env
	$(PYTHON) -u scripts/run_ft_aws.py --base google/gemma-4-E4B-it \
	                                   --adapter-out adapters/gemma4-e4b-upsc-v1

validate-gemma:
	$(PYTHON) scripts/validate_adapter.py --base mlx-community/gemma-4-e4b-it-4bit \
	                                   --adapter adapters/gemma4-e4b-upsc-v1

validate-qwen:
	$(PYTHON) scripts/validate_adapter.py --base mlx-community/Qwen3.5-4B-MLX-4bit \
	                                   --adapter adapters/qwen35-4b-upsc-v1

infer-c1a:
	$(PYTHON) scripts/run_inference.py --condition C1a --run-id $(RUN_ID)

infer-c1b:
	$(PYTHON) scripts/run_inference.py --condition C1b --run-id $(RUN_ID)

infer-c2:
	$(PYTHON) scripts/run_inference.py --condition C2 --run-id $(RUN_ID)

infer-c3:
	$(PYTHON) scripts/run_inference.py --condition C3 --run-id $(RUN_ID)

infer: infer-c1a infer-c1b infer-c2 infer-c3

score-tier1:
	$(PYTHON) scripts/score_tier1.py

aggregate: score-tier1
	$(PYTHON) scripts/aggregate.py

test-hypotheses: score-tier1
	$(PYTHON) scripts/test_hypotheses.py

# Renders the §6 + §7 tables of experiment-report.md from aggregate +
# hypothesis_tests + stratum_heatmap. Idempotent; run --check first to
# see which (condition, task, metric) cells would fill.
render-report: aggregate test-hypotheses
	$(PYTHON) scripts/render_report.py --check
	$(PYTHON) scripts/render_report.py

# Streamlit dashboard — reporting view + interactive playground.
# Reads results/*.parquet for the reporting pages; the playground page lazy-loads
# MLX adapters (one at a time, ~5 GB peak) and calls Gemini API for C2/C3.
# Default port 8501; override with `make dashboard PORT=8502`.
# Sources .env if present so GEMINI_API_KEY etc. flow into the Streamlit process
# without requiring `export` in every shell.
PORT ?= 8501
dashboard:
	set -a; [ -f .env ] && . ./.env; set +a; \
	  $(PYTHON) -m streamlit run dashboard/app.py --server.port $(PORT)

# ---------- Corpus acquisition ----------
# Each target writes to data/cpt_raw/<source>/ with a per-source manifest.jsonl.
# Re-runs are idempotent — items already in the manifest are skipped.
# Override defaults inline: `make acquire-ncert NCERT_ARGS="--only kegy1dd --dry-run"`.
NCERT_ARGS ?=
LOCAL_DB_ARGS ?=
SLIMPAJAMA_ARGS ?= --target-tokens 700000000
IPCC_ARGS ?=

acquire-ncert:
	$(PYTHON) -m training.data.acquire.ncert $(NCERT_ARGS)

acquire-local-db: snapshot
	$(PYTHON) -m training.data.acquire.local_db $(LOCAL_DB_ARGS)

acquire-slimpajama:
	$(PYTHON) -m training.data.acquire.slimpajama $(SLIMPAJAMA_ARGS)

acquire-ipcc:
	$(PYTHON) -m training.data.acquire.ipcc $(IPCC_ARGS)

ARC2_ARGS ?=
PRS_ARGS ?=
PMF_IAS_ARGS ?=
MRUNAL_ARGS ?=

acquire-arc2:
	$(PYTHON) -m training.data.acquire.arc2 $(ARC2_ARGS)

acquire-prs:
	$(PYTHON) -m training.data.acquire.prs $(PRS_ARGS)

acquire-pmf-ias:
	$(PYTHON) -m training.data.acquire.pmf_ias $(PMF_IAS_ARGS)

acquire-mrunal:
	$(PYTHON) -m training.data.acquire.mrunal $(MRUNAL_ARGS)

# Run a curated YAML by short name. Example: `make acquire-curated CURATED=economic_survey`
CURATED ?= economic_survey
acquire-curated:
	$(PYTHON) -m training.data.acquire.curated --source $(CURATED)

# Run every implemented source in sequence. Heavy.
# Approx wall-clock at default rate:
#   NCERT ~30m | ARC2 ~10m | IPCC ~1m | Eco Survey ~3m
#   local-DB ~3m | PRS ~15m | PMF IAS ~10m | Mrunal ~25m
#   SlimPajama ~2-3h
acquire-cpt-data: acquire-ncert acquire-arc2 acquire-ipcc acquire-curated \
                  acquire-local-db acquire-prs acquire-pmf-ias acquire-mrunal \
                  acquire-slimpajama

# ---------- Corpus build (OCR → clean → leakage → tokenize+pack) ----------
# Each stage runs standalone for re-runs / smoke tests; build-cpt-corpus
# chains them in order with the leakage gate as a hard barrier.
PHASE1_SOURCE ?=
PHASE1_WORKERS ?= 4
PHASE1_TOKENIZER ?= both
PHASE1_FLAGS ?=

cpt-ocr:
	$(PYTHON) -m training.data.ocr $(if $(PHASE1_SOURCE),--source $(PHASE1_SOURCE)) --workers $(PHASE1_WORKERS)

cpt-clean:
	$(PYTHON) -m training.data.clean $(if $(PHASE1_SOURCE),--source $(PHASE1_SOURCE))

cpt-leakage:
	$(PYTHON) -m training.data.leakage $(if $(PHASE1_SOURCE),--source $(PHASE1_SOURCE))

cpt-tokenize:
	$(PYTHON) -m training.data.tokenize_pack --tokenizer $(PHASE1_TOKENIZER)

build-cpt-corpus:
	$(PYTHON) -m training.data.build_cpt_corpus $(if $(PHASE1_SOURCE),--source $(PHASE1_SOURCE)) \
	    --workers $(PHASE1_WORKERS) --tokenizer $(PHASE1_TOKENIZER) $(PHASE1_FLAGS)

# v2 SFT corpus: filter v1 ft_corpus.parquet to EN, join target_word_count
# from `pyqs.word_count` for Task B, split train/valid, write JSONL.
SFT_BUILD_FLAGS ?=
build-sft-corpus:
	$(PYTHON) -m training.data.build_sft_corpus $(SFT_BUILD_FLAGS)

# ---------- CPT then SFT training ----------
# Run on a CUDA host (L40S/A10G). Each target loads the per-model + runtime
# YAML, builds the QLoRA+LoRA model, and starts training with auto-resume
# from the latest checkpoint in --output-dir.
CPT_FLAGS ?=
SFT_FLAGS ?=

cpt-gemma: verify-aws-env
	$(PYTHON) -u -m training.orchestration.run_cpt \
	    --config training/configs/cpt_gemma.yaml \
	    --runtime training/configs/runtime_l40s.yaml $(CPT_FLAGS)

cpt-qwen: verify-aws-env
	$(PYTHON) -u -m training.orchestration.run_cpt \
	    --config training/configs/cpt_qwen.yaml \
	    --runtime training/configs/runtime_l40s.yaml $(CPT_FLAGS)

sft-gemma-v2: verify-aws-env
	$(PYTHON) -u -m training.orchestration.run_sft \
	    --config training/configs/sft_gemma.yaml \
	    --runtime training/configs/runtime_l40s.yaml $(SFT_FLAGS)

sft-qwen-v2: verify-aws-env
	$(PYTHON) -u -m training.orchestration.run_sft \
	    --config training/configs/sft_qwen.yaml \
	    --runtime training/configs/runtime_l40s.yaml $(SFT_FLAGS)

# Smoke-test the full CPT chain (20 steps) on a CUDA host before
# committing to a 55-hour run. Runs `training.orchestration.smoke_cpt`
# which: builds QLoRA + LoRA model, runs 20 grad-accum steps over a
# 1K-sequence corpus (real or synthetic), saves+reloads adapter,
# reports peak VRAM. Aborts with a clear error if any stage fails.
smoke-cpt-gemma: verify-aws-env
	$(PYTHON) -u -m training.orchestration.smoke_cpt --model gemma

smoke-cpt-qwen: verify-aws-env
	$(PYTHON) -u -m training.orchestration.smoke_cpt --model qwen

# Standalone pre-flight leakage gate — verifies eval-set non-contamination
# in the tokenized CPT corpus + per-source manifests. Same gate that
# `run_cpt.py` and `run_sft.py` run automatically at startup (skippable
# only via --skip-preflight, which is debug-only).
preflight-v2:
	$(PYTHON) -m training.eval.preflight

# 6-cell ablation grid driver (methodology §8). Trains each cell
# sequentially on a single GPU, writes per-cell results under
# runs/ablation/<cell>__<model>/. Resume-aware via done.txt markers.
ABLATION_FLAGS ?=
ablation-gemma: verify-aws-env
	$(PYTHON) -u -m training.orchestration.run_ablation --model gemma $(ABLATION_FLAGS)

ablation-qwen: verify-aws-env
	$(PYTHON) -u -m training.orchestration.run_ablation --model qwen $(ABLATION_FLAGS)

# v2 inference + scoring — reuse the v1 pipeline against a v2 adapter.
# ADAPTER = merged HF dir (run scripts/merge_adapter.py on the trained
# LoRA first), e.g. ADAPTER=adapters/gemma4-e4b-upsc-v2-sft/final-merged
infer-v2-gemma:
	@test -n "$(ADAPTER)" || (echo "usage: make infer-v2-gemma ADAPTER=<merged-hf-dir>" && exit 1)
	$(PYTHON) -u scripts/run_inference.py --condition C1a --backend hf \
		--adapter-dir $(ADAPTER) --batch-size 8 --run-id $(RUN_ID)

infer-v2-qwen:
	@test -n "$(ADAPTER)" || (echo "usage: make infer-v2-qwen ADAPTER=<merged-hf-dir>" && exit 1)
	$(PYTHON) -u scripts/run_inference.py --condition C1b --backend hf \
		--adapter-dir $(ADAPTER) --batch-size 8 --run-id $(RUN_ID)

score-v2: score-tier1
	$(PYTHON) scripts/aggregate.py
	$(PYTHON) scripts/test_hypotheses.py

test:
	$(PYTHON) -m pytest tests/ training/tests/ -v

clean:
	find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache
