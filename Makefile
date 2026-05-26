.PHONY: verify-env snapshot build-facts freeze build-ft-corpus export-corpus \
        probe-hindi gate-hindi \
        ft-gemma ft-qwen materialize-ft-split ft-gemma-aws ft-qwen-aws \
        validate-gemma validate-qwen \
        infer infer-c1a infer-c1b infer-c2 infer-c3 \
        score-tier1 aggregate test-hypotheses \
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
# Run only on a CUDA host (e.g. g6.xlarge). Expects data/ft_split/{train,valid}.jsonl
# already on disk — produced locally via `make materialize-ft-split` then S3-synced.
materialize-ft-split: build-ft-corpus
	$(PYTHON) scripts/run_ft.py --materialize-only

ft-qwen-aws:
	$(PYTHON) scripts/run_ft_aws.py --base Qwen/Qwen3.5-4B \
	                                --adapter-out adapters/qwen35-4b-upsc-v1

ft-gemma-aws:
	$(PYTHON) scripts/run_ft_aws.py --base google/gemma-4-E4B-it \
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

test:
	$(PYTHON) -m pytest tests/ -v

clean:
	find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache
