.DEFAULT_GOAL := help

.PHONY: help
help: ## show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

.PHONY: sync
sync: ## install the venv
	# aarch64 pulls ~2 GB of CUDA wheels; uv's default 30 s timeout aborts the download
	UV_HTTP_TIMEOUT=600 uv sync

.PHONY: precommit-hook
precommit-hook: ## install the pre-push hook that runs `make precommit`
	@common_dir="$$(git rev-parse --git-common-dir 2>/dev/null)"; \
	hooks_path="$$(git config --get core.hooksPath 2>/dev/null)"; \
	if [ -n "$$common_dir" ] && [ -z "$$hooks_path" ]; then \
	mkdir -p "$$common_dir/hooks" && printf '#!/bin/sh\nmake precommit\n' > "$$common_dir/hooks/pre-push" && chmod +x "$$common_dir/hooks/pre-push"; \
	fi

.PHONY: fmt
fmt: ## format src and tests
	uvx ruff format src tests

.PHONY: lint
lint: ## lint src and tests
	uvx ruff check src tests

.PHONY: tests
tests: ## run the test suite (CPU, no GPU needed)
	uv run --with pytest pytest -W ignore tests/

.PHONY: precommit
precommit: ## sync + hook + fmt + lint + tests
	$(MAKE) sync
	$(MAKE) precommit-hook
	$(MAKE) fmt
	$(MAKE) lint
	$(MAKE) tests

.PHONY: envs
envs: ## list every registered task
	uv run list-envs

.PHONY: smoke
smoke: ## 64 envs, 5 iters -- run this before every long run
	uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5 $(ARGS)

.PHONY: train
train: ## 4096 envs, 6000 iters, ~1-2 h for a usable gait
	uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max_iterations 6000 $(ARGS)

.PHONY: play
play: ## view a checkpoint in the sim; CKPT=logs/rsl_rl/velocity/<run>/model_3000.pt
	@[ -n "$(CKPT)" ] || { echo "CKPT=<path/to/model_3000.pt> required"; exit 2; }
	uv run play Mjlab-Velocity-Flat-MicroDuck --checkpoint-file "$(CKPT)" $(ARGS)

.PHONY: export
export: ## checkpoint -> weights/output.onnx, obs normalizer baked in; CKPT=3000
	@[ -n "$(CKPT)" ] || { echo "CKPT=<iteration> required (e.g. CKPT=3000)"; exit 2; }
	uv run export Mjlab-Velocity-Flat-MicroDuck --checkpoint "$(CKPT)" $(ARGS)

.PHONY: infer
infer: ## CPU MuJoCo deployment rehearsal of weights/output.onnx
	uv run infer --walking weights/output.onnx --new-cmd-obs $(ARGS)
