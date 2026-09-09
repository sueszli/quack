# Formatting / lint config lives in [tool.ruff] in pyproject.toml so editors,
# pre-commit and `make` all agree.
#
# Workflow targets take variables, e.g.
#   make train TASK=Mjlab-StandUp-Flat-MicroDuck ENVS=4096
#   make play RUN=entity/project/run_id
#   make publish RUN=entity/project/run_id CHECKPOINT=3000 REPO=user/microduck-bow KIND=episodic DURATION=4.0

TASK ?= Mjlab-Velocity-Flat-MicroDuck
ENVS ?= 4096
ONNX ?= output.onnx
KIND ?= episodic
ARGS ?=

.DEFAULT_GOAL := help

.PHONY: help
help:
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

.PHONY: sync
sync: ## Install/refresh the environment (UV_HTTP_TIMEOUT is raised for slow CUDA wheel downloads)
	UV_HTTP_TIMEOUT=$${UV_HTTP_TIMEOUT:-600} uv sync

.PHONY: envs
envs: ## Print the live task registry
	uv run list-envs

.PHONY: train
train: ## Train TASK on ENVS envs (ARGS is appended, e.g. ARGS="--agent.max_iterations 2000")
	uv run train $(TASK) --env.scene.num-envs $(ENVS) $(ARGS)

.PHONY: resume
resume: ## Resume TASK from CHECKPOINT (a model_XXXX.pt file name)
	@[ -n "$(CHECKPOINT)" ] || { echo "set CHECKPOINT=model_29999.pt"; exit 1; }
	uv run train $(TASK) --env.scene.num-envs $(ENVS) --agent.run-name resume --agent.load-checkpoint $(CHECKPOINT) --agent.resume True $(ARGS)

.PHONY: play
play: ## Watch TASK's policy from wandb RUN in the viewer (VIEWER=viser --num-envs 1 over an SSH tunnel on a remote GPU box)
	@[ -n "$(RUN)" ] || { echo "set RUN=entity/project/run_id"; exit 1; }
	uv run play $(TASK) --wandb-run-path $(RUN) $(if $(VIEWER),--viewer $(VIEWER),) $(ARGS)

.PHONY: export
export: ## Export TASK's wandb RUN to ONNX with the obs normalizer baked in
	@[ -n "$(RUN)" ] || { echo "set RUN=entity/project/run_id"; exit 1; }
	uv run export $(TASK) --wandb-run-path $(RUN) $(ARGS)

.PHONY: infer
infer: ## Drive ONNX in CPU MuJoCo with the keyboard (deployment rehearsal)
	uv run infer --walking $(ONNX) $(ARGS)

.PHONY: publish
publish: ## Upload TASK's wandb RUN to REPO on the Hub (KIND=episodic needs DURATION, perpetual needs SLOT or UNWIND)
	@[ -n "$(RUN)" ] || { echo "set RUN=entity/project/run_id"; exit 1; }
	@[ -n "$(REPO)" ] || { echo "set REPO=user/microduck-<name>"; exit 1; }
	uv run publish --task $(TASK) --wandb-run-path $(RUN) $(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),) --repo $(REPO) --kind $(KIND) $(if $(DURATION),--duration-s $(DURATION),) $(if $(SLOT),--slot $(SLOT),) $(if $(UNWIND),--unwind-s $(UNWIND),) $(ARGS)

.PHONY: publish-dry
publish-dry: ## Same as publish, but only shows what would be uploaded
	$(MAKE) publish ARGS="$(ARGS) --dry-run"

.PHONY: fmt
fmt: ## Format src/ and tests/
	uvx ruff format src tests

.PHONY: lint
lint: ## Lint src/ and tests/
	uvx ruff check src tests

.PHONY: tests
tests: ## Run the CPU-only test suite
	uv run --with pytest pytest -W ignore tests/

.PHONY: smoke
smoke: ## 5-iteration, 64-env smoke test of TASK — always run before a long training run
	uv run train $(TASK) --env.scene.num-envs 64 --agent.max_iterations 5

.PHONY: precommit-hook
precommit-hook:
	@common_dir="$$(git rev-parse --git-common-dir 2>/dev/null)"; \
	hooks_path="$$(git config --get core.hooksPath 2>/dev/null)"; \
	if [ -n "$$common_dir" ] && [ -z "$$hooks_path" ]; then \
	mkdir -p "$$common_dir/hooks" && printf '#!/bin/sh\nmake precommit\n' > "$$common_dir/hooks/pre-push" && chmod +x "$$common_dir/hooks/pre-push"; \
	fi

# Note: `lint` is not part of precommit yet — master currently has 151 ruff
# findings. It is wired up here so the backlog can be cleared incrementally;
# add it to precommit once `make lint` is clean.
.PHONY: precommit
precommit: ## Sync, install the pre-push hook, run the tests
	uv sync
	$(MAKE) precommit-hook
	$(MAKE) tests
