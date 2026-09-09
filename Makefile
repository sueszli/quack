.PHONY: precommit-hook
precommit-hook:
	@common_dir="$$(git rev-parse --git-common-dir 2>/dev/null)"; \
	hooks_path="$$(git config --get core.hooksPath 2>/dev/null)"; \
	if [ -n "$$common_dir" ] && [ -z "$$hooks_path" ]; then \
	mkdir -p "$$common_dir/hooks" && printf '#!/bin/sh\nmake precommit\n' > "$$common_dir/hooks/pre-push" && chmod +x "$$common_dir/hooks/pre-push"; \
	fi

.PHONY: fmt
fmt:
	uvx ruff format src tests

.PHONY: lint
lint:
	uvx ruff check src tests

.PHONY: tests
tests:
	uv run --with pytest pytest -W ignore tests/

.PHONY: precommit
precommit:
	uv sync
	$(MAKE) precommit-hook
	$(MAKE) fmt
	$(MAKE) lint
	$(MAKE) tests

.PHONY: sync
sync:
	UV_HTTP_TIMEOUT=600 uv sync

.PHONY: envs
envs:
	uv run list-envs

.PHONY: smoke
smoke:
	uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5

.PHONY: train
train:
	uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096 $(ARGS)

.PHONY: play
play:
	uv run play Mjlab-Velocity-Flat-MicroDuck --wandb-run-path $(RUN) $(ARGS)

.PHONY: export
export:
	uv run export Mjlab-Velocity-Flat-MicroDuck --wandb-run-path $(RUN) $(ARGS)

.PHONY: infer
infer:
	uv run infer --walking $(ONNX) $(ARGS)
