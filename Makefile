RUFF_FLAGS := --line-length 5000 --target-version py312
RUFF_CHECK_FLAGS := $(RUFF_FLAGS) --extend-select I --ignore F403,F405,F821,E731,E402,PLE0643,B008,UP040,RUF016,PLC0206,SIM115

.PHONY: precommit-hook
precommit-hook:
	@common_dir="$$(git rev-parse --git-common-dir 2>/dev/null)"; \
	hooks_path="$$(git config --get core.hooksPath 2>/dev/null)"; \
	if [ -n "$$common_dir" ] && [ -z "$$hooks_path" ]; then \
	mkdir -p "$$common_dir/hooks" && printf '#!/bin/sh\nmake precommit\n' > "$$common_dir/hooks/pre-push" && chmod +x "$$common_dir/hooks/pre-push"; \
	fi

.PHONY: fmt
fmt:
	uvx ruff check --fix $(RUFF_CHECK_FLAGS) src tests
	uvx ruff format $(RUFF_FLAGS) src tests

.PHONY: lint
lint:
	uvx ruff check $(RUFF_CHECK_FLAGS) src tests
	uv run --with vulture vulture --min-confidence 80 src tests
	uv run --with pyright pyright src

# Regenerate the checked-in type stubs in typings/ after a mujoco or bam bump.
# mujoco and bam are C-extension / untyped packages; without stubs pyright sees
# `object` for MjModel and Incomplete for the BAM actuator, which is ~300 of the
# original ~470 errors. Hand-applied fixes on top of stubgen output are marked
# with comments in the .pyi files — re-apply them after regenerating.
.PHONY: stubs
stubs:
	uv run --with mypy stubgen -p mujoco -p bam -o typings

.PHONY: tests
tests:
	uv run --with pytest pytest -W ignore tests/

.PHONY: smoke
smoke:
	uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5

.PHONY: precommit
precommit:
	uv sync
	$(MAKE) precommit-hook
	$(MAKE) fmt
	$(MAKE) lint
	$(MAKE) tests
