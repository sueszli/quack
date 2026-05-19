# HF Jobs training

Train mjlab-microduck on Hugging Face's managed GPUs. Auth is the cached HF
token from `hf auth login` — no GitHub PAT, no Docker registry to manage.

## One-time setup

```fish
uv tool install huggingface_hub   # installs the `hf` CLI globally
hf auth login                     # cached token used by everything below
# optional: forward your wandb key into jobs
set -x WANDB_API_KEY <your-key>
```

## Submit a run

```fish
uv run scripts/hf/train_hf.py Mjlab-VelStand-Flat-MicroDuck \
    --env.scene.num-envs 4096 --agent.max-iterations 5000
```

Everything after the task id is forwarded to `uv run train` inside the job.

Useful flags:
- `--flavor l4x1` (default) / `a10g-large` / `a100-large` — see `hf jobs hardware`
- `--timeout 12h` (default) — job is killed past this
- `--detach` — submit and return immediately (default streams logs)
- `--dry-run` — build tarball, print the `hf jobs run` command, do not submit
- `--run-name <tag>` — overrides the auto-generated `<task>-<timestamp>` name

## What happens under the hood

1. `git ls-files` snapshots tracked + uncommitted files → `src-<stamp>.tar.gz`.
2. Tarball is uploaded to private dataset `<user>/mjlab-microduck-src`.
3. Private model repo `<user>/<run-name>` is created for checkpoints.
4. `hf jobs run` launches a container that:
   - installs `uv`, extracts the tarball, runs `uv sync`,
   - starts `scripts/hf/uploader.py` in background (watches `logs/rsl_rl/**/model_*.pt`, pushes every 60s),
   - runs `uv run train <task> <args>`,
   - does a final one-shot upload on exit.
5. wandb (if `WANDB_API_KEY` is set locally) is forwarded as a secret — runs show up live in your wandb project.

## Browsing checkpoints

The submitter prints `https://huggingface.co/<user>/<run-name>` at start; new
`.pt` files appear there during training.

## Managing jobs

```fish
hf jobs ps              # list
hf jobs logs <job_id>   # tail
hf jobs cancel <job_id>
```
