"""Submit a mjlab-microduck training run as a Hugging Face Job.

Usage:
    uv run scripts/hf/train_hf.py Mjlab-VelStand-Flat-MicroDuck \
        --env.scene.num-envs 4096 --agent.max-iterations 5000

Anything after the task id is forwarded to `uv run train` inside the job.

Auth: uses the cached HF token from `hf auth login` (read via `HfApi.whoami`).
Source: a `git archive HEAD` tarball is uploaded to a private HF dataset repo
and mounted read-only inside the job. Checkpoints are pushed by a watcher
running alongside training to a private HF model repo.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"
DEFAULT_FLAVOR = "l4x1"
DEFAULT_TIMEOUT = "12h"

# Bootstrap script run inside the container. Single-quoted so `$VAR` is
# expanded by the container shell, not by the local shell that calls hf.
BOOTSTRAP = r"""
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -qq -y --no-install-recommends git curl ca-certificates xz-utils >/dev/null
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
export PATH="/root/.local/bin:$PATH"

mkdir -p /work && cd /work
echo "[bootstrap] extracting source $SRC_TARBALL"
tar -xzf "/src/$SRC_TARBALL"

echo "[bootstrap] uv sync"
uv sync --no-progress

echo "[bootstrap] launching checkpoint uploader"
mkdir -p logs/rsl_rl
nohup uv run python scripts/hf/uploader.py > /tmp/uploader.log 2>&1 &
UPLOADER_PID=$!

echo "[bootstrap] starting training: uv run train $TRAIN_ARGS"
set +e
uv run train $TRAIN_ARGS
TRAIN_RC=$?
set -e

echo "[bootstrap] training exited with code $TRAIN_RC, final upload pass"
# kill watcher loop, then run one final upload pass synchronously
kill $UPLOADER_PID 2>/dev/null || true
CKPT_ONE_SHOT=1 uv run python scripts/hf/uploader.py || true

exit $TRAIN_RC
"""


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    return subprocess.run(cmd, check=True, **kw)


def _hf_user(api: HfApi) -> str:
    info = api.whoami()
    name = info.get("name") or info.get("email")
    if not name:
        raise RuntimeError(
            "Could not determine HF username. Run `hf auth login` first."
        )
    return name


def _build_tarball(out_path: Path) -> str:
    """Create a tarball of HEAD + uncommitted tracked changes. Returns short SHA."""
    sha = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT
    ).decode().strip()

    # Use `git ls-files` so we include tracked-but-modified files (working tree state)
    # but skip ignored junk (.venv, logs, *.onnx, wandb/, etc.).
    files = subprocess.check_output(
        ["git", "ls-files", "-co", "--exclude-standard"], cwd=REPO_ROOT
    ).decode().splitlines()

    with tarfile.open(out_path, "w:gz") as tar:
        for rel in files:
            p = REPO_ROOT / rel
            if p.exists() and p.is_file():
                tar.add(p, arcname=rel)
    return sha


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Submit a microduck training run to HF Jobs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("task", help="mjlab task id, e.g. Mjlab-VelStand-Flat-MicroDuck")
    ap.add_argument("--flavor", default=DEFAULT_FLAVOR, help="HF Jobs hardware flavor")
    ap.add_argument("--image", default=DEFAULT_IMAGE, help="Docker image to run in")
    ap.add_argument("--timeout", default=DEFAULT_TIMEOUT, help="Job max duration")
    ap.add_argument(
        "--run-name",
        default=None,
        help="Short tag for this run; defaults to task+timestamp",
    )
    ap.add_argument(
        "--detach", action="store_true",
        help="Submit and return immediately (do not stream logs).",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Build tarball and print the hf jobs command without submitting.",
    )
    ap.add_argument(
        "--src-repo",
        default=None,
        help="HF dataset repo for source tarballs. Defaults to <user>/mjlab-microduck-src",
    )
    ap.add_argument(
        "--ckpt-repo",
        default=None,
        help="HF model repo for checkpoints. Defaults to <user>/<run-name>",
    )
    args, train_args = ap.parse_known_args()

    # check `hf` CLI is available
    if shutil.which("hf") is None:
        print(
            "error: `hf` CLI not found. Install it with `uv tool install huggingface_hub` "
            "or `pip install huggingface_hub`.",
            file=sys.stderr,
        )
        return 1

    api = HfApi()
    try:
        user = _hf_user(api)
    except Exception as e:
        print(f"error: HF auth failed ({e}). Run `hf auth login`.", file=sys.stderr)
        return 1
    print(f"[hf] authenticated as {user}")

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"{args.task}-{stamp}".lower()
    src_repo = args.src_repo or f"{user}/mjlab-microduck-src"
    ckpt_repo = args.ckpt_repo or f"{user}/{run_name}"

    # 1. Build tarball
    with tempfile.TemporaryDirectory() as td:
        tar_path = Path(td) / f"src-{stamp}.tar.gz"
        print(f"[src] building tarball -> {tar_path.name}")
        sha = _build_tarball(tar_path)
        size_mb = tar_path.stat().st_size / 1e6
        print(f"[src] HEAD={sha}, {size_mb:.1f} MB")

        tar_filename = tar_path.name
        if not args.dry_run:
            # 2. Upload tarball
            api.create_repo(src_repo, repo_type="dataset", private=True, exist_ok=True)
            print(f"[src] uploading to dataset {src_repo}")
            api.upload_file(
                path_or_fileobj=str(tar_path),
                path_in_repo=tar_filename,
                repo_id=src_repo,
                repo_type="dataset",
            )

            # 3. Pre-create checkpoint repo (uploader also creates it, but doing
            # it here lets us print the URL immediately).
            api.create_repo(ckpt_repo, repo_type="model", private=True, exist_ok=True)

        train_args_str = " ".join(shlex.quote(a) for a in [args.task, *train_args])

        # 4. Build hf jobs command
        cmd: list[str] = [
            "hf", "jobs", "run",
            "--flavor", args.flavor,
            "--timeout", args.timeout,
            "--secrets", "HF_TOKEN",
            "-v", f"hf://datasets/{src_repo}:/src:ro",
            "-e", f"SRC_TARBALL={tar_filename}",
            "-e", f"CKPT_REPO={ckpt_repo}",
            "-e", f"TRAIN_ARGS={train_args_str}",
            "-e", f"GIT_SHA={sha}",
        ]
        # Forward WANDB_API_KEY if set locally
        if os.environ.get("WANDB_API_KEY"):
            cmd += ["--secrets", f"WANDB_API_KEY={os.environ['WANDB_API_KEY']}"]
            # also forward project/entity if user has them set
            for k in ("WANDB_PROJECT", "WANDB_ENTITY"):
                if os.environ.get(k):
                    cmd += ["-e", f"{k}={os.environ[k]}"]
        if args.detach:
            cmd.append("--detach")
        cmd += [args.image, "bash", "-c", BOOTSTRAP]

        if args.dry_run:
            print("[dry-run] would run:")
            print("  " + " ".join(shlex.quote(c) for c in cmd))
            return 0

        print(f"[ckpt] checkpoints -> https://huggingface.co/{ckpt_repo}")
        print(f"[job] submitting (flavor={args.flavor}, timeout={args.timeout})")
        _run(cmd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
