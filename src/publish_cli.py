# Uploads policy.onnx + a schema-2 manifest.json + README in the shape the microduck daemon loads.

from __future__ import annotations

import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NoReturn

import tyro

from . import publish_manifest as m
from .utils import data_path


@dataclass(frozen=True)
class PublishConfig:
    # Hub repo id, `<user-or-org>/microduck-<name>`. Created (private) if it does not exist.
    repo: str
    # episodic: runs `duration_s` and comes back on its own. perpetual: holds until told.
    kind: Literal["episodic", "perpetual"]

    # Task id to export from, e.g. Mjlab-PoliteBow-Flat-MicroDuck. Needs a checkpoint.
    task: str | None = None
    # Checkpoint iteration (model_<N>.pt) under logs/rsl_rl/<experiment_name>/. Default: the latest.
    checkpoint: int | None = None
    # An explicit path to a model_<N>.pt.
    checkpoint_file: str | None = None
    # An already-exported ONNX. Validated, not re-exported.
    onnx: str | None = None

    # What a client asks for (`robotctl robot do <name>`). Default: the repo's stem minus `microduck-`.
    name: str | None = None
    # One line. Default: the task id.
    description: str | None = None
    # episodic only: seconds it runs.
    duration_s: float | None = None
    # episodic only: a held button chains another run (roulade does, a kick does not).
    chain: bool = False
    # perpetual held pose (flamingo): seconds the daemon drives `idle` before handing back. Leave unset for a gait.
    unwind_s: float | None = None
    # perpetual gait: which slot it is for (walk, stand, ...). Display-only; drives the install hint.
    slot: Literal["walk", "stand", "sitstand", "ground_pick", "kick_left", "kick_right", "roulade"] | None = None
    # The twist that means 'stop doing the thing'. Zeros for every one-shot published so far.
    idle: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # The policy's own output scale, if it wants one. Default: the gait's.
    action_scale: float | None = None
    # The pose the policy expects to start from.
    entry_pose: str = "standing"
    # Prose for `command.twist` when the slots mean something (flamingo: '[flag, side, 0]').
    twist_help: str | None = None

    # Create the repo private (--no-private for public). Existing repos keep their visibility.
    private: bool = True
    # Overwrite an existing policy.onnx in the repo.
    force: bool = False
    # Tag the resulting revision, e.g. v1.
    tag: str | None = None
    # Run the network on plausible inputs and refuse NaNs before uploading.
    smoke: bool = True
    # Write policy.onnx, manifest.json and README.md to ./publish-<name>/ and stop.
    dry_run: bool = False
    # Export device. Default: cuda:0 if available, else cpu.
    device: str | None = None


def _fail(msg: str) -> NoReturn:
    print(f"[publish] error: {msg}", file=sys.stderr)
    sys.exit(2)


def _resolve_weights(cfg: PublishConfig, workdir: Path) -> tuple[Path, dict]:
    from_checkpoint = cfg.task is not None or cfg.checkpoint_file is not None
    if (cfg.onnx is None) == (not from_checkpoint):
        _fail("give exactly one source: --onnx <file>, or --task <id> with --checkpoint/--checkpoint-file")

    training: dict = {"repo": "pollen-robotics/microduck_rl", **m.git_provenance()}
    if cfg.onnx is not None:
        src = Path(cfg.onnx)
        training["source_file"] = src.name
        if cfg.task:
            training["task_id"] = cfg.task
        return src, training

    if cfg.task is None:
        _fail("--checkpoint-file needs --task <id> to build the env it was trained in")

    # Heavy imports only on this path: the --onnx path must work without a GPU or mjlab's registry.
    import mjlab.tasks  # noqa: F401

    from .export import ExportConfig, run_export

    out = workdir / m.POLICY_FILE
    result = run_export(cfg.task, ExportConfig(onnx_file=str(out), checkpoint=cfg.checkpoint, checkpoint_file=cfg.checkpoint_file, num_envs=1, device=cfg.device))
    training["task_id"] = cfg.task
    if result.checkpoint_iteration is not None:
        training["checkpoint"] = result.checkpoint_iteration
    return result.onnx_path, training


def _default_name(repo: str) -> str:
    stem = repo.rsplit("/", 1)[-1]
    return stem.removeprefix("microduck-").removeprefix("microduck_") or stem


def run(cfg: PublishConfig) -> int:
    if "/" not in cfg.repo:
        _fail("--repo must be `<user-or-org>/<name>`")
    name = cfg.name or _default_name(cfg.repo)

    workdir = Path(tempfile.mkdtemp(prefix="publish-", dir=str(data_path("publish"))))
    try:
        onnx_path, training = _resolve_weights(cfg, workdir)
        shape = m.check_onnx(onnx_path)
        print(f"[publish] {onnx_path.name}: {shape.obs_len} -> {shape.action_len}, ok")
        if cfg.smoke:
            m.smoke_run_onnx(onnx_path)
            print("[publish] smoke run: finite, non-constant output")

        command_help = {"twist": cfg.twist_help} if cfg.twist_help else None
        manifest = m.build_manifest(name=name, kind=cfg.kind, description=cfg.description or training.get("task_id") or name, duration_s=cfg.duration_s, chain=cfg.chain, unwind_s=cfg.unwind_s, idle=cfg.idle, action_scale=cfg.action_scale, entry_pose=cfg.entry_pose, slot=cfg.slot, command_help=command_help, training=training)
        m.validate_manifest(manifest)

        staged = workdir / "repo"
        staged.mkdir()
        shutil.copyfile(onnx_path, staged / m.POLICY_FILE)
        (staged / "manifest.json").write_text(m.dump_manifest(manifest))
        (staged / "README.md").write_text(m.render_readme(manifest, cfg.repo))

        if cfg.dry_run:
            dest = Path.cwd() / f"publish-{name}"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(staged, dest)
            print(f"[publish] dry run: wrote {dest}/ (policy.onnx, manifest.json, README.md)")
            return 0

        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(cfg.repo, repo_type="model", private=cfg.private, exist_ok=True)
        existing = set(api.list_repo_files(cfg.repo))
        onnx_files = {f for f in existing if f.endswith(".onnx")}
        if onnx_files and not cfg.force:
            _fail(f"{cfg.repo} already carries {sorted(onnx_files)}; --force overwrites. A repo carries exactly one .onnx, so a second name is a new repo.")
        stale = onnx_files - {m.POLICY_FILE}
        commit = api.upload_folder(repo_id=cfg.repo, folder_path=str(staged), commit_message=f"publish {name}: {cfg.kind}, {training.get('task_id', onnx_path.name)}", delete_patterns=sorted(stale) or None)
        url = getattr(commit, "commit_url", None) or f"https://huggingface.co/{cfg.repo}"
        print(f"[publish] uploaded: {url}")
        if cfg.tag:
            api.create_tag(cfg.repo, tag=cfg.tag, tag_message=f"{name} {cfg.tag}")
            print(f"[publish] tagged {cfg.tag}")
        first = m.install_commands(manifest, cfg.repo).splitlines()[0]
        print(f"[publish] on a robot: {first}")
        return 0
    except AssertionError as e:
        _fail(str(e))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return 0


def main() -> int:
    cfg = tyro.cli(PublishConfig)
    return run(cfg)


if __name__ == "__main__":
    sys.exit(main())
