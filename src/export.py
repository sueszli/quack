# Export a trained checkpoint to ONNX, with the observation normalizer baked in.
#
# This is the ONE path from a checkpoint to a deployable `.onnx`: `runner.export_policy_to_onnx`
# emits `actor(normalizer(obs))`, so what the robot runs is what training saw. In-sim `play`
# applies the normalizer itself and hides a hand-converted checkpoint that forgot it — never
# convert by hand.
#
# `uv run export` is the command-line entry (:func:`main`); `src.publish_cli` calls
# :func:`run_export` directly so a published policy cannot skip this step.

import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from rsl_rl.runners import OnPolicyRunner


@dataclass(frozen=True)
class ExportConfig:
    onnx_file: str = "output.onnx"
    agent: Literal["untrained", "trained"] = "trained"
    checkpoint: int | None = None  # Select checkpoint by iteration number (e.g. 3000)
    checkpoint_file: str | None = None
    motion_file: str | None = None
    num_envs: int | None = None
    device: str | None = None
    video: bool = False
    video_length: int = 200
    video_height: int | None = None
    video_width: int | None = None
    camera: int | str | None = None
    viewer: Literal["auto", "native", "viser"] = "auto"

    # Internal flag used by demo script.
    _demo_mode: tyro.conf.Suppress[bool] = False


@dataclass(frozen=True)
class ExportResult:
    # What an export produced and where it came from, for the publisher's provenance block.

    onnx_path: Path
    checkpoint_path: Path | None
    checkpoint_iteration: int | None


# Only `model_<N>.pt` is a checkpoint. See the comment at the fallback branch in `run_export`.
CHECKPOINT_PATTERN = r"model_\d+\.pt$"


def _iteration_of(checkpoint_path: Path | None) -> int | None:
    if checkpoint_path is None:
        return None
    match = re.search(r"model_(\d+)\.pt$", checkpoint_path.name)
    return int(match.group(1)) if match else None


def run_export(task_id: str, cfg: ExportConfig) -> ExportResult:
    configure_torch_backends()

    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)

    DUMMY_MODE = cfg.agent == "untrained"
    TRAINED_MODE = not DUMMY_MODE

    # Check if this is a motion tracking task.
    is_motion_tracking = env_cfg.commands is not None and "motion" in env_cfg.commands and isinstance(env_cfg.commands["motion"], MotionCommandCfg)
    is_tracking_task = is_motion_tracking

    if is_tracking_task and cfg._demo_mode:
        # Demo mode: use uniform sampling to see more diversity with num_envs > 1.
        assert env_cfg.commands is not None
        motion_cmd = env_cfg.commands["motion"]
        assert isinstance(motion_cmd, MotionCommandCfg)
        motion_cmd.sampling_mode = "uniform"

    if is_tracking_task:
        assert env_cfg.commands is not None
        motion_cmd = env_cfg.commands["motion"]
        assert isinstance(motion_cmd, MotionCommandCfg)

        # Check if motion file is already set and exists
        motion_file_already_set = hasattr(motion_cmd, "motion_file") and motion_cmd.motion_file is not None and Path(motion_cmd.motion_file).exists()

        if cfg.motion_file is not None:
            print(f"[INFO]: Using motion file from CLI: {cfg.motion_file}")
            motion_cmd.motion_file = cfg.motion_file
        elif motion_file_already_set:
            print(f"[INFO]: Using motion file from env config: {motion_cmd.motion_file}")
        else:
            raise ValueError("Tracking tasks require `motion_file`: pass `--motion-file <path/to/motion.npz>` (there is no remote artifact store).")

    log_dir: Path | None = None
    resume_path: Path | None = None
    if TRAINED_MODE:
        log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
        if cfg.checkpoint_file is None and not log_root_path.exists():
            raise FileNotFoundError(f"No local runs for this task: {log_root_path} does not exist. Train it first, or point at a checkpoint you copied over with `--checkpoint-file <path/to/model_N.pt>`.")
        if cfg.checkpoint_file is not None:
            resume_path = Path(cfg.checkpoint_file)
            if not resume_path.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
            print(f"[INFO]: Loading checkpoint: {resume_path.name}")
        elif cfg.checkpoint is not None:
            # Select a specific checkpoint iteration from the local log tree.
            checkpoint_filename = f"model_{cfg.checkpoint}.pt"
            resume_path = get_checkpoint_path(log_root_path, checkpoint=re.escape(checkpoint_filename))
            print(f"[INFO]: Loading checkpoint: {resume_path.name}")
        else:
            # Latest checkpoint of the latest run under logs/rsl_rl/<experiment_name>/.
            # The pattern matters: mjlab's default (".*") matches every entry in the run
            # directory, and alphabetical order puts `events.out.tfevents.*` and `params/`
            # AFTER `model_*.pt` — so the default would hand back the TensorBoard event
            # file (which logger="tensorboard" guarantees is sitting right there).
            resume_path = get_checkpoint_path(log_root_path, checkpoint=CHECKPOINT_PATTERN)
            print(f"[INFO]: Loading checkpoint: {resume_path.name} (latest in {log_root_path})")
        log_dir = resume_path.parent

    if cfg.num_envs is not None:
        env_cfg.scene.num_envs = cfg.num_envs
    if cfg.video_height is not None:
        env_cfg.viewer.height = cfg.video_height
    if cfg.video_width is not None:
        env_cfg.viewer.width = cfg.video_width

    render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
    if cfg.video and DUMMY_MODE:
        print("[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir).")
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

    if TRAINED_MODE and cfg.video:
        print("[INFO] Recording videos during play")
        assert log_dir is not None  # log_dir is set in TRAINED_MODE block
        env = VideoRecorder(env, video_folder=log_dir / "videos" / "play", step_trigger=lambda step: step == 0, video_length=cfg.video_length, disable_logger=True)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task_id) or OnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    if TRAINED_MODE:
        runner.load(str(resume_path), map_location=device)
    else:
        print("[WARN] --agent untrained: random-init weights. A shape fixture only — NOT deployable, and `publish` will refuse it.")

    # mjlab 1.3.0: ONNX export + metadata moved to mjlab.rl.exporter_utils and
    # the runner's built-in export_policy_to_onnx. Observation normalization is
    # baked into the exported graph automatically — EmpiricalNormalization is a
    # submodule of the policy's MLPModel (obs_normalization=True in RslRlModelCfg),
    # so export_policy_to_onnx emits actor(normalizer(obs)). No manual normalizer
    # handling needed (the old export_velocity_policy_as_onnx path is gone).
    from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata

    onnx_path = os.path.abspath(cfg.onnx_file)
    path = os.path.dirname(onnx_path)
    filename = os.path.basename(onnx_path)

    runner.export_policy_to_onnx(path, filename)

    metadata = get_base_metadata(runner.env.unwrapped, run_path=str(resume_path) if resume_path is not None else None)
    if DUMMY_MODE:
        metadata["untrained"] = "true"
    attach_metadata_to_onnx(onnx_path, metadata)

    print(f"Written {onnx_path}")

    env.close()
    return ExportResult(onnx_path=Path(onnx_path), checkpoint_path=resume_path, checkpoint_iteration=_iteration_of(resume_path))


def main():
    # Parse first argument to choose the task.
    # Import tasks to populate the registry.
    import mjlab.tasks  # noqa: F401

    all_tasks = list_tasks()
    chosen_task, remaining_args = tyro.cli(tyro.extras.literal_type_from_choices(all_tasks), add_help=False, return_unknown_args=True)

    # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
    agent_cfg = load_rl_cfg(chosen_task)

    args = tyro.cli(ExportConfig, args=remaining_args, default=ExportConfig(), prog=sys.argv[0] + f" {chosen_task}", config=(tyro.conf.AvoidSubcommands, tyro.conf.FlagConversionOff))
    del remaining_args, agent_cfg

    run_export(chosen_task, args)


if __name__ == "__main__":
    main()
