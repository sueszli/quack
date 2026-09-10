# One vocabulary for two shapes — a single-policy repo (fields at the top level) and the official
# set (the same fields per entry under ``policies``). This module writes the first; the daemon
# (`pollen-robotics/microduck`, ``updater/src/policy.rs`` and ``robotd-params``) reads both. The
# contract is `docs/policy-manifest.md` over there; the numbers below are what the daemon publishes
# in ``duck_ipc_proto`` and refuses a policy for disagreeing with.
#
# Deliberately free of mjlab / torch imports so the tests run on a laptop in milliseconds and the
# CLI can validate an ONNX file without a GPU.

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2
# `duck_ipc_proto`: the daemon refuses a policy whose manifest disagrees with these, and refuses
# at load a network whose graph does. 61 = 48 proprioception + 13 command; 14 = the servos.
MODEL_API = 1
OBS_LEN = 61
ACTION_LEN = 14
ROBOT: dict[str, Any] = {"model": "microduck", "hw_rev": 1, "servos": "xl330", "control_hz": 50}

# The one `.onnx` a repo carries. The daemon takes the sole `.onnx` in a repo and refuses several.
POLICY_FILE = "policy.onnx"

KINDS: tuple[str, ...] = ("episodic", "perpetual")

ZERO_TWIST: tuple[float, float, float] = (0.0, 0.0, 0.0)

# The daemon's policy slots, for a gait's `slot` hint (display-only: `robotctl policy load <slot>`).
SLOTS: tuple[str, ...] = ("walk", "stand", "sitstand", "ground_pick", "kick_left", "kick_right", "roulade")


def git_provenance(repo_root: Path | None = None) -> dict[str, Any]:
    root = str(repo_root or Path(__file__).resolve().parents[1])

    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", "-C", root, *args], capture_output=True, text=True, check=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip()

    commit = git("rev-parse", "--short=9", "HEAD")
    if commit is None:
        return {}
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "branch": branch, "dirty": bool(status)}


def build_manifest(*, name: str, kind: str, description: str, duration_s: float | None = None, chain: bool = False, unwind_s: float | None = None, idle: tuple[float, float, float] = ZERO_TWIST, action_scale: float | None = None, entry_pose: str = "standing", slot: str | None = None, command_help: dict[str, Any] | None = None, training: dict[str, Any] | None = None, eval: dict[str, Any] | None = None) -> dict[str, Any]:
    # Only the constant-command family is publishable from here — a skill's network is fed a fixed
    # twist. Phase and posture-flag encodings are the official set's own arms and are not something
    # a community policy can be.
    assert kind in KINDS, f"kind {kind!r} not in {KINDS}"
    assert name and "/" not in name and name == name.strip(), f"name {name!r}: bare word, no slash, no surrounding space"
    if kind == "episodic":
        assert duration_s is not None and duration_s > 0, "episodic needs duration_s > 0"
        assert not unwind_s, "unwind_s is perpetual-only"
    else:
        # Two things are perpetual: a gait, which lives in a slot (`policy load walk <repo>`) and
        # needs nothing here, and a held pose like the flamingo, which the owner runs as a
        # one-shot with `policy add --hold` and which then needs `unwind_s` so the robot is not
        # let go of on one foot. `unwind_s` is what says which.
        assert duration_s is None, "perpetual has no duration_s (a held pose gets --hold at `policy add`)"
        assert unwind_s is None or unwind_s > 0, f"unwind_s {unwind_s}: must be > 0"
        assert not chain, "chain is episodic-only"
    assert slot is None or slot in SLOTS, f"slot {slot!r} not in {SLOTS}"
    assert action_scale is None or 0 < action_scale <= 2.0, f"action_scale {action_scale} outside (0, 2]"
    assert len(idle) == 3, f"idle: 3-vector twist, got len {len(idle)}"

    command: dict[str, Any] = {"encoding": "constant", "idle": [float(v) for v in idle], "twist": "unused (zeros)", "head": "unused (zeros)", "body": "unused (zeros)"}
    if command_help:
        command.update(command_help)

    manifest: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "model_api": MODEL_API, "obs_len": OBS_LEN, "action_len": ACTION_LEN, "robot": dict(ROBOT), "name": name, "kind": kind, "entry_pose": entry_pose, "description": description, "command": command}
    if kind == "episodic":
        manifest["duration_s"] = float(duration_s)  # type: ignore[arg-type]
        manifest["chain"] = bool(chain)
    else:
        manifest["duration_s"] = None
        if unwind_s is not None:
            manifest["unwind_s"] = float(unwind_s)
    if slot is not None:
        manifest["slot"] = slot
    if action_scale is not None:
        manifest["action_scale"] = float(action_scale)
    if training:
        manifest["training"] = training
    if eval:
        manifest["eval"] = eval
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> None:
    # Accepts both shapes and any schema version, because absence is not evidence — a repo is under
    # no obligation to carry any field. Only a claim that is present and wrong fails.
    if "policies" in manifest:
        for entry in manifest["policies"]:
            assert "file" in entry, "set entry missing `file`"
            validate_manifest({k: v for k, v in entry.items() if k != "file"})
        return
    obs = manifest.get("obs_len")
    assert obs is None or obs == OBS_LEN, f"obs_len {obs}: robot builds {OBS_LEN}"
    act = manifest.get("action_len")
    assert act is None or act == ACTION_LEN, f"action_len {act}: robot has {ACTION_LEN}"
    api = manifest.get("model_api")
    assert api is None or api <= MODEL_API, f"model_api {api}: repo targets {MODEL_API}"
    model = (manifest.get("robot") or {}).get("model")
    assert model is None or model.lower() == ROBOT["model"], f"robot.model {model!r}: expected {ROBOT['model']}"
    kind = manifest.get("kind")
    assert kind is None or kind in (*KINDS, "scripted"), f"kind {kind!r} not in episodic, perpetual, scripted"
    encoding = (manifest.get("command") or {}).get("encoding")
    assert encoding is None or encoding in ("constant", "phase", "posture_flag"), f"command.encoding {encoding!r} not driven by the daemon"
    if kind == "episodic" and encoding in (None, "constant"):
        duration = manifest.get("duration_s")
        assert duration is not None and duration > 0, "episodic constant-command needs duration_s > 0"
    idle = (manifest.get("command") or {}).get("idle")
    assert idle is None or len(idle) == 3, f"command.idle: 3-vector twist, got len {len(idle)}"


@dataclass(frozen=True)
class OnnxShape:
    input_name: str
    output_name: str
    obs_len: int
    action_len: int


def inspect_onnx(path: Path) -> OnnxShape:
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    graph = model.graph
    initializers = {i.name for i in graph.initializer}
    inputs = [i for i in graph.input if i.name not in initializers]
    assert len(inputs) == 1 and len(graph.output) == 1, f"{path.name}: want 1 input and 1 output, got {[i.name for i in inputs]} -> {[o.name for o in graph.output]}"

    def last_dim(value) -> int:
        dims = value.type.tensor_type.shape.dim
        assert dims, f"{path.name}: {value.name} has no shape"
        last = dims[-1]
        assert last.HasField("dim_value"), f"{path.name}: {value.name} last dim is symbolic"
        return int(last.dim_value)

    return OnnxShape(input_name=inputs[0].name, output_name=graph.output[0].name, obs_len=last_dim(inputs[0]), action_len=last_dim(graph.output[0]))


def is_untrained_onnx(path: Path) -> bool:
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    return any(prop.key == "untrained" and prop.value == "true" for prop in model.metadata_props)


def check_onnx(path: Path) -> OnnxShape:
    assert path.exists(), f"{path}: no such file"
    assert not is_untrained_onnx(path), f"{path.name}: untrained export (random-init weights), not a policy"
    shape = inspect_onnx(path)
    assert shape.obs_len == OBS_LEN, f"{path.name}: obs width {shape.obs_len}, robot builds {OBS_LEN} (51 = legacy 3-value command)"
    assert shape.action_len == ACTION_LEN, f"{path.name}: {shape.action_len} actions, robot has {ACTION_LEN}"
    return shape


def smoke_run_onnx(path: Path, steps: int = 50, seed: int = 0) -> None:
    # Not a physics rehearsal — `infer.py` is that — but it catches a broken export
    # (an un-baked normalizer producing NaNs on raw observations, a graph that will not execute)
    # before anything is uploaded.
    import numpy as np
    import onnxruntime as ort

    shape = inspect_onnx(path)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(seed)
    obs = np.zeros((1, shape.obs_len), dtype=np.float32)
    outputs = []
    for _ in range(steps):
        (out,) = session.run([shape.output_name], {shape.input_name: obs})
        assert np.all(np.isfinite(out)), f"{path.name}: non-finite action"
        outputs.append(out)
        # Feed the action back into the last-action slots and jitter the rest, the way an
        # observation evolves on the robot; enough to leave the zero point.
        obs = rng.normal(0.0, 0.05, size=obs.shape).astype(np.float32)
        obs[0, -ACTION_LEN - 13 : -13] = np.clip(out[0], -1, 1)
    spread = float(np.std(np.stack(outputs)))
    assert spread > 0.0, f"{path.name}: output never changes over {steps} steps"


def install_commands(manifest: dict[str, Any], repo_id: str) -> str:
    name = manifest["name"]
    if manifest["kind"] == "episodic":
        return f"sudo robotctl policy add {name} {repo_id}\nrobotctl robot do {name}"
    if manifest.get("unwind_s") is not None:
        return f"sudo robotctl policy add {name} {repo_id} --hold <seconds>\nrobotctl robot do {name}"
    slot = manifest.get("slot", "<slot>")
    return f"sudo robotctl policy load {slot} {repo_id}"


def render_readme(manifest: dict[str, Any], repo_id: str) -> str:
    kind = manifest["kind"]
    name = manifest["name"]
    description = manifest.get("description", "")
    training = manifest.get("training", {})
    run = install_commands(manifest, repo_id)
    if kind == "episodic":
        timing = f"Runs {manifest['duration_s']} s and returns itself to a standing pose."
        if manifest.get("chain"):
            timing += " Holding the button chains another run."
    elif manifest.get("unwind_s") is not None:
        timing = f"Holds until told otherwise; the daemon drives `command.idle` for {manifest['unwind_s']} s before handing back to the gait."
    else:
        slot = manifest.get("slot")
        timing = "Runs until told otherwise" + (f" — a gait for the `{slot}` slot." if slot else " — a gait, loaded into a policy slot.")
    lines = ["---", "tags:", "- microduck", "- robotics", "- reinforcement-learning", "- onnx", "library_name: onnx", "---", "", f"# {name}", "", description, "", f"A **{kind}** policy for the [microduck](https://github.com/pollen-robotics/microduck) ({OBS_LEN}-D observation, {ACTION_LEN} actions, {ROBOT['control_hz']} Hz). {timing}", "", "## Run it on a robot", "", "```bash", run, "```", "", "The observation normalizer is baked into `policy.onnx`; feed raw observations.", "`manifest.json` follows schema 2 of the microduck policy manifest (`docs/policy-manifest.md` in the daemon repo)."]
    if training:
        lines += ["", "## Training", ""]
        for key in ("task_id", "repo", "branch", "commit", "checkpoint", "exported"):
            if key in training:
                lines.append(f"- **{key}**: `{training[key]}`")
        if training.get("dirty"):
            lines.append("- exported from a checkout with uncommitted changes")
    return "\n".join(lines) + "\n"


def dump_manifest(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, indent=2) + "\n"
