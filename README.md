# Microduck RL

<img width="2215" height="884" alt="image" src="https://github.com/user-attachments/assets/5db7cc83-b3ce-4f7c-83f0-0572a63baed7" />

RL training environments for [Microduck](https://github.com/pollen-robotics/microduck) —
a ~800 g, ~25 cm tall bipedal robot — built on
[mjlab](https://github.com/mujocolab/mjlab) (MuJoCo Warp) with PPO.
Policies are trained here at 50 Hz, exported to ONNX, and deployed on the real
robot by the runtime in [pollen-robotics/microduck](https://github.com/pollen-robotics/microduck).

https://github.com/user-attachments/assets/50c3d537-8db2-4005-9d9c-3472faeec4d0

The repo encodes the full sim2real recipe: [BAM](https://github.com/Rhoban/bam)
actuator physics, domain randomization, backlash simulation, and the
reward-design lessons that made it work
(see [CLAUDE.md](CLAUDE.md) for the distilled playbook).

## Quickstart

Requires a CUDA GPU (training runs through MuJoCo Warp), [uv](https://docs.astral.sh/uv/),
and `make`. `make help` lists every target; each takes variables
(`TASK`, `ENVS`, `RUN`, `ONNX`, ..., plus `ARGS` for extra CLI flags).

```bash
git clone https://github.com/pollen-robotics/microduck_rl
cd microduck_rl
make sync                       # install (raises uv's HTTP timeout; ARM boxes pull ~2 GB of CUDA wheels)

make smoke                      # 64 envs, 5 iters — always run this before a long run
make train TASK=Mjlab-Velocity-Flat-MicroDuck ENVS=4096   # ~1-2 h for a usable gait
make resume CHECKPOINT=model_29999.pt

make play RUN=<entity/project/run_id>       # viewer
make export RUN=<entity/project/run_id>     # -> ONNX, obs normalizer baked in
make infer ONNX=output.onnx                 # CPU MuJoCo, keyboard-driven
make publish RUN=<...> REPO=<user>/microduck-<name> KIND=episodic DURATION=4.0
```

On a remote GPU box, tunnel the viewer: `ssh -L 8080:localhost:8080 USER@HOST`,
then `make play RUN=... VIEWER=viser ARGS="--num-envs 1"` and open
http://localhost:8080 locally (keep the SSH session open).

## Tasks

`make envs` prints the live registry.

<!-- SHOWCASE GRID — one short GIF per task family (sim or real), 3 per row.
     Priority order if you only record a few: Velocity, VelStand (fall+recover),
     Roulade, SitStand, Rollers/Swizzle, BallKick. -->

| Task id | Terrain | Description |
|---|---|---|
| `Mjlab-Velocity-{Flat,Rough}-MicroDuck` | flat/rough | **The main task**: walking with velocity commands + head-pose commands |
| `Mjlab-VelStand-{Flat,Rough}-MicroDuck` | flat/rough | Walking + fall recovery in one policy |
| `Mjlab-StandUp-{Flat,Rough}-MicroDuck` | flat/rough | Stand up from face-down/face-up/sitting, then hold the stand + body-pose control |
| `Mjlab-SitStand-{Flat,Rough}-MicroDuck` | flat/rough | Commanded sit ↔ stand in one policy, gently, head commandable |
| `Mjlab-GroundPick-{Flat,Rough}-MicroDuck` | flat/rough | Crouch and touch the ground with the mouth tip, return to stand |
| `Mjlab-BallKick-Flat-MicroDuck` | flat | Kick a 70 mm / 15 g ball forward (actor is ball-blind) |
| `Mjlab-Roulade-Flat-MicroDuck` | flat | Forward roll over the head, land back on the feet |
| `Mjlab-Velocity-Flat-MicroDuck-Rollers` | flat | Roller-skate velocity tracking (passive wheels under the feet) |
| `Mjlab-Velocity-Swizzle-MicroDuck` | flat | Classic symmetric swizzle skating |
| `Mjlab-RollerCrouch-Flat-MicroDuck` | flat | Crouch while gliding on rollers |
| `Mjlab-RollerSlope-Flat-MicroDuck` | slope | Glide down slopes on rollers |
| `Mjlab-RollerStandUp-Flat-MicroDuck` | flat | Stand up from the ground onto the wheels |
| `Mjlab-Spin-Flat-MicroDuck` | flat | Fast spin in place on rollers |

At deployment the runtime hot-swaps these policies (walk / recover / trick)
behind a shared 61-dimensional observation contract, so any of them can take
over the robot at any moment. `make infer` rehearses exactly that — load one
ONNX per slot and trigger them from the keyboard:

```bash
make infer ONNX=walk.onnx ARGS="--standing stand.onnx --sitstand sitstand.onnx --roulade roulade.onnx --new-cmd-obs"
```

The servos are simulated with the same BAM M6 XL330 model the policies are
trained against (voltage control + load-dependent friction, via
`bam.mujoco.MujocoController`); `uv run infer --help` lists the flags that pin
the training DR ranges to one value or fall back to the XML PD actuators, plus
the debug/CSV/record options used for sim2real comparisons.

### Backlash variants

Every main task has a **Backlash** twin that trains on a model with ±1° of gear
play (2° total) in series with each of the 14 servo joints: insert `-Backlash`
before `MicroDuck` in the task id, e.g. `Mjlab-Velocity-Flat-Backlash-MicroDuck`.

The backlash is modeled properly for sim2real: each servo gets an unactuated
`passive_<joint>_backlash` hinge, and because the real encoder sits on the
output side of the play, both the firmware PD emulation
(`BacklashEncoderBamActuator`) and the `joint_pos`/`joint_vel` observations
read *through* the backlash (`qpos[servo] + qpos[backlash]`). Observation and
action dims are unchanged, so ONNX export and the runtime need no changes.
See `src/task_backlash.py`.

## Actuator model

All tasks use the [BAM](https://github.com/Rhoban/bam) M6 actuator model for
the Dynamixel XL330 (voltage control law, back-EMF, Coulomb/Stribeck/load-dependent
friction), with per-env domain randomization on battery voltage, voltage sag
under load, command delay, and friction magnitude
(`FrictionDRBamActuator` in `src/robot_actuator.py`).

At this scale — tiny servos driving a ~800 g biped — actuator fidelity is most
of the sim2real gap, which is why the actuator is modeled down to its voltage
control law instead of an ideal PD.

## Robot models

MJCF models live in `assets/mjcf/` (meshes in `assets/meshes/`) and are exported
from Onshape with [onshape-to-robot](https://github.com/Rhoban/onshape-to-robot),
one `config_mjcf_*.json` per model:

| XML | Used by |
|---|---|
| `robot_walk.xml` | Velocity (stripped trunk/head contacts — falling is cheap) |
| `robot_groundcontact.xml` | VelStand, StandUp, SitStand, GroundPick, BallKick, Roulade (curated collision set for the parts that touch the floor — body can physically lie on the ground; formerly `robot_allcollisions.xml`) |
| `robot_groundcontact_rollers.xml` | Roller tasks (passive wheels) |
| `robot_allcollisions.xml` | True full-collision model — every part has a collision geom. No task uses it yet |
| `robot_*_backlash.xml` | Backlash task variants (generated by `add_backlash.py`) |

`scene*.xml` files wrap the robots with a floor + keyframes (STAND/SIT/FOLD)
for quick viewing and for `infer.py`.

<!-- IMAGE — side-by-side render: walk model vs rollers model (or a collision-geom
     visualization). One image here makes the model-variant story instant. -->

## Conventions

`src/` is a flat namespace package (no `__init__.py`); file names are prefixed
by role (`robot_*`, `task_*`) and one `task_<name>.py` holds each env + RL cfg.
The rules that are not obvious from the code:

- The observation layout is shared across every policy (61-dim actor obs:
  48 proprioception + commands `[twist(3), head_pose(4), body_pose(6)]`), which
  is what makes runtime policy hot-swapping possible. Envs that don't use a
  command slot zero-pad it rather than dropping it.
- Unactuated joints are all named `passive_*` (roller wheels, backlash
  hinges); actuators, joint observations and pose rewards select servo joints
  with `^(?!passive_).*`.
- Joint layout (14 servos): 0–4 left leg (hip_yaw, hip_roll, hip_pitch, knee,
  ankle), 5–8 neck/head (neck_pitch, head_pitch, head_yaw, head_roll),
  9–13 right leg.
- Domain-randomization toggles are `ENABLE_*` booleans at the top of each
  env cfg file.
- The exporter bakes the observation normalizer into the ONNX graph — always
  deploy ONNX produced by `make export`, never a hand-converted checkpoint, or
  the policy sees unnormalized observations at runtime.

[CLAUDE.md](CLAUDE.md) documents the env-building workflow and the reward-design
rules learned across the project (also aimed at AI coding agents working in
this repo).

## Publishing a policy

`make publish` puts a policy on the Hugging Face Hub in the shape the robot's
daemon loads: one `policy.onnx` with the observation normalizer baked in, a
`manifest.json` following schema 2 of the
[microduck policy manifest](https://github.com/pollen-robotics/microduck/blob/main/docs/policy-manifest.md),
and a README saying how to run it. Anyone with a microduck can then install it
with one command, no daemon release needed.

```bash
make publish RUN=<entity/project/run_id> CHECKPOINT=3000 \
    TASK=Mjlab-PoliteBow-Flat-MicroDuck REPO=<user>/microduck-polite-bow \
    KIND=episodic DURATION=4.0 ARGS='--description "Bows and comes back up."'
make publish-dry RUN=<...> REPO=<...> KIND=episodic DURATION=4.0   # show, don't upload
```

`uv run publish --help` covers the rest (uploading an already-exported
`--onnx`, `--force`, `--no-private`, `--tag`, `--chain`, `--idle`).

What `KIND` means, and what each needs:

- **episodic** — runs for `DURATION` seconds and returns itself to a standing
  pose (kicks, roulade, a bow). Add `ARGS=--chain` if holding the button should
  repeat it.
- **perpetual** — runs until told otherwise. Two shapes:
  - a **gait** (a new walk or stand): add `SLOT=walk` (or `stand`) and nothing
    else; the owner installs it with `robotctl policy load walk <repo>`.
  - a **held pose** (a flamingo): give `UNWIND=1.5`, how long the daemon drives
    the idle twist before handing back to the gait, so the robot is not let go
    of on one foot. The owner runs it as a one-shot with
    `robotctl policy add ... --hold <seconds>`.

Before anything is uploaded, `publish` checks the graph is `[1,61] -> [1,14]`
(a 51-D legacy policy is refused with a message), runs it on plausible inputs
and refuses NaNs or a constant output, fills the `training` block from git and
wandb (task, commit, branch, dirty flag, run, checkpoint), and refuses to
overwrite an existing `.onnx` in the repo without `--force`.

Only constant-command policies are publishable this way. Phase-driven moves
(the ground pick) and the posture-flag sit↔stand are driven by the daemon
itself and live in the official set, `pollen-robotics/microduck-policies`.

## Tests

```bash
make tests      # CPU-only cfg-invariant and reward regression tests
make fmt lint
```
