
# MjLab Microduck

RL training environments for the [microduck](https://github.com/apirrone/microduck), built on [mjlab](https://github.com/mujocolab/mjlab) (MuJoCo Warp). Policies are trained here, exported to ONNX, and deployed with [microduck_runtime](https://github.com/apirrone/microduck_runtime).

https://github.com/user-attachments/assets/e9a0d4de-8a3d-4e44-b490-e873728cf2bf

## Training

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096
```

Resume from a checkpoint:

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096 \
    --agent.run-name resume --agent.load-checkpoint model_29999.pt --agent.resume True
```

Run the same command on Hugging Face Jobs instead of locally (submission flags:
`--flavor`, `--namespace`, `--detach`, ... — see `src/mjlab_microduck/hf_jobs.py`):

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096 --hf-jobs
```

## Tasks

`uv run list-envs` prints the live registry. Flat/Rough variants exist where noted.

| Task id | Terrain | Description |
|---|---|---|
| `Mjlab-Velocity-{Flat,Rough}-MicroDuck` | flat/rough | Walking with velocity commands (main task) + head-pose commands |
| `Mjlab-Velocity2-{Flat,Rough}-MicroDuck` | flat/rough | Velocity walking with the microban reward/regularization recipe |
| `Mjlab-VelStand-{Flat,Rough}-MicroDuck` | flat/rough | Walking + fall recovery + body-pose control in one policy |
| `Mjlab-VelStandTipToe-{Flat,Rough}-MicroDuck` | flat/rough | VelStand + tiptoe feet-alignment reward |
| `Mjlab-StandUp-{Flat,Rough}-MicroDuck` | flat/rough | Stand up from face-down/face-up/sitting, then hold the stand |
| `Mjlab-Sit-{Flat,Rough}-MicroDuck` | flat/rough | Gentle stand → sit (companion to StandUp) |
| `Mjlab-GroundPick-{Flat,Rough}-MicroDuck` | flat/rough | Crouch and touch the ground with the mouth tip, return to stand |
| `Mjlab-BallKick-Flat-MicroDuck` | flat | Kick a 70 mm / 15 g ball forward with the right foot (actor is ball-blind) |
| `Mjlab-Shoot-Flat-MicroDuck` | flat | Standing kick with the right leg, left leg planted |
| `Mjlab-Velocity-Flat-MicroDuck-Rollers` | flat | Roller-skate velocity tracking (passive wheels under the feet) |
| `Mjlab-Velocity-Swizzle-MicroDuck` | flat | Classic symmetric swizzle skating |
| `Mjlab-RollerCrouch-Flat-MicroDuck` | flat | Crouch while gliding on rollers |
| `Mjlab-RollerSlope-Flat-MicroDuck` | slope | Glide down slopes on rollers |

### Backlash variants

Every task above has a **Backlash** twin that trains on a model with ±1° of
gear play (2° total) in series with each of the 14 servo joints:
insert `-Backlash` before `MicroDuck` in the task id, e.g.
`Mjlab-Velocity-Flat-Backlash-MicroDuck`,
`Mjlab-Velocity-Flat-Backlash-MicroDuck-Rollers`,
`Mjlab-Velocity-Swizzle-Backlash-MicroDuck`.

The backlash is modeled properly for sim2real: each servo gets an unactuated
`passive_<joint>_backlash` hinge, and because the real encoder sits on the
output side of the play, both the firmware PD emulation
(`BacklashEncoderBamActuator`) and the `joint_pos`/`joint_vel` observations
read *through* the backlash (`qpos[servo] + qpos[backlash]`). Observation and
action dims are unchanged, so ONNX export and the runtime need no changes.
See `src/mjlab_microduck/tasks/backlash.py`.

## Play

```bash
uv run play Mjlab-Velocity-Flat-MicroDuck --wandb-run-path <entity/project/run_id>
```

`scripts/play_latest.py` finds and plays your latest wandb run
(`--crouch`, `--roller`, `--swizzle`, `--slope` filter by task type).

## ONNX export

```bash
uv run scripts/export.py Mjlab-Velocity-Flat-MicroDuck --wandb-run-path <...>
# or from a local file: --checkpoint-file logs/.../model_XXXX.pt
```

The exporter bakes the observation normalizer into the ONNX graph — always
deploy ONNX produced by this script, never a hand-converted checkpoint,
or the policy sees unnormalized observations at runtime.

## Infer a policy in MuJoCo (CPU)

```bash
uv run scripts/infer_policy.py --walking output.onnx
# multiple policies at once, e.g.:
uv run scripts/infer_policy.py --walking walk.onnx --standing stand.onnx --sit sit.onnx
uv run scripts/infer_policy.py --roller --walking roller.onnx
```

Keyboard-driven (velocity commands, G = ground pick, Y = sit/slope toggle);
`--debug`, `--save-csv`, `--record` for sim2real comparisons.

## Robot models

MJCF models live in `src/mjlab_microduck/robot/microduck/` and are exported
from Onshape with [onshape-to-robot](https://github.com/Rhoban/onshape-to-robot),
one `config_mjcf_*.json` per model:

| XML | Export config | Used by |
|---|---|---|
| `robot_walk.xml` | `config_mjcf_walk.json` | Velocity, Velocity2 |
| `robot_allcollisions.xml` | `config_mjcf_allcollisions.json` | VelStand, StandUp, Sit, GroundPick, BallKick, Shoot |
| `robot_allcollisions_rollers.xml` | `config_mjcf_allcollisions_rollers.json` | Roller tasks |
| `robot_*_backlash.xml` | `config_mjcf_*_backlash.json` | Backlash task variants |

The backlash models are produced by `add_backlash.py`, which runs as the last
`post_import_commands` step of the backlash export configs. `--backlash-deg`
is the TOTAL peak-to-peak play (default 2.0 → joint range ±1°); it also works
standalone on an already-exported xml:

```bash
cd src/mjlab_microduck/robot/microduck
cp robot_walk.xml robot_walk_backlash.xml
python3 add_backlash.py robot_walk_backlash.xml --backlash-deg 2.0
```

`scene*.xml` files wrap the robots with a floor + keyframes (STAND/SIT/FOLD)
for quick viewing and for `infer_policy.py`. On the backlash scenes, keyframe
qpos vectors interleave a `0` after each servo value for its backlash hinge.

## Actuator model

All tasks use the [BAM](https://github.com/Rhoban/bam) M6 actuator model for
the XL330 (voltage control law, back-EMF, Coulomb/Stribeck/load-dependent
friction), with per-env domain randomization on battery voltage, voltage sag
under load, command delay, and friction magnitude
(`FrictionDRBamActuator` in `src/mjlab_microduck/actuator/`).

## Project structure

```
src/mjlab_microduck/
├── robot/
│   ├── microduck/                    # MJCF exports, export configs, scenes, add_backlash.py
│   └── microduck_constants.py        # robot cfgs, HOME frame, BAM actuator cfg
├── actuator/friction_dr_bam.py       # BAM + friction DR + backlash encoder feedback
├── tasks/
│   ├── __init__.py                   # task registration (base + backlash variants)
│   ├── mdp.py                        # rewards, events, observations, custom classes
│   ├── backlash.py                   # make_backlash_variant() env-cfg wrapper
│   └── microduck_*_env_cfg.py        # one cfg module per task family
├── train_cli.py                      # `train` entry point (+ --hf-jobs)
└── hf_jobs.py                        # Hugging Face Jobs submission
```

Conventions worth knowing:

- Unactuated joints are all named `passive_*` (roller wheels, backlash
  hinges); actuators, joint observations and pose rewards select servo joints
  with `^(?!passive_).*`.
- Domain-randomization toggles are `ENABLE_*` booleans at the top of each
  env cfg file.
- Joint layout (14 servos, ctrl indices): 0-4 left leg (hip_yaw, hip_roll,
  hip_pitch, knee, ankle), 5-8 neck/head (neck_pitch, head_pitch, head_yaw,
  head_roll), 9-13 right leg.

## Tests

```bash
uv run --with pytest pytest tests/
```
