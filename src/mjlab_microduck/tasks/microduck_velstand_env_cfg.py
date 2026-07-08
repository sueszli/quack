"""Microduck VelStand environment: walking + fall recovery, one policy.

REBASED (2026-07, audit follow-up) on the **velocity2** recipe — the proven
walker — instead of the abandoned base-velocity recipe the old velstand used.
The 2026-07 audit found the old design starved the walk: only ~25% of
experience was clean commanded walking (2/3 prone resets + fallen envs farming
recovery reward for full 20 s episodes), the recovery rewards taxed the gait
(always-on posture double-counting, a bounce incentive from com_upward_velocity
below walk height), and the prone init dropped the robot from 0.20–0.25 m
(function defaults — a violent uncontrolled impact opening most episodes).

Design now:
  - Walk layer  = make_microduck_velocity2_env_cfg, verbatim. Everything the
    good walker has (tracking weights, air_time, turn-in-place bucket, fixed
    command ranges, DR/noise/obs) flows in by construction.
  - Robot       = all-collision standup XML (body can physically lie down).
  - Recovery    = a small reward layer GATED on actually-being-fallen
    (trunk z < 0.10 m OR tilt > 40°): contributes exactly zero during clean
    walking, steers only when down. upright_linear gives an orientation
    gradient everywhere; com_upward_velocity pays for rising. (The old
    com_height_recovery was dropped: flat/no-gradient inside its band and
    redundant with the two above — audit finding 3.)
  - Impact penalties (trunk/head) discourage hard landings, ungated.
  - joint_torque_rate_l2 (standup's proven anti-jitter) for transfer
    smoothness — penalizes torque CHANGE, never blocks the recovery flip.

Phases (as before, but with a recovery backstop):
  Phase 1 (0 → 500 iters): `fell_over` termination active (70°) → clean
    walking first.
  Phase 2 (500+): fell_over disabled (limit → π) so falls become recovery
    opportunities — but `fallen_too_long` (5 s continuously down) recycles
    failed recoveries instead of letting them farm the full 20 s episode.
  Phase 3 (1500+): prone-init ramp: face-down first (easier), face-up mixed
    in later, capped at 45% prone so the walking data share stays ≥ ~55%
    (was 2/3 prone → ~25% walking share).
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlModelCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity2_env_cfg import (
    make_microduck_velocity2_env_cfg,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg

# Phase boundaries (PPO iterations; env step counter scales by num_steps_per_env=24)
FELL_OVER_DISABLE_ITER = 500
NUM_STEPS_PER_ENV = 24

# Fallen gate shared by the recovery rewards and the fallen_too_long backstop:
# fallen = trunk z below GATE_Z OR tilt beyond GATE_TILT. Walking sits at
# z ≈ 0.115–0.13 with tilt < 25°, so the gate is firmly closed during gait.
GATE_Z = 0.10
GATE_TILT_DEG = 40.0

# Failed-recovery backstop: continuously fallen this long → terminate/reset.
FALLEN_TIMEOUT_S = 5.0

# Prone-init ramp (phase 3): capped at 45% prone (was 2/3 — starved the walk).
# Face-down introduced first (easier recovery), face-up mixed in later.
PRONE_RAMP_STAGES = [
    {"step": 0,                        "params": {"prone_prob": 0.00, "face_down_prob": 1.0}},
    {"step": 1500 * NUM_STEPS_PER_ENV, "params": {"prone_prob": 0.15, "face_down_prob": 0.80}},
    {"step": 2000 * NUM_STEPS_PER_ENV, "params": {"prone_prob": 0.30, "face_down_prob": 0.65}},
    {"step": 2500 * NUM_STEPS_PER_ENV, "params": {"prone_prob": 0.45, "face_down_prob": 0.50}},
]


def make_microduck_velstand_env_cfg(play: bool = False, rough: bool = False) -> ManagerBasedRlEnvCfg:
    # Walk layer: the PROVEN velocity2 recipe, verbatim.
    cfg = make_microduck_velocity2_env_cfg(play=play, rough=rough)

    # In play mode the curriculum doesn't run, so the fall-termination disable
    # below never fires — just delete the termination outright.
    if play:
        cfg.terminations.pop("fell_over", None)

    # Full-collision standup XML: trunk/head shells keep their contacts so the
    # robot can physically lie on the ground and push off it.
    cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}

    # Impact sensors for the recovery penalties.
    trunk_impact_cfg = ContactSensorCfg(
        name="trunk_impact_contact",
        primary=ContactMatch(mode="body", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("force",),
        reduce="netforce",
        num_slots=1,
    )
    head_impact_cfg = ContactSensorCfg(
        name="head_impact_contact",
        primary=ContactMatch(mode="subtree", pattern="neck", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("force",),
        reduce="netforce",
        num_slots=1,
    )
    cfg.scene.sensors = (*cfg.scene.sensors, trunk_impact_cfg, head_impact_cfg)

    # ── Recovery reward layer — GATED on actually-being-fallen ────────────────
    # Exactly zero while walking upright (gate closed) → no dilution of the
    # velocity2 tracking rewards, no bounce farming. See _fallen_mask in mdp.py.
    cfg.rewards["upright_linear"] = RewardTermCfg(
        func=microduck_mdp.body_upright_linear,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
            "gate_z_below": GATE_Z,
            "gate_tilt_above_deg": GATE_TILT_DEG,
        },
    )
    cfg.rewards["com_upward_velocity"] = RewardTermCfg(
        func=microduck_mdp.com_upward_velocity,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
            # Height gate slightly above standing (standup uses 0.125) so the
            # rising reward keeps paying until fully up; the fallen gate is
            # what prevents gait-bounce farming, not this ceiling.
            "max_height": 0.125,
            "gate_z_below": GATE_Z,
            "gate_tilt_above_deg": GATE_TILT_DEG,
        },
    )
    # Impact penalties: discourage slamming the trunk shell / head into the
    # ground during falls and recovery pushes. Ungated (always relevant).
    cfg.rewards["trunk_impact_penalty"] = RewardTermCfg(
        func=microduck_mdp.body_impact_cost,
        weight=-0.1,
        params={"sensor_name": trunk_impact_cfg.name, "threshold": 5.0},
    )
    cfg.rewards["head_impact_penalty"] = RewardTermCfg(
        func=microduck_mdp.body_impact_cost,
        weight=-1.0,
        params={"sensor_name": head_impact_cfg.name, "threshold": 2.0},
    )
    # Standup's proven anti-jitter term: penalizes torque CHANGE (not magnitude
    # or rotation) → smooths transfer without blocking the recovery flip.
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torque_rate_l2,
        weight=-2e-3,
    )

    # ── Events: prone init ────────────────────────────────────────────────────
    # z fix (audit BUG): the function defaults were 0.20–0.25 m — a 15–20 cm
    # free-fall opening every prone episode. Face-down trunk rests at ~0.044 m;
    # spawn just above the ground instead.
    cfg.events["random_prone_init"] = EventTermCfg(
        func=microduck_mdp.maybe_set_random_prone_orientation,
        mode="reset",
        params={
            "prone_prob": 0.0,        # ramped by the prone_init_prob curriculum
            "face_down_prob": 1.0,
            "prone_z_min": 0.05,
            "prone_z_max": 0.09,
        },
    )

    # ── Terminations ──────────────────────────────────────────────────────────
    # Failed-recovery backstop (see module docstring, Phase 2).
    cfg.terminations["fallen_too_long"] = TerminationTermCfg(
        func=microduck_mdp.fallen_too_long,
        time_out=False,
        params={
            "gate_z_below": GATE_Z,
            "gate_tilt_above_deg": GATE_TILT_DEG,
            "max_duration_s": FALLEN_TIMEOUT_S,
        },
    )

    # ── Curricula ─────────────────────────────────────────────────────────────
    # Phase 1 → 2: disable fell_over at iter 500 (limit 70° → 180°) so falls
    # become recovery training instead of episode ends.
    if not play:
        cfg.curriculum["fell_over_disable"] = CurriculumTermCfg(
            func=microduck_mdp.termination_param_curriculum,
            params={
                "term_name": "fell_over",
                "param_stages": [
                    {"step": 0,
                     "params": {"limit_angle": math.radians(70.0)}},
                    {"step": FELL_OVER_DISABLE_ITER * NUM_STEPS_PER_ENV,
                     "params": {"limit_angle": math.pi}},
                ],
            },
        )

    # Phase 3: prone-init ramp (face-down first, face-up later, capped 45%).
    cfg.curriculum["prone_init_prob"] = CurriculumTermCfg(
        func=microduck_mdp.event_param_curriculum,
        params={
            "event_name": "random_prone_init",
            "param_stages": PRONE_RAMP_STAGES,
        },
    )

    return cfg


MicroduckVelStandRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="velstand",
    run_name="velstand",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=20_000,
)
