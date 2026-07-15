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

from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity2_env_cfg import (
    make_microduck_velocity2_env_cfg,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg

# Phase boundaries (PPO iterations; env step counter scales by num_steps_per_env=24)
FELL_OVER_DISABLE_ITER = 500
NUM_STEPS_PER_ENV = 24

# Fallen gates. LESSON (first rebase training run): the recovery REWARDS must
# gate on TILT ONLY. Gating them on low height too made SITTING (z≈0.07, trunk
# upright) open the gate → the policy learned to sit and farm upright_linear
# while bobbing for com_upward_velocity and shaking its legs through the
# air_time window. Gating a positive reward on a bad state rewards entering
# the state. Tilt>40° can't be farmed from a comfortable pose — you're
# genuinely toppled. The TERMINATION keeps the z-condition so sitters and
# stuck-low envs get recycled (terminated) rather than paid.
REWARD_GATE_TILT_DEG = 40.0   # recovery rewards: fallen = tilt > 40° ONLY
# TERM z-gate at 0.08, NOT 0.10 (run-3 lesson): a normally wobbling upright
# robot dips to z=0.084-0.096 — 0.10 sits inside the early-learning envelope
# and recycled crouch-walking explorers every 5 s. 0.08 still catches sitting
# (z≈0.07) and prone (z≈0.05).
TERM_GATE_Z = 0.08            # fallen_too_long: z < 0.08 OR tilt > 40°
TERM_GATE_TILT_DEG = 40.0

# The tax and bounty exist FOR THE RECOVERY PHASE. Run-3 lesson: fallen_tax
# active from step 0 (dense, -0.5) taught "avoid tilt at all costs" within ~25
# iters → crouch-freeze local optimum before walking could bootstrap (ep_len
# pinned at the 5 s recycle, air_time never grew). Both ramp in at iter 1200 —
# after the walk is established, before the prone ramp begins at 1500.
RECOVERY_ECON_KICKIN_ITER = 1200

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

    # ── Recovery reward layer ─────────────────────────────────────────────────
    # LESSON (runs 1/2/4 — sitting, lying, head-tripod): ANY positive reward for
    # BEING in a fallen-ish state gets farmed from some comfortable pose. The
    # orientation reward is therefore POTENTIAL-BASED (Δcos tilt): rising pays,
    # falling costs, holding anything pays zero. Unfarmable, ungated, and also
    # rewards catching a stumble while walking. (Run 4 specifically: removing
    # the head-impact penalty unlocked a head-tripod at ~55° farming the gated
    # +2·cos(tilt) — run 2 had only been protected from it by that penalty.)
    cfg.rewards["upright_progress"] = RewardTermCfg(
        func=microduck_mdp.upright_progress,
        weight=5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    cfg.rewards["com_upward_velocity"] = RewardTermCfg(
        func=microduck_mdp.com_upward_velocity,
        weight=0.0,  # recovery term — ramped in at RECOVERY_ECON_KICKIN_ITER
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
            # Height gate slightly above standing (standup uses 0.125) so the
            # rising reward keeps paying until fully up; the fallen gate is
            # what prevents gait-bounce farming, not this ceiling.
            "max_height": 0.125,
            # tilt-only gate: z=0.0 never triggers (see LESSON above)
            "gate_z_below": 0.0,
            "gate_tilt_above_deg": REWARD_GATE_TILT_DEG,
        },
    )
    # NO impact penalties (first run lesson #2): the standup SPECIALIST has
    # none — the duck's recovery pushes off with head/trunk, and the head
    # penalty (-1.0 @ 2 N) taxed exactly that strategy. Falls stayed cheaper
    # than getting up. joint_torque_rate_l2 below covers landing harshness.
    # Standup's proven anti-jitter term: penalizes torque CHANGE (not magnitude
    # or rotation) → smooths transfer without blocking the recovery flip.
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torque_rate_l2,
        weight=-2e-3,
    )

    # ── Recovery economics (first-run lessons #3-#5) ──────────────────────────
    # air_time zeroed while fallen: a robot lying on its trunk can rhythmically
    # tap its feet through the swing window — the observed "shaking a leg" farm.
    at = cfg.rewards["air_time"]
    at_params = dict(at.params)
    cfg.rewards["air_time"] = RewardTermCfg(
        func=microduck_mdp.feet_air_time_upright,
        weight=at.weight,
        params={**at_params, "gate_tilt_above_deg": REWARD_GATE_TILT_DEG},
    )
    # Flat tax while fallen: lying still must be strictly worse than trying.
    # (Without it, waiting 5 s for the fallen_too_long recycle was rational —
    # recovery attempts cost action-rate/torque penalties, waiting cost 0.)
    cfg.rewards["fallen_tax"] = RewardTermCfg(
        func=microduck_mdp.fallen_state_penalty,
        weight=0.0,  # ramped to -0.5 at RECOVERY_ECON_KICKIN_ITER (see curriculum)
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
            "gate_tilt_above_deg": REWARD_GATE_TILT_DEG,
        },
    )
    # One-shot bounty on a COMPLETED recovery (fallen ≥0.5 s → genuinely up),
    # with hysteresis so gate-oscillation pays nothing. The strong endpoint
    # signal the dense gated terms lack.
    cfg.rewards["recovery_success"] = RewardTermCfg(
        func=microduck_mdp.recovery_success,
        weight=0.0,  # ramped to +10 at RECOVERY_ECON_KICKIN_ITER (see curriculum)
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
            "fallen_tilt_deg": REWARD_GATE_TILT_DEG,
            "min_fallen_s": 0.5,
            "up_tilt_deg": 25.0,
            "up_z": 0.105,
        },
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
            "gate_z_below": TERM_GATE_Z,
            "gate_tilt_above_deg": TERM_GATE_TILT_DEG,
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

    # Recovery economics ramp: tax + bounty OFF until the walk is established
    # (see RECOVERY_ECON_KICKIN_ITER note above — run-3 crouch-freeze lesson).
    cfg.curriculum["fallen_tax_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "fallen_tax",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": RECOVERY_ECON_KICKIN_ITER * NUM_STEPS_PER_ENV, "weight": -0.5},
            ],
        },
    )
    cfg.curriculum["recovery_success_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "recovery_success",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": RECOVERY_ECON_KICKIN_ITER * NUM_STEPS_PER_ENV, "weight": 10.0},
            ],
        },
    )
    cfg.curriculum["com_upward_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "com_upward_velocity",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": RECOVERY_ECON_KICKIN_ITER * NUM_STEPS_PER_ENV, "weight": 2.0},
            ],
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
