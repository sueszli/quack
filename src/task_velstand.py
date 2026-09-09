"""Microduck VelStand environment: walking + fall recovery, one policy.

Design:
  - Walk layer = make_microduck_velocity_env_cfg, verbatim, so the proven
    walker's tracking weights, air_time, command ranges and DR/noise/obs flow
    in by construction.
  - Robot = all-collision standup XML, so the body can physically lie down.
  - Recovery = a small reward layer gated on actually being fallen, so it
    contributes exactly zero during clean walking.

Keeping the walk alive is the hard constraint: an earlier design left only ~25%
of experience as clean commanded walking (2/3 prone resets plus fallen envs
farming recovery reward for full 20 s episodes) and the walk starved.

Phases:
  1 (0→500):   `fell_over` active (70°) — learn to walk first.
  2 (500+):    fell_over disabled (limit → π) so falls become recovery
               opportunities, with `fallen_too_long` recycling failed ones.
  3 (1500+):   prone-init ramp, face-down first, capped at 45% so the walking
               data share stays ≥ ~55%.
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg, EventTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg

from . import task_mdp as microduck_mdp
from .robot import MICRODUCK_STANDUP_ROBOT_CFG
from .task_symmetry import PpoWithSymmetryCfg
from .task_velocity import make_microduck_velocity_env_cfg

# Phase boundaries (PPO iterations; env step counter scales by num_steps_per_env=24)
FELL_OVER_DISABLE_ITER = 500
NUM_STEPS_PER_ENV = 24

# Recovery REWARDS gate on TILT ONLY. Gating them on low height too made SITTING
# (z≈0.07, trunk upright) open the gate, and the policy sat there farming
# upright_linear while bobbing for com_upward_velocity and shaking its legs
# through the air_time window. Tilt > 40° can't be farmed from a comfortable pose.
REWARD_GATE_TILT_DEG = 40.0
# The TERMINATION keeps a z-condition so sitters and stuck-low envs get recycled
# rather than paid. 0.08, NOT 0.10: a normally wobbling upright robot dips to
# 0.084-0.096, so 0.10 sits inside the normal envelope and recycled crouch-walking
# explorers every few seconds. 0.08 still catches sitting (≈0.07) and prone (≈0.05).
TERM_GATE_Z = 0.08
TERM_GATE_TILT_DEG = 40.0

# "Recovery COMPLETE", shared by the recovery_success bounty and the fallen_tax
# release. The z threshold must sit INSIDE the policy's real standing envelope
# (measured 0.084–0.096 while wobbling; the STAND keyframe settles at ≈0.117).
# up_z=0.105 demanded standing TALLER than the policy ever is, so the bounty never
# fired and recoveries parked in a deep crouch just past the 40° gates, where every
# dense recovery term stops paying. 0.09 is reachable yet still 2 cm above sitting.
RECOVERED_UP_TILT_DEG = 25.0
RECOVERED_UP_Z = 0.09

# Tax and bounty are for the RECOVERY phase only. Active from step 0 they taught
# "avoid tilt at all costs" within ~25 iters → crouch-freeze before walking
# bootstrapped. This must also stay well after fell_over is disabled at 500: the
# gap is a TAX-FREE window where natural-fall get-up attempts cost nothing and the
# dense progress terms alone can teach them. Starting it at 800 instead left
# hopeless prone episodes bleeding -0.5/step and prone recovery never bootstrapped.
RECOVERY_ECON_KICKIN_ITER = 1200

# Failed-recovery backstop: continuously fallen this long → terminate/reset. At 5 s
# a face-down recovery spent most of its budget getting TO the deep crouch and was
# recycled right at the frontier, starving the last mile of on-policy data.
FALLEN_TIMEOUT_S = 8.0

# Phase 3 init ramp. Prone is capped at 45% (2/3 starved the walk), face-down first
# since it is the easier recovery. crouch_prob is a REVERSE-CURRICULUM slice:
# resetting directly into random mid-recovery crouches gives the crouch→stand last
# mile dense data instead of only reaching it at the tail of rare good rollouts.
# Prone starts AFTER the econ kick-in (hence after the tax-free window); the crouch
# slice can start earlier since it is near-upright and doubles as stand-tall data.
PRONE_RAMP_STAGES = [{"step": 0, "params": {"prone_prob": 0.00, "face_down_prob": 1.0, "crouch_prob": 0.00}}, {"step": 800 * NUM_STEPS_PER_ENV, "params": {"prone_prob": 0.00, "face_down_prob": 1.0, "crouch_prob": 0.15}}, {"step": 1500 * NUM_STEPS_PER_ENV, "params": {"prone_prob": 0.15, "face_down_prob": 0.80, "crouch_prob": 0.15}}, {"step": 2000 * NUM_STEPS_PER_ENV, "params": {"prone_prob": 0.30, "face_down_prob": 0.65, "crouch_prob": 0.15}}, {"step": 2500 * NUM_STEPS_PER_ENV, "params": {"prone_prob": 0.45, "face_down_prob": 0.50, "crouch_prob": 0.15}}]


def make_microduck_velstand_env_cfg(play: bool = False, rough: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)

    # In play mode the curriculum doesn't run, so the fall-termination disable below
    # never fires — delete the termination outright.
    if play:
        cfg.terminations.pop("fell_over", None)

    # Trunk/head shells keep their contacts, so the robot can lie on the ground and
    # push off it.
    cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}

    # head_pose_bias arrives UNGATED from the velocity env, which is fine there only
    # because fell_over ends fallen episodes. Velstand episodes SURVIVE falls, so an
    # ungated EMA would charge head "droop" throughout the ground phase — a flat tax
    # on being fallen that the recovery economics never priced in.
    cfg.rewards["head_pose_bias"].params.update({"gate_height_low": 0.09, "gate_height_high": 0.11, "gate_tilt_full_deg": 20.0, "gate_tilt_zero_deg": REWARD_GATE_TILT_DEG})

    # ── Recovery reward layer ─────────────────────────────────────────────────
    # ANY positive reward for BEING in a fallen-ish state gets farmed from some
    # comfortable pose (observed: sitting, lying, a head-tripod at ~55°). So this is
    # POTENTIAL-BASED (Δcos tilt): rising pays, falling costs, holding anything pays
    # zero. Unfarmable and ungated, and it also rewards catching a stumble mid-walk.
    cfg.rewards["upright_progress"] = RewardTermCfg(func=microduck_mdp.upright_progress, weight=5.0, params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))})
    # z-axis companion to upright_progress: the crouch→stand last mile is mostly a
    # HEIGHT change at modest tilt, where Δcos(tilt) is tiny and the Gaussian
    # upright/pose rewards are flat. Same potential-based construction. A full
    # prone→stand rise collects ≈+2, the crouch→stand mile ≈+1.
    cfg.rewards["height_progress"] = RewardTermCfg(func=microduck_mdp.height_progress, weight=30.0, params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)), "ceiling": 0.115})
    cfg.rewards["com_upward_velocity"] = RewardTermCfg(
        func=microduck_mdp.com_upward_velocity,
        weight=0.0,  # recovery term — ramped in at RECOVERY_ECON_KICKIN_ITER
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
            # Slightly above standing, so the rising reward keeps paying until fully
            # up. The fallen gate, not this ceiling, is what prevents bounce-farming.
            "max_height": 0.125,
            "gate_z_below": 0.0,  # tilt-only gate: never triggers (see above)
            "gate_tilt_above_deg": REWARD_GATE_TILT_DEG,
        },
    )
    # NO impact penalties: recovery pushes off with head/trunk, so a head-impact
    # penalty taxed exactly that strategy and left falling cheaper than getting up.
    # joint_torque_rate_l2 covers landing harshness instead — it penalizes torque
    # CHANGE, not magnitude, so it smooths transfer without blocking the flip.
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(func=microduck_mdp.joint_torque_rate_l2, weight=-2e-3)

    # ── Recovery economics (first-run lessons #3-#5) ──────────────────────────
    # air_time is zeroed while fallen: a robot lying on its trunk can rhythmically
    # tap its feet through the swing window and farm it.
    at = cfg.rewards["air_time"]
    at_params = dict(at.params)
    cfg.rewards["air_time"] = RewardTermCfg(func=microduck_mdp.feet_air_time_upright, weight=at.weight, params={**at_params, "gate_tilt_above_deg": REWARD_GATE_TILT_DEG})
    # Flat tax while fallen: lying still must be strictly worse than trying. Without
    # it, waiting for the fallen_too_long recycle was rational — recovery attempts
    # cost action-rate/torque penalties, waiting cost 0.
    cfg.rewards["fallen_tax"] = RewardTermCfg(
        func=microduck_mdp.fallen_state_penalty,
        weight=0.0,  # ramped to -0.5 at RECOVERY_ECON_KICKIN_ITER (see curriculum)
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
            "gate_tilt_above_deg": REWARD_GATE_TILT_DEG,
            # Hysteresis: recoveries parked in a deep crouch just under the 40° gate,
            # past every recovery term's gate but short of standing. Releasing only on
            # the recovery_success conditions keeps a fall taxed until the stand is
            # FINISHED. Arms only on tilt > 40°, so normal gait is never taxed.
            "release_tilt_below_deg": RECOVERED_UP_TILT_DEG,
            "release_z_above": RECOVERED_UP_Z,
        },
    )
    # One-shot bounty on a COMPLETED recovery (fallen ≥0.5 s → genuinely up), with
    # hysteresis so gate-oscillation pays nothing. The endpoint signal the dense
    # terms lack.
    cfg.rewards["recovery_success"] = RewardTermCfg(
        func=microduck_mdp.recovery_success,
        weight=0.0,  # ramped to +10 at RECOVERY_ECON_KICKIN_ITER (see curriculum)
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)), "fallen_tilt_deg": REWARD_GATE_TILT_DEG, "min_fallen_s": 0.5, "up_tilt_deg": RECOVERED_UP_TILT_DEG, "up_z": RECOVERED_UP_Z},
    )

    # ── Events: prone init ────────────────────────────────────────────────────
    # Spawn just above the ground: a face-down trunk rests at ~0.044 m, and the
    # function defaults (0.20–0.25 m) opened every prone episode with a 15–20 cm
    # free-fall impact.
    cfg.events["random_prone_init"] = EventTermCfg(
        func=microduck_mdp.maybe_set_random_prone_orientation,
        mode="reset",
        params={
            "prone_prob": 0.0,  # ramped by the prone_init_prob curriculum
            "face_down_prob": 1.0,
            "prone_z_min": 0.05,
            "prone_z_max": 0.09,
            "crouch_prob": 0.0,  # ramped by the prone_init_prob curriculum
        },
    )

    # ── Terminations ──────────────────────────────────────────────────────────
    cfg.terminations["fallen_too_long"] = TerminationTermCfg(func=microduck_mdp.fallen_too_long, time_out=False, params={"gate_z_below": TERM_GATE_Z, "gate_tilt_above_deg": TERM_GATE_TILT_DEG, "max_duration_s": FALLEN_TIMEOUT_S})

    # ── Curricula ─────────────────────────────────────────────────────────────
    # Phase 1 → 2: falls become recovery training instead of episode ends.
    if not play:
        cfg.curriculum["fell_over_disable"] = CurriculumTermCfg(func=microduck_mdp.termination_param_curriculum, params={"term_name": "fell_over", "param_stages": [{"step": 0, "params": {"limit_angle": math.radians(70.0)}}, {"step": FELL_OVER_DISABLE_ITER * NUM_STEPS_PER_ENV, "params": {"limit_angle": math.pi}}]})

    cfg.curriculum["prone_init_prob"] = CurriculumTermCfg(func=microduck_mdp.event_param_curriculum, params={"event_name": "random_prone_init", "param_stages": PRONE_RAMP_STAGES})

    # Tax + bounty stay OFF until the walk is established — see
    # RECOVERY_ECON_KICKIN_ITER.
    cfg.curriculum["fallen_tax_weight"] = CurriculumTermCfg(func=microduck_mdp.reward_weight, params={"reward_name": "fallen_tax", "weight_stages": [{"step": 0, "weight": 0.0}, {"step": RECOVERY_ECON_KICKIN_ITER * NUM_STEPS_PER_ENV, "weight": -0.5}]})
    cfg.curriculum["recovery_success_weight"] = CurriculumTermCfg(func=microduck_mdp.reward_weight, params={"reward_name": "recovery_success", "weight_stages": [{"step": 0, "weight": 0.0}, {"step": RECOVERY_ECON_KICKIN_ITER * NUM_STEPS_PER_ENV, "weight": 10.0}]})
    cfg.curriculum["com_upward_weight"] = CurriculumTermCfg(func=microduck_mdp.reward_weight, params={"reward_name": "com_upward_velocity", "weight_stages": [{"step": 0, "weight": 0.0}, {"step": RECOVERY_ECON_KICKIN_ITER * NUM_STEPS_PER_ENV, "weight": 2.0}]})

    return cfg


MicroduckVelStandRlCfg = RslRlOnPolicyRunnerCfg(actor=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True, distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}), critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True), algorithm=PpoWithSymmetryCfg(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.01, num_learning_epochs=5, num_mini_batches=4, learning_rate=1.0e-3, schedule="adaptive", gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0, symmetry_cfg=None), wandb_project="mjlab_microduck", experiment_name="velstand", run_name="velstand", save_interval=250, num_steps_per_env=24, max_iterations=20_000)
