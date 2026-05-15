"""Microduck Pose environment: stand still, follow head + body pose commands.

This is a dedicated "animation playback" policy. The robot's only jobs are:
  1. Stay upright at the nominal standing height.
  2. Track the 4D head_pose command (neck_pitch, head_pitch, head_yaw, head_roll).
  3. Track the 6D body_pose command (x, y, z, roll, pitch, yaw).
  4. Be modestly robust to small pushes / IMU noise / CoM randomization.

NO walking. NO fall recovery. The fundamental conflict that prevented velstand
from learning body tracking (walking gait demands rhythmic joint motion ↔
pose tracking demands joint hold) is eliminated by removing the walking task
entirely. Switch between this policy and a velocity-tracking one at deployment
based on what behavior you want.

Obs shape stays at 61D (unified runtime obs); the twist slot is fed with the
near-zero command this env samples.
"""

import math
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.manager_term_config import (
    CurriculumTermCfg,
    RewardTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
)
from mjlab.tasks.velocity import mdp as velocity_mdp

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG


NUM_STEPS_PER_ENV = 24

# Final body-pose command ranges (reached at end of curriculum).
BODY_CMD_MAX_XY    = 0.02                # ±20 mm lateral/forward
BODY_CMD_MAX_Z     = 0.03                # ±30 mm height
BODY_CMD_MAX_ANGLE = math.radians(30)    # ±30° per Euler axis

# Final head-pose command ranges — same per-joint mechanical caps as vel env.
HEAD_NECK_PITCH_MAX  = 1.10
HEAD_HEAD_PITCH_MAX  = 1.10
HEAD_HEAD_YAW_MAX    = 1.40
HEAD_HEAD_ROLL_MAX   = 0.31

# Trunk z that the body-pose z command is measured against. Matches vel env
# reset_base z (0.12–0.13), which is the trunk height at steady upright stance.
NOMINAL_HEIGHT = 0.12


def make_microduck_pose_env_cfg(play: bool = False, rough: bool = False) -> ManagerBasedRlEnvCfg:
    # Inherit walking obs / action / robot setup from vel env so the runtime
    # 61D obs layout (and ONNX export) stay identical across all microduck
    # policies. We then aggressively strip everything walking-related.
    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)

    # ── COMMANDS ─────────────────────────────────────────────────────────────
    # Twist command pinned to ~0. We can't fully zero it because input neurons
    # for the command obs would die — keep a tiny range (±1 mm/s lin, ±0.05 rad/s
    # ang) so the slot stays alive but never asks the policy to actually move.
    twist = cfg.commands["twist"]
    twist.rel_standing_envs = 1.0   # every env is "standing"
    twist.rel_heading_envs  = 0.0
    twist.ranges.lin_vel_x = (-0.001, 0.001)
    twist.ranges.lin_vel_y = (-0.001, 0.001)
    twist.ranges.ang_vel_z = (-0.05, 0.05)
    twist.heading_command = False
    twist.ranges.heading = None

    # Head and body command terms come from the vel env (same UniformPoseCommand
    # objects, narrow initial "kept-alive" ranges); the curricula below widen
    # them.

    # ── REWARDS: walking → off, pose tracking → primary ──────────────────────
    # Zero out everything related to forward locomotion.
    for name in (
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "soft_landing",
        "angular_momentum",
        "stillness_at_zero_command",   # cmd is always ~0; this is just a free reward, drop
    ):
        if name in cfg.rewards:
            cfg.rewards[name].weight = 0.0

    # Keep the structural rewards that prevent the policy from degenerating:
    # upright, leg pose (head/neck excluded → command-driven), com height,
    # action smoothness, dof limits, self collisions, torque cost.
    # `pose` already excludes head/neck (via regex set in vel env).
    cfg.rewards["upright"].weight = 2.0

    # com_height_target wide enough to allow z body commands (nominal ± max_z
    # plus a margin) without the cliff that broke velstand.
    cfg.rewards["com_height_target"].params["target_height_min"] = NOMINAL_HEIGHT - BODY_CMD_MAX_Z - 0.01
    cfg.rewards["com_height_target"].params["target_height_max"] = NOMINAL_HEIGHT + BODY_CMD_MAX_Z + 0.01
    cfg.rewards["com_height_target"].weight = 2.0

    # Head pose tracking: already wired by vel env (weight 3, std 0.5).
    # Bump it slightly to make sure it stays dominant on the head axes.
    cfg.rewards["head_pose_tracking"].weight = 4.0

    # Body pose tracking: vel env left it at weight 0.05 (kept-alive only).
    # Replace with the locomotion-relative reward — no vel gating since the
    # robot is always standing, but it lets us use the same function and the
    # feet-centroid relative xy/yaw computations work fine when stationary.
    cfg.rewards["body_pose_tracking"] = RewardTermCfg(
        func=microduck_mdp.body_pose_tracking_locomotion,
        weight=6.0,
        params={
            "command_name": "body_pose",
            "nominal_height": NOMINAL_HEIGHT,
            # std = max/2 keeps the gradient alive at full miss while making
            # the baseline-at-cmd=0 reasonable (matches head_pose recipe).
            "xy_std":    BODY_CMD_MAX_XY    / 2.0,
            "z_std":     BODY_CMD_MAX_Z     / 2.0,
            "angle_std": BODY_CMD_MAX_ANGLE / 2.0,
            # Track ALL 6 axes here — no walking conflict, the xy ↔ pitch/roll
            # coupling is just a constraint the policy needs to navigate, not
            # an active fight with another reward.
            "axis_weights": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            # No vel gate — we're always standing.
            "feet_cfg":  SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
        },
    )

    # ── CURRICULA ────────────────────────────────────────────────────────────
    # Drop the curricula that ramp walking-related stuff. Keep the head_pose
    # range ramp + add a body_pose range ramp + a tracking weight ramp.

    # standing_envs is forced to 1.0 always — override the inherited ramp.
    cfg.curriculum["standing_envs"].params["standing_stages"] = [
        {"step": 0, "rel_standing_envs": 1.0},
    ]

    # velocity_command_ranges curriculum becomes a no-op (twist range is
    # already tiny and we don't want it to widen).
    cfg.curriculum["velocity_command_ranges"].params["velocity_stages"] = [
        {"step": 0, "lin_vel_range": 0.001, "ang_vel_range": 0.05},
    ]

    # head_pose range curriculum — keep the per-joint final caps from vel env.
    # Inherited as-is; no override needed.

    # body_pose range curriculum — widen aggressively (this is the main signal).
    cfg.curriculum["body_pose_range"].params["range_stages"] = [
        {"step": 0, "ranges": (
            (-0.005, 0.005), (-0.005, 0.005), (-0.005, 0.005),
            (-math.radians(3), math.radians(3)),
            (-math.radians(3), math.radians(3)),
            (-math.radians(3), math.radians(3)),
        )},
        {"step": 250 * NUM_STEPS_PER_ENV, "ranges": (
            (-0.008, 0.008), (-0.008, 0.008), (-0.012, 0.012),
            (-math.radians(10), math.radians(10)),
            (-math.radians(10), math.radians(10)),
            (-math.radians(10), math.radians(10)),
        )},
        {"step": 750 * NUM_STEPS_PER_ENV, "ranges": (
            (-0.015, 0.015), (-0.015, 0.015), (-0.020, 0.020),
            (-math.radians(20), math.radians(20)),
            (-math.radians(20), math.radians(20)),
            (-math.radians(20), math.radians(20)),
        )},
        {"step": 1500 * NUM_STEPS_PER_ENV, "ranges": (
            (-BODY_CMD_MAX_XY, BODY_CMD_MAX_XY),
            (-BODY_CMD_MAX_XY, BODY_CMD_MAX_XY),
            (-BODY_CMD_MAX_Z,  BODY_CMD_MAX_Z),
            (-BODY_CMD_MAX_ANGLE, BODY_CMD_MAX_ANGLE),
            (-BODY_CMD_MAX_ANGLE, BODY_CMD_MAX_ANGLE),
            (-BODY_CMD_MAX_ANGLE, BODY_CMD_MAX_ANGLE),
        )},
    ]

    # ── EVENTS ───────────────────────────────────────────────────────────────
    # Pushes: keep at the vel env default (±0.3 m/s) — provides modest
    # disturbance robustness without being a primary objective. No prone init.
    # (vel env already sets push_robot up.)

    return cfg


MicroduckPoseRlCfg = RslRlOnPolicyRunnerCfg(
    policy=RslRlPpoActorCriticCfg(
        init_noise_std=0.5,    # less exploration needed than a walking policy
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        activation="elu",
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
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
    experiment_name="pose",
    run_name="pose",
    save_interval=250,
    num_steps_per_env=NUM_STEPS_PER_ENV,
    max_iterations=5_000,
)
