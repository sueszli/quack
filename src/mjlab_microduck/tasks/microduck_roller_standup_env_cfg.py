"""Microduck roller standup: get up onto the rollers from prone/supine and hold the stand.

Port of the standup recipe to the roller model, derived from the roller velocity env so
robot, sensors, DR and the 61D obs are inherited (runtime-interchangeable). Differences from
standup: passive wheels INTERLEAVE the joint order (see _LEG_JOINTS, locked by
tests/test_roller_standup_cfg.py), and head/body command slots stay zero-padded with the
head held by neck_joint_pos_l2 (resolved by name). New piece: an inverted rolling-friction
curriculum (braked → free wheels) because free wheels give no grip to push on.
"""

import math
import os

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    RewardTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_rollers_env_cfg import (
    make_microduck_velocity_rollers_env_cfg,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg

# Trunk heights (m), measured by exact kinematics on scene_rollers.xml: standing 0.1407
# (minus ~2 mm load sag, as standup measured), prone rest 0.0752.
ROLLER_STAND_Z = 0.138
ROLLER_PRONE_Z = 0.075

EPISODE_LENGTH_S  = 6.0
NUM_STEPS_PER_ENV = 24

# Play-only override of the face-up spawn fraction. Play rebuilds the env with
# common_step_counter = 0, so the ground_state_mix curriculum sits at stage 0 where
# face_up_prob = 0 and the hardest case is never shown. Env STANDUP_PLAY_FACE_UP:
# 1.0 = all supine, 0.4 = final curriculum mix, unset/none/random = default.
PLAY_FACE_UP = None
# face_down : standing ratio of the final curriculum stage (0.40 / 0.20).
_PLAY_FACE_DOWN_SHARE = 2.0 / 3.0


def _resolve_play_face_up():
    raw = os.environ.get("STANDUP_PLAY_FACE_UP")
    if raw is None:
        return PLAY_FACE_UP
    raw = raw.strip().lower()
    if raw in ("", "none", "random"):
        return None
    try:
        return max(0.0, min(1.0, float(raw)))
    except ValueError:
        print(f"[roller_standup] STANDUP_PLAY_FACE_UP='{raw}' invalid -> default {PLAY_FACE_UP}")
        return PLAY_FACE_UP

# Roller model joint order (passive wheels interleaved): 0-4 left leg, 5-6 LF/LR wheels,
# 7-10 neck/head, 11-15 right leg, 16-17 RF/RR wheels. Standup's [0-4, 9-13] indices are
# WRONG here. Only _LEG_JOINTS is consumed; the neck is resolved by name and wheels by
# the ^passive_.* regex.
_LEG_JOINTS   = [0, 1, 2, 3, 4, 11, 12, 13, 14, 15]
_NECK_JOINTS  = [7, 8, 9, 10]
_WHEEL_JOINTS = [5, 6, 16, 17]

# Skating rewards make no sense on the ground; feet_flat/hip_roll_neutral would fight the
# rise; pose/com_height_target/upright are replaced by the standup targets below.
_SKATING_REWARDS = (
    "wheel_speed",
    "braking",
    "skating_air_time",
    "glide",
    "single_support",
    "gait_symmetry",
    "forward_lean",
    "heading_hold",
    "feet_flat",
    "hip_roll_neutral",
    "pose",
    "com_height_target",
    "upright",
)


def make_microduck_roller_standup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Env « se relever sur rollers » : départ au sol, cible = debout sur roues."""
    cfg = make_microduck_velocity_rollers_env_cfg(play=play)

    cfg.episode_length_s = EPISODE_LENGTH_S

    for name in _SKATING_REWARDS:
        cfg.rewards.pop(name, None)

    # Neutralise the twist slot (tiny non-zero range keeps its inputs alive); replaces the
    # roller env's heading command. head/body slots stay zero-padded.
    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs  = 0.0
    command.heading_command   = False
    command.ranges.heading    = None
    command.resampling_time_range = (EPISODE_LENGTH_S, EPISODE_LENGTH_S * 2)
    command.debug_vis = False
    command.ranges.lin_vel_x = (-0.01, 0.01)
    command.ranges.lin_vel_y = (-0.01, 0.01)
    command.ranges.ang_vel_z = (-0.05, 0.05)
    cfg.commands["twist"] = microduck_mdp.VelocityCommandCommandOnlyCfg(**vars(command))

    # A rare contact (~1/25M steps) NaNs the free joint; sanitise the obs, the env resets.
    for grp in ("actor", "critic"):
        cfg.observations[grp].nan_policy = "sanitize"

    # Standup reward stack with remapped joint indices and heights. A NEW SceneEntityCfg per
    # term: mjlab resolves and mutates them in place, a shared one gives stale indices.
    cfg.rewards["pose_stand_legs"] = RewardTermCfg(
        func=microduck_mdp.pose_target_match,
        weight=8.0,
        params={
            "std": 0.5,
            "joint_indices": _LEG_JOINTS,
            "target_overrides": None,
        },
    )
    cfg.rewards["pose_stand_l1"] = RewardTermCfg(
        func=microduck_mdp.pose_l1_penalty,
        weight=5.0,
        params={
            "joint_indices": _LEG_JOINTS,
            "target_overrides": None,
        },
    )

    # Height: wide Gaussian (pull from the ground), sharp Gaussian (last cm), strong L1 so
    # "stay on the ground" is net negative.
    cfg.rewards["height_stand"] = RewardTermCfg(
        func=microduck_mdp.height_target_gaussian,
        weight=4.0,
        params={
            "std": 0.04,
            "target_height": ROLLER_STAND_Z,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    cfg.rewards["height_stand_sharp"] = RewardTermCfg(
        func=microduck_mdp.height_target_gaussian,
        weight=4.0,
        params={
            "std": 0.015,
            "target_height": ROLLER_STAND_Z,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    cfg.rewards["height_stand_l1"] = RewardTermCfg(
        func=microduck_mdp.height_l1_penalty,
        weight=30.0,
        params={
            "target_height": ROLLER_STAND_Z,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # Pays for the rising MOTION; cutoff 10 mm above target or the policy parks at the cutoff.
    cfg.rewards["com_upward_velocity"] = RewardTermCfg(
        func=microduck_mdp.com_upward_velocity,
        weight=3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
            "max_height": ROLLER_STAND_Z + 0.010,
        },
    )
    # |a_z| penalty. POSITIVE weight: trunk_vertical_accel_penalty already returns -|a_z|;
    # the inherited -0.02 double-negated into a reward for violence. Kept small because
    # |a_z| is unavoidable when flipping from the back; joint_torque_rate_l2 does the damping.
    cfg.rewards["gentle_rise"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=+0.02,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )

    # Two-layer upright: cos(tilt) pulls from lying, the height-gated Gaussian kills the
    # back-lean near vertical.
    cfg.rewards["upright_linear"] = RewardTermCfg(
        func=microduck_mdp.body_upright_linear,
        weight=6.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    cfg.rewards["upright_sharp"] = RewardTermCfg(
        func=microduck_mdp.upright_gaussian_at_height,
        weight=6.0,
        params={
            "std": 0.3,
            "height_low": ROLLER_PRONE_Z,
            "height_high": ROLLER_STAND_Z,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # Multiplicative height × upright × pose kills 2-of-3 compromises; broad stds so it is
    # visible during the rise (tight stds scored ~5e-5).
    cfg.rewards["standing_composite"] = RewardTermCfg(
        func=microduck_mdp.standing_composite_score,
        weight=15.0,
        params={
            "target_height": ROLLER_STAND_Z,
            "height_std": 0.04,
            "upright_std": 0.40,
            "pose_std": 0.40,
            "joint_indices": _LEG_JOINTS,
            "target_overrides": None,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # Anti-jitter on torque CHANGE: the one damper that doesn't block the flip from the
    # back. If still violent, raise THIS, not body_ang_vel/action_rate (motion-blockers).
    # Raw |Δτ|² ≈ 0.1 at convergence, so contribution ≈ 0.1 × |weight|.
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torque_rate_l2,
        weight=-0.2,
    )

    # No head-impact penalty: this robot PIVOTS on its head to get up from the back, and
    # penalising the pivot made the policy lie still. If a head slam returns, gate any
    # penalty on height so the ground phase is spared.

    # Added LAST so it overrides reset_base / reset_robot_joints. The standing bucket is
    # required or the policy learns to rise but not to HOLD. Probabilities = curriculum
    # stage 0.
    cfg.events["set_ground_state"] = EventTermCfg(
        func=microduck_mdp.set_random_ground_state,
        mode="reset",
        params={
            "face_down_prob": 0.50,
            "face_up_prob":   0.00,
            "sitting_prob":   0.00,
            "standing_prob":  0.50,
            "sitting_joint_overrides": None,
            # One z range for prone and supine: 0.076 avoids belly interpenetration (rest
            # 0.0752) at the cost of the back starting ~3 cm above its 0.0475 rest.
            "prone_z_min":    0.076,
            "prone_z_max":    0.09,
            "standing_z_min": 0.134,
            "standing_z_max": 0.144,
            # Also applies to standing spawns (they reuse the sitting quaternion) — wanted.
            "sitting_tilt_max": math.radians(10),
        },
    )

    cfg.terminations.pop("fell_over", None)

    # Spawn mix easy → hard; a flat mix from step 0 leaves the back under-trained.
    cfg.curriculum["ground_state_mix"] = CurriculumTermCfg(
        func=microduck_mdp.event_param_curriculum,
        params={
            "event_name": "set_ground_state",
            "param_stages": [
                {"step": 0, "params": {
                    "standing_prob": 0.50, "sitting_prob": 0.00,
                    "face_down_prob": 0.50, "face_up_prob": 0.00}},
                {"step": 600 * NUM_STEPS_PER_ENV, "params": {
                    "standing_prob": 0.35, "sitting_prob": 0.00,
                    "face_down_prob": 0.45, "face_up_prob": 0.20}},
                {"step": 1500 * NUM_STEPS_PER_ENV, "params": {
                    "standing_prob": 0.25, "sitting_prob": 0.00,
                    "face_down_prob": 0.40, "face_up_prob": 0.35}},
                {"step": 2500 * NUM_STEPS_PER_ENV, "params": {
                    "standing_prob": 0.20, "sitting_prob": 0.00,
                    "face_down_prob": 0.40, "face_up_prob": 0.40}},
            ],
        },
    )

    # Play override: also remove the curriculum, which runs BEFORE reset events and would
    # rewrite the probabilities with stage 0.
    if play:
        play_face_up = _resolve_play_face_up()
        if play_face_up is not None:
            remainder = 1.0 - play_face_up
            cfg.events["set_ground_state"].params.update({
                "face_up_prob":    play_face_up,
                "face_down_prob":  remainder * _PLAY_FACE_DOWN_SHARE,
                "standing_prob":   remainder * (1.0 - _PLAY_FACE_DOWN_SHARE),
                "sitting_prob":    0.00,
            })
            del cfg.curriculum["ground_state_mix"]

    # Inverted rolling-friction curriculum (braked → free): bootstrap the motion with
    # near-locked wheels (≈ feet), then impose real rolling. If standing_composite collapses
    # at a stage, the gripping-feet motion doesn't transfer. Only checkpoints AFTER the last
    # stage (iter 4000+) are deployable — earlier ones rely on friction the robot lacks.
    _WHEEL_FRICTION_STAGE0 = (0.0500, 0.0500)
    cfg.curriculum["wheel_friction"] = CurriculumTermCfg(
        func=microduck_mdp.wheel_friction_curriculum,
        params={
            "event_name": "randomize_wheel_friction",
            "ranges_stages": [
                {"step": 0,                        "ranges": _WHEEL_FRICTION_STAGE0},
                {"step": 1000 * NUM_STEPS_PER_ENV, "ranges": (0.0200, 0.0200)},
                {"step": 2000 * NUM_STEPS_PER_ENV, "ranges": (0.0080, 0.0080)},
                {"step": 3000 * NUM_STEPS_PER_ENV, "ranges": (0.0030, 0.0030)},
                {"step": 4000 * NUM_STEPS_PER_ENV, "ranges": (0.0015, 0.0015)},
            ],
        },
    )
    # Keep the event default consistent with stage 0 in case the curriculum is removed.
    cfg.events["randomize_wheel_friction"].params["ranges"] = _WHEEL_FRICTION_STAGE0

    # Standup's ramp, not the roller env's -2.0: action_rate is a motion-blocker for the flip.
    cfg.rewards["action_rate_l2"].weight = -0.6
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0,                       "weight": -0.4},
                {"step": 250 * NUM_STEPS_PER_ENV, "weight": -0.8},
                {"step": 500 * NUM_STEPS_PER_ENV, "weight": -1.0},
            ],
        },
    )

    # Pushes from step 0 disrupt the rise bootstrap; ramp them like standup.
    cfg.curriculum["push_magnitude"] = CurriculumTermCfg(
        func=microduck_mdp.push_curriculum,
        params={
            "event_name": "push_robot",
            "push_stages": [
                {"step": 0, "velocity_range": {
                    "x": (0.0, 0.0), "y": (0.0, 0.0)}},
                {"step": 500 * NUM_STEPS_PER_ENV, "velocity_range": {
                    "x": (-0.08, 0.08), "y": (-0.08, 0.08)}},
                {"step": 1000 * NUM_STEPS_PER_ENV, "velocity_range": {
                    "x": (-0.2, 0.2), "y": (-0.2, 0.2)}},
            ],
        },
    )

    return cfg


MicroduckRollerStandUpRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,  # normalizer must be baked into the ONNX (export.py)
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
    experiment_name="roller_standup",
    run_name="roller_standup",
    save_interval=250,
    num_steps_per_env=NUM_STEPS_PER_ENV,
    max_iterations=15_000,
)
