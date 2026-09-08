"""Microduck roller SWIZZLE environment — clean classic swizzle.

A separate roller task producing a CLASSIC SWIZZLE: both blades stay on the ground,
the legs spread out and pull back in SYMMETRICALLY (hourglass pattern), propelling
the duck forward. Simpler / more stable alternative to the alternating stride
(`Mjlab-Velocity-Flat-MicroDuck-Rollers`), which does not transfer well to the real
robot. The stride env is left untouched.

Approach A (see docs/superpowers/specs/2026-07-23-swizzle-env-design.md): the base
roller recipe NATURALLY converges to a swizzle, so we reuse the stride env wholesale
(robot, 61D obs, command, full DR, curricula, sim2real — deploys identically with
`--roller`) and only swap the reward recipe:
  - REMOVE the anti-swizzle / stride terms.
  - ADD leg_symmetry (legs mirror) + grounded (both blades down).
"""

import dataclasses

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg, ObservationTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_rollers_env_cfg import (
    MicroduckRollersRlCfg,
    make_microduck_velocity_rollers_env_cfg,
)

# Stride / anti-swizzle rewards to drop for the swizzle task.
_ANTI_SWIZZLE = ("single_support", "glide", "skating_air_time", "gait_symmetry", "hip_roll_neutral")


def make_microduck_velocity_swizzle_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Roller swizzle env: the stride env minus its anti-swizzle terms, plus symmetry
    and grounded rewards. Everything else (robot, obs, command, DR) is identical."""
    cfg = make_microduck_velocity_rollers_env_cfg(play=play)

    for name in _ANTI_SWIZZLE:
        if name in cfg.rewards:
            del cfg.rewards[name]

    # Legs mirror each other (the swizzle's defining symmetry).
    cfg.rewards["leg_symmetry"] = RewardTermCfg(
        func=microduck_mdp.leg_symmetry_reward,
        weight=2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # Keep both blades on the ground (classic swizzle: no lifting).
    cfg.rewards["grounded"] = RewardTermCfg(
        func=microduck_mdp.grounded_reward,
        weight=1.0,
        params={"sensor_name": "feet_ground_contact", "command_name": "twist"},
    )

    # cmd_x < 0 means GO BACKWARD here (not brake): braking reward dropped, range symmetric.
    cfg.rewards["wheel_speed"].params["bidirectional"] = True
    if "braking" in cfg.rewards:
        del cfg.rewards["braking"]
    cfg.commands["twist"].ranges.lin_vel_x = (-0.6, 0.6)

    # Heading: straight first (heading_hold), then follow (heading_tracking) via curriculum.
    # cmd[2] = heading error, clipped to ±0.5: ±1.0 turned too violently on the robot.
    cfg.commands["twist"].ranges.ang_vel_z = (-0.5, 0.5)

    cfg.rewards["heading_tracking"] = RewardTermCfg(
        func=microduck_mdp.heading_tracking_reward,
        weight=0.0,  # ramped up by the curriculum below (must match its step-0 value)
        params={"command_name": "twist", "std": 0.5},
    )

    cfg.curriculum["heading_hold_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "heading_hold",
            "weight_stages": [
                {"step": 0,          "weight": 1.0},   # must match heading_hold's initial weight
                {"step": 1000 * 24,  "weight": 1.0},   # hold straight while the swizzle solidifies
                {"step": 1750 * 24,  "weight": 0.5},
                {"step": 2500 * 24,  "weight": 0.0},
            ],
        },
    )
    cfg.curriculum["heading_tracking_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "heading_tracking",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},
                {"step": 1000 * 24,  "weight": 0.0},   # straight-only until here
                {"step": 1750 * 24,  "weight": 1.5},
                {"step": 2500 * 24,  "weight": 3.0},
            ],
        },
    )

    # Head pose command; ranges widened by curriculum.
    cfg.commands["head_pose"] = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=(2.0, 5.0),
        ranges=(
            (-0.05, 0.05),    # neck_pitch
            (-0.05, 0.05),    # head_pitch
            (-0.07, 0.07),    # head_yaw
            (-0.015, 0.015),  # head_roll
        ),
    )

    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "head_pose"},
        )

    # Ramped in late so it doesn't disturb the swizzle before it's solid.
    cfg.rewards["head_pose_tracking"] = RewardTermCfg(
        func=microduck_mdp.head_pose_tracking,
        weight=0.0,
        params={"command_name": "head_pose", "std": 0.5},
    )

    # Remove the HOME-pullers on neck/head that would fight head_pose_tracking.
    if "neck_joint_pos_l2" in cfg.rewards:
        del cfg.rewards["neck_joint_pos_l2"]
    for std_key in ["std_standing", "std_walking", "std_running"]:
        if std_key in cfg.rewards["pose"].params:
            std_dict = cfg.rewards["pose"].params[std_key]
            cfg.rewards["pose"].params[std_key] = {
                k: v for k, v in std_dict.items()
                if "neck" not in k and "head" not in k and "passive" not in k
            }
    cfg.rewards["pose"].params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=(r"^(?!passive_|.*neck.*|.*head.*).*",)
    )

    cfg.curriculum["head_pose_tracking_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "head_pose_tracking",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},   # must match initial weight
                {"step": 1500 * 24,  "weight": 0.0},   # head off while swizzle solidifies
                {"step": 2250 * 24,  "weight": 2.0},
                {"step": 3000 * 24,  "weight": 4.0},
            ],
        },
    )
    cfg.curriculum["head_pose_range"] = CurriculumTermCfg(
        func=microduck_mdp.pose_command_range_curriculum,
        params={
            "command_name": "head_pose",
            "range_stages": [
                # step,               ((neck_pitch), (head_pitch), (head_yaw),  (head_roll))
                {"step": 0,          "ranges": ((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015))},
                {"step": 1500 * 24,  "ranges": ((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015))},
                {"step": 2250 * 24,  "ranges": ((-0.55, 0.55), (-0.55, 0.55), (-0.70, 0.70), (-0.15, 0.15))},
                {"step": 3000 * 24,  "ranges": ((-1.10, 1.10), (-1.10, 1.10), (-1.40, 1.40), (-0.31, 0.31))},
            ],
        },
    )

    return cfg


MicroduckSwizzleRlCfg = dataclasses.replace(
    MicroduckRollersRlCfg,
    experiment_name="velocity_swizzle",
    run_name="velocity_swizzle",
)
