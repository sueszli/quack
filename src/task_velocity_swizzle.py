"""Microduck roller SWIZZLE environment — clean classic swizzle.

Both blades stay on the ground and the legs spread out and pull back in
SYMMETRICALLY (hourglass pattern), propelling the duck forward. A more stable
alternative to the alternating stride of `Mjlab-Velocity-Flat-MicroDuck-Rollers`,
which does not transfer well to the real robot.

The base roller recipe naturally converges to a swizzle, so this reuses the stride
env wholesale (robot, 61D obs, command, DR, curricula — deploys identically with
`--roller`) and only swaps the reward recipe.
"""

import dataclasses

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg, ObservationTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp

from . import task_mdp as microduck_mdp
from .task_velocity_rollers import MicroduckRollersRlCfg, make_microduck_velocity_rollers_env_cfg

# Stride / anti-swizzle rewards to drop for the swizzle task.
_ANTI_SWIZZLE = ("single_support", "glide", "skating_air_time", "gait_symmetry", "hip_roll_neutral")


def make_microduck_velocity_swizzle_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """The stride env minus its anti-swizzle terms, plus symmetry and grounded."""
    cfg = make_microduck_velocity_rollers_env_cfg(play=play)

    for name in _ANTI_SWIZZLE:
        if name in cfg.rewards:
            del cfg.rewards[name]

    # Legs mirror each other (the swizzle's defining symmetry).
    cfg.rewards["leg_symmetry"] = RewardTermCfg(func=microduck_mdp.leg_symmetry_reward, weight=2.0, params={"asset_cfg": SceneEntityCfg("robot")})
    # Keep both blades on the ground (classic swizzle: no lifting).
    cfg.rewards["grounded"] = RewardTermCfg(func=microduck_mdp.grounded_reward, weight=1.0, params={"sensor_name": "feet_ground_contact", "command_name": "twist"})

    # cmd_x < 0 means GO BACKWARD here, NOT brake (unlike the stride env): wheel_speed
    # pays spin in the commanded direction, braking is dropped, and the range is
    # symmetric. cmd_x ≈ 0 is how you stop. grounded uses |cmd_x|, so it holds the
    # blades down both ways.
    cfg.rewards["wheel_speed"].params["bidirectional"] = True
    if "braking" in cfg.rewards:
        del cfg.rewards["braking"]
    cfg.commands["twist"].ranges.lin_vel_x = (-0.6, 0.6)

    # Go STRAIGHT first, then FOLLOW a commanded direction: the curriculum below
    # swaps heading_hold out for heading_tracking. cmd[2] is the heading-error clip;
    # ±0.5 rather than ±1.0 bounds the OBSERVED error so turn correction is gentler
    # (a ±1.0-trained policy turned so violently it needed --max-angular-vel 0.3).
    # Any heading is still reachable — the error just saturates.
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
                {"step": 0, "weight": 1.0},  # must match heading_hold's initial weight
                {"step": 1000 * 24, "weight": 1.0},  # hold straight while the swizzle solidifies
                {"step": 1750 * 24, "weight": 0.5},
                {"step": 2500 * 24, "weight": 0.0},
            ],
        },
    )
    cfg.curriculum["heading_tracking_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "heading_tracking",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": 1000 * 24, "weight": 0.0},  # straight-only until here
                {"step": 1750 * 24, "weight": 1.5},
                {"step": 2500 * 24, "weight": 3.0},
            ],
        },
    )

    # Head-pose control (Y button): 4D deltas from HOME — [neck_pitch, head_pitch,
    # head_yaw, head_roll]. Ranges start small and are widened by the curriculum.
    cfg.commands["head_pose"] = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=(2.0, 5.0),
        ranges=(
            (-0.05, 0.05),  # neck_pitch
            (-0.05, 0.05),  # head_pitch
            (-0.07, 0.07),  # head_yaw
            (-0.015, 0.015),  # head_roll (tighter — small mechanical range)
        ),
    )

    # Replaces the roller env's zero_command_padding. body_command stays zero-padded.
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "head_pose"})

    # Weight 0 here — ramped in late by the curriculum, so it doesn't disturb the
    # swizzle before it's solid.
    cfg.rewards["head_pose_tracking"] = RewardTermCfg(func=microduck_mdp.head_pose_tracking, weight=0.0, params={"command_name": "head_pose", "std": 0.5})

    # Remove the two HOME-pullers that would fight head_pose_tracking: this one, and
    # the pose reward's neck/head coverage (scoped to LEG joints below). The std
    # dicts must be filtered to match the scoped asset_cfg.
    if "neck_joint_pos_l2" in cfg.rewards:
        del cfg.rewards["neck_joint_pos_l2"]
    for std_key in ["std_standing", "std_walking", "std_running"]:
        if std_key in cfg.rewards["pose"].params:
            std_dict = cfg.rewards["pose"].params[std_key]
            cfg.rewards["pose"].params[std_key] = {k: v for k, v in std_dict.items() if "neck" not in k and "head" not in k and "passive" not in k}
    cfg.rewards["pose"].params["asset_cfg"] = SceneEntityCfg("robot", joint_names=(r"^(?!passive_|.*neck.*|.*head.*).*",))

    # Head control is added on top of an already-stable swizzle.
    cfg.curriculum["head_pose_tracking_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "head_pose_tracking",
            "weight_stages": [
                {"step": 0, "weight": 0.0},  # must match initial weight
                {"step": 1500 * 24, "weight": 0.0},  # head off while swizzle solidifies
                {"step": 2250 * 24, "weight": 2.0},
                {"step": 3000 * 24, "weight": 4.0},
            ],
        },
    )
    # Widens over the SAME window as the weight ramp above.
    cfg.curriculum["head_pose_range"] = CurriculumTermCfg(
        func=microduck_mdp.pose_command_range_curriculum,
        params={
            "command_name": "head_pose",
            "range_stages": [
                # step,               ((neck_pitch), (head_pitch), (head_yaw),  (head_roll))
                {"step": 0, "ranges": ((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015))},
                {"step": 1500 * 24, "ranges": ((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015))},
                {"step": 2250 * 24, "ranges": ((-0.55, 0.55), (-0.55, 0.55), (-0.70, 0.70), (-0.15, 0.15))},
                {"step": 3000 * 24, "ranges": ((-1.10, 1.10), (-1.10, 1.10), (-1.40, 1.40), (-0.31, 0.31))},
            ],
        },
    )

    return cfg


MicroduckSwizzleRlCfg = dataclasses.replace(MicroduckRollersRlCfg, experiment_name="velocity_swizzle", run_name="velocity_swizzle")
