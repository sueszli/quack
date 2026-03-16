"""Microduck velocity environment — roller skate variant.

Inherits everything from make_microduck_velocity_env_cfg() and applies only
the minimal changes needed for the roller-skate robot.
"""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.manager_term_config import EventTermCfg, ObservationTermCfg, RewardTermCfg
from mjlab.tasks.velocity import mdp
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_ROLLERS_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from mjlab_microduck.tasks.microduck_velocity_env_cfg import make_microduck_velocity_env_cfg


def make_microduck_velocity_rollers_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create Microduck roller skate velocity tracking environment configuration."""

    # Roller-specific foot contact sensor (different body pattern from walk robot)
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(roller_foot1|roller_foot2)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    cfg = make_microduck_velocity_env_cfg(play=play)

    # Swap robot and sensors
    cfg.scene.entities = {"robot": MICRODUCK_WALK_ROLLERS_ROBOT_CFG}
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)

    # Roller robot sits higher than walk robot
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.1335, 0.1435)

    # Rollers have no foot friction geoms
    del cfg.events["foot_friction"]

    # Warm-start a fraction of episodes with forward velocity to seed wheel_speed gradient
    cfg.events["reset_forward_velocity"] = EventTermCfg(
        func=microduck_mdp.reset_with_forward_velocity,
        mode="reset",
        params={
            "velocity_range": (0.5, 1.5),
            "fraction_stages": [
                {"step": 0,          "fraction": 0.2},
                {"step": 2000 * 24,  "fraction": 0.1},
                {"step": 4000 * 24,  "fraction": 0.0},
            ],
        },
    )

    # Adjust CoM height target for roller robot (sits ~1cm higher than walk robot)
    cfg.rewards["com_height_target"].params["target_height_min"] = 0.0935
    cfg.rewards["com_height_target"].params["target_height_max"] = 0.1235

    # Exclude passive wheel joints from policy obs and any joint-indexed rewards.
    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    cfg.observations["policy"].terms["joint_pos"].params["asset_cfg"] = passive_excluded
    cfg.observations["policy"].terms["joint_vel"].params["asset_cfg"] = passive_excluded
    cfg.observations["critic"].terms["joint_pos"].params["asset_cfg"] = deepcopy(passive_excluded)
    cfg.observations["critic"].terms["joint_vel"].params["asset_cfg"] = deepcopy(passive_excluded)

    # Roller skating requires lateral foot sliding — slip penalty is counterproductive
    del cfg.rewards["foot_slip"]

    # Pose reward uses joint_names=".*" by default → would include 18 joints but std
    # dicts only have 14 entries → size mismatch. Restrict to non-passive joints.
    cfg.rewards["pose"].params["asset_cfg"] = deepcopy(passive_excluded)

    # Wheel velocities in critic only
    wheel_cfg = SceneEntityCfg("robot", joint_names=(r"^passive_.*",))
    cfg.observations["critic"].terms["wheel_vel"] = ObservationTermCfg(
        func=mdp.joint_vel_rel,
        scale=1.0,
        params={"asset_cfg": wheel_cfg},
    )

    # Forward-only commands: always positive, no sideways, no turning
    command: UniformVelocityCommandCfg = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    command.ranges.lin_vel_x = (0.5, 1.5)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)

    # stillness_at_zero_command rewards being still — counterproductive for skating
    del cfg.rewards["stillness_at_zero_command"]

    # Walk-specific rewards — irrelevant for skating, zero them out
    cfg.rewards["foot_clearance"].weight = 0.0
    cfg.rewards["foot_swing_height"].weight = 0.0
    cfg.rewards["air_time"].weight = 0.0
    cfg.rewards["soft_landing"].weight = 0.0

    # Reduce pose weight — standing still shouldn't be too comfortable
    cfg.rewards["pose"].weight = 0.5  # was 2.0

    cfg.rewards["feet_flat"] = RewardTermCfg(
        func=microduck_mdp.feet_flat_penalty,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))},
    )

    # Reward spinning wheels forward
    cfg.rewards["wheel_speed"] = RewardTermCfg(
        func=microduck_mdp.wheel_speed_reward,
        weight=50.0,
        params={"command_name": "twist", "vel_scale": 0.5},
    )

    return cfg


MicroduckRollersRlCfg = RslRlOnPolicyRunnerCfg(
    policy=RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        activation="elu",
    ),
    algorithm=RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.03,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="velocity_rollers",
    run_name="velocity_rollers",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=50_000,
)
