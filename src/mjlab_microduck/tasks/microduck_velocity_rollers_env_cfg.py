"""Microduck velocity environment — roller skate variant"""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_ROLLERS_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp


def make_microduck_velocity_rollers_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create Microduck roller skate velocity tracking environment configuration."""

    # passive_.*: 999.0 means passive wheel joints are included but effectively ignored
    std_standing = {
        r".*hip_yaw.*": 0.05,
        r".*hip_roll.*": 0.05,
        r".*hip_pitch.*": 0.05,
        r".*knee.*": 0.05,
        r".*ankle.*": 0.05,
        r".*neck.*": 0.05,
        r".*head.*": 0.05,
        r".*passive_.*": 999.0,
    }

    std_walking = {
        r".*hip_yaw.*": 0.3,
        r".*hip_roll.*": 0.6,  # loosened: skating requires wide lateral push
        r".*hip_pitch.*": 0.4,
        r".*knee.*": 0.4,
        r".*ankle.*": 0.25,
        r".*neck.*": 0.05,
        r".*head.*": 0.05,
        r".*passive_.*": 999.0,
    }

    std_running = {
        r".*hip_yaw.*": 0.5,
        r".*hip_roll.*": 0.8,  # loosened: skating requires wide lateral push
        r".*hip_pitch.*": 0.8,
        r".*knee.*": 0.8,
        r".*ankle.*": 0.5,
        r".*neck.*": 0.05,
        r".*head.*": 0.05,
        r".*passive_.*": 999.0,
    }

    site_names = ["left_foot", "right_foot"]

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

    cfg = make_velocity_env_cfg()

    cfg.observations["critic"].terms["foot_height"].params[
        "asset_cfg"
    ].site_names = site_names

    # Robot setup
    cfg.scene.entities = {"robot": MICRODUCK_WALK_ROLLERS_ROBOT_CFG}
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)
    cfg.viewer.body_name = "trunk_base"

    # Action configuration
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # === REWARDS ===
    keep = {"pose", "upright", "body_ang_vel", "angular_momentum", "action_rate_l2"}
    for name in list(cfg.rewards.keys()):
        if name not in keep:
            del cfg.rewards[name]

    cfg.rewards["pose"].params["std_standing"] = std_standing
    cfg.rewards["pose"].params["std_walking"] = std_walking
    cfg.rewards["pose"].params["std_running"] = std_running
    cfg.rewards["pose"].params["walking_threshold"] = 0.01
    cfg.rewards["pose"].params["running_threshold"] = 0.5
    cfg.rewards["pose"].weight = 2.0

    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["upright"].weight = 2.0

    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02
    cfg.rewards["action_rate_l2"].weight = -1.0

    cfg.rewards["com_height_target"] = RewardTermCfg(
        func=microduck_mdp.com_height_target,
        weight=2.0,
        params={"target_height_min": 0.0935, "target_height_max": 0.1235},
    )
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": "self_collision"},
    )
    cfg.rewards["feet_flat"] = RewardTermCfg(
        func=microduck_mdp.feet_flat_penalty,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))},
    )
    cfg.rewards["neck_action_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.neck_action_rate_l2, weight=-0.5
    )
    cfg.rewards["neck_joint_pos_l2"] = RewardTermCfg(
        func=microduck_mdp.neck_joint_pos_l2, weight=-0.5
    )
    cfg.rewards["joint_torques_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torques_l2, weight=-1e-3
    )
    # Sole positive task reward — robot must spin wheels to get anything
    cfg.rewards["wheel_speed"] = RewardTermCfg(
        func=microduck_mdp.wheel_speed_reward,
        weight=10.0,
        params={"command_name": "twist", "vel_scale": 0.5},
    )
    # Brake: reward stopping when cmd_x < 0. Silent at cmd_x >= 0 (coast/push).
    cfg.rewards["braking"] = RewardTermCfg(
        func=microduck_mdp.braking_reward,
        weight=1.0,
        params={"command_name": "twist", "vel_std": 0.3},
    )
    # Small air time reward during push to encourage lifting feet in recovery phase.
    cfg.rewards["skating_air_time"] = RewardTermCfg(
        func=microduck_mdp.skating_air_time_reward,
        weight=1.0,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "threshold_min": 0.05,
            "threshold_max": 0.4,
        },
    )
    # Encourage slight forward lean when pushing to counteract backward torque.
    cfg.rewards["forward_lean"] = RewardTermCfg(
        func=microduck_mdp.forward_lean_reward,
        weight=1.5,
        params={"command_name": "twist", "target_pitch": 0.262, "std": 0.1},
    )
    # Heading: cmd[2] = heading error (0=straight, +right/-left).
    # Start low so wheel_speed dominates early; ramp up once skating is learned.
    cfg.rewards["heading_tracking"] = RewardTermCfg(
        func=microduck_mdp.heading_tracking_reward,
        weight=1.0,
        params={"command_name": "twist", "std": 0.8},
    )

    # === EVENTS ===
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )

    del cfg.events["foot_friction"]

    cfg.events["push_robot"] = EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 6.0),
        params={
            "velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    cfg.events["reset_base"].params["pose_range"]["z"] = (0.1335, 0.1435)

    cfg.events["randomize_wheel_friction"] = EventTermCfg(
        func=mdp.randomize_field,
        mode="reset",
        domain_randomization=True,  # required: expands dof_frictionloss to per-env array
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(r"^passive_.*",)),
            "operation": "abs",
            "field": "dof_frictionloss",
            "ranges": (0.000, 0.000),  # ramped up by wheel_friction_curriculum
        },
    )

    cfg.events["randomize_com"] = EventTermCfg(
        func=mdp.randomize_field,
        mode="reset",
        domain_randomization=True,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
            "operation": "add",
            "field": "body_ipos",
            "ranges": (-0.003, 0.003),
        },
    )
    cfg.events["randomize_motor_gains"] = EventTermCfg(
        func=microduck_mdp.randomize_delayed_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "operation": "scale",
            "kp_range": (0.85, 1.15),
            "kd_range": (0.9, 1.1),
        },
    )
    cfg.events["randomize_mass_inertia"] = EventTermCfg(
        func=microduck_mdp.randomize_mass_and_inertia,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
            "scale_range": (0.95, 1.05),
        },
    )
    cfg.events["randomize_imu_orientation"] = EventTermCfg(
        func=microduck_mdp.randomize_imu_orientation,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "max_angle_deg": 1.0,
        },
    )

    # === OBSERVATIONS ===
    del cfg.observations["policy"].terms["base_lin_vel"]
    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel,
        scale=1.0,
    )

    gravity_term_name = "projected_gravity"
    cfg.observations["policy"].terms[gravity_term_name] = deepcopy(
        cfg.observations["policy"].terms[gravity_term_name]
    )
    cfg.observations["policy"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["policy"].terms["base_ang_vel"]
    )
    cfg.observations["policy"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["policy"].terms["base_ang_vel"].delay_max_lag = 3
    cfg.observations["policy"].terms["base_ang_vel"].delay_update_period = 64
    cfg.observations["policy"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["policy"].terms[gravity_term_name].delay_max_lag = 3
    cfg.observations["policy"].terms[gravity_term_name].delay_update_period = 64
    cfg.observations["policy"].terms["base_ang_vel"].noise = Unoise(n_min=-0.024, n_max=0.024)
    cfg.observations["policy"].terms[gravity_term_name].noise = Unoise(n_min=-0.007, n_max=0.007)
    cfg.observations["policy"].terms["joint_pos"].noise = Unoise(n_min=-0.0006, n_max=0.0006)
    cfg.observations["policy"].terms["joint_vel"].noise = Unoise(n_min=-0.024, n_max=0.024)

    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    cfg.observations["policy"].terms["joint_pos"].params["asset_cfg"] = passive_excluded
    cfg.observations["policy"].terms["joint_vel"].params["asset_cfg"] = passive_excluded
    cfg.observations["critic"].terms["joint_pos"].params["asset_cfg"] = deepcopy(passive_excluded)
    cfg.observations["critic"].terms["joint_vel"].params["asset_cfg"] = deepcopy(passive_excluded)

    wheel_cfg = SceneEntityCfg("robot", joint_names=(r"^passive_.*",))
    cfg.observations["critic"].terms["wheel_vel"] = ObservationTermCfg(
        func=mdp.joint_vel_rel,
        scale=1.0,
        params={"asset_cfg": wheel_cfg},
    )

    # === COMMANDS ===
    command: UniformVelocityCommandCfg = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    command.heading_command = False  # RelativeHeadingVelocityCommand handles heading internally
    command.ranges.heading = None    # must be None when heading_command=False
    # cmd_x semantics: 0=coast, >0=push to accelerate, <0=brake to stop
    command.ranges.lin_vel_x = (-0.5, 0.6)
    command.ranges.lin_vel_y = (0.0, 0.0)
    # ang_vel_z range is used as clip limit for cmd[2] = heading error (rad)
    command.ranges.ang_vel_z = (-1.0, 1.0)
    command.viz.z_offset = 0.5
    command.class_type = microduck_mdp.RelativeHeadingVelocityCommand

    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # === CURRICULUM ===
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.5},
                {"step": 250 * 24, "weight": -0.8},
                {"step": 500 * 24, "weight": -1.0},
            ],
        },
    )

    # cfg.curriculum["neck_joint_pos_l2_weight"] = CurriculumTermCfg(
        # func=mdp.reward_weight,
        # params={
            # "reward_name": "neck_joint_pos_l2",
            # "weight_stages": [
                # {"step": 0,          "weight": 0.0},
                # {"step": 1500 * 24,  "weight": -0.05},
                # {"step": 2500 * 24,  "weight": -0.2},
                # {"step": 4000 * 24,  "weight": -0.5},
                # {"step": 6000 * 24,  "weight": -2.0},
            # ],
        # },
    # )

    del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    cfg.curriculum["wheel_friction"] = CurriculumTermCfg(
        func=microduck_mdp.wheel_friction_curriculum,
        params={
            "event_name": "randomize_wheel_friction",
            "ranges_stages": [
                {"step":    0 * 24,  "ranges": (0.000, 0.000)},
                {"step":  750 * 24,  "ranges": (0.001, 0.001)},
                {"step": 1500 * 24,  "ranges": (0.002, 0.002)},
                {"step": 2500 * 24,  "ranges": (0.003, 0.003)},
            ],
        },
    )

    cfg.curriculum["heading_tracking_weight"] = CurriculumTermCfg(
        func=mdp.reward_weight,
        params={
            "reward_name": "heading_tracking",
            "weight_stages": [
                {"step": 0,           "weight": 1.0},   # must match initial RewardTermCfg weight
                {"step": 500  * 24,   "weight": 3.0},   # robot should be skating by now
                {"step": 1000 * 24,   "weight": 5.0},
                {"step": 2000 * 24,   "weight": 8.0},
            ],
        },
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
