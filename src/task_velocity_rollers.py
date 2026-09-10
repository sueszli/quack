# Symmetry — OFF: SYMMETRY_CFG's obs permutation is hardcoded for the old 51D
# layout and breaks on the 61D obs (same situation as all other v1.5+ envs).
ENABLE_SYMMETRY = False

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import CurriculumTermCfg, EventTermCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from . import task_dr
from . import task_mdp as microduck_mdp
from .robot import MICRODUCK_WALK_ROLLERS_ROBOT_CFG
from .task_symmetry import SYMMETRY_CFG, PpoWithSymmetryCfg
from .task_velocity import HEAD_BODY_NAMES, LOCAL_CHECKPOINTS_ONLY

DR = task_dr.ROLLER_DR
ENCODER_BIAS_RANGE = DR.encoder_bias_range


def make_microduck_velocity_rollers_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    # passive_.*: 999.0 → passive wheel joints are matched but effectively ignored
    std_standing = {r".*hip_yaw.*": 0.05, r".*hip_roll.*": 0.05, r".*hip_pitch.*": 0.05, r".*knee.*": 0.05, r".*ankle.*": 0.05, r".*neck.*": 0.05, r".*head.*": 0.05, r".*passive_.*": 999.0}

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

    # 2026-07 model: the roller_blade bodies were merged into the ankles (blade
    # mesh is now a visual geom on ankle_{l,r}_v1); the tires hang directly off
    # the ankles. Each ankle subtree's only collision geoms are its two tires,
    # so this keeps the old per-foot semantics: 2 slots, left first.
    feet_ground_cfg = ContactSensorCfg(name="feet_ground_contact", primary=ContactMatch(mode="subtree", pattern=r"^(ankle_l_v1|ankle_r_v1)$", entity="robot"), secondary=ContactMatch(mode="body", pattern="terrain"), fields=("found", "force"), reduce="netforce", num_slots=1, track_air_time=True)

    self_collision_cfg = ContactSensorCfg(name="self_collision", primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"), secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"), fields=("found",), reduce="none", num_slots=1)

    cfg = make_velocity_env_cfg()

    cfg.scene.entities = {"robot": MICRODUCK_WALK_ROLLERS_ROBOT_CFG}
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)
    cfg.viewer.body_name = "trunk_base"

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0
    # NOTE: an env-side action clip was tried here to bound the target, but the
    # deployment pipeline (infer.py) does NOT clip → the clip would only
    # exist in sim, a train/deploy mismatch. The over-command deterrent lives
    # policy-side instead (action_over_limit reward below), baked into the network
    # so it transfers with the ONNX.

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

    cfg.rewards["com_height_target"] = RewardTermCfg(func=microduck_mdp.com_height_target, weight=2.0, params={"target_height_min": 0.0935, "target_height_max": 0.1235})
    cfg.rewards["self_collisions"] = RewardTermCfg(func=mdp.self_collision_cost, weight=-1.0, params={"sensor_name": "self_collision"})
    cfg.rewards["feet_flat"] = RewardTermCfg(func=microduck_mdp.feet_flat_penalty, weight=-2.0, params={"asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot")), "sensor_name": "feet_ground_contact"})
    cfg.rewards["neck_action_rate_l2"] = RewardTermCfg(func=microduck_mdp.neck_action_rate_l2, weight=-0.5)
    cfg.rewards["neck_joint_pos_l2"] = RewardTermCfg(func=microduck_mdp.neck_joint_pos_l2, weight=-0.5)
    cfg.rewards["joint_torques_l2"] = RewardTermCfg(func=microduck_mdp.joint_torques_l2, weight=-1e-3)
    # Deter OVER-COMMANDING a joint past its hard stop (policy-side, transfers via
    # the ONNX). hip_roll's ±0.38 rad limit vs the ±10 rad ctrlrange let the low-kp
    # servo be commanded far past the stop and slam it with max torque — a fragile
    # sim-only trick. This penalises only the COMMAND beyond (limit + 0.3 overshoot),
    # so the joint keeps its full reachable range (a qpos penalty stole that range
    # and broke the gait) while the wild over-drive is discouraged.
    cfg.rewards["action_over_limit"] = RewardTermCfg(func=microduck_mdp.action_over_limit_penalty, weight=-0.5, params={"action_name": "joint_pos", "overshoot": 0.3})
    cfg.rewards["hip_roll_neutral"] = RewardTermCfg(
        func=microduck_mdp.joint_deviation_l1,
        weight=-2.0,  # -1.0 -> -2.0: stronger centring pull. Sim already keeps hip_roll
        # narrow, but a stronger corrective may help the REAL robot resist
        # whatever spreads the legs (deployment/disturbance). Lower if it
        # flattens the push.
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(r".*hip_roll.*",))},
    )
    # vel_scale 0.5 -> 0.3: the tanh target speed. Measured on a trained ckpt, the
    # policy only reaches ~0.33 m/s at max push, so a 0.5 target sat on the
    # un-saturated tanh slope and kept pushing it to go faster than it can (over-
    # reach -> launch instability). 0.3 saturates near the achievable speed, so it
    # is 'content' there instead of over-driving.
    cfg.rewards["wheel_speed"] = RewardTermCfg(func=microduck_mdp.wheel_speed_reward, weight=10.0, params={"command_name": "twist", "vel_scale": 0.3})
    cfg.rewards["braking"] = RewardTermCfg(func=microduck_mdp.braking_reward, weight=1.0, params={"command_name": "twist", "vel_std": 0.3})
    cfg.rewards["skating_air_time"] = RewardTermCfg(func=microduck_mdp.skating_air_time_reward, weight=1.5, params={"sensor_name": "feet_ground_contact", "command_name": "twist", "threshold_min": 0.15, "threshold_max": 0.45, "vel_gate_ref": 0.2})
    cfg.rewards["glide"] = RewardTermCfg(func=microduck_mdp.glide_reward, weight=4.0, params={"sensor_name": "feet_ground_contact", "command_name": "twist", "vel_ref": 0.2})
    # NOTE: a recover_pose reward (reward default leg pose + quiet + coasting during
    # the pause) was tried to get "stroke -> recover-to-neutral -> stroke", but
    # rewarding the SYMMETRIC default posture + dropping single_support's double
    # penalty re-opened the symmetric swizzle -> reverted. A proper retry must be
    # PHASE-GATED (reward the neutral only briefly right after a stroke, not
    # continuously) and keep the double-support penalty.
    cfg.rewards["single_support"] = RewardTermCfg(func=microduck_mdp.single_support_reward, weight=3.0, params={"sensor_name": "feet_ground_contact", "command_name": "twist", "vel_gate_ref": 0.2})
    cfg.rewards["gait_symmetry"] = RewardTermCfg(func=microduck_mdp.gait_symmetry_penalty, weight=-1.0, params={"sensor_name": "feet_ground_contact"})
    # NOTE: a contact_frequency penalty was tried here to slow the cadence, but it
    # penalises contact CHANGES — minimised by never lifting a foot (the swizzle),
    # so it pushes toward exactly the gait we fought to leave. Reverted; the
    # widened air-time window above is the safe cadence-slower (it forbids short
    # swings without rewarding not-stepping).
    cfg.rewards["forward_lean"] = RewardTermCfg(func=microduck_mdp.forward_lean_reward, weight=1.5, params={"command_name": "twist", "target_pitch": 0.262, "std": 0.1})
    # Heading command DISABLED (straight-line focus), but we hold the heading so it
    # doesn't drift: heading_hold rewards the yaw ANGLE staying near the spawn
    # heading. Corrective (allows yaw to steer back) — unlike a yaw-RATE penalty,
    # which froze the yaw and made drift WORSE (tried and reverted). Re-add real
    # heading_tracking (turning) once the stride is solid.
    cfg.rewards["heading_hold"] = RewardTermCfg(func=microduck_mdp.heading_hold_reward, weight=1.0, params={"std": 0.4, "asset_cfg": SceneEntityCfg("robot")})

    cfg.terminations["nan_state"] = TerminationTermCfg(func=microduck_mdp.robot_state_is_nan, time_out=False)

    cfg.events["reset_action_history"] = EventTermCfg(func=microduck_mdp.reset_action_history, mode="reset")

    del cfg.events["foot_friction"]  # wheels roll; ground friction lives in the XML

    cfg.events["reset_base"].params["pose_range"]["z"] = (0.1335, 0.1435)

    task_dr.apply_dr(cfg, DR, HEAD_BODY_NAMES, play=play)

    del cfg.observations["actor"].terms["base_lin_vel"]
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(func=mdp.base_lin_vel, scale=1.0)

    microduck_mdp.wire_sim2real_obs(cfg, imu_delay_max_lag=1, imu_misalignment_deg=DR.imu_orientation_angle_deg if DR.imu_orientation else None, encoder_bias_range=DR.encoder_bias_range if DR.encoder_bias else None, sanitize_critic_sensors=False)

    wheel_cfg = SceneEntityCfg("robot", joint_names=(r"^passive_.*wheel",))
    cfg.observations["critic"].terms["wheel_vel"] = ObservationTermCfg(func=mdp.joint_vel_rel, scale=1.0, params={"asset_cfg": wheel_cfg})

    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(func=microduck_mdp.zero_command_padding, params={"dim": 4})
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(func=microduck_mdp.zero_command_padding, params={"dim": 6})

    command: UniformVelocityCommandCfg = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    command.heading_command = False
    command.ranges.heading = None  # must be None when heading_command=False
    # cmd_x semantics: 0=coast, >0=push to accelerate, <0=brake to stop
    command.ranges.lin_vel_x = (-0.5, 0.6)
    command.ranges.lin_vel_y = (0.0, 0.0)
    # ang_vel_z range is the clip limit for cmd[2] = heading error (rad).
    # Set to 0 → cmd[2] is always 0 → no turning demand (straight-line focus).
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.viz.z_offset = 0.5
    cfg.commands["twist"] = microduck_mdp.RelativeHeadingVelocityCommandCfg(**vars(command))

    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(func=microduck_mdp.reward_weight, params={"reward_name": "action_rate_l2", "weight_stages": [{"step": 0, "weight": -1.0}, {"step": 250 * 24, "weight": -1.5}, {"step": 500 * 24, "weight": -2.0}]})

    if DR.wheel_friction:
        # Delayed + softened ramp: the previous schedule started adding bearing
        # drag at iter 750 — right when wheel_speed peaked — and reached 0.003,
        # which (with the heading ramp below) pushed the policy off skating into
        # a heading-farming local optimum. Keep the wheels free until skating is
        # robust, then add gentle, realistic drag.
        cfg.curriculum["wheel_friction"] = CurriculumTermCfg(func=microduck_mdp.wheel_friction_curriculum, params={"event_name": "randomize_wheel_friction", "ranges_stages": [{"step": 0 * 24, "ranges": (0.0000, 0.0000)}, {"step": 2000 * 24, "ranges": (0.0005, 0.0005)}, {"step": 3500 * 24, "ranges": (0.0010, 0.0010)}, {"step": 5000 * 24, "ranges": (0.0015, 0.0015)}]})

    # CoM randomization curricula — velocity's ramp, capped lower for the
    # balance-sensitive skating task (audit lesson: ±30 mm forced a nervous
    # gait on the walker; skates are even less forgiving).
    if DR.com:
        cfg.curriculum["com_range"] = CurriculumTermCfg(func=microduck_mdp.com_range_curriculum, params={"event_name": "randomize_com", "range_stages": [{"step": 0, "range": 0.003}, {"step": 500 * 24, "range": 0.005}, {"step": 1000 * 24, "range": 0.01}]})
    if DR.head_com:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(func=microduck_mdp.com_range_curriculum, params={"event_name": "randomize_head_com", "range_stages": [{"step": 0, "range": 0.003}, {"step": 500 * 24, "range": 0.005}, {"step": 1000 * 24, "range": 0.01}]})

    return cfg


MicroduckRollersRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,  # matches the family; normalizer baked into ONNX by export.py
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.03,  # roller-specific: higher exploration than the walk envs
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,
    ),
    logger=LOCAL_CHECKPOINTS_ONLY,
    experiment_name="velocity_rollers",
    run_name="velocity_rollers",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=50_000,
)
