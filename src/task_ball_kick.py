import math

KICK_FOOT = "right"
assert KICK_FOOT in ("right", "left")

# Symmetry — must stay OFF: the kick task is inherently one-footed.
ENABLE_SYMMETRY = False

# Long enough for kick + several seconds of ball-rolling reward + settle-back.
EPISODE_LENGTH_S = 5.0

# 70mm-diameter / 15g ball (see ball.xml).
BALL_RADIUS = 0.035
# Nominal ball-center offset in the robot's yaw frame. Measured at HOME: foot
# centers at (0, ±0.042), toe tip x≈0.034. With radius 0.035 and ±0.015 noise
# the ball's rear surface is at worst x=0.040 → always ≥6mm clear of the toe.
# (0.08 ± 0.02 allowed spawn-penetration with the toe: the solver ejected the
# ball at reset — free "kick" reward with no kick.)
BALL_OFFSET_X = 0.09
BALL_OFFSET_ABS_Y = 0.042
# Uniform ± placement noise per axis. This is the DR that makes the BLIND
# policy's swing robust to real-world aiming error.
BALL_POS_NOISE_XY = 0.015

# Target kick speed (m/s). The first trained policy (linear reward capped at
# 5 m/s) kicked much harder than needed — this tames the kick to a gentle,
# controlled tap. NOTE: the kick reward weights below are scaled to keep the
# at-target payoff ≈ +3/step regardless of this value (weight ≈ 3/target for
# the capped term) — if you change the target, rescale the weights with it.
BALL_TARGET_SPEED = 1.0

# Trunk standing height (measured natural equilibrium at HOME — see standup env).
STAND_Z = 0.115

_LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
_NECK_JOINTS = [5, 6, 7, 8]

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import CurriculumTermCfg, EventTermCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from . import task_dr
from . import task_mdp as microduck_mdp
from .robot import MICRODUCK_BALL_CFG, MICRODUCK_STANDUP_ROBOT_CFG
from .task_symmetry import SYMMETRY_CFG, PpoWithSymmetryCfg
from .task_velocity import HEAD_BODY_NAMES, LOCAL_CHECKPOINTS_ONLY

DR = task_dr.DEFAULT_DR
ENCODER_BIAS_RANGE = DR.encoder_bias_range


def make_microduck_ball_kick_env_cfg(play: bool = False, kick_foot: str | None = None) -> ManagerBasedRlEnvCfg:
    kick_foot = kick_foot or KICK_FOOT
    assert kick_foot in ("right", "left")
    support_foot = "left" if kick_foot == "right" else "right"

    feet_ground_cfg = ContactSensorCfg(name="feet_ground_contact", primary=ContactMatch(mode="geom", pattern=r"^(left_foot_collision|right_foot_collision)$", entity="robot"), secondary=ContactMatch(mode="body", pattern="terrain"), fields=("found", "force"), reduce="netforce", num_slots=1, track_air_time=True)

    support_foot_ground_cfg = ContactSensorCfg(name="support_foot_ground_contact", primary=ContactMatch(mode="geom", pattern=rf"^{support_foot}_foot_collision$", entity="robot"), secondary=ContactMatch(mode="body", pattern="terrain"), fields=("found",), reduce="netforce", num_slots=1)

    self_collision_cfg = ContactSensorCfg(name="self_collision", primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"), secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"), fields=("found",), reduce="none", num_slots=1)

    foot_frictions_geom_names = ("left_foot_collision", "right_foot_collision")

    cfg = make_velocity_env_cfg()

    # Full-collision robot (same spec as standup/ground-pick): the ball must be
    # able to contact the whole leg, not just the foot pads of the walk model.
    # Robot MUST stay the first entity (set_random_ground_state and the base
    # reset events write robot root state at qpos[:, 0:7]).
    cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG, "ball": MICRODUCK_BALL_CFG}
    cfg.scene.sensors = (feet_ground_cfg, support_foot_ground_cfg, self_collision_cfg)
    cfg.viewer.body_name = "trunk_base"

    cfg.episode_length_s = EPISODE_LENGTH_S

    # Extra contact headroom for the ball (ball-terrain + ball-robot contacts
    # on top of the full-collision robot's budget).
    cfg.sim.nconmax = 50

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    for name in [
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "pose",  # gait-conditioned; replaced by pose_target_match below
        "soft_landing",
    ]:
        if name in cfg.rewards:
            del cfg.rewards[name]

    #   • ball_speed_overshoot_penalty (weight -4.0): each m/s above target
    #     costs -4/step while it persists. Needed because the cap alone does
    #     NOT tame the kick — a harder kick keeps the ball at the cap for more
    #     steps, so total (per-step × rolling time) reward still grows with
    #     strike speed.
    cfg.rewards["ball_forward_velocity"] = RewardTermCfg(func=microduck_mdp.ball_forward_velocity, weight=12.0, params={"asset_name": "ball", "max_speed": BALL_TARGET_SPEED})
    cfg.rewards["ball_speed_overshoot"] = RewardTermCfg(func=microduck_mdp.ball_speed_overshoot_penalty, weight=-4.0, params={"asset_name": "ball", "target_speed": BALL_TARGET_SPEED})

    # Always-on anti-hop — swinging the kicking leg is free, lifting the
    # support foot costs this every step. Also suppresses walking/dribbling
    # exploits (any gait loses this reward half the time).
    cfg.rewards["support_foot_grounded"] = RewardTermCfg(func=microduck_mdp.single_foot_grounded_reward, weight=2.0, params={"sensor_name": support_foot_ground_cfg.name})

    # Legs at HOME. std=0.5 is deliberately loose: the kick itself is a big
    # transient leg deviation and must stay affordable.
    cfg.rewards["pose_stand_legs"] = RewardTermCfg(
        func=microduck_mdp.pose_target_match,
        weight=2.0,
        params={
            "std": 0.5,
            "joint_indices": _LEG_JOINTS,
            "target_overrides": None,  # HOME = standing
        },
    )

    cfg.rewards["pose_stand_neck"] = RewardTermCfg(func=microduck_mdp.pose_target_match, weight=1.0, params={"std": 0.3, "joint_indices": _NECK_JOINTS, "target_overrides": None})

    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["upright"].weight = 2.0
    cfg.rewards["upright"].params["std"] = math.sqrt(0.05)

    cfg.rewards["height_stand"] = RewardTermCfg(func=microduck_mdp.height_target_gaussian, weight=1.0, params={"std": 0.04, "target_height": STAND_Z, "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))})

    cfg.rewards["action_rate_l2"].weight = -0.1
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02

    cfg.rewards["self_collisions"] = RewardTermCfg(func=mdp.self_collision_cost, weight=-1.0, params={"sensor_name": self_collision_cfg.name})

    del cfg.observations["actor"].terms["base_lin_vel"]

    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(func=mdp.base_lin_vel, scale=1.0)
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    microduck_mdp.wire_sim2real_obs(cfg, imu_delay_max_lag=1, imu_misalignment_deg=DR.imu_orientation_angle_deg if DR.imu_orientation else None, encoder_bias_range=DR.encoder_bias_range if DR.encoder_bias else None, sanitize_critic_sensors=False)

    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(func=microduck_mdp.zero_command_padding, params={"dim": 4})
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(func=microduck_mdp.zero_command_padding, params={"dim": 6})

    # CRITIC-ONLY ball state (asymmetric actor-critic): the actor stays blind
    # to the ball (no ball sensing on the real robot), the critic uses it to
    # predict the kick payoff.
    cfg.observations["critic"].terms["ball_position"] = ObservationTermCfg(func=microduck_mdp.ball_pos_in_base, params={"asset_name": "ball"})
    cfg.observations["critic"].terms["ball_velocity"] = ObservationTermCfg(func=microduck_mdp.ball_vel_in_base, params={"asset_name": "ball"})

    # Command: tiny noise around zero (obs-shape parity only)
    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    command.heading_command = False
    command.ranges.heading = None
    command.resampling_time_range = (EPISODE_LENGTH_S, EPISODE_LENGTH_S * 2)
    command.debug_vis = False
    command.ranges.lin_vel_x = (-0.01, 0.01)
    command.ranges.lin_vel_y = (-0.01, 0.01)
    command.ranges.ang_vel_z = (-0.05, 0.05)
    cfg.commands["twist"] = microduck_mdp.VelocityCommandCommandOnlyCfg(**vars(command))

    # fell_over KEPT (robot starts standing and must stay up through the kick).
    cfg.terminations["nan_state"] = TerminationTermCfg(func=microduck_mdp.robot_state_is_nan, time_out=False)

    cfg.events["reset_action_history"] = EventTermCfg(func=microduck_mdp.reset_action_history, mode="reset")
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_frictions_geom_names
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)

    # Joint noise on the standing start: deployment hands off from the walk /
    # velstand policy, whose settled stand won't match HOME exactly.
    cfg.events["reset_robot_joints"].params["position_range"] = (-0.05, 0.05)

    cfg.events["set_ground_state"] = EventTermCfg(
        func=microduck_mdp.set_random_ground_state,
        mode="reset",
        params={
            "face_down_prob": 0.0,
            "face_up_prob": 0.0,
            "sitting_prob": 0.0,
            "standing_prob": 1.0,
            "sitting_tilt_max": math.radians(5),  # ±5° pitch/roll on the stand
            "standing_z_min": 0.11,
            "standing_z_max": 0.12,
        },
    )

    # Ball placement — MUST come after set_ground_state (events run in dict
    # insertion order; the ball position derives from the final robot pose).
    ball_offset_y = -BALL_OFFSET_ABS_Y if kick_foot == "right" else BALL_OFFSET_ABS_Y
    cfg.events["reset_ball"] = EventTermCfg(func=microduck_mdp.reset_ball_in_front_of_foot, mode="reset", params={"offset": (BALL_OFFSET_X, ball_offset_y), "noise_xy": BALL_POS_NOISE_XY, "ball_radius": BALL_RADIUS, "asset_name": "ball"})

    task_dr.apply_dr(cfg, DR, HEAD_BODY_NAMES, play=play)

    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(func=microduck_mdp.reward_weight, params={"reward_name": "action_rate_l2", "weight_stages": [{"step": 0, "weight": -0.1}, {"step": 500 * 24, "weight": -0.2}, {"step": 750 * 24, "weight": -0.4}, {"step": 1000 * 24, "weight": -0.6}, {"step": 1250 * 24, "weight": -0.8}, {"step": 1500 * 24, "weight": -1.0}]})

    if DR.com:
        cfg.curriculum["com_range"] = CurriculumTermCfg(func=microduck_mdp.com_range_curriculum, params={"event_name": "randomize_com", "range_stages": [{"step": 0, "range": 0.003}, {"step": 500 * 24, "range": 0.005}, {"step": 1000 * 24, "range": 0.01}, {"step": 1500 * 24, "range": 0.015}]})

    if DR.head_com:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(func=microduck_mdp.com_range_curriculum, params={"event_name": "randomize_head_com", "range_stages": [{"step": 0, "range": 0.003}, {"step": 500 * 24, "range": 0.005}, {"step": 1000 * 24, "range": 0.01}]})

    if DR.pushes:
        cfg.curriculum["push_magnitude"] = CurriculumTermCfg(func=microduck_mdp.push_curriculum, params={"event_name": "push_robot", "push_stages": [{"step": 0, "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}}, {"step": 500 * 24, "velocity_range": {"x": (-0.08, 0.08), "y": (-0.08, 0.08)}}, {"step": 1000 * 24, "velocity_range": {"x": DR.push_range, "y": DR.push_range}}]})

    return cfg


MicroduckBallKickRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,  # normalizer MUST be baked into ONNX by export.py
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
    algorithm=PpoWithSymmetryCfg(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.01, num_learning_epochs=5, num_mini_batches=4, learning_rate=1.0e-3, schedule="adaptive", gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0, symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None),
    logger=LOCAL_CHECKPOINTS_ONLY,
    experiment_name=f"ball_kick_{KICK_FOOT}",
    run_name=f"ball_kick_{KICK_FOOT}",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=10_000,
)
