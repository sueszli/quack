"""Microduck sit/stand task (v1.5 new model).

Episodic policy that gently sits down (deep squat) and stands back up while
remaining upright and stable, robust to pushes.

Phase encoding (in the command slot, 3-D):
    command = [cos(2π·phase), sin(2π·phase), 0]
    phase ∈ [0, 0.5]  → sit-down (target = sitting pose, peaks at phase 0.25)
    phase ∈ [0.5, 1]  → stand-up (target = standing pose, peaks at phase 0.75)

Phase is randomised per-env on episode reset to decorrelate environments.
PERIOD = 8 s (4 s sit-down + ~1.5 s rest-window + 4 s stand-up).

Joint layout (16 entries in asset.data.joint_pos; 14 actuated + 2 passive):
    0-4 : left  leg (hip_yaw, hip_roll, hip_pitch, knee, ankle)
    5-8 : neck/head (neck_pitch, head_pitch, head_yaw, head_roll)
    9-10: passive_1, passive_2  (jaw linkage — not actuated)
    11-15: right leg (hip_yaw, hip_roll, hip_pitch, knee, ankle)

Sitting target (from scene.xml `SIT` keyframe, deltas from HOME):
    left  knee  (idx 3)  =  1.0472    (HOME 0)
    left  ankle (idx 4)  =  0          (HOME +0.5236)
    right knee  (idx 14) = -1.0472    (HOME 0)
    right ankle (idx 15) =  0          (HOME -0.5236)
    left/right hip_roll  =  0          (HOME ±0.0873, sole-angle compensation)
    neck_pitch  (idx 5)  =  1.2217    (HOME +0.3491)
    head_pitch  (idx 6)  = -1.2217    (HOME -0.3491)
    everything else = home/standing pose
"""

from copy import deepcopy

# Symmetry
ENABLE_SYMMETRY = False

# ── Domain randomisation ──────────────────────────────────────────────────────
ENABLE_COM_RANDOMIZATION             = True
ENABLE_KP_RANDOMIZATION              = True
ENABLE_KD_RANDOMIZATION              = True
ENABLE_MASS_INERTIA_RANDOMIZATION    = True
ENABLE_VELOCITY_PUSHES               = True
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_NECK_OFFSET_RANDOMIZATION     = False  # head is part of pose target

# ── Ranges ────────────────────────────────────────────────────────────────────
COM_RANDOMIZATION_RANGE             = 0.003
MASS_INERTIA_RANDOMIZATION_RANGE    = (0.95, 1.05)
KP_RANDOMIZATION_RANGE              = (0.85, 1.15)
KD_RANDOMIZATION_RANGE              = (0.9, 1.1)
VELOCITY_PUSH_INTERVAL_S            = (3.0, 6.0)
VELOCITY_PUSH_RANGE                 = (-0.3, 0.3)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 1.0

# ── Sitting target pose (asset.data.joint_pos index → angle in rad) ───────────
SITTING_TARGET_OVERRIDES = {
    1:   0.0,      # left  hip_roll  (HOME -0.0873)
    3:   1.0472,   # left  knee      (HOME 0)
    4:   0.0,      # left  ankle     (HOME +0.5236)
    5:   1.2217,   # neck_pitch      (HOME +0.3491)
    6:  -1.2217,   # head_pitch      (HOME -0.3491)
    12:  0.0,      # right hip_roll  (HOME +0.0873)
    14: -1.0472,   # right knee      (HOME 0)
    15:  0.0,      # right ankle     (HOME -0.5236)
}

# Articulation indices (account for passive_1, passive_2 at 9, 10).
_LEG_JOINTS  = [0, 1, 2, 3, 4, 11, 12, 13, 14, 15]
_NECK_JOINTS = [5, 6, 7, 8]
# Joints with the largest deltas between SIT and STAND — focused reward so the
# gradient isn't diluted by joints that sit at home in both poses.
_SIT_STAND_CRITICAL_JOINTS = [3, 4, 14, 15]  # knees + ankles

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import MICRODUCK_ROUGH_TERRAINS_CFG
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG


def make_microduck_sitstand_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create Microduck sit/stand environment configuration."""

    site_names = ["left_foot", "right_foot"]

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(left_foot_collision|right_foot_collision)$",
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

    # Trunk-ground contact: detects "butt on ground" during the sit phase.
    trunk_ground_cfg = ContactSensorCfg(
        name="trunk_ground_contact",
        primary=ContactMatch(mode="body", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
    )

    foot_frictions_geom_names = ("left_foot_collision", "right_foot_collision")

    # ── Base config ───────────────────────────────────────────────────────────
    cfg = make_velocity_env_cfg()

    # Use the standup robot variant: has full collision meshes (head shells, jaw,
    # upper legs, neck_support, np_f970, battery_holder) needed to physically rest
    # the body on the ground during the deep-squat sit phase.
    cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}
    cfg.scene.sensors  = (feet_ground_cfg, self_collision_cfg, trunk_ground_cfg)
    cfg.viewer.body_name = "trunk_base"

    # ── Actions ───────────────────────────────────────────────────────────────
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0
    # No NeckOffsetJointPositionAction — head is part of the home/sit pose target

    # ── Rewards: remove walking-specific terms ────────────────────────────────
    for name in [
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "pose",  # replaced by phase-conditioned sit/stand pose terms
    ]:
        if name in cfg.rewards:
            del cfg.rewards[name]

    # ── Rewards: sit phase — reward END-STATE (low trunk) not exact joints ────
    # Replaces the previous sit_pose_legs + sit_pose_neck joint-matching rewards.
    # The policy is free to find any motion strategy (deep squat, head-supported
    # descent, body fold) as long as the trunk z reaches ~0.07 at sit peak.
    # Target z interpolates smoothly: stand_z=0.12 ↔ sit_z=0.07 via sin(2π·phase).
    cfg.rewards["phase_height_track"] = RewardTermCfg(
        func=microduck_mdp.phase_height_track,
        weight=5.0,
        params={
            "command_name": "twist",
            "stand_z": 0.12,
            "sit_z": 0.07,
            "std": 0.04,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # Reward butt-on-ground during the sit window (sin > 0.7 ≈ ±1.5 s around
    # sit peak with PERIOD=8s). Direct positive signal for the actual goal.
    cfg.rewards["sit_grounded"] = RewardTermCfg(
        func=microduck_mdp.sit_grounded,
        weight=5.0,
        params={
            "command_name": "twist",
            "sensor_name": trunk_ground_cfg.name,
            "sin_threshold": 0.7,
        },
    )

    # Bonus for stillness once seated — encourages "rest" rather than twitching.
    cfg.rewards["sit_stability"] = RewardTermCfg(
        func=microduck_mdp.sit_stability,
        weight=3.0,
        params={
            "command_name": "twist",
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
            "ang_vel_std": 0.5,
            "sin_threshold": 0.7,
        },
    )

    # Soft sit pose hint — the keyframe joint angles are a suggestion, not a
    # constraint. Wide std + low weight means the policy is free to find any
    # ergonomic sit; this just biases toward the designed pose.
    cfg.rewards["sit_pose_soft"] = RewardTermCfg(
        func=microduck_mdp.phase_pose_match,
        weight=1.0,
        params={
            "std": 0.5,
            "command_name": "twist",
            "joint_indices": _LEG_JOINTS + _NECK_JOINTS,
            "target_overrides": SITTING_TARGET_OVERRIDES,
            "phase": "approach",
        },
    )

    # Penalize hard trunk landings — encourages gentle descent.
    cfg.rewards["trunk_impact_penalty"] = RewardTermCfg(
        func=microduck_mdp.body_impact_cost,
        weight=-0.05,
        params={"sensor_name": trunk_ground_cfg.name, "threshold": 8.0},
    )

    # ── Rewards: stand-up phase (legs) ────────────────────────────────────────
    # General leg pose match (10 joints, relaxed std).
    cfg.rewards["stand_pose_legs"] = RewardTermCfg(
        func=microduck_mdp.phase_pose_match,
        weight=2.0,
        params={
            "std": 0.3,
            "command_name": "twist",
            "joint_indices": _LEG_JOINTS,
            "target_overrides": None,  # default = home/standing pose
            "phase": "return",
        },
    )

    # Focused reward on the 4 joints that actually change between sit↔stand
    # (knees + ankles, both legs). Without this, the 6 always-at-home leg
    # joints inflate the average and the policy under-extends. Tight std forces
    # full convergence to home.
    cfg.rewards["stand_pose_critical"] = RewardTermCfg(
        func=microduck_mdp.phase_pose_match,
        weight=6.0,
        params={
            "std": 0.15,
            "command_name": "twist",
            "joint_indices": _SIT_STAND_CRITICAL_JOINTS,
            "target_overrides": None,
            "phase": "return",
        },
    )

    # Stand-up phase (neck): tight.
    cfg.rewards["stand_pose_neck"] = RewardTermCfg(
        func=microduck_mdp.phase_pose_match,
        weight=4.0,
        params={
            "std": 0.15,
            "command_name": "twist",
            "joint_indices": _NECK_JOINTS,
            "target_overrides": None,
            "phase": "return",
        },
    )

    # ── Rewards: stability (kept across both phases) ──────────────────────────
    # Upright: enforced throughout — robot must keep trunk vertical even when sitting.
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    # Dropped to 0.2: don't fight the natural trunk-fold needed to actually sit
    # the body onto the ground. The phase-conditioned stand_pose rewards already
    # enforce upright trunk during the stand half via joint targeting.
    cfg.rewards["upright"].weight = 0.2

    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.05

    cfg.rewards["angular_momentum"].weight = -0.02

    cfg.rewards["soft_landing"].weight = -1e-5

    # ── Rewards: regularisation ───────────────────────────────────────────────
    # Action smoothness — high weight to encourage gentle motion.
    cfg.rewards["action_rate_l2"] = RewardTermCfg(
        func=mdp.action_rate_l2, weight=-2.0
    )

    # Neck/head smoothness.
    cfg.rewards["neck_action_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.neck_action_rate_l2, weight=-1.0
    )

    # Joint torque penalty — discourages forceful moves.
    cfg.rewards["joint_torques_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torques_l2, weight=-5e-3
    )

    # Self-collision — folded legs could clip with neck/head during sit.
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )

    # ── Observations (identical layout to walking policy) ─────────────────────
    del cfg.observations["policy"].terms["base_lin_vel"]

    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel, scale=1.0,
    )
    cfg.observations["critic"].terms["foot_height"].params[
        "asset_cfg"
    ].site_names = site_names

    gravity_term_name = "projected_gravity"
    cfg.observations["policy"].terms[gravity_term_name] = deepcopy(
        cfg.observations["policy"].terms[gravity_term_name]
    )
    cfg.observations["policy"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["policy"].terms["base_ang_vel"]
    )

    # Sensor delays — match velocity env
    cfg.observations["policy"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["policy"].terms["base_ang_vel"].delay_max_lag = 3
    cfg.observations["policy"].terms["base_ang_vel"].delay_update_period = 64
    cfg.observations["policy"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["policy"].terms[gravity_term_name].delay_max_lag = 3
    cfg.observations["policy"].terms[gravity_term_name].delay_update_period = 64

    # Observation noise — match velocity env
    cfg.observations["policy"].terms["base_ang_vel"].noise    = Unoise(n_min=-0.024, n_max=0.024)
    cfg.observations["policy"].terms[gravity_term_name].noise = Unoise(n_min=-0.007, n_max=0.007)
    cfg.observations["policy"].terms["joint_pos"].noise       = Unoise(n_min=-0.0006, n_max=0.0006)
    cfg.observations["policy"].terms["joint_vel"].noise       = Unoise(n_min=-0.24, n_max=0.24)

    # 1-ctrl-step lag on joint_vel (matches Dynamixel firmware).
    cfg.observations["policy"].terms["joint_vel"] = deepcopy(
        cfg.observations["policy"].terms["joint_vel"]
    )
    cfg.observations["policy"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["policy"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["policy"].terms["joint_vel"].delay_update_period = 0

    # Exclude passive_* joints from joint_pos/vel obs (action space stays 14-dim).
    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    cfg.observations["policy"].terms["joint_pos"].params["asset_cfg"] = passive_excluded
    cfg.observations["policy"].terms["joint_vel"].params["asset_cfg"] = deepcopy(passive_excluded)
    cfg.observations["critic"].terms["joint_pos"].params["asset_cfg"] = deepcopy(passive_excluded)
    cfg.observations["critic"].terms["joint_vel"].params["asset_cfg"] = deepcopy(passive_excluded)

    # ── Command: cyclic phase encoding (reuse GroundPickPhaseCommand) ─────────
    # PERIOD = 8 s — gives the policy time for a gentle sit-down with a ~1.5 s
    # rest window at the sit peak (where sin > 0.7). Phase randomized per env
    # on reset to decorrelate environments.
    command: UniformVelocityCommandCfg = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs  = 0.0
    cfg.commands["twist"] = microduck_mdp.GroundPickPhaseCommandCfg(
        **{**vars(command), "class_type": microduck_mdp.GroundPickPhaseCommand, "period": 8.0}
    )

    # ── Events ────────────────────────────────────────────────────────────────
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_frictions_geom_names

    # Always reset to the standing pose at standing height (matches STAND keyframe trunk z=0.12).
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.115, 0.125)

    # NaN-state termination
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
    )

    if ENABLE_VELOCITY_PUSHES:
        interval = (0.5, 1.0) if play else VELOCITY_PUSH_INTERVAL_S
        cfg.events["push_robot"] = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=interval,
            params={
                "velocity_range": {
                    "x": VELOCITY_PUSH_RANGE,
                    "y": VELOCITY_PUSH_RANGE,
                },
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

    if ENABLE_COM_RANDOMIZATION:
        cfg.events["randomize_com"] = EventTermCfg(
            func=mdp.randomize_field,
            mode="reset",
            domain_randomization=True,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "operation": "add",
                "field": "body_ipos",
                "ranges": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_KP_RANDOMIZATION or ENABLE_KD_RANDOMIZATION:
        kp_range = KP_RANDOMIZATION_RANGE if ENABLE_KP_RANDOMIZATION else (1.0, 1.0)
        kd_range = KD_RANDOMIZATION_RANGE if ENABLE_KD_RANDOMIZATION else (1.0, 1.0)
        cfg.events["randomize_motor_gains"] = EventTermCfg(
            func=microduck_mdp.randomize_delayed_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "operation": "scale",
                "kp_range": kp_range,
                "kd_range": kd_range,
            },
        )

    if ENABLE_MASS_INERTIA_RANDOMIZATION:
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=microduck_mdp.randomize_mass_and_inertia,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "scale_range": MASS_INERTIA_RANDOMIZATION_RANGE,
            },
        )

    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        cfg.events["randomize_imu_orientation"] = EventTermCfg(
            func=microduck_mdp.randomize_imu_orientation,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE,
            },
        )

    # ── Terrain ───────────────────────────────────────────────────────────────
    if not rough:
        cfg.scene.terrain.terrain_type = "plane"
        cfg.scene.terrain.terrain_generator = None
    else:
        cfg.scene.terrain.terrain_type = "generator"
        cfg.scene.terrain.terrain_generator = MICRODUCK_ROUGH_TERRAINS_CFG
        if play:
            cfg.scene.terrain.terrain_generator.curriculum = False
            cfg.scene.terrain.terrain_generator.num_cols = 5
            cfg.scene.terrain.terrain_generator.num_rows = 5

    # ── Curriculum ────────────────────────────────────────────────────────────
    if not rough:
        del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    # Push curriculum: zero pushes for 500 iterations (learn the sit/stand cycle
    # cleanly), then ramp up to full magnitude.
    if ENABLE_VELOCITY_PUSHES:
        cfg.curriculum["push_magnitude"] = CurriculumTermCfg(
            func=microduck_mdp.push_curriculum,
            params={
                "event_name": "push_robot",
                "push_stages": [
                    {"step": 0,         "velocity_range": {"x": (0.0, 0.0),    "y": (0.0, 0.0)}},
                    {"step": 500 * 24,  "velocity_range": {"x": (-0.15, 0.15), "y": (-0.15, 0.15)}},
                    {"step": 750 * 24,  "velocity_range": {"x": VELOCITY_PUSH_RANGE, "y": VELOCITY_PUSH_RANGE}},
                ],
            },
        )

    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0,        "weight": -0.4},
                {"step": 250 * 24, "weight": -0.8},
                {"step": 500 * 24, "weight": -1.0},
            ],
        },
    )

    return cfg


# ── RL runner config ──────────────────────────────────────────────────────────

MicroduckSitStandRlCfg = RslRlOnPolicyRunnerCfg(
    policy=RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
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
        entropy_coef=0.01,
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
    wandb_project="mjlab_microduck",
    experiment_name="microduck_sitstand",
    run_name="microduck_sitstand",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=20_000,
)
