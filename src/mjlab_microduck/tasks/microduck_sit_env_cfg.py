"""Microduck *sit* task (v1.5).

Episodic policy that gently descends from the standing keyframe to the sitting
keyframe and rests there. Companion to the standup env — together they replace
the older cyclic sitstand env.

Reset:  standing keyframe (trunk z ≈ 0.12, home joints).
Target: sitting keyframe (trunk z ≈ 0.07, knees/ankles/head set per SIT key).
Gentleness is treated as a first-class objective: strong action-rate, torque,
and impact penalties are active from the very start (no curriculum ramp).

Joint layout (16 entries in asset.data.joint_pos; 14 actuated + 2 passive):
    0-4 : left  leg (hip_yaw, hip_roll, hip_pitch, knee, ankle)
    5-8 : neck/head (neck_pitch, head_pitch, head_yaw, head_roll)
    9-10: passive_1, passive_2  (jaw linkage — not actuated)
    11-15: right leg (hip_yaw, hip_roll, hip_pitch, knee, ankle)
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

# ── Ranges ────────────────────────────────────────────────────────────────────
COM_RANDOMIZATION_RANGE             = 0.003
MASS_INERTIA_RANDOMIZATION_RANGE    = (0.95, 1.05)
KP_RANDOMIZATION_RANGE              = (0.85, 1.15)
KD_RANDOMIZATION_RANGE              = (0.9, 1.1)
# Lighter pushes than sitstand — the robot is descending, big lateral kicks
# while it's already low-stability would teach a "brace" behavior we don't need.
VELOCITY_PUSH_INTERVAL_S            = (3.0, 6.0)
VELOCITY_PUSH_RANGE                 = (-0.15, 0.15)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 1.0

# Episode length: long enough for a controlled descent + several seconds of rest.
EPISODE_LENGTH_S = 6.0

# ── SIT keyframe (joint_pos index → angle in rad). Single fixed target. ─────
# No intermediate waypoints — the policy is free to discover its own descent
# path. Anything else than "land gently in this exact pose" pays a cost via
# pose error + a_z penalty + smoothness regularisers.
SITTING_TARGET_OVERRIDES = {
    1:   0.0,      # left  hip_roll  (HOME -0.0873)
    3:   1.0472,   # left  knee      (HOME 0)
    4:   0.0,      # left  ankle     (HOME +0.5236)
    # neck/head intentionally omitted → stay at HOME (head forward-facing).
    12:  0.0,      # right hip_roll  (HOME +0.0873)
    14: -1.0472,   # right knee      (HOME 0)
    15:  0.0,      # right ankle     (HOME -0.5236)
}

# Articulation indices (passive_1, passive_2 occupy 9, 10).
_LEG_JOINTS  = [0, 1, 2, 3, 4, 11, 12, 13, 14, 15]
_NECK_JOINTS = [5, 6, 7, 8]

# Trunk height targets (m).
STAND_Z = 0.12
SIT_Z   = 0.07

# Upright gating window for ``upright_while_tall``: full upright incentive
# above STAND_UPRIGHT_Z (still tall, must stay vertical), fades to 0 at
# SIT_UPRIGHT_Z (committed to sit, butt-down orientation is fine).
STAND_UPRIGHT_Z = 0.10
SIT_UPRIGHT_Z   = 0.085

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
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import MICRODUCK_ROUGH_TERRAINS_CFG
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG


def make_microduck_sit_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create Microduck sit environment configuration."""

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

    foot_frictions_geom_names = ("left_foot_collision", "right_foot_collision")

    # ── Base config ───────────────────────────────────────────────────────────
    cfg = make_velocity_env_cfg()

    # Standup robot variant: full collision meshes — needed so the body can
    # physically rest on the ground during the seated phase.
    cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}
    cfg.scene.sensors  = (feet_ground_cfg, self_collision_cfg)
    cfg.viewer.body_name = "trunk_base"

    cfg.episode_length_s = EPISODE_LENGTH_S

    # ── Actions ───────────────────────────────────────────────────────────────
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # ── Rewards: drop walking-specific terms ──────────────────────────────────
    for name in [
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "pose",
    ]:
        if name in cfg.rewards:
            del cfg.rewards[name]

    # ── Rewards: minimum-viable set for an organic sit policy ────────────────
    # Single fixed target (SIT keyframe) active from t=0. No trajectory, no
    # waypoints, no episode-progress gating. The policy is free to discover
    # any descent path that satisfies:
    #   (1) end-state matches the SIT pose + trunk height
    #   (2) descent is gentle (low |a_z| throughout)
    #   (3) trunk stays upright until committed to the sit (low z)
    #   (4) joint/action motion stays smooth (sim2real regularisers)
    # Anything else (fall backward, snap to floor, face-plant) costs more
    # than the rewarded path of "slow controlled crouch-and-sit".

    # Pose target — legs+hips+knees+ankles. Generous std (0.5) gives a useful
    # gradient even from HOME (the SIT pose is ~1 rad away on knees).
    cfg.rewards["pose_sit_legs"] = RewardTermCfg(
        func=microduck_mdp.pose_target_match,
        weight=8.0,
        params={
            "std": 0.5,
            "joint_indices": _LEG_JOINTS,
            "target_overrides": SITTING_TARGET_OVERRIDES,
        },
    )

    # Pose target — neck/head fixed at HOME (forward-facing).
    cfg.rewards["pose_sit_neck"] = RewardTermCfg(
        func=microduck_mdp.pose_target_match,
        weight=4.0,
        params={
            "std": 0.3,
            "joint_indices": _NECK_JOINTS,
            "target_overrides": SITTING_TARGET_OVERRIDES,
        },
    )

    # L1 bootstrap — constant gradient toward SIT joints even when far from
    # the Gaussian's effective basin.
    cfg.rewards["pose_sit_l1"] = RewardTermCfg(
        func=microduck_mdp.pose_l1_penalty,
        weight=2.0,
        params={
            "joint_indices": _LEG_JOINTS + _NECK_JOINTS,
            "target_overrides": SITTING_TARGET_OVERRIDES,
        },
    )

    # Trunk height target (Gaussian + L1) — pulls the body down to SIT_Z.
    cfg.rewards["height_sit"] = RewardTermCfg(
        func=microduck_mdp.height_target_gaussian,
        weight=4.0,
        params={
            "std":           0.02,
            "target_height": SIT_Z,
            "asset_cfg":     SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    cfg.rewards["height_sit_l1"] = RewardTermCfg(
        func=microduck_mdp.height_l1_penalty,
        weight=10.0,
        params={
            "target_height": SIT_Z,
            "asset_cfg":     SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # Gentle descent — penalty on |a_z| of the trunk. This is the only
    # "no hard shocks" signal. Hard impacts and fall-onto-butt strategies
    # produce large a_z spikes; controlled quasi-static descent gives a_z ≈ 0.
    cfg.rewards["gentle_descent"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=-0.02,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )

    # Upright while tall — full upright incentive above STAND_UPRIGHT_Z, fades
    # to 0 by SIT_UPRIGHT_Z. Blocks the "tip backward to sit" exploit (which
    # requires losing uprightness while still high) without fighting the
    # natural butt-down orientation once the robot has committed.
    cfg.rewards["upright_while_tall"] = RewardTermCfg(
        func=microduck_mdp.upright_while_tall,
        weight=4.0,
        params={
            "height_low":  SIT_UPRIGHT_Z,
            "height_high": STAND_UPRIGHT_Z,
            "asset_cfg":   SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # ── Sim2real regularisers (smoothness) ────────────────────────────────────
    cfg.rewards["action_rate_l2"]        = RewardTermCfg(func=mdp.action_rate_l2,           weight=-0.5)
    cfg.rewards["joint_torque_rate_l2"]  = RewardTermCfg(func=microduck_mdp.joint_torque_rate_l2, weight=-5e-4)
    cfg.rewards["joint_torques_l2"]      = RewardTermCfg(func=microduck_mdp.joint_torques_l2,     weight=-5e-3)

    # ── Stability ─────────────────────────────────────────────────────────────
    # Body angular velocity: keep wobble cheap to suppress at rest. Inherited
    # from the velocity env, just retarget to trunk_base and bump.
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.3

    # Self collisions still penalised.
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )

    # Drop the velocity-env upright Gaussian and angular_momentum — replaced
    # by upright_while_tall above. Drop soft_landing — its job is taken over
    # by trunk_vertical_accel_penalty, which has a cleaner gradient.
    for name in ("upright", "angular_momentum", "soft_landing"):
        if name in cfg.rewards:
            del cfg.rewards[name]

    # ── Observations (identical layout to walking / sitstand policies) ────────
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

    cfg.observations["policy"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["policy"].terms["base_ang_vel"].delay_max_lag = 3
    cfg.observations["policy"].terms["base_ang_vel"].delay_update_period = 64
    cfg.observations["policy"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["policy"].terms[gravity_term_name].delay_max_lag = 3
    cfg.observations["policy"].terms[gravity_term_name].delay_update_period = 64

    cfg.observations["policy"].terms["base_ang_vel"].noise    = Unoise(n_min=-0.024, n_max=0.024)
    cfg.observations["policy"].terms[gravity_term_name].noise = Unoise(n_min=-0.007, n_max=0.007)
    cfg.observations["policy"].terms["joint_pos"].noise       = Unoise(n_min=-0.0006, n_max=0.0006)
    cfg.observations["policy"].terms["joint_vel"].noise       = Unoise(n_min=-0.24, n_max=0.24)

    cfg.observations["policy"].terms["joint_vel"] = deepcopy(
        cfg.observations["policy"].terms["joint_vel"]
    )
    cfg.observations["policy"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["policy"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["policy"].terms["joint_vel"].delay_update_period = 0

    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    cfg.observations["policy"].terms["joint_pos"].params["asset_cfg"] = passive_excluded
    cfg.observations["policy"].terms["joint_vel"].params["asset_cfg"] = deepcopy(passive_excluded)
    cfg.observations["critic"].terms["joint_pos"].params["asset_cfg"] = deepcopy(passive_excluded)
    cfg.observations["critic"].terms["joint_vel"].params["asset_cfg"] = deepcopy(passive_excluded)

    # ── Command padding: zero head/body command slots to keep 13D layout ──────
    # The sit env doesn't track head or body pose commands — the target is
    # baked into the joint-target reward. The slots are zero-padded so the
    # runtime can feed a single 13D command buffer across all microduck policies.
    for group in ("policy", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 4},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 6},
        )

    # ── Command: tiny noise around zero (kept for obs-shape parity) ──────────
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

    # ── Events ────────────────────────────────────────────────────────────────
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_frictions_geom_names

    # Always start standing (matches STAND keyframe trunk z=0.12, home joints).
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.115, 0.125)

    # Disable fall termination: tilt-based termination would cut episodes short
    # whenever the robot wobbles during the descent, robbing the policy of the
    # impact-penalty signal that teaches it to land gently.
    if "fell_over" in cfg.terminations:
        del cfg.terminations["fell_over"]

    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
    )

    # MuJoCo physics robustness for the sit task. The standup XML has full
    # collisions on every body, and the sit pose puts trunk + folded legs +
    # head all in close contact with the ground and each other. Default
    # nconmax=35 and solver iterations=10 overflow the contact solver on
    # sit attempts, producing NaN. nan_state then terminates the episode,
    # cutting the policy off mid-sit with zero future rewards.
    #
    # This is the real culprit behind the "learn-to-sit-by-iter-250-then-
    # unlearn-it-by-iter-500" pattern: sit attempts crash the physics, NaN-
    # terminated episodes accumulate negative gradient, policy backs off
    # the descent.
    cfg.sim.nconmax = 200
    cfg.sim.mujoco.iterations = 30
    cfg.sim.mujoco.ls_iterations = 50

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

    # Push curriculum — delayed significantly. The sit task is much more
    # fragile to pushes than walking: a push mid-descent tips the robot
    # into a configuration it can't recover from before the policy has
    # consolidated the sit motion. Previous schedule (pushes at iter 500)
    # caused exactly the failure mode seen in wandb: policy learns to sit
    # by iter 250, pushes start at 500, episode rewards drop, policy
    # unlearns sitting and converges to "just stand doing nothing".
    #
    # Hold pushes off until the policy has been sitting cleanly for a
    # while (~iter 1500), then ramp gently to the final ±0.15 m/s.
    if ENABLE_VELOCITY_PUSHES:
        cfg.curriculum["push_magnitude"] = CurriculumTermCfg(
            func=microduck_mdp.push_curriculum,
            params={
                "event_name": "push_robot",
                "push_stages": [
                    {"step": 0,         "velocity_range": {"x": (0.0, 0.0),    "y": (0.0, 0.0)}},
                    {"step": 1500 * 24, "velocity_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05)}},
                    {"step": 2000 * 24, "velocity_range": {"x": (-0.10, 0.10), "y": (-0.10, 0.10)}},
                    {"step": 2500 * 24, "velocity_range": {"x": VELOCITY_PUSH_RANGE, "y": VELOCITY_PUSH_RANGE}},
                ],
            },
        )

    return cfg


# ── RL runner config ──────────────────────────────────────────────────────────

MicroduckSitRlCfg = RslRlOnPolicyRunnerCfg(
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
    experiment_name="microduck_sit",
    run_name="microduck_sit",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=15_000,
)
