"""Microduck *stand* task (v1.5) — specialized: sitting pose → standing.

Episodic policy that gently rises from the sitting keyframe to the standing
keyframe. Companion to the sit env — together they form a clean sit↔stand
pair, each policy doing one direction.

Reset:  sitting keyframe (trunk z ≈ 0.07, knees/ankles bent, head at HOME).
Target: standing keyframe (trunk z ≈ 0.12, HOME joints).
Reward design (mirror of sit env): a single fixed target is rewarded from
t=0 to end of episode; gentleness is enforced via |a_z| only; smoothness is
enforced by the usual sim2real regularisers. No trajectory waypoints, no
episode-progress gating — the policy is free to discover its own rise path.
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
VELOCITY_PUSH_INTERVAL_S            = (3.0, 6.0)
VELOCITY_PUSH_RANGE                 = (-0.15, 0.15)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 1.0

# Episode length: long enough for a gentle rise + brief stabilisation.
EPISODE_LENGTH_S = 6.0

# ── Sitting source pose (asset.data.joint_pos index → angle in rad) ───────────
# Must match the *actual end-state* of the sit policy (head at HOME, knees bent
# ~60°, ankles 0). Neck/head intentionally omitted → reset stays at HOME so the
# standup policy starts from exactly where the sit policy converges.
SITTING_JOINT_OVERRIDES = {
    1:   0.0,      # left  hip_roll
    3:   1.0472,   # left  knee
    4:   0.0,      # left  ankle
    12:  0.0,      # right hip_roll
    14: -1.0472,   # right knee
    15:  0.0,      # right ankle
}

# Articulation indices (account for passive_1, passive_2 at 9, 10).
_LEG_JOINTS  = [0, 1, 2, 3, 4, 11, 12, 13, 14, 15]
_NECK_JOINTS = [5, 6, 7, 8]

# Trunk height targets (m).
SIT_Z   = 0.07
STAND_Z = 0.12

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


def make_microduck_standup_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create Microduck stand environment configuration (sit-keyframe start)."""

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

    # ── Rewards: minimum-viable set for an organic standup policy ────────────
    # Single fixed target (STAND = HOME pose + STAND_Z), active from t=0. No
    # trajectory, no waypoints, no episode-progress gating. The policy is free
    # to discover any rise path that satisfies:
    #   (1) end-state matches the HOME pose + STAND_Z
    #   (2) rise is gentle (low |a_z| throughout)
    #   (3) trunk stays upright throughout (failure mode: tip backward while
    #       extending legs; no "low z is safe" regime as in sit)
    #   (4) joint/action motion stays smooth (sim2real regularisers)

    # Pose target — legs+hips+knees+ankles. target_overrides=None → HOME.
    cfg.rewards["pose_stand_legs"] = RewardTermCfg(
        func=microduck_mdp.pose_target_match,
        weight=8.0,
        params={
            "std": 0.5,
            "joint_indices": _LEG_JOINTS,
            "target_overrides": None,   # HOME = standing
        },
    )

    # Pose target — neck/head at HOME.
    cfg.rewards["pose_stand_neck"] = RewardTermCfg(
        func=microduck_mdp.pose_target_match,
        weight=4.0,
        params={
            "std": 0.3,
            "joint_indices": _NECK_JOINTS,
            "target_overrides": None,
        },
    )

    # L1 bootstrap — constant gradient even when far from HOME.
    # Bumped 2 → 5: at convergence the policy parks ~0.18 rad off-HOME (mostly
    # bent knees) costing only -0.35/step at weight 2 — cheap enough to ignore.
    # At weight 5 that error costs -0.9/step, forcing the policy to actually
    # close the gap on the remaining joints.
    cfg.rewards["pose_stand_l1"] = RewardTermCfg(
        func=microduck_mdp.pose_l1_penalty,
        weight=5.0,
        params={
            "joint_indices": _LEG_JOINTS + _NECK_JOINTS,
            "target_overrides": None,
        },
    )

    # Trunk height target — two-layer Gaussian to get both bootstrap reach
    # AND a sharp peak at STAND_Z.
    #  - ``height_stand``: wide std (0.04), for the bootstrap pull from sit.
    #  - ``height_stand_sharp``: narrow std (0.015), creates a strong gradient
    #    in the final cm. Earlier runs converged at z ≈ 0.109 because the
    #    wide-std Gaussian was already saturated (0.93/1.0) — no gradient to
    #    pull the last cm. The sharp layer adds 0.36→1.0 reward jump in that
    #    same range, ~3× the marginal pull.
    cfg.rewards["height_stand"] = RewardTermCfg(
        func=microduck_mdp.height_target_gaussian,
        weight=4.0,
        params={
            "std":           0.04,
            "target_height": STAND_Z,
            "asset_cfg":     SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    cfg.rewards["height_stand_sharp"] = RewardTermCfg(
        func=microduck_mdp.height_target_gaussian,
        weight=4.0,
        params={
            "std":           0.015,
            "target_height": STAND_Z,
            "asset_cfg":     SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    # L1 bumped 10 → 30: previous run plateaued sitting still because the
    # static-sit basin (-0.5 reward from L1 + everything else positive) was
    # net positive. At weight 30, sitting still costs -1.5/step — net cost
    # of "stay sitting" forces exploration.
    cfg.rewards["height_stand_l1"] = RewardTermCfg(
        func=microduck_mdp.height_l1_penalty,
        weight=30.0,
        params={
            "target_height": STAND_Z,
            "asset_cfg":     SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # Reward upward CoM velocity below STAND_Z — pays for the *motion* of
    # rising, not just for the destination. Critical bootstrap: with only
    # destination rewards, "stay sitting upright collecting most-of-pose +
    # upright" was the dominant local optimum. Rewarding vz > 0 directly
    # makes any rise attempt immediately positive. Gates off above
    # max_height so the policy can't farm it by bobbing.
    # max_height set just above STAND_Z (0.12 → 0.125) so the reward stays
    # active through the final cm of rise. Earlier 0.11 caused the policy to
    # park at ~0.108 (gate-off altitude) and never finish the climb.
    cfg.rewards["com_upward_velocity"] = RewardTermCfg(
        func=microduck_mdp.com_upward_velocity,
        weight=3.0,
        params={
            "asset_cfg":  SceneEntityCfg("robot", body_names=("trunk_base",)),
            "max_height": 0.125,
        },
    )

    # Gentle rise — penalty on |a_z|. Compatible with com_upward_velocity:
    # constant positive vz collects upward-velocity reward AND has a_z = 0,
    # so the two pressures together select for smooth constant-velocity rise.
    cfg.rewards["gentle_rise"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=-0.02,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )

    # Upright — two-layer like the height reward.
    #  - ``upright_linear``: cos(tilt). Strong gradient at high tilt (e.g.,
    #    while inverted at the start of a recovery), weak near vertical.
    #    Provides bootstrap pull from any orientation.
    #  - ``upright_sharp``: exp(-tilt²/std²) with std ≈ 6°. Gradient is
    #    STRONGEST in the near-vertical regime where the linear version
    #    runs out of steam. Previous run converged at ~37° back-lean because
    #    the linear pull at small tilt becomes weak; this term punishes that
    #    exact regime.
    cfg.rewards["upright_linear"] = RewardTermCfg(
        func=microduck_mdp.body_upright_linear,
        weight=6.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    cfg.rewards["upright_sharp"] = RewardTermCfg(
        func=microduck_mdp.body_upright_gaussian,
        weight=6.0,
        params={
            "std":       0.1,   # ≈ 5.7° — full reward only when near vertical
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # Discrete goal-state bonus: +10 only when the robot is *fully* standing
    # (height, upright AND pose all within tolerance simultaneously). The
    # surrounding gradient rewards bring the policy CLOSE to the goal but
    # can't escape compromise basins (lean trunk to compensate head-forward
    # CoM, park 1cm short of STAND_Z) — those compromises collect partial
    # gradient credit but ZERO bonus. The bonus is the "carrot" that makes
    # the genuine goal state strictly more valuable than the compromises.
    cfg.rewards["standing_bonus"] = RewardTermCfg(
        func=microduck_mdp.standing_success_bonus,
        weight=10.0,
        params={
            "target_height":     STAND_Z,
            "height_tol":        0.01,    # ±1 cm
            "upright_threshold": 0.97,    # within ≈ 14° of vertical
            "pose_tol":          0.15,    # ≈ 8.6° per joint, max
            "joint_indices":     _LEG_JOINTS + _NECK_JOINTS,
            "target_overrides":  None,
            "asset_cfg":         SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # ── Sim2real regularisers (smoothness) ────────────────────────────────────
    cfg.rewards["action_rate_l2"]        = RewardTermCfg(func=mdp.action_rate_l2,                 weight=-0.5)
    cfg.rewards["joint_torque_rate_l2"]  = RewardTermCfg(func=microduck_mdp.joint_torque_rate_l2, weight=-5e-4)
    cfg.rewards["joint_torques_l2"]      = RewardTermCfg(func=microduck_mdp.joint_torques_l2,     weight=-5e-3)

    # ── Stability ─────────────────────────────────────────────────────────────
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.3

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )

    # Hip yaw/roll drift penalty — keeps a narrow base while pushing up. Without
    # it the policy can splay legs sideways while extending knees, which gives
    # a low-cost "push trunk up via leg spread" exploit instead of a clean rise.
    cfg.rewards["hip_yaw_roll_deviation"] = RewardTermCfg(
        func=microduck_mdp.joint_deviation_l1,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=(r".*hip_yaw.*", r".*hip_roll.*"),
            ),
        },
    )

    # Drop velocity-env terms that are either subsumed (upright Gaussian,
    # angular_momentum, soft_landing) or irrelevant here.
    for name in ("upright", "angular_momentum", "soft_landing"):
        if name in cfg.rewards:
            del cfg.rewards[name]

    # ── Observations (identical layout to walking / sit policies) ─────────────
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

    # ── Command padding: zero head/body slots for 13D unified obs ─────────────
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

    # ── Terminations ──────────────────────────────────────────────────────────
    # Robot starts seated — tilt-based fall termination doesn't apply here.
    if "fell_over" in cfg.terminations:
        del cfg.terminations["fell_over"]
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
    )

    # ── Events ────────────────────────────────────────────────────────────────
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_frictions_geom_names

    # Always start in the sitting keyframe. ``set_random_ground_state`` writes
    # the upright orientation, the sitting trunk z, and the sit joint angles
    # for every env (sitting_prob = 1.0). It runs after ``reset_base`` and
    # ``reset_robot_joints`` so it cleanly overrides whatever they set.
    cfg.events["set_ground_state"] = EventTermCfg(
        func=microduck_mdp.set_random_ground_state,
        mode="reset",
        params={
            "face_down_prob": 0.0,
            "face_up_prob":   0.0,
            "sitting_prob":   1.0,
            "sitting_joint_overrides": SITTING_JOINT_OVERRIDES,
        },
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

    if ENABLE_VELOCITY_PUSHES:
        cfg.curriculum["push_magnitude"] = CurriculumTermCfg(
            func=microduck_mdp.push_curriculum,
            params={
                "event_name": "push_robot",
                "push_stages": [
                    {"step": 0,         "velocity_range": {"x": (0.0, 0.0),    "y": (0.0, 0.0)}},
                    {"step": 500 * 24,  "velocity_range": {"x": (-0.08, 0.08), "y": (-0.08, 0.08)}},
                    {"step": 1000 * 24, "velocity_range": {"x": VELOCITY_PUSH_RANGE, "y": VELOCITY_PUSH_RANGE}},
                ],
            },
        )

    return cfg


# ── RL runner config ──────────────────────────────────────────────────────────

MicroduckStandUpRlCfg = RslRlOnPolicyRunnerCfg(
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
    experiment_name="microduck_stand",
    run_name="microduck_stand",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=15_000,
)
