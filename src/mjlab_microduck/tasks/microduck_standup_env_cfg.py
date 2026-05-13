"""Microduck stand-up environment configuration.

The robot is initialized lying on its back (body inverted 180° from upright)
and must learn to right itself and reach a stable standing posture.

Command layout is unified with the velocity env (13D, see mdp.py):
  twist (3)     : velocity commands — kept at ~0 here (we're standing)
  head_pose (4) : neck/head joint deltas from HOME (tracked)
  body_pose (6) : body delta [x, y, z, roll, pitch, yaw] from nominal standing
                  pose (tracked — primary objective once upright).
"""

import math
from copy import deepcopy

# Symmetry
ENABLE_SYMMETRY = False

# Domain randomization toggles
ENABLE_COM_RANDOMIZATION = True
ENABLE_KP_RANDOMIZATION = True
ENABLE_KD_RANDOMIZATION = True
ENABLE_MASS_INERTIA_RANDOMIZATION = True
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True

# Domain randomization ranges
COM_RANDOMIZATION_RANGE = 0.003
MASS_INERTIA_RANDOMIZATION_RANGE = (0.95, 1.05)
KP_RANDOMIZATION_RANGE = (0.85, 1.15)
KD_RANDOMIZATION_RANGE = (0.9, 1.1)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 1.0

# Body pose command control
# v1.5 robot CoM sits ~2 cm higher than the previous robot, so all standup-height
# thresholds are shifted up by 0.02 m vs the main-branch values.
BODY_CMD_NOMINAL_HEIGHT = 0.115
# Tight height range for the standup com_height_target reward.
# Must exclude face-down reset heights (0.20–0.25 m) so the robot is always
# penalized for lying flat and must stand up to earn this reward.
STANDUP_HEIGHT_MIN = 0.095
STANDUP_HEIGHT_MAX = 0.130

# Final body-pose command ranges (reached at end of curriculum). Body pose is
# tracked as a delta from the nominal standing pose.
BODY_CMD_MAX_XY        = 0.02                # ±20 mm lateral/forward
BODY_CMD_MAX_Z         = 0.03                # ±30 mm height
BODY_CMD_MAX_ANGLE     = math.radians(30)    # ±30° per Euler axis
# Head pose: per-joint final caps come from the mechanical XML limits minus
# HOME offset, with ~10% safety margin. See the curriculum below for values.

# Resampling intervals for the pose commands
HEAD_POSE_CMD_RESAMPLE_S = (2.0, 5.0)
BODY_POSE_CMD_RESAMPLE_S = (4.0, 8.0)

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
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import MICRODUCK_ROUGH_TERRAINS_CFG
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG


def make_microduck_standup_env_cfg(play: bool = False, rough: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Microduck stand-up environment configuration.

    The robot starts lying on its back (upside down) and must learn to
    right itself and reach a stable upright stance.
    """

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

    # Net terrain contact force on the trunk shell only (excludes legs and head).
    # Detects when the robot belly-flops onto the ground.
    trunk_impact_cfg = ContactSensorCfg(
        name="trunk_impact_contact",
        primary=ContactMatch(mode="body", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("force",),
        reduce="netforce",
        num_slots=1,
    )

    # Net terrain contact force across the entire head/neck subtree.
    # Detects the robot using its beak or head to break a forward fall.
    head_impact_cfg = ContactSensorCfg(
        name="head_impact_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern="neck",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("force",),
        reduce="netforce",
        num_slots=1,
    )

    foot_frictions_geom_names = (
        "left_foot_collision",
        "right_foot_collision",
    )

    cfg = make_velocity_env_cfg()

    cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg, trunk_impact_cfg, head_impact_cfg)
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
    cfg.viewer.body_name = "trunk_base"

    cfg.episode_length_s = 20.0

    # Action configuration
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # === OBSERVATIONS ===
    del cfg.observations["policy"].terms["base_lin_vel"]

    cfg.observations["critic"].terms["foot_height"].params[
        "asset_cfg"
    ].site_names = site_names
    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=velocity_mdp.base_lin_vel,
        scale=1.0,
    )

    cfg.observations["policy"].terms["projected_gravity"] = deepcopy(
        cfg.observations["policy"].terms["projected_gravity"]
    )
    cfg.observations["policy"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["policy"].terms["base_ang_vel"]
    )

    cfg.observations["policy"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["policy"].terms["base_ang_vel"].delay_max_lag = 3
    cfg.observations["policy"].terms["base_ang_vel"].delay_update_period = 64
    cfg.observations["policy"].terms["projected_gravity"].delay_min_lag = 0
    cfg.observations["policy"].terms["projected_gravity"].delay_max_lag = 3
    cfg.observations["policy"].terms["projected_gravity"].delay_update_period = 64

    cfg.observations["policy"].terms["base_ang_vel"].noise = Unoise(n_min=-0.024, n_max=0.024)
    cfg.observations["policy"].terms["projected_gravity"].noise = Unoise(n_min=-0.007, n_max=0.007)
    cfg.observations["policy"].terms["joint_pos"].noise = Unoise(n_min=-0.0006, n_max=0.0006)
    cfg.observations["policy"].terms["joint_vel"].noise = Unoise(n_min=-0.24, n_max=0.24)

    # 1-ctrl-step lag on joint_vel: the Dynamixel firmware computes
    # present_velocity via a moving-average over the previous position-sample
    # window, so the value the policy actually reads is ~1 control period old.
    cfg.observations["policy"].terms["joint_vel"] = deepcopy(
        cfg.observations["policy"].terms["joint_vel"]
    )
    cfg.observations["policy"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["policy"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["policy"].terms["joint_vel"].delay_update_period = 0
    cfg.observations["policy"].enable_corruption = not play

    # v1.5: exclude passive_* jaw joints so obs is 14-dim (matches action space).
    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    cfg.observations["policy"].terms["joint_pos"].params["asset_cfg"] = passive_excluded
    cfg.observations["policy"].terms["joint_vel"].params["asset_cfg"] = deepcopy(passive_excluded)
    cfg.observations["critic"].terms["joint_pos"].params["asset_cfg"] = deepcopy(passive_excluded)
    cfg.observations["critic"].terms["joint_vel"].params["asset_cfg"] = deepcopy(passive_excluded)

    # === COMMANDS ===
    # twist: kept around for runtime obs-shape parity (3 slots) but mostly idle —
    # standing robot, no walking. Tiny non-zero range to keep input neurons alive.
    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    command.heading_command = False
    command.ranges.heading = None
    command.resampling_time_range = (4.0, 8.0)
    command.debug_vis = False
    command.ranges.lin_vel_x = (-0.01, 0.01)
    command.ranges.lin_vel_y = (-0.01, 0.01)
    command.ranges.ang_vel_z = (-0.05, 0.05)
    cfg.commands["twist"] = microduck_mdp.VelocityCommandCommandOnlyCfg(**vars(command))

    # Head pose command (4D deltas from HOME). Per-joint final caps reflect
    # mechanical limits — see microduck_velocity_env_cfg.py for the full table.
    cfg.commands["head_pose"] = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=HEAD_POSE_CMD_RESAMPLE_S,
        ranges=(
            (-0.05, 0.05),    # neck_pitch
            (-0.05, 0.05),    # head_pitch
            (-0.07, 0.07),    # head_yaw
            (-0.015, 0.015),  # head_roll
        ),
    )
    # Body pose command (6D delta from nominal standing pose).
    cfg.commands["body_pose"] = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=BODY_POSE_CMD_RESAMPLE_S,
        ranges=(
            (-0.005, 0.005),  # x (m)
            (-0.005, 0.005),  # y (m)
            (-0.005, 0.005),  # z (m)
            (-0.05, 0.05),    # roll
            (-0.05, 0.05),    # pitch
            (-0.05, 0.05),    # yaw
        ),
    )

    # Append head + body command obs terms to both policy and critic groups
    # so the obs vector ends with [twist(3), head(4), body(6)].
    for group in ("policy", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=velocity_mdp.generated_commands,
            params={"command_name": "head_pose"},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=velocity_mdp.generated_commands,
            params={"command_name": "body_pose"},
        )

    # === REWARDS ===
    cfg.rewards = {
        # Linear upright reward: +1 when vertical, 0 when horizontal, -1 when inverted.
        # Provides non-zero gradient at every tilt angle, unlike a narrow Gaussian
        # which is ~0 at the 90° prone starting position.
        "upright_linear": RewardTermCfg(
            func=microduck_mdp.body_upright_linear,
            weight=4.0,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
        ),
        # Reward upward CoM velocity: directly incentivizes the dynamic push needed
        # to go from prone to standing. Clamped to zero on the way down so the robot
        # isn't penalized for settling once upright.
        "com_upward_velocity": RewardTermCfg(
            func=microduck_mdp.com_upward_velocity,
            weight=3.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "max_height": 0.115,  # = BODY_CMD_NOMINAL_HEIGHT — no dead zone above STANDUP_HEIGHT_MIN where upward velocity stops paying but com_height isn't yet maxed
            },
        ),
        # Height reward: quadratic penalty below target, +1 when in standing range.
        # Fixed range — intentionally NOT derived from BODY_CMD_MAX_Z so widening
        # the body control range doesn't accidentally include face-down heights.
        "com_height_target": RewardTermCfg(
            func=microduck_mdp.com_height_target,
            weight=5.0,
            params={
                "target_height_min": STANDUP_HEIGHT_MIN,
                "target_height_max": STANDUP_HEIGHT_MAX,
            },
        ),
        # Body pose tracking: 6D Gaussian (x, y, z, roll, pitch, yaw) — primary
        # objective once standing. Weight starts at 0; curriculum kicks it in.
        "body_pose_tracking": RewardTermCfg(
            func=microduck_mdp.body_pose_tracking_6d,
            weight=0.0,
            params={
                "command_name": "body_pose",
                "nominal_height": BODY_CMD_NOMINAL_HEIGHT,
                "xy_std": 0.02,
                "z_std": 0.02,
                "angle_std": math.radians(10),
            },
        ),
        # Head pose tracking: Gaussian over the 4 neck/head joint deltas.
        # Weight ramped by curriculum once standup is solved.
        "head_pose_tracking": RewardTermCfg(
            func=microduck_mdp.head_pose_tracking,
            weight=0.0,
            params={"command_name": "head_pose", "std": 0.5},
        ),
        # Pose reward. Bumped to 3.0 now that the flip is learned — pulls
        # joints back to HOME so the policy stops using saturated hip_yaw /
        # extreme knee/ankle angles to balance.
        "pose": RewardTermCfg(
            func=velocity_mdp.variable_posture,
            weight=3.0,
            params={
                # Legs only — head/neck are command-driven via head_pose_tracking.
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r"^(?!passive_|.*neck.*|.*head.*).*",)),
                "command_name": "twist",
                "std_standing": {r".*": 0.5},
                "std_walking": {r".*": 0.5},
                "std_running": {r".*": 0.5},
                "walking_threshold": 0.01,
                "running_threshold": 1.5,
            },
        ),
        # Regularization — kept very light so motion penalties don't outweigh
        # the upward-velocity and upright rewards during the standup phase.
        "action_rate_l2": RewardTermCfg(
            func=velocity_mdp.action_rate_l2,
            weight=-0.01,
        ),
        "joint_torques_l2": RewardTermCfg(
            func=microduck_mdp.joint_torques_l2,
            weight=-1e-5,
        ),
        "dof_pos_limits": RewardTermCfg(
            func=velocity_mdp.joint_pos_limits,
            weight=-1.0,
        ),
        # Focused L1 penalty on hip_yaw + hip_roll deviation from HOME.
        # Fights the wide-base stance the policy converges to: pose reward
        # (Gaussian) saturates near 1 for any small deviation, and dof_pos_limits
        # only fires past the 90% soft limit — so the policy can splay legs
        # outward to ~25° without paying. L1 gives linear gradient everywhere.
        "hip_yaw_roll_deviation": RewardTermCfg(
            func=microduck_mdp.joint_deviation_l1,
            weight=-2.0,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=(r".*hip_yaw.*", r".*hip_roll.*")
                ),
            },
        ),
        "self_collisions": RewardTermCfg(
            func=velocity_mdp.self_collision_cost,
            weight=-3.0,  # bumped -1 → -3: policy was saturating hip_yaw, causing leg-into-trunk contacts; stronger penalty makes that net-negative vs the gain from saturated stance
            params={"sensor_name": self_collision_cfg.name},
        ),
        # Penalize hard impacts of the trunk shell against the ground.
        # Starts at 0; curriculum ramps up after standup is learned.
        "trunk_impact_penalty": RewardTermCfg(
            func=microduck_mdp.body_impact_cost,
            weight=0.0,
            params={"sensor_name": trunk_impact_cfg.name, "threshold": 5.0},
        ),
        # Penalize hard impacts of the head/neck against the ground.
        # Higher weight than trunk because the head servo is more fragile.
        "head_impact_penalty": RewardTermCfg(
            func=microduck_mdp.body_impact_cost,
            weight=0.0,
            params={"sensor_name": head_impact_cfg.name, "threshold": 2.0},
        ),
        # Penalize sudden torque spikes (gearbox shock proxy).
        # Starts at 0; curriculum ramps up after standup is learned.
        "joint_torque_rate_l2": RewardTermCfg(
            func=microduck_mdp.joint_torque_rate_l2,
            weight=0.0,
        ),
    }

    # === TERMINATIONS ===
    # Remove fell_over — robot starts inverted, would terminate immediately
    del cfg.terminations["fell_over"]

    # === EVENTS ===
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )

    # Robot is pitched 90° forward (belly/front facing ground). At z=0.12 the
    # head collision mesh (≈12–15 cm along neck chain from trunk CoM) clips into
    # the floor, causing immediate MuJoCo NaN. Use z=0.20–0.25 to ensure full
    # clearance. Heights above STANDUP_HEIGHT_MAX still generate a penalty, so
    # the task reward structure is preserved.
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.20, 0.25)
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_frictions_geom_names

    # Override orientation: randomly face-down (belly) or face-up (back), with random yaw.
    cfg.events["set_face_down"] = EventTermCfg(
        func=microduck_mdp.set_random_prone_orientation,
        mode="reset",
    )

    # Terminate environments where MuJoCo physics went NaN (contact instability).
    # The standup task is especially prone to this: the robot starts face-down and
    # generates large contact forces while flipping over. NaN states corrupt network
    # weights — terminating immediately ensures the observation buffer stays finite.
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
    )

    # Domain randomization
    if ENABLE_COM_RANDOMIZATION:
        cfg.events["randomize_com"] = EventTermCfg(
            func=velocity_mdp.randomize_field,
            mode="startup",
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

    # === CURRICULUM ===
    if not rough:
        del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=velocity_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0,          "weight": -0.01},
                {"step": 500 * 24,   "weight": -0.1},
                {"step": 1000 * 24,  "weight": -0.3},
                {"step": 1500 * 24,  "weight": -0.6},
                {"step": 2000 * 24,  "weight": -0.8},
                {"step": 2500 * 24,  "weight": -1.0},
            ],
        },
    )

    # Body pose tracking weight — starts at 0, ramps up after the robot is standing.
    cfg.curriculum["body_pose_tracking_weight"] = CurriculumTermCfg(
        func=velocity_mdp.reward_weight,
        params={
            "reward_name": "body_pose_tracking",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},
                {"step": 1000 * 24,  "weight": 2.0},
                {"step": 1500 * 24,  "weight": 3.5},
                {"step": 2000 * 24,  "weight": 5.0},
            ],
        },
    )

    # Pose reward weight ramp — kept flat at 3.0 until iter 2000 (so the flip
    # exploration phase is unchanged), then bumped late to drive joints to HOME.
    cfg.curriculum["pose_weight"] = CurriculumTermCfg(
        func=velocity_mdp.reward_weight,
        params={
            "reward_name": "pose",
            "weight_stages": [
                {"step": 0,          "weight": 3.0},
                {"step": 3000 * 24,  "weight": 5.0},
                {"step": 4000 * 24,  "weight": 7.0},
            ],
        },
    )

    # Body pose command range: ramped from "alive" tiny range up to the full
    # final range once the standup phase is complete.
    cfg.curriculum["body_pose_range"] = CurriculumTermCfg(
        func=microduck_mdp.pose_command_range_curriculum,
        params={
            "command_name": "body_pose",
            "range_stages": [
                {"step": 0, "ranges": (
                    (-0.005, 0.005),  # x
                    (-0.005, 0.005),  # y
                    (-0.005, 0.005),  # z
                    (-0.05, 0.05),    # roll
                    (-0.05, 0.05),    # pitch
                    (-0.05, 0.05),    # yaw
                )},
                {"step": 1000 * 24, "ranges": (
                    (-0.010, 0.010), (-0.010, 0.010), (-0.010, 0.010),
                    (-math.radians(10), math.radians(10)),
                    (-math.radians(10), math.radians(10)),
                    (-math.radians(10), math.radians(10)),
                )},
                {"step": 1500 * 24, "ranges": (
                    (-0.015, 0.015), (-0.015, 0.015), (-0.020, 0.020),
                    (-math.radians(20), math.radians(20)),
                    (-math.radians(20), math.radians(20)),
                    (-math.radians(20), math.radians(20)),
                )},
                {"step": 2000 * 24, "ranges": (
                    (-BODY_CMD_MAX_XY, BODY_CMD_MAX_XY),
                    (-BODY_CMD_MAX_XY, BODY_CMD_MAX_XY),
                    (-BODY_CMD_MAX_Z,  BODY_CMD_MAX_Z),
                    (-BODY_CMD_MAX_ANGLE, BODY_CMD_MAX_ANGLE),
                    (-BODY_CMD_MAX_ANGLE, BODY_CMD_MAX_ANGLE),
                    (-BODY_CMD_MAX_ANGLE, BODY_CMD_MAX_ANGLE),
                )},
            ],
        },
    )

    # Head pose command range: per-joint, same final caps as vel env.
    # neck/head pitch ±1.10, head_yaw ±1.40, head_roll ±0.31 (mechanical limit).
    cfg.curriculum["head_pose_range"] = CurriculumTermCfg(
        func=microduck_mdp.pose_command_range_curriculum,
        params={
            "command_name": "head_pose",
            "range_stages": [
                # step,                ranges = ((neck_pitch), (head_pitch), (head_yaw),  (head_roll))
                {"step": 0,         "ranges": ((-0.05, 0.05),  (-0.05, 0.05),  (-0.07, 0.07),  (-0.015, 0.015))},
                {"step": 1000 * 24, "ranges": ((-0.55, 0.55),  (-0.55, 0.55),  (-0.70, 0.70),  (-0.15, 0.15))},
                {"step": 2000 * 24, "ranges": ((-1.10, 1.10),  (-1.10, 1.10),  (-1.40, 1.40),  (-0.31, 0.31))},
            ],
        },
    )

    # Head pose tracking weight — ramped after standup is solved.
    cfg.curriculum["head_pose_tracking_weight"] = CurriculumTermCfg(
        func=velocity_mdp.reward_weight,
        params={
            "reward_name": "head_pose_tracking",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},
                {"step": 1000 * 24,  "weight": 1.0},
                {"step": 1500 * 24,  "weight": 2.0},
                {"step": 2000 * 24,  "weight": 3.0},
            ],
        },
    )

    _MAX_PUSH = (-1.0, 1.0)
    cfg.curriculum["push_magnitude"] = CurriculumTermCfg(
        func=microduck_mdp.push_curriculum,
        params={
            "event_name": "push_robot",
            "push_stages": [
                {"step": 0,          "velocity_range": {"x": (-0.3, 0.3),   "y": (-0.3, 0.3)}},
                {"step": 1500 * 24,  "velocity_range": {"x": (-0.6, 0.6),   "y": (-0.6, 0.6)}},
                {"step": 2500 * 24,  "velocity_range": {"x": _MAX_PUSH,     "y": _MAX_PUSH}},
            ],
        },
    )

    if play and "push_robot" in cfg.events:
        cfg.events["push_robot"].params["velocity_range"] = {
            "x": _MAX_PUSH,
            "y": _MAX_PUSH,
        }

    cfg.curriculum["trunk_impact_weight"] = CurriculumTermCfg(
        func=velocity_mdp.reward_weight,
        params={
            "reward_name": "trunk_impact_penalty",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},
                {"step": 1000 * 24,  "weight": -0.05},
                {"step": 2000 * 24,  "weight": -0.2},
            ],
        },
    )

    cfg.curriculum["head_impact_weight"] = CurriculumTermCfg(
        func=velocity_mdp.reward_weight,
        params={
            "reward_name": "head_impact_penalty",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},
                {"step": 1000 * 24,  "weight": -0.2},
                {"step": 2000 * 24,  "weight": -1.0},
                {"step": 3000 * 24,  "weight": -2.5},  # late: kill remaining head-on-ground tendency
            ],
        },
    )

    cfg.curriculum["torque_rate_weight"] = CurriculumTermCfg(
        func=velocity_mdp.reward_weight,
        params={
            "reward_name": "joint_torque_rate_l2",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},
                {"step": 1000 * 24,  "weight": -1e-4},
                {"step": 2000 * 24,  "weight": -5e-4},
            ],
        },
    )

    return cfg


MicroduckStandUpRlCfg = RslRlOnPolicyRunnerCfg(
    policy=RslRlPpoActorCriticCfg(
        init_noise_std=0.3,  # conservative: face-down start + scale=1.0 → init_noise_std=1.0 causes MuJoCo NaN
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
        entropy_coef=0.03,  # needed for back-flip exploration; dropping to 0.02 caused it to be unlearned
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
    experiment_name="standup",
    run_name="standup",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=10_000,
)
