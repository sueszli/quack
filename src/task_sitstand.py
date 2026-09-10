import math

ENABLE_SYMMETRY = False

# Episode length: room for 2-3 posture segments (dwell 3.5-6.5 s each), i.e.
# at least one full sit → rest → rise → rest cycle per episode.
EPISODE_LENGTH_S = 12.0
# Dwell time in each commanded posture before a resample may flip it. The
# lower bound must comfortably exceed a gentle transition (~1.5 s) plus some
# rest, so "arrive, then hold still" is always trained.
POSTURE_DWELL_S = (3.5, 6.5)
SIT_PROB = 0.5

# SIT keyframe (joint_pos index → angle in rad). Single fixed target.
# STABILITY-VERIFIED 2026-07-27 (sit env, scratchpad sweep_sit_pose2.py):
# knee ±1.35, hip_pitch = HOME ∓ 0.05 lean, ankle 0, hip_roll 0 settles at
# 3-5° tilt for 95-100% of noisy resets. The old keyframe (knee ±1.0472,
# hip_pitch HOME) is NOT statically stable — it tips to ~88° in 1 s and
# silently drove the sit env's whole hop/back-flop/plank exploit chain.
# If the robot or keyframe changes, RE-RUN THE SWEEP — verify tilt, not z.
# Keep in sync with task_sitstand.SITTING_TARGET_OVERRIDES and
# task_standup.SITTING_JOINT_OVERRIDES.
SITTING_TARGET_OVERRIDES = {
    1: 0.0,  # left  hip_roll   (HOME -0.0873)
    2: -0.4079,  # left  hip_pitch  (HOME -0.4579; +0.05 = slight fwd lean)
    3: 1.35,  # left  knee       (HOME -0.0049)
    4: 0.0,  # left  ankle      (HOME +0.4530)
    # neck/head intentionally omitted → steered by the head_pose command.
    10: 0.0,  # right hip_roll   (HOME +0.0873)
    11: 0.4079,  # right hip_pitch  (HOME +0.4579)
    12: -1.35,  # right knee       (HOME +0.0049)
    13: 0.0,  # right ankle      (HOME -0.4530)
}

_LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
_NECK_JOINTS = [5, 6, 7, 8]

# Trunk height targets (m) — both MEASURED in sim, never carried across robot
# or keyframe changes (sit run-1 / standup lessons).
STAND_Z = 0.115
SIT_Z = 0.060

# Upright gating window for ``upright_while_tall``: full upright incentive
# above STAND_UPRIGHT_Z, fades to 0 at SIT_UPRIGHT_Z (committed to the sit).
# Blocks the "tip backward while still high" descent exploit; the always-on
# upright_linear floor covers the seated regime.
STAND_UPRIGHT_Z = 0.10
SIT_UPRIGHT_Z = 0.075

# Target-ramp duration (s): the command term slews an internal target blend
# STAND↔SIT over this time, and the posture rewards track the MOVING target.
# THE anti-crash mechanism (run-1 failure: near-instant transitions). With a
# binary target, arriving early pays the full goal jackpot (~7/step) for
# every step saved, while the linear speed caps integrate to a bounded
# excess-distance cost (~50 total for an instant drop) — crashing won ~7×.
# With the ramp, being AHEAD of the setpoint zeroes the height/composite
# stack for the ramp remainder, so tracking the slow setpoint is the argmax.
# 55 mm over 2 s ≈ 0.028 m/s, comfortably under both caps below.
POSTURE_RAMP_S = 2.0

# Vertical-speed caps (m/s) — now BACKSTOPS for overshoot/bounce around the
# slewed target (see POSTURE_RAMP_S), not the primary gentleness mechanism.
# The rise cap is looser (rising against gravity needs some momentum to get
# over the heels) and is introduced by curriculum only after the rise motion
# has been discovered — see the rise_speed_weight curriculum.
MAX_DESCENT_SPEED = 0.05
MAX_RISE_SPEED = 0.08

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
from .robot import MICRODUCK_STANDUP_ROBOT_CFG
from .task_symmetry import SYMMETRY_CFG, PpoWithSymmetryCfg
from .task_velocity import HEAD_BODY_NAMES, HEAD_POSE_CMD_RESAMPLE_S, LOCAL_CHECKPOINTS_ONLY, MICRODUCK_ROUGH_TERRAINS_CFG

DR = task_dr.DEFAULT_DR
ENCODER_BIAS_RANGE = DR.encoder_bias_range


def make_microduck_sitstand_env_cfg(play: bool = False, rough: bool = False) -> ManagerBasedRlEnvCfg:
    feet_ground_cfg = ContactSensorCfg(name="feet_ground_contact", primary=ContactMatch(mode="geom", pattern=r"^(left_foot_collision|right_foot_collision)$", entity="robot"), secondary=ContactMatch(mode="body", pattern="terrain"), fields=("found", "force"), reduce="netforce", num_slots=1, track_air_time=True)

    self_collision_cfg = ContactSensorCfg(name="self_collision", primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"), secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"), fields=("found",), reduce="none", num_slots=1)

    # NOTE: no head-ground contact penalty here (unlike the sit env). Using the
    # head as a third support point during transitions is explicitly allowed —
    # the plank-as-terminal-rest exploit is anti-selected by posture_composite
    # + posture_stillness instead (both ≈0 at plank tilt/height).

    foot_frictions_geom_names = ("left_foot_collision", "right_foot_collision")

    cfg = make_velocity_env_cfg()

    # Standup robot variant: full collision meshes — the body must physically
    # rest on the ground while seated, and knees/head may touch mid-transition.
    cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)
    cfg.viewer.body_name = "trunk_base"

    cfg.episode_length_s = EPISODE_LENGTH_S

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    for name in ["track_linear_velocity", "track_angular_velocity", "air_time", "foot_clearance", "foot_swing_height", "foot_slip", "pose"]:
        if name in cfg.rewards:
            del cfg.rewards[name]

    # Pose target — legs only (head is command-steered). Generous std keeps
    # gradient alive from either end (~1.35 rad knee delta).
    cfg.rewards["posture_pose_legs"] = RewardTermCfg(func=microduck_mdp.posture_pose_match, weight=4.0, params={"command_name": "twist", "std": 0.5, "joint_indices": _LEG_JOINTS, "sit_overrides": SITTING_TARGET_OVERRIDES})

    cfg.rewards["head_pose_tracking"] = RewardTermCfg(func=microduck_mdp.head_pose_tracking, weight=0.75, params={"command_name": "head_pose", "std": 0.5})

    cfg.rewards["posture_pose_l1"] = RewardTermCfg(func=microduck_mdp.posture_pose_l1, weight=1.0, params={"command_name": "twist", "joint_indices": _LEG_JOINTS, "sit_overrides": SITTING_TARGET_OVERRIDES})

    # Trunk height — two-layer Gaussian (standup recipe: wide layer for the
    # bootstrap pull across the 55 mm travel, sharp layer so the final cm has
    # real gradient instead of a saturated plateau) + L1 transition driver.
    cfg.rewards["posture_height"] = RewardTermCfg(func=microduck_mdp.posture_height_gaussian, weight=1.0, params={"command_name": "twist", "sit_z": SIT_Z, "stand_z": STAND_Z, "std": 0.04})
    cfg.rewards["posture_height_sharp"] = RewardTermCfg(func=microduck_mdp.posture_height_gaussian, weight=1.0, params={"command_name": "twist", "sit_z": SIT_Z, "stand_z": STAND_Z, "std": 0.015})
    # L1 weight 6.0: between sit's 5.0 and standup's 7.5 — resting in the
    # WRONG posture must be clearly net-negative in both directions (staying
    # seated under a stand command was the standup env's stall mode at low L1).
    cfg.rewards["posture_height_l1"] = RewardTermCfg(func=microduck_mdp.posture_height_l1, weight=6.0, params={"command_name": "twist", "sit_z": SIT_Z, "stand_z": STAND_Z})

    # Rise bootstrap — pays for upward motion itself when STAND is commanded
    # and the trunk is below 0.125 (just ABOVE the target so the final cm
    # still pays). Destination-only rewards have zero gradient at zero motion;
    # without this the standup env parked seated. Zero under a SIT command.
    cfg.rewards["rise_bootstrap"] = RewardTermCfg(
        func=microduck_mdp.posture_rise_bootstrap,
        weight=0.75,
        params={
            "command_name": "twist",
            "max_height": 0.125,
            "max_vz": MAX_RISE_SPEED,  # explosive launch can't out-earn a gentle rise
        },
    )

    # ⚠️ POSITIVE weights, deliberately: these three functions ALREADY return
    # negative values (-clamp(...), -|a_z|), same convention as the *_l1_penalty
    # helpers (used with +1/+6 here). Run 7ev90yd9 (2026-08-12) had them at
    # negative weights — the double negative made them REWARDS for violence
    # (logs: Episode_Reward/descent_speed +4.6, rise_speed +2.1, gentle_motion
    # +0.57, the three biggest positive terms) and trained a butt-hopping,
    # crash-sitting policy. Same bug class roller_standup found in gentle_rise.
    # After any reward change, check Episode_Reward/<penalty> stays ≤ 0.
    cfg.rewards["descent_speed"] = RewardTermCfg(func=microduck_mdp.trunk_downward_velocity_penalty, weight=10.0, params={"max_down_vel": MAX_DESCENT_SPEED, "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))})
    cfg.rewards["rise_speed"] = RewardTermCfg(func=microduck_mdp.trunk_upward_velocity_penalty, weight=0.0, params={"max_up_vel": MAX_RISE_SPEED, "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))})
    cfg.rewards["gentle_motion"] = RewardTermCfg(func=microduck_mdp.trunk_vertical_accel_penalty, weight=0.05, params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))})

    # Two-layer upright pressure (sit env values — the anti-flop calibration):
    #  - always-on linear floor: holds the trunk vertical at BOTH rests; at 2.5
    #    "lie on your back" trails upright rest by ~4.5/step (sit run-2 fix).
    #  - height-gated booster: blocks the "tip backward while tall" descent
    #    exploit; during the rise it doubles as an arrival-uprightness pull.
    cfg.rewards["upright_linear"] = RewardTermCfg(func=microduck_mdp.body_upright_linear, weight=2.5, params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))})
    cfg.rewards["upright_while_tall"] = RewardTermCfg(func=microduck_mdp.upright_while_tall, weight=1.5, params={"height_low": SIT_UPRIGHT_Z, "height_high": STAND_UPRIGHT_Z, "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))})

    # Stillness at the commanded posture — "arrive, then rest QUIETLY, UPRIGHT"
    # as an explicit positive peak. The z gate is a band around the commanded
    # height (inactive during transitions); the tilt gate pays nothing for a
    # tilted rest (back/face/side flops earn zero — the sit run-2 exploit).
    cfg.rewards["posture_stillness"] = RewardTermCfg(func=microduck_mdp.posture_stillness, weight=2.0, params={"command_name": "twist", "sit_z": SIT_Z, "stand_z": STAND_Z, "band_full": 0.012, "band_zero": 0.03, "vel_std": 0.05, "tilt_full_deg": 25.0, "tilt_zero_deg": 60.0})

    # Multiplicative goal score vs the COMMANDED target — kills partial-sum
    # farming in both postures (plank, flop, lean, park-1cm-short). Broad stds
    # keep gradient visible far from the goal (standup's proven calibration).
    # head_std adds the neck/head-at-command factor: the first sign-fixed run
    # rested with the head DANGLING to the floor (trunk/legs/z all on target →
    # full composite, only the 0.75 tracking term lost, and the hanging head
    # adds passive stability). With the head factor, the goal state itself
    # requires the head up at its commanded pose; transient head assist
    # mid-transition stays free (composite ≈0 there anyway).
    cfg.rewards["posture_composite"] = RewardTermCfg(
        func=microduck_mdp.posture_composite,
        weight=3.0,
        params={
            "command_name": "twist",
            "sit_overrides": SITTING_TARGET_OVERRIDES,
            "joint_indices": _LEG_JOINTS,
            "sit_z": SIT_Z,
            "stand_z": STAND_Z,
            "height_std": 0.03,
            "upright_std": 0.40,  # ≈ 23° effective — plank (~70°+) scores ~0
            "pose_std": 0.40,
            "head_std": 0.40,  # head fully dropped (~1.2 rad) → factor ~0.01
        },
    )

    cfg.rewards["action_rate_l2"] = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1)
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(func=microduck_mdp.joint_torque_rate_l2, weight=0.0)

    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.05  # velocity value
    cfg.rewards["angular_momentum"].weight = -0.02  # velocity value
    cfg.rewards.pop("soft_landing", None)  # velocity removes it

    cfg.rewards["self_collisions"] = RewardTermCfg(func=mdp.self_collision_cost, weight=-1.0, params={"sensor_name": self_collision_cfg.name})

    if "upright" in cfg.rewards:
        del cfg.rewards["upright"]

    del cfg.observations["actor"].terms["base_lin_vel"]

    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(func=mdp.base_lin_vel, scale=1.0)
    # mjlab 1.3.0 base template adds sensor-based foot_height + height_scan obs.
    # Sitstand has no terrain-height sensor (and drops the walking foot rewards),
    # so remove these terms. foot_air_time/foot_contact(_forces) use the
    # feet_ground_contact sensor, which sitstand does define, so they stay.
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    microduck_mdp.wire_sim2real_obs(cfg, imu_delay_max_lag=1, imu_misalignment_deg=DR.imu_orientation_angle_deg if DR.imu_orientation else None, encoder_bias_range=DR.encoder_bias_range if DR.encoder_bias else None, sanitize_critic_sensors=False)

    cfg.commands["head_pose"] = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=HEAD_POSE_CMD_RESAMPLE_S,
        ranges=(
            (-0.05, 0.05),  # neck_pitch
            (-0.05, 0.05),  # head_pitch
            (-0.07, 0.07),  # head_yaw
            (-0.015, 0.015),  # head_roll
        ),
    )

    # Layout parity with velocity/standup: [twist(3), head_pose(4), body_pose(6)].
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "head_pose"})
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(func=microduck_mdp.zero_command_padding, params={"dim": 6})

    # cmd = [sit_flag, 0, 0]; dwell-time resampling flips the posture mid-
    # episode. "Stand" is the all-zero command (deployment idle parity). The
    # runtime drives this by writing 0/1 into the vx slot of the command
    # buffer. Internally the term slews a target blend over POSTURE_RAMP_S
    # that the posture rewards track (see the constant's comment); the OBS
    # stays the raw binary flag.
    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    command.heading_command = False
    command.ranges.heading = None
    command.resampling_time_range = POSTURE_DWELL_S
    command.debug_vis = False
    cfg.commands["twist"] = microduck_mdp.SitStandCommandCfg(**{**vars(command), "sit_prob": SIT_PROB, "ramp_s": POSTURE_RAMP_S, "sit_z": SIT_Z, "stand_z": STAND_Z})

    # No fall termination: wobbles/tips during transitions must play out so the
    # policy experiences the impact/upright costs instead of a truncated episode.
    if "fell_over" in cfg.terminations:
        del cfg.terminations["fell_over"]
    cfg.terminations["nan_state"] = TerminationTermCfg(func=microduck_mdp.robot_state_is_nan, time_out=False)

    cfg.events["reset_action_history"] = EventTermCfg(func=microduck_mdp.reset_action_history, mode="reset")
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_frictions_geom_names
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)  # match velocity

    cfg.events["reset_base"].params["pose_range"]["z"] = (0.11, 0.12)

    # Reset-state mix: 50% standing / 50% already seated (SIT keyframe with
    # joint/tilt noise). Combined with the independent 50/50 posture command
    # this trains all four cases — sit-from-stand, rise-from-sit, hold-stand,
    # hold-sit — and hands the policy both goal states' values directly (the
    # sit env's discovery-bootstrap lesson, extended to both ends).
    cfg.events["set_ground_state"] = EventTermCfg(
        func=microduck_mdp.set_random_ground_state,
        mode="reset",
        params={
            "face_down_prob": 0.0,
            "face_up_prob": 0.0,
            "sitting_prob": 0.5,
            "standing_prob": 0.5,
            "sitting_joint_overrides": SITTING_TARGET_OVERRIDES,
            "sitting_joint_noise_std": 0.10,  # ≈ 6° per joint
            "sitting_tilt_max": math.radians(8),
            "sitting_z_min": 0.06,  # settles to the 0.060 rest
            "sitting_z_max": 0.075,
            "standing_z_min": 0.11,
            "standing_z_max": 0.12,
        },
    )

    # MuJoCo physics robustness (sit env's contact NaN fix). The standup XML
    # has full collisions on every body; the seated pose puts trunk + folded
    # legs + head all in close ground/self contact. Default nconmax=35 and
    # solver iters=10 overflow the contact solver on sit attempts → NaN →
    # nan_state terminations that punish the descent itself ("learn then
    # unlearn by iter 500" pattern).
    cfg.sim.nconmax = 200
    cfg.sim.mujoco.iterations = 30
    cfg.sim.mujoco.ls_iterations = 50

    task_dr.apply_dr(cfg, DR, HEAD_BODY_NAMES, play=play)

    # NOTE: IMU mounting-misalignment is applied at the OBSERVATION level above
    # (matching velocity) — the old event-based randomize_imu_orientation wrote
    # site_quat, which under mjlab 1.3.0 is neither per-env nor read by the obs.

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

    if not rough:
        del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    # Head pose command range curriculum — same per-joint widening as the
    # velocity/standup envs (5% → 100% of each joint's reachable delta).
    cfg.curriculum["head_pose_range"] = CurriculumTermCfg(func=microduck_mdp.pose_command_range_curriculum, params={"command_name": "head_pose", "range_stages": [{"step": 0, "ranges": ((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015))}, {"step": 500 * 24, "ranges": ((-0.17, 0.17), (-0.17, 0.17), (-0.21, 0.21), (-0.047, 0.047))}, {"step": 1000 * 24, "ranges": ((-0.39, 0.39), (-0.39, 0.39), (-0.49, 0.49), (-0.11, 0.11))}, {"step": 1500 * 24, "ranges": ((-0.72, 0.72), (-0.72, 0.72), (-0.91, 0.91), (-0.20, 0.20))}, {"step": 2000 * 24, "ranges": ((-1.10, 1.10), (-1.10, 1.10), (-1.40, 1.40), (-0.31, 0.31))}]})

    # CoM-randomization range curricula — match velocity (trunk capped at ±15 mm,
    # head at ±10 mm, per the 2026-07 audit).
    if DR.com:
        cfg.curriculum["com_range"] = CurriculumTermCfg(func=microduck_mdp.com_range_curriculum, params={"event_name": "randomize_com", "range_stages": [{"step": 0, "range": 0.003}, {"step": 500 * 24, "range": 0.005}, {"step": 1000 * 24, "range": 0.01}, {"step": 1500 * 24, "range": 0.015}]})

    if DR.head_com:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(func=microduck_mdp.com_range_curriculum, params={"event_name": "randomize_head_com", "range_stages": [{"step": 0, "range": 0.003}, {"step": 500 * 24, "range": 0.005}, {"step": 1000 * 24, "range": 0.01}]})

    # Push curriculum — delayed significantly (sit env lesson): a push
    # mid-transition tips the robot into configurations it can't recover from
    # before the motions have consolidated; early pushes made the sit policy
    # unlearn sitting and converge to "just stand doing nothing".
    if DR.pushes:
        cfg.curriculum["push_magnitude"] = CurriculumTermCfg(func=microduck_mdp.push_curriculum, params={"event_name": "push_robot", "push_stages": [{"step": 0, "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}}, {"step": 1000 * 24, "velocity_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05)}}, {"step": 1500 * 24, "velocity_range": {"x": (-0.10, 0.10), "y": (-0.10, 0.10)}}, {"step": 2000 * 24, "velocity_range": {"x": (-0.20, 0.20), "y": (-0.20, 0.20)}}, {"step": 2500 * 24, "velocity_range": {"x": DR.push_range, "y": DR.push_range}}]})

    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(func=microduck_mdp.reward_weight, params={"reward_name": "action_rate_l2", "weight_stages": [{"step": 0, "weight": -0.1}, {"step": 500 * 24, "weight": -0.2}, {"step": 750 * 24, "weight": -0.4}, {"step": 1000 * 24, "weight": -0.6}, {"step": 1250 * 24, "weight": -0.8}, {"step": 1500 * 24, "weight": -1.0}]})

    # Descent-speed cap tightening: discover the sit under magnitude 10
    # (crash-sit already net-negative), then tighten to 20. POSITIVE weights —
    # the function is self-negating (see the sign-convention warning at the
    # reward definitions).
    cfg.curriculum["descent_speed_weight"] = CurriculumTermCfg(func=microduck_mdp.reward_weight, params={"reward_name": "descent_speed", "weight_stages": [{"step": 0, "weight": 10.0}, {"step": 500 * 24, "weight": 20.0}]})

    # Rise-speed cap — introduced only AFTER the rise motion exists (the
    # standup attempt-tax lesson: any motion-tax during discovery makes
    # exploratory attempts net-negative and the skill is never found).
    # Pushed 750/1250 → 1500/2500: the rise needs a brief dynamic burst to
    # rock over the heels (vz > 0.08 for a few steps), and the first
    # sign-fixed run stalled in a head-down forward fold — a half-finished
    # rise — consistent with the cap taxing the final weight shift while it
    # was still being consolidated. Sit-direction gentleness doesn't depend
    # on this cap (descent_speed covers it), so late is cheap. If the rise
    # degrades when this kicks in, soften the final stage — never earlier.
    cfg.curriculum["rise_speed_weight"] = CurriculumTermCfg(func=microduck_mdp.reward_weight, params={"reward_name": "rise_speed", "weight_stages": [{"step": 0, "weight": 0.0}, {"step": 1500 * 24, "weight": 5.0}, {"step": 2500 * 24, "weight": 10.0}]})

    cfg.curriculum["torque_rate_weight"] = CurriculumTermCfg(func=microduck_mdp.reward_weight, params={"reward_name": "joint_torque_rate_l2", "weight_stages": [{"step": 0, "weight": 0.0}, {"step": 750 * 24, "weight": -5e-4}, {"step": 1250 * 24, "weight": -1e-3}]})

    return cfg


MicroduckSitStandRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,  # matches velocity; normalizer MUST be baked into ONNX by export.py
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
    algorithm=PpoWithSymmetryCfg(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.01, num_learning_epochs=5, num_mini_batches=4, learning_rate=1.0e-3, schedule="adaptive", gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0, symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None),
    logger=LOCAL_CHECKPOINTS_ONLY,
    experiment_name="microduck_sitstand",
    run_name="microduck_sitstand",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=15_000,
)
