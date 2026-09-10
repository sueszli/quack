# Episodic policy: robot starts standing, rolls forward over the flat top of
# its head, and lands back on its feet. Triggered at deployment like sit/standup
# (policy switch = roll starts immediately; no phase clock, no reference motion).

import math

ENABLE_SYMMETRY = True

# Episode: a CONTROLLED roll takes ~2 s + rise ~1.5 s + settle. Run-3: 4 → 5 s
# (4 s left no room for the rise after a paced roll).
EPISODE_LENGTH_S = 5.0

# Empirically-measured standing trunk height (standup lesson: don't guess).
STAND_Z = 0.115

# (0, 0) = roll from a standstill (run 1). Widen to e.g. (0.0, 0.3) to train
# rolls entered with forward momentum — standing spawns then get a random
# initial forward base velocity, approximating a hand-off from the walking
# policy without simulating the walk itself.
ROULADE_FORWARD_VEL_RANGE = (0.0, 0.0)

# 90° = balanced on the head, 180° = on the back, 270° = supine, ~340° = seated
# leaning back, >260° opens the landing gate. Run-3 change: MAX widened
# 185° → 340° — run-2 logs showed the second half of the roll (supine →
# seated → rise) was never spawned and never learned; spawns past ~300° open
# the landing gate at birth, giving dense on-policy data on the crouch→stand
# last mile (the velstand run-5 crouch-basin lesson).
MIDROLL_PITCH_MIN = math.radians(50.0)
MIDROLL_PITCH_MAX = math.radians(340.0)
MIDROLL_OMEGA_RANGE = (0.0, 3.0)  # rad/s forward momentum at spawn
# Tuck anchor: legs folded (crouch-anchor values from the velstand crouch
# reset) + CHIN TUCK (run-5: neck_pitch −1 / head_pitch +1 puts the flat head
# top squarely on the floor — measured axis_z −0.99 vs +0.6 for the passive
# face-plant; the head-top latch requires this, so mid-roll spawns must
# demonstrate the tucked configuration). Servo-index keyed; mid-roll spawns
# lerp HOME→tuck by a per-env factor.
TUCK_OVERRIDES = {
    2: -1.15,  # left  hip_pitch
    3: 1.25,  # left  knee
    4: 1.05,  # left  ankle
    5: -1.0,  # neck_pitch  (chin tuck)
    6: 1.0,  # head_pitch  (chin tuck)
    11: 1.15,  # right hip_pitch
    12: -1.25,  # right knee
    13: -1.05,  # right ankle
}

LANDING_GATE_LO = math.radians(260.0)
LANDING_GATE_HI = math.radians(330.0)
RISE_GATE_LO = math.radians(180.0)
RISE_GATE_HI = math.radians(260.0)

_LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
_NECK_JOINTS = [5, 6, 7, 8]

import dataclasses

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
from .task_velocity import HEAD_BODY_NAMES, LOCAL_CHECKPOINTS_ONLY

DR = dataclasses.replace(task_dr.DEFAULT_DR, pushes=False)
ENCODER_BIAS_RANGE = DR.encoder_bias_range


def make_microduck_roulade_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    feet_ground_cfg = ContactSensorCfg(name="feet_ground_contact", primary=ContactMatch(mode="geom", pattern=r"^(left_foot_collision|right_foot_collision)$", entity="robot"), secondary=ContactMatch(mode="body", pattern="terrain"), fields=("found", "force"), reduce="netforce", num_slots=1, track_air_time=True)

    self_collision_cfg = ContactSensorCfg(name="self_collision", primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"), secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"), fields=("found",), reduce="none", num_slots=1)

    # Head-ground contact — the roll's pivot signal. jaw_soft is the body that
    # carries the head collision geoms (top_head_shell = the flat top, jaw,
    # bottom_head_shell) in robot_groundcontact.xml. NAME IS LOAD-BEARING:
    # _update_roulade_accum reads it for the over-the-head latch.
    head_ground_cfg = ContactSensorCfg(name="head_ground_contact", primary=ContactMatch(mode="body", pattern="jaw_soft", entity="robot"), secondary=ContactMatch(mode="body", pattern="terrain"), fields=("found",), reduce="none", num_slots=1)

    # Whole-robot ground contact — the SUPPORT GATE (run-2 fix): the rotation
    # accumulator only integrates while some robot geom touches the terrain,
    # so ballistic flips ("breakdance") earn no progress and never complete.
    # NAME IS LOAD-BEARING: _update_roulade_accum reads it.
    robot_ground_cfg = ContactSensorCfg(name="robot_ground_contact", primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"), secondary=ContactMatch(mode="body", pattern="terrain"), fields=("found",), reduce="none", num_slots=1)

    foot_frictions_geom_names = ("left_foot_collision", "right_foot_collision")

    cfg = make_velocity_env_cfg()

    cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg, head_ground_cfg, robot_ground_cfg)
    cfg.viewer.body_name = "trunk_base"

    cfg.episode_length_s = EPISODE_LENGTH_S

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    for name in ["track_linear_velocity", "track_angular_velocity", "air_time", "foot_clearance", "foot_swing_height", "foot_slip", "pose"]:
        if name in cfg.rewards:
            del cfg.rewards[name]

    # Progress increments — the one dense task signal during the roll. During
    # a 1.5 s roll it averages ~0.7/step; total payout per full roll from a
    # standing spawn ≈ weight × (episode steps it took) × mean ≈ weight × 50.
    cfg.rewards["roulade_progress"] = RewardTermCfg(
        func=microduck_mdp.roulade_progress,
        weight=8.0,
        # max_paid_rate: run-4 raised 3 → 5 rad/s. Measured physics (run-3
        # checkpoint eval): the over-the-top transit runs at 3.5–5.5 rad/s —
        # this robot is 10 cm tall, its natural tumble timescale is fast, and
        # the 3 rad/s cap was forfeiting most of the physically-necessary
        # rotation. Style pressure lives in |a_z| / action_rate / the support
        # gate, not in fighting gravity's clock.
        params={"target_angle": 2 * math.pi, "max_paid_rate": 5.0},
    )

    # Whip-speed tax — run-4 threshold 4 → 7 rad/s (above the measured p90
    # transit speed of ~5.5): taxes genuine whips, not the natural tumble.
    cfg.rewards["roulade_overspeed"] = RewardTermCfg(func=microduck_mdp.roulade_overspeed_penalty, weight=-0.1, params={"omega_max": 7.0})

    # Head-as-pivot shaping: contact × mid-roll window × forward-rate factor
    # (the rate factor kills the "rest face-down with head on floor" farm).
    cfg.rewards["roulade_head_pivot"] = RewardTermCfg(func=microduck_mdp.roulade_head_pivot, weight=0.5, params={"sensor_name": head_ground_cfg.name, "angle_lo": math.radians(30.0), "angle_hi": math.radians(240.0), "rate_norm": 2.0})

    # Completion-gated standing annuity — the dominant attractor. Broad stds
    # (standup composite lesson: partial landing must score visibly, ~0.2+).
    cfg.rewards["roulade_landing_composite"] = RewardTermCfg(func=microduck_mdp.roulade_landing_composite, weight=4.0, params={"target_height": STAND_Z, "height_std": 0.04, "upright_std": 0.40, "pose_std": 0.40, "joint_indices": _LEG_JOINTS, "gate_lo": LANDING_GATE_LO, "gate_hi": LANDING_GATE_HI, "target_overrides": None})

    # Completion-gated bootstrap layers (gradient far from the goal, where the
    # composite product is ≈0): linear upright + broad height Gaussian.
    cfg.rewards["roulade_upright_after_roll"] = RewardTermCfg(func=microduck_mdp.roulade_upright_after_roll, weight=1.5, params={"gate_lo": LANDING_GATE_LO, "gate_hi": LANDING_GATE_HI})
    cfg.rewards["roulade_height_after_roll"] = RewardTermCfg(func=microduck_mdp.roulade_height_after_roll, weight=1.0, params={"target_height": STAND_Z, "std": 0.04, "gate_lo": LANDING_GATE_LO, "gate_hi": LANDING_GATE_HI})

    # Sharp landing layer (run-4): tight-std upright × height product on top
    # of the broad composite. Run-3 eval showed EVERY completed episode
    # parking at the same z≈0.105 / 27°-lean pose — the broad stds score ~0.5
    # there, no gradient to finish. Sharp layer: ~0.1 at the basin, ~1.0
    # upright — 10× differential across the last mile.
    cfg.rewards["roulade_landing_sharp"] = RewardTermCfg(func=microduck_mdp.roulade_landing_sharp, weight=2.0, params={"target_height": STAND_Z, "height_std": 0.015, "upright_std": 0.3, "gate_lo": LANDING_GATE_LO, "gate_hi": LANDING_GATE_HI})

    # Completion-gated stand tax (run-3, THE standup lesson): once the
    # rotation is done, every step spent below STAND_Z costs — "crumple in a
    # heap after the roll" flips from free to net-negative, the same fix that
    # broke standup's static-sit basin (its height L1 at ÷4-scaled weight
    # 7.5). Gate closed during the roll, so the roll itself is never taxed;
    # mid/late-roll spawns are born with it active, which is the point.
    cfg.rewards["roulade_stand_tax"] = RewardTermCfg(func=microduck_mdp.roulade_stand_tax, weight=5.0, params={"target_height": STAND_Z, "gate_lo": LANDING_GATE_LO, "gate_hi": LANDING_GATE_HI})

    # Exit-rise bootstrap: upward CoM velocity, gated to the late-roll region
    # (supine → up is the face-up-recovery problem; end-state rewards have zero
    # gradient at zero motion there — standup lesson #2).
    cfg.rewards["roulade_rise_velocity"] = RewardTermCfg(func=microduck_mdp.roulade_rise_velocity, weight=0.75, params={"max_height": STAND_Z + 0.01, "gate_lo": RISE_GATE_LO, "gate_hi": RISE_GATE_HI})

    # Straightness — run-5: the run-4 policy rolled over the SHOULDER (lower
    # energy path than straight over the head — it avoids the fully-inverted
    # configuration, same cheat human beginners default to). The structural
    # fix is the flatness gate on the accumulator + the head-top latch (side
    # rolls no longer count as rotation at all); these penalties provide the
    # dense per-step gradient back toward the plane, weights raised 5× from
    # the run-2 values that were noise against progress@8.
    cfg.rewards["roulade_sagittal"] = RewardTermCfg(func=microduck_mdp.roulade_sagittal_penalty, weight=-0.1)
    cfg.rewards["roulade_lateral_vel"] = RewardTermCfg(func=microduck_mdp.roulade_lateral_velocity_penalty, weight=-0.5)
    cfg.rewards["roulade_flatness"] = RewardTermCfg(func=microduck_mdp.roulade_flatness_penalty, weight=-0.5)

    # Motion-blockers stay near zero during discovery (the roll IS a large
    # angular-velocity + impact event); the settle/polish pressure comes from
    # the LATE-introduced gated terms below (arrival_damping, |a_z|, torque
    # rate) — the standup timing lesson.
    cfg.rewards["action_rate_l2"] = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1)
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(func=microduck_mdp.joint_torque_rate_l2, weight=0.0)

    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.002  # must stay ≈0: the roll is ω
    cfg.rewards["angular_momentum"].weight = -0.001
    cfg.rewards.pop("soft_landing", None)

    cfg.rewards["arrival_damping"] = RewardTermCfg(func=microduck_mdp.body_ang_vel_at_height, weight=0.0, params={"height_low": 0.09, "height_high": 0.11, "tilt_full_deg": 20.0, "tilt_zero_deg": 45.0, "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))})

    # NOTE: trunk_vertical_accel_penalty is SELF-NEGATING (returns -|a_z|) →
    # POSITIVE weight (penalty sign convention; a negative weight here would
    # reward violence — caught in the run-2 smoke test, sum was positive).
    cfg.rewards["gentle_landing"] = RewardTermCfg(func=microduck_mdp.trunk_vertical_accel_penalty, weight=0.002, params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))})

    # Self-collision — LIGHT: a tucked roll needs body-on-body contact
    # (knees against trunk); standup's -1.0 would fight the tuck.
    cfg.rewards["self_collisions"] = RewardTermCfg(func=mdp.self_collision_cost, weight=-0.1, params={"sensor_name": self_collision_cfg.name})

    # Always-on upright would oppose the flip (the old attempt's core failure);
    # landing uprightness is handled by the completion-gated terms above.
    if "upright" in cfg.rewards:
        del cfg.rewards["upright"]

    del cfg.observations["actor"].terms["base_lin_vel"]

    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(func=mdp.base_lin_vel, scale=1.0)
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    microduck_mdp.wire_sim2real_obs(cfg, imu_delay_max_lag=1, imu_misalignment_deg=DR.imu_orientation_angle_deg if DR.imu_orientation else None, encoder_bias_range=DR.encoder_bias_range if DR.encoder_bias else None, sanitize_critic_sensors=False)

    # Command obs slots: zero padding for BOTH head (4) and body (6) — the head
    # is part of the task (it's the pivot), so no head_pose command here, but
    # the 61D obs layout parity with velocity/standup is kept so the runtime
    # stack works unchanged (send zeros).
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(func=microduck_mdp.zero_command_padding, params={"dim": 4})
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(func=microduck_mdp.zero_command_padding, params={"dim": 6})

    # Command: tiny noise around zero (kept for obs-shape parity)
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

    # Falling over is the task — keep only the NaN guard + timeout.
    if "fell_over" in cfg.terminations:
        del cfg.terminations["fell_over"]
    cfg.terminations["nan_state"] = TerminationTermCfg(func=microduck_mdp.robot_state_is_nan, time_out=False)

    cfg.events["reset_action_history"] = EventTermCfg(func=microduck_mdp.reset_action_history, mode="reset")
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_frictions_geom_names
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)

    # Standing start + mid-roll reverse-curriculum spawns; also resets the
    # rotation accumulator (must run after reset_robot_joints — dict insertion
    # order — since mid-roll tuck lerps FROM the HOME pose it wrote).
    cfg.events["set_roulade_state"] = EventTermCfg(func=microduck_mdp.reset_roulade_state, mode="reset", params={"standing_prob": 0.5, "midroll_prob": 0.5, "standing_z_min": 0.11, "standing_z_max": 0.12, "standing_tilt_max": math.radians(5.0), "forward_vel_range": ROULADE_FORWARD_VEL_RANGE, "midroll_pitch_min": MIDROLL_PITCH_MIN, "midroll_pitch_max": MIDROLL_PITCH_MAX, "midroll_z_min": 0.05, "midroll_z_max": 0.10, "midroll_omega_range": MIDROLL_OMEGA_RANGE, "tuck_overrides": TUCK_OVERRIDES, "tuck_factor_range": (0.3, 1.0), "joint_noise_std": 0.08})

    if "push_robot" in cfg.events:
        del cfg.events["push_robot"]

    task_dr.apply_dr(cfg, DR, HEAD_BODY_NAMES, play=play)

    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    if "terrain_levels" in cfg.curriculum:
        del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    # Reverse-curriculum mix: heavy mid-roll early (the completion sub-task is
    # learnable from day 0 — it overlaps face-up recovery), shift toward
    # standing starts as the full roll gets discovered. Mid-roll never goes to
    # zero: it keeps the second half practiced and is realistic DR anyway.
    cfg.curriculum["roulade_spawn_mix"] = CurriculumTermCfg(func=microduck_mdp.event_param_curriculum, params={"event_name": "set_roulade_state", "param_stages": [{"step": 0, "params": {"standing_prob": 0.50, "midroll_prob": 0.50}}, {"step": 3000 * 24, "params": {"standing_prob": 0.65, "midroll_prob": 0.35}}, {"step": 6000 * 24, "params": {"standing_prob": 0.80, "midroll_prob": 0.20}}]})

    if DR.com:
        cfg.curriculum["com_range"] = CurriculumTermCfg(func=microduck_mdp.com_range_curriculum, params={"event_name": "randomize_com", "range_stages": [{"step": 0, "range": 0.003}, {"step": 500 * 24, "range": 0.005}, {"step": 1000 * 24, "range": 0.01}, {"step": 1500 * 24, "range": 0.015}]})

    if DR.head_com:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(func=microduck_mdp.com_range_curriculum, params={"event_name": "randomize_head_com", "range_stages": [{"step": 0, "range": 0.003}, {"step": 500 * 24, "range": 0.005}, {"step": 1000 * 24, "range": 0.01}]})

    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(func=microduck_mdp.reward_weight, params={"reward_name": "action_rate_l2", "weight_stages": [{"step": 0, "weight": -0.1}, {"step": 1500 * 24, "weight": -0.2}, {"step": 3000 * 24, "weight": -0.4}]})

    cfg.curriculum["arrival_damping_weight"] = CurriculumTermCfg(func=microduck_mdp.reward_weight, params={"reward_name": "arrival_damping", "weight_stages": [{"step": 0, "weight": 0.0}, {"step": 2500 * 24, "weight": -0.025}, {"step": 3500 * 24, "weight": -0.05}]})
    cfg.curriculum["torque_rate_weight"] = CurriculumTermCfg(func=microduck_mdp.reward_weight, params={"reward_name": "joint_torque_rate_l2", "weight_stages": [{"step": 0, "weight": 0.0}, {"step": 2500 * 24, "weight": -5e-4}, {"step": 3500 * 24, "weight": -1e-3}]})
    cfg.curriculum["gentle_landing_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            # POSITIVE weights: the func is self-negating (returns -|a_z|).
            "reward_name": "gentle_landing",
            "weight_stages": [{"step": 0, "weight": 0.002}, {"step": 2500 * 24, "weight": 0.005}],
        },
    )

    return cfg


MicroduckRouladeRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,  # normalizer MUST be baked into ONNX by export.py
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
    algorithm=PpoWithSymmetryCfg(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.01, num_learning_epochs=5, num_mini_batches=4, learning_rate=1.0e-3, schedule="adaptive", gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0, symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None),
    logger=LOCAL_CHECKPOINTS_ONLY,
    experiment_name="microduck_roulade",
    run_name="microduck_roulade",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=10_000,
)
