# Microduck SPIN task — fast in-place rotation, on rollers.
#
# Cyclic gesture triggered by button A via the runtime's --ground-pick slot:
# ~1 counter-clockwise turn at ~3 rad/s then a clean stop, standing.
#
# Hybrid:
#   - physics / roller robot  ← task_velocity_rollers.py
#   - cyclic phase machinery ← task_roller_crouch.py
#     (GroundPickPhaseCommand command: [cos(2πφ), sin(2πφ), 0], period 4 s)
#
# Fundamental difference from the crouch: the phase drives a target YAW RATE
# (outcome objective) and not a joint pose. Two decaying primers
# push towards differential rolling — the only certain physical mechanism on
# 4 passive wheels: left skate backwards, right skate forwards.
#
# Unified 61D obs → interchangeable at runtime with roller / ground_pick / crouch.
# See docs/design-notes


# L/R symmetry would turn a left spin into a right spin: forbidden here.
ENABLE_SYMMETRY = False

# DR — taken from the roller env


# The button can be pressed at rest OR while rolling slowly: the policy learns
# to kill the residual momentum before/during the launch of the rotation.
ENTRY_VELOCITY_X = (0.0, 0.3)


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

# Phase envelope: canonical constants defined in task_mdp.py.
SPIN_PERIOD = microduck_mdp.SPIN_PERIOD
_ENVELOPE = {"rate_max": microduck_mdp.SPIN_RATE_MAX, "accel_end": microduck_mdp.SPIN_ACCEL_END, "hold_end": microduck_mdp.SPIN_HOLD_END, "brake_end": microduck_mdp.SPIN_BRAKE_END}
# Neck/head held near neutral EXCEPT head_yaw, left free: it can serve
# as a flywheel to launch the rotation.
NECK_PATTERN_NO_YAW = r"^(neck_pitch|head_pitch|head_roll)$"


def make_microduck_spin_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    # Spin env on rollers, driven by the phase of the ground-pick slot.

    feet_ground_cfg = ContactSensorCfg(name="feet_ground_contact", primary=ContactMatch(mode="subtree", pattern=r"^(ankle_l_v1|ankle_r_v1)$", entity="robot"), secondary=ContactMatch(mode="body", pattern="terrain"), fields=("found", "force"), reduce="netforce", num_slots=1, track_air_time=True)
    self_collision_cfg = ContactSensorCfg(name="self_collision", primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"), secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"), fields=("found",), reduce="none", num_slots=1)

    cfg = make_velocity_env_cfg()
    cfg.scene.entities = {"robot": MICRODUCK_WALK_ROLLERS_ROBOT_CFG}
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)
    cfg.viewer.body_name = "trunk_base"

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # REWARDS
    # ⚠️ angular_momentum is NOT kept: it penalizes the 3D norm of the angular
    # momentum, so it would directly fight the spin. body_ang_vel, on the other hand,
    # only penalizes x/y ("Don't penalize z-angular velocity" in mjlab) →
    # kept, it tames the roll/pitch sway without hindering the rotation.
    keep = {"upright", "body_ang_vel", "action_rate_l2"}
    for name in list(cfg.rewards.keys()):
        if name not in keep:
            del cfg.rewards[name]

    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["upright"].weight = 2.0
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["action_rate_l2"].weight = -1.0

    # Main objective: track the target yaw rate ω*(φ) (trapezoid).
    cfg.rewards["spin_rate_track"] = RewardTermCfg(func=microduck_mdp.spin_rate_track, weight=6.0, params={"command_name": "twist", "std": 1.5, **_ENVELOPE})
    # L1 bootstrap: constant gradient when the Gaussian saturates far from the target.
    cfg.rewards["spin_rate_l1"] = RewardTermCfg(func=microduck_mdp.spin_rate_l1, weight=0.5, params={"command_name": "twist", **_ENVELOPE})
    # Turn IN PLACE, and kill the entry momentum. Strengthened -1.0 -> -3.0: in the
    # calibration run at 500 it. the trunk translated at ~0.35 m/s (~ω·half-track), the
    # signature of a pivot on a single skate rather than a spin centered on the body — this is
    # the only term that distinguishes a centered spin from an off-center pivot.
    # Attenuated during the launch ramp [0, ACCEL_END): this is when the
    # robot must push on the ground to inject angular momentum, and when the entry
    # momentum (up to 0.3 m/s) must be CONVERTED into rotation — charging it full
    # price there would oppose the launch. Full price on cruise/braking/rest.
    cfg.rewards["spin_stay_in_place"] = RewardTermCfg(func=microduck_mdp.spin_stay_in_place, weight=-3.0, params={"command_name": "twist", "launch_scale": microduck_mdp.SPIN_LAUNCH_DRIFT_SCALE, "accel_end": microduck_mdp.SPIN_ACCEL_END})
    # Primer 1: turn BY ROLLING (skates in opposite directions), not by skidding.
    cfg.rewards["spin_wheel_differential"] = RewardTermCfg(func=microduck_mdp.spin_wheel_differential, weight=1.0, params={"command_name": "twist", "omega_scale": microduck_mdp.SPIN_WHEEL_OMEGA_SCALE, **_ENVELOPE})
    # Primer 2: leg scissoring (decays via curriculum, see below).
    cfg.rewards["leg_antisymmetry"] = RewardTermCfg(func=microduck_mdp.leg_antisymmetry, weight=1.0, params={"command_name": "twist", "joint_bases": ("hip_pitch", "knee"), **_ENVELOPE})
    # Both blades on the ground during the spin (no airborne twirl).
    cfg.rewards["spin_grounded"] = RewardTermCfg(func=microduck_mdp.spin_grounded, weight=0.5, params={"sensor_name": "feet_ground_contact", "command_name": "twist", **_ENVELOPE})
    # Stability / sim2real
    cfg.rewards["feet_flat"] = RewardTermCfg(func=microduck_mdp.feet_flat_penalty, weight=-2.0, params={"asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot")), "sensor_name": "feet_ground_contact"})
    cfg.rewards["self_collisions"] = RewardTermCfg(func=mdp.self_collision_cost, weight=-1.0, params={"sensor_name": "self_collision"})
    cfg.rewards["neck_action_rate_l2"] = RewardTermCfg(func=microduck_mdp.neck_action_rate_l2, weight=-0.5)
    cfg.rewards["neck_joint_pos_l2"] = RewardTermCfg(func=microduck_mdp.neck_joint_pos_l2, weight=-0.2, params={"pattern": NECK_PATTERN_NO_YAW})
    cfg.rewards["joint_torques_l2"] = RewardTermCfg(func=microduck_mdp.joint_torques_l2, weight=-1e-3)

    # TERMINATIONS
    cfg.terminations["nan_state"] = TerminationTermCfg(func=microduck_mdp.robot_state_is_nan, time_out=False)

    # EVENTS
    cfg.events["reset_action_history"] = EventTermCfg(func=microduck_mdp.reset_action_history, mode="reset")
    del cfg.events["foot_friction"]

    cfg.events["reset_base"].params["pose_range"]["z"] = (0.1335, 0.1435)
    # Entry momentum: injected via reset_root_state_uniform (CLEAN default state
    # + range), and NOT via push_by_setting_velocity in reset mode, which adds to
    # a potentially divergent root velocity and blows up the base free-joint
    # -> NaN. Known regression from roller_crouch.
    cfg.events["reset_base"].params["velocity_range"] = {"x": ENTRY_VELOCITY_X}

    task_dr.apply_dr(cfg, DR, HEAD_BODY_NAMES, play=play)

    # OBSERVATIONS (unified 61D layout)
    del cfg.observations["actor"].terms["base_lin_vel"]
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]
    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(func=mdp.base_lin_vel, scale=1.0)

    microduck_mdp.wire_sim2real_obs(cfg, imu_delay_max_lag=1, imu_misalignment_deg=DR.imu_orientation_angle_deg if DR.imu_orientation else None, encoder_bias_range=DR.encoder_bias_range if DR.encoder_bias else None, sanitize_critic_sensors=False)

    wheel_cfg = SceneEntityCfg("robot", joint_names=(r"^passive_.*",))
    cfg.observations["critic"].terms["wheel_vel"] = ObservationTermCfg(func=mdp.joint_vel_rel, scale=1.0, params={"asset_cfg": wheel_cfg})

    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(func=microduck_mdp.zero_command_padding, params={"dim": 4})
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(func=microduck_mdp.zero_command_padding, params={"dim": 6})

    # COMMAND: phase (like ground_pick / roller_crouch)
    command: UniformVelocityCommandCfg = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    # period=4.0 = default of --ground-pick-period (nothing to pass to the runtime);
    # randomize_phase=False -> every episode starts standing at phase 0, like the
    # button at deployment. 20 s episode = 5 full cycles of the gesture.
    cfg.commands["twist"] = microduck_mdp.GroundPickPhaseCommandCfg(**{**vars(command), "class_type": microduck_mdp.GroundPickPhaseCommand, "period": SPIN_PERIOD, "randomize_phase": False})

    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # CURRICULUM
    del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(func=microduck_mdp.reward_weight, params={"reward_name": "action_rate_l2", "weight_stages": [{"step": 0, "weight": -0.5}, {"step": 250 * 24, "weight": -0.8}, {"step": 500 * 24, "weight": -1.0}]})
    # The scissor primer fades out: it launches the right mechanism then lets the policy
    # refine its own gesture (free pumping frequency).
    cfg.curriculum["leg_antisym_weight"] = CurriculumTermCfg(func=microduck_mdp.reward_weight, params={"reward_name": "leg_antisymmetry", "weight_stages": [{"step": 0, "weight": 1.0}, {"step": 1500 * 24, "weight": 0.5}, {"step": 3000 * 24, "weight": 0.25}]})
    if DR.com:
        cfg.curriculum["com_range"] = CurriculumTermCfg(func=microduck_mdp.com_range_curriculum, params={"event_name": "randomize_com", "range_stages": [{"step": 0, "range": 0.003}, {"step": 500 * 24, "range": 0.005}, {"step": 1000 * 24, "range": 0.01}]})
    if DR.head_com:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(func=microduck_mdp.com_range_curriculum, params={"event_name": "randomize_head_com", "range_stages": [{"step": 0, "range": 0.003}, {"step": 500 * 24, "range": 0.005}, {"step": 1000 * 24, "range": 0.01}]})

    return cfg


MicroduckSpinRlCfg = RslRlOnPolicyRunnerCfg(actor=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True, distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}), critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True), algorithm=PpoWithSymmetryCfg(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.01, num_learning_epochs=5, num_mini_batches=4, learning_rate=1.0e-3, schedule="adaptive", gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0, symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None), logger=LOCAL_CHECKPOINTS_ONLY, experiment_name="spin", run_name="spin", save_interval=250, num_steps_per_env=24, max_iterations=8_000)
