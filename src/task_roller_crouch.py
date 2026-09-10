ENABLE_SYMMETRY = False

ENTRY_VELOCITY_X = (0.2, 0.5)  # m/s: the robot arrives rolling

# Cycle timing (phase), 4 segments over a 5 s period:
#   descent      [0, DESCENT_END]        = 0.10*5 = 0.5 s  (go down)
#   low/crouched [DESCENT_END, HOLD_END] = 0.40*5 = 2.0 s  (crouched glide)
#   rise         [HOLD_END, RISE_END]    = 0.10*5 = 0.5 s  (stand up)
#   high/standing [RISE_END, 1.0]        = 0.40*5 = 2.0 s  (standing rest)
# NB: the period MUST match --ground-pick-period at deployment (5.0).
CROUCH_PERIOD = 5.0
DESCENT_END = 0.10
HOLD_END = 0.50
RISE_END = 0.60

# ⚠️ at deployment, at the end of the trick the runtime hands control back to the roller policy
# which restarts from HOME — keep STAND_POSE close to HOME for a clean return.
STAND_POSE = {
    # Read on the REAL robot (read_pose.py) — desired standing stance for the trick.
    "left_hip_yaw": -0.0476,
    "left_hip_roll": -0.0629,
    "left_hip_pitch": -0.2869,
    "left_knee": 0.9618,
    "left_ankle": 1.1674,
    "neck_pitch": 0.6029,
    "head_pitch": 0.543,
    "head_yaw": -0.069,
    "head_roll": -0.0414,
    "right_hip_yaw": -0.0337,
    "right_hip_roll": -0.0061,
    "right_hip_pitch": 0.1534,
    "right_knee": -0.9725,
    "right_ankle": -1.0646,
}

CROUCH_POSE = {
    # Read on the REAL robot (Dynamixel XL330, read_pose.py) — holdable pose.
    "left_hip_yaw": -0.0184,
    "left_hip_roll": 0.0307,
    "left_hip_pitch": 1.4082,
    "left_knee": 1.5248,
    "left_ankle": -0.0675,
    "neck_pitch": 1.0937,
    "head_pitch": 1.2149,
    "head_yaw": -0.0184,
    "head_roll": -0.0368,
    "right_hip_yaw": 0.0184,
    "right_hip_roll": -0.0169,
    "right_hip_pitch": -1.4757,
    "right_knee": -1.5907,
    "right_ankle": 0.0568,
}
CROUCH_POSE_STD = 0.4  # per-joint Gaussian tolerance (rad)
CROUCH_LEAN_PITCH = 0.08  # slight forward lean during the crouch (rad ≈ 4.6°)

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


def make_microduck_roller_crouch_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    feet_ground_cfg = ContactSensorCfg(name="feet_ground_contact", primary=ContactMatch(mode="subtree", pattern=r"^(ankle_l_v1|ankle_r_v1)$", entity="robot"), secondary=ContactMatch(mode="body", pattern="terrain"), fields=("found", "force"), reduce="netforce", num_slots=1, track_air_time=True)
    self_collision_cfg = ContactSensorCfg(name="self_collision", primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"), secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"), fields=("found",), reduce="none", num_slots=1)

    cfg = make_velocity_env_cfg()
    cfg.scene.entities = {"robot": MICRODUCK_WALK_ROLLERS_ROBOT_CFG}
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)
    cfg.viewer.body_name = "trunk_base"

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    keep = {"upright", "body_ang_vel", "angular_momentum", "action_rate_l2"}
    for name in list(cfg.rewards.keys()):
        if name not in keep:
            del cfg.rewards[name]

    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["upright"].weight = 2.0
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02
    cfg.rewards["action_rate_l2"].weight = -1.0

    _pose_params = {"command_name": "twist", "crouch_pose": CROUCH_POSE, "stand_pose": STAND_POSE, "descent_end": DESCENT_END, "hold_end": HOLD_END, "rise_end": RISE_END}
    cfg.rewards["crouch_glide_pose"] = RewardTermCfg(func=microduck_mdp.crouch_glide_pose_by_phase, weight=6.0, params={**_pose_params, "std": CROUCH_POSE_STD})
    cfg.rewards["crouch_glide_pose_l1"] = RewardTermCfg(func=microduck_mdp.crouch_glide_pose_l1, weight=2.0, params=_pose_params)
    cfg.rewards["forward_speed"] = RewardTermCfg(func=microduck_mdp.forward_speed_reward, weight=1.0, params={"vel_ref": 0.2})
    # Slight forward lean during the crouch -> counters the backward tipping observed
    # on the real robot during the fast descent. Gated by the blend (crouch only).
    cfg.rewards["crouch_forward_lean"] = RewardTermCfg(func=microduck_mdp.crouch_forward_lean, weight=1.0, params={"command_name": "twist", "target_pitch": CROUCH_LEAN_PITCH, "std": 0.1, "descent_end": DESCENT_END, "hold_end": HOLD_END, "rise_end": RISE_END})
    cfg.rewards["feet_flat"] = RewardTermCfg(func=microduck_mdp.feet_flat_penalty, weight=-2.0, params={"asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot")), "sensor_name": "feet_ground_contact"})
    cfg.rewards["self_collisions"] = RewardTermCfg(func=mdp.self_collision_cost, weight=-1.0, params={"sensor_name": "self_collision"})
    cfg.rewards["neck_action_rate_l2"] = RewardTermCfg(func=microduck_mdp.neck_action_rate_l2, weight=-0.5)
    cfg.rewards["joint_torques_l2"] = RewardTermCfg(func=microduck_mdp.joint_torques_l2, weight=-1e-3)

    cfg.terminations["nan_state"] = TerminationTermCfg(func=microduck_mdp.robot_state_is_nan, time_out=False)

    cfg.events["reset_action_history"] = EventTermCfg(func=microduck_mdp.reset_action_history, mode="reset")
    del cfg.events["foot_friction"]

    cfg.events["reset_base"].params["pose_range"]["z"] = (0.1335, 0.1435)
    cfg.events["reset_base"].params["velocity_range"] = {"x": ENTRY_VELOCITY_X}

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
    # period=CROUCH_PERIOD (slower descent); randomize_phase=False -> every
    # episode starts standing (phase 0), as at deployment (the button starts the
    # cycle at phase 0). Avoids learning "stay low" from already-low starts.
    cfg.commands["twist"] = microduck_mdp.GroundPickPhaseCommandCfg(**{**vars(command), "class_type": microduck_mdp.GroundPickPhaseCommand, "period": CROUCH_PERIOD, "randomize_phase": False})

    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(func=microduck_mdp.reward_weight, params={"reward_name": "action_rate_l2", "weight_stages": [{"step": 0, "weight": -0.5}, {"step": 250 * 24, "weight": -0.8}, {"step": 500 * 24, "weight": -1.0}]})
    if DR.com:
        cfg.curriculum["com_range"] = CurriculumTermCfg(func=microduck_mdp.com_range_curriculum, params={"event_name": "randomize_com", "range_stages": [{"step": 0, "range": 0.003}, {"step": 500 * 24, "range": 0.005}, {"step": 1000 * 24, "range": 0.01}]})
    if DR.head_com:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(func=microduck_mdp.com_range_curriculum, params={"event_name": "randomize_head_com", "range_stages": [{"step": 0, "range": 0.003}, {"step": 500 * 24, "range": 0.005}, {"step": 1000 * 24, "range": 0.01}]})

    return cfg


MicroduckRollerCrouchRlCfg = RslRlOnPolicyRunnerCfg(actor=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True, distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}), critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True), algorithm=PpoWithSymmetryCfg(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.01, num_learning_epochs=5, num_mini_batches=4, learning_rate=1.0e-3, schedule="adaptive", gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0, symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None), logger=LOCAL_CHECKPOINTS_ONLY, experiment_name="roller_crouch", run_name="roller_crouch", save_interval=250, num_steps_per_env=24, max_iterations=8_000)
