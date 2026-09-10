import math
from dataclasses import dataclass as _dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers import CommandTermCfg
from mjlab.managers.command_manager import CommandTerm
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.reward_manager import RewardManager as _RewardManager
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import observations as _velocity_obs
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg
from mjlab.utils.lab_api.math import euler_xyz_from_quat, matrix_from_quat, quat_apply, quat_from_angle_axis, wrap_to_pi
from rsl_rl.algorithms.ppo import PPO as _PPO

# Patch 1: RewardManager.compute — sanitize NaN rewards before they enter the
# PPO buffer.  mjlab computes rewards BEFORE resetting environments, so any
# reward term operating on a NaN physics state returns NaN.  That NaN
# propagates: NaN reward → NaN advantage → NaN loss → NaN gradient →
# NaN/negative std → crash in torch.normal on the next mini-batch.
_orig_reward_compute = _RewardManager.compute


def _nan_safe_reward_compute(self, dt: float) -> torch.Tensor:
    result = _orig_reward_compute(self, dt)
    # _episode_sums is updated inside compute() before nan_to_num can act.
    # Sanitize in-place so per-term metrics don't show NaN.
    for key in self._episode_sums:
        torch.nan_to_num_(self._episode_sums[key], nan=0.0)
    return torch.nan_to_num(result, nan=0.0)


_RewardManager.compute = _nan_safe_reward_compute

# Patch 2: PPO.compute_returns — sanitize advantages before normalization.
# At a sudden curriculum step (e.g. reward weight ×2.5) the value function is
# badly wrong: all TD errors shift by the same amount, std(advantages) → tiny,
# and (A − mean) / (std + 1e-8) → huge.  That blows up the gradient for std,
# which the optimizer then pushes below zero.  Zeroing NaN/Inf advantages
# before normalization keeps them in a safe range.
_orig_compute_returns = _PPO.compute_returns


def _safe_compute_returns(self, obs) -> None:
    _orig_compute_returns(self, obs)
    st = self.storage
    torch.nan_to_num_(st.advantages, nan=0.0, posinf=0.0, neginf=0.0)
    torch.nan_to_num_(st.returns, nan=0.0, posinf=0.0, neginf=0.0)


_PPO.compute_returns = _safe_compute_returns


print("[mdp] Patches 1-2 active: NaN-safe reward/advantage")

# Patch 4: exporter_utils.get_base_metadata — the new microduck model has
# passive joints (jaw linkage closed via equality constraints) that are part
# of the articulation but have no XML actuator.  The upstream exporter
# iterates robot.joint_names (16) and indexes joint_name_to_ctrl_id (14),
# crashing with KeyError on passive_*.  Filter passive joints out of the
# exported metadata so policies stay consistent with the 14-dim action space.
from mjlab.envs.mdp.actions import JointPositionAction as _JointAction
from mjlab.rl import exporter_utils as _exporter_utils


def _get_base_metadata_no_passive(env, run_path):
    robot = env.scene["robot"]
    joint_action = env.action_manager.get_term("joint_pos")
    assert isinstance(joint_action, _JointAction)
    full_names = list(robot.joint_names)
    keep_idx = [i for i, n in enumerate(full_names) if not n.startswith("passive_")]
    joint_names = [full_names[i] for i in keep_idx]
    joint_name_to_ctrl_id = {a.target.split("/")[-1]: a.id for a in robot.spec.actuators}
    ctrl_ids = [joint_name_to_ctrl_id[n] for n in joint_names]
    stiffness = env.sim.mj_model.actuator_gainprm[ctrl_ids, 0]
    damping = -env.sim.mj_model.actuator_biasprm[ctrl_ids, 2]
    default_jp = robot.data.default_joint_pos[0].cpu().tolist()
    return {"run_path": run_path, "joint_names": joint_names, "joint_stiffness": stiffness.tolist(), "joint_damping": damping.tolist(), "default_joint_pos": [default_jp[i] for i in keep_idx], "command_names": list(env.command_manager.active_terms), "observation_names": env.observation_manager.active_terms["actor"], "action_scale": joint_action._scale[0].cpu().tolist() if isinstance(joint_action._scale, torch.Tensor) else joint_action._scale}


_exporter_utils.get_base_metadata = _get_base_metadata_no_passive
try:
    from mjlab.tasks.velocity.rl import exporter as _vel_exporter

    if hasattr(_vel_exporter, "get_base_metadata"):
        _vel_exporter.get_base_metadata = _get_base_metadata_no_passive
except Exception:
    pass

print("[mdp] Patch 4 active: ONNX export filters passive_* joints")

if TYPE_CHECKING:
    from mjlab.viewer.debug_visualizer import DebugVisualizer


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

_NECK_JOINT_PATTERNS = [r".*neck_pitch.*", r".*head_pitch.*", r".*head_yaw.*", r".*head_roll.*"]


def _servo_joint_ids(env: "ManagerBasedRlEnv", asset: Entity) -> list:
    # Entity-local indices of the servo (non-``passive_``) joints, cached.
    #
    # All joint-index-based reward/event params in this module (``joint_indices``,
    # ``target_overrides``, qpos-column math) are written against the canonical
    # 14-servo layout. On models with extra unactuated joints — backlash hinges,
    # roller wheels, the jaw linkage, all named ``passive_*`` — the entity joint
    # array is wider and interleaved, so raw indices would select the wrong
    # joints. Index through this list to recover the servo-only view; on plain
    # models it is the identity.
    cache = env.__dict__.setdefault("_servo_joint_ids_cache", {})
    key = id(asset)
    ids = cache.get(key)
    if ids is None:
        ids, _ = asset.find_joints(r"^(?!passive_).*")
        cache[key] = ids
    return ids


def _servo_joint_pos(env: "ManagerBasedRlEnv", asset: Entity) -> torch.Tensor:
    return asset.data.joint_pos[:, _servo_joint_ids(env, asset)]


def _servo_joint_vel(env: "ManagerBasedRlEnv", asset: Entity) -> torch.Tensor:
    return asset.data.joint_vel[:, _servo_joint_ids(env, asset)]


def _servo_default_joint_pos(env: "ManagerBasedRlEnv", asset: Entity) -> torch.Tensor:
    return asset.data.default_joint_pos[:, _servo_joint_ids(env, asset)]


def reset_action_history(env: ManagerBasedRlEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG):
    if len(env_ids) == 0:
        return

    asset: Entity = env.scene[asset_cfg.name]

    if hasattr(env, "_prev_leg_actions"):
        if hasattr(env, "action_manager") and env.action_manager.action is not None:
            leg_joint_indices = list(range(5)) + list(range(9, 14))
            env._prev_leg_actions[env_ids] = env.action_manager.action[env_ids][:, leg_joint_indices]
        else:
            env._prev_leg_actions[env_ids] = 0.0

    if hasattr(env, "_prev_neck_actions"):
        if hasattr(env, "action_manager") and env.action_manager.action is not None:
            neck_joint_indices = list(range(5, 9))
            env._prev_neck_actions[env_ids] = env.action_manager.action[env_ids][:, neck_joint_indices]
        else:
            env._prev_neck_actions[env_ids] = 0.0

    if hasattr(env, "_prev_leg_actions_for_acc"):
        if hasattr(env, "action_manager") and env.action_manager.action is not None:
            leg_joint_indices = list(range(5)) + list(range(9, 14))
            current_action = env.action_manager.action[env_ids][:, leg_joint_indices]
            env._prev_leg_actions_for_acc[env_ids] = current_action
            env._prev_prev_leg_actions_for_acc[env_ids] = current_action
        else:
            env._prev_leg_actions_for_acc[env_ids] = 0.0
            env._prev_prev_leg_actions_for_acc[env_ids] = 0.0

    if hasattr(env, "_prev_neck_actions_for_acc"):
        if hasattr(env, "action_manager") and env.action_manager.action is not None:
            neck_joint_indices = list(range(5, 9))
            current_action = env.action_manager.action[env_ids][:, neck_joint_indices]
            env._prev_neck_actions_for_acc[env_ids] = current_action
            env._prev_prev_neck_actions_for_acc[env_ids] = current_action
        else:
            env._prev_neck_actions_for_acc[env_ids] = 0.0
            env._prev_prev_neck_actions_for_acc[env_ids] = 0.0

    if hasattr(asset.data, "_prev_joint_vel"):
        joint_vel = asset.data.joint_vel[env_ids, :][:, asset_cfg.joint_ids]
        asset.data._prev_joint_vel[env_ids] = joint_vel

    if hasattr(env, "_contact_change_count"):
        env._contact_change_count[env_ids] = 0.0
    if hasattr(env, "_contact_change_timer"):
        env._contact_change_timer[env_ids] = 0.0
    if hasattr(env, "_prev_contacts_for_freq"):
        if "feet_ground_contact" in env.scene.sensors:
            contacts = env.scene.sensors["feet_ground_contact"].data.found[env_ids, :2]
            env._prev_contacts_for_freq[env_ids] = contacts

    if hasattr(env, "_prev_foot_forces"):
        if "feet_ground_contact" in env.scene.sensors:
            forces = env.scene.sensors["feet_ground_contact"].data.found[env_ids, :2].squeeze(-1)
            env._prev_foot_forces[env_ids] = forces

    if hasattr(env, "_prev_actuator_forces"):
        env._prev_actuator_forces[env_ids] = asset.data.actuator_force[env_ids].clone()


def neck_action_rate_l2(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    neck_joint_indices = list(range(5, 9))

    if not hasattr(env, "action_manager"):
        return torch.zeros(env.num_envs, device=env.device)

    actions = env.action_manager.action
    if actions.shape[1] < 14:
        return torch.zeros(env.num_envs, device=env.device)

    neck_actions = actions[:, neck_joint_indices]

    if not hasattr(env, "_prev_neck_actions"):
        env._prev_neck_actions = neck_actions.clone()
        return torch.zeros(env.num_envs, device=env.device)

    action_rate = neck_actions - env._prev_neck_actions
    env._prev_neck_actions = neck_actions.clone()

    return torch.sum(torch.square(action_rate), dim=1)


def _fallen_mask(env: ManagerBasedRlEnv, asset, gate_z_below: float, gate_tilt_above_deg: float) -> torch.Tensor:
    # Per-env float mask: 1.0 where the robot counts as FALLEN — trunk height
    # below `gate_z_below` OR tilt beyond `gate_tilt_above_deg`. Used to gate the
    # recovery rewards so they only steer while actually fallen and contribute
    # exactly zero during clean walking (no walk tax / bounce farming).
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    quat = asset.data.root_link_quat_w
    # cos(tilt) = R22 = 1 - 2(qx² + qy²)
    cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    fallen = (z < gate_z_below) | (cos_tilt < math.cos(math.radians(gate_tilt_above_deg)))
    return fallen.float()


def feet_air_time_upright(env: ManagerBasedRlEnv, gate_tilt_above_deg: float = 40.0, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, **air_time_kwargs) -> torch.Tensor:
    # velocity template feet_air_time, zeroed while FALLEN (tilt > gate).
    #
    # velstand: a robot lying on its trunk can still tap its feet rhythmically
    # through the air-time window — the observed "lies there shaking a leg"
    # exploit. Air time is only meaningful upright.
    from mjlab.tasks.velocity.mdp import feet_air_time as _template_air_time

    reward = _template_air_time(env, **air_time_kwargs)
    asset: Entity = env.scene[asset_cfg.name]
    upright = 1.0 - _fallen_mask(env, asset, 0.0, gate_tilt_above_deg)
    return reward * upright


def upright_progress(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    # Potential-based upright shaping: Δcos(tilt) per step.
    #
    # Pays for PROGRESS toward upright, charges for progress toward fallen, and
    # pays exactly ZERO for holding any pose — so no state can farm it (the gated
    # state-reward it replaces was farmed from sitting, lying flat, and a
    # head-tripod lean across three velstand runs). Potential-based shaping is
    # policy-invariant (Ng et al.): it accelerates learning of recovery without
    # creating new optima. A full prone→stand recovery collects Δ≈+1 total
    # (× weight); a fall costs the same on the way down.
    asset: Entity = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w
    cos_tilt = torch.nan_to_num(1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2), nan=1.0)
    if not hasattr(env, "_upright_potential_prev"):
        env._upright_potential_prev = cos_tilt.clone()
    # Freshly reset envs: no spurious delta from the previous episode's pose.
    fresh = env.episode_length_buf <= 1
    env._upright_potential_prev[fresh] = cos_tilt[fresh]
    delta = cos_tilt - env._upright_potential_prev
    env._upright_potential_prev = cos_tilt.clone()
    return delta


def height_progress(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, ceiling: float = 0.115) -> torch.Tensor:
    # Potential-based height shaping: Δ min(trunk z, ceiling) per step.
    #
    # The z-axis companion to ``upright_progress`` (velstand crouch-endpoint
    # lesson): the last mile of a recovery — extending the knees out of a deep
    # crouch — is mostly a HEIGHT change at modest tilt, exactly where the
    # Gaussian upright/pose rewards are flat and Δcos(tilt) is tiny. Rising pays,
    # falling charges, holding pays zero, so gait bobbing nets zero and nothing
    # can farm it. Capped at ``ceiling`` (just below full-stand trunk z ≈ 0.117)
    # so hopping above stance height pays nothing extra.
    asset: Entity = env.scene[asset_cfg.name]
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    pot = torch.clamp(z, max=ceiling)
    if not hasattr(env, "_height_potential_prev"):
        env._height_potential_prev = pot.clone()
    fresh = env.episode_length_buf <= 1
    env._height_potential_prev[fresh] = pot[fresh]
    delta = pot - env._height_potential_prev
    env._height_potential_prev = pot.clone()
    return delta


def fallen_state_penalty(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, gate_tilt_above_deg: float = 40.0, release_tilt_below_deg: float | None = None, release_z_above: float | None = None) -> torch.Tensor:
    # 1.0 while FALLEN (weight it negative): a flat per-step tax on staying
    # down. Without it, lying still is ~0/step while attempting recovery costs
    # action-rate/torque penalties — waiting for the fallen_too_long recycle was
    # the rational policy. (Penalties on bad states are safe; it's POSITIVE
    # rewards gated on bad states that get farmed.)
    #
    # With ``release_*`` set, the tax has HYSTERESIS (velstand crouch-endpoint
    # lesson): a fall arms it and it keeps paying until the robot is genuinely
    # up (tilt < release_tilt AND z > release_z), not merely under the arming
    # gate. Without it, a crouch just below the 40° gate is a zero-cost rest
    # state — recoveries learned to park there instead of finishing the stand.
    # Arms only on a genuine fall, so gait-cycle tilt wobble is never taxed.
    asset: Entity = env.scene[asset_cfg.name]
    fallen = _fallen_mask(env, asset, 0.0, gate_tilt_above_deg).bool()
    if release_tilt_below_deg is None:
        return fallen.float()
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    quat = asset.data.root_link_quat_w
    cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    up = cos_tilt > math.cos(math.radians(release_tilt_below_deg))
    if release_z_above is not None:
        up &= z > release_z_above
    if not hasattr(env, "_fallen_tax_armed"):
        env._fallen_tax_armed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    fresh = env.episode_length_buf <= 1
    env._fallen_tax_armed[fresh] = False
    env._fallen_tax_armed |= fallen
    env._fallen_tax_armed &= ~up
    return env._fallen_tax_armed.float()


def recovery_success(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, fallen_tilt_deg: float = 40.0, min_fallen_s: float = 0.5, up_tilt_deg: float = 25.0, up_z: float = 0.105) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    quat = asset.data.root_link_quat_w
    cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    fallen = cos_tilt < math.cos(math.radians(fallen_tilt_deg))
    up = (cos_tilt > math.cos(math.radians(up_tilt_deg))) & (z > up_z)
    if not hasattr(env, "_recovery_fallen_s"):
        env._recovery_fallen_s = torch.zeros(env.num_envs, device=env.device)
        env._recovery_armed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    fresh = env.episode_length_buf <= 1
    env._recovery_fallen_s[fresh] = 0.0
    env._recovery_armed[fresh] = False
    env._recovery_fallen_s = torch.where(fallen, env._recovery_fallen_s + env.step_dt, torch.zeros_like(env._recovery_fallen_s))
    env._recovery_armed |= env._recovery_fallen_s >= min_fallen_s
    fired = env._recovery_armed & up
    env._recovery_armed &= ~fired
    return fired.float()


def body_upright_linear(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, gate_z_below: float | None = None, gate_tilt_above_deg: float = 40.0) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w
    qx = quat[:, 1]
    qy = quat[:, 2]
    reward = 1.0 - 2.0 * (qx * qx + qy * qy)
    if gate_z_below is not None:
        # Recovery-gated variant (velstand): active only while fallen, exactly
        # zero during clean walking so it can't dilute the tracking rewards.
        reward = reward * _fallen_mask(env, asset, gate_z_below, gate_tilt_above_deg)
    return reward


def body_upright_gaussian(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, std: float = 0.1) -> torch.Tensor:
    # Uses ``2*(qx² + qy²) = 1 - cos(tilt) ≈ tilt²/2`` as a tilt-squared
    # proxy and applies ``exp(-tilt²/std²)``. Default std=0.1 rad ≈ 5.7°.
    asset: Entity = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w
    qx = quat[:, 1]
    qy = quat[:, 2]
    tilt_sq = 2.0 * (qx * qx + qy * qy)
    return torch.exp(-tilt_sq / (std * std))


def upright_gaussian_at_height(env: ManagerBasedRlEnv, std: float, height_low: float, height_high: float, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w
    qx = quat[:, 1]
    qy = quat[:, 2]
    tilt_sq = 2.0 * (qx * qx + qy * qy)
    upright_g = torch.exp(-tilt_sq / (std * std))
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    t = torch.clamp((z - height_low) / max(height_high - height_low, 1e-6), 0.0, 1.0)
    smooth = t * t * (3.0 - 2.0 * t)
    return upright_g * smooth


def body_ang_vel_at_height(env: ManagerBasedRlEnv, height_low: float, height_high: float, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, tilt_full_deg: float | None = None, tilt_zero_deg: float = 45.0) -> torch.Tensor:
    # Height-gated arrival damper: zero below ``height_low`` (ground recovery —
    # flips/rolls need large trunk rotation and must stay free), full above
    # ``height_high``. Same formula as mjlab's body_angular_velocity_penalty
    # (world-frame ω_xy, z-rotation free) but returns the gated POSITIVE cost;
    # use a negative weight.
    #
    # ``tilt_full_deg`` (optional but STRONGLY recommended): additionally gate
    # by tilt — full cost only when tilt ≤ tilt_full_deg, zero when
    # ≥ tilt_zero_deg, smoothstep between. LESSON (2026-07 run that broke
    # front-recovery): with a height gate alone, the final straighten of a
    # bent-over rise (tilt 60°→0 happening INSIDE the z gate) is itself a
    # large trunk rotation — taxing it builds a reward wall right before the
    # finish, and the policy parks bent-over below the gate instead. With the
    # tilt gate, the approach TO vertical is free; only residual wobble
    # AROUND vertical (the overshoot→tip→retry oscillation) is damped.
    asset = env.scene[asset_cfg.name]
    ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :].squeeze(1)
    cost = torch.sum(torch.square(ang_vel[:, :2]), dim=1)
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    t = torch.clamp((z - height_low) / max(height_high - height_low, 1e-6), 0.0, 1.0)
    gate = t * t * (3.0 - 2.0 * t)
    if tilt_full_deg is not None:
        quat = asset.data.root_link_quat_w
        cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
        tilt_deg = torch.rad2deg(torch.acos(cos_tilt.clamp(-1.0, 1.0)))
        s = torch.clamp((tilt_zero_deg - tilt_deg) / max(tilt_zero_deg - tilt_full_deg, 1e-6), 0.0, 1.0)
        gate = gate * (s * s * (3.0 - 2.0 * s))
    return cost * gate


def standing_composite_score(env: ManagerBasedRlEnv, target_height: float, height_std: float, upright_std: float, pose_std: float, joint_indices: list, target_overrides: dict | None = None, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]

    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    height_score = torch.exp(-(((z - target_height) / height_std) ** 2))

    quat = asset.data.root_link_quat_w
    qx = quat[:, 1]
    qy = quat[:, 2]
    tilt_sq = 2.0 * (qx * qx + qy * qy)
    upright_score = torch.exp(-tilt_sq / (upright_std * upright_std))

    target = _servo_default_joint_pos(env, asset).clone()
    if target_overrides:
        for idx, val in target_overrides.items():
            target[:, idx] = val
    joint_pos = _servo_joint_pos(env, asset)[:, joint_indices]
    target = target[:, joint_indices]
    pose_err_sq = ((joint_pos - target) ** 2).mean(dim=-1)
    pose_score = torch.exp(-pose_err_sq / (pose_std * pose_std))

    return height_score * upright_score * pose_score


def com_upward_velocity(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, max_height: float = 0.08, gate_z_below: float | None = None, gate_tilt_above_deg: float = 40.0, max_vz: float | None = None) -> torch.Tensor:
    # ``max_vz`` (optional): cap the rewarded velocity. Uncapped, the reward is
    # proportional to vz, which pays MORE per step for an explosive launch —
    # a violent-rise incentive. With a cap, any rise ≥ max_vz earns the same,
    # so the gentlest rise that reaches the cap is optimal (the |a_z| penalty
    # then picks the smooth one). The bootstrap property is preserved: any
    # upward motion still pays immediately.
    asset: Entity = env.scene[asset_cfg.name]
    # nan_to_num: MuJoCo can produce NaN on contact instability; treat as z=0
    com_z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    vz = torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0)
    below_target = (com_z < max_height).float()
    reward = torch.clamp(vz, min=0.0, max=max_vz) * below_target
    if gate_z_below is not None:
        # Recovery-gated (velstand): without the gate this pays for dip-and-rise
        # during gait whenever the trunk crosses max_height → bounce incentive.
        reward = reward * _fallen_mask(env, asset, gate_z_below, gate_tilt_above_deg)
    return reward


def fallen_too_long(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, gate_z_below: float = 0.10, gate_tilt_above_deg: float = 40.0, max_duration_s: float = 5.0) -> torch.Tensor:
    # For envs that mix walking with fall recovery (velstand): the fell_over
    # termination gets disabled by curriculum so the policy can attempt recovery,
    # but without a backstop a failed recovery farms recovery-reward for the whole
    # 20 s episode, starving the walk of data (audit: ~25% walking share). This
    asset: Entity = env.scene[asset_cfg.name]
    fallen = _fallen_mask(env, asset, gate_z_below, gate_tilt_above_deg).bool()
    if not hasattr(env, "_fallen_timer_s"):
        env._fallen_timer_s = torch.zeros(env.num_envs, device=env.device)
    env._fallen_timer_s[env.episode_length_buf <= 1] = 0.0
    env._fallen_timer_s = torch.where(fallen, env._fallen_timer_s + env.step_dt, torch.zeros_like(env._fallen_timer_s))
    return env._fallen_timer_s >= max_duration_s


def robot_state_is_nan(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, sensor_names: tuple[str, ...] = ()) -> torch.Tensor:
    # MuJoCo's contact solver can overflow to NaN under extreme penetration or
    # impulse (e.g. robot landing at high velocity). A NaN simulation state
    # propagates into observations, corrupting the policy network weights.
    #
    # Note: the reward at THIS terminal step may still be NaN from the simulation;
    # mjlab computes rewards before resetting (see manager_based_rl_env.py step()).
    # Our custom reward functions guard against NaN internally with nan_to_num,
    # but standard mjlab rewards can still be NaN here. One NaN reward is
    # tolerable because done=True prevents it propagating backward through GAE.
    #
    # Covers the WHOLE physical state, not just joint_pos: contact divergence
    # often blows up the base FREE-JOINT (position/orientation/velocity) or the
    # passive WHEELS, not the actuated joints. These quantities
    # feed critic obs terms (base_lin_vel, base_ang_vel,
    # projected_gravity, wheel_vel); if they are not monitored, the env does not
    # reset and the NaN reaches the obs → rsl_rl's check_nan kills the whole
    # training. We test non-finiteness (NaN AND inf, the inf becoming NaN
    # downstream during the normalization of projected_gravity).
    asset: Entity = env.scene[asset_cfg.name]
    d = asset.data
    bad = ~torch.isfinite(d.joint_pos).all(dim=1)
    bad |= ~torch.isfinite(d.joint_vel).all(dim=1)
    bad |= ~torch.isfinite(d.root_link_pos_w).all(dim=1)
    bad |= ~torch.isfinite(d.root_link_quat_w).all(dim=1)
    bad |= ~torch.isfinite(d.root_link_lin_vel_w).all(dim=1)
    bad |= ~torch.isfinite(d.root_link_ang_vel_w).all(dim=1)

    # Contact FORCES can blow up a step before qpos/qvel do: MuJoCo resolves a
    # degenerate contact into an inf/NaN impulse while the integrated state is
    # still finite. That force feeds the critic-only `foot_contact_forces` obs
    # (sign(F)*log1p(|F|)), which the state checks above do NOT cover — so the
    # env was not reset and the NaN reached the runner's check_nan, killing the
    # whole run (crash 2026-08-21, Velocity2-Rough-Backlash with hfield slopes).
    for name in sensor_names:
        if name not in env.scene.sensors:
            continue
        force = getattr(env.scene.sensors[name].data, "force", None)
        if force is not None:
            bad |= ~torch.isfinite(force).flatten(start_dim=1).all(dim=1)
    return bad


def root_height_below(env: ManagerBasedRlEnv, min_height: float, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    # Used by roller_slope as "fell into the void": the terrain has an
    # exit flat at the bottom of the ramp, so a normal descent never goes
    # below the level of the lowest exit flat. Choosing min_height
    # below that level => the termination only triggers if the robot
    # leaves solid ground and falls into the void. Independent of the exact
    # ramp geometry (length/slope).
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w[:, 2] < min_height


def descent_speed_reward(env: ManagerBasedRlEnv, cap: float = 0.8, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    # The ramp descends in +x, so the world linear velocity in x measures the
    # descent progress. Capped at ``cap`` m/s: encourages letting oneself
    # glide without pushing to hurtle down faster and faster. Zero if the robot
    # goes backwards/uphill (vx < 0). Without this reward, the optimum is to stay
    # motionless and upright (the robot "brakes" instead of gliding). NaN-safe.
    asset: Entity = env.scene[asset_cfg.name]
    vx = torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 0], nan=0.0, posinf=0.0, neginf=0.0)
    return torch.clamp(vx, min=0.0, max=cap)


def reset_rolling_entry(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, speed_range: tuple = (0.25, 0.45), wheel_radius: float = 0.0175, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> None:
    # Draws a forward speed v per env; sets the base LINEAR velocity (world
    # x) = v AND the ROTATION velocity of the 4 passive wheels = v / r, so
    # ω·r = v => zero slip at contact. Avoids the jolt of the old base-only
    # push (moving base, stationary wheels = brutal skidding on the 1st step).
    # To be executed AFTER reset_base (which places the base; stop giving it a
    # velocity_range).
    asset: Entity = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n = int(env_ids.shape[0])
    lo, hi = speed_range
    v = torch.rand(n, device=env.device) * (hi - lo) + lo

    root_vel = torch.zeros(n, 6, device=env.device)
    root_vel[:, 0] = v
    asset.write_root_link_velocity_to_sim(root_vel, env_ids=env_ids)

    # Rotation of the 4 passive wheels = v / r (positive = forward, cf. wheel_speed).
    wheel_ids = []
    for name in ("passive_LF_?wheel", "passive_LR_?wheel", "passive_RF_?wheel", "passive_RR_?wheel"):
        ids, _ = asset.find_joints(name)
        wheel_ids.append(ids[0])
    wheel_ids_t = torch.tensor(wheel_ids, device=env.device)
    omega = (v / wheel_radius).unsqueeze(1).repeat(1, len(wheel_ids))
    asset.write_joint_velocity_to_sim(omega, joint_ids=wheel_ids_t, env_ids=env_ids)


def wheel_glide_reward(env: ManagerBasedRlEnv, cap_speed: float = 0.35, wheel_radius: float = 0.0175) -> torch.Tensor:
    # Unlike descent_speed (velocity of the BASE, reachable by
    # "running"/pushing), we reward the rotation of the passive WHEELS = true
    # glide by rolling. Independent of any command (the slope task has a
    # zero command: the glide comes from gravity). Capped at ``cap_speed``
    # (m/s of rolling speed) -> NO incentive to accelerate beyond; zero
    # if the wheels go backwards (uphill). NaN-safe.
    asset: Entity = env.scene["robot"]
    lf, _ = asset.find_joints("passive_LF_?wheel")
    lr, _ = asset.find_joints("passive_LR_?wheel")
    rf, _ = asset.find_joints("passive_RF_?wheel")
    rr, _ = asset.find_joints("passive_RR_?wheel")
    vel = asset.data.joint_vel
    # The 4 wheels spin positive for forward (cf. wheel_speed_reward).
    omega = (vel[:, lf[0]] + vel[:, lr[0]] + vel[:, rf[0]] + vel[:, rr[0]]) / 4.0
    speed = torch.nan_to_num(omega * wheel_radius, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.clamp(speed, min=0.0, max=cap_speed)


def is_alive(env: ManagerBasedRlEnv) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


def com_height_target(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, target_height_min: float = 0.1, target_height_max: float = 0.15) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]

    # env_origins[:, 2] is 0 for flat ground, so this is safe unconditionally.
    # nan_to_num: MuJoCo can produce NaN on contact instability; treat as z=0
    # so the penalty is finite (small, since 0 is near the target range).
    com_height = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)

    below_min = com_height < target_height_min
    above_max = com_height > target_height_max
    in_range = ~(below_min | above_max)

    penalty_below = torch.square(com_height - target_height_min) * below_min.float()
    penalty_above = torch.square(com_height - target_height_max) * above_max.float()

    reward = in_range.float() - (penalty_below + penalty_above)

    return reward


def crouch_height_target(phase: torch.Tensor, height_low: float, height_high: float, hold_lo: float = 0.375, hold_hi: float = 0.625) -> torch.Tensor:
    descend = phase < hold_lo
    hold = (phase >= hold_lo) & (phase < hold_hi)

    frac_d = phase / hold_lo
    t_descend = height_high + (height_low - height_high) * frac_d

    t_hold = torch.full_like(phase, height_low)

    frac_r = (phase - hold_hi) / (1.0 - hold_hi)
    t_rise = height_low + (height_high - height_low) * frac_r

    return torch.where(descend, t_descend, torch.where(hold, t_hold, t_rise))


def crouch_glide_reward_from_values(com_height: torch.Tensor, cmd_cos: torch.Tensor, cmd_sin: torch.Tensor, height_low: float, height_high: float, hold_lo: float = 0.375, hold_hi: float = 0.625, std: float = 0.02) -> torch.Tensor:
    phase = (torch.atan2(cmd_sin, cmd_cos) / (2 * torch.pi)) % 1.0
    target = crouch_height_target(phase, height_low, height_high, hold_lo, hold_hi)
    return torch.exp(-(((com_height - target) / std) ** 2))


def forward_speed_reward(env: ManagerBasedRlEnv, vel_ref: float = 0.2, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    # Independent of the command (the command carries the phase, not the speed).
    asset: Entity = env.scene[asset_cfg.name]
    vx = asset.data.root_link_lin_vel_b[:, 0]
    return torch.tanh(torch.clamp(vx, min=0.0) / vel_ref)


def crouch_pose_blend(phase: torch.Tensor, descent_end: float, hold_end: float, rise_end: float) -> torch.Tensor:
    b = torch.zeros_like(phase)
    descend = phase < descent_end
    b = torch.where(descend, phase / descent_end, b)
    low = (phase >= descent_end) & (phase < hold_end)
    b = torch.where(low, torch.ones_like(phase), b)
    rise = (phase >= hold_end) & (phase < rise_end)
    b = torch.where(rise, 1.0 - (phase - hold_end) / (rise_end - hold_end), b)
    return b


def _crouch_pose_error(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, command_name: str, crouch_pose: dict, descent_end: float, hold_end: float, rise_end: float, stand_pose: dict | None = None):
    # given, else the model DEFAULT (HOME). Joints are resolved BY NAME so the
    # passive-wheel interspersing on the roller robot never shifts an index.
    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    phase = (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * torch.pi)) % 1.0
    blend = crouch_pose_blend(phase, descent_end, hold_end, rise_end)

    names = list(crouch_pose.keys())
    ids = [int(asset.find_joints([n])[0][0]) for n in names]
    default = asset.data.default_joint_pos[:, ids]

    stand = default.clone()
    if stand_pose:
        for j, n in enumerate(names):
            if n in stand_pose:
                stand[:, j] = stand_pose[n]
    crouch = torch.tensor([crouch_pose[n] for n in names], device=env.device, dtype=default.dtype).unsqueeze(0)

    target = stand + blend.unsqueeze(-1) * (crouch - stand)
    cur = asset.data.joint_pos[:, ids]
    return cur, target


def crouch_glide_pose_by_phase(env: ManagerBasedRlEnv, command_name: str = "twist", crouch_pose: dict | None = None, stand_pose: dict | None = None, std: float = 0.4, descent_end: float = 0.10, hold_end: float = 0.50, rise_end: float = 0.60, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    cur, target = _crouch_pose_error(env, asset_cfg, command_name, crouch_pose or {}, descent_end, hold_end, rise_end, stand_pose)
    return torch.exp(-(((cur - target) / std) ** 2)).mean(dim=-1)


def crouch_glide_pose_l1(env: ManagerBasedRlEnv, command_name: str = "twist", crouch_pose: dict | None = None, stand_pose: dict | None = None, descent_end: float = 0.10, hold_end: float = 0.50, rise_end: float = 0.60, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    cur, target = _crouch_pose_error(env, asset_cfg, command_name, crouch_pose or {}, descent_end, hold_end, rise_end, stand_pose)
    return -(cur - target).abs().mean(dim=-1)


def crouch_forward_lean(env: ManagerBasedRlEnv, command_name: str = "twist", target_pitch: float = 0.08, std: float = 0.1, descent_end: float = 0.10, hold_end: float = 0.50, rise_end: float = 0.60, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=("trunk_base",))) -> torch.Tensor:
    # Counters the backward tipping induced by the fast hip flexion. Pitch
    # proxy = projected_gravity_b[:,0] (positive = forward, verified). The gate
    # (blend) is 1 during descent+low, 0 standing → ONLY biases the crouch.
    # small target_pitch = "by very little".
    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    phase = (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * torch.pi)) % 1.0
    gate = crouch_pose_blend(phase, descent_end, hold_end, rise_end)
    lean = asset.data.projected_gravity_b[:, 0]
    return gate * torch.exp(-((lean - target_pitch) ** 2) / std**2)


_NECK_JOINT_CFG = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*(neck|head).*",))
_HIP_PITCH_KNEE_CFG = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*(hip_pitch|knee).*",))
_ROLLER_FEET_SITE_CFG = SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))


def feet_flat_penalty(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROLLER_FEET_SITE_CFG, sensor_name: str | None = None) -> torch.Tensor:
    # When ``sensor_name`` is given, each foot's penalty is GATED by that foot's own
    # ground contact: the airborne (swing) foot is free to tilt, only the stance
    # blade is asked to stay flat (so its wheels keep gripping). Without this gate
    import torch.nn.functional as F
    from mjlab.utils.lab_api.math import quat_apply_inverse

    asset: Entity = env.scene[asset_cfg.name]
    gravity_w_n = F.normalize(asset.data.gravity_vec_w, dim=-1)

    foot_quats = asset.data.site_quat_w[:, asset_cfg.site_ids, :]
    per_foot = torch.zeros(env.num_envs, foot_quats.shape[1], device=env.device)
    for i in range(foot_quats.shape[1]):
        proj = quat_apply_inverse(foot_quats[:, i, :], gravity_w_n)
        per_foot[:, i] = torch.sum(torch.square(proj[:, :2]), dim=1)

    if sensor_name is not None:
        from mjlab.sensor import ContactSensor

        sensor: ContactSensor = env.scene[sensor_name]
        contact_time = sensor.data.current_contact_time
        assert contact_time is not None
        per_foot = per_foot * (contact_time > 0.0).float()

    return per_foot.sum(dim=1)


def neck_joint_pos_l2(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _NECK_JOINT_CFG, pattern: str = r".*(neck|head).*") -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    # Exclude passive_* joints (backlash hinges also contain "neck"/"head").
    if not pattern.startswith(r"^(?!passive_)"):
        pattern = r"^(?!passive_)" + pattern.lstrip("^")
    joint_ids, _ = asset.find_joints(pattern)
    error = asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]
    return torch.sum(torch.square(error), dim=1)


def joint_torques_l2(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]

    actuator_forces = asset.data.actuator_force

    return torch.sum(torch.square(actuator_forces), dim=1)


def joint_torque_rate_l2(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    current = asset.data.actuator_force

    if not hasattr(env, "_prev_actuator_forces"):
        env._prev_actuator_forces = current.clone()
        return torch.zeros(env.num_envs, device=env.device)

    rate = current - env._prev_actuator_forces
    env._prev_actuator_forces = current.clone()
    return torch.sum(torch.square(rate), dim=1)


def feet_grounded_reward(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    if sensor_name not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)
    sensor = env.scene.sensors[sensor_name]
    found = sensor.data.found
    if found.dim() > 1:
        found = found.sum(dim=-1)
    return torch.clamp(found, 0.0, 2.0) / 2.0


def body_impact_cost(env: ManagerBasedRlEnv, sensor_name: str, threshold: float = 1.0) -> torch.Tensor:
    if sensor_name not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)

    sensor = env.scene.sensors[sensor_name]
    forces = sensor.data.force
    total_force = forces.sum(dim=1)
    force_mag = torch.norm(total_force, dim=1)
    return torch.clamp(force_mag - threshold, min=0.0)


def wheel_speed_reward(env: ManagerBasedRlEnv, command_name: str, wheel_radius: float = 0.0175, vel_scale: float = 0.5, bidirectional: bool = False) -> torch.Tensor:
    #   cmd_x > 0, silent otherwise (cmd_x < 0 handled by the braking reward).
    # - ``bidirectional=True``: reward wheel spin in the COMMANDED direction —
    #   forward for cmd_x > 0, backward for cmd_x < 0 — with magnitude |cmd_x|.
    #   Lets cmd_x < 0 mean "go backward" instead of "brake".
    cmd_x = env.command_manager.get_command(command_name)[:, 0]  # (B,)

    asset: Entity = env.scene["robot"]
    lf_ids, _ = asset.find_joints("passive_LF_?wheel")
    lr_ids, _ = asset.find_joints("passive_LR_?wheel")
    rf_ids, _ = asset.find_joints("passive_RF_?wheel")
    rr_ids, _ = asset.find_joints("passive_RR_?wheel")

    vel = asset.data.joint_vel
    # All 4 wheels spin positive for forward motion (verified by test_wheel_direction.py)
    forward_omega = (vel[:, lf_ids[0]] + vel[:, lr_ids[0]] + vel[:, rf_ids[0]] + vel[:, rr_ids[0]]) / 4.0

    omega_scale = vel_scale / wheel_radius
    if bidirectional:
        # spin aligned with the command sign (fwd for +, back for -)
        aligned = torch.sign(cmd_x) * forward_omega
        return torch.abs(cmd_x) * torch.tanh(torch.clamp(aligned, min=0.0) / omega_scale)
    return torch.clamp(cmd_x, min=0.0) * torch.tanh(torch.clamp(forward_omega, min=0.0) / omega_scale)


def braking_reward(env: ManagerBasedRlEnv, command_name: str, vel_std: float = 0.3) -> torch.Tensor:
    # - At cmd_x = -1 and vel = 0: reward = 1.0 (full stop achieved).
    # - At cmd_x = -1 and vel = vel_std: reward ≈ 0.37 (strong gradient).
    # vel_std=0.3 m/s gives meaningful gradient down to walking-pace speeds.
    cmd = env.command_manager.get_command(command_name)
    cmd_x = cmd[:, 0]
    braking_strength = torch.clamp(-cmd_x, min=0.0)
    fwd_vel = env.scene["robot"].data.root_link_lin_vel_b[:, 0]
    stopped = torch.exp(-(fwd_vel.clamp(min=0.0) ** 2) / (vel_std**2))
    return braking_strength * stopped


def joint_deviation_l1(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    jnt_ids = asset_cfg.joint_ids
    err = asset.data.joint_pos[:, jnt_ids] - asset.data.default_joint_pos[:, jnt_ids]
    return torch.sum(torch.abs(err), dim=-1)


def pose_target_match(env: ManagerBasedRlEnv, target_overrides: dict | None = None, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, std: float = 0.3, joint_indices: list | None = None) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    target = _servo_default_joint_pos(env, asset).clone()
    if target_overrides:
        for idx, val in target_overrides.items():
            target[:, idx] = val
    joint_pos = _servo_joint_pos(env, asset)
    if joint_indices is not None:
        joint_pos = joint_pos[:, joint_indices]
        target = target[:, joint_indices]
    return torch.exp(-(((joint_pos - target) / std) ** 2)).mean(dim=-1)


def pose_l1_penalty(env: ManagerBasedRlEnv, target_overrides: dict | None = None, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, joint_indices: list | None = None) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    target = _servo_default_joint_pos(env, asset).clone()
    if target_overrides:
        for idx, val in target_overrides.items():
            target[:, idx] = val
    joint_pos = _servo_joint_pos(env, asset)
    if joint_indices is not None:
        joint_pos = joint_pos[:, joint_indices]
        target = target[:, joint_indices]
    return -torch.abs(joint_pos - target).mean(dim=-1)


def height_target_gaussian(env: ManagerBasedRlEnv, target_height: float, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, std: float = 0.02) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    return torch.exp(-(((z - target_height) / std) ** 2))


def height_l1_penalty(env: ManagerBasedRlEnv, target_height: float, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    return -torch.abs(z - target_height)


def trunk_vertical_accel_penalty(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    vz = torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0)
    prev = getattr(env, "_prev_trunk_vz", None)
    if prev is None or prev.shape[0] != vz.shape[0]:
        prev = vz.detach().clone()
    a_z = (vz - prev) / env.step_dt
    # Zero out a_z at reset steps to suppress the cross-episode transient.
    if hasattr(env, "episode_length_buf"):
        reset_mask = env.episode_length_buf <= 1
        a_z = torch.where(reset_mask, torch.zeros_like(a_z), a_z)
    env._prev_trunk_vz = vz.detach().clone()
    return -torch.abs(a_z)


def trunk_downward_velocity_penalty(env: ManagerBasedRlEnv, max_down_vel: float = 0.05, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    # one impact spike at the bottom — cheap relative to arriving at the target
    # pose sooner. This term makes every step of a too-fast descent cost reward,
    # so the gentlest descent that stays under the cap is optimal. Zero at rest
    # and for any motion slower than the cap (including all upward motion).
    asset = env.scene[asset_cfg.name]
    vz = torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0)
    return -torch.clamp(-vz - max_down_vel, min=0.0)


def upright_while_tall(env: ManagerBasedRlEnv, height_low: float, height_high: float, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    # fades to zero once it has committed to the lower sit configuration (where
    # butt-on-ground orientation is fine). Prevents the policy from learning to
    # tip backward while still high (which would otherwise farm the descent
    # reward via a controlled fall).
    asset = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w
    qx = quat[:, 1]
    qy = quat[:, 2]
    upright = 1.0 - 2.0 * (qx * qx + qy * qy)
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    t = torch.clamp((z - height_low) / max(height_high - height_low, 1e-6), 0.0, 1.0)
    smooth = t * t * (3.0 - 2.0 * t)
    return upright * smooth


def phase_pose_blend(phase: torch.Tensor, descent_end: float, hold_end: float, rise_end: float) -> torch.Tensor:
    # [0, descent_end)       : 0 -> 1  (go down)
    # [descent_end, hold_end): 1       (low)
    # [hold_end, rise_end)   : 1 -> 0  (stand up)
    # [rise_end, 1.0)        : 0       (high / rest)
    b = torch.zeros_like(phase)
    descend = phase < descent_end
    b = torch.where(descend, phase / descent_end, b)
    low = (phase >= descent_end) & (phase < hold_end)
    b = torch.where(low, torch.ones_like(phase), b)
    rise = (phase >= hold_end) & (phase < rise_end)
    b = torch.where(rise, 1.0 - (phase - hold_end) / (rise_end - hold_end), b)
    return b


def _phase_pose_error(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, command_name: str, target_pose: dict, descent_end: float, hold_end: float, rise_end: float, source_pose: dict | None = None):
    assert target_pose, "_phase_pose_error: target_pose is empty"

    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    phase = (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * torch.pi)) % 1.0
    blend = phase_pose_blend(phase, descent_end, hold_end, rise_end)

    names = list(target_pose.keys())
    ids = [int(asset.find_joints([n])[0][0]) for n in names]
    default = asset.data.default_joint_pos[:, ids]

    source = default.clone()
    if source_pose:
        for j, n in enumerate(names):
            if n in source_pose:
                source[:, j] = source_pose[n]
    target_vec = torch.tensor([target_pose[n] for n in names], device=env.device, dtype=default.dtype).unsqueeze(0)

    target = source + blend.unsqueeze(-1) * (target_vec - source)
    cur = asset.data.joint_pos[:, ids]
    return cur, target


def phase_pose_track(env: ManagerBasedRlEnv, command_name: str = "twist", target_pose: dict | None = None, source_pose: dict | None = None, std: float = 0.3, descent_end: float = 0.15, hold_end: float = 0.50, rise_end: float = 0.65, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    cur, target = _phase_pose_error(env, asset_cfg, command_name, target_pose or {}, descent_end, hold_end, rise_end, source_pose)
    return torch.exp(-(((cur - target) / std) ** 2)).mean(dim=-1)


def phase_pose_track_l1(env: ManagerBasedRlEnv, command_name: str = "twist", target_pose: dict | None = None, source_pose: dict | None = None, descent_end: float = 0.15, hold_end: float = 0.50, rise_end: float = 0.65, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    cur, target = _phase_pose_error(env, asset_cfg, command_name, target_pose or {}, descent_end, hold_end, rise_end, source_pose)
    return -(cur - target).abs().mean(dim=-1)


def phase_rise_gate(phase: torch.Tensor, hold_end: float, rise_end: float) -> torch.Tensor:
    g = torch.zeros_like(phase)
    rising = (phase >= hold_end) & (phase < rise_end)
    g = torch.where(rising, (phase - hold_end) / (rise_end - hold_end), g)
    g = torch.where(phase >= rise_end, torch.ones_like(phase), g)
    return g


def _gp_phase(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)
    return (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * torch.pi)) % 1.0


def mouth_ground_proximity_phased(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=["mouth_tip"]), std: float = 0.10, target_height: float = 0.0, command_name: str = "twist", descent_end: float = 0.25, hold_end: float = 0.35, rise_end: float = 0.60) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    mouth_z = asset.data.site_pos_w[:, asset_cfg.site_ids[0], 2]
    proximity = torch.exp(-(((mouth_z - target_height) / std) ** 2))
    gate = phase_pose_blend(_gp_phase(env, command_name), descent_end, hold_end, rise_end)
    return gate * proximity


def mouth_perpendicular_phased(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=["mouth_tip"]), command_name: str = "twist", descent_end: float = 0.25, hold_end: float = 0.35, rise_end: float = 0.60) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    q = asset.data.site_quat_w[:, asset_cfg.site_ids[0], :]
    w, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    x_axis_z = 2.0 * (qx * qz - w * qy)
    alignment = -x_axis_z  # 1 = mouth points straight down
    gate = phase_pose_blend(_gp_phase(env, command_name), descent_end, hold_end, rise_end)
    return gate * alignment


def ground_pick_return_pose_phased(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, std: float = 0.3, command_name: str = "twist", joint_indices: list | None = None, hold_end: float = 0.35, rise_end: float = 0.60) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    joint_pos = _servo_joint_pos(env, asset)
    default_pos = _servo_default_joint_pos(env, asset)
    if joint_indices is not None:
        joint_pos = joint_pos[:, joint_indices]
        default_pos = default_pos[:, joint_indices]
    pose_reward = torch.exp(-(((joint_pos - default_pos) / std) ** 2)).mean(dim=-1)
    gate = phase_rise_gate(_gp_phase(env, command_name), hold_end, rise_end)
    return gate * pose_reward


def ground_pick_return_upright_phased(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, std: float = 0.4, command_name: str = "twist", hold_end: float = 0.35, rise_end: float = 0.60) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w
    tilt_sq = 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    upright = torch.exp(-tilt_sq / (std * std))
    gate = phase_rise_gate(_gp_phase(env, command_name), hold_end, rise_end)
    return gate * upright


def neck_vel_descent_penalty(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, command_name: str = "twist", joint_indices: list | None = None, hold_end: float = 0.35) -> torch.Tensor:
    # Cost = mean(joint_vel²) over the given joints, gated to 1 for phase < hold_end
    # (descent + low hold) and 0 afterwards (rise + rest) -> does NOT hinder
    # raising the neck. Returns a positive cost; to be used with a negative weight.
    asset = env.scene[asset_cfg.name]
    vel = _servo_joint_vel(env, asset)
    if joint_indices is not None:
        vel = vel[:, joint_indices]
    cost = (vel**2).mean(dim=-1)
    phase = _gp_phase(env, command_name)
    gate = (phase < hold_end).to(vel.dtype)  # descent + low hold only
    return gate * cost


def sample_mouth_payload(env: ManagerBasedRlEnv, env_ids: torch.Tensor, min_kg: float = 0.01, max_kg: float = 0.04) -> None:
    buf = getattr(env, "_mouth_payload_kg", None)
    if buf is None:
        buf = torch.zeros(env.num_envs, device=env.device)
        env._mouth_payload_kg = buf
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    buf[env_ids] = torch.rand(len(env_ids), device=env.device) * (max_kg - min_kg) + min_kg


def apply_mouth_payload_force(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["jaw_soft"], site_names=["mouth_tip"]), command_name: str = "twist", hold_end: float = 0.35, ramp: float = 0.05, gravity: float = 9.81) -> torch.Tensor:
    # Emulates a point mass at the tip of the mouth while standing up: the force
    # m·g is applied at the body CoM + the torque (p_mouth - p_com) × F, which
    # is equivalent to applying it at mouth_tip (proper lever arm for the neck). Returns
    asset: Entity = env.scene[asset_cfg.name]
    payload = getattr(env, "_mouth_payload_kg", None)
    if payload is None:
        return torch.zeros(env.num_envs, device=env.device)
    phase = _gp_phase(env, command_name)
    gate = ((phase - hold_end) / ramp).clamp(0.0, 1.0)  # 0 before grab -> 1 after
    fz = -(gate * payload) * gravity  # (N,) vertical force (down)

    bid = int(asset_cfg.body_ids[0])
    sid = int(asset_cfg.site_ids[0])
    p_mouth = asset.data.site_pos_w[:, sid, :]
    p_com = asset.data.body_com_pos_w[:, bid, :]
    F = torch.zeros((env.num_envs, 3), device=env.device, dtype=p_mouth.dtype)
    F[:, 2] = fz
    tau = torch.cross(p_mouth - p_com, F, dim=-1)  # applies F at mouth_tip
    asset.write_external_wrench_to_sim(forces=F.unsqueeze(1), torques=tau.unsqueeze(1), body_ids=[bid])
    return torch.zeros(env.num_envs, device=env.device)


def randomize_delayed_actuator_gains(env: ManagerBasedRlEnv, env_ids: torch.Tensor, kp_range: tuple[float, float], kd_range: tuple[float, float], asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, operation: str = "scale"):
    # Under the canonical BAM actuator (``bam.mjlab.BamActuator``) gains are scaled
    # per-env via ``set_gains``/``reset_gains`` (the actuator owns ``kp_scale``/
    # ``kd_scale``), so we never touch the MuJoCo model — no accumulation risk. The
    # sampled per-joint factors are averaged into a single scalar per env (the
    # actuator applies one scale across its joints), matching the previous behavior.
    # Non-BAM actuators are skipped (e.g. the roller XmlActuator, which doesn't
    # expose set_gains).
    del operation
    from bam.mjlab import BamActuator

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]

    for actuator in asset.actuators:
        if not isinstance(actuator, BamActuator):
            continue
        n_joints = len(actuator.ctrl_ids)
        kp_samples = torch.rand(len(env_ids), n_joints, device=env.device) * (kp_range[1] - kp_range[0]) + kp_range[0]
        kd_samples = torch.rand(len(env_ids), n_joints, device=env.device) * (kd_range[1] - kd_range[0]) + kd_range[0]
        # Restore nominal first (prevents accumulation), then apply fresh scale.
        actuator.reset_gains(env_ids)
        actuator.set_gains(env_ids, kp_scale=kp_samples.mean(dim=1, keepdim=True), kd_scale=kd_samples.mean(dim=1, keepdim=True))


@requires_model_fields("dof_frictionloss", "dof_damping")
def expand_bam_friction_fields(env: ManagerBasedRlEnv, env_ids: torch.Tensor):
    # bam's BamActuator (mjlab_frictionloss branch) writes a per-env friction
    # budget into MuJoCo's dof_frictionloss/dof_damping every step, which
    # requires those model fields to be expanded per world. mjlab expands
    # exactly the fields declared by event functions via requires_model_fields,
    # so every env using the BAM actuator must register this event.
    pass


def randomize_bam_friction(env: ManagerBasedRlEnv, env_ids: torch.Tensor, scale_range: tuple[float, float], asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG):
    # Under BAM, MuJoCo's dof_frictionloss is zeroed (BAM computes friction in
    # compute()), so stock dr.dof_frictionloss is a no-op. Instead this samples a
    # per-env scalar in ``scale_range`` and applies it to the FrictionDRBamActuator's
    # ``friction_scale``, which multiplies BAM's velocity-independent friction budget
    # (Coulomb + Stribeck + load). Restores nominal (1.0) first to avoid accumulation.
    # No-op on actuators without a friction_scale hook.
    from .robot_actuator import FrictionDRBamActuator

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]
    lo, hi = scale_range
    for actuator in asset.actuators:
        if isinstance(actuator, FrictionDRBamActuator):
            actuator.reset_friction_scale(env_ids)
            samples = torch.rand(len(env_ids), 1, device=env.device) * (hi - lo) + lo
            actuator.set_friction_scale(env_ids, samples)


def standing_envs_curriculum(env: ManagerBasedRlEnv, env_ids: torch.Tensor, command_name: str, standing_stages: list[dict]) -> torch.Tensor:
    del env_ids

    from typing import cast

    from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

    command_term = env.command_manager.get_term(command_name)
    assert command_term is not None, f"Command term '{command_name}' not found"

    cfg = cast(UniformVelocityCommandCfg, command_term.cfg)

    for stage in standing_stages:
        if env.common_step_counter > stage["step"]:
            cfg.rel_standing_envs = stage["rel_standing_envs"]

    return torch.tensor([cfg.rel_standing_envs])


def push_curriculum(env: ManagerBasedRlEnv, env_ids: torch.Tensor, event_name: str, push_stages: list[dict]) -> torch.Tensor:
    del env_ids

    # NOTE: must update the live EventManager term_cfg, not env.cfg.events —
    # EventManager.__init__ does deepcopy(cfg), so mutating env.cfg.events is a no-op.
    event_cfg = env.event_manager.get_term_cfg(event_name)

    current_range = push_stages[0]["velocity_range"]

    for stage in push_stages:
        if env.common_step_counter > stage["step"]:
            current_range = stage["velocity_range"]

    event_cfg.params["velocity_range"] = current_range

    max_push = max(abs(current_range["x"][0]), abs(current_range["x"][1]))
    return torch.tensor([max_push])


def wheel_friction_curriculum(env: ManagerBasedRlEnv, env_ids: torch.Tensor, event_name: str, ranges_stages: list[dict]) -> torch.Tensor:
    del env_ids

    current_ranges = ranges_stages[0]["ranges"]
    for stage in ranges_stages:
        if env.common_step_counter > stage["step"]:
            current_ranges = stage["ranges"]

    env.event_manager.get_term_cfg(event_name).params["ranges"] = current_ranges
    return torch.tensor([current_ranges[0]])


def reward_weight(env: ManagerBasedRlEnv, env_ids: torch.Tensor, reward_name: str, weight_stages: list[dict]) -> torch.Tensor:
    del env_ids
    term_cfg = env.reward_manager.get_term_cfg(reward_name)
    for stage in weight_stages:
        if env.common_step_counter > stage["step"]:
            term_cfg.weight = stage["weight"]
    return torch.tensor([term_cfg.weight])


def com_range_curriculum(env: ManagerBasedRlEnv, env_ids: torch.Tensor, event_name: str, range_stages: list[dict]) -> torch.Tensor:
    del env_ids

    # NOTE: must update the live EventManager term_cfg, not env.cfg.events —
    # EventManager.__init__ does deepcopy(cfg), so mutating env.cfg.events is a no-op.
    event_cfg = env.event_manager.get_term_cfg(event_name)

    current_range = range_stages[0]["range"]
    for stage in range_stages:
        if env.common_step_counter > stage["step"]:
            current_range = stage["range"]

    event_cfg.params["ranges"] = (-current_range, current_range)
    return torch.tensor([current_range])


def slope_move_masks(distance: "torch.Tensor", size_x: float):
    # move_up   : traveled more than 40% of the tile → it went down the ramp,
    #             make it steeper. Aligned with the terrain_edge_reached
    #             termination (~3.8 m, threshold_fraction=0.95 by
    #             default on size_x=8.0), which ends the episode before the
    #             half threshold (4.0 m) — without this alignment a successful
    #             traverser is never promoted.
    # move_down : barely advanced (< 20% of the tile) → early fall/stall,
    #             make the ramp gentler.
    move_up = distance > size_x * 0.4
    move_down = (distance < size_x * 0.2) & (~move_up)
    return move_up, move_down


def terrain_levels_slope(env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> torch.Tensor:
    # Progression based on the x distance traveled from the spawn origin.
    asset = env.scene["robot"]
    terrain = env.scene.terrain
    assert terrain is not None
    terrain_generator = terrain.cfg.terrain_generator
    assert terrain_generator is not None

    distance = asset.data.root_link_pos_w[env_ids, 0] - env.scene.env_origins[env_ids, 0]
    move_up, move_down = slope_move_masks(distance, terrain_generator.size[0])
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def _imu_misalignment_quat(env: ManagerBasedRlEnv, max_angle_rad: float) -> torch.Tensor:
    # (like a startup randomization), so it's a *systematic per-robot bias*, not
    # per-step noise. Replaces the old randomize_imu_orientation event, which wrote
    # site_quat (not per-env expanded under mjlab 1.3.0, and not read by the
    # projected_gravity / base_ang_vel observations anyway).
    #
    # Returns a (num_envs, 4) unit quaternion (w, x, y, z).
    q = getattr(env, "_imu_misalign_quat", None)
    if q is None:
        n = env.num_envs
        axis = torch.randn(n, 3, device=env.device)
        axis = axis / (torch.norm(axis, dim=-1, keepdim=True) + 1e-8)
        angle = torch.rand(n, device=env.device) * max_angle_rad  # [0, max]
        q = quat_from_angle_axis(angle, axis)
        env._imu_misalign_quat = q
    return q


def projected_gravity_imu_misaligned(env: ManagerBasedRlEnv, max_angle_deg: float = 1.0, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    q = _imu_misalignment_quat(env, math.radians(max_angle_deg))
    return quat_apply(q, asset.data.projected_gravity_b)


def base_ang_vel_imu_misaligned(env: ManagerBasedRlEnv, max_angle_deg: float = 1.0, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    q = _imu_misalignment_quat(env, math.radians(max_angle_deg))
    return quat_apply(q, asset.data.root_link_ang_vel_b)


def raw_accelerometer(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]

    # The accelerometer sensor is the 5th sensor (index 4) in robot.xml
    # Sensors: framequat, gyro, gyro, velocimeter, accelerometer, subtreeangmom
    mj_model = asset.data.model

    sensor_adr_array = mj_model.sensor_adr
    sensor_id = 4  # imu_accel is the 5th sensor (0-indexed)
    sensor_adr = int(sensor_adr_array[sensor_id].item())

    accel_raw = asset.data.data.sensordata[:, sensor_adr : sensor_adr + 3]

    # MuJoCo accelerometer measures specific force (like real sensor)
    # Negate to match convention: when at rest upright, should point down
    accel_negated = -accel_raw

    accel_norm = torch.norm(accel_negated, dim=-1, keepdim=True)
    accel_normalized = torch.where(
        accel_norm > 0.1,
        accel_negated / accel_norm,
        asset.data.projected_gravity_b,  # Fallback to projected gravity
    )

    return accel_normalized


def randomize_base_orientation(env: ManagerBasedRlEnv, env_ids: torch.Tensor, max_pitch_deg: float = 10.0, max_roll_deg: float = 5.0, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)

    num_envs = len(env_ids)

    max_pitch_rad = max_pitch_deg * torch.pi / 180.0
    max_roll_rad = max_roll_deg * torch.pi / 180.0

    pitch = (torch.rand(num_envs, device=env.device) * 2 - 1) * max_pitch_rad
    roll = (torch.rand(num_envs, device=env.device) * 2 - 1) * max_roll_rad
    yaw = torch.zeros(num_envs, device=env.device)

    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)

    quat_w = cr * cp * cy + sr * sp * sy
    quat_x = sr * cp * cy - cr * sp * sy
    quat_y = cr * sp * cy + sr * cp * sy
    quat_z = cr * cp * sy - sr * sp * cy

    new_quat = torch.stack([quat_w, quat_x, quat_y, quat_z], dim=1)

    new_quat = new_quat / torch.norm(new_quat, dim=1, keepdim=True)

    root_quat_idx = 3

    env.sim.data.qpos[env_ids, root_quat_idx : root_quat_idx + 4] = new_quat


def set_random_prone_orientation(env: ManagerBasedRlEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, face_down_prob: float = 0.5):
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.int)
    num = len(env_ids)

    yaw = torch.rand(num, device=env.device) * 2 * np.pi - np.pi
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    s = 2.0**-0.5  # sqrt(2)/2

    face_down = torch.stack([s * cy, -s * sy, s * cy, s * sy], dim=1)
    face_up = torch.stack([s * cy, s * sy, -s * cy, s * sy], dim=1)

    mask = torch.rand(num, device=env.device) < face_down_prob  # True → face-down
    new_quat = torch.where(mask.unsqueeze(1), face_down, face_up)

    env.sim.data.qpos[env_ids, 3:7] = new_quat
    env.sim.data.qvel[env_ids, :6] = 0.0


def set_random_ground_state(env: ManagerBasedRlEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, face_down_prob: float = 0.4, face_up_prob: float = 0.4, sitting_prob: float = 0.2, standing_prob: float = 0.0, prone_z_min: float = 0.20, prone_z_max: float = 0.25, sitting_z_min: float = 0.07, sitting_z_max: float = 0.09, standing_z_min: float = 0.11, standing_z_max: float = 0.12, sitting_joint_overrides: dict | None = None, sitting_joint_noise_std: float = 0.0, sitting_tilt_max: float = 0.0, face_up_roll_max: float = 0.0):
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.int)
    num = len(env_ids)

    total = face_down_prob + face_up_prob + sitting_prob + standing_prob
    p_fd = face_down_prob / total
    p_fu = (face_down_prob + face_up_prob) / total
    p_sit = (face_down_prob + face_up_prob + sitting_prob) / total

    yaw = torch.rand(num, device=env.device) * 2 * np.pi - np.pi
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    s = 2.0**-0.5

    face_down = torch.stack([s * cy, -s * sy, s * cy, s * sy], dim=1)
    face_up = torch.stack([s * cy, s * sy, -s * cy, s * sy], dim=1)
    # Upright sitting: yaw-only by default, with optional ±sitting_tilt_max
    # pitch/roll noise so the policy doesn't overfit to perfectly-upright starts.
    if sitting_tilt_max > 0.0:
        pitch = (torch.rand(num, device=env.device) * 2 - 1) * sitting_tilt_max
        roll = (torch.rand(num, device=env.device) * 2 - 1) * sitting_tilt_max
        cp = torch.cos(pitch * 0.5)
        sp = torch.sin(pitch * 0.5)
        cr = torch.cos(roll * 0.5)
        sr = torch.sin(roll * 0.5)
        # ZYX intrinsic Euler → quaternion (yaw * pitch * roll).
        sit_w = cr * cp * cy + sr * sp * sy
        sit_x = sr * cp * cy - cr * sp * sy
        sit_y = cr * sp * cy + sr * cp * sy
        sit_z = cr * cp * sy - sr * sp * cy
        sitting = torch.stack([sit_w, sit_x, sit_y, sit_z], dim=1)
    else:
        sitting = torch.stack([cy, torch.zeros_like(cy), torch.zeros_like(cy), sy], dim=1)

    u = torch.rand(num, device=env.device)
    # is_fd (u < p_fd) is implicit: face_down is the base value of new_quat below.
    is_fu = (u >= p_fd) & (u < p_fu)
    is_sit = (u >= p_fu) & (u < p_sit)
    is_stand = u >= p_sit

    # seed-lucky): the reward landscape between supine and prone is FLAT —
    # upright_linear (cos tilt) is ≈0 through the whole roll, height doesn't
    # change — so rolling off the back only pays via the front-rise path that
    # follows, a long-horizon dependency that noisy exploration rarely finds
    # from a perfectly flat supine start. With roll noise, a fraction of
    # face-up spawns start near-on-side (partway along the roll): the policy
    # learns roll-completion from easy starts and generalizes back to flat
    # supine — a built-in reverse curriculum. Uniform sampling keeps every
    # difficulty represented (flat back |roll|<15° ≈ 17% at ±90°), so no
    # annealing schedule is needed, and varied post-fall poses are realistic
    # DR for deployment anyway.
    if face_up_roll_max > 0.0:
        theta = (torch.rand(num, device=env.device) * 2 - 1) * face_up_roll_max
        ct = torch.cos(theta * 0.5)
        st = torch.sin(theta * 0.5)
        # Log-roll = rotation about the body's LONG axis, which is body z (the
        # spine: trunk z is up when standing → horizontal when lying). NOT body
        # x — supine leaves body x pointing skyward, so an x-roll would only
        # spin the robot in place like the yaw noise already does.
        # Body-frame rotation → right-multiply: q_fu ⊗ [ct, 0, 0, st].
        w, x, y, z = face_up[:, 0], face_up[:, 1], face_up[:, 2], face_up[:, 3]
        face_up = torch.stack([w * ct - z * st, x * ct + y * st, y * ct - x * st, w * st + z * ct], dim=1)

    new_quat = face_down.clone()
    new_quat[is_fu] = face_up[is_fu]
    new_quat[is_sit] = sitting[is_sit]
    new_quat[is_stand] = sitting[is_stand]

    z_prone = torch.rand(num, device=env.device) * (prone_z_max - prone_z_min) + prone_z_min
    z_sit = torch.rand(num, device=env.device) * (sitting_z_max - sitting_z_min) + sitting_z_min
    z_stand = torch.rand(num, device=env.device) * (standing_z_max - standing_z_min) + standing_z_min
    new_z = z_prone.clone()
    new_z = torch.where(is_sit, z_sit, new_z)
    new_z = torch.where(is_stand, z_stand, new_z)

    env.sim.data.qpos[env_ids, 2] = new_z
    env.sim.data.qpos[env_ids, 3:7] = new_quat
    env.sim.data.qvel[env_ids, :6] = 0.0

    # Sitting-bucket joint overrides (e.g. knee/ankle bent to keyframe).
    # Override keys are SERVO indices (14-joint layout); translate to entity
    # joint indices so models with interleaved passive_* joints (backlash)
    # write the intended joints. qpos column = 7 + entity joint index
    # (robot free joint first, all hinges 1-dof).
    asset: Entity = env.scene[asset_cfg.name]
    servo_ids = _servo_joint_ids(env, asset)
    if sitting_joint_overrides:
        sit_env_ids = env_ids[is_sit]
        if len(sit_env_ids) > 0:
            for jnt_idx, angle in sitting_joint_overrides.items():
                env.sim.data.qpos[sit_env_ids, 7 + servo_ids[jnt_idx]] = angle

    # Joint noise for sitting envs: Gaussian noise on every actuated joint
    # so the policy sees a distribution of plausible "sit" starts rather than
    # a single canonical pose. Captures real-world transfer where the robot's
    # joint angles won't match the SIT keyframe exactly when the standup
    # policy takes over from the sit policy.
    if sitting_joint_noise_std > 0.0:
        sit_env_ids = env_ids[is_sit]
        if len(sit_env_ids) > 0:
            # Servo joints only: passive_* joints (backlash hinges) have tiny
            # ranges and must stay at 0 on reset.
            n_sit = len(sit_env_ids)
            cols = torch.tensor([7 + j for j in servo_ids], device=env.device, dtype=torch.long)
            noise = torch.randn(n_sit, len(cols), device=env.device) * sitting_joint_noise_std
            env.sim.data.qpos[sit_env_ids.unsqueeze(1).long(), cols.unsqueeze(0)] += noise


# Deep-crouch anchor pose (velstand run-5): the "stuck" mid-recovery basin —
# knees folded under the body, trunk pitched forward, feet flat. Values chosen
# by extending the HOME zig-zag (hip fwd / knee back / ankle fwd, sign
# conventions per the SIT keyframe fold directions) to deep flexion, inside
# the ±1.57 joint limits. hip_yaw/hip_roll/neck stay at HOME.
_CROUCH_ANCHOR_BY_NAME = {"left_hip_pitch": -1.15, "left_knee": 1.25, "left_ankle": 1.05, "right_hip_pitch": 1.15, "right_knee": -1.25, "right_ankle": -1.05}


def set_random_crouch_state(env: ManagerBasedRlEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, depth_min: float = 0.35, depth_max: float = 1.0, pitch_max_deg: float = 55.0, joint_noise: float = 0.12, z_stand: float = 0.115, z_deep: float = 0.06):
    # Reverse curriculum for the recovery last mile (velstand run-5 lesson):
    # prone-init episodes spend most of their fallen budget getting TO the deep
    # crouch and are recycled shortly after reaching it, so the crouch→stand
    # mile gets almost no on-policy data — the policy converged to parking
    # there. Seeding resets ACROSS that mile (depth λ ∈ [depth_min, depth_max]
    # between standing and the deep-crouch anchor, trunk pitch and z scaled
    # with λ) makes the frontier dense from step 0 of the episode.
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.long)
    num = len(env_ids)
    asset: Entity = env.scene[asset_cfg.name]

    lam = torch.rand(num, device=env.device) * (depth_max - depth_min) + depth_min

    # Joints: lerp HOME → anchor on the leg pitch chain, uniform noise on the
    # servo joints only (passive_* backlash hinges have ±1° ranges — noise
    # there would spawn them pinned outside their limits).
    joints = asset.data.default_joint_pos[env_ids].clone()
    for name, anchor in _CROUCH_ANCHOR_BY_NAME.items():
        ids, _ = asset.find_joints(f"^{name}$")
        j = ids[0]
        joints[:, j] = joints[:, j] + lam * (anchor - joints[:, j])
    noise_mask = torch.zeros(joints.shape[1], device=joints.device)
    noise_mask[_servo_joint_ids(env, asset)] = 1.0
    joints += (torch.rand_like(joints) * 2 - 1) * joint_noise * noise_mask

    # Base orientation: forward pitch scaled with depth (the stuck basin is a
    # forward crouch from both fall directions), random yaw, small roll noise.
    pitch = lam * math.radians(pitch_max_deg) + (torch.rand(num, device=env.device) * 2 - 1) * math.radians(10.0)
    pitch = torch.clamp(pitch, min=math.radians(5.0))
    roll = (torch.rand(num, device=env.device) * 2 - 1) * math.radians(8.0)
    yaw = torch.rand(num, device=env.device) * 2 * np.pi - np.pi
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    # ZYX intrinsic Euler → quaternion (yaw * pitch * roll), as in
    # set_random_ground_state's sitting branch.
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    quat = torch.stack([qw, qx, qy, qz], dim=1)

    # Trunk height scaled with depth, small upward margin to settle cleanly.
    z = z_stand + lam * (z_deep - z_stand) + torch.rand(num, device=env.device) * 0.01

    env.sim.data.qpos[env_ids, 2] = z
    env.sim.data.qpos[env_ids, 3:7] = quat
    env.sim.data.qpos[env_ids, 7:] = joints
    env.sim.data.qvel[env_ids, :] = 0.0


def maybe_set_random_prone_orientation(env: ManagerBasedRlEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, prone_prob: float = 0.0, face_down_prob: float = 0.5, prone_z_min: float = 0.20, prone_z_max: float = 0.25, crouch_prob: float = 0.0):
    if prone_prob <= 0.0 and crouch_prob <= 0.0:
        return
    # env_ids=None means "all envs" (the initial global reset passes None —
    # the old early-return silently skipped prone init there).
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if len(env_ids) == 0:
        return
    env_ids_t = env_ids.to(env.device, dtype=torch.long) if isinstance(env_ids, torch.Tensor) else torch.tensor(env_ids, device=env.device, dtype=torch.long)
    u = torch.rand(len(env_ids_t), device=env.device)
    selected = env_ids_t[u < prone_prob]
    crouch_selected = env_ids_t[(u >= prone_prob) & (u < prone_prob + crouch_prob)]
    if len(selected) > 0:
        set_random_prone_orientation(env, selected, asset_cfg=asset_cfg, face_down_prob=face_down_prob)
        # Override z so the prone body has head/neck clearance when settling.
        z = torch.rand(len(selected), device=env.device) * (prone_z_max - prone_z_min) + prone_z_min
        env.sim.data.qpos[selected, 2] = z
    if len(crouch_selected) > 0:
        set_random_crouch_state(env, crouch_selected, asset_cfg=asset_cfg)


def event_param_curriculum(env: ManagerBasedRlEnv, env_ids: torch.Tensor, event_name: str, param_stages: list[dict]) -> torch.Tensor:
    del env_ids
    event_cfg = env.event_manager.get_term_cfg(event_name)
    current = param_stages[0]["params"]
    for stage in param_stages:
        if env.common_step_counter >= stage["step"]:
            current = stage["params"]
    event_cfg.params.update(current)
    first_val = next(iter(current.values()))
    return torch.tensor(float(first_val) if isinstance(first_val, (int, float)) else 0.0)


class VelocityCommandCommandOnly(UniformVelocityCommand):
    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        # Turn-in-place practice: for a fraction of envs, zero the linear velocity
        # and force a meaningful (away-from-zero) yaw command. Independent uniform
        # sampling almost never produces "lin≈0, |ang| large" (~2% of samples), so
        # spinning on the spot was effectively untrained → slow/unstable real-robot
        # turning. Mirrors the base rel_forward_envs mechanism.
        p = getattr(self.cfg, "rel_turn_in_place_envs", 0.0)
        if p <= 0.0:
            return
        r = torch.empty(len(env_ids), device=self.device)
        turn_ids = env_ids[r.uniform_(0.0, 1.0) < p]
        if len(turn_ids) == 0:
            return
        self.vel_command_b[turn_ids, 0] = 0.0
        self.vel_command_b[turn_ids, 1] = 0.0
        lo, hi = self.cfg.ranges.ang_vel_z
        maxr = max(abs(lo), abs(hi))
        rr = torch.empty(len(turn_ids), device=self.device)
        sign = torch.where(rr.uniform_(0.0, 1.0) < 0.5, -1.0, 1.0)
        mag = torch.empty(len(turn_ids), device=self.device).uniform_(0.4 * maxr, maxr)
        self.vel_command_b[turn_ids, 2] = sign * mag
        # These envs must actually turn — un-mark them as standing (which would
        # zero the command) and refresh the world-frame reference copy.
        self.is_standing_env[turn_ids] = False
        self.vel_command_w[turn_ids] = self.vel_command_b[turn_ids]

    def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
        batch = visualizer.env_idx
        if batch >= self.num_envs:
            return

        cmds = self.command.cpu().numpy()
        base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
        base_quat_w = self.robot.data.root_link_quat_w
        base_mat_ws = matrix_from_quat(base_quat_w).cpu().numpy()

        base_pos_w = base_pos_ws[batch]
        base_mat_w = base_mat_ws[batch]
        cmd = cmds[batch]

        if np.linalg.norm(base_pos_w) < 1e-6:
            return

        def local_to_world(vec: np.ndarray) -> np.ndarray:
            return base_pos_w + base_mat_w @ vec

        scale = self.cfg.viz.scale * 2.0
        z_offset = self.cfg.viz.z_offset

        cmd_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
        cmd_lin_to = local_to_world((np.array([0, 0, z_offset]) + np.array([cmd[0], cmd[1], 0])) * scale)
        visualizer.add_arrow(cmd_lin_from, cmd_lin_to, color=(0.2, 0.2, 0.6, 0.6), width=0.015)


@_dataclass(kw_only=True)
class VelocityCommandCommandOnlyCfg(UniformVelocityCommandCfg):
    # Fraction of envs commanded to turn in place (lin=0, |ang| forced to
    # [0.4·max, max]) each resample. 0 = disabled (base uniform sampling only).
    rel_turn_in_place_envs: float = 0.0

    def build(self, env: ManagerBasedRlEnv) -> "VelocityCommandCommandOnly":
        return VelocityCommandCommandOnly(self, env)


class RelativeHeadingVelocityCommand(VelocityCommandCommandOnly):
    # cmd[0] = lin_vel_x  (throttle: 0=coast, +push, -brake)
    # cmd[1] = lin_vel_y  (unused, 0)
    # cmd[2] = heading_error  (+ = target is to the right/CW, - = to the left/CCW)
    #          0 → go straight, ±max = target is max_angle rad to the right/left
    # Set heading_command=False and rel_heading_envs=0.0 in the cfg (we handle
    # heading internally).  ang_vel_z range in cfg is used as the clip limit for cmd[2].

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._target_heading_w = torch.zeros(self.num_envs, device=self.device)
        ang_rng = cfg.ranges.ang_vel_z
        self._heading_max = float(ang_rng[1]) if ang_rng else 1.0

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        n = len(env_ids)
        self._target_heading_w[env_ids] = torch.rand(n, device=self.device) * 2.0 * math.pi - math.pi
        self.vel_command_b[env_ids, 2] = 0.0

    def _update_command(self) -> None:
        # Do NOT call super()._update_command() — it would run the heading
        # proportional controller and overwrite cmd[2] with a yaw rate.
        # Instead recompute heading error from scratch each step.
        quat = self.robot.data.root_link_quat_w
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        current_yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        delta = self._target_heading_w - current_yaw
        heading_error = torch.atan2(torch.sin(delta), torch.cos(delta))
        self.vel_command_b[:, 2] = heading_error.clamp(-self._heading_max, self._heading_max)

    def _update_metrics(self) -> None:
        pass


class RelativeHeadingVelocityCommandCfg(UniformVelocityCommandCfg):
    def build(self, env: ManagerBasedRlEnv) -> "RelativeHeadingVelocityCommand":
        return RelativeHeadingVelocityCommand(self, env)


def heading_tracking_reward(env: ManagerBasedRlEnv, command_name: str, std: float = 0.5) -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)
    heading_error = cmd[:, 2]
    return torch.exp(-(heading_error**2) / (std**2))


def skating_air_time_reward(env: ManagerBasedRlEnv, sensor_name: str, command_name: str, threshold_min: float = 0.05, threshold_max: float = 0.4, vel_gate_ref: float = 0.0) -> torch.Tensor:
    from mjlab.sensor import ContactSensor

    sensor: ContactSensor = env.scene[sensor_name]
    current_air_time = sensor.data.current_air_time
    assert current_air_time is not None

    in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
    reward = torch.sum(in_range.float(), dim=1)

    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    reward = reward * torch.clamp(cmd_x, min=0.0)
    gate = _forward_progress_gate(env, vel_gate_ref)
    if gate is not None:
        reward = reward * gate
    return reward


def _forward_progress_gate(env: ManagerBasedRlEnv, v_ref: float) -> torch.Tensor | None:
    if v_ref <= 0.0:
        return None
    v_fwd = env.scene["robot"].data.root_link_lin_vel_b[:, 0]
    return (v_fwd.clamp(min=0.0) / v_ref).clamp(max=1.0)


def single_support_reward(env: ManagerBasedRlEnv, sensor_name: str, command_name: str, vel_gate_ref: float = 0.0, double_penalty: float = 0.25) -> torch.Tensor:
    # Per step, counting blades in contact:
    #   - exactly 1 blade down (stride)  → + clamp(cmd_x,0) · gate
    #   - 2 blades down    (double supp) → − double_penalty · clamp(cmd_x,0)
    #   - 0 blades down    (flight/hop)  →  0
    #
    # The POSITIVE single-support reward is gated by forward speed (``vel_gate_ref``)
    # so stepping in place (no propulsion) earns nothing — kills the tap-dance hack.
    # The double-support penalty is small and UNGATED: brief double support during
    # weight transfer / push-off is NORMAL skating, so we only lightly discourage
    # PERMANENT double support (the swizzle) rather than forbid it. The real
    # anti-swizzle signal is skating_air_time — the swizzle never lifts a foot.
    from mjlab.sensor import ContactSensor

    sensor: ContactSensor = env.scene[sensor_name]
    contact_time = sensor.data.current_contact_time  # (num_envs, num_feet)
    assert contact_time is not None

    n_contact = torch.sum((contact_time > 0.0).float(), dim=1)  # (num_envs,)
    single = (n_contact == 1).float()
    double = (n_contact >= 2).float()

    cmd_x = torch.clamp(env.command_manager.get_command(command_name)[:, 0], min=0.0)
    single_r = single * cmd_x
    gate = _forward_progress_gate(env, vel_gate_ref)
    if gate is not None:
        single_r = single_r * gate
    return single_r - double_penalty * double * cmd_x


def glide_reward(env: ManagerBasedRlEnv, sensor_name: str, command_name: str, vel_ref: float = 0.2, stillness_std: float = 5.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=(r".*(hip|knee|ankle).*",))) -> torch.Tensor:
    #
    # - single_support: exactly ONE blade in contact. REQUIRED — this is the fix vs
    #   the earlier broken glide, which omitted it and let a two-blade swizzle-coast
    #   farm the reward and regress the gait.
    # - forward_gate = clamp(v_fwd,0,vel_ref)/vel_ref → 0 when not moving forward.
    # - stillness = exp(-Σ leg_joint_vel² / stillness_std²) → high only when legs
    #   are quiet; a kick (fast joint motion) gets ~0, so only a real glide pays.
    # - active on push/coast only (cmd_x >= 0); silent on brake.
    from mjlab.sensor import ContactSensor

    sensor: ContactSensor = env.scene[sensor_name]
    contact_time = sensor.data.current_contact_time  # (num_envs, num_feet)
    assert contact_time is not None
    single = (torch.sum((contact_time > 0.0).float(), dim=1) == 1).float()

    forward_gate = _forward_progress_gate(env, vel_ref)
    if forward_gate is None:
        forward_gate = torch.ones(env.num_envs, device=env.device)

    asset: Entity = env.scene[asset_cfg.name]
    joint_vel_sq = torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
    stillness = torch.exp(-joint_vel_sq / stillness_std**2)

    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    active = (cmd_x >= 0.0).float()
    return single * forward_gate * stillness * active


def leg_symmetry_reward(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, joint_bases: tuple = ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")) -> torch.Tensor:
    # The robot uses mirrored L/R sign conventions, so a bilaterally-symmetric config
    # satisfies q_left + q_right ≈ 0 per matched joint pair. Returns
    # ``-mean_pairs |q_left + q_right|`` (L1, constant gradient); use with a POSITIVE
    # weight so asymmetry is penalised and the symmetric swizzle is favoured. L/R index
    # pairs are resolved once by name and cached on env.
    asset: Entity = env.scene[asset_cfg.name]
    if not hasattr(env, "_leg_sym_ids"):
        left, right = [], []
        for base in joint_bases:
            li, _ = asset.find_joints([f"left_{base}"])
            ri, _ = asset.find_joints([f"right_{base}"])
            left.append(li[0])
            right.append(ri[0])
        env._leg_sym_ids = (torch.tensor(left, device=env.device), torch.tensor(right, device=env.device))
    lids, rids = env._leg_sym_ids
    q = asset.data.joint_pos
    return -torch.abs(q[:, lids] + q[:, rids]).mean(dim=-1)


def grounded_reward(env: ManagerBasedRlEnv, sensor_name: str, command_name: str) -> torch.Tensor:
    # Mirror of single_support_reward but rewarding double support (n_contact >= 2),
    # scaled by |cmd_x| so it shapes the push phase in EITHER direction (forward or
    # backward — the swizzle env drives cmd_x < 0 as "go backward").
    from mjlab.sensor import ContactSensor

    sensor: ContactSensor = env.scene[sensor_name]
    contact_time = sensor.data.current_contact_time
    assert contact_time is not None
    n_contact = torch.sum((contact_time > 0.0).float(), dim=1)
    grounded = (n_contact >= 2).float()
    cmd_x = torch.abs(env.command_manager.get_command(command_name)[:, 0])
    return grounded * cmd_x


def gait_symmetry_penalty(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    # launch). Accumulates per-foot swing time over the episode and penalises the
    # normalised imbalance |L - R| / (L + R):
    #   - balanced alternating stride  -> ~0 (no penalty)
    #   - one foot swinging much more   -> ~1 (max penalty)
    # Only the CUMULATIVE imbalance is penalised — the instantaneous single-support
    # asymmetry of a real stride (one foot swinging now) is fine.
    from mjlab.sensor import ContactSensor

    sensor: ContactSensor = env.scene[sensor_name]
    air = sensor.data.current_air_time
    assert air is not None

    if not hasattr(env, "_swing_accum") or env._swing_accum.shape[0] != env.num_envs:
        env._swing_accum = torch.zeros(env.num_envs, air.shape[1], device=env.device)
    reset = env.episode_length_buf <= 1
    env._swing_accum[reset] = 0.0
    env._swing_accum += (air > 0.0).float() * env.step_dt

    L = env._swing_accum[:, 0]
    R = env._swing_accum[:, 1]
    return torch.abs(L - R) / (L + R + 1e-3)


def heading_hold_reward(env: ManagerBasedRlEnv, std: float = 0.4, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    # The spawn heading is captured per-env on the first step(s) after reset
    # (episode_length_buf <= 1), when the robot is still ~at its spawn pose. Reads
    # root_link_quat_w, which is fresh at reward time (post physics step). Heading-
    # invariant: the reference is each env's own random spawn yaw, so it works with
    # the full-circle yaw randomisation at reset.
    asset: Entity = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    if not hasattr(env, "_heading_ref") or env._heading_ref.shape[0] != env.num_envs:
        env._heading_ref = yaw.clone()
    just_reset = env.episode_length_buf <= 1
    env._heading_ref = torch.where(just_reset, yaw, env._heading_ref)

    err = yaw - env._heading_ref
    err = torch.atan2(torch.sin(err), torch.cos(err))
    return torch.exp(-(err**2) / std**2)


def action_over_limit_penalty(env: ManagerBasedRlEnv, action_name: str = "joint_pos", overshoot: float = 0.3) -> torch.Tensor:
    # e.g. hip_roll has a ±0.38 rad limit but a ±10 rad ctrlrange, so the low-kp
    # servo can be commanded far past the stop to slam it with max torque — a
    # fragile sim-only trick that will not transfer.
    #
    # Reads the commanded target (raw_action · scale + offset) and penalises only
    # the part BEYOND (hard_limit + overshoot):
    #
    #     penalty = Σ relu(target - (hi + overshoot)) + relu((lo - overshoot) - target)
    #
    # Unlike a qpos-limit penalty, this fires on the COMMAND, not the joint
    # position — so the joint may still reach its full range (command ≈ limit) and
    # no usable amplitude is stolen. Because it constrains the policy's OUTPUT, the
    # learned behaviour is baked into the network and transfers to deployment
    # WITHOUT any env-side action clip (which would only exist in sim → mismatch).
    # ``overshoot`` gives the low-kp servo the headroom to reach near-limit targets
    # under load; only the wild over-drive past that is penalised.
    term = env.action_manager.get_term(action_name)
    target = term.raw_action * term.scale + term.offset
    jnt_ids = term.target_ids
    hard = env.scene["robot"].data.joint_pos_limits[:, jnt_ids]
    lo = hard[..., 0] - overshoot
    hi = hard[..., 1] + overshoot
    over = (target - hi).clip(min=0.0) + (lo - target).clip(min=0.0)
    return torch.sum(over, dim=-1)


def forward_lean_reward(env: ManagerBasedRlEnv, command_name: str, target_pitch: float = 0.08, std: float = 0.08, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=("trunk_base",))) -> torch.Tensor:
    # Uses projected_gravity_b x-component as a pitch proxy:
    #   forward_lean = -gravity_b[:, 0]  (positive when leaning forward)
    #
    asset: Entity = env.scene[asset_cfg.name]
    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    forward_lean = asset.data.projected_gravity_b[:, 0]
    push = torch.clamp(cmd_x, min=0.0)
    return push * torch.exp(-((forward_lean - target_pitch) ** 2) / (std**2))


class GroundPickPhaseCommand(UniformVelocityCommand):
    PERIOD: float = 4.0  # default; cfg.period overrides

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._gp_phase = torch.zeros(self.num_envs, device=self.device)
        self._period = float(getattr(cfg, "period", self.PERIOD))
        # When False, each episode starts at phase 0 (standing) instead of a
        # random phase. Matches the runtime, where the button starts the cycle
        # at phase 0 from standing. Default True keeps the historical ground_pick
        # behavior (random phase to decorrelate envs).
        self._randomize_phase = bool(getattr(cfg, "randomize_phase", True))

    @property
    def command(self) -> torch.Tensor:
        return self.vel_command_b

    def compute(self, dt: float) -> None:
        self._gp_phase = (self._gp_phase + dt / self._period) % 1.0
        self.vel_command_b[:, 0] = torch.cos(2 * torch.pi * self._gp_phase)
        self.vel_command_b[:, 1] = torch.sin(2 * torch.pi * self._gp_phase)
        self.vel_command_b[:, 2] = 0.0

    def reset(self, env_ids: torch.Tensor | None) -> dict:
        if env_ids is not None and len(env_ids) > 0:
            if self._randomize_phase:
                self._gp_phase[env_ids] = torch.rand(len(env_ids), device=self.device)
            else:
                self._gp_phase[env_ids] = 0.0
        return {}

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        pass

    def _update_command(self) -> None:
        pass

    def _update_metrics(self) -> None:
        pass


from dataclasses import dataclass as _dataclass


@_dataclass(kw_only=True)
class GroundPickPhaseCommandCfg(UniformVelocityCommandCfg):
    class_type: type = GroundPickPhaseCommand
    period: float = 4.0  # cycle length in seconds; sitstand uses 8.0
    randomize_phase: bool = True  # False -> each episode starts at phase 0 (standing)

    def build(self, env: ManagerBasedRlEnv) -> "GroundPickPhaseCommand":
        return GroundPickPhaseCommand(self, env)


# Layout, unified across all microduck policies for runtime obs compatibility:
#   command vector (13D) = [vx, vy, vtheta,           ← "twist" (velocity)
#                           neck_pitch, head_pitch,   ← "head_pose" (deltas)
#                           head_yaw, head_roll,
#                           body_x, body_y, body_z,   ← "body_pose" (deltas)
#                           body_roll, body_pitch, body_yaw]
# Total policy obs becomes 61D (51 - 3 + 13).


from dataclasses import dataclass


class UniformPoseCommand(CommandTerm):
    cfg: "UniformPoseCommandCfg"

    def __init__(self, cfg: "UniformPoseCommandCfg", env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.dim = len(cfg.ranges)
        self._command = torch.zeros(self.num_envs, self.dim, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _update_metrics(self) -> None:
        pass

    def _update_command(self) -> None:
        pass

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        if n == 0:
            return
        r = torch.empty(n, device=self.device)
        for i, (lo, hi) in enumerate(self.cfg.ranges):
            self._command[env_ids, i] = r.uniform_(lo, hi)
        # Explicit zero-command bucket. Uniform sampling essentially never
        # produces the all-zero command, so the deployment idle case ("hold the
        # nominal pose") would otherwise be absent from training (velocity
        # body-control run-1 lesson: the policy only stood still when a command
        # was present).
        if self.cfg.zero_command_prob > 0.0:
            zero_mask = torch.rand(n, device=self.device) < self.cfg.zero_command_prob
            self._command[env_ids[zero_mask]] = 0.0


@dataclass(kw_only=True)
class UniformPoseCommandCfg(CommandTermCfg):
    ranges: tuple[tuple[float, float], ...] = ()
    zero_command_prob: float = 0.0

    def build(self, env: ManagerBasedRlEnv) -> "UniformPoseCommand":
        return UniformPoseCommand(self, env)


def zero_command_padding(env: ManagerBasedRlEnv, dim: int) -> torch.Tensor:
    return torch.zeros(env.num_envs, dim, device=env.device)


def head_pose_tracking(env: ManagerBasedRlEnv, command_name: str = "head_pose", std: float = 0.5, fine_std: float | None = None, fine_weight: float = 0.5, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    # `std` is the per-joint tolerance: at err=std the per-joint reward is 1/e
    # (~0.37). Pick std on the order of the command range so the gradient
    # doesn't die as the curriculum widens.
    #
    # `fine_std` (optional) blends in a second, narrow Gaussian:
    # (1-fine_weight)·exp(-(err/std)²) + fine_weight·exp(-(err/fine_std)²).
    # Rationale: a single wide std (0.5 rad ≈ 29°) makes small errors nearly
    # free — a 10° gravity sag on the heavy head costs ~0.03 reward, so the
    # policy lets it droop. The narrow component (~0.1 rad) prices those small
    # errors while the wide one keeps gradient alive at far commands during
    # curriculum widening.
    # On backlash models the measured angle is qpos[servo] + qpos[backlash] —
    # the OUTPUT link, which is also what the encoder obs
    # (joint_pos_rel_backlash) reports. Measuring the servo alone would let the
    # head droop the backlash play reward-free AND penalize the policy for
    # compensating it (servo biased up = servo-side "error"). On models without
    # passive_*_backlash joints the mask is 0 and this reduces to the servo.
    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)

    if not hasattr(env, "_head_pose_neck_ids"):
        ids, names = asset.find_joints_by_actuator_names(_NECK_JOINT_PATTERNS)
        env._head_pose_neck_ids = torch.tensor(ids, device=env.device, dtype=torch.long)
        name_to_id = {n: i for i, n in enumerate(asset.joint_names)}
        bl = [name_to_id.get(f"passive_{n}_backlash") for n in names]
        env._head_pose_bl_ids = torch.tensor([0 if b is None else b for b in bl], device=env.device, dtype=torch.long)
        env._head_pose_bl_mask = torch.tensor([0.0 if b is None else 1.0 for b in bl], device=env.device)

    neck_ids = env._head_pose_neck_ids
    joint_pos = asset.data.joint_pos
    measured = joint_pos[:, neck_ids] + joint_pos[:, env._head_pose_bl_ids] * env._head_pose_bl_mask
    actual = measured - asset.data.default_joint_pos[:, neck_ids]
    err = actual - cmd
    per_joint = torch.exp(-((err / std) ** 2))
    if fine_std is not None:
        per_joint = (1.0 - fine_weight) * per_joint + fine_weight * torch.exp(-((err / fine_std) ** 2))
    return per_joint.mean(dim=-1)


# `robot_state_is_nan` covers joint + root state, so every obs derived from
# those is protected by the reset it triggers. The three terms below are NOT:
# they read sensor data (raycast heights, contact air-time, contact forces),
# which MuJoCo can return non-finite for while the integrated robot state is
# still clean. They are critic-only, so a single sanitized step costs the
# policy nothing, whereas letting the value through kills the entire run via
# rsl_rl's check_nan. Sanitizing here does not hide real physics blowups —
# those still terminate through nan_state and show up as
# Episode_Termination/nan_state in the training logs.


def _finite(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def foot_contact_forces_safe(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    return _finite(_velocity_obs.foot_contact_forces(env, sensor_name))


def foot_height_safe(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    return _finite(_velocity_obs.foot_height(env, sensor_name))


def foot_air_time_safe(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    return _finite(_velocity_obs.foot_air_time(env, sensor_name))


def head_pose_bias_penalty(env: ManagerBasedRlEnv, command_name: str = "head_pose", tau_s: float = 1.0, gate_height_low: float | None = None, gate_height_high: float = 0.11, gate_tilt_full_deg: float = 20.0, gate_tilt_zero_deg: float = 45.0, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    # walking unavoidably shakes a head that is 38% of the robot's mass, so an
    # instantaneous tight-tolerance term is a permanent tax on walking that no
    # policy can escape — measured at ~0.77/step against an air_time reward of
    # ~1.01/step, which is exactly what made velocity run 2026-08-20 abandon
    # stepping altogether (run 5yay13u4). The steady-state droop IS escapable:
    # the policy can bias its neck command up to cancel gravity sag. Averaging
    # over ``tau_s`` lets the oscillation cancel and prices only the bias.
    #
    # L1 (not Gaussian) on purpose: the gradient stays constant at large bias,
    # where a tight Gaussian would be flat and dead.
    #
    # velstand), same smoothstep shape and semantics as body_ang_vel_at_height —
    # zero below gate_height_low or above gate_tilt_zero_deg tilt, full above
    # gate_height_high and below gate_tilt_full_deg. The gate multiplies the
    # ERROR feeding the EMA (not just the output): while fallen/rising the EMA
    # sees zero and decays, so arriving upright starts the bias clock from ~0
    # instead of charging the whole ground phase's accumulated error at the
    # finish line — that would be a reward wall right before recovery completes,
    # the exact failure mode of the retired head_impact_penalty. The output is
    # gated too, so a fresh fall stops the charge immediately.
    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)

    if not hasattr(env, "_head_pose_neck_ids"):
        # Share the id cache with head_pose_tracking (either may run first).
        head_pose_tracking(env, command_name=command_name, asset_cfg=asset_cfg)

    neck_ids = env._head_pose_neck_ids
    joint_pos = asset.data.joint_pos
    measured = joint_pos[:, neck_ids] + joint_pos[:, env._head_pose_bl_ids] * env._head_pose_bl_mask
    err = (measured - asset.data.default_joint_pos[:, neck_ids]) - cmd

    if gate_height_low is not None:
        z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
        t = torch.clamp((z - gate_height_low) / max(gate_height_high - gate_height_low, 1e-6), 0.0, 1.0)
        gate = t * t * (3.0 - 2.0 * t)
        quat = asset.data.root_link_quat_w
        cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
        tilt_deg = torch.rad2deg(torch.acos(cos_tilt.clamp(-1.0, 1.0)))
        st = torch.clamp((gate_tilt_zero_deg - tilt_deg) / max(gate_tilt_zero_deg - gate_tilt_full_deg, 1e-6), 0.0, 1.0)
        gate = gate * (st * st * (3.0 - 2.0 * st))
        err = err * gate.unsqueeze(-1)
    else:
        gate = None

    if not hasattr(env, "_head_bias_ema"):
        env._head_bias_ema = torch.zeros_like(err)
    # Freshly reset envs: drop the previous episode's accumulated bias.
    fresh = env.episode_length_buf <= 1
    env._head_bias_ema[fresh] = 0.0

    alpha = min(1.0, float(env.step_dt) / max(tau_s, 1e-6))
    env._head_bias_ema = (1.0 - alpha) * env._head_bias_ema + alpha * err
    out = -env._head_bias_ema.abs().mean(dim=-1)
    if gate is not None:
        out = out * gate
    return out


def body_pose_tracking_6d(env: ManagerBasedRlEnv, command_name: str = "body_pose", nominal_height: float = 0.095, xy_std: float = 0.02, z_std: float = 0.01, angle_std: float = math.radians(8), asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    dx, dy, dz = cmd[:, 0], cmd[:, 1], cmd[:, 2]
    droll, dpitch, dyaw = cmd[:, 3], cmd[:, 4], cmd[:, 5]

    # Position relative to env spawn origin. nan_to_num because MuJoCo can
    # produce NaN on contact instability and we don't want to taint the reward.
    pos_w = asset.data.root_link_pos_w
    origin = env.scene.terrain.env_origins
    rel = torch.nan_to_num(pos_w - origin, nan=0.0)
    x_err = rel[:, 0] - dx
    y_err = rel[:, 1] - dy
    z_err = rel[:, 2] - (nominal_height + dz)

    roll, pitch, yaw = euler_xyz_from_quat(asset.data.root_link_quat_w)

    roll_err = roll - droll
    pitch_err = pitch - dpitch
    yaw_err = wrap_to_pi(yaw - dyaw)

    r_x = torch.exp(-((x_err / xy_std) ** 2))
    r_y = torch.exp(-((y_err / xy_std) ** 2))
    r_z = torch.exp(-((z_err / z_std) ** 2))
    r_r = torch.exp(-((roll_err / angle_std) ** 2))
    r_p = torch.exp(-((pitch_err / angle_std) ** 2))
    r_w = torch.exp(-((yaw_err / angle_std) ** 2))

    return (r_x + r_y + r_z + r_r + r_p + r_w) / 6.0


def termination_param_curriculum(env: ManagerBasedRlEnv, env_ids: torch.Tensor, term_name: str, param_stages: list[dict]) -> torch.Tensor:
    # TerminationManager keeps its own deepcopy of the cfg dict, so the live
    # term_cfgs list must be edited directly — env.cfg.terminations is a no-op.
    # Useful for disabling a termination later in training (e.g. set
    # bad_orientation's limit_angle to pi at iter N so the robot can fall over
    # without ending the episode and learn to recover).
    #
    # param_stages: list of {step: int, params: dict}. The dict is shallow-merged
    # into the live term_cfg.params at the latest matching stage.
    del env_ids
    tm = env.termination_manager
    if term_name not in tm._term_names:
        # Term was removed (e.g. play mode disables fell_over entirely).
        return torch.tensor(0.0)
    idx = tm._term_names.index(term_name)
    term_cfg = tm._term_cfgs[idx]

    current = param_stages[0]["params"]
    for stage in param_stages:
        if env.common_step_counter >= stage["step"]:
            current = stage["params"]
    term_cfg.params.update(current)

    first_val = next(iter(current.values()))
    return torch.tensor(float(first_val) if isinstance(first_val, (int, float)) else 0.0)


def body_pose_tracking_locomotion(env: ManagerBasedRlEnv, command_name: str = "body_pose", nominal_height: float = 0.105, xy_std: float = 0.02, z_std: float = 0.03, angle_std: float = math.radians(30), axis_weights: tuple[float, float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0), vel_gate_command_name: str | None = None, vel_gate_std: float = 0.1, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, feet_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    dx, dy, dz = cmd[:, 0], cmd[:, 1], cmd[:, 2]
    droll, dpitch, dyaw = cmd[:, 3], cmd[:, 4], cmd[:, 5]

    pos_w = asset.data.root_link_pos_w
    roll, pitch, trunk_yaw = euler_xyz_from_quat(asset.data.root_link_quat_w)

    foot_pos = asset.data.site_pos_w[:, feet_cfg.site_ids]
    foot_quat = asset.data.site_quat_w[:, feet_cfg.site_ids]
    feet_centroid = foot_pos.mean(dim=1)

    # Trunk xy in body frame relative to feet centroid (rotate world Δxy by −yaw).
    dx_w = pos_w[:, 0] - feet_centroid[:, 0]
    dy_w = pos_w[:, 1] - feet_centroid[:, 1]
    cos_y = torch.cos(trunk_yaw)
    sin_y = torch.sin(trunk_yaw)
    x_body = cos_y * dx_w + sin_y * dy_w
    y_body = -sin_y * dx_w + cos_y * dy_w

    # Z relative to spawn-origin terrain height (still in world).
    origin = env.scene.terrain.env_origins
    z_world = torch.nan_to_num(pos_w[:, 2] - origin[:, 2], nan=0.0)

    # Feet yaws → circular mean. NOTE: this depends on the site orientation
    # matching the foot pointing direction; if the site frame is rotated, this
    # yaw reference may have an offset (constant per-env, so dyaw=0 still maps
    # to "feet-aligned").
    fqw, fqx, fqy, fqz = foot_quat[..., 0], foot_quat[..., 1], foot_quat[..., 2], foot_quat[..., 3]
    foot_yaws = torch.atan2(2.0 * (fqw * fqz + fqx * fqy), 1.0 - 2.0 * (fqy * fqy + fqz * fqz))
    mean_foot_yaw = torch.atan2(torch.sin(foot_yaws).mean(dim=1), torch.cos(foot_yaws).mean(dim=1))

    x_err = x_body - dx
    y_err = y_body - dy
    z_err = z_world - (nominal_height + dz)
    roll_err = roll - droll
    pitch_err = pitch - dpitch
    yaw_err = wrap_to_pi(trunk_yaw - mean_foot_yaw - dyaw)

    r_x = torch.exp(-((x_err / xy_std) ** 2))
    r_y = torch.exp(-((y_err / xy_std) ** 2))
    r_z = torch.exp(-((z_err / z_std) ** 2))
    r_r = torch.exp(-((roll_err / angle_std) ** 2))
    r_p = torch.exp(-((pitch_err / angle_std) ** 2))
    r_w = torch.exp(-((yaw_err / angle_std) ** 2))

    # Per-axis weighted mean. Pass axis_weights=(0,0,1,1,1,1) to disable xy
    # tracking — useful when xy lean is mechanically coupled to pitch/roll on
    # the robot, making independent xy commands a noise source rather than a
    # learnable objective.
    wx, wy, wz, wr, wp, wyaw = axis_weights
    total_w = wx + wy + wz + wr + wp + wyaw
    reward = (wx * r_x + wy * r_y + wz * r_z + wr * r_r + wp * r_p + wyaw * r_w) / max(total_w, 1e-6)

    # Optional gate: when vel_gate_command_name is set, scale the reward by a
    # Gaussian on the velocity command's magnitude. With vel_gate_std ≈ 0.1,
    # the gate is ~1 when commanded velocity is 0 and decays to ~exp(-9)≈0
    # by |vel_cmd|≥0.3 — body tracking only meaningfully contributes when the
    # robot is supposed to be standing still. Avoids the tracking vs walking
    # conflict that prevented the previous run from learning either well.
    if vel_gate_command_name is not None:
        # Gate on commanded LINEAR velocity only (xy) — turning in place still
        # leaves body pose meaningful, but walking forward/sideways doesn't.
        vel_cmd = env.command_manager.get_command(vel_gate_command_name)
        vel_mag = torch.linalg.vector_norm(vel_cmd[:, :2], dim=-1)
        gate = torch.exp(-((vel_mag / vel_gate_std) ** 2))
        reward = reward * gate

    return reward


def pose_command_range_curriculum(env: ManagerBasedRlEnv, env_ids: torch.Tensor, command_name: str, range_stages: list[dict]) -> torch.Tensor:
    # range_stages: list of {step: int, ranges: tuple[(lo, hi), ...]}.
    # The first stage applies before its step; latest passed stage wins.
    # Always uses the live CommandManager term cfg (NOT env.cfg.commands) so
    # updates take effect — CommandManager keeps its own term refs and reads
    # `term.cfg.ranges` each resample.
    del env_ids

    term = env.command_manager.get_term(command_name)
    assert term is not None, f"Command term '{command_name}' not found"
    cfg = term.cfg  # type: ignore[assignment]

    current = range_stages[0]["ranges"]
    for stage in range_stages:
        if env.common_step_counter >= stage["step"]:
            current = stage["ranges"]

    cfg.ranges = tuple(current)
    max_abs = max((max(abs(lo), abs(hi)) for lo, hi in current), default=0.0)
    return torch.tensor(max_abs)


def randomize_dof_field_scaled(env: ManagerBasedRlEnv, env_ids: torch.Tensor, field: str, scale_range: tuple[float, float], asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    # Scale a per-dof model field (e.g. dof_frictionloss/dof_damping) per episode
    # WITHOUT accumulating: restore nominal, then apply a fresh scale.
    #
    # ``field`` doubles as the domain_randomization field name. NOTE: under the BAM
    # actuator, dof_frictionloss and dof_damping are zeroed in edit_spec (BAM models
    # friction itself), so scaling them is a no-op — these only matter with the XML
    # position actuator. Kept correct to avoid the accumulation footgun if re-enabled.
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, slice):
        joint_ids = list(range(len(asset.indexing.joint_ids)))[joint_ids]
    dof_indices = asset.indexing.joint_v_adr[joint_ids]

    mf = getattr(env.sim.model, field)
    cache_attr = f"_original_{field}"
    if not hasattr(env, cache_attr):
        setattr(env, cache_attr, mf[0, dof_indices].clone())
    nominal = getattr(env, cache_attr)

    num_envs = len(env_ids)
    num_dofs = len(dof_indices)

    mf[env_ids[:, None], dof_indices] = nominal.unsqueeze(0).expand(num_envs, -1)
    lo, hi = scale_range
    scales = torch.rand(num_envs, num_dofs, device=env.device) * (hi - lo) + lo
    mf[env_ids[:, None], dof_indices] *= scales
    return torch.tensor(float(hi))


def _ball_kick_dir(env: ManagerBasedRlEnv) -> torch.Tensor:
    # Set by ``reset_ball_in_front_of_foot`` to the robot's forward direction at
    # episode reset. Frozen for the episode so the policy can't redefine "forward"
    # by turning after the kick.
    if not hasattr(env, "_ball_kick_dir_w"):
        env._ball_kick_dir_w = torch.zeros(env.num_envs, 2, device=env.device)
        env._ball_kick_dir_w[:, 0] = 1.0
    return env._ball_kick_dir_w


def reset_ball_in_front_of_foot(env: ManagerBasedRlEnv, env_ids: torch.Tensor, offset: tuple = (0.09, -0.042), noise_xy: float = 0.015, ball_radius: float = 0.035, asset_name: str = "ball"):
    # ``offset`` is the nominal ball-center position in the robot's yaw frame:
    # at HOME the right foot is centered at (0, -0.042) with the toe tip at
    # x≈0.034, so (0.08, -0.042) puts a 35mm-radius ball ~1cm in front of the
    # toe. ``noise_xy`` (uniform ± per axis) is the placement DR: the policy is
    # BLIND to the ball, so this is what forces a swing that works across the
    # real-world placement error.
    #
    # Reads the robot root from qpos directly (root_link_pos_w lags until the
    # next forward()); must be registered AFTER reset_base / set_ground_state
    # (events run in dict insertion order) so the robot pose is final.
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device)
    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene[asset_name]

    root = env.sim.data.qpos[env_ids][:, robot.indexing.free_joint_q_adr]
    qw, qx, qy, qz = root[:, 3], root[:, 4], root[:, 5], root[:, 6]
    yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)

    n = len(env_ids)
    off = torch.tensor(offset, device=env.device, dtype=torch.float).repeat(n, 1)
    off += (torch.rand(n, 2, device=env.device) * 2.0 - 1.0) * noise_xy

    pose = torch.zeros(n, 7, device=env.device)
    pose[:, 0] = root[:, 0] + cos_y * off[:, 0] - sin_y * off[:, 1]
    pose[:, 1] = root[:, 1] + sin_y * off[:, 0] + cos_y * off[:, 1]
    pose[:, 2] = env.scene.terrain.env_origins[env_ids, 2] + ball_radius
    pose[:, 3] = 1.0
    ball.write_root_link_pose_to_sim(pose, env_ids)
    ball.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=env.device), env_ids)

    kick_dir = _ball_kick_dir(env)
    kick_dir[env_ids, 0] = cos_y
    kick_dir[env_ids, 1] = sin_y


def ball_forward_velocity(env: ManagerBasedRlEnv, asset_name: str = "ball", max_speed: float = 5.0) -> torch.Tensor:
    # Dense and linear-in-speed up to ``max_speed``: every extra bit of forward
    # ball speed pays more every step the ball keeps rolling, so exploration
    # nudges bootstrap the kick with no peak-detection machinery. Backward /
    # lateral ball motion earns 0 rather than a penalty — a mis-hit shouldn't
    # scare the policy away from contacting the ball at all.
    #
    # With ``max_speed`` set to a TARGET speed (rather than a large cap), pair
    # with ``ball_speed_overshoot_penalty``: the reward saturating at the target
    # alone does NOT remove "harder is better" — a harder kick keeps the ball
    # at/above the cap for more steps, so the rolling-time integral still grows
    # with strike speed. The overshoot penalty is what makes the target the
    # actual optimum.
    ball: Entity = env.scene[asset_name]
    vel_xy = ball.data.root_link_lin_vel_w[:, :2]
    fwd = (vel_xy * _ball_kick_dir(env)).sum(dim=1)
    return torch.nan_to_num(fwd, nan=0.0).clamp(0.0, max_speed)


def ball_speed_overshoot_penalty(env: ManagerBasedRlEnv, asset_name: str = "ball", target_speed: float = 1.0, max_penalty: float = 5.0) -> torch.Tensor:
    ball: Entity = env.scene[asset_name]
    vel_xy = ball.data.root_link_lin_vel_w[:, :2]
    fwd = (vel_xy * _ball_kick_dir(env)).sum(dim=1)
    over = torch.nan_to_num(fwd, nan=0.0) - target_speed
    return over.clamp(0.0, max_penalty)


def single_foot_grounded_reward(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    if sensor_name not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)
    found = env.scene.sensors[sensor_name].data.found
    if found.dim() > 1:
        found = found.sum(dim=-1)
    return torch.clamp(found, 0.0, 1.0)


def ball_pos_in_base(env: ManagerBasedRlEnv, asset_name: str = "ball") -> torch.Tensor:
    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene[asset_name]
    rel = ball.data.root_link_pos_w - robot.data.root_link_pos_w
    rot = matrix_from_quat(robot.data.root_link_quat_w)
    return torch.bmm(rot.transpose(1, 2), rel.unsqueeze(-1)).squeeze(-1)


def ball_vel_in_base(env: ManagerBasedRlEnv, asset_name: str = "ball") -> torch.Tensor:
    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene[asset_name]
    rot = matrix_from_quat(robot.data.root_link_quat_w)
    vel = ball.data.root_link_lin_vel_w
    return torch.bmm(rot.transpose(1, 2), vel.unsqueeze(-1)).squeeze(-1)


# Phase envelope: the button slot command carries a phase, which drives
# a trapezoid target YAW RATE (and not a pose like the crouch).
#   [0, accel_end)        0.5 s   0 -> rate_max    (launch)
#   [accel_end, hold_end) 1.6 s   rate_max         (cruise)
#   [hold_end, brake_end) 0.5 s   rate_max -> 0    (braking)
#   [brake_end, 1.0)      1.4 s   0                (standing rest)
# Area under the envelope over one cycle = 2.1 * SPIN_RATE_MAX rad. At 3.0 rad/s:
# 2.1 * 3.0 = 6.3 rad ~ 1 turn (and not ~2, as with the old 6.0 target).
SPIN_PERIOD = 4.0
SPIN_RATE_MAX = 3.0
SPIN_ACCEL_END = 0.125
SPIN_HOLD_END = 0.525
SPIN_BRAKE_END = 0.650


def spin_rate_by_phase(phase: torch.Tensor, rate_max: float = SPIN_RATE_MAX, accel_end: float = SPIN_ACCEL_END, hold_end: float = SPIN_HOLD_END, brake_end: float = SPIN_BRAKE_END) -> torch.Tensor:
    w = torch.zeros_like(phase)
    accel = phase < accel_end
    w = torch.where(accel, rate_max * phase / accel_end, w)
    hold = (phase >= accel_end) & (phase < hold_end)
    w = torch.where(hold, torch.full_like(phase, rate_max), w)
    brake = (phase >= hold_end) & (phase < brake_end)
    w = torch.where(brake, rate_max * (1.0 - (phase - hold_end) / (brake_end - hold_end)), w)
    return w


def spin_gate_by_phase(phase: torch.Tensor, rate_max: float = SPIN_RATE_MAX, accel_end: float = SPIN_ACCEL_END, hold_end: float = SPIN_HOLD_END, brake_end: float = SPIN_BRAKE_END) -> torch.Tensor:
    return spin_rate_by_phase(phase, rate_max, accel_end, hold_end, brake_end) / rate_max


def spin_phase_from_command(cmd: torch.Tensor) -> torch.Tensor:
    return (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * torch.pi)) % 1.0


def _spin_target_rate(env: ManagerBasedRlEnv, command_name: str, rate_max: float, accel_end: float, hold_end: float, brake_end: float) -> torch.Tensor:
    phase = spin_phase_from_command(env.command_manager.get_command(command_name))
    return spin_rate_by_phase(phase, rate_max, accel_end, hold_end, brake_end)


def _spin_gate(env: ManagerBasedRlEnv, command_name: str, rate_max: float, accel_end: float, hold_end: float, brake_end: float) -> torch.Tensor:
    phase = spin_phase_from_command(env.command_manager.get_command(command_name))
    return spin_gate_by_phase(phase, rate_max, accel_end, hold_end, brake_end)


def spin_rate_reward_from_values(omega_z: torch.Tensor, omega_target: torch.Tensor, std: float) -> torch.Tensor:
    return torch.exp(-(((omega_z - omega_target) / std) ** 2))


def spin_rate_track(env: ManagerBasedRlEnv, command_name: str = "twist", std: float = 1.5, rate_max: float = SPIN_RATE_MAX, accel_end: float = SPIN_ACCEL_END, hold_end: float = SPIN_HOLD_END, brake_end: float = SPIN_BRAKE_END, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    # ω_z is taken in the body frame (this is what the IMU gyro sees, hence what
    # the policy observes). A rotation in the wrong direction is punished more than
    # standing still, the Gaussian being centered on a positive target.
    asset: Entity = env.scene[asset_cfg.name]
    omega_z = asset.data.root_link_ang_vel_b[:, 2]
    target = _spin_target_rate(env, command_name, rate_max, accel_end, hold_end, brake_end)
    return spin_rate_reward_from_values(omega_z, target, std)


def spin_rate_l1(env: ManagerBasedRlEnv, command_name: str = "twist", rate_max: float = SPIN_RATE_MAX, accel_end: float = SPIN_ACCEL_END, hold_end: float = SPIN_HOLD_END, brake_end: float = SPIN_BRAKE_END, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    # L1 bootstrap: constant gradient towards the target even when the Gaussian
    # of `spin_rate_track` saturates far from the target. To be used with a
    # POSITIVE weight (the returned value is already negative).
    asset: Entity = env.scene[asset_cfg.name]
    omega_z = asset.data.root_link_ang_vel_b[:, 2]
    target = _spin_target_rate(env, command_name, rate_max, accel_end, hold_end, brake_end)
    return -torch.abs(omega_z - target)


SPIN_LAUNCH_DRIFT_SCALE = 0.2  # attenuation of the drift cost during the launch


def spin_stay_in_place(env: ManagerBasedRlEnv, command_name: str = "twist", launch_scale: float = SPIN_LAUNCH_DRIFT_SCALE, accel_end: float = SPIN_ACCEL_END, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    # ATTENUATED DURING THE LAUNCH: over `[0, accel_end)` the robot must push on the
    # ground to inject angular momentum, and the entry state gives it up to
    # 0.3 m/s that it is supposed to CONVERT into rotation. Charging translation at full
    # price at that moment therefore directly opposes the objective. The cost is
    # multiplied by `launch_scale` on that segment only, and is full price afterwards
    # (cruise, braking, rest) where "in place" is the real criterion.
    #
    # Unlike the other spin primers, this term is NOT switched off by
    # `spin_gate_by_phase`: during the rest we precisely want it to stay full,
    # since that is when the robot must be motionless.
    asset: Entity = env.scene[asset_cfg.name]
    v_xy = asset.data.root_link_lin_vel_b[:, :2]
    cost = torch.sum(torch.square(v_xy), dim=1)

    phase = spin_phase_from_command(env.command_manager.get_command(command_name))
    scale = torch.where(phase < accel_end, torch.full_like(cost, launch_scale), torch.ones_like(cost))
    return cost * scale


# Half-track measured on the rollers model (HOME pose, left_foot/right_foot sites):
# 0.0499 m, vs 0.03 m estimated in the spec. Mechanical consequence of SPIN_RATE_MAX
# (A1): expected differential = 2*SPIN_RATE_MAX*half_track/r, r = 0.0175 m.
# At the old 6.0 rad/s target: 2*6.0*0.0499/0.0175 = 34.2 rad/s (kept as
# 34.0, i.e. +71% vs the 20.0 estimated in the spec -> 30% threshold exceeded).
# At the new 3.0 rad/s target: 2*3.0*0.0499/0.0175 = 17.1 rad/s. Leaving 34.0
# here would cap the term at tanh(17.1/34) = 0.47 of its own maximum, which
# would weaken exactly the shaping we want to strengthen (cf. spin_stay_in_place).
SPIN_WHEEL_OMEGA_SCALE = 17.0  # rad/s; recalibrated on the measured half-track and SPIN_RATE_MAX = 3.0


def spin_wheel_differential_from_values(diff: torch.Tensor, gate: torch.Tensor, omega_scale: float) -> torch.Tensor:
    return gate * torch.tanh(torch.clamp(diff, min=0.0) / omega_scale)


def spin_wheel_differential(env: ManagerBasedRlEnv, command_name: str = "twist", omega_scale: float = SPIN_WHEEL_OMEGA_SCALE, rate_max: float = SPIN_RATE_MAX, accel_end: float = SPIN_ACCEL_END, hold_end: float = SPIN_HOLD_END, brake_end: float = SPIN_BRAKE_END) -> torch.Tensor:
    # For a counter-clockwise spin, the left skate goes backwards and the right one forwards; the 4
    # wheels spinning positive when going forward, this gives ω_R − ω_L > 0. The tanh
    # saturates at `omega_scale` to avoid a race for wheel speed.
    asset: Entity = env.scene["robot"]
    lf_ids, _ = asset.find_joints("passive_LF_?wheel")
    lr_ids, _ = asset.find_joints("passive_LR_?wheel")
    rf_ids, _ = asset.find_joints("passive_RF_?wheel")
    rr_ids, _ = asset.find_joints("passive_RR_?wheel")

    vel = asset.data.joint_vel
    omega_left = (vel[:, lf_ids[0]] + vel[:, lr_ids[0]]) / 2.0
    omega_right = (vel[:, rf_ids[0]] + vel[:, rr_ids[0]]) / 2.0
    gate = _spin_gate(env, command_name, rate_max, accel_end, hold_end, brake_end)
    return spin_wheel_differential_from_values(omega_right - omega_left, gate, omega_scale)


def spin_grounded(env: ManagerBasedRlEnv, sensor_name: str, command_name: str = "twist", rate_max: float = SPIN_RATE_MAX, accel_end: float = SPIN_ACCEL_END, hold_end: float = SPIN_HOLD_END, brake_end: float = SPIN_BRAKE_END) -> torch.Tensor:
    # Variant of the swizzle's `grounded_reward`, which is not reusable here:
    # it weights by cmd_x, which is cos(2πφ) on the phase command.
    from mjlab.sensor import ContactSensor

    sensor: ContactSensor = env.scene[sensor_name]
    contact_time = sensor.data.current_contact_time
    assert contact_time is not None
    n_contact = torch.sum((contact_time > 0.0).float(), dim=1)
    grounded = (n_contact >= 2).float()
    gate = _spin_gate(env, command_name, rate_max, accel_end, hold_end, brake_end)
    return grounded * gate


def leg_antisymmetry(env: ManagerBasedRlEnv, command_name: str = "twist", asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, joint_bases: tuple = ("hip_pitch", "knee"), rate_max: float = SPIN_RATE_MAX, accel_end: float = SPIN_ACCEL_END, hold_end: float = SPIN_HOLD_END, brake_end: float = SPIN_BRAKE_END) -> torch.Tensor:
    # The robot has MIRRORED left/right sign conventions: a symmetric
    # pose satisfies q_L + q_R ≈ 0 (cf. `leg_symmetry_reward`), so the
    # scissor satisfies q_L ≈ q_R. Returns `gate(φ) · (−mean|q_L − q_R|)` — to be
    # used with a POSITIVE weight, decaying via curriculum: the primer
    # fades out to let the policy refine its own gesture.
    asset: Entity = env.scene[asset_cfg.name]
    left, right = [], []
    for base in joint_bases:
        li, _ = asset.find_joints([f"left_{base}"])
        ri, _ = asset.find_joints([f"right_{base}"])
        left.append(li[0])
        right.append(ri[0])
    lids = torch.tensor(left, device=env.device)
    rids = torch.tensor(right, device=env.device)

    q = asset.data.joint_pos
    scissor = -torch.abs(q[:, lids] - q[:, rids]).mean(dim=-1)
    gate = _spin_gate(env, command_name, rate_max, accel_end, hold_end, brake_end)
    return gate * scissor


# The backlash model (robot_groundcontact_backlash.xml) puts an unactuated
# ``passive_<joint>_backlash`` hinge in series with each servo joint. The link
# angle is qpos[servo] + qpos[backlash], and the real encoder sits on the
# OUTPUT side of the play — it reads the sum. These obs replace joint_pos_rel /
# joint_vel_rel in backlash tasks (see task_backlash.py) so the policy sees
# exactly what the runtime will feed it. The asset_cfg regex is expected to
# select only the servo joints (the usual ``^(?!passive_).*``).


def _backlash_encoder_ids(env: "ManagerBasedRlEnv", asset: Entity, asset_cfg: SceneEntityCfg) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # mask is 1.0 where a matching passive_<name>_backlash joint exists, so the
    # same obs functions run unchanged on models without backlash joints.
    key = (asset_cfg.name, str(asset_cfg.joint_ids))
    cache = env.__dict__.setdefault("_backlash_encoder_cache", {})
    hit = cache.get(key)
    if hit is not None:
        return hit

    names = asset.joint_names
    jnt_ids = asset_cfg.joint_ids
    if isinstance(jnt_ids, slice):
        main_ids = list(range(len(names)))[jnt_ids]
    else:
        main_ids = [int(i) for i in jnt_ids]
    name_to_id = {n: i for i, n in enumerate(names)}
    bl_ids, mask = [], []
    for i in main_ids:
        bl = name_to_id.get(f"passive_{names[i]}_backlash")
        bl_ids.append(0 if bl is None else bl)
        mask.append(0.0 if bl is None else 1.0)

    device = asset.data.joint_pos.device
    out = (torch.tensor(main_ids, dtype=torch.long, device=device), torch.tensor(bl_ids, dtype=torch.long, device=device), torch.tensor(mask, dtype=torch.float32, device=device))
    cache[key] = out
    return out


def joint_pos_rel_backlash(env: "ManagerBasedRlEnv", biased: bool = False, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    # Returns (qpos[servo] + qpos[backlash]) - default[servo]. With biased=True
    # the per-env encoder-calibration bias is applied to the servo reading (one
    # encoder per servo → one bias per joint; the backlash summand stays raw).
    asset: Entity = env.scene[asset_cfg.name]
    main_ids, bl_ids, mask = _backlash_encoder_ids(env, asset, asset_cfg)
    joint_pos = asset.data.joint_pos_biased if biased else asset.data.joint_pos
    pos = joint_pos[:, main_ids] + asset.data.joint_pos[:, bl_ids] * mask
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    return pos - default_joint_pos[:, main_ids]


def joint_vel_rel_backlash(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    # The firmware derives present_velocity from encoder positions, so it also
    # sees the backlash motion: qvel[servo] + qvel[backlash].
    asset: Entity = env.scene[asset_cfg.name]
    main_ids, bl_ids, mask = _backlash_encoder_ids(env, asset, asset_cfg)
    vel = asset.data.joint_vel[:, main_ids] + asset.data.joint_vel[:, bl_ids] * mask
    default_joint_vel = asset.data.default_joint_vel
    assert default_joint_vel is not None
    return vel - default_joint_vel[:, main_ids]


# One policy, both directions: the command is a single sit/stand flag carried
# in the twist slot (cmd = [sit_flag, 0, 0], so "stand" is the all-zero
# command — same deployment idle as every other policy). All task rewards
# below select their target (SIT keyframe + SIT_Z vs HOME + STAND_Z) from the
# live command, per env, so the same reward stack drives the descent, the
# seated rest, the rise and the standing rest. Uses the _servo_* helpers →
# backlash-model compatible.


class SitStandCommand(UniformVelocityCommand):
    # ``alpha`` (0 = STAND target, 1 = SIT target) slews toward the flag at a
    # constant rate (full transition in cfg.ramp_s seconds) and is what the
    # posture_* rewards track. THE anti-crash mechanism: with a binary target,
    # arriving early pays the full goal-state jackpot for every step saved,
    # while the linear speed-cap penalties integrate to a bounded excess-
    # distance cost — an instant drop beat a 1 s descent by ~7×. With the
    # slewed target, being AHEAD of the ramp scores ~0 on the height/composite
    # stack (z far from the commanded height), so tracking the slow setpoint IS
    # the argmax; the caps remain as backstops for overshoot/bounce. The OBS
    # stays the raw binary flag (deployment: runtime writes 0/1; the trained
    # response to a flip is the ~ramp_s glide).
    #
    # On episode reset, alpha is initialised from the robot's ACTUAL trunk
    # height, not the flag — a seated spawn must not be dragged upward by a
    # stand-initialised ramp (and vice versa).

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._sit_prob = float(getattr(cfg, "sit_prob", 0.5))
        self._ramp_s = float(getattr(cfg, "ramp_s", 2.0))
        self._sit_z = float(getattr(cfg, "sit_z", 0.060))
        self._stand_z = float(getattr(cfg, "stand_z", 0.115))
        self._env_ref = env
        self._alpha = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.vel_command_b

    @property
    def alpha(self) -> torch.Tensor:
        return self._alpha

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        if n == 0:
            return
        sit = (torch.rand(n, device=self.device) < self._sit_prob).float()
        self.vel_command_b[env_ids] = 0.0
        self.vel_command_b[env_ids, 0] = sit

    def _alpha_from_height(self) -> torch.Tensor:
        z = torch.nan_to_num(self.robot.data.root_link_pos_w[:, 2] - self._env_ref.scene.terrain.env_origins[:, 2], nan=self._stand_z)
        return torch.clamp((self._stand_z - z) / max(self._stand_z - self._sit_z, 1e-6), 0.0, 1.0)

    def compute(self, dt: float) -> None:
        super().compute(dt)
        # Episode-start re-init of the blend from the ACTUAL trunk height.
        # Done here (not in reset()) because the command manager resets BEFORE
        # the set_ground_state event teleports the robot, so reset() would read
        # the pre-teleport height. On the first compute of an episode the spawn
        # state is in place.
        fresh = self._env_ref.episode_length_buf <= 1
        if fresh.any():
            self._alpha = torch.where(fresh, self._alpha_from_height(), self._alpha)
        step = dt / max(self._ramp_s, 1e-6)
        delta = self.vel_command_b[:, 0] - self._alpha
        self._alpha += torch.clamp(delta, -step, step)

    def _update_command(self) -> None:
        pass

    def _update_metrics(self) -> None:
        pass


@_dataclass(kw_only=True)
class SitStandCommandCfg(UniformVelocityCommandCfg):
    class_type: type = SitStandCommand
    sit_prob: float = 0.5
    ramp_s: float = 2.0
    sit_z: float = 0.060
    stand_z: float = 0.115

    def build(self, env: ManagerBasedRlEnv) -> "SitStandCommand":
        return SitStandCommand(self, env)


def _posture_blend(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    term = env.command_manager.get_term(command_name)
    alpha = getattr(term, "alpha", None)
    if alpha is not None:
        return alpha
    return env.command_manager.get_command(command_name)[:, 0]


def _posture_targets(env: ManagerBasedRlEnv, asset: Entity, command_name: str, sit_overrides: dict) -> tuple[torch.Tensor, torch.Tensor]:
    # STAND target = default_joint_pos (HOME); SIT target = HOME with the
    # keyframe overrides applied; the SLEWED blend interpolates between them,
    # so mid-ramp the rewarded pose folds in sync with the descending height.
    blend = _posture_blend(env, command_name)
    stand_target = _servo_default_joint_pos(env, asset)
    sit_target = stand_target.clone()
    for idx, val in sit_overrides.items():
        sit_target[:, idx] = val
    target = stand_target + blend.unsqueeze(-1) * (sit_target - stand_target)
    return blend, target


def _posture_height(env: ManagerBasedRlEnv, command_name: str, sit_z: float, stand_z: float) -> tuple[torch.Tensor, torch.Tensor]:
    blend = _posture_blend(env, command_name)
    target_z = stand_z + blend * (sit_z - stand_z)
    asset = env.scene["robot"]
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    return target_z, z


def posture_pose_match(env: ManagerBasedRlEnv, command_name: str, sit_overrides: dict, joint_indices: list, std: float = 0.5, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    _, target = _posture_targets(env, asset, command_name, sit_overrides)
    joint_pos = _servo_joint_pos(env, asset)[:, joint_indices]
    target = target[:, joint_indices]
    return torch.exp(-(((joint_pos - target) / std) ** 2)).mean(dim=-1)


def posture_pose_l1(env: ManagerBasedRlEnv, command_name: str, sit_overrides: dict, joint_indices: list, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    _, target = _posture_targets(env, asset, command_name, sit_overrides)
    joint_pos = _servo_joint_pos(env, asset)[:, joint_indices]
    target = target[:, joint_indices]
    return -torch.abs(joint_pos - target).mean(dim=-1)


def posture_height_gaussian(env: ManagerBasedRlEnv, command_name: str, sit_z: float, stand_z: float, std: float = 0.02, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    del asset_cfg  # trunk z read via _posture_height
    target_z, z = _posture_height(env, command_name, sit_z, stand_z)
    return torch.exp(-(((z - target_z) / std) ** 2))


def posture_height_l1(env: ManagerBasedRlEnv, command_name: str, sit_z: float, stand_z: float, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    # While the robot rests in the *wrong* posture this charges a constant
    # per-step cost (~|Δz| = 55 mm), which is what makes "ignore the command"
    # a net-negative strategy in both directions.
    del asset_cfg
    target_z, z = _posture_height(env, command_name, sit_z, stand_z)
    return -torch.abs(z - target_z)


def posture_composite(env: ManagerBasedRlEnv, command_name: str, sit_overrides: dict, joint_indices: list, sit_z: float, stand_z: float, height_std: float = 0.03, upright_std: float = 0.40, pose_std: float = 0.40, head_std: float | None = None, head_command_name: str = "head_pose", asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    # The posture-conditioned version of ``standing_composite_score``: a
    # deficiency in any factor collapses the whole term, so partial-sum
    # compromises (plank, flop, lean) never pay. Both rest states demand an
    # upright trunk, so the upright factor is posture-independent.
    #
    # ``head_std`` (optional): adds a fourth factor on the neck/head joints vs
    # the ``head_pose`` command (same error convention as head_pose_tracking).
    # Without it the goal state is head-blind: the trained policy rested with
    # the head dangling to the floor — trunk upright, legs in pose, z on target
    # all held while the head hung, costing only the light tracking term. With
    # the factor, "arrived" REQUIRES the head at its commanded pose, so head
    # assist stays free mid-transition (composite is ≈0 there anyway) but must
    # be retracted to collect the goal reward.
    asset = env.scene[asset_cfg.name]
    _, target = _posture_targets(env, asset, command_name, sit_overrides)
    target_z, z = _posture_height(env, command_name, sit_z, stand_z)

    height_score = torch.exp(-(((z - target_z) / height_std) ** 2))

    quat = asset.data.root_link_quat_w
    tilt_sq = 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    upright_score = torch.exp(-tilt_sq / (upright_std * upright_std))

    joint_pos = _servo_joint_pos(env, asset)[:, joint_indices]
    pose_err_sq = ((joint_pos - target[:, joint_indices]) ** 2).mean(dim=-1)
    pose_score = torch.exp(-pose_err_sq / (pose_std * pose_std))

    score = height_score * upright_score * pose_score

    if head_std is not None:
        if not hasattr(env, "_head_pose_neck_ids"):
            ids, _ = asset.find_joints_by_actuator_names(_NECK_JOINT_PATTERNS)
            env._head_pose_neck_ids = torch.tensor(ids, device=env.device, dtype=torch.long)
        neck_ids = env._head_pose_neck_ids
        head_cmd = env.command_manager.get_command(head_command_name)
        actual = asset.data.joint_pos[:, neck_ids] - asset.data.default_joint_pos[:, neck_ids]
        head_err_sq = ((actual - head_cmd) ** 2).mean(dim=-1)
        score = score * torch.exp(-head_err_sq / (head_std * head_std))

    return score


def posture_stillness(env: ManagerBasedRlEnv, command_name: str, sit_z: float, stand_z: float, band_full: float = 0.012, band_zero: float = 0.03, vel_std: float = 0.05, tilt_full_deg: float = 25.0, tilt_zero_deg: float = 60.0, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    # Generalizes ``seated_stillness`` to both rest states: exp(-(|v|/std)²)
    # gated by a smoothstep on |z − commanded z| (full inside ``band_full``,
    # zero beyond ``band_zero`` → inactive during transitions) and by trunk
    # tilt (a tilted rest — back/face/side — earns nothing). Additionally gated
    # on the target ramp being COMPLETE (|flag − alpha| small), so stillness
    # never pays mid-transition. Makes "rest quietly, upright, at the commanded
    # height" the peak of the stack.
    asset = env.scene[asset_cfg.name]
    target_z, z = _posture_height(env, command_name, sit_z, stand_z)
    v = torch.nan_to_num(asset.data.root_link_lin_vel_w, nan=0.0).norm(dim=-1)

    flag = env.command_manager.get_command(command_name)[:, 0]
    blend = _posture_blend(env, command_name)
    ramp_done = ((flag - blend).abs() < 0.02).float()

    err = torch.abs(z - target_z)
    t = torch.clamp((band_zero - err) / max(band_zero - band_full, 1e-6), 0.0, 1.0)
    z_gate = t * t * (3.0 - 2.0 * t)

    quat = asset.data.root_link_quat_w
    cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    cos_full = math.cos(math.radians(tilt_full_deg))
    cos_zero = math.cos(math.radians(tilt_zero_deg))
    u = torch.clamp((cos_tilt - cos_zero) / max(cos_full - cos_zero, 1e-6), 0.0, 1.0)
    tilt_gate = u * u * (3.0 - 2.0 * u)

    return torch.exp(-((v / vel_std) ** 2)) * z_gate * tilt_gate * ramp_done


def posture_rise_bootstrap(env: ManagerBasedRlEnv, command_name: str, max_height: float, max_vz: float | None = None, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    # The standup-env lesson: destination-only rewards have zero gradient at
    # zero motion, so "stay seated and eat the L1" is a local optimum — paying
    # for the rise *motion* itself makes any attempt immediately positive.
    # Gated off above ``max_height`` (set just ABOVE the stand target so the
    # final cm still pays; gating at exactly STAND_Z parks the policy short).
    # Zero whenever SIT is commanded, so it can never fight the descent.
    # ``max_vz`` caps the rewarded speed (any rise ≥ the cap earns the same, so
    # an explosive launch can't out-earn a gentle one).
    asset = env.scene[asset_cfg.name]
    sit = env.command_manager.get_command(command_name)[:, 0]
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    vz = torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0)
    return torch.clamp(vz, min=0.0, max=max_vz) * (z < max_height).float() * (1.0 - sit)


def trunk_upward_velocity_penalty(env: ManagerBasedRlEnv, max_up_vel: float = 0.08, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    # Mirror of ``trunk_downward_velocity_penalty`` for the rise: charges every
    # step of a too-fast (violent) stand-up, so the explosive rise can't be
    # amortised against arriving-standing reward. Zero at rest, for any rise
    # slower than the cap, and for all downward motion. Introduce via
    # curriculum AFTER the rise is discovered (attempt-tax lesson).
    asset = env.scene[asset_cfg.name]
    vz = torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0)
    return -torch.clamp(vz - max_up_vel, min=0.0)


_ROULADE_FWD_SIGN = 1.0

# Sensor names read by the accumulator update (must match the env cfg).
_ROULADE_SUPPORT_SENSOR = "robot_ground_contact"
_ROULADE_HEAD_SENSOR = "head_ground_contact"

# Head-latch window: head-ground contact while accum is inside this window
# marks the episode as a genuine over-the-head roll. In a real roulade the
# head plants at ~60–120° of body rotation; the window is generous around it.
_HEAD_LATCH_LO = math.radians(20.0)
_HEAD_LATCH_HI = math.radians(170.0)

# Head-top axis in jaw_soft's LOCAL frame (measured empirically 2026-08-13:
# world-up expressed in jaw_soft's frame with the robot settled at HOME).
# The latch requires this axis to point DOWN at contact — "the flat top of
# the head on the floor", not the face or the side of the shell (run-5 fix:
# the run-4 policy rolled over the shoulder, which still touched jaw_soft).
_HEAD_TOP_AXIS = (0.882, 0.0, 0.471)
# dot(top_axis_world, -z) threshold. Measured landmarks (trunk pitched 110°):
# passive face-plant (neck at HOME) reads +0.6, full chin-tuck (neck_pitch −1,
# head_pitch +1) reads −0.99 — 0.3 accepts partial tucks while staying far
# from any face/side contact.
_HEAD_TOP_DOWN_MIN = 0.3

# Sagittal flatness gate on the accumulator (run-5): in a clean forward roll
# the body's LATERAL axis stays horizontal the whole way — its world-z
# component is 2(q_y·q_z + q_w·q_x) ≈ 0 for ANY amount of pure pitch, and
# grows toward ±1 as the roll goes over the shoulder instead. Full rotation
# credit while the lateral axis is within ~30° of horizontal, zero beyond
# ~60°: a side roll does not count as rotation, earns no progress, and never
# opens the landing gate.
_FLAT_FULL = 0.5  # |lateral_axis_z| = sin(30°): full credit below
_FLAT_ZERO = 0.866  # sin(60°): zero credit above


def _lateral_axis_z(quat: torch.Tensor) -> torch.Tensor:
    # World-z component of the body's lateral (y) axis. 0 = flat/sagittal.
    return 2.0 * (quat[:, 2] * quat[:, 3] + quat[:, 0] * quat[:, 1])


def _head_top_down(env: ManagerBasedRlEnv, asset: Entity) -> torch.Tensor:
    # True where the head-top axis points at the floor (dot with -z > min).
    if not hasattr(env, "_roulade_head_body_id"):
        ids, _ = asset.find_bodies("jaw_soft")
        env._roulade_head_body_id = ids[0]
    q = asset.data.body_link_quat_w[:, env._roulade_head_body_id]
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    a, b, c = _HEAD_TOP_AXIS
    # z-component of R(q) @ axis_local
    axis_world_z = 2.0 * (x * z - w * y) * a + 2.0 * (y * z + w * x) * b + (1.0 - 2.0 * (x * x + y * y)) * c
    return axis_world_z < -_HEAD_TOP_DOWN_MIN


def _sensor_any_contact(env: ManagerBasedRlEnv, name: str) -> torch.Tensor | None:
    if name not in env.scene.sensors:
        return None
    found = env.scene.sensors[name].data.found
    return (found.view(found.shape[0], -1) > 0).any(dim=-1)


def _roulade_state(env: ManagerBasedRlEnv) -> tuple:
    if not hasattr(env, "_roulade_accum"):
        z = torch.zeros(env.num_envs, device=env.device)
        env._roulade_accum = z.clone()
        env._roulade_max = z.clone()
        env._roulade_paid = z.clone()
        env._roulade_head_latch = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._roulade_last_update_step = -1
    return env._roulade_accum, env._roulade_max, env._roulade_paid


def _update_roulade_accum(env: ManagerBasedRlEnv, asset: Entity) -> None:
    # SUPPORT GATE (run-1 fix): rotation is integrated only while the robot
    # touches the terrain — a roulade is a supported motion; ballistic flips
    # accumulate nothing, so they neither get paid nor open the completion gate.
    #
    # Also latches env._roulade_head_latch when the head touches the ground
    # while accum is inside the first-quadrant window — the landing annuity
    # requires this, making "over the head" a hard requirement of the task.
    _roulade_state(env)
    step = int(env.common_step_counter)
    if step != env._roulade_last_update_step:
        omega_fwd = _ROULADE_FWD_SIGN * asset.data.root_link_ang_vel_b[:, 1]
        delta = torch.nan_to_num(omega_fwd, nan=0.0) * env.step_dt
        supported = _sensor_any_contact(env, _ROULADE_SUPPORT_SENSOR)
        if supported is not None:
            delta = delta * supported.float()
        # Sagittal flatness gate (run-5): side/shoulder rolls don't count.
        y_z = torch.nan_to_num(_lateral_axis_z(asset.data.root_link_quat_w), nan=1.0).abs()
        t = torch.clamp((_FLAT_ZERO - y_z) / (_FLAT_ZERO - _FLAT_FULL), 0.0, 1.0)
        delta = delta * (t * t * (3.0 - 2.0 * t))
        env._roulade_accum = env._roulade_accum + delta
        env._roulade_max = torch.maximum(env._roulade_max, env._roulade_accum)

        head_contact = _sensor_any_contact(env, _ROULADE_HEAD_SENSOR)
        if head_contact is not None:
            in_window = (env._roulade_accum > _HEAD_LATCH_LO) & (env._roulade_accum < _HEAD_LATCH_HI)
            # Run-5: contact must be with the FLAT TOP of the head (top axis
            # pointing at the floor) — face/side shell contacts don't latch.
            env._roulade_head_latch = env._roulade_head_latch | (head_contact & in_window & _head_top_down(env, asset))
        env._roulade_last_update_step = step


def _roulade_completion_gate(env: ManagerBasedRlEnv, gate_lo: float, gate_hi: float, require_head: bool = False) -> torch.Tensor:
    # State-based replacement for the old phase-clock landing window — it can
    # only be opened by actually rotating (while SUPPORTED — the accumulator is
    # contact-gated), so neither pre-roll standing nor a ballistic flip collects.
    # With require_head=True the gate additionally requires the head latch —
    # the episode must have rolled over the head to unlock the landing annuity.
    _, max_accum, _ = _roulade_state(env)
    t = torch.clamp((max_accum - gate_lo) / max(gate_hi - gate_lo, 1e-6), 0.0, 1.0)
    gate = t * t * (3.0 - 2.0 * t)
    if require_head:
        gate = gate * env._roulade_head_latch.float()
    return gate


def reset_roulade_state(env: ManagerBasedRlEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, standing_prob: float = 0.5, midroll_prob: float = 0.5, standing_z_min: float = 0.11, standing_z_max: float = 0.12, standing_tilt_max: float = 0.0, forward_vel_range: tuple = (0.0, 0.0), midroll_pitch_min: float = math.radians(50.0), midroll_pitch_max: float = math.radians(185.0), midroll_z_min: float = 0.05, midroll_z_max: float = 0.10, midroll_omega_range: tuple = (0.0, 0.0), tuck_overrides: dict | None = None, tuck_factor_range: tuple = (0.3, 1.0), joint_noise_std: float = 0.0):
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.long)
    num = len(env_ids)
    asset: Entity = env.scene[asset_cfg.name]
    accum, max_accum, paid = _roulade_state(env)

    total = standing_prob + midroll_prob
    is_mid = torch.rand(num, device=env.device) < (midroll_prob / max(total, 1e-6))

    yaw = torch.rand(num, device=env.device) * 2 * np.pi - np.pi
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)

    pitch = (torch.rand(num, device=env.device) * 2 - 1) * standing_tilt_max
    mid_pitch = torch.rand(num, device=env.device) * (midroll_pitch_max - midroll_pitch_min) + midroll_pitch_min
    pitch = torch.where(is_mid, mid_pitch, pitch)
    roll = (torch.rand(num, device=env.device) * 2 - 1) * max(standing_tilt_max, math.radians(5.0))

    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    # ZYX intrinsic Euler → quaternion (yaw * pitch * roll), as in
    # set_random_ground_state.
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    quat = torch.stack([qw, qx, qy, qz], dim=1)

    z_stand = torch.rand(num, device=env.device) * (standing_z_max - standing_z_min) + standing_z_min
    z_mid = torch.rand(num, device=env.device) * (midroll_z_max - midroll_z_min) + midroll_z_min
    new_z = torch.where(is_mid, z_mid, z_stand)

    env.sim.data.qpos[env_ids, 2] = new_z
    env.sim.data.qpos[env_ids, 3:7] = quat
    env.sim.data.qvel[env_ids, :6] = 0.0

    servo_ids = _servo_joint_ids(env, asset)

    mid_env_ids = env_ids[is_mid]
    if len(mid_env_ids) > 0 and tuck_overrides:
        u = torch.rand(len(mid_env_ids), device=env.device) * (tuck_factor_range[1] - tuck_factor_range[0]) + tuck_factor_range[0]
        for jnt_idx, angle in tuck_overrides.items():
            col = 7 + servo_ids[jnt_idx]
            home = env.sim.data.qpos[mid_env_ids, col]
            env.sim.data.qpos[mid_env_ids, col] = home + u * (angle - home)
    if len(mid_env_ids) > 0 and joint_noise_std > 0.0:
        cols = torch.tensor([7 + j for j in servo_ids], device=env.device, dtype=torch.long)
        noise = torch.randn(len(mid_env_ids), len(cols), device=env.device) * joint_noise_std
        env.sim.data.qpos[mid_env_ids.unsqueeze(1), cols.unsqueeze(0)] += noise

    if len(mid_env_ids) > 0 and midroll_omega_range[1] > 0.0:
        omega = torch.rand(len(mid_env_ids), device=env.device) * (midroll_omega_range[1] - midroll_omega_range[0]) + midroll_omega_range[0]
        env.sim.data.qvel[mid_env_ids, 4] = _ROULADE_FWD_SIGN * omega

    stand_env_ids = env_ids[~is_mid]
    if len(stand_env_ids) > 0 and forward_vel_range[1] > 0.0:
        vx = torch.rand(len(stand_env_ids), device=env.device) * (forward_vel_range[1] - forward_vel_range[0]) + forward_vel_range[0]
        yaw_s = yaw[~is_mid]
        env.sim.data.qvel[stand_env_ids, 0] = vx * torch.cos(yaw_s)
        env.sim.data.qvel[stand_env_ids, 1] = vx * torch.sin(yaw_s)

    spawn_angle = torch.where(is_mid, mid_pitch, torch.zeros_like(mid_pitch))
    accum[env_ids] = spawn_angle
    max_accum[env_ids] = spawn_angle
    paid[env_ids] = spawn_angle
    env._roulade_head_latch[env_ids] = is_mid


def roulade_progress(env: ManagerBasedRlEnv, target_angle: float = 2 * math.pi, max_paid_rate: float = 3.0, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    _update_roulade_accum(env, asset)
    _, max_accum, paid = _roulade_state(env)
    new_paid = torch.clamp(max_accum, max=target_angle)
    delta = torch.clamp(new_paid - torch.clamp(paid, max=target_angle), min=0.0)
    delta = torch.clamp(delta, max=max_paid_rate * env.step_dt)
    env._roulade_paid = torch.maximum(paid, new_paid)
    return delta / (env.step_dt * target_angle)


def roulade_head_pivot(env: ManagerBasedRlEnv, sensor_name: str = "head_ground_contact", angle_lo: float = math.radians(30.0), angle_hi: float = math.radians(240.0), rate_norm: float = 2.0, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    _update_roulade_accum(env, asset)
    accum, _, _ = _roulade_state(env)

    if sensor_name not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)
    found = env.scene.sensors[sensor_name].data.found
    contact = (found.view(found.shape[0], -1) > 0).any(dim=-1).float()

    in_window = ((accum > angle_lo) & (accum < angle_hi)).float()
    omega_fwd = _ROULADE_FWD_SIGN * asset.data.root_link_ang_vel_b[:, 1]
    rate = torch.clamp(torch.nan_to_num(omega_fwd, nan=0.0) / rate_norm, 0.0, 1.0)
    top = 0.3 + 0.7 * _head_top_down(env, asset).float()
    return contact * in_window * rate * top


def roulade_landing_composite(env: ManagerBasedRlEnv, target_height: float, height_std: float, upright_std: float, pose_std: float, joint_indices: list, gate_lo: float = math.radians(260.0), gate_hi: float = math.radians(330.0), target_overrides: dict | None = None, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    _update_roulade_accum(env, asset)
    score = standing_composite_score(env, target_height=target_height, height_std=height_std, upright_std=upright_std, pose_std=pose_std, joint_indices=joint_indices, target_overrides=target_overrides, asset_cfg=asset_cfg)
    return score * _roulade_completion_gate(env, gate_lo, gate_hi, require_head=True)


def roulade_upright_after_roll(env: ManagerBasedRlEnv, gate_lo: float = math.radians(260.0), gate_hi: float = math.radians(330.0), asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    _update_roulade_accum(env, asset)
    quat = asset.data.root_link_quat_w
    upright = 1.0 - 2.0 * (quat[:, 1].pow(2) + quat[:, 2].pow(2))
    return torch.clamp(upright, min=0.0) * _roulade_completion_gate(env, gate_lo, gate_hi, require_head=True)


def roulade_height_after_roll(env: ManagerBasedRlEnv, target_height: float, std: float = 0.04, gate_lo: float = math.radians(260.0), gate_hi: float = math.radians(330.0), asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    _update_roulade_accum(env, asset)
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    g = torch.exp(-(((z - target_height) / std) ** 2))
    return g * _roulade_completion_gate(env, gate_lo, gate_hi, require_head=True)


def roulade_landing_sharp(env: ManagerBasedRlEnv, target_height: float, height_std: float = 0.015, upright_std: float = 0.3, gate_lo: float = math.radians(260.0), gate_hi: float = math.radians(330.0), asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    _update_roulade_accum(env, asset)
    quat = asset.data.root_link_quat_w
    tilt_sq = 2.0 * (quat[:, 1].pow(2) + quat[:, 2].pow(2))
    upright_g = torch.exp(-tilt_sq / (upright_std * upright_std))
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    height_g = torch.exp(-(((z - target_height) / height_std) ** 2))
    gate = _roulade_completion_gate(env, gate_lo, gate_hi, require_head=True)
    return upright_g * height_g * gate


def roulade_stand_tax(env: ManagerBasedRlEnv, target_height: float, gate_lo: float = math.radians(260.0), gate_hi: float = math.radians(330.0), asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    _update_roulade_accum(env, asset)
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    shortfall = torch.clamp(target_height - z, min=0.0)
    return -shortfall * _roulade_completion_gate(env, gate_lo, gate_hi, require_head=True)


def roulade_rise_velocity(env: ManagerBasedRlEnv, max_height: float = 0.125, gate_lo: float = math.radians(180.0), gate_hi: float = math.radians(260.0), asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    _update_roulade_accum(env, asset)
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    vz = torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0)
    reward = torch.clamp(vz, min=0.0) * (z < max_height).float()
    return reward * _roulade_completion_gate(env, gate_lo, gate_hi, require_head=True)


def roulade_overspeed_penalty(env: ManagerBasedRlEnv, omega_max: float = 4.0, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    omega_y = torch.nan_to_num(asset.data.root_link_ang_vel_b[:, 1], nan=0.0)
    excess = torch.clamp(omega_y.abs() - omega_max, min=0.0)
    return excess.pow(2)


def roulade_flatness_penalty(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return torch.nan_to_num(_lateral_axis_z(asset.data.root_link_quat_w), nan=0.0).pow(2)


def roulade_sagittal_penalty(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    omega_b = asset.data.root_link_ang_vel_b
    return torch.nan_to_num(omega_b[:, 0].pow(2) + omega_b[:, 2].pow(2), nan=0.0)


def roulade_lateral_velocity_penalty(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return torch.nan_to_num(asset.data.root_link_lin_vel_b[:, 1].pow(2), nan=0.0)


_PASSIVE_EXCLUDED_JOINTS = (r"^(?!passive_).*",)


def wire_sim2real_obs(cfg, *, gravity_term_name: str = "projected_gravity", imu_delay_max_lag: int = 1, imu_misalignment_deg: float | None = 6.0, misalign_gravity: bool = True, encoder_bias_range: tuple[float, float] | None = (-0.015, 0.015), sanitize_critic_sensors: bool = True) -> None:
    from copy import deepcopy

    from mjlab.utils.noise import UniformNoiseCfg as Unoise

    actor = cfg.observations["actor"].terms
    critic = cfg.observations["critic"].terms

    for name in (gravity_term_name, "base_ang_vel"):
        actor[name] = deepcopy(actor[name])

    for name in ("base_ang_vel", gravity_term_name):
        actor[name].delay_min_lag = 0
        actor[name].delay_max_lag = imu_delay_max_lag
        actor[name].delay_update_period = 64

    if sanitize_critic_sensors:
        for term, safe in (("foot_contact_forces", foot_contact_forces_safe), ("foot_height", foot_height_safe), ("foot_air_time", foot_air_time_safe)):
            if term in critic:
                critic[term].func = safe

    actor["base_ang_vel"].noise = Unoise(n_min=-0.03, n_max=0.03)
    actor[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)
    actor["joint_pos"].noise = Unoise(n_min=-0.001, n_max=0.001)
    actor["joint_vel"].noise = Unoise(n_min=-0.25, n_max=0.25)

    if imu_misalignment_deg is not None:
        actor["base_ang_vel"].func = base_ang_vel_imu_misaligned
        actor["base_ang_vel"].params = {"max_angle_deg": imu_misalignment_deg}
        if misalign_gravity:
            actor[gravity_term_name].func = projected_gravity_imu_misaligned
            actor[gravity_term_name].params = {"max_angle_deg": imu_misalignment_deg}

    actor["joint_vel"] = deepcopy(actor["joint_vel"])
    actor["joint_vel"].delay_min_lag = 1
    actor["joint_vel"].delay_max_lag = 1
    actor["joint_vel"].delay_update_period = 0

    passive_excluded = SceneEntityCfg("robot", joint_names=_PASSIVE_EXCLUDED_JOINTS)
    for group in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[group].terms[term] = deepcopy(cfg.observations[group].terms[term])
            cfg.observations[group].terms[term].params["asset_cfg"] = deepcopy(passive_excluded)

    if encoder_bias_range is not None:
        cfg.events["encoder_bias"].params["bias_range"] = encoder_bias_range
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)
