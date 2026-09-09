"""MDP functions for microduck tasks"""

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
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply, quat_from_angle_axis, wrap_to_pi
from rsl_rl.algorithms.ppo import PPO as _PPO

# Patch 1: mjlab computes rewards BEFORE resetting envs, so a term reading a NaN
# physics state returns NaN, which propagates: reward → advantage → loss →
# gradient → NaN/negative std → crash in torch.normal on the next mini-batch.
_orig_reward_compute = _RewardManager.compute


def _nan_safe_reward_compute(self, dt: float) -> torch.Tensor:
    result = _orig_reward_compute(self, dt)
    # compute() already updated _episode_sums; sanitize in-place or per-term
    # metrics show NaN.
    for key in self._episode_sums:
        torch.nan_to_num_(self._episode_sums[key], nan=0.0)
    return torch.nan_to_num(result, nan=0.0)


_RewardManager.compute = _nan_safe_reward_compute

# Patch 2: at a sudden curriculum weight step the value function is badly wrong,
# all TD errors shift together, std(advantages) → tiny, and (A − mean) / std →
# huge, blowing up the std gradient until the optimizer pushes std below zero.
_orig_compute_returns = _PPO.compute_returns


def _safe_compute_returns(self, obs) -> None:
    _orig_compute_returns(self, obs)
    st = self.storage
    torch.nan_to_num_(st.advantages, nan=0.0, posinf=0.0, neginf=0.0)
    torch.nan_to_num_(st.returns, nan=0.0, posinf=0.0, neginf=0.0)


_PPO.compute_returns = _safe_compute_returns

# Patch 3 (std-clamp) dropped in the mjlab 1.3.0 migration: rsl_rl 5.0.1 has no
# ActorCritic. If std-blowup recurs, reinstate it on GaussianDistribution.

print("[mdp] Patches 1-2 active: NaN-safe reward/advantage")

# Patch 4: passive joints are in the articulation but have no XML actuator, so
# the upstream exporter iterates robot.joint_names (16) and indexes
# joint_name_to_ctrl_id (14) → KeyError on passive_*. Exported metadata must
# match the 14-dim action space.
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
# The velocity task exporter already imported the symbol by value.
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
    """Entity-local indices of the servo (non-``passive_``) joints, cached.

    Every joint-index param in this module is written against the canonical
    14-servo layout, but models with ``passive_*`` joints (backlash hinges,
    roller wheels, jaw linkage) have a wider, INTERLEAVED joint array where raw
    indices select the wrong joints. Identity on plain models.
    """
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


def reset_with_forward_velocity(env: ManagerBasedRlEnv, env_ids: torch.Tensor, velocity_range: tuple[float, float] = (0.3, 0.8), fraction_stages: list[dict] | None = None, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> None:
    """Warm-start a fraction of reset envs with a random body-forward velocity.

    Decaying the fraction over training makes the robot earn the speed from rest.

    fraction_stages: [{"step": int, "fraction": float}, ...] sorted by step; the
    last stage whose step has been reached wins.
    """
    if fraction_stages is None:
        fraction_stages = [{"step": 0, "fraction": 0.8}]

    step = env.common_step_counter
    fraction = fraction_stages[0]["fraction"]
    for stage in fraction_stages:
        if step >= stage["step"]:
            fraction = stage["fraction"]

    if len(env_ids) == 0 or fraction <= 0.0:
        return

    n_warmstart = max(1, int(len(env_ids) * fraction))
    perm = torch.randperm(len(env_ids), device=env.device)[:n_warmstart]
    warmstart_ids = env_ids[perm]

    lo, hi = velocity_range
    vx = lo + torch.rand(n_warmstart, device=env.device) * (hi - lo)

    # Read the quaternion from qpos, NOT root_link_quat_w: xquat needs a current
    # sim.forward(), so after reset_base writes a new yaw it is still stale from
    # the old episode. write_root_pose updates qpos immediately.
    asset: Entity = env.scene[asset_cfg.name]
    qpos_q_adr = asset.data.indexing.free_joint_q_adr[3:7]
    q = asset.data.data.qpos[warmstart_ids][:, qpos_q_adr]
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    forward_world = torch.stack([torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)], dim=-1)

    velocities = torch.zeros(n_warmstart, 6, device=env.device)
    velocities[:, :3] = vx.unsqueeze(-1) * forward_world

    asset.write_root_link_velocity_to_sim(velocities, env_ids=warmstart_ids)

    # Spin wheels to match, else no-slip contact brakes the spawn instantly.
    # Radius measured; all 4 wheels take +ω for forward (test_wheel_direction.py).
    _WHEEL_RADIUS = 0.0175
    all_wheel_ids, _ = asset.find_joints(r"^passive_.*")

    if all_wheel_ids:
        joint_pos = asset.data.joint_pos[warmstart_ids].clone()
        joint_vel = asset.data.joint_vel[warmstart_ids].clone()
        omega = vx / _WHEEL_RADIUS
        joint_vel[:, all_wheel_ids] = omega.unsqueeze(-1).expand(-1, len(all_wheel_ids))
        asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=warmstart_ids)


def reset_action_history(env: ManagerBasedRlEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG):
    """Reset cached action history so action-rate/acceleration penalties do not
    charge the first step of a new episode for the last step of the old one."""
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


def joint_accelerations_l2(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Finite-difference joint acceleration cost (≥ 0 → negative weight)."""
    asset: Entity = env.scene[asset_cfg.name]

    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]

    if not hasattr(asset.data, "_prev_joint_vel"):
        asset.data._prev_joint_vel = joint_vel.clone()
        return torch.zeros(env.num_envs, device=env.device)

    dt = env.step_dt
    joint_acc = (joint_vel - asset.data._prev_joint_vel) / dt

    asset.data._prev_joint_vel = joint_vel.clone()

    return torch.sum(torch.square(joint_acc), dim=1)


def leg_action_rate_l2(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Leg action-rate cost (≥ 0 → negative weight)."""
    leg_joint_indices = list(range(5)) + list(range(9, 14))

    if not hasattr(env, "action_manager"):
        return torch.zeros(env.num_envs, device=env.device)

    actions = env.action_manager.action
    if actions.shape[1] < 14:
        return torch.zeros(env.num_envs, device=env.device)

    leg_actions = actions[:, leg_joint_indices]

    if not hasattr(env, "_prev_leg_actions"):
        env._prev_leg_actions = leg_actions.clone()
        return torch.zeros(env.num_envs, device=env.device)

    action_rate = leg_actions - env._prev_leg_actions
    env._prev_leg_actions = leg_actions.clone()

    return torch.sum(torch.square(action_rate), dim=1)


def neck_action_rate_l2(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Neck action-rate cost (≥ 0 → negative weight)."""
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


def leg_action_acceleration_l2(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Leg action-acceleration cost (≥ 0 → negative weight)."""
    leg_joint_indices = list(range(5)) + list(range(9, 14))

    if not hasattr(env, "action_manager"):
        return torch.zeros(env.num_envs, device=env.device)

    actions = env.action_manager.action
    if actions.shape[1] < 14:
        return torch.zeros(env.num_envs, device=env.device)

    leg_actions = actions[:, leg_joint_indices]

    if not hasattr(env, "_prev_leg_actions_for_acc"):
        env._prev_leg_actions_for_acc = leg_actions.clone()
        env._prev_prev_leg_actions_for_acc = leg_actions.clone()
        return torch.zeros(env.num_envs, device=env.device)

    action_acc = leg_actions - 2 * env._prev_leg_actions_for_acc + env._prev_prev_leg_actions_for_acc

    env._prev_prev_leg_actions_for_acc = env._prev_leg_actions_for_acc.clone()
    env._prev_leg_actions_for_acc = leg_actions.clone()

    return torch.sum(torch.square(action_acc), dim=1)


def neck_action_acceleration_l2(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Neck action-acceleration cost (≥ 0 → negative weight)."""
    neck_joint_indices = list(range(5, 9))

    if not hasattr(env, "action_manager"):
        return torch.zeros(env.num_envs, device=env.device)

    actions = env.action_manager.action
    if actions.shape[1] < 14:
        return torch.zeros(env.num_envs, device=env.device)

    neck_actions = actions[:, neck_joint_indices]

    if not hasattr(env, "_prev_neck_actions_for_acc"):
        env._prev_neck_actions_for_acc = neck_actions.clone()
        env._prev_prev_neck_actions_for_acc = neck_actions.clone()
        return torch.zeros(env.num_envs, device=env.device)

    action_acc = neck_actions - 2 * env._prev_neck_actions_for_acc + env._prev_prev_neck_actions_for_acc

    env._prev_prev_neck_actions_for_acc = env._prev_neck_actions_for_acc.clone()
    env._prev_neck_actions_for_acc = neck_actions.clone()

    return torch.sum(torch.square(action_acc), dim=1)


def _fallen_mask(env: ManagerBasedRlEnv, asset, gate_z_below: float, gate_tilt_above_deg: float) -> torch.Tensor:
    """1.0 where FALLEN (z < gate_z_below OR tilt > gate_tilt_above_deg).

    Gates recovery rewards to exactly zero during clean walking: no walk tax,
    no bounce farming.
    """
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    quat = asset.data.root_link_quat_w
    # cos(tilt) = R22
    cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    fallen = (z < gate_z_below) | (cos_tilt < math.cos(math.radians(gate_tilt_above_deg)))
    return fallen.float()


def feet_air_time_upright(env: ManagerBasedRlEnv, gate_tilt_above_deg: float = 40.0, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, **air_time_kwargs) -> torch.Tensor:
    """velocity template feet_air_time, zeroed while FALLEN (tilt > gate).

    A robot lying on its trunk can otherwise farm the air-time window by
    rhythmically tapping a leg (observed in velstand).
    """
    from mjlab.tasks.velocity.mdp import feet_air_time as _template_air_time

    reward = _template_air_time(env, **air_time_kwargs)
    asset: Entity = env.scene[asset_cfg.name]
    upright = 1.0 - _fallen_mask(env, asset, 0.0, gate_tilt_above_deg)
    return reward * upright


def upright_progress(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Potential-based upright shaping: Δcos(tilt) per step.

    Holding any pose pays exactly zero, so nothing can farm it — the gated
    state-reward this replaces was farmed from sitting, lying flat, and a
    head-tripod lean. A full prone→stand recovery collects Δ≈+1 total.
    """
    asset: Entity = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w
    cos_tilt = torch.nan_to_num(1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2), nan=1.0)
    if not hasattr(env, "_upright_potential_prev"):
        env._upright_potential_prev = cos_tilt.clone()
    # Else the first step charges the delta against the old episode's pose.
    fresh = env.episode_length_buf <= 1
    env._upright_potential_prev[fresh] = cos_tilt[fresh]
    delta = cos_tilt - env._upright_potential_prev
    env._upright_potential_prev = cos_tilt.clone()
    return delta


def height_progress(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, ceiling: float = 0.115) -> torch.Tensor:
    """Potential-based height shaping: Δ min(trunk z, ceiling) per step.

    z-axis companion to ``upright_progress``: the last mile of a recovery
    (knees extending out of a deep crouch) is mostly height at modest tilt,
    where the Gaussian upright/pose rewards are flat and Δcos(tilt) is tiny.
    ``ceiling`` sits just below full-stand trunk z so hopping pays nothing.
    """
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
    """1.0 while FALLEN — WEIGHT IT NEGATIVE. Per-step tax on staying down.

    Without it, lying still costs ~0/step while attempting recovery pays
    action-rate/torque penalties, so waiting for the fallen_too_long recycle is
    rational. (Penalties on bad states are safe; positive rewards gated on bad
    states get farmed.)

    ``release_*`` adds hysteresis: the tax keeps paying until genuinely up
    (tilt < release_tilt AND z > release_z), not merely under the arming gate —
    a crouch just below the gate was otherwise a zero-cost rest state that
    recoveries parked in instead of finishing the stand. Arms only on a real
    fall, so gait tilt wobble is never taxed.
    """
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
    """One-shot bounty on a COMPLETED recovery (fallen ≥ min_fallen_s → upright).

    Re-arms only by falling again, so oscillating around the gate pays nothing.
    """
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
    """Body-Z world z-component: +1 upright, 0 horizontal, -1 inverted.

    Unlike the Gaussian flat_orientation, gradient is non-zero everywhere, so a
    prone robot still has a signal to rotate toward upright.
    """
    asset: Entity = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w
    qx = quat[:, 1]
    qy = quat[:, 2]
    reward = 1.0 - 2.0 * (qx * qx + qy * qy)
    if gate_z_below is not None:
        # Zero during clean walking, so it can't dilute the tracking rewards.
        reward = reward * _fallen_mask(env, asset, gate_z_below, gate_tilt_above_deg)
    return reward


def body_upright_gaussian(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, std: float = 0.1) -> torch.Tensor:
    """Gaussian on tilt magnitude — sharp pull toward fully vertical.

    Complements ``body_upright_linear``, whose gradient ``sin(tilt)`` vanishes
    exactly at the target.
    """
    asset: Entity = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w
    qx = quat[:, 1]
    qy = quat[:, 2]
    tilt_sq = 2.0 * (qx * qx + qy * qy)  # = 1 − cos(tilt); small-angle tilt²/2
    return torch.exp(-tilt_sq / (std * std))


def upright_gaussian_at_height(env: ManagerBasedRlEnv, std: float, height_low: float, height_high: float, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """``body_upright_gaussian`` smoothstep-gated on trunk z.

    Ungated, the policy finds a "crouch low and vertical" optimum that collects
    upright reward without ever rising.
    """
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
    """Trunk ``sum(ω_xy²)`` arrival damper, gated by trunk z and optionally tilt.

    Returns a POSITIVE cost → use a negative weight. Zero below ``height_low``
    so ground recovery (flips/rolls need large trunk rotation) stays free.

    ``tilt_full_deg`` is strongly recommended: with a height gate alone, the
    final straighten of a bent-over rise happens INSIDE the z gate and is
    itself a large rotation, so taxing it builds a reward wall right before the
    finish and the policy parks bent-over below the gate. Gating on tilt leaves
    the approach to vertical free and damps only wobble AROUND vertical.
    """
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
    """Product of height/upright/pose Gaussians, each ∈ [0, 1].

    Multiplying means a deficiency in any one factor collapses the reward, so
    the policy cannot claim 80% by being perfect on 2-of-3. Breaks the
    compromise basins that additive stacks leave open (e.g. "lean trunk at the
    right height").
    """
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


def standing_success_bonus(env: ManagerBasedRlEnv, target_height: float, height_tol: float, upright_threshold: float, pose_tol: float, joint_indices: list, target_overrides: dict | None = None, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """1.0 iff height, uprightness AND pose are all within tolerance.

    Nearby compromises (lean to balance a head-forward CoM, park 1 cm short of
    target z) collect partial credit from the dense terms but ZERO bonus.
    """
    asset = env.scene[asset_cfg.name]

    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    height_ok = (z - target_height).abs() <= height_tol

    quat = asset.data.root_link_quat_w
    qx = quat[:, 1]
    qy = quat[:, 2]
    upright = 1.0 - 2.0 * (qx * qx + qy * qy)
    upright_ok = upright >= upright_threshold

    target = _servo_default_joint_pos(env, asset).clone()
    if target_overrides:
        for idx, val in target_overrides.items():
            target[:, idx] = val
    joint_pos = _servo_joint_pos(env, asset)[:, joint_indices]
    target = target[:, joint_indices]
    pose_err = (joint_pos - target).abs().max(dim=-1).values
    pose_ok = pose_err <= pose_tol

    return (height_ok & upright_ok & pose_ok).float()


def com_upward_velocity(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, max_height: float = 0.08, gate_z_below: float | None = None, gate_tilt_above_deg: float = 40.0, max_vz: float | None = None) -> torch.Tensor:
    """Reward upward CoM velocity, zero above ``max_height`` so a standing robot
    cannot farm it by squatting and re-rising.

    ``max_vz`` caps the rewarded velocity: uncapped, reward ∝ vz pays more for
    an explosive launch. Capped, every rise ≥ max_vz earns the same and the
    |a_z| penalty picks the smooth one.
    """
    asset: Entity = env.scene[asset_cfg.name]
    # MuJoCo can produce NaN on contact instability; treat as z=0.
    com_z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    vz = torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0)
    below_target = (com_z < max_height).float()
    reward = torch.clamp(vz, min=0.0, max=max_vz) * below_target
    if gate_z_below is not None:
        # Ungated, gait dip-and-rise across max_height pays → bounce incentive.
        reward = reward * _fallen_mask(env, asset, gate_z_below, gate_tilt_above_deg)
    return reward


def fallen_too_long(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, gate_z_below: float = 0.10, gate_tilt_above_deg: float = 40.0, max_duration_s: float = 5.0) -> torch.Tensor:
    """Terminate envs continuously FALLEN for `max_duration_s`.

    Envs that mix walking with recovery disable the fell_over termination by
    curriculum; without this backstop a failed recovery farms recovery reward
    for the whole episode and starves the walk of data (measured: ~25% walking).
    """
    asset: Entity = env.scene[asset_cfg.name]
    fallen = _fallen_mask(env, asset, gate_z_below, gate_tilt_above_deg).bool()
    if not hasattr(env, "_fallen_timer_s"):
        env._fallen_timer_s = torch.zeros(env.num_envs, device=env.device)
    env._fallen_timer_s[env.episode_length_buf <= 1] = 0.0
    env._fallen_timer_s = torch.where(fallen, env._fallen_timer_s + env.step_dt, torch.zeros_like(env._fallen_timer_s))
    return env._fallen_timer_s >= max_duration_s


def robot_state_is_nan(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, sensor_names: tuple[str, ...] = ()) -> torch.Tensor:
    """Terminate envs where MuJoCo's contact solver diverged to non-finite state.

    Must cover the WHOLE physical state, not just joint_pos: divergence usually
    blows up the base free joint or the passive wheels, which feed critic obs
    (base_lin_vel, base_ang_vel, projected_gravity, wheel_vel). Unmonitored,
    the env is not reset and the NaN reaches rsl_rl's check_nan, killing the
    run. Tests non-finiteness because inf becomes NaN downstream when
    projected_gravity is normalized.

    The reward at this terminal step can still be NaN (mjlab computes rewards
    before resets), but done=True stops it propagating back through GAE.
    """
    asset: Entity = env.scene[asset_cfg.name]
    d = asset.data
    bad = ~torch.isfinite(d.joint_pos).all(dim=1)
    bad |= ~torch.isfinite(d.joint_vel).all(dim=1)
    bad |= ~torch.isfinite(d.root_link_pos_w).all(dim=1)
    bad |= ~torch.isfinite(d.root_link_quat_w).all(dim=1)
    bad |= ~torch.isfinite(d.root_link_lin_vel_w).all(dim=1)
    bad |= ~torch.isfinite(d.root_link_ang_vel_w).all(dim=1)

    # Contact forces can blow up a step before qpos/qvel do: a degenerate
    # contact resolves to an inf/NaN impulse while the integrated state is still
    # finite. That force feeds the critic-only `foot_contact_forces` obs, which
    # the state checks above do not cover, and killed a run.
    for name in sensor_names:
        if name not in env.scene.sensors:
            continue
        force = getattr(env.scene.sensors[name].data, "force", None)
        if force is not None:
            bad |= ~torch.isfinite(force).flatten(start_dim=1).all(dim=1)
    return bad


def root_height_below(env: ManagerBasedRlEnv, min_height: float, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Terminate when the trunk drops below ``min_height`` in world z.

    roller_slope's "fell into the void": set min_height below the ramp's exit
    flat, so only leaving solid ground triggers it. Independent of ramp
    length/slope.
    """
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w[:, 2] < min_height


def descent_speed_reward(env: ManagerBasedRlEnv, cap: float = 0.8, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Forward speed DOWN the slope (the ramp descends in world +x), capped.

    Without it the optimum is to brake and stand still; the cap stops the
    policy hurtling down ever faster.
    """
    asset: Entity = env.scene[asset_cfg.name]
    vx = torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 0], nan=0.0, posinf=0.0, neginf=0.0)
    return torch.clamp(vx, min=0.0, max=cap)


def reset_rolling_entry(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, speed_range: tuple = (0.25, 0.45), wheel_radius: float = 0.0175, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> None:
    """Rolling start with ω·r = v, so there is zero slip at contact.

    A base-only push (moving base, stationary wheels) skids brutally on the
    first step. Run AFTER reset_base, and leave reset_base's velocity_range
    unset.
    """
    asset: Entity = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n = int(env_ids.shape[0])
    lo, hi = speed_range
    v = torch.rand(n, device=env.device) * (hi - lo) + lo

    root_vel = torch.zeros(n, 6, device=env.device)
    root_vel[:, 0] = v
    asset.write_root_link_velocity_to_sim(root_vel, env_ids=env_ids)

    # Positive ω = forward, cf. wheel_speed.
    wheel_ids = []
    for name in ("passive_LF_?wheel", "passive_LR_?wheel", "passive_RF_?wheel", "passive_RR_?wheel"):
        ids, _ = asset.find_joints(name)
        wheel_ids.append(ids[0])
    wheel_ids_t = torch.tensor(wheel_ids, device=env.device)
    omega = (v / wheel_radius).unsqueeze(1).repeat(1, len(wheel_ids))
    asset.write_joint_velocity_to_sim(omega, joint_ids=wheel_ids_t, env_ids=env_ids)


def wheel_glide_reward(env: ManagerBasedRlEnv, cap_speed: float = 0.35, wheel_radius: float = 0.0175) -> torch.Tensor:
    """Rolling speed of the passive wheels, capped and clamped at zero.

    Unlike descent_speed (base velocity, reachable by running/pushing), wheel
    rotation measures true glide.
    """
    asset: Entity = env.scene["robot"]
    lf, _ = asset.find_joints("passive_LF_?wheel")
    lr, _ = asset.find_joints("passive_LR_?wheel")
    rf, _ = asset.find_joints("passive_RF_?wheel")
    rr, _ = asset.find_joints("passive_RR_?wheel")
    vel = asset.data.joint_vel
    # All 4 wheels spin positive for forward, cf. wheel_speed_reward.
    omega = (vel[:, lf[0]] + vel[:, lr[0]] + vel[:, rf[0]] + vel[:, rr[0]]) / 4.0
    speed = torch.nan_to_num(omega * wheel_radius, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.clamp(speed, min=0.0, max=cap_speed)


def is_alive(env: ManagerBasedRlEnv) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


def com_height_target(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, target_height_min: float = 0.1, target_height_max: float = 0.15) -> torch.Tensor:
    """+1 inside the CoM height band, −(distance to band)² outside."""
    asset: Entity = env.scene[asset_cfg.name]

    # Height above the terrain spawn origin. NaN (contact instability) → z=0,
    # which keeps the penalty finite and small since 0 is near the band.
    com_height = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)

    below_min = com_height < target_height_min
    above_max = com_height > target_height_max
    in_range = ~(below_min | above_max)

    penalty_below = torch.square(com_height - target_height_min) * below_min.float()
    penalty_above = torch.square(com_height - target_height_max) * above_max.float()

    reward = in_range.float() - (penalty_below + penalty_above)

    return reward


def crouch_height_target(phase: torch.Tensor, height_low: float, height_high: float, hold_lo: float = 0.375, hold_hi: float = 0.625) -> torch.Tensor:
    """Trapezoid trunk-height target over phase ∈ [0, 1):

    descend high→low, hold low (the crouched glide), rise low→high.
    """
    descend = phase < hold_lo
    hold = (phase >= hold_lo) & (phase < hold_hi)

    frac_d = phase / hold_lo
    t_descend = height_high + (height_low - height_high) * frac_d

    t_hold = torch.full_like(phase, height_low)

    frac_r = (phase - hold_hi) / (1.0 - hold_hi)
    t_rise = height_low + (height_high - height_low) * frac_r

    return torch.where(descend, t_descend, torch.where(hold, t_hold, t_rise))


def crouch_glide_reward_from_values(com_height: torch.Tensor, cmd_cos: torch.Tensor, cmd_sin: torch.Tensor, height_low: float, height_high: float, hold_lo: float = 0.375, hold_hi: float = 0.625, std: float = 0.02) -> torch.Tensor:
    """Gaussian on the trapezoid height target, phase decoded from [cos, sin]."""
    phase = (torch.atan2(cmd_sin, cmd_cos) / (2 * torch.pi)) % 1.0
    target = crouch_height_target(phase, height_low, height_high, hold_lo, hold_hi)
    return torch.exp(-(((com_height - target) / std) ** 2))


def crouch_glide_height_by_phase(env: ManagerBasedRlEnv, command_name: str = "twist", height_low: float = 0.075, height_high: float = 0.11, hold_lo: float = 0.375, hold_hi: float = 0.625, std: float = 0.02, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Main reward: trunk height tracking the phase trapezoid.

    Phase comes from the GroundPick command.
    """
    asset: Entity = env.scene[asset_cfg.name]
    com_height = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    cmd = env.command_manager.get_command(command_name)
    return crouch_glide_reward_from_values(com_height, cmd[:, 0], cmd[:, 1], height_low, height_high, hold_lo, hold_hi, std)


def forward_speed_reward(env: ManagerBasedRlEnv, vel_ref: float = 0.2, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Trunk forward speed, saturating — keep momentum, don't brake.

    Command-independent: the command carries the phase, not a speed.
    """
    asset: Entity = env.scene[asset_cfg.name]
    vx = asset.data.root_link_lin_vel_b[:, 0]
    return torch.tanh(torch.clamp(vx, min=0.0) / vel_ref)


def crouch_pose_blend(phase: torch.Tensor, descent_end: float, hold_end: float, rise_end: float) -> torch.Tensor:
    """Phase → blend in [0, 1]: 0 = standing pose, 1 = crouched pose."""
    b = torch.zeros_like(phase)
    descend = phase < descent_end
    b = torch.where(descend, phase / descent_end, b)
    low = (phase >= descent_end) & (phase < hold_end)
    b = torch.where(low, torch.ones_like(phase), b)
    rise = (phase >= hold_end) & (phase < rise_end)
    b = torch.where(rise, 1.0 - (phase - hold_end) / (rise_end - hold_end), b)
    return b


def _crouch_pose_error(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, command_name: str, crouch_pose: dict, descent_end: float, hold_end: float, rise_end: float, stand_pose: dict | None = None):
    """(cur, target) joints for the phase-interpolated crouch pose.

    STAND is `stand_pose` where given, else the model default (HOME). Joints
    resolve BY NAME so interleaved passive wheels never shift an index.
    """
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
    """Gaussian match to the phase-interpolated joint pose (stand <-> crouch).

    Symmetric by construction: standing back up pays exactly like crouching.
    """
    cur, target = _crouch_pose_error(env, asset_cfg, command_name, crouch_pose or {}, descent_end, hold_end, rise_end, stand_pose)
    return torch.exp(-(((cur - target) / std) ** 2)).mean(dim=-1)


def crouch_glide_pose_l1(env: ManagerBasedRlEnv, command_name: str = "twist", crouch_pose: dict | None = None, stand_pose: dict | None = None, descent_end: float = 0.10, hold_end: float = 0.50, rise_end: float = 0.60, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """L1 bootstrap toward the phase-interpolated crouch pose. SELF-NEGATING
    (≤ 0) → POSITIVE weight.

    Constant gradient everywhere, so there is still a direction where the
    Gaussian above has saturated to ~0.
    """
    cur, target = _crouch_pose_error(env, asset_cfg, command_name, crouch_pose or {}, descent_end, hold_end, rise_end, stand_pose)
    return -(cur - target).abs().mean(dim=-1)


def crouch_forward_lean(env: ManagerBasedRlEnv, command_name: str = "twist", target_pitch: float = 0.08, std: float = 0.1, descent_end: float = 0.10, hold_end: float = 0.50, rise_end: float = 0.60, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=("trunk_base",))) -> torch.Tensor:
    """Slight forward trunk lean during the crouch, gated by the crouch blend.

    Counters the backward tipping that fast hip flexion induces. Pitch proxy =
    projected_gravity_b[:,0], positive = forward (verified).
    """
    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    phase = (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * torch.pi)) % 1.0
    gate = crouch_pose_blend(phase, descent_end, hold_end, rise_end)
    lean = asset.data.projected_gravity_b[:, 0]
    return gate * torch.exp(-((lean - target_pitch) ** 2) / std**2)


def neck_joint_vel_l2(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Neck joint velocity cost (≥ 0 → negative weight)."""
    asset: Entity = env.scene[asset_cfg.name]

    neck_joint_indices = list(range(5, 9))
    joint_vel = _servo_joint_vel(env, asset)
    neck_joint_vel = joint_vel[:, neck_joint_indices]

    return torch.sum(torch.square(neck_joint_vel), dim=1)


def leg_joint_vel_l2(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Leg joint velocity cost (≥ 0 → negative weight)."""
    asset: Entity = env.scene[asset_cfg.name]

    leg_joint_indices = list(range(5)) + list(range(9, 14))
    joint_vel = _servo_joint_vel(env, asset)
    leg_joint_vel = joint_vel[:, leg_joint_indices]

    return torch.sum(torch.square(leg_joint_vel), dim=1)


_NECK_JOINT_CFG = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*(neck|head).*",))
_HIP_PITCH_KNEE_CFG = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*(hip_pitch|knee).*",))
_ROLLER_FEET_SITE_CFG = SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))


def feet_flat_penalty(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROLLER_FEET_SITE_CFG, sensor_name: str | None = None) -> torch.Tensor:
    """Cost for foot sites not parallel to the ground (≥ 0 → negative weight).

    Gravity projected into each foot site frame; flat → [0,0,-1], so the xy²
    magnitude measures tilt. ≈ 2.0 per fully-sideways foot.

    ``sensor_name`` gates each foot on its OWN ground contact, leaving the swing
    foot free to tilt. Ungated, the penalty is minimised by keeping both blades
    flat on the ground — i.e. the swizzle — and punishes the foot lift a stride
    needs. Assumes site order (left, right) matches the sensor slot order.

    Gravity must be normalized PER ENV (dim=-1): a dimensionless torch.norm()
    reduces over envs too, shrinking the vector ~1/sqrt(num_envs).
    """
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


def feet_tiptoe_alignment(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROLLER_FEET_SITE_CFG, command_name: str = "twist", command_threshold: float = 0.01) -> torch.Tensor:
    """Reward each foot site's local x-axis pointing down — tiptoe stance.

    Toe-down rotates the site x-axis toward world −Z; alignment ∈ [−2, 2] over
    both feet. Gated on |vel_cmd_xy| so tiptoes are not required at rest.
    Mutually exclusive with feet_flat_penalty — the two fight.
    """
    asset: Entity = env.scene[asset_cfg.name]
    quats = asset.data.site_quat_w[:, asset_cfg.site_ids, :]
    w, qx, qy, qz = quats[:, :, 0], quats[:, :, 1], quats[:, :, 2], quats[:, :, 3]
    x_axis_z = 2.0 * (qx * qz - w * qy)
    alignment = (-x_axis_z).sum(dim=-1)

    cmd = env.command_manager.get_command(command_name)
    cmd_mag = torch.linalg.norm(cmd[:, :2], dim=1)
    active = (cmd_mag > command_threshold).float()
    return alignment * active


def hip_pitch_knee_vel_l2(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _HIP_PITCH_KNEE_CFG) -> torch.Tensor:
    """Sagittal leg-joint velocity cost (≥ 0 → negative weight).

    Suppresses the walking oscillation without blocking static balance;
    skating instead uses hip_roll laterally.
    """
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def neck_joint_pos_l2(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _NECK_JOINT_CFG, pattern: str = r".*(neck|head).*") -> torch.Tensor:
    """Neck/head deviation-from-default cost (≥ 0 → negative weight).

    Resolves joints every call: the same SceneEntityCfg singleton is reused
    across robots whose passive wheels shift the neck indices.

    The spin task passes a pattern EXCLUDING `head_yaw` so the head can serve
    as a flywheel when launching the rotation.
    """
    asset: Entity = env.scene[asset_cfg.name]
    # Backlash hinges also contain "neck"/"head".
    if not pattern.startswith(r"^(?!passive_)"):
        pattern = r"^(?!passive_)" + pattern.lstrip("^")
    joint_ids, _ = asset.find_joints(pattern)
    error = asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]
    return torch.sum(torch.square(error), dim=1)


def joint_torques_l2(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Actuator force cost (≥ 0 → negative weight)."""
    asset: Entity = env.scene[asset_cfg.name]
    actuator_forces = asset.data.actuator_force
    return torch.sum(torch.square(actuator_forces), dim=1)


def joint_torque_rate_l2(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Torque-rate cost (≥ 0 → negative weight): a gearbox-shock proxy.

    Torque spikes when actuators resist a landing impulse; taxing the rate buys
    soft landings.
    """
    asset: Entity = env.scene[asset_cfg.name]
    current = asset.data.actuator_force

    if not hasattr(env, "_prev_actuator_forces"):
        env._prev_actuator_forces = current.clone()
        return torch.zeros(env.num_envs, device=env.device)

    rate = current - env._prev_actuator_forces
    env._prev_actuator_forces = current.clone()
    return torch.sum(torch.square(rate), dim=1)


def feet_grounded_reward(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """Fraction of the two feet touching the ground ∈ {0, 0.5, 1.0}."""
    if sensor_name not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)
    sensor = env.scene.sensors[sensor_name]
    found = sensor.data.found
    if found.dim() > 1:
        found = found.sum(dim=-1)
    return torch.clamp(found, 0.0, 2.0) / 2.0


def body_impact_cost(env: ManagerBasedRlEnv, sensor_name: str, threshold: float = 1.0) -> torch.Tensor:
    """Contact force above ``threshold`` on protected bodies (≥ 0 → negative
    weight): discourages slamming the trunk shell or head into the ground.

    ``sensor_name`` must be a ContactSensorCfg with fields=("force",),
    reduce="netforce".
    """
    if sensor_name not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)

    sensor = env.scene.sensors[sensor_name]
    forces = sensor.data.force
    total_force = forces.sum(dim=1)
    force_mag = torch.norm(total_force, dim=1)
    return torch.clamp(force_mag - threshold, min=0.0)


def wheel_speed_reward(env: ManagerBasedRlEnv, command_name: str, wheel_radius: float = 0.0175, vel_scale: float = 0.5, bidirectional: bool = False) -> torch.Tensor:
    """Reward wheel spin proportional to commanded push, tanh-saturated.

    All 4 wheels spin positive for forward motion. ``bidirectional=False``
    leaves cmd_x < 0 to ``braking_reward``; True makes it mean "go backward".
    """
    cmd_x = env.command_manager.get_command(command_name)[:, 0]

    asset: Entity = env.scene["robot"]
    lf_ids, _ = asset.find_joints("passive_LF_?wheel")
    lr_ids, _ = asset.find_joints("passive_LR_?wheel")
    rf_ids, _ = asset.find_joints("passive_RF_?wheel")
    rr_ids, _ = asset.find_joints("passive_RR_?wheel")

    vel = asset.data.joint_vel
    forward_omega = (vel[:, lf_ids[0]] + vel[:, lr_ids[0]] + vel[:, rf_ids[0]] + vel[:, rr_ids[0]]) / 4.0

    omega_scale = vel_scale / wheel_radius
    if bidirectional:
        aligned = torch.sign(cmd_x) * forward_omega
        return torch.abs(cmd_x) * torch.tanh(torch.clamp(aligned, min=0.0) / omega_scale)
    return torch.clamp(cmd_x, min=0.0) * torch.tanh(torch.clamp(forward_omega, min=0.0) / omega_scale)


def coasting_reward(env: ManagerBasedRlEnv, command_name: str, vel_std: float = 0.3, stillness_std: float = 5.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=(r".*(hip|knee|ankle).*",))) -> torch.Tensor:
    """Reward coasting: at target speed AND legs still, multiplicatively — so
    stomping at speed collapses the reward instead of half-earning it.
    """
    cmd = env.command_manager.get_command(command_name)
    vel_b = env.scene["robot"].data.root_link_lin_vel_b[:, :2]
    vel_error = torch.sum(torch.square(cmd[:, :2] - vel_b), dim=1)
    at_speed = torch.exp(-vel_error / vel_std**2)

    asset: Entity = env.scene[asset_cfg.name]
    joint_vel_sq = torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
    stillness = torch.exp(-joint_vel_sq / stillness_std**2)

    return at_speed * stillness


def braking_reward(env: ManagerBasedRlEnv, command_name: str, vel_std: float = 0.3) -> torch.Tensor:
    """Reward stopping when cmd_x < 0, silent otherwise.

    vel_std ≈ 0.3 m/s keeps a usable gradient down to walking-pace speeds.
    """
    cmd = env.command_manager.get_command(command_name)
    cmd_x = cmd[:, 0]
    braking_strength = torch.clamp(-cmd_x, min=0.0)
    fwd_vel = env.scene["robot"].data.root_link_lin_vel_b[:, 0]
    stopped = torch.exp(-(fwd_vel.clamp(min=0.0) ** 2) / (vel_std**2))
    return braking_strength * stopped


def contact_frequency_penalty(env: ManagerBasedRlEnv, sensor_name: str = "feet_ground_contact", max_contact_changes_per_sec: float = 4.0, command_threshold: float = 0.01) -> torch.Tensor:
    """Quadratic tax on contact changes above ``max_contact_changes_per_sec``,
    for slower stepping. SELF-NEGATING (≤ 0) → POSITIVE weight.
    """
    if sensor_name not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)

    if "twist" in env.command_manager._terms:
        cmd = env.command_manager.get_command("twist")
        cmd_vel = cmd[:, :3]
        cmd_norm = torch.linalg.norm(cmd_vel, dim=1)
        active_mask = cmd_norm > command_threshold
    else:
        active_mask = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)

    sensor = env.scene.sensors[sensor_name]
    contacts = sensor.data.found[:, :2]

    if not hasattr(env, "_contact_change_count"):
        env._contact_change_count = torch.zeros(env.num_envs, device=env.device)
        env._contact_change_timer = torch.zeros(env.num_envs, device=env.device)
        env._prev_contacts_for_freq = contacts.clone()
        return torch.zeros(env.num_envs, device=env.device)

    contact_changed = torch.any(contacts != env._prev_contacts_for_freq, dim=1)

    env._contact_change_count += contact_changed.float()
    env._contact_change_timer += env.step_dt

    freq = env._contact_change_count / torch.clamp(env._contact_change_timer, min=0.01)

    reset_mask = env._contact_change_timer >= 1.0
    env._contact_change_count[reset_mask] = 0.0
    env._contact_change_timer[reset_mask] = 0.0

    excess_freq = torch.clamp(freq - max_contact_changes_per_sec, min=0.0)
    penalty = -torch.square(excess_freq)

    env._prev_contacts_for_freq = contacts.clone()

    penalty = penalty * active_mask.float()

    return penalty


# ── Ground pick ──────────────────────────────────────────────────────────────


def mouth_ground_proximity(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=["mouth_tip"]), std: float = 0.03, target_height: float = 0.0, command_name: str = "twist") -> torch.Tensor:
    """Gaussian on mouth-tip height, weighted by the approach phase.

    The ground-pick command is [cos(2π·phase), sin(2π·phase), 0], so
    max(0, sin) selects the descent half-cycle and peaks at phase 0.25.
    """
    asset = env.scene[asset_cfg.name]
    mouth_z = asset.data.site_pos_w[:, asset_cfg.site_ids[0], 2]
    proximity = torch.exp(-(((mouth_z - target_height) / std) ** 2))

    cmd = env.command_manager.get_command(command_name)
    approach_weight = torch.clamp(cmd[:, 1], min=0.0)

    return approach_weight * proximity


def mouth_perpendicular_to_ground(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=["mouth_tip"]), command_name: str = "twist") -> torch.Tensor:
    """Mouth-tip x-axis pointing down (+1) vs up (−1), weighted by the approach
    phase so it only applies during the descent.
    """
    asset = env.scene[asset_cfg.name]
    q = asset.data.site_quat_w[:, asset_cfg.site_ids[0], :]
    w, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    x_axis_z = 2.0 * (qx * qz - w * qy)
    alignment = -x_axis_z

    cmd = env.command_manager.get_command(command_name)
    approach_weight = torch.clamp(cmd[:, 1], min=0.0)

    return approach_weight * alignment


def sit_grounded(env: ManagerBasedRlEnv, sensor_name: str, command_name: str | None = None, sin_threshold: float = 0.7, min_progress_frac: float = 0.0, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, upright_cos_threshold: float = 0.5) -> torch.Tensor:
    """Positive reward for trunk-ground contact WHILE upright.

    The upright gate is mandatory: without it the policy earns the contact bonus
    by tipping sideways or face-forward and converges to a "fallen" mode that
    competes with the real sit pose.

    ``command_name`` gates to a phase command's sit window; otherwise always-on,
    optionally restricted to the late episode via ``min_progress_frac``.
    """
    if sensor_name not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)
    sensor = env.scene.sensors[sensor_name]
    found = sensor.data.found
    if found.dim() > 1:
        found = found.sum(dim=-1)
    has_contact = (found > 0).float()

    asset: Entity = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w
    qx, qy = quat[:, 1], quat[:, 2]
    upright_cos = 1.0 - 2.0 * (qx * qx + qy * qy)
    is_upright = (upright_cos >= upright_cos_threshold).float()

    contact_upright = has_contact * is_upright

    if command_name is None:
        if min_progress_frac > 0.0:
            progress = env.episode_length_buf.float() / float(env.max_episode_length)
            late_enough = (progress >= min_progress_frac).float()
            return late_enough * contact_upright
        return contact_upright
    cmd = env.command_manager.get_command(command_name)
    in_sit_window = (cmd[:, 1] > sin_threshold).float()
    return in_sit_window * contact_upright


def sit_stability(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, command_name: str | None = None, ang_vel_std: float = 0.5, sin_threshold: float = 0.7, min_progress_frac: float = 0.0) -> torch.Tensor:
    """Bonus for low body angular velocity — a stable rest pose.

    Gating matches ``sit_grounded``.
    """
    asset = env.scene[asset_cfg.name]
    ang_vel_norm = asset.data.root_link_ang_vel_w.norm(dim=-1)
    stillness = torch.exp(-((ang_vel_norm / ang_vel_std) ** 2))
    if command_name is None:
        if min_progress_frac > 0.0:
            progress = env.episode_length_buf.float() / float(env.max_episode_length)
            late_enough = (progress >= min_progress_frac).float()
            return late_enough * stillness
        return stillness
    cmd = env.command_manager.get_command(command_name)
    in_sit_window = (cmd[:, 1] > sin_threshold).float()
    return in_sit_window * stillness


def joint_deviation_l1(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """L1 penalty for joint positions deviating from their default (HOME).

    Returns sum of |joint_pos - default| over the selected joints. Unlike the
    Gaussian `pose` reward (which saturates near 1.0 for any small deviation),
    this gives a *linear* gradient at all deviation magnitudes — useful as a
    focused penalty on a subset of joints (e.g. hip_yaw / hip_roll) to prevent
    them drifting to wide-base stances even when other joints are near HOME.
    """
    asset = env.scene[asset_cfg.name]
    jnt_ids = asset_cfg.joint_ids
    err = asset.data.joint_pos[:, jnt_ids] - asset.data.default_joint_pos[:, jnt_ids]
    return torch.sum(torch.abs(err), dim=-1)


def joint_pos_limit_proximity(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, margin: float = 0.15) -> torch.Tensor:
    """L1 cost for joints inside ``margin`` (rad) of their HARD limits
    (≥ 0 → negative weight).

    The stock ``joint_pos_limits`` only fires past the soft limit (last ~7.5% of
    range) and only by the overshoot, so it barely deters a joint parked on its
    stop. Command-side penalties don't work either: wide ctrlrange overshoot is
    intentional for low-kp servos, so the deterrent must live on the qpos side
    and bite well before the stop.
    """
    asset = env.scene[asset_cfg.name]
    jnt_ids = asset_cfg.joint_ids
    q = asset.data.joint_pos[:, jnt_ids]
    hard = asset.data.joint_pos_limits[:, jnt_ids]
    soft_lo = hard[..., 0] + margin
    soft_hi = hard[..., 1] - margin
    below = (soft_lo - q).clip(min=0.0)
    above = (q - soft_hi).clip(min=0.0)
    return torch.sum(below + above, dim=-1)


def phase_height_track(env: ManagerBasedRlEnv, command_name: str, stand_z: float, sit_z: float, std: float = 0.02, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Trunk z tracking a sin-interpolated stand↔sit target.

    Rewards the END STATE, not a prescribed descent, so the policy can pick any
    strategy (deep squat, head-supported descent). cmd[:, 1] = sin(2π·phase):
    +1 → sit_z, −1 → stand_z, 0 → midpoint.
    """
    cmd = env.command_manager.get_command(command_name)
    sin_phase = cmd[:, 1]
    target_z = (stand_z + sit_z) * 0.5 - (stand_z - sit_z) * 0.5 * sin_phase
    asset = env.scene[asset_cfg.name]
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    return torch.exp(-(((z - target_z) / std) ** 2))


def pose_target_match(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, std: float = 0.3, joint_indices: list | None = None, target_overrides: dict | None = None) -> torch.Tensor:
    """Always-on Gaussian on joint positions vs a target pose.

    Non-phase analog of ``phase_pose_match`` for episodic tasks with a constant
    target. ``target_overrides`` is ``{joint_index: angle_rad}``; unlisted
    joints keep the HOME default.
    """
    asset = env.scene[asset_cfg.name]
    joint_pos = _servo_joint_pos(env, asset)
    target = _servo_default_joint_pos(env, asset).clone()
    if target_overrides:
        for idx, val in target_overrides.items():
            target[:, idx] = val
    if joint_indices is not None:
        joint_pos = joint_pos[:, joint_indices]
        target = target[:, joint_indices]
    return torch.exp(-(((joint_pos - target) / std) ** 2)).mean(dim=-1)


def interpolated_pose_target_match(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, std: float = 0.3, joint_indices: list | None = None, source_overrides: dict | None = None, target_overrides: dict | None = None, ramp_start_frac: float = 0.0, ramp_end_frac: float = 1.0) -> torch.Tensor:
    """Gaussian on joint positions vs a source→target pose ramped over the
    episode-progress window, clamped outside it.

    Enforces a smooth descent: snapping to the final pose early leaves the
    robot off-target versus where the ramp currently is, so it forfeits pose
    reward for the whole mismatch.
    """
    asset = env.scene[asset_cfg.name]
    joint_pos = _servo_joint_pos(env, asset)
    source = _servo_default_joint_pos(env, asset).clone()
    target = _servo_default_joint_pos(env, asset).clone()
    if source_overrides:
        for idx, val in source_overrides.items():
            source[:, idx] = val
    if target_overrides:
        for idx, val in target_overrides.items():
            target[:, idx] = val

    progress = env.episode_length_buf.float() / float(env.max_episode_length)
    span = max(ramp_end_frac - ramp_start_frac, 1e-6)
    tau = ((progress - ramp_start_frac) / span).clamp(0.0, 1.0).unsqueeze(-1)
    interp = source * (1.0 - tau) + target * tau

    if joint_indices is not None:
        joint_pos = joint_pos[:, joint_indices]
        interp = interp[:, joint_indices]
    return torch.exp(-(((joint_pos - interp) / std) ** 2)).mean(dim=-1)


def interpolated_pose_l1_penalty(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, joint_indices: list | None = None, source_overrides: dict | None = None, target_overrides: dict | None = None, ramp_start_frac: float = 0.0, ramp_end_frac: float = 1.0) -> torch.Tensor:
    """L1 companion to ``interpolated_pose_target_match``. SELF-NEGATING (≤ 0)
    → POSITIVE weight.

    Constant gradient everywhere, so it still points at the target where the
    Gaussian has saturated to zero.
    """
    asset = env.scene[asset_cfg.name]
    joint_pos = _servo_joint_pos(env, asset)
    source = _servo_default_joint_pos(env, asset).clone()
    target = _servo_default_joint_pos(env, asset).clone()
    if source_overrides:
        for idx, val in source_overrides.items():
            source[:, idx] = val
    if target_overrides:
        for idx, val in target_overrides.items():
            target[:, idx] = val

    progress = env.episode_length_buf.float() / float(env.max_episode_length)
    span = max(ramp_end_frac - ramp_start_frac, 1e-6)
    tau = ((progress - ramp_start_frac) / span).clamp(0.0, 1.0).unsqueeze(-1)
    interp = source * (1.0 - tau) + target * tau

    if joint_indices is not None:
        joint_pos = joint_pos[:, joint_indices]
        interp = interp[:, joint_indices]
    return -torch.abs(joint_pos - interp).mean(dim=-1)


def interpolated_height_l1_penalty(env: ManagerBasedRlEnv, start_height: float, end_height: float, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, ramp_start_frac: float = 0.0, ramp_end_frac: float = 1.0) -> torch.Tensor:
    """``interpolated_pose_l1_penalty`` on trunk z. SELF-NEGATING (≤ 0) →
    POSITIVE weight.
    """
    progress = env.episode_length_buf.float() / float(env.max_episode_length)
    span = max(ramp_end_frac - ramp_start_frac, 1e-6)
    tau = ((progress - ramp_start_frac) / span).clamp(0.0, 1.0)
    target_z = start_height * (1.0 - tau) + end_height * tau

    asset = env.scene[asset_cfg.name]
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    return -torch.abs(z - target_z)


def interpolated_height_target(env: ManagerBasedRlEnv, start_height: float, end_height: float, std: float = 0.02, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, ramp_start_frac: float = 0.0, ramp_end_frac: float = 1.0) -> torch.Tensor:
    """``interpolated_pose_target_match`` on trunk z."""
    progress = env.episode_length_buf.float() / float(env.max_episode_length)
    span = max(ramp_end_frac - ramp_start_frac, 1e-6)
    tau = ((progress - ramp_start_frac) / span).clamp(0.0, 1.0)
    target_z = start_height * (1.0 - tau) + end_height * tau

    asset = env.scene[asset_cfg.name]
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    return torch.exp(-(((z - target_z) / std) ** 2))


def bilateral_symmetry_penalty(env: ManagerBasedRlEnv, left_indices: list, right_indices: list, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """L1 on left/right leg asymmetry. SELF-NEGATING (≤ 0) → POSITIVE weight.

    Left/right joints use mirrored sign conventions, so every symmetric pose
    (HOME, FOLD, SIT) satisfies ``q_left + q_right == 0`` per matched pair.
    Counters the one-leg-correct local minimum that ``mean()``-ed pose rewards
    permit: half the reward comes free and the gradient to fix the second leg is
    too weak to escape.
    """
    asset: Entity = env.scene[asset_cfg.name]
    pos = asset.data.joint_pos
    left = pos[:, left_indices]
    right = pos[:, right_indices]
    return -torch.abs(left + right).mean(dim=-1)


def _multistage_target_pose(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, waypoints) -> torch.Tensor:
    """Time-interpolated joint target across N waypoints, clamped at the ends.

    waypoints: ordered [{"frac": [0,1], "overrides": {joint_idx: rad} | None}];
    the first should be frac=0.0.
    """
    asset = env.scene[asset_cfg.name]
    default = _servo_default_joint_pos(env, asset)

    def build_pose(overrides):
        pose = default.clone()
        if overrides:
            for idx, val in overrides.items():
                pose[:, idx] = val
        return pose

    progress = env.episode_length_buf.float() / float(env.max_episode_length)
    out = build_pose(waypoints[0]["overrides"])
    for i in range(1, len(waypoints)):
        f0 = waypoints[i - 1]["frac"]
        f1 = waypoints[i]["frac"]
        span = max(f1 - f0, 1e-6)
        tau = ((progress - f0) / span).clamp(0.0, 1.0).unsqueeze(-1)
        prev_pose = build_pose(waypoints[i - 1]["overrides"])
        next_pose = build_pose(waypoints[i]["overrides"])
        seg = prev_pose * (1.0 - tau) + next_pose * tau
        mask = (progress >= f0).float().unsqueeze(-1)
        out = torch.where(mask > 0, seg, out)
    return out


def _multistage_target_height(env: ManagerBasedRlEnv, waypoints) -> torch.Tensor:
    """``_multistage_target_pose`` for trunk z; waypoints carry "height"."""
    progress = env.episode_length_buf.float() / float(env.max_episode_length)
    out = torch.full_like(progress, waypoints[0]["height"])
    for i in range(1, len(waypoints)):
        f0 = waypoints[i - 1]["frac"]
        f1 = waypoints[i]["frac"]
        span = max(f1 - f0, 1e-6)
        tau = ((progress - f0) / span).clamp(0.0, 1.0)
        seg = waypoints[i - 1]["height"] * (1.0 - tau) + waypoints[i]["height"] * tau
        mask = (progress >= f0).float()
        out = torch.where(mask > 0, seg, out)
    return out


def multistage_pose_target_match(env: ManagerBasedRlEnv, waypoints: list, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, std: float = 0.3, joint_indices: list | None = None) -> torch.Tensor:
    """Multi-waypoint ``interpolated_pose_target_match`` — forces a trajectory
    through intermediate poses (e.g. stand → fold → sit).

    Waypoint trajectories invite waypoint-camping; prefer a single fixed target
    for episodic pose-landing tasks.
    """
    asset = env.scene[asset_cfg.name]
    target = _multistage_target_pose(env, asset_cfg, waypoints)
    joint_pos = _servo_joint_pos(env, asset)
    if joint_indices is not None:
        joint_pos = joint_pos[:, joint_indices]
        target = target[:, joint_indices]
    return torch.exp(-(((joint_pos - target) / std) ** 2)).mean(dim=-1)


def multistage_pose_l1_penalty(env: ManagerBasedRlEnv, waypoints: list, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, joint_indices: list | None = None) -> torch.Tensor:
    """L1 companion to ``multistage_pose_target_match``. SELF-NEGATING (≤ 0) →
    POSITIVE weight.
    """
    asset = env.scene[asset_cfg.name]
    target = _multistage_target_pose(env, asset_cfg, waypoints)
    joint_pos = _servo_joint_pos(env, asset)
    if joint_indices is not None:
        joint_pos = joint_pos[:, joint_indices]
        target = target[:, joint_indices]
    return -torch.abs(joint_pos - target).mean(dim=-1)


def multistage_height_target(env: ManagerBasedRlEnv, waypoints: list, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, std: float = 0.03) -> torch.Tensor:
    """Multi-waypoint Gaussian on trunk z."""
    target_z = _multistage_target_height(env, waypoints)
    asset = env.scene[asset_cfg.name]
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    return torch.exp(-(((z - target_z) / std) ** 2))


def multistage_height_l1_penalty(env: ManagerBasedRlEnv, waypoints: list, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """L1 companion to ``multistage_height_target``. SELF-NEGATING (≤ 0) →
    POSITIVE weight.
    """
    target_z = _multistage_target_height(env, waypoints)
    asset = env.scene[asset_cfg.name]
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    return -torch.abs(z - target_z)


def pose_target_match(env: ManagerBasedRlEnv, target_overrides: dict | None = None, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, std: float = 0.3, joint_indices: list | None = None) -> torch.Tensor:
    """Gaussian pose-match against a single fixed target, constant from t=0."""
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
    """L1 companion to ``pose_target_match``. SELF-NEGATING (≤ 0) → POSITIVE
    weight.
    """
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
    """Gaussian on trunk z against a single fixed target."""
    asset = env.scene[asset_cfg.name]
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    return torch.exp(-(((z - target_height) / std) ** 2))


def height_l1_penalty(env: ManagerBasedRlEnv, target_height: float, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """L1 companion to ``height_target_gaussian``. SELF-NEGATING (≤ 0) →
    POSITIVE weight.
    """
    asset = env.scene[asset_cfg.name]
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    return -torch.abs(z - target_height)


def trunk_vertical_accel_penalty(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Trunk ``|a_z|``. SELF-NEGATING (≤ 0) → POSITIVE weight.

    Catches landing impacts and buys a quasi-static descent (constant velocity
    → a_z ≈ 0); a robot at rest pays nothing.
    """
    asset = env.scene[asset_cfg.name]
    vz = torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0)
    prev = getattr(env, "_prev_trunk_vz", None)
    if prev is None or prev.shape[0] != vz.shape[0]:
        prev = vz.detach().clone()
    a_z = (vz - prev) / env.step_dt
    # Else the first step charges the previous episode's final velocity.
    if hasattr(env, "episode_length_buf"):
        reset_mask = env.episode_length_buf <= 1
        a_z = torch.where(reset_mask, torch.zeros_like(a_z), a_z)
    env._prev_trunk_vz = vz.detach().clone()
    return -torch.abs(a_z)


def trunk_downward_velocity_penalty(env: ManagerBasedRlEnv, max_down_vel: float = 0.05, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Downward trunk speed beyond ``max_down_vel``. SELF-NEGATING (≤ 0) →
    POSITIVE weight.

    ``trunk_vertical_accel_penalty`` alone cannot cap descent SPEED: a fast
    constant-velocity drop has a_z ≈ 0 all the way down and pays a single
    impact spike, which is cheap next to reaching the target pose sooner.
    """
    asset = env.scene[asset_cfg.name]
    vz = torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0)
    return -torch.clamp(-vz - max_down_vel, min=0.0)


def seated_stillness(env: ManagerBasedRlEnv, height_full: float = 0.06, height_zero: float = 0.08, vel_std: float = 0.05, tilt_full_deg: float = 25.0, tilt_zero_deg: float = 60.0, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Reward trunk stillness while seated UPRIGHT: |v| Gaussian, z- and
    tilt-gated, so quiet-upright-at-seated-height is the only rewarded rest.

    The tilt gate is mandatory: without it "lie still on your back" scores as
    well as "sit still upright" (a trunk on its back is inside the seated z
    band and perfectly motionless) — a converged exploit.
    """
    asset = env.scene[asset_cfg.name]
    v = torch.nan_to_num(asset.data.root_link_lin_vel_w, nan=0.0).norm(dim=-1)
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    t = torch.clamp((height_zero - z) / max(height_zero - height_full, 1e-6), 0.0, 1.0)
    z_gate = t * t * (3.0 - 2.0 * t)
    quat = asset.data.root_link_quat_w
    cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    cos_full = math.cos(math.radians(tilt_full_deg))
    cos_zero = math.cos(math.radians(tilt_zero_deg))
    u = torch.clamp((cos_tilt - cos_zero) / max(cos_full - cos_zero, 1e-6), 0.0, 1.0)
    tilt_gate = u * u * (3.0 - 2.0 * u)
    return torch.exp(-((v / vel_std) ** 2)) * z_gate * tilt_gate


def upright_while_tall(env: ManagerBasedRlEnv, height_low: float, height_high: float, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """``body_upright_linear`` smoothstep-weighted by trunk z: full while still
    standing tall, fading out once committed to the sit (where butt-on-ground
    orientation is fine).

    Stops the policy tipping backward while still high, i.e. farming the
    descent reward with a controlled fall.
    """
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
    """Phase → blend in [0, 1]: 0 = STAND pose, 1 = DOWN pose."""
    b = torch.zeros_like(phase)
    descend = phase < descent_end
    b = torch.where(descend, phase / descent_end, b)
    low = (phase >= descent_end) & (phase < hold_end)
    b = torch.where(low, torch.ones_like(phase), b)
    rise = (phase >= hold_end) & (phase < rise_end)
    b = torch.where(rise, 1.0 - (phase - hold_end) / (rise_end - hold_end), b)
    return b


def kick_pose_target(phase: torch.Tensor, stand: torch.Tensor, back: torch.Tensor, forward: torch.Tensor, windup_end: float, kick_end: float, return_end: float) -> torch.Tensor:
    """Interpolated joint target of a 4-keyframe kicking gesture:
    STAND → BACK (wind-up) → FORWARD (strike) → STAND (return, then rest).
    """
    p = phase.unsqueeze(-1)

    def interp(a, b, s):
        return a + s * (b - a)

    s1 = (p / windup_end).clamp(0.0, 1.0)
    s2 = ((p - windup_end) / (kick_end - windup_end)).clamp(0.0, 1.0)
    s3 = ((p - kick_end) / (return_end - kick_end)).clamp(0.0, 1.0)

    seg1 = interp(stand, back, s1)
    seg2 = interp(back, forward, s2)
    seg3 = interp(forward, stand, s3)

    out = seg1
    out = torch.where(p >= windup_end, seg2, out)
    out = torch.where(p >= kick_end, seg3, out)
    return out


def _kick_pose_error(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, command_name: str, stand_pose: dict, back_pose: dict, forward_pose: dict, windup_end: float, kick_end: float, return_end: float, joint_names: list | None = None):
    """(cur, target) for the kicking gesture, joints resolved BY NAME.

    The 3 poses share keys; name order comes from `stand_pose`, or from
    `joint_names` to score the gesture and support leg with different stds.
    """
    if not stand_pose:
        raise ValueError("_kick_pose_error requires a non-empty stand_pose dict")
    asset: Entity = env.scene[asset_cfg.name]
    names = list(joint_names) if joint_names is not None else list(stand_pose.keys())
    ids = [int(asset.find_joints([n])[0][0]) for n in names]

    def vec(d):
        return torch.tensor([d[n] for n in names], device=env.device, dtype=asset.data.joint_pos.dtype)

    stand_v, back_v, fwd_v = vec(stand_pose), vec(back_pose), vec(forward_pose)

    cmd = env.command_manager.get_command(command_name)
    phase = (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * torch.pi)) % 1.0
    target = kick_pose_target(phase, stand_v, back_v, fwd_v, windup_end, kick_end, return_end)
    cur = asset.data.joint_pos[:, ids]
    return cur, target


def kick_pose_track(env: ManagerBasedRlEnv, command_name: str = "twist", stand_pose: dict | None = None, back_pose: dict | None = None, forward_pose: dict | None = None, std: float = 0.4, windup_end: float = 0.35, kick_end: float = 0.45, return_end: float = 0.75, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, joint_names: list | None = None) -> torch.Tensor:
    """Gaussian on the joint pose vs the interpolated kick target.

    `joint_names` restricts evaluation to a subset, e.g. track the kicking leg
    tightly and the support leg loosely so it can still balance.
    """
    cur, target = _kick_pose_error(env, asset_cfg, command_name, stand_pose or {}, back_pose or {}, forward_pose or {}, windup_end, kick_end, return_end, joint_names)
    return torch.exp(-(((cur - target) / std) ** 2)).mean(dim=-1)


def kick_pose_track_l1(env: ManagerBasedRlEnv, command_name: str = "twist", stand_pose: dict | None = None, back_pose: dict | None = None, forward_pose: dict | None = None, windup_end: float = 0.35, kick_end: float = 0.45, return_end: float = 0.75, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, joint_names: list | None = None) -> torch.Tensor:
    """L1 companion to ``kick_pose_track``. SELF-NEGATING (≤ 0) → POSITIVE
    weight.
    """
    cur, target = _kick_pose_error(env, asset_cfg, command_name, stand_pose or {}, back_pose or {}, forward_pose or {}, windup_end, kick_end, return_end, joint_names)
    return -(cur - target).abs().mean(dim=-1)


def kick_engagement(phase: torch.Tensor, windup_end: float, return_end: float) -> torch.Tensor:
    """Gesture engagement gate ∈ [0, 1]: ramps up over the wind-up, holds 1
    through the strike (single-leg support), 0 at the STAND rest.

    Weights the single-leg balance rewards, which must not apply at rest where
    two-leg support and a centered CoM are correct.
    """
    g = torch.zeros_like(phase)
    ramp = phase < windup_end
    g = torch.where(ramp, phase / windup_end, g)
    hold = (phase >= windup_end) & (phase < return_end)
    g = torch.where(hold, torch.ones_like(phase), g)
    return g


def com_over_support_foot(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, command_name: str = "twist", std: float = 0.04, windup_end: float = 0.35, return_end: float = 0.75) -> torch.Tensor:
    """Gaussian on CoM↔support-foot horizontal distance, gated on the strike
    phase. Teaches the lateral weight transfer.

    Without it, a one-footed gesture built from two-leg-support keyframes keeps
    the CoM centered between the feet and tips over the moment a foot lifts.

    `asset_cfg` must target the support foot site; `std` in metres (≈ foot size).
    """
    asset: Entity = env.scene[asset_cfg.name]
    com_xy = asset.data.root_com_pos_w[:, :2]
    foot_id = asset_cfg.site_ids[0]
    foot_xy = asset.data.site_pos_w[:, foot_id, :2]
    dist2 = ((com_xy - foot_xy) ** 2).sum(dim=-1)
    reward = torch.exp(-dist2 / (std**2))

    cmd = env.command_manager.get_command(command_name)
    phase = (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * torch.pi)) % 1.0
    gate = kick_engagement(phase, windup_end, return_end)
    return gate * reward


def _phase_pose_error(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, command_name: str, target_pose: dict, descent_end: float, hold_end: float, rise_end: float, source_pose: dict | None = None):
    """(cur, target) for the phase-interpolated pose, resolved BY NAME.

    Source is `source_pose` if given, else the model default (HOME).
    """
    if not target_pose:
        raise ValueError("_phase_pose_error requires a non-empty target_pose dict")

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
    """Gaussian on the joint pose vs the interpolated STAND↔DOWN target.

    Symmetric by construction: standing up pays exactly like going down.
    """
    cur, target = _phase_pose_error(env, asset_cfg, command_name, target_pose or {}, descent_end, hold_end, rise_end, source_pose)
    return torch.exp(-(((cur - target) / std) ** 2)).mean(dim=-1)


def phase_pose_track_l1(env: ManagerBasedRlEnv, command_name: str = "twist", target_pose: dict | None = None, source_pose: dict | None = None, descent_end: float = 0.15, hold_end: float = 0.50, rise_end: float = 0.65, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """L1 companion to ``phase_pose_track``. SELF-NEGATING (≤ 0) → POSITIVE
    weight.

    Constant gradient everywhere, so it still points at the target where the
    Gaussian has saturated to ~0.
    """
    cur, target = _phase_pose_error(env, asset_cfg, command_name, target_pose or {}, descent_end, hold_end, rise_end, source_pose)
    return -(cur - target).abs().mean(dim=-1)


def phase_pose_match(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, std: float = 0.3, command_name: str = "twist", joint_indices: list | None = None, target_overrides: dict | None = None, phase: str = "approach") -> torch.Tensor:
    """Gaussian pose match weighted by a phase-cycle command.

    The command encodes phase as [cos(2π·phase), sin(2π·phase), 0]; "approach"
    peaks at phase 0.25, "return" at 0.75. ``target_overrides`` is
    {joint_index: angle_rad}; unlisted joints keep the HOME default.
    """
    asset = env.scene[asset_cfg.name]
    joint_pos = _servo_joint_pos(env, asset)
    target = _servo_default_joint_pos(env, asset).clone()
    if target_overrides:
        for idx, val in target_overrides.items():
            target[:, idx] = val
    if joint_indices is not None:
        joint_pos = joint_pos[:, joint_indices]
        target = target[:, joint_indices]
    pose_reward = torch.exp(-(((joint_pos - target) / std) ** 2)).mean(dim=-1)

    cmd = env.command_manager.get_command(command_name)
    if phase == "approach":
        weight = torch.clamp(cmd[:, 1], min=0.0)
    else:
        weight = torch.clamp(-cmd[:, 1], min=0.0)
    return weight * pose_reward


def ground_pick_return_pose(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, std: float = 0.3, command_name: str = "twist", joint_indices: list | None = None) -> torch.Tensor:
    """Gaussian return-to-HOME pose match, weighted by the return half-cycle.

    Call twice with different ``joint_indices`` to give legs and neck/head
    different stds.
    """
    asset = env.scene[asset_cfg.name]
    joint_pos = _servo_joint_pos(env, asset)
    default_pos = _servo_default_joint_pos(env, asset)

    if joint_indices is not None:
        joint_pos = joint_pos[:, joint_indices]
        default_pos = default_pos[:, joint_indices]

    pose_reward = torch.exp(-(((joint_pos - default_pos) / std) ** 2)).mean(dim=-1)

    cmd = env.command_manager.get_command(command_name)
    return_weight = torch.clamp(-cmd[:, 1], min=0.0)

    return return_weight * pose_reward


def ground_pick_return_upright(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, std: float = 0.4, command_name: str = "twist") -> torch.Tensor:
    """Reward trunk verticality, weighted by the RETURN phase (stand-up aid).

    Return-weighted like ``ground_pick_return_pose`` so it never fights the
    forward lean of the approach. A broad std keeps gradient available even
    from a deep crouch.
    """
    asset: Entity = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w
    tilt_sq = 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    upright = torch.exp(-tilt_sq / (std * std))
    cmd = env.command_manager.get_command(command_name)
    return_weight = torch.clamp(-cmd[:, 1], min=0.0)
    return return_weight * upright


# ── Ground-pick segmented phase gating ───────────────────────────────────────
# Independent descent/hold/rise/rest durations, replacing the sinusoidal
# max(0, ±sin) weighting: down-gate = phase_pose_blend, up-gate = phase_rise_gate.
def phase_rise_gate(phase: torch.Tensor, hold_end: float, rise_end: float) -> torch.Tensor:
    """Rising gate for the RETURN: 0 before hold_end, 0->1 over [hold_end,
    rise_end), 1 after (standing rest)."""
    g = torch.zeros_like(phase)
    rising = (phase >= hold_end) & (phase < rise_end)
    g = torch.where(rising, (phase - hold_end) / (rise_end - hold_end), g)
    g = torch.where(phase >= rise_end, torch.ones_like(phase), g)
    return g


def _gp_phase(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)
    return (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * torch.pi)) % 1.0


def mouth_ground_proximity_phased(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=["mouth_tip"]), std: float = 0.10, target_height: float = 0.0, command_name: str = "twist", descent_end: float = 0.25, hold_end: float = 0.35, rise_end: float = 0.60) -> torch.Tensor:
    """mouth_ground_proximity gated by the segmented down-gate (descent+hold)."""
    asset = env.scene[asset_cfg.name]
    mouth_z = asset.data.site_pos_w[:, asset_cfg.site_ids[0], 2]
    proximity = torch.exp(-(((mouth_z - target_height) / std) ** 2))
    gate = phase_pose_blend(_gp_phase(env, command_name), descent_end, hold_end, rise_end)
    return gate * proximity


def mouth_perpendicular_phased(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=["mouth_tip"]), command_name: str = "twist", descent_end: float = 0.25, hold_end: float = 0.35, rise_end: float = 0.60) -> torch.Tensor:
    """mouth_perpendicular_to_ground gated by the segmented down-gate."""
    asset = env.scene[asset_cfg.name]
    q = asset.data.site_quat_w[:, asset_cfg.site_ids[0], :]
    w, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    x_axis_z = 2.0 * (qx * qz - w * qy)
    alignment = -x_axis_z
    gate = phase_pose_blend(_gp_phase(env, command_name), descent_end, hold_end, rise_end)
    return gate * alignment


def ground_pick_return_pose_phased(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, std: float = 0.3, command_name: str = "twist", joint_indices: list | None = None, hold_end: float = 0.35, rise_end: float = 0.60) -> torch.Tensor:
    """ground_pick_return_pose gated by the segmented up-gate (rise+rest)."""
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
    """ground_pick_return_upright gated by the segmented up-gate."""
    asset: Entity = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w
    tilt_sq = 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    upright = torch.exp(-tilt_sq / (std * std))
    gate = phase_rise_gate(_gp_phase(env, command_name), hold_end, rise_end)
    return gate * upright


def neck_vel_descent_penalty(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, command_name: str = "twist", joint_indices: list | None = None, hold_end: float = 0.35) -> torch.Tensor:
    """Neck joint velocity cost during descent+hold only, so it slows the head's
    nosedive without hindering the rise (≥ 0 → negative weight).
    """
    asset = env.scene[asset_cfg.name]
    vel = _servo_joint_vel(env, asset)
    if joint_indices is not None:
        vel = vel[:, joint_indices]
    cost = (vel**2).mean(dim=-1)
    phase = _gp_phase(env, command_name)
    gate = (phase < hold_end).to(vel.dtype)
    return gate * cost


def sample_mouth_payload(env: ManagerBasedRlEnv, env_ids: torch.Tensor, min_kg: float = 0.01, max_kg: float = 0.04) -> None:
    """Sample the mass (kg) of the object held in the mouth, per env."""
    buf = getattr(env, "_mouth_payload_kg", None)
    if buf is None:
        buf = torch.zeros(env.num_envs, device=env.device)
        env._mouth_payload_kg = buf
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    buf[env_ids] = torch.rand(len(env_ids), device=env.device) * (max_kg - min_kg) + min_kg


def apply_mouth_payload_force(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["jaw_soft"], site_names=["mouth_tip"]), command_name: str = "twist", hold_end: float = 0.35, ramp: float = 0.05, gravity: float = 9.81) -> torch.Tensor:
    """Applies the mouth payload's weight at mouth_tip during the rise.

    NOT a reward: registered as a weight-0 reward term purely to get a per-step
    hook, and always returns 0. Force at the CoM plus torque (p_mouth − p_com)
    × F is equivalent to applying it at the tip, giving the neck its real lever
    arm.
    """
    asset: Entity = env.scene[asset_cfg.name]
    payload = getattr(env, "_mouth_payload_kg", None)
    if payload is None:
        return torch.zeros(env.num_envs, device=env.device)
    phase = _gp_phase(env, command_name)
    gate = ((phase - hold_end) / ramp).clamp(0.0, 1.0)
    fz = -(gate * payload) * gravity

    bid = int(asset_cfg.body_ids[0])
    sid = int(asset_cfg.site_ids[0])
    p_mouth = asset.data.site_pos_w[:, sid, :]
    p_com = asset.data.body_com_pos_w[:, bid, :]
    F = torch.zeros((env.num_envs, 3), device=env.device, dtype=p_mouth.dtype)
    F[:, 2] = fz
    tau = torch.cross(p_mouth - p_com, F, dim=-1)
    asset.write_external_wrench_to_sim(forces=F.unsqueeze(1), torques=tau.unsqueeze(1), body_ids=[bid])
    return torch.zeros(env.num_envs, device=env.device)


# ── Domain randomization events ────────────────────────────────────────────


def randomize_delayed_actuator_gains(env: ManagerBasedRlEnv, env_ids: torch.Tensor, kp_range: tuple[float, float], kd_range: tuple[float, float], asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, operation: str = "scale"):
    """Randomize firmware PD gains per episode (NON-accumulating).

    Scales gains through the BAM actuator's own kp_scale/kd_scale rather than
    the MuJoCo model, so there is no accumulation risk. Per-joint samples are
    averaged to one scalar because the actuator applies a single scale across
    its joints. Actuators without set_gains (e.g. roller XmlActuator) are
    skipped.

    ``operation`` is unused; kept for cfg compatibility.
    """
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
        # Restore-then-apply: DR must not accumulate across resets.
        actuator.reset_gains(env_ids)
        actuator.set_gains(env_ids, kp_scale=kp_samples.mean(dim=1, keepdim=True), kd_scale=kd_samples.mean(dim=1, keepdim=True))


@requires_model_fields("dof_frictionloss", "dof_damping")
def expand_bam_friction_fields(env: ManagerBasedRlEnv, env_ids: torch.Tensor):
    """No-op startup event; the decorator above is the entire point.

    BamActuator writes a per-env friction budget into dof_frictionloss/
    dof_damping every step, and mjlab only expands per-world the model fields
    declared via requires_model_fields. EVERY env using BAM must register this.
    """


def randomize_bam_friction(env: ManagerBasedRlEnv, env_ids: torch.Tensor, scale_range: tuple[float, float], asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG):
    """Per-episode joint-friction DR for the BAM actuator (NON-accumulating).

    Under BAM, dof_frictionloss is zeroed (BAM computes friction itself), so
    stock dr.dof_frictionloss is a SILENT NO-OP. This scales the actuator's own
    friction budget instead. No-op without a friction_scale hook.
    """
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


def randomize_mass_and_inertia(env: ManagerBasedRlEnv, env_ids: torch.Tensor, scale_range: tuple[float, float], asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG):
    """Randomize body mass and inertia by ONE shared factor (NON-accumulating).

    They must scale together, or the inertia tensor becomes physically
    inconsistent and destabilizes the sim.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]

    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, slice):
        body_ids = list(range(asset.num_bodies))[body_ids]
    body_indices = asset.indexing.body_ids[body_ids]

    num_envs = len(env_ids)
    num_bodies = len(body_indices)
    scales = torch.rand(num_envs, num_bodies, device=env.device) * (scale_range[1] - scale_range[0]) + scale_range[0]

    if not hasattr(env, "_original_mass_inertia"):
        env._original_mass_inertia = {"mass": env.sim.model.body_mass[0, body_indices].clone(), "inertia": env.sim.model.body_inertia[0, body_indices].clone()}

    # Restore-then-apply: DR must not accumulate across resets.
    original = env._original_mass_inertia
    env.sim.model.body_mass[env_ids[:, None], body_indices] = original["mass"].unsqueeze(0).expand(num_envs, -1)
    env.sim.model.body_inertia[env_ids[:, None], body_indices] = original["inertia"].unsqueeze(0).expand(num_envs, -1, -1)

    env.sim.model.body_mass[env_ids[:, None], body_indices] *= scales
    env.sim.model.body_inertia[env_ids[:, None], body_indices] *= scales.unsqueeze(-1)


def standing_envs_curriculum(env: ManagerBasedRlEnv, env_ids: torch.Tensor, command_name: str, standing_stages: list[dict]) -> torch.Tensor:
    """Step ``rel_standing_envs`` through ``standing_stages``
    ([{"step", "rel_standing_envs"}, ...]).
    """
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


def velocity_tracking_std_curriculum(env: ManagerBasedRlEnv, env_ids: torch.Tensor, reward_name: str, std_stages: list[dict]) -> torch.Tensor:
    """Step a tracking reward's ``std`` through ``std_stages``
    ([{"step", "std"}, ...]).

    Start loose to learn walking at all, then tighten for accuracy.
    """
    del env_ids  # Unused

    reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)

    current_std = std_stages[0]["std"]

    for stage in std_stages:
        if env.common_step_counter > stage["step"]:
            current_std = stage["std"]

    reward_term_cfg.params["std"] = current_std

    return torch.tensor([current_std])


def push_curriculum(env: ManagerBasedRlEnv, env_ids: torch.Tensor, event_name: str, push_stages: list[dict]) -> torch.Tensor:
    """Step the push event's ``velocity_range`` through ``push_stages``
    ([{"step", "velocity_range"}, ...]).

    Start with no pushes so clean walking can form, then build robustness.
    """
    del env_ids

    # Must mutate the live EventManager term cfg: EventManager.__init__
    # deepcopies cfg, so writes to env.cfg.events are silent no-ops.
    event_cfg = env.event_manager.get_term_cfg(event_name)

    current_range = push_stages[0]["velocity_range"]

    for stage in push_stages:
        if env.common_step_counter > stage["step"]:
            current_range = stage["velocity_range"]

    event_cfg.params["velocity_range"] = current_range

    max_push = max(abs(current_range["x"][0]), abs(current_range["x"][1]))
    return torch.tensor([max_push])


def wheel_friction_curriculum(env: ManagerBasedRlEnv, env_ids: torch.Tensor, event_name: str, ranges_stages: list[dict]) -> torch.Tensor:
    """Step wheel friction through ``ranges_stages``."""
    del env_ids

    current_ranges = ranges_stages[0]["ranges"]
    for stage in ranges_stages:
        if env.common_step_counter > stage["step"]:
            current_ranges = stage["ranges"]

    env.event_manager.get_term_cfg(event_name).params["ranges"] = current_ranges
    return torch.tensor([current_ranges[0]])


def reward_weight(env: ManagerBasedRlEnv, env_ids: torch.Tensor, reward_name: str, weight_stages: list[dict]) -> torch.Tensor:
    """Step-staged reward weight curriculum.

    mjlab 1.3.0 dropped the built-in ``mdp.reward_weight``. ``weight_stages`` is
    [{"step", "weight"}, ...] and the latest elapsed stage wins — a STEP
    function, not an interpolation, so discretize ramps into stages.

    Mutates the live RewardManager term cfg; env.cfg is a deepcopy.
    """
    del env_ids
    term_cfg = env.reward_manager.get_term_cfg(reward_name)
    for stage in weight_stages:
        if env.common_step_counter > stage["step"]:
            term_cfg.weight = stage["weight"]
    return torch.tensor([term_cfg.weight])


def com_range_curriculum(env: ManagerBasedRlEnv, env_ids: torch.Tensor, event_name: str, range_stages: list[dict]) -> torch.Tensor:
    """Step the CoM-randomization range (metres) through ``range_stages``
    ([{"step", "range"}, ...]), widening the CoM uncertainty over training.
    """
    del env_ids

    # Must mutate the live EventManager term cfg: EventManager.__init__
    # deepcopies cfg, so writes to env.cfg.events are silent no-ops.
    event_cfg = env.event_manager.get_term_cfg(event_name)

    current_range = range_stages[0]["range"]
    for stage in range_stages:
        if env.common_step_counter > stage["step"]:
            current_range = stage["range"]

    event_cfg.params["ranges"] = (-current_range, current_range)
    return torch.tensor([current_range])


def slope_move_masks(distance: "torch.Tensor", size_x: float):
    """Promotion/demotion masks for the slope curriculum.

    The 40% promotion threshold must stay below the terrain_edge_reached
    termination (~3.8 m of an 8 m tile), which ends the episode before the 50%
    mark — otherwise a successful traverser is never promoted.
    """
    move_up = distance > size_x * 0.4
    move_down = (distance < size_x * 0.2) & (~move_up)
    return move_up, move_down


def terrain_levels_slope(env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> torch.Tensor:
    """Steepness curriculum for roller_slope (no commanded velocity), driven by
    x-distance travelled from the spawn origin.
    """
    asset = env.scene["robot"]
    terrain = env.scene.terrain
    assert terrain is not None
    terrain_generator = terrain.cfg.terrain_generator
    assert terrain_generator is not None

    distance = asset.data.root_link_pos_w[env_ids, 0] - env.scene.env_origins[env_ids, 0]
    move_up, move_down = slope_move_masks(distance, terrain_generator.size[0])
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def velocity_command_ranges_curriculum(env: ManagerBasedRlEnv, env_ids: torch.Tensor, command_name: str, velocity_stages: list[dict], update_lin_vel_y: bool = True, update_ang_vel_z: bool = True, forward_only: bool = False) -> torch.Tensor:
    """Widen the commanded velocity ranges through ``velocity_stages``
    ([{"step", "lin_vel_range", "ang_vel_range"}, ...]).
    """
    del env_ids

    from typing import cast

    from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

    command_term = env.command_manager.get_term(command_name)
    assert command_term is not None, f"Command term '{command_name}' not found"

    cfg = cast(UniformVelocityCommandCfg, command_term.cfg)

    current_lin_vel = velocity_stages[0]["lin_vel_range"]
    current_ang_vel = velocity_stages[0]["ang_vel_range"]

    for stage in velocity_stages:
        if env.common_step_counter > stage["step"]:
            current_lin_vel = stage["lin_vel_range"]
            current_ang_vel = stage["ang_vel_range"]

    if forward_only:
        cfg.ranges.lin_vel_x = (0.0, current_lin_vel)
    else:
        cfg.ranges.lin_vel_x = (-current_lin_vel, current_lin_vel)
    if update_lin_vel_y:
        cfg.ranges.lin_vel_y = (-current_lin_vel, current_lin_vel)
    if update_ang_vel_z:
        cfg.ranges.ang_vel_z = (-current_ang_vel, current_ang_vel)

    return torch.tensor([current_lin_vel])


def projected_gravity(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Gravity in body frame — pure orientation, no linear acceleration."""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b


def _imu_misalignment_quat(env: ManagerBasedRlEnv, max_angle_rad: float) -> torch.Tensor:
    """Per-env constant IMU mounting-misalignment quaternion, sampled once.

    Cached for the whole run, so it is a systematic per-robot bias rather than
    per-step noise. Zero-centered DR like this trains tolerance to misalignment
    MAGNITUDE and cannot compensate a real mounting bias — that is a runtime
    calibration.

    Supersedes randomize_imu_orientation, which wrote site_quat: not per-env
    expanded under mjlab 1.3.0, and not read by the gravity/ang_vel obs anyway.
    """
    q = getattr(env, "_imu_misalign_quat", None)
    if q is None:
        n = env.num_envs
        axis = torch.randn(n, 3, device=env.device)
        axis = axis / (torch.norm(axis, dim=-1, keepdim=True) + 1e-8)
        angle = torch.rand(n, device=env.device) * max_angle_rad
        q = quat_from_angle_axis(angle, axis)
        env._imu_misalign_quat = q
    return q


def projected_gravity_imu_misaligned(env: ManagerBasedRlEnv, max_angle_deg: float = 1.0, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """projected_gravity with a per-env constant IMU mounting misalignment."""
    asset: Entity = env.scene[asset_cfg.name]
    q = _imu_misalignment_quat(env, math.radians(max_angle_deg))
    return quat_apply(q, asset.data.projected_gravity_b)


def base_ang_vel_imu_misaligned(env: ManagerBasedRlEnv, max_angle_deg: float = 1.0, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """base angular velocity with the SAME per-env IMU misalignment as gravity."""
    asset: Entity = env.scene[asset_cfg.name]
    q = _imu_misalignment_quat(env, math.radians(max_angle_deg))
    return quat_apply(q, asset.data.root_link_ang_vel_b)


def raw_accelerometer(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Normalized raw accelerometer (gravity + linear acceleration), as a real
    IMU measures it — unlike ``projected_gravity``, which is orientation only.
    """
    asset: Entity = env.scene[asset_cfg.name]

    mj_model = asset.data.model

    # robot.xml sensor order: framequat, gyro, gyro, velocimeter, accelerometer,
    # subtreeangmom — imu_accel is index 4.
    sensor_adr_array = mj_model.sensor_adr
    sensor_id = 4
    sensor_adr = int(sensor_adr_array[sensor_id].item())

    accel_raw = asset.data.data.sensordata[:, sensor_adr : sensor_adr + 3]

    # MuJoCo reports specific force; negate so upright-at-rest points down.
    accel_negated = -accel_raw

    accel_norm = torch.norm(accel_negated, dim=-1, keepdim=True)
    accel_normalized = torch.where(accel_norm > 0.1, accel_negated / accel_norm, asset.data.projected_gravity_b)

    return accel_normalized


def randomize_imu_orientation(env: ManagerBasedRlEnv, env_ids: torch.Tensor, max_angle_deg: float = 2.0, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG):
    """Randomize IMU mounting orientation by small angles.

    Superseded by ``_imu_misalignment_quat``: site_quat is not per-env expanded
    under mjlab 1.3.0 and the gravity/ang_vel observations never read it.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]

    # robot.xml site order: imu (0), left_foot (1), right_foot (2).
    site_id = 0

    if not hasattr(env, "_original_imu_quat"):
        env._original_imu_quat = env.sim.model.site_quat[0, site_id].clone()

    num_envs = len(env_ids)
    max_angle_rad = max_angle_deg * torch.pi / 180.0

    angles = (torch.rand(num_envs, 3, device=env.device) * 2 - 1) * max_angle_rad

    # Small-angle quaternion: [1, θx/2, θy/2, θz/2].
    half_angles = angles / 2.0
    quats_delta = torch.zeros(num_envs, 4, device=env.device)
    quats_delta[:, 0] = 1.0
    quats_delta[:, 1:] = half_angles

    quats_delta = quats_delta / torch.norm(quats_delta, dim=1, keepdim=True)

    original_quat = env._original_imu_quat.unsqueeze(0).expand(num_envs, -1)

    # q_new = q_delta * q_original
    w1, x1, y1, z1 = quats_delta[:, 0], quats_delta[:, 1], quats_delta[:, 2], quats_delta[:, 3]
    w2, x2, y2, z2 = original_quat[:, 0], original_quat[:, 1], original_quat[:, 2], original_quat[:, 3]

    new_quat = torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,  # w
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,  # x
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,  # y
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,  # z
        ],
        dim=1,
    )

    env.sim.model.site_quat[env_ids, site_id] = new_quat


def standing_phase(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Time-based phase in [0, 1), cycling every 2 s, so a standing policy still
    has a time-varying input.
    """
    phase_period = 2.0
    time = env.episode_length_buf * env.step_dt
    phase = (time % phase_period) / phase_period

    return phase.unsqueeze(-1)


def air_time_adaptive(env: ManagerBasedRlEnv, sensor_name: str, command_name: str = "twist", command_threshold: float = 0.01, running_threshold: float = 0.5, walk_threshold_min: float = 0.10, walk_threshold_max: float = 0.25, run_threshold_min: float = 0.05, run_threshold_max: float = 0.25) -> torch.Tensor:
    """Air-time reward with separate swing-time windows for walking vs running,
    so the walk keeps a deliberate cadence while running can step faster.

    Zero below ``command_threshold`` (standing).
    """
    sensor = env.scene.sensors[sensor_name]
    current_air_time = sensor.data.current_air_time
    assert current_air_time is not None

    command = env.command_manager.get_command(command_name)
    total_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])

    is_walking = ((total_speed >= command_threshold) & (total_speed < running_threshold)).float()
    is_running = (total_speed >= running_threshold).float()

    tmin = (is_walking * walk_threshold_min + is_running * run_threshold_min).unsqueeze(1)
    tmax = (is_walking * walk_threshold_max + is_running * run_threshold_max).unsqueeze(1)

    in_range = (current_air_time > tmin) & (current_air_time < tmax)
    reward = torch.sum(in_range.float(), dim=1)

    active = (total_speed >= command_threshold).float()
    return reward * active


def stillness_at_zero_command(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, command_name: str = "twist", command_threshold: float = 0.01, vel_std: float = 0.1) -> torch.Tensor:
    """Reward stillness while the command is near zero.

    Monotonically decreasing in body speed, so unlike a gate-based stepping
    penalty there is no threshold the robot can cross to escape it.
    """
    asset: Entity = env.scene[asset_cfg.name]

    command = env.command_manager.get_command(command_name)
    total_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    is_standing_cmd = (total_speed < command_threshold).float()

    body_vel = torch.norm(asset.data.root_link_vel_w[:, :2], dim=1)
    stillness = torch.exp(-(body_vel**2) / vel_std**2)

    return is_standing_cmd * stillness


def joint_vel_l2_when_standing(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, command_name: str = "twist", command_threshold: float = 0.01) -> torch.Tensor:
    """Leg joint velocity cost while the command is near zero (≥ 0 → negative
    weight): kills standing shake without touching the walking gait.
    """
    asset: Entity = env.scene[asset_cfg.name]

    command = env.command_manager.get_command(command_name)
    total_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    is_standing_cmd = (total_speed < command_threshold).float()

    leg_indices = list(range(5)) + list(range(9, 14))
    joint_vel = asset.data.joint_vel[:, leg_indices]
    vel_sq = torch.sum(joint_vel**2, dim=-1)

    return is_standing_cmd * vel_sq


def foot_step_penalty_when_standing(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, command_name: str = "twist", command_threshold: float = 0.01, body_vel_threshold: float = 0.2, air_time_threshold: float = 0.05) -> torch.Tensor:
    """Stepping at zero command ∈ [0, 1] (≥ 0 → negative weight): the mirror of
    the air_time reward, which pays for stepping while commanded to move.

    The body-velocity gate spares recovery steps after a push, so a robot
    already moving fast can still catch itself.
    """
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor = env.scene.sensors["feet_ground_contact"]

    air_time = contact_sensor.data.last_air_time[:, :2]
    any_foot_stepped = (air_time > air_time_threshold).any(dim=1).float()

    command = env.command_manager.get_command(command_name)
    total_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    is_standing = (total_speed < command_threshold).float()

    body_vel = torch.norm(asset.data.root_link_vel_w[:, :2], dim=1)
    is_still = (body_vel < body_vel_threshold).float()

    return any_foot_stepped * is_standing * is_still


def recovery_stepping_reward(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, command_name: str = "twist", command_threshold: float = 0.01, velocity_threshold: float = 0.3, air_time_threshold: float = 0.05) -> torch.Tensor:
    """Reward stepping only at zero command AND high body velocity, i.e. while
    recovering from a push; silent during normal walking.
    """
    asset: Entity = env.scene[asset_cfg.name]

    command = env.command_manager.get_command(command_name)
    total_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    is_standing_cmd = (total_speed < command_threshold).float()

    base_lin_vel = asset.data.root_link_vel_w[:, :3]
    vel_magnitude = torch.norm(base_lin_vel[:, :2], dim=1)

    should_step = vel_magnitude > velocity_threshold

    contact_sensor = env.scene.sensors["feet_ground_contact"]
    air_time = contact_sensor.data.last_air_time[:, :2]

    foot_in_air = (air_time > air_time_threshold).any(dim=1)

    reward = is_standing_cmd * should_step.float() * foot_in_air.float()

    return reward


def adaptive_pose_weight(env: ManagerBasedRlEnv, base_pose_reward: torch.Tensor, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, velocity_threshold: float = 0.3, min_weight: float = 0.3) -> torch.Tensor:
    """Fade pose-tracking weight down to ``min_weight`` at high velocity, so
    recovery steps may deviate from the standing pose while a still robot is
    still held to it.
    """
    asset: Entity = env.scene[asset_cfg.name]

    base_lin_vel = asset.data.root_link_vel_w[:, :3]
    vel_magnitude = torch.norm(base_lin_vel[:, :2], dim=1)

    weight = min_weight + (1.0 - min_weight) * torch.exp(-(((vel_magnitude - velocity_threshold) / velocity_threshold).clamp(min=0.0) ** 2))

    return base_pose_reward * weight


def randomize_base_orientation(env: ManagerBasedRlEnv, env_ids: torch.Tensor, max_pitch_deg: float = 10.0, max_roll_deg: float = 5.0, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG):
    """Random pitch/roll at episode start, so the policy cannot memorize one
    initial state and must use feedback.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]
    num_envs = len(env_ids)

    max_pitch_rad = max_pitch_deg * torch.pi / 180.0
    max_roll_rad = max_roll_deg * torch.pi / 180.0

    pitch = (torch.rand(num_envs, device=env.device) * 2 - 1) * max_pitch_rad
    roll = (torch.rand(num_envs, device=env.device) * 2 - 1) * max_roll_rad
    yaw = torch.zeros(num_envs, device=env.device)

    # Aerospace ZYX Euler → quaternion.
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

    # Freejoint qpos: [x, y, z, qw, qx, qy, qz].
    root_quat_idx = 3

    env.sim.data.qpos[env_ids, root_quat_idx : root_quat_idx + 4] = new_quat


def set_face_down_orientation(env: ManagerBasedRlEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG):
    """Spawn belly-down with random yaw (+90° pitch about Y) for stand-up
    training.

    quat = quat_yaw * quat_pitch90, with s = sqrt(2)/2.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.int)
    num = len(env_ids)

    yaw = torch.rand(num, device=env.device) * 2 * np.pi - np.pi
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    s = 2.0**-0.5

    new_quat = torch.stack([s * cy, -s * sy, s * cy, s * sy], dim=1)

    env.sim.data.qpos[env_ids, 3:7] = new_quat
    env.sim.data.qvel[env_ids, :6] = 0.0


def set_random_prone_orientation(env: ManagerBasedRlEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, face_down_prob: float = 0.5):
    """Spawn face-down (±90° pitch) or face-up with random yaw.

    Face-up recovery is the harder half, so a curriculum ramps
    ``face_down_prob`` from high down toward 0.5.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.int)
    num = len(env_ids)

    yaw = torch.rand(num, device=env.device) * 2 * np.pi - np.pi
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    s = 2.0**-0.5

    face_down = torch.stack([s * cy, -s * sy, s * cy, s * sy], dim=1)
    face_up = torch.stack([s * cy, s * sy, -s * cy, s * sy], dim=1)

    mask = torch.rand(num, device=env.device) < face_down_prob
    new_quat = torch.where(mask.unsqueeze(1), face_down, face_up)

    env.sim.data.qpos[env_ids, 3:7] = new_quat
    env.sim.data.qvel[env_ids, :6] = 0.0


def set_random_ground_state(env: ManagerBasedRlEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, face_down_prob: float = 0.4, face_up_prob: float = 0.4, sitting_prob: float = 0.2, standing_prob: float = 0.0, prone_z_min: float = 0.20, prone_z_max: float = 0.25, sitting_z_min: float = 0.07, sitting_z_max: float = 0.09, standing_z_min: float = 0.11, standing_z_max: float = 0.12, sitting_joint_overrides: dict | None = None, sitting_joint_noise_std: float = 0.0, sitting_tilt_max: float = 0.0, face_up_roll_max: float = 0.0):
    """Reset to a random ground state: face-down, face-up, sitting, or standing.

    Broader than ``set_random_prone_orientation``: includes the sit keyframe
    (the sit policy's rest state, which standup takes over from) and an
    already-standing pose, so the policy also learns to HOLD a stand.

    Probabilities are normalized and need not sum to 1.
    ``sitting_joint_overrides`` is ``{servo_index: angle_rad}``.
    """
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
    # Tilt noise stops the policy overfitting to perfectly-upright starts.
    if sitting_tilt_max > 0.0:
        pitch = (torch.rand(num, device=env.device) * 2 - 1) * sitting_tilt_max
        roll = (torch.rand(num, device=env.device) * 2 - 1) * sitting_tilt_max
        cp = torch.cos(pitch * 0.5)
        sp = torch.sin(pitch * 0.5)
        cr = torch.cos(roll * 0.5)
        sr = torch.sin(roll * 0.5)
        # ZYX intrinsic Euler → quaternion.
        sit_w = cr * cp * cy + sr * sp * sy
        sit_x = sr * cp * cy - cr * sp * sy
        sit_y = cr * sp * cy + sr * cp * sy
        sit_z = cr * cp * sy - sr * sp * cy
        sitting = torch.stack([sit_w, sit_x, sit_y, sit_z], dim=1)
    else:
        sitting = torch.stack([cy, torch.zeros_like(cy), torch.zeros_like(cy), sy], dim=1)

    u = torch.rand(num, device=env.device)
    is_fd = u < p_fd
    is_fu = (u >= p_fd) & (u < p_fu)
    is_sit = (u >= p_fu) & (u < p_sit)
    is_stand = u >= p_sit

    # Face-up roll noise = built-in reverse curriculum for back recovery. The
    # reward landscape between supine and prone is FLAT (cos tilt ≈ 0 and height
    # constant through the whole roll), so rolling off the back only pays via
    # the front-rise that follows — a long-horizon dependency exploration
    # rarely finds from flat supine. Starting some spawns partway through the
    # roll teaches roll-completion, which generalizes back to flat. Uniform
    # sampling keeps every difficulty represented, so no annealing is needed.
    if face_up_roll_max > 0.0:
        theta = (torch.rand(num, device=env.device) * 2 - 1) * face_up_roll_max
        ct = torch.cos(theta * 0.5)
        st = torch.sin(theta * 0.5)
        # Log-roll is about body z (the spine), NOT body x: supine leaves body x
        # pointing skyward, so an x-roll would just spin the robot in place like
        # the yaw noise already does. Body-frame → right-multiply.
        w, x, y, z = face_up[:, 0], face_up[:, 1], face_up[:, 2], face_up[:, 3]
        face_up = torch.stack([w * ct - z * st, x * ct + y * st, y * ct - x * st, w * st + z * ct], dim=1)

    # Sitting and standing share the upright orientation, differing only in
    # trunk height and joint pose.
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

    # Override keys are SERVO indices, so translate through servo_ids or models
    # with interleaved passive_* joints write the wrong joints. qpos column =
    # 7 + entity joint index (free joint first, all hinges 1-dof).
    asset: Entity = env.scene[asset_cfg.name]
    servo_ids = _servo_joint_ids(env, asset)
    if sitting_joint_overrides:
        sit_env_ids = env_ids[is_sit]
        if len(sit_env_ids) > 0:
            for jnt_idx, angle in sitting_joint_overrides.items():
                env.sim.data.qpos[sit_env_ids, 7 + servo_ids[jnt_idx]] = angle

    # A distribution of sit starts, not one canonical pose: on the real robot
    # the joints won't match the SIT keyframe exactly when standup takes over.
    if sitting_joint_noise_std > 0.0:
        sit_env_ids = env_ids[is_sit]
        if len(sit_env_ids) > 0:
            # Servo joints only: passive_* backlash hinges have ~±1° ranges and
            # must stay at 0 on reset.
            n_sit = len(sit_env_ids)
            cols = torch.tensor([7 + j for j in servo_ids], device=env.device, dtype=torch.long)
            noise = torch.randn(n_sit, len(cols), device=env.device) * sitting_joint_noise_std
            env.sim.data.qpos[sit_env_ids.unsqueeze(1).long(), cols.unsqueeze(0)] += noise


# The "stuck" mid-recovery basin: knees folded under the body, trunk pitched
# forward, feet flat. Extends the HOME zig-zag (hip fwd / knee back / ankle fwd)
# to deep flexion, inside the ±1.57 joint limits; hip_yaw/hip_roll/neck at HOME.
_CROUCH_ANCHOR_BY_NAME = {"left_hip_pitch": -1.15, "left_knee": 1.25, "left_ankle": 1.05, "right_hip_pitch": 1.15, "right_knee": -1.25, "right_ankle": -1.05}


def set_random_crouch_state(env: ManagerBasedRlEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, depth_min: float = 0.35, depth_max: float = 1.0, pitch_max_deg: float = 55.0, joint_noise: float = 0.12, z_stand: float = 0.115, z_deep: float = 0.06):
    """Reset selected envs into a random mid-recovery crouch.

    Reverse curriculum for the recovery last mile: prone-init episodes burn
    their fallen budget getting TO the deep crouch and are recycled soon after,
    so crouch→stand gets almost no on-policy data and the policy converged to
    parking there. Spawning across that mile (depth λ scaling joints, pitch and
    z) makes the frontier dense from step 0.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.long)
    num = len(env_ids)
    asset: Entity = env.scene[asset_cfg.name]

    lam = torch.rand(num, device=env.device) * (depth_max - depth_min) + depth_min

    # Noise on servo joints only: passive_* backlash hinges have ~±1° ranges, so
    # noise there spawns them pinned outside their limits.
    joints = asset.data.default_joint_pos[env_ids].clone()
    for name, anchor in _CROUCH_ANCHOR_BY_NAME.items():
        ids, _ = asset.find_joints(f"^{name}$")
        j = ids[0]
        joints[:, j] = joints[:, j] + lam * (anchor - joints[:, j])
    noise_mask = torch.zeros(joints.shape[1], device=joints.device)
    noise_mask[_servo_joint_ids(env, asset)] = 1.0
    joints += (torch.rand_like(joints) * 2 - 1) * joint_noise * noise_mask

    # The stuck basin is a forward crouch from both fall directions.
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
    # ZYX intrinsic Euler → quaternion.
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    quat = torch.stack([qw, qx, qy, qz], dim=1)

    # Small upward margin so the spawn settles instead of interpenetrating.
    z = z_stand + lam * (z_deep - z_stand) + torch.rand(num, device=env.device) * 0.01

    env.sim.data.qpos[env_ids, 2] = z
    env.sim.data.qpos[env_ids, 3:7] = quat
    env.sim.data.qpos[env_ids, 7:] = joints
    env.sim.data.qvel[env_ids, :] = 0.0


def maybe_set_random_prone_orientation(env: ManagerBasedRlEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, prone_prob: float = 0.0, face_down_prob: float = 0.5, prone_z_min: float = 0.20, prone_z_max: float = 0.25, crouch_prob: float = 0.0):
    """Override reset_base's upright orientation with a prone one for a
    ``prone_prob`` slice of envs, and a mid-recovery crouch for ``crouch_prob``.

    Prone envs also need z lifted to [prone_z_min, prone_z_max]: the vel-env
    reset z (~0.125) clips the head through the ground at 90° pitch.

    prone_prob=2/3 with face_down_prob=0.5 gives the standard balanced
    33/33/33 upright / face-down / face-up mixture.
    """
    if prone_prob <= 0.0 and crouch_prob <= 0.0:
        return
    # env_ids=None means all envs; the initial global reset passes None, and an
    # early return here silently skipped prone init for it.
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if len(env_ids) == 0:
        return
    env_ids_t = env_ids.to(env.device, dtype=torch.long) if isinstance(env_ids, torch.Tensor) else torch.tensor(env_ids, device=env.device, dtype=torch.long)
    # One draw → exclusive prone / crouch / untouched slices.
    u = torch.rand(len(env_ids_t), device=env.device)
    selected = env_ids_t[u < prone_prob]
    crouch_selected = env_ids_t[(u >= prone_prob) & (u < prone_prob + crouch_prob)]
    if len(selected) > 0:
        set_random_prone_orientation(env, selected, asset_cfg=asset_cfg, face_down_prob=face_down_prob)
        z = torch.rand(len(selected), device=env.device) * (prone_z_max - prone_z_min) + prone_z_min
        env.sim.data.qpos[selected, 2] = z
    if len(crouch_selected) > 0:
        set_random_crouch_state(env, crouch_selected, asset_cfg=asset_cfg)


def event_param_curriculum(env: ManagerBasedRlEnv, env_ids: torch.Tensor, event_name: str, param_stages: list[dict]) -> torch.Tensor:
    """Shallow-merge an event term's params at scheduled steps.

    ``param_stages`` is [{"step", "params"}, ...]. Mutates the live EventManager
    term cfg; env.cfg.events is a deepcopy.
    """
    del env_ids
    event_cfg = env.event_manager.get_term_cfg(event_name)
    current = param_stages[0]["params"]
    for stage in param_stages:
        if env.common_step_counter >= stage["step"]:
            current = stage["params"]
    event_cfg.params.update(current)
    first_val = next(iter(current.values()))
    return torch.tensor(float(first_val) if isinstance(first_val, (int, float)) else 0.0)


def face_down_prob_curriculum(env: ManagerBasedRlEnv, env_ids: torch.Tensor, event_name: str, prob_stages: list[dict]) -> torch.Tensor:
    """Ramp ``face_down_prob`` through ``prob_stages`` ([{"step", "prob"}, ...]).

    Higher prob = more face-down resets = easier; ramp toward 0.5.
    """
    del env_ids

    # Must mutate the live EventManager term cfg: EventManager.__init__
    # deepcopies cfg, so writes to env.cfg.events are silent no-ops.
    event_cfg = env.event_manager.get_term_cfg(event_name)

    current_prob = prob_stages[0]["prob"]
    for stage in prob_stages:
        if env.common_step_counter > stage["step"]:
            current_prob = stage["prob"]

    event_cfg.params["face_down_prob"] = current_prob
    return torch.tensor([current_prob])


class VelocityCommandCommandOnly(UniformVelocityCommand):
    """UniformVelocityCommand that draws only the command arrows."""

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        # Turn-in-place needs its own bucket: independent uniform sampling makes
        # "lin≈0, |ang| large" ~2% of experience, so spinning on the spot went
        # untrained and real-robot turning was slow/unstable.
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
        # Un-mark as standing, which would zero the command we just wrote.
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
    # Fraction of envs forced to turn in place each resample; 0 = disabled.
    rel_turn_in_place_envs: float = 0.0

    def build(self, env: ManagerBasedRlEnv) -> "VelocityCommandCommandOnly":
        return VelocityCommandCommandOnly(self, env)


class RelativeHeadingVelocityCommand(VelocityCommandCommandOnly):
    """Velocity command whose cmd[2] slot carries body-frame HEADING ERROR
    instead of a yaw rate: cmd[0] = throttle, cmd[1] = 0, cmd[2] = heading error.

    Training samples a world-frame target heading per episode and recomputes the
    error each step; at inference the user writes cmd[2] directly, and holding it
    constant yields an approximately constant turn rate.

    The cfg MUST set heading_command=False and rel_heading_envs=0.0 (heading is
    handled here); ang_vel_z is reused as the clip limit for cmd[2].
    """

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
        # Deliberately NOT calling super(): its heading P-controller would
        # overwrite cmd[2] with a yaw rate.
        quat = self.robot.data.root_link_quat_w
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        current_yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        # Positive = target is CCW (left) of the robot.
        delta = self._target_heading_w - current_yaw
        heading_error = torch.atan2(torch.sin(delta), torch.cos(delta))
        self.vel_command_b[:, 2] = heading_error.clamp(-self._heading_max, self._heading_max)

    def _update_metrics(self) -> None:
        pass


class RelativeHeadingVelocityCommandCfg(UniformVelocityCommandCfg):
    def build(self, env: ManagerBasedRlEnv) -> "RelativeHeadingVelocityCommand":
        return RelativeHeadingVelocityCommand(self, env)


def heading_tracking_reward(env: ManagerBasedRlEnv, command_name: str, std: float = 0.5) -> torch.Tensor:
    """Gaussian on the heading error carried in cmd[2].

    std ≈ 0.5 rad keeps a usable gradient across the expected error range.
    """
    cmd = env.command_manager.get_command(command_name)
    heading_error = cmd[:, 2]
    return torch.exp(-(heading_error**2) / (std**2))


def skating_air_time_reward(env: ManagerBasedRlEnv, sensor_name: str, command_name: str, threshold_min: float = 0.05, threshold_max: float = 0.4, vel_gate_ref: float = 0.0) -> torch.Tensor:
    """Reward feet air time only when pushing (cmd_x > 0), so the recovery foot
    lifts instead of dragging.

    ``vel_gate_ref`` > 0 additionally requires forward progress, killing
    tap-dancing on the spot. Raise ``threshold_min`` to forbid high-cadence
    flutter.
    """
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
    """0→1 ramp in body forward speed; None when disabled (v_ref <= 0).

    Gates stride-FORM rewards on the stride doing its JOB, so stepping that does
    not propel the body earns nothing.
    """
    if v_ref <= 0.0:
        return None
    v_fwd = env.scene["robot"].data.root_link_lin_vel_b[:, 0]
    return (v_fwd.clamp(min=0.0) / v_ref).clamp(max=1.0)


def single_support_reward(env: ManagerBasedRlEnv, sensor_name: str, command_name: str, vel_gate_ref: float = 0.0, double_penalty: float = 0.25) -> torch.Tensor:
    """Reward single support (a real stride), mildly tax double support.

    wheel_speed alone converges to the swizzle: both blades stay grounded and
    the wheels still spin. Single support is what makes it a stride.

    The positive term is speed-gated so stepping in place earns nothing. The
    double-support tax is small and UNGATED on purpose — brief double support
    during weight transfer is normal skating, so only PERMANENT double support
    is discouraged; skating_air_time is the real anti-swizzle signal.
    """
    from mjlab.sensor import ContactSensor

    sensor: ContactSensor = env.scene[sensor_name]
    contact_time = sensor.data.current_contact_time
    assert contact_time is not None

    n_contact = torch.sum((contact_time > 0.0).float(), dim=1)
    single = (n_contact == 1).float()
    double = (n_contact >= 2).float()

    cmd_x = torch.clamp(env.command_manager.get_command(command_name)[:, 0], min=0.0)
    single_r = single * cmd_x
    gate = _forward_progress_gate(env, vel_gate_ref)
    if gate is not None:
        single_r = single_r * gate
    return single_r - double_penalty * double * cmd_x


def glide_reward(env: ManagerBasedRlEnv, sensor_name: str, command_name: str, vel_ref: float = 0.2, stillness_std: float = 5.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=(r".*(hip|knee|ankle).*",))) -> torch.Tensor:
    """Reward the GLIDE phase: coast on ONE blade with quiet legs, so the policy
    commits to each stroke instead of maximising swing frequency (which
    skating_air_time alone rewards — frantic kicking).

    The single-support factor is REQUIRED: without it a two-blade swizzle-coast
    farms this reward and the gait regresses. Silent on brake.
    """
    from mjlab.sensor import ContactSensor

    sensor: ContactSensor = env.scene[sensor_name]
    contact_time = sensor.data.current_contact_time
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
    """Reward left/right mirroring — the swizzle's defining symmetry.
    SELF-NEGATING (≤ 0) → POSITIVE weight.

    Mirrored L/R sign conventions mean a symmetric config satisfies
    q_left + q_right ≈ 0 per pair.
    """
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
    """Reward BOTH blades in contact — the swizzle never lifts a foot.

    Inverse of ``single_support_reward``, scaled by |cmd_x| so it shapes the push
    in either direction (the swizzle env drives cmd_x < 0 as "go backward").
    """
    from mjlab.sensor import ContactSensor

    sensor: ContactSensor = env.scene[sensor_name]
    contact_time = sensor.data.current_contact_time
    assert contact_time is not None
    n_contact = torch.sum((contact_time > 0.0).float(), dim=1)
    grounded = (n_contact >= 2).float()
    cmd_x = torch.abs(env.command_manager.get_command(command_name)[:, 0])
    return grounded * cmd_x


def gait_symmetry_penalty(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """Normalized CUMULATIVE swing-time imbalance |L−R|/(L+R) ∈ [0, 1]
    (≥ 0 → negative weight).

    With symmetry augmentation off, nothing stops a one-legged stride that veers
    and destabilises at launch. Only cumulative imbalance is taxed — the
    instantaneous asymmetry of a real stride is fine.
    """
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
    """Gaussian on yaw drift from the heading captured at reset.

    Angle-based, not yaw-RATE: penalising the rate just says "never turn", so
    the policy cannot steer back and drifts open-loop. Here drift costs reward
    and correcting it earns the reward back. Per-env reference makes it
    compatible with full-circle spawn-yaw randomization.
    """
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
    """Cost for commanding a joint target beyond its hard limit + ``overshoot``
    (≥ 0 → negative weight).

    Deters slamming a joint onto its stop with max torque: e.g. hip_roll has a
    ±0.38 rad limit but ±10 rad ctrlrange, and that trick is sim-only.

    Fires on the COMMAND, not qpos, so the joint keeps its full usable range.
    Constraining the policy's OUTPUT bakes the behaviour into the network and
    transfers without an env-side action clip (which would exist only in sim).
    ``overshoot`` preserves the headroom a low-kp servo needs under load.
    """
    term = env.action_manager.get_term(action_name)
    target = term.raw_action * term.scale + term.offset
    jnt_ids = term.target_ids
    hard = env.scene["robot"].data.joint_pos_limits[:, jnt_ids]
    lo = hard[..., 0] - overshoot
    hi = hard[..., 1] + overshoot
    over = (target - hi).clip(min=0.0) + (lo - target).clip(min=0.0)
    return torch.sum(over, dim=-1)


def forward_lean_reward(env: ManagerBasedRlEnv, command_name: str, target_pitch: float = 0.08, std: float = 0.08, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=("trunk_base",))) -> torch.Tensor:
    """Reward a slight forward lean while pushing, countering the backward
    torque of a skating stroke. Only fires when cmd_x > 0.
    """
    asset: Entity = env.scene[asset_cfg.name]
    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    forward_lean = asset.data.projected_gravity_b[:, 0]
    push = torch.clamp(cmd_x, min=0.0)
    return push * torch.exp(-((forward_lean - target_pitch) ** 2) / (std**2))


class GroundPickPhaseCommand(UniformVelocityCommand):
    """Cyclic phase command [cos(2π·phase), sin(2π·phase), 0] replacing the
    velocity command: phase ∈ [0, 0.5] descends, [0.5, 1.0] returns.
    """

    PERIOD: float = 4.0

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._gp_phase = torch.zeros(self.num_envs, device=self.device)
        self._period = float(getattr(cfg, "period", self.PERIOD))
        # False matches the runtime, where the button starts the cycle at phase 0
        # from standing; True (default) decorrelates envs.
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
    period: float = 4.0
    randomize_phase: bool = True

    def build(self, env: ManagerBasedRlEnv) -> "GroundPickPhaseCommand":
        return GroundPickPhaseCommand(self, env)


# ── Unified pose commands ─────────────────────────────────────────────────
# Head/body pose are COMMANDS (dense policy inputs with tracking rewards), not
# disturbances to be robust to — the retired NeckOffsetJointPositionAction
# approach trained only a weak indirect signal.
#
# The 13D command block is shared by every microduck policy so the runtime can
# feed them all the same buffer, giving a 61D actor obs:
#   [vx, vy, vtheta,                                    ← twist
#    neck_pitch, head_pitch, head_yaw, head_roll,       ← head_pose deltas
#    body_x, body_y, body_z, body_roll, body_pitch, body_yaw]  ← body_pose deltas


from dataclasses import dataclass


class UniformPoseCommand(CommandTerm):
    """Generic N-dim uniform pose command, held between resamples."""

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
        # Explicit zero bucket: uniform sampling essentially never produces the
        # all-zero command, so the deployment idle state would be untrained.
        if self.cfg.zero_command_prob > 0.0:
            zero_mask = torch.rand(n, device=self.device) < self.cfg.zero_command_prob
            self._command[env_ids[zero_mask]] = 0.0


@dataclass(kw_only=True)
class UniformPoseCommandCfg(CommandTermCfg):
    """Per-dim uniform ranges; builds a UniformPoseCommand."""

    # Length defines the command dim.
    ranges: tuple[tuple[float, float], ...] = ()
    # Probability a resample yields the exact all-zero command.
    zero_command_prob: float = 0.0

    def build(self, env: ManagerBasedRlEnv) -> "UniformPoseCommand":
        return UniformPoseCommand(self, env)


def zero_command_padding(env: ManagerBasedRlEnv, dim: int) -> torch.Tensor:
    """Constant-zero obs term of width `dim`.

    Envs that don't track head/body commands still must keep the slot so the
    obs stays 61D and policies remain hot-swappable. Never delete a slot.
    """
    return torch.zeros(env.num_envs, dim, device=env.device)


def head_pose_tracking(env: ManagerBasedRlEnv, command_name: str = "head_pose", std: float = 0.5, fine_std: float | None = None, fine_weight: float = 0.5, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Per-joint Gaussian on the commanded neck/head deltas, MEANED over the 4
    joints so one bad joint doesn't kill the whole reward (as SOS would).

    cmd is (N, 4) of deltas from default, ordered [neck_pitch, head_pitch,
    head_yaw, head_roll]. Keep ``std`` on the order of the command range so the
    gradient survives curriculum widening.

    ``fine_std`` blends in a narrow Gaussian: a single wide std makes small
    errors nearly free (a 10° sag on the heavy head costs ~0.03), so the policy
    lets the head droop. The narrow term prices that while the wide one keeps
    far commands learnable.

    On backlash models the measurement reads the OUTPUT link (servo + play), the
    same view as the encoder obs — measuring the servo alone would let the head
    droop the play for free AND penalize the policy for compensating it.
    """
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


# ── NaN-safe sensor-derived critic observations ──────────────────────────────
# `robot_state_is_nan` covers joint + root state, but not sensor data (raycast
# heights, contact air-time, contact forces), which MuJoCo can return
# non-finite while the integrated state is still clean. These are critic-only,
# so a sanitized step costs the policy nothing while an escaping NaN kills the
# run via rsl_rl's check_nan. Real blowups still terminate through nan_state.


def _finite(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def foot_contact_forces_safe(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """NaN-safe `foot_contact_forces` (see note above)."""
    return _finite(_velocity_obs.foot_contact_forces(env, sensor_name))


def foot_height_safe(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """NaN-safe `foot_height` (see note above)."""
    return _finite(_velocity_obs.foot_height(env, sensor_name))


def foot_air_time_safe(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """NaN-safe `foot_air_time` (see note above)."""
    return _finite(_velocity_obs.foot_air_time(env, sensor_name))


def head_pose_bias_penalty(env: ManagerBasedRlEnv, command_name: str = "head_pose", tau_s: float = 1.0, gate_height_low: float | None = None, gate_height_high: float = 0.11, gate_tilt_full_deg: float = 20.0, gate_tilt_zero_deg: float = 45.0, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Penalize the time-averaged (DC) neck/head tracking error: -mean(|EMA(err)|).

    Companion to ``head_pose_tracking``, which scores the INSTANTANEOUS error.
    Why a separate DC term instead of just tightening that Gaussian's std:
    walking unavoidably shakes a head that is 38% of the robot's mass, so an
    instantaneous tight-tolerance term is a permanent tax on walking that no
    policy can escape — measured at ~0.77/step against an air_time reward of
    ~1.01/step, which is exactly what made velocity run 2026-08-20 abandon
    stepping altogether (wandb 5yay13u4). The steady-state droop IS escapable:
    the policy can bias its neck command up to cancel gravity sag. Averaging
    over ``tau_s`` lets the oscillation cancel and prices only the bias.

    L1 (not Gaussian) on purpose: the gradient stays constant at large bias,
    where a tight Gaussian would be flat and dead.

    On backlash models the measured angle reads through the play, matching
    head_pose_tracking and the encoder obs.

    ``gate_height_low`` (optional): upright gate for recovery envs (standup /
    velstand), same smoothstep shape and semantics as body_ang_vel_at_height —
    zero below gate_height_low or above gate_tilt_zero_deg tilt, full above
    gate_height_high and below gate_tilt_full_deg. The gate multiplies the
    ERROR feeding the EMA (not just the output): while fallen/rising the EMA
    sees zero and decays, so arriving upright starts the bias clock from ~0
    instead of charging the whole ground phase's accumulated error at the
    finish line — that would be a reward wall right before recovery completes,
    the exact failure mode of the retired head_impact_penalty. The output is
    gated too, so a fresh fall stops the charge immediately.
    """
    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)  # (N, 4)

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
    fresh = env.episode_length_buf <= 1
    env._head_bias_ema[fresh] = 0.0

    alpha = min(1.0, float(env.step_dt) / max(tau_s, 1e-6))
    env._head_bias_ema = (1.0 - alpha) * env._head_bias_ema + alpha * err
    out = -env._head_bias_ema.abs().mean(dim=-1)
    if gate is not None:
        out = out * gate
    return out


def body_pose_tracking_6d(env: ManagerBasedRlEnv, command_name: str = "body_pose", nominal_height: float = 0.095, xy_std: float = 0.02, z_std: float = 0.01, angle_std: float = math.radians(8), asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Mean of 6 per-axis Gaussians tracking the commanded body pose.

    cmd is (N, 6) = [x, y, z, roll, pitch, yaw] as deltas from the nominal
    standing pose: xy from the spawn origin, z from ``nominal_height``, angles
    from upright.
    """
    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    dx, dy, dz = cmd[:, 0], cmd[:, 1], cmd[:, 2]
    droll, dpitch, dyaw = cmd[:, 3], cmd[:, 4], cmd[:, 5]

    pos_w = asset.data.root_link_pos_w
    origin = env.scene.terrain.env_origins
    rel = torch.nan_to_num(pos_w - origin, nan=0.0)
    x_err = rel[:, 0] - dx
    y_err = rel[:, 1] - dy
    z_err = rel[:, 2] - (nominal_height + dz)

    quat = asset.data.root_link_quat_w
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    roll = torch.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    pitch = torch.asin(torch.clamp(2.0 * (qw * qy - qz * qx), -1.0, 1.0))
    yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

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
    """Shallow-merge a termination term's params at scheduled steps.

    ``param_stages`` is [{"step", "params"}, ...]. TerminationManager deepcopies
    the cfg, so the live term_cfgs list must be edited directly;
    env.cfg.terminations is a no-op. Typical use is relaxing a termination late
    in training so the robot may fall and learn to recover.
    """
    del env_ids
    tm = env.termination_manager
    if term_name not in tm._term_names:
        # e.g. play mode disables fell_over entirely.
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
    """Locomotion-aware 6D body pose tracking.

    Like ``body_pose_tracking_6d``, but x/y/yaw are measured relative to the
    FEET (centroid in trunk body frame, yaw vs circular-mean foot yaw) instead
    of the spawn origin, whose gradient dies as soon as the robot translates or
    turns. z/roll/pitch stay in world frame, which is locomotion-neutral.
    """
    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    dx, dy, dz = cmd[:, 0], cmd[:, 1], cmd[:, 2]
    droll, dpitch, dyaw = cmd[:, 3], cmd[:, 4], cmd[:, 5]

    pos_w = asset.data.root_link_pos_w
    quat = asset.data.root_link_quat_w
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    trunk_yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    roll = torch.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    pitch = torch.asin(torch.clamp(2.0 * (qw * qy - qz * qx), -1.0, 1.0))

    foot_pos = asset.data.site_pos_w[:, feet_cfg.site_ids]
    foot_quat = asset.data.site_quat_w[:, feet_cfg.site_ids]
    feet_centroid = foot_pos.mean(dim=1)

    dx_w = pos_w[:, 0] - feet_centroid[:, 0]
    dy_w = pos_w[:, 1] - feet_centroid[:, 1]
    cos_y = torch.cos(trunk_yaw)
    sin_y = torch.sin(trunk_yaw)
    x_body = cos_y * dx_w + sin_y * dy_w
    y_body = -sin_y * dx_w + cos_y * dy_w

    origin = env.scene.terrain.env_origins
    z_world = torch.nan_to_num(pos_w[:, 2] - origin[:, 2], nan=0.0)

    # Assumes the foot site frame points along the foot; a rotated site adds a
    # constant per-env offset, so dyaw=0 still means "feet-aligned".
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

    # axis_weights=(0,0,1,1,1,1) disables xy tracking, which is worth doing when
    # xy lean is mechanically coupled to pitch/roll: independent xy commands are
    # then a noise source, not a learnable objective.
    wx, wy, wz, wr, wp, wyaw = axis_weights
    total_w = wx + wy + wz + wr + wp + wyaw
    reward = (wx * r_x + wy * r_y + wz * r_z + wr * r_r + wp * r_p + wyaw * r_w) / max(total_w, 1e-6)

    # Gating body tracking on a near-zero velocity command resolves the
    # tracking-vs-walking conflict that stopped a run learning either well.
    # Linear xy only: turning in place leaves body pose meaningful.
    if vel_gate_command_name is not None:
        vel_cmd = env.command_manager.get_command(vel_gate_command_name)
        vel_mag = torch.linalg.vector_norm(vel_cmd[:, :2], dim=-1)
        gate = torch.exp(-((vel_mag / vel_gate_std) ** 2))
        reward = reward * gate

    return reward


def pose_command_range_curriculum(env: ManagerBasedRlEnv, env_ids: torch.Tensor, command_name: str, range_stages: list[dict]) -> torch.Tensor:
    """Ramp a UniformPoseCommand's per-dim ranges over training.

    ``range_stages`` is [{"step", "ranges"}, ...]; the latest passed stage wins.
    Mutates the live CommandManager term cfg, which is what each resample reads;
    env.cfg.commands is a no-op.
    """
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


# ── Gait-shaping penalties (from the microban velocity recipe) ────────────────
def no_stepping_penalty(env: ManagerBasedRlEnv, sensor_name: str, command_name: str = "twist", command_threshold: float = 0.01) -> torch.Tensor:
    """Count of airborne feet while the commanded speed is below threshold
    (≥ 0 → negative weight): discourages marching in place.
    """
    command = env.command_manager.get_command(command_name)
    cmd_speed = torch.norm(command[:, :2], dim=-1) + torch.abs(command[:, 2])
    below_threshold = cmd_speed < command_threshold

    sensor = env.scene.sensors[sensor_name]
    found = sensor.data.found
    if found.dim() == 3:
        found = found.any(dim=-1)
    in_air = ~found.bool()

    return in_air.float().sum(dim=-1) * below_threshold.float()


def feet_distance_penalty(env: ManagerBasedRlEnv, min_dist: float, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Horizontal foot-separation shortfall below ``min_dist`` (≥ 0 → negative
    weight). Not wired into velocity yet.
    """
    asset: Entity = env.scene[asset_cfg.name]
    foot_pos_xy = asset.data.site_pos_w[:, asset_cfg.site_ids, :2]
    dist = torch.norm(foot_pos_xy[:, 0] - foot_pos_xy[:, 1], dim=-1)
    return torch.clamp(min_dist - dist, min=0.0)


# ── Non-accumulating domain randomization (restore-nominal-then-apply) ───────
# Older mjlab's randomize_field with operation="add"/"scale" + mode="reset" read
# the CURRENT model value without restoring nominal, so every reset STACKED the
# perturbation and the parameter random-walked away. On body_ipos this drifted
# the CoM centimetres off-centre over hundreds of resets and collapsed
# reward/episode length after the early peak. Always restore, then apply.
def randomize_com(env: ManagerBasedRlEnv, env_ids: torch.Tensor, ranges: tuple[float, float], field: str = "body_ipos", asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Randomize body CoM per episode WITHOUT accumulating.

    ``ranges`` is (lo, hi) applied to all 3 axes and is also what the com_range
    curriculum mutates. ``field`` must be declared as a param: mjlab reads it to
    expand that model field per-env under ``domain_randomization=True``.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, slice):
        body_ids = list(range(asset.num_bodies))[body_ids]
    body_indices = asset.indexing.body_ids[body_ids]

    mf = getattr(env.sim.model, field)
    # Cache key must include the body set: trunk and head both randomize
    # body_ipos, and a shared attr would collide (their body counts differ).
    _bidx = body_indices.tolist() if hasattr(body_indices, "tolist") else list(body_indices)
    cache_attr = f"_original_{field}_" + "_".join(str(int(i)) for i in _bidx)
    # model[0] is still nominal on the first call.
    if not hasattr(env, cache_attr):
        setattr(env, cache_attr, mf[0, body_indices].clone())
    nominal = getattr(env, cache_attr)

    num_envs = len(env_ids)
    num_bodies = len(body_indices)

    mf[env_ids[:, None], body_indices] = nominal.unsqueeze(0).expand(num_envs, -1, -1)
    lo, hi = ranges
    offsets = torch.rand(num_envs, num_bodies, 3, device=env.device) * (hi - lo) + lo
    mf[env_ids[:, None], body_indices] += offsets
    return torch.tensor(float(hi))


def randomize_dof_field_scaled(env: ManagerBasedRlEnv, env_ids: torch.Tensor, field: str, scale_range: tuple[float, float], asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Scale a per-dof model field per episode WITHOUT accumulating.

    ``field`` doubles as the domain_randomization field name. Under BAM,
    dof_frictionloss/dof_damping are zeroed in edit_spec, so scaling them is a
    SILENT NO-OP — they matter only with the XML position actuator.
    """
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


# ── BallKick ──────────────────────────────────────────────────────────


def _ball_kick_dir(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Per-env world-frame kick direction (XY unit vector), lazily allocated.

    Frozen for the episode so the policy cannot redefine "forward" by turning
    after the kick.
    """
    if not hasattr(env, "_ball_kick_dir_w"):
        env._ball_kick_dir_w = torch.zeros(env.num_envs, 2, device=env.device)
        env._ball_kick_dir_w[:, 0] = 1.0
    return env._ball_kick_dir_w


def reset_ball_in_front_of_foot(env: ManagerBasedRlEnv, env_ids: torch.Tensor, offset: tuple = (0.09, -0.042), noise_xy: float = 0.015, ball_radius: float = 0.035, asset_name: str = "ball"):
    """Place the ball in front of the right foot; store the kick direction.

    ``offset`` is the nominal ball centre in the robot's yaw frame: at HOME the
    right foot sits at (0, -0.042) with its toe tip at x≈0.034. ``noise_xy`` is
    the placement DR that matters most — the policy is BLIND to the ball, so it
    is what forces a swing robust to real placement error.

    Reads the root from qpos because root_link_pos_w lags until the next
    forward(), and MUST be registered after reset_base / set_ground_state
    (events run in insertion order) so the robot pose is final.
    """
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
    """Ball XY velocity along the per-env kick direction, clamped to [0, max].

    Dense and linear in speed, so exploration nudges bootstrap the kick without
    peak detection. Backward/lateral motion earns 0 rather than a penalty — a
    mis-hit must not scare the policy off contacting the ball at all.

    Saturating at a TARGET speed does NOT by itself remove "harder is better":
    a harder kick holds the ball above the cap for more steps, so the integral
    still grows. ``ball_speed_overshoot_penalty`` is what makes the target
    optimal.
    """
    ball: Entity = env.scene[asset_name]
    vel_xy = ball.data.root_link_lin_vel_w[:, :2]
    fwd = (vel_xy * _ball_kick_dir(env)).sum(dim=1)
    return torch.nan_to_num(fwd, nan=0.0).clamp(0.0, max_speed)


def ball_speed_overshoot_penalty(env: ManagerBasedRlEnv, asset_name: str = "ball", target_speed: float = 1.0, max_penalty: float = 5.0) -> torch.Tensor:
    """Ball forward speed above ``target_speed`` (≥ 0 → negative weight).

    Keep |weight| BELOW ``ball_forward_velocity``'s weight so the landscape
    peaks at the target with a gentler overshoot slope: erring slightly hard
    must stay cheaper than not kicking at all.
    """
    ball: Entity = env.scene[asset_name]
    vel_xy = ball.data.root_link_lin_vel_w[:, :2]
    fwd = (vel_xy * _ball_kick_dir(env)).sum(dim=1)
    over = torch.nan_to_num(fwd, nan=0.0) - target_speed
    return over.clamp(0.0, max_penalty)


def single_foot_grounded_reward(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """1 while the sensed foot touches the terrain.

    Pins the SUPPORT foot during a kick (anti-hop): swinging the kicking leg is
    free, lifting the support foot forfeits this every step.
    """
    if sensor_name not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)
    found = env.scene.sensors[sensor_name].data.found
    if found.dim() > 1:
        found = found.sum(dim=-1)
    return torch.clamp(found, 0.0, 1.0)


def ball_pos_in_base(env: ManagerBasedRlEnv, asset_name: str = "ball") -> torch.Tensor:
    """Ball position in the robot's base frame.

    CRITIC-ONLY: the deployed robot has no ball sensing, so the actor must stay
    blind; the critic may still use it to predict the kick payoff.
    """
    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene[asset_name]
    rel = ball.data.root_link_pos_w - robot.data.root_link_pos_w
    rot = matrix_from_quat(robot.data.root_link_quat_w)
    return torch.bmm(rot.transpose(1, 2), rel.unsqueeze(-1)).squeeze(-1)


def ball_vel_in_base(env: ManagerBasedRlEnv, asset_name: str = "ball") -> torch.Tensor:
    """Ball linear velocity in the robot's base frame. CRITIC-ONLY (see above)."""
    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene[asset_name]
    rot = matrix_from_quat(robot.data.root_link_quat_w)
    vel = ball.data.root_link_lin_vel_w
    return torch.bmm(rot.transpose(1, 2), vel.unsqueeze(-1)).squeeze(-1)


# ── Spin: fast in-place rotation on rollers ────────────────────────────────
# The phase command drives a trapezoid target YAW RATE (launch / cruise / brake
# / rest), not a pose. Envelope area per cycle = 2.1 · SPIN_RATE_MAX rad, so
# 3.0 rad/s gives ~1 turn per cycle.
SPIN_PERIOD = 4.0
SPIN_RATE_MAX = 3.0
SPIN_ACCEL_END = 0.125
SPIN_HOLD_END = 0.525
SPIN_BRAKE_END = 0.650


def spin_rate_by_phase(phase: torch.Tensor, rate_max: float = SPIN_RATE_MAX, accel_end: float = SPIN_ACCEL_END, hold_end: float = SPIN_HOLD_END, brake_end: float = SPIN_BRAKE_END) -> torch.Tensor:
    """Target yaw rate (rad/s, positive = counter-clockwise) along the phase."""
    w = torch.zeros_like(phase)
    accel = phase < accel_end
    w = torch.where(accel, rate_max * phase / accel_end, w)
    hold = (phase >= accel_end) & (phase < hold_end)
    w = torch.where(hold, torch.full_like(phase, rate_max), w)
    brake = (phase >= hold_end) & (phase < brake_end)
    w = torch.where(brake, rate_max * (1.0 - (phase - hold_end) / (brake_end - hold_end)), w)
    return w


def spin_gate_by_phase(phase: torch.Tensor, rate_max: float = SPIN_RATE_MAX, accel_end: float = SPIN_ACCEL_END, hold_end: float = SPIN_HOLD_END, brake_end: float = SPIN_BRAKE_END) -> torch.Tensor:
    """Normalized envelope ∈ [0, 1] used as a shaping gate.

    Zero through the rest segment, so the primers (leg scissoring, wheel
    differential) apply only during launch and cruise and the robot returns to a
    neutral stance before control hands back to the roller policy.
    """
    return spin_rate_by_phase(phase, rate_max, accel_end, hold_end, brake_end) / rate_max


def spin_phase_from_command(cmd: torch.Tensor) -> torch.Tensor:
    """Recovers the phase [0,1) from the slot command [cos(2πφ), sin(2πφ), 0]."""
    return (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * torch.pi)) % 1.0


def _spin_target_rate(env: ManagerBasedRlEnv, command_name: str, rate_max: float, accel_end: float, hold_end: float, brake_end: float) -> torch.Tensor:
    phase = spin_phase_from_command(env.command_manager.get_command(command_name))
    return spin_rate_by_phase(phase, rate_max, accel_end, hold_end, brake_end)


def _spin_gate(env: ManagerBasedRlEnv, command_name: str, rate_max: float, accel_end: float, hold_end: float, brake_end: float) -> torch.Tensor:
    phase = spin_phase_from_command(env.command_manager.get_command(command_name))
    return spin_gate_by_phase(phase, rate_max, accel_end, hold_end, brake_end)


def spin_rate_reward_from_values(omega_z: torch.Tensor, omega_target: torch.Tensor, std: float) -> torch.Tensor:
    """Gaussian on the yaw rate error (pure, testable function)."""
    return torch.exp(-(((omega_z - omega_target) / std) ** 2))


def spin_rate_track(env: ManagerBasedRlEnv, command_name: str = "twist", std: float = 1.5, rate_max: float = SPIN_RATE_MAX, accel_end: float = SPIN_ACCEL_END, hold_end: float = SPIN_HOLD_END, brake_end: float = SPIN_BRAKE_END, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Main spin objective: track the target yaw rate ω*(φ).

    ω_z in BODY frame, which is what the IMU gyro sees and the policy observes.
    The Gaussian centres on a positive target, so spinning the wrong way scores
    worse than standing still.
    """
    asset: Entity = env.scene[asset_cfg.name]
    omega_z = asset.data.root_link_ang_vel_b[:, 2]
    target = _spin_target_rate(env, command_name, rate_max, accel_end, hold_end, brake_end)
    return spin_rate_reward_from_values(omega_z, target, std)


def spin_rate_l1(env: ManagerBasedRlEnv, command_name: str = "twist", rate_max: float = SPIN_RATE_MAX, accel_end: float = SPIN_ACCEL_END, hold_end: float = SPIN_HOLD_END, brake_end: float = SPIN_BRAKE_END, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """L1 companion to ``spin_rate_track``. SELF-NEGATING (≤ 0) → POSITIVE
    weight.

    Constant gradient where the Gaussian has saturated far from target.
    """
    asset: Entity = env.scene[asset_cfg.name]
    omega_z = asset.data.root_link_ang_vel_b[:, 2]
    target = _spin_target_rate(env, command_name, rate_max, accel_end, hold_end, brake_end)
    return -torch.abs(omega_z - target)


SPIN_LAUNCH_DRIFT_SCALE = 0.2


def spin_stay_in_place(env: ManagerBasedRlEnv, command_name: str = "twist", launch_scale: float = SPIN_LAUNCH_DRIFT_SCALE, accel_end: float = SPIN_ACCEL_END, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Trunk ‖v_xy‖² cost — turn IN PLACE and kill entry momentum
    (≥ 0 → negative weight).

    Stateless (no since-reset reference), so it stays valid across all cycles of
    an episode.

    Attenuated during the launch: there the robot must push off the ground and
    convert entry speed into rotation, so full-price translation cost would
    oppose the objective. Full price afterwards, where "in place" is the real
    criterion.

    Deliberately NOT gated by ``spin_gate_by_phase``: the rest segment is
    exactly when stillness must stay priced.
    """
    asset: Entity = env.scene[asset_cfg.name]
    v_xy = asset.data.root_link_lin_vel_b[:, :2]
    cost = torch.sum(torch.square(v_xy), dim=1)

    phase = spin_phase_from_command(env.command_manager.get_command(command_name))
    scale = torch.where(phase < accel_end, torch.full_like(cost, launch_scale), torch.ones_like(cost))
    return cost * scale


# = 2 · SPIN_RATE_MAX · half_track / r, with half_track 0.0499 m MEASURED on the
# rollers model at HOME (not the 0.03 m the spec estimated) and r = 0.0175 m.
# Must be recomputed whenever SPIN_RATE_MAX changes: a stale, too-large scale
# caps tanh() well below 1 and silently weakens this shaping term.
SPIN_WHEEL_OMEGA_SCALE = 17.0


def spin_wheel_differential_from_values(diff: torch.Tensor, gate: torch.Tensor, omega_scale: float) -> torch.Tensor:
    """tanh of the wheel differential, gated, clamped ≥ 0."""
    return gate * torch.tanh(torch.clamp(diff, min=0.0) / omega_scale)


def spin_wheel_differential(env: ManagerBasedRlEnv, command_name: str = "twist", omega_scale: float = SPIN_WHEEL_OMEGA_SCALE, rate_max: float = SPIN_RATE_MAX, accel_end: float = SPIN_ACCEL_END, hold_end: float = SPIN_HOLD_END, brake_end: float = SPIN_BRAKE_END) -> torch.Tensor:
    """Reward rotating BY ROLLING rather than skidding.

    A counter-clockwise spin runs the left skate backwards and the right
    forwards, and all 4 wheels spin positive for forward, so ω_R − ω_L > 0.
    tanh saturation stops a race for wheel speed.
    """
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
    """Both blades grounded during the spin — blocks "jump and twirl".

    The swizzle's ``grounded_reward`` is not reusable here: it weights by cmd_x,
    which on a phase command is cos(2πφ).
    """
    from mjlab.sensor import ContactSensor

    sensor: ContactSensor = env.scene[sensor_name]
    contact_time = sensor.data.current_contact_time
    assert contact_time is not None
    n_contact = torch.sum((contact_time > 0.0).float(), dim=1)
    grounded = (n_contact >= 2).float()
    gate = _spin_gate(env, command_name, rate_max, accel_end, hold_end, brake_end)
    return grounded * gate


def leg_antisymmetry(env: ManagerBasedRlEnv, command_name: str = "twist", asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, joint_bases: tuple = ("hip_pitch", "knee"), rate_max: float = SPIN_RATE_MAX, accel_end: float = SPIN_ACCEL_END, hold_end: float = SPIN_HOLD_END, brake_end: float = SPIN_BRAKE_END) -> torch.Tensor:
    """Prime the leg SCISSOR (one forward, one back) during the spin.
    SELF-NEGATING (≤ 0) → POSITIVE weight.

    Mirrored L/R sign conventions mean a symmetric pose has q_L + q_R ≈ 0, so a
    scissor has q_L ≈ q_R. Decay this by curriculum so the primer fades and the
    policy refines its own gesture.
    """
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


# ── Backlash model: encoder-through-backlash joint observations ──────────────
# The backlash model puts an unactuated ``passive_<joint>_backlash`` hinge in
# series with each servo, so the link angle is qpos[servo] + qpos[backlash] and
# the real encoder — sitting on the OUTPUT side of the play — reads that sum.
# These replace joint_pos_rel / joint_vel_rel in backlash tasks so the policy
# sees what the runtime will feed it. asset_cfg must select servo joints only.


def _backlash_encoder_ids(env: "ManagerBasedRlEnv", asset: Entity, asset_cfg: SceneEntityCfg) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(main_ids, backlash_ids, mask) — cached per (entity, joint selection).

    mask is 1.0 where a matching passive_<name>_backlash joint exists, so the
    same obs functions run unchanged on models without backlash joints.
    """
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
    """joint_pos_rel read through the backlash hinge.

    ``biased`` applies the per-env encoder-calibration bias to the servo
    reading only: one encoder per servo, so the backlash summand stays raw.
    """
    asset: Entity = env.scene[asset_cfg.name]
    main_ids, bl_ids, mask = _backlash_encoder_ids(env, asset, asset_cfg)
    joint_pos = asset.data.joint_pos_biased if biased else asset.data.joint_pos
    pos = joint_pos[:, main_ids] + asset.data.joint_pos[:, bl_ids] * mask
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    return pos - default_joint_pos[:, main_ids]


def joint_vel_rel_backlash(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """joint_vel_rel read through the backlash hinge.

    The firmware derives present_velocity from encoder positions, so it sees the
    backlash motion too.
    """
    asset: Entity = env.scene[asset_cfg.name]
    main_ids, bl_ids, mask = _backlash_encoder_ids(env, asset, asset_cfg)
    vel = asset.data.joint_vel[:, main_ids] + asset.data.joint_vel[:, bl_ids] * mask
    default_joint_vel = asset.data.default_joint_vel
    assert default_joint_vel is not None
    return vel - default_joint_vel[:, main_ids]


# ── Sit↔Stand posture command + posture-conditioned rewards ─────────────────
# One policy, both directions: a single sit/stand flag rides in the twist slot
# (cmd = [sit_flag, 0, 0]), so "stand" is the all-zero deployment idle command
# every other policy also uses. Every task reward below picks its target (SIT
# keyframe + SIT_Z vs HOME + STAND_Z) from the live command per env, so one
# reward stack drives descent, seated rest, rise and standing rest.


class SitStandCommand(UniformVelocityCommand):
    """Posture command cmd = [sit_flag, 0, 0] with dwell-time resampling and a
    SLEWED internal target blend.

    ``sit_prob`` is the chance a resample commands SIT; combined with the
    reset-state mix this trains all four (start-state × command) cases,
    including "hold what you're already doing".

    ``alpha`` (0 = STAND, 1 = SIT) slews toward the flag over ``ramp_s`` and is
    what the posture_* rewards track. THIS is the anti-crash mechanism: against
    a binary target, arriving early collects the goal-state jackpot for every
    step saved while speed-cap penalties integrate to a bounded cost, and an
    instant drop beat a 1 s descent by ~7×. Tracking a slewed setpoint pays ~0
    for being AHEAD of the ramp, so slow IS the argmax; the caps stay as
    overshoot backstops. The OBS remains the raw binary flag, matching the
    runtime, and the trained response to a flip is the ~ramp_s glide.
    """

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
        """Slewed target blend: 0 = STAND target, 1 = SIT target."""
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
        # Re-init the blend from the ACTUAL spawn height, so a seated spawn is
        # not dragged upward by a stand-initialised ramp. Must happen here, not
        # in reset(): the command manager resets BEFORE set_ground_state
        # teleports the robot, so reset() would read the pre-teleport height.
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
    # Seconds for the internal blend to traverse STAND↔SIT in full.
    ramp_s: float = 2.0
    # Rest heights; also used to initialise the blend from the spawn state.
    sit_z: float = 0.060
    stand_z: float = 0.115

    def build(self, env: ManagerBasedRlEnv) -> "SitStandCommand":
        return SitStandCommand(self, env)


def _posture_blend(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Target blend ∈ [0, 1] (0 = STAND, 1 = SIT) for the posture rewards.

    Prefers the slewed ``alpha`` setpoint; falls back to the raw binary flag.
    """
    term = env.command_manager.get_term(command_name)
    alpha = getattr(term, "alpha", None)
    if alpha is not None:
        return alpha
    return env.command_manager.get_command(command_name)[:, 0]


def _posture_targets(env: ManagerBasedRlEnv, asset: Entity, command_name: str, sit_overrides: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """(target blend, per-env joint target) for the commanded posture.

    The slewed blend interpolates HOME ↔ the SIT keyframe, so mid-ramp the
    rewarded pose folds in sync with the descending height.
    """
    blend = _posture_blend(env, command_name)
    stand_target = _servo_default_joint_pos(env, asset)
    sit_target = stand_target.clone()
    for idx, val in sit_overrides.items():
        sit_target[:, idx] = val
    target = stand_target + blend.unsqueeze(-1) * (sit_target - stand_target)
    return blend, target


def _posture_height(env: ManagerBasedRlEnv, command_name: str, sit_z: float, stand_z: float) -> tuple[torch.Tensor, torch.Tensor]:
    """(slewed target trunk z, actual trunk z) per env."""
    blend = _posture_blend(env, command_name)
    target_z = stand_z + blend * (sit_z - stand_z)
    asset = env.scene["robot"]
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    return target_z, z


def posture_pose_match(env: ManagerBasedRlEnv, command_name: str, sit_overrides: dict, joint_indices: list, std: float = 0.5, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Gaussian pose-match against the commanded posture's target pose."""
    asset = env.scene[asset_cfg.name]
    _, target = _posture_targets(env, asset, command_name, sit_overrides)
    joint_pos = _servo_joint_pos(env, asset)[:, joint_indices]
    target = target[:, joint_indices]
    return torch.exp(-(((joint_pos - target) / std) ** 2)).mean(dim=-1)


def posture_pose_l1(env: ManagerBasedRlEnv, command_name: str, sit_overrides: dict, joint_indices: list, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """L1 companion to ``posture_pose_match``. SELF-NEGATING (≤ 0) → POSITIVE
    weight.
    """
    asset = env.scene[asset_cfg.name]
    _, target = _posture_targets(env, asset, command_name, sit_overrides)
    joint_pos = _servo_joint_pos(env, asset)[:, joint_indices]
    target = target[:, joint_indices]
    return -torch.abs(joint_pos - target).mean(dim=-1)


def posture_height_gaussian(env: ManagerBasedRlEnv, command_name: str, sit_z: float, stand_z: float, std: float = 0.02, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Gaussian on trunk z against the commanded posture's target height."""
    del asset_cfg
    target_z, z = _posture_height(env, command_name, sit_z, stand_z)
    return torch.exp(-(((z - target_z) / std) ** 2))


def posture_height_l1(env: ManagerBasedRlEnv, command_name: str, sit_z: float, stand_z: float, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """L1 companion to ``posture_height_gaussian`` — the transition driver.
    SELF-NEGATING (≤ 0) → POSITIVE weight.

    Resting in the WRONG posture charges a constant per-step cost, which is what
    makes "ignore the command" net-negative in both directions.
    """
    del asset_cfg
    target_z, z = _posture_height(env, command_name, sit_z, stand_z)
    return -torch.abs(z - target_z)


def posture_composite(env: ManagerBasedRlEnv, command_name: str, sit_overrides: dict, joint_indices: list, sit_z: float, stand_z: float, height_std: float = 0.03, upright_std: float = 0.40, pose_std: float = 0.40, head_std: float | None = None, head_command_name: str = "head_pose", asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Multiplicative goal score vs the commanded posture (height·upright·pose
    [·head]).

    The posture-conditioned version of ``standing_composite_score``: a
    deficiency in any factor collapses the whole term, so partial-sum
    compromises (plank, flop, lean) never pay. Both rest states demand an
    upright trunk, so the upright factor is posture-independent.

    ``head_std`` (optional): adds a fourth factor on the neck/head joints vs
    the ``head_pose`` command (same error convention as head_pose_tracking).
    Without it the goal state is head-blind: the trained policy rested with
    the head dangling to the floor — trunk upright, legs in pose, z on target
    all held while the head hung, costing only the light tracking term. With
    the factor, "arrived" REQUIRES the head at its commanded pose, so head
    assist stays free mid-transition (composite is ≈0 there anyway) but must
    be retracted to collect the goal reward.
    """
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
    """Reward trunk stillness while AT the commanded posture, upright.

    Generalizes ``seated_stillness`` to both rest states: exp(-(|v|/std)²)
    gated by a smoothstep on |z − commanded z| (full inside ``band_full``,
    zero beyond ``band_zero`` → inactive during transitions) and by trunk
    tilt (a tilted rest — back/face/side — earns nothing). Additionally gated
    on the target ramp being COMPLETE (|flag − alpha| small), so stillness
    never pays mid-transition. Makes "rest quietly, upright, at the commanded
    height" the peak of the stack.
    """
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
    """Upward-vz reward, active only when STAND is commanded and z < max_height.

    The standup-env lesson: destination-only rewards have zero gradient at
    zero motion, so "stay seated and eat the L1" is a local optimum — paying
    for the rise *motion* itself makes any attempt immediately positive.
    Gated off above ``max_height`` (set just ABOVE the stand target so the
    final cm still pays; gating at exactly STAND_Z parks the policy short).
    Zero whenever SIT is commanded, so it can never fight the descent.
    ``max_vz`` caps the rewarded speed (any rise ≥ the cap earns the same, so
    an explosive launch can't out-earn a gentle one).
    """
    asset = env.scene[asset_cfg.name]
    sit = env.command_manager.get_command(command_name)[:, 0]
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    vz = torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0)
    return torch.clamp(vz, min=0.0, max=max_vz) * (z < max_height).float() * (1.0 - sit)


def trunk_upward_velocity_penalty(env: ManagerBasedRlEnv, max_up_vel: float = 0.08, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Penalty on upward trunk velocity beyond ``max_up_vel``.

    Mirror of ``trunk_downward_velocity_penalty`` for the rise: charges every
    step of a too-fast (violent) stand-up, so the explosive rise can't be
    amortised against arriving-standing reward. Zero at rest, for any rise
    slower than the cap, and for all downward motion. Introduce via
    curriculum AFTER the rise is discovered (attempt-tax lesson).
    """
    asset = env.scene[asset_cfg.name]
    vz = torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0)
    return -torch.clamp(vz - max_up_vel, min=0.0)


# ==============================================================================
# Roulade (forward roll) task — episodic dynamic maneuver
# ==============================================================================
#
# Third attempt at the roulade. What the first two taught us:
#   • origin/roulade (phase-clock + time-windowed reward stages): plateaued
#     face-down at ~90° — time windows are keyframes-in-time, campable local
#     optima (the sit/standup lesson exactly). Also integrated -ω_y as forward
#     progress, which by this codebase's own convention (face-down = +90° pitch
#     = rotation about +y, see set_random_ground_state) is the WRONG SIGN — the
#     progress reward paid for backward rotation.
#   • origin/roulade later commits (keyframe imitation): same waypoint-camping
#     family, dropped per feedback-episodic-pose-landing.
#
# This design uses the proven episodic recipe instead:
#   • ONE dense progress signal: paid INCREMENTS of the max-so-far cumulative
#     forward rotation (potential-based — a camping policy earns zero/step, a
#     full roll earns exactly 2π worth no matter the path or speed).
#   • Landing rewards (composite product, upright, height, rise velocity) are
#     gated on ROLL COMPLETION (max rotation ≥ threshold) — state-based gates,
#     not clock-based. "Do nothing" earns nothing; standing at spawn earns
#     nothing; only rolling opens the standing-attractor annuity.
#   • Reverse curriculum via mid-roll spawns (the face-up partial-roll trick
#     that fixed back-recovery): a slice of episodes starts pitched 50°–185°
#     into the roll, tucked, optionally with forward angular momentum, and the
#     rotation accumulator is initialized to the spawn angle so the progress
#     accounting stays consistent.
#
# RUN-1 LESSON (2026-08): with unsupported rotation counting and uncapped
# paid rate, the optimal policy is a violent ballistic whip ("breakdance") —
# same 2π, finishes sooner, more discounted annuity. Doesn't transfer. Fixes:
#   • SUPPORT GATE: the accumulator only integrates while some robot geom
#     touches the terrain (robot_ground_contact sensor) — a real roulade never
#     leaves the ground; airborne rotation now earns nothing and cannot open
#     the completion gate.
#   • HEAD LATCH: the landing annuity additionally requires head-ground
#     contact to have occurred while accum was in the first-quadrant window —
#     "went over the head" is a requirement, not a 0.5-weight suggestion.
#   • PAID-RATE CAP: progress increments are capped at max_paid_rate; rotation
#     faster than the cap FORFEITS the excess (not deferred), so speed no
#     longer pays. An explicit overspeed penalty backs this up.
#
# Per-env state on the env object (created lazily, reset by
# reset_roulade_state):
#   env._roulade_accum      — supported-only integral of forward pitch rate (rad)
#   env._roulade_max        — max(accum) so far this episode (progress frontier)
#   env._roulade_paid       — frontier already paid out by roulade_progress
#   env._roulade_head_latch — True once the head touched ground mid-first-quadrant

# Forward-roll sign: face-down is +90° pitch = rotation about body +y
# (set_random_ground_state convention), so forward roll = POSITIVE body-frame
# ω_y. Verified empirically (see claude_experiments smoke test): a positive
# qvel about +y pitches the robot nose-down/forward and drives accum upward.
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
    """World-z component of the body's lateral (y) axis. 0 = flat/sagittal."""
    return 2.0 * (quat[:, 2] * quat[:, 3] + quat[:, 0] * quat[:, 1])


def _head_top_down(env: ManagerBasedRlEnv, asset: Entity) -> torch.Tensor:
    """True where the head-top axis points at the floor (dot with -z > min)."""
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
    """Integrate forward pitch rate into the per-env rotation accumulator.

    Step-guarded so that multiple reward terms reading the accumulator in the
    same control step don't double-integrate. The frontier (max) only moves
    forward; backward rocking (wind-up) neither pays nor un-pays.

    SUPPORT GATE (run-1 fix): rotation is integrated only while the robot
    touches the terrain — a roulade is a supported motion; ballistic flips
    accumulate nothing, so they neither get paid nor open the completion gate.

    Also latches env._roulade_head_latch when the head touches the ground
    while accum is inside the first-quadrant window — the landing annuity
    requires this, making "over the head" a hard requirement of the task.
    """
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
    """Smoothstep on the progress frontier: 0 below gate_lo rad, 1 above gate_hi.

    State-based replacement for the old phase-clock landing window — it can
    only be opened by actually rotating (while SUPPORTED — the accumulator is
    contact-gated), so neither pre-roll standing nor a ballistic flip collects.
    With require_head=True the gate additionally requires the head latch —
    the episode must have rolled over the head to unlock the landing annuity.
    """
    _, max_accum, _ = _roulade_state(env)
    t = torch.clamp((max_accum - gate_lo) / max(gate_hi - gate_lo, 1e-6), 0.0, 1.0)
    gate = t * t * (3.0 - 2.0 * t)
    if require_head:
        gate = gate * env._roulade_head_latch.float()
    return gate


def reset_roulade_state(env: ManagerBasedRlEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, standing_prob: float = 0.5, midroll_prob: float = 0.5, standing_z_min: float = 0.11, standing_z_max: float = 0.12, standing_tilt_max: float = 0.0, forward_vel_range: tuple = (0.0, 0.0), midroll_pitch_min: float = math.radians(50.0), midroll_pitch_max: float = math.radians(185.0), midroll_z_min: float = 0.05, midroll_z_max: float = 0.10, midroll_omega_range: tuple = (0.0, 0.0), tuck_overrides: dict | None = None, tuck_factor_range: tuple = (0.3, 1.0), joint_noise_std: float = 0.0):
    """Reset to a standing start or a mid-roll state (reverse curriculum).

    Standing bucket: upright (±standing_tilt_max pitch/roll noise), random yaw,
    HOME joints (left from reset_robot_joints), z in [standing_z_min, _max].
    ``forward_vel_range`` is the run-up hook: a per-env forward base velocity
    (body x, mapped to world through the spawn yaw) sampled uniformly — 0 for
    a standstill roll, widen it later to train rolls out of a walk.

    Mid-roll bucket: pitched ``midroll_pitch_min..max`` into the roll (90° =
    on the head, 180° = on the back), random yaw, legs lerped HOME→tuck by a
    per-env factor in ``tuck_factor_range``, z in [midroll_z_min, _max],
    optional forward angular momentum from ``midroll_omega_range``. The
    rotation accumulator is initialized to the spawn pitch so progress
    accounting (and the completion gates) stay consistent: a 170° spawn only
    gets paid for the remaining ~190°.
    """
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

    # Pitch per bucket: small noise for standing, mid-roll angle otherwise.
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

    # Mid-roll joints: lerp HOME → tuck on the overridden joints, noise on all
    # servo joints (passive_* backlash hinges must stay at 0).
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

    # Mid-roll forward angular momentum: rotation about body +y. MuJoCo free
    # joint qvel[3:6] is the angular velocity in the BODY frame, so [0, ω, 0]
    # is the forward-roll axis regardless of spawn yaw (verified in the smoke
    # test — a yawed spawn still rolls straight ahead in its own frame).
    if len(mid_env_ids) > 0 and midroll_omega_range[1] > 0.0:
        omega = torch.rand(len(mid_env_ids), device=env.device) * (midroll_omega_range[1] - midroll_omega_range[0]) + midroll_omega_range[0]
        env.sim.data.qvel[mid_env_ids, 4] = _ROULADE_FWD_SIGN * omega

    # Run-up hook: forward base velocity for STANDING spawns, body x → world xy
    # through the spawn yaw. (0, 0) = standstill start, disabled.
    stand_env_ids = env_ids[~is_mid]
    if len(stand_env_ids) > 0 and forward_vel_range[1] > 0.0:
        vx = torch.rand(len(stand_env_ids), device=env.device) * (forward_vel_range[1] - forward_vel_range[0]) + forward_vel_range[0]
        yaw_s = yaw[~is_mid]
        env.sim.data.qvel[stand_env_ids, 0] = vx * torch.cos(yaw_s)
        env.sim.data.qvel[stand_env_ids, 1] = vx * torch.sin(yaw_s)

    # Progress accounting: standing starts at 0, mid-roll at the spawn pitch.
    spawn_angle = torch.where(is_mid, mid_pitch, torch.zeros_like(mid_pitch))
    accum[env_ids] = spawn_angle
    max_accum[env_ids] = spawn_angle
    paid[env_ids] = spawn_angle
    # Head latch: mid-roll spawns are considered already past the head phase
    # (the reverse curriculum teaches roll COMPLETION; requiring a latch they
    # never had the chance to earn would keep their landing gate shut forever).
    # Standing spawns must earn it by actually rolling over the head.
    env._roulade_head_latch[env_ids] = is_mid


def roulade_progress(env: ManagerBasedRlEnv, target_angle: float = 2 * math.pi, max_paid_rate: float = 3.0, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Pay increments of the progress frontier, up to one full roll.

    reward = Δ(min(max_accum, target)) / (step_dt · target), CAPPED at
    max_paid_rate rad/s of paid rotation. Nothing to farm by camping
    face-down (0/step), rocking below the frontier (0/step), or spinning past
    2π (clamped). The accumulator is support-gated, so airborne rotation pays
    nothing either.

    max_paid_rate (run-1 fix): rotation faster than the cap FORFEITS the
    excess — the paid pointer still jumps to the frontier, it just pays the
    capped amount. A violent whip therefore collects LESS total progress
    reward than a controlled ≤cap roll, instead of the same total sooner.
    """
    asset: Entity = env.scene[asset_cfg.name]
    _update_roulade_accum(env, asset)
    _, max_accum, paid = _roulade_state(env)
    new_paid = torch.clamp(max_accum, max=target_angle)
    delta = torch.clamp(new_paid - torch.clamp(paid, max=target_angle), min=0.0)
    delta = torch.clamp(delta, max=max_paid_rate * env.step_dt)
    env._roulade_paid = torch.maximum(paid, new_paid)
    return delta / (env.step_dt * target_angle)


def roulade_head_pivot(env: ManagerBasedRlEnv, sensor_name: str = "head_ground_contact", angle_lo: float = math.radians(30.0), angle_hi: float = math.radians(240.0), rate_norm: float = 2.0, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Reward head-ground contact while rotating forward mid-roll.

    contact × window(accum ∈ [angle_lo, angle_hi]) × clamp(ω_fwd/rate_norm, 0, 1)
    × (0.3 + 0.7·top_down).
    The rate factor is the anti-camping guard: a face-planted robot resting its
    head on the floor has ω_fwd ≈ 0 and earns nothing — the term only pays for
    pivoting OVER the head. The top_down factor (run-5) aligns this dense
    shaping with the latch: any head contact mid-roll pays 30%, contact on the
    FLAT TOP (chin tucked) pays full — the gradient that teaches the tuck.
    """
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
    """standing_composite_score × completion gate.

    The big annuity: once the roll is (nearly) complete, every step spent
    standing at HOME pose pays — finishing on the feet and staying there
    dominates every partial outcome. Zero before gate_lo of rotation, so the
    standing spawn cannot farm it by doing nothing.
    """
    asset: Entity = env.scene[asset_cfg.name]
    _update_roulade_accum(env, asset)
    score = standing_composite_score(env, target_height=target_height, height_std=height_std, upright_std=upright_std, pose_std=pose_std, joint_indices=joint_indices, target_overrides=target_overrides, asset_cfg=asset_cfg)
    return score * _roulade_completion_gate(env, gate_lo, gate_hi, require_head=True)


def roulade_upright_after_roll(env: ManagerBasedRlEnv, gate_lo: float = math.radians(260.0), gate_hi: float = math.radians(330.0), asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Linear cos(tilt) × completion gate — bootstrap pull toward vertical.

    Gradient from ANY orientation (the composite is near-zero far from the
    goal), but only after the roll: before gate_lo it is exactly zero, so it
    cannot oppose the flip the way the old always-on upright term did.
    """
    asset: Entity = env.scene[asset_cfg.name]
    _update_roulade_accum(env, asset)
    quat = asset.data.root_link_quat_w
    upright = 1.0 - 2.0 * (quat[:, 1].pow(2) + quat[:, 2].pow(2))
    return torch.clamp(upright, min=0.0) * _roulade_completion_gate(env, gate_lo, gate_hi, require_head=True)


def roulade_height_after_roll(env: ManagerBasedRlEnv, target_height: float, std: float = 0.04, gate_lo: float = math.radians(260.0), gate_hi: float = math.radians(330.0), asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Broad height Gaussian × completion gate — pull up to standing height."""
    asset: Entity = env.scene[asset_cfg.name]
    _update_roulade_accum(env, asset)
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    g = torch.exp(-(((z - target_height) / std) ** 2))
    return g * _roulade_completion_gate(env, gate_lo, gate_hi, require_head=True)


def roulade_landing_sharp(env: ManagerBasedRlEnv, target_height: float, height_std: float = 0.015, upright_std: float = 0.3, gate_lo: float = math.radians(260.0), gate_hi: float = math.radians(330.0), asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Tight-std upright × height Gaussians × completion gate — the last mile.

    Run-4 fix for the 27°-lean / 1-cm-crouch end basin: the broad landing
    composite (upright_std 0.40) scores ~0.5 at that pose, so the policy
    parks there. This is standup's two-layer lesson — the broad layers reach,
    the sharp layers finish. At 27° tilt this term scores ~0.1 (real
    gradient); at vertical it pays ~1.
    """
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
    """SELF-NEGATING height L1 below target, active only after roll completion.

    Returns −max(0, target − z) × completion_gate — use a POSITIVE weight
    (penalty sign convention). The run-3 fix for post-roll crumple-camping:
    the gated landing rewards made standing better than lying in a heap, but
    the heap itself was FREE — with only positive gated terms, "stay crumpled"
    collects ≈0/step, a comfortable basin (the standup static-sit lesson:
    the basin must be net NEGATIVE to force the rise). The gate keeps the
    roll itself untaxed, and requires the head latch so a no-roll episode
    can't be punished into weird avoidance behaviors.
    """
    asset: Entity = env.scene[asset_cfg.name]
    _update_roulade_accum(env, asset)
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    shortfall = torch.clamp(target_height - z, min=0.0)
    return -shortfall * _roulade_completion_gate(env, gate_lo, gate_hi, require_head=True)


def roulade_rise_velocity(env: ManagerBasedRlEnv, max_height: float = 0.125, gate_lo: float = math.radians(180.0), gate_hi: float = math.radians(260.0), asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """com_upward_velocity × late-roll gate — bootstrap the exit rise.

    The second half of a roulade (supine → sitting-up → standing) is the
    face-up recovery problem, and the standup env proved end-state rewards
    alone have zero gradient at zero motion there: pay for rising vz directly.
    Gated to open from ~180° (on the back) so pre-roll bobbing earns nothing,
    and gated off above max_height so it can't be farmed by hopping.
    """
    asset: Entity = env.scene[asset_cfg.name]
    _update_roulade_accum(env, asset)
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    vz = torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0)
    reward = torch.clamp(vz, min=0.0) * (z < max_height).float()
    return reward * _roulade_completion_gate(env, gate_lo, gate_hi, require_head=True)


def roulade_overspeed_penalty(env: ManagerBasedRlEnv, omega_max: float = 4.0, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """max(0, |ω_y| − omega_max)² — quadratic tax on whip-speed rotation.

    Positive quantity; use a negative weight. Complements the paid-rate cap
    in roulade_progress: the cap removes the INCENTIVE to rotate faster than
    ~3 rad/s, this adds an explicit COST above omega_max, so "violent" is
    strictly worse than "controlled" rather than merely not-better. A
    controlled full roll (~2–3 rad/s average) never touches it.
    """
    asset: Entity = env.scene[asset_cfg.name]
    omega_y = torch.nan_to_num(asset.data.root_link_ang_vel_b[:, 1], nan=0.0)
    excess = torch.clamp(omega_y.abs() - omega_max, min=0.0)
    return excess.pow(2)


def roulade_flatness_penalty(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """(lateral-axis world-z)² — dense gradient toward a sagittal roll.

    Positive quantity; use a negative weight. Zero when standing, zero
    through an arbitrarily deep CLEAN forward roll (pure pitch keeps the
    lateral axis horizontal), up to 1 when tipped fully onto a shoulder.
    The accumulator's flatness gate makes side rolls unprofitable; this term
    adds the per-step gradient that steers back toward the plane.
    """
    asset: Entity = env.scene[asset_cfg.name]
    return torch.nan_to_num(_lateral_axis_z(asset.data.root_link_quat_w), nan=0.0).pow(2)


def roulade_sagittal_penalty(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Rotation out of the sagittal plane: body-frame ω_x² + ω_z² (positive;
    use a negative weight). ω_y is the roll axis and stays free."""
    asset: Entity = env.scene[asset_cfg.name]
    omega_b = asset.data.root_link_ang_vel_b
    return torch.nan_to_num(omega_b[:, 0].pow(2) + omega_b[:, 2].pow(2), nan=0.0)


def roulade_lateral_velocity_penalty(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Body-frame lateral (y) linear velocity² — keeps the roll straight."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.nan_to_num(asset.data.root_link_lin_vel_b[:, 1].pow(2), nan=0.0)
