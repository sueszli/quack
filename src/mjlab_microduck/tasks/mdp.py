"""MDP functions for microduck tasks"""

import math

import numpy as np
import torch
from typing import TYPE_CHECKING, Optional
import mujoco

from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.reward_manager import RewardManager as _RewardManager
from mjlab.entity import Entity
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg
from mjlab.utils.lab_api.math import matrix_from_quat
from mjlab.envs.mdp.actions import JointPositionActionCfg as _JointPositionActionCfg

# Patch RewardManager.compute to sanitize NaN rewards before they enter the
# PPO buffer.  mjlab computes rewards BEFORE resetting environments, so any
# reward term that operates on a NaN physics state (e.g. after MuJoCo contact
# instability) returns NaN.  That NaN propagates: NaN reward → NaN advantage
# → NaN loss → NaN gradient → NaN log_std → crash in torch.normal on the
# next mini-batch.  Replacing NaN rewards with 0.0 breaks the chain without
# altering the training signal for healthy environments.
_orig_reward_compute = _RewardManager.compute

def _nan_safe_reward_compute(self, dt: float) -> torch.Tensor:
    return torch.nan_to_num(_orig_reward_compute(self, dt), nan=0.0)

_RewardManager.compute = _nan_safe_reward_compute

if TYPE_CHECKING:
    from mjlab.viewer.debug_visualizer import DebugVisualizer


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

# Name patterns to look up neck/head joint entity-local IDs at runtime.
# Must use names rather than ctrl indices because joint_pos_target is indexed
# by entity-local joint order, which differs from ctrl order.
_NECK_JOINT_PATTERNS = [r".*neck_pitch.*", r".*head_pitch.*", r".*head_yaw.*", r".*head_roll.*"]
# Time constant (seconds) for smooth offset interpolation toward target
_NECK_OFFSET_SMOOTHING_TAU = 0.5

class NeckOffsetJointPositionAction(_JointPositionActionCfg.class_type):
    """JointPositionAction that adds a random offset to neck/head joint targets.

    After the policy output is applied as joint position targets, adds
    env._neck_offset to the joint_pos_target buffer for neck joints.
    This trains robustness to external head movement and enables independent
    head control at deployment (add any offset on top of policy output).

    The offset smoothly follows env._neck_offset_target, which is updated by
    randomize_neck_offset_target() interval events.
    """

    def apply_actions(self) -> None:
        # Apply standard joint position control from policy output.
        # In new mjlab this writes to entity.data.joint_pos_target (not ctrl directly).
        # entity.write_data_to_sim() then copies joint_pos_target → ctrl after all
        # apply_actions() calls, so the offset must be added here to joint_pos_target.
        super().apply_actions()

        env = self._env

        # Initialize offset tensors on first call
        if not hasattr(env, "_neck_offset"):
            env._neck_offset = torch.zeros(env.num_envs, 4, device=env.device)
            env._neck_offset_target = torch.zeros(env.num_envs, 4, device=env.device)

        # Cache entity-local neck joint IDs (looked up by name, not ctrl index)
        if not hasattr(self, "_neck_joint_ids"):
            ids, _ = self._entity.find_joints_by_actuator_names(_NECK_JOINT_PATTERNS)
            self._neck_joint_ids = torch.tensor(ids, device=env.device, dtype=torch.long)

        # Exponential smoothing: offset tracks target with time constant tau
        alpha = min(1.0, env.step_dt / _NECK_OFFSET_SMOOTHING_TAU)
        env._neck_offset.lerp_(env._neck_offset_target, alpha)

        # Add offset to joint_pos_target — the buffer super().apply_actions() wrote to
        self._entity.data.joint_pos_target[:, self._neck_joint_ids] += env._neck_offset


class NeckOffsetJointPositionActionCfg(_JointPositionActionCfg):
    def build(self, env: ManagerBasedRlEnv) -> "NeckOffsetJointPositionAction":
        return NeckOffsetJointPositionAction(self, env)


def reset_neck_offset(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
):
    """Reset neck joint offsets to zero at episode start."""
    if not hasattr(env, "_neck_offset"):
        env._neck_offset = torch.zeros(env.num_envs, 4, device=env.device)
        env._neck_offset_target = torch.zeros(env.num_envs, 4, device=env.device)

    if len(env_ids) > 0:
        env._neck_offset[env_ids] = 0.0
        env._neck_offset_target[env_ids] = 0.0


def randomize_neck_offset_target(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    max_offset: float = 0.3,
):
    """Sample new random neck offset targets (called at intervals).

    Draws uniform random targets in [-max_offset, max_offset] for each of the
    4 neck/head joints. The offset smoothly interpolates toward the new target.
    """
    if not hasattr(env, "_neck_offset_target"):
        env._neck_offset = torch.zeros(env.num_envs, 4, device=env.device)
        env._neck_offset_target = torch.zeros(env.num_envs, 4, device=env.device)

    if len(env_ids) > 0:
        env._neck_offset_target[env_ids] = (
            torch.rand(len(env_ids), 4, device=env.device) * 2 - 1
        ) * max_offset


def reset_with_forward_velocity(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    velocity_range: tuple[float, float] = (0.3, 0.8),
    fraction_stages: list[dict] | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Warm-start a fraction of reset environments with a random forward velocity.

    The robot spawns already moving in its body-forward direction, so it first
    discovers what coasting at speed feels like. The fraction decreases over
    training, forcing it to progressively earn that speed from rest.

    Args:
        velocity_range: (min, max) forward speed in m/s.
        fraction_stages: list of {"step": int, "fraction": float} dicts, sorted by step.
            The fraction active at the current training step is used.
            Example: [{"step":0,"fraction":0.8}, {"step":2000*24,"fraction":0.0}]
        asset_cfg: robot entity config.
    """
    if fraction_stages is None:
        fraction_stages = [{"step": 0, "fraction": 0.8}]

    # Determine current fraction from training step
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

    # Build horizontal forward direction from yaw only — ignoring pitch/roll.
    # IMPORTANT: read quaternion from qpos, NOT from root_link_quat_w.
    # root_link_quat_w reads xquat which requires sim.forward() to be current.
    # After reset_base writes a new yaw to qpos, xquat is still stale (old episode).
    # qpos is updated immediately by write_root_pose, so it's always fresh.
    asset: Entity = env.scene[asset_cfg.name]
    qpos_q_adr = asset.data.indexing.free_joint_q_adr[3:7]  # quat indices in qpos
    q = asset.data.data.qpos[warmstart_ids][:, qpos_q_adr]  # (n, 4) [w, x, y, z]
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    forward_world = torch.stack([torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)], dim=-1)

    velocities = torch.zeros(n_warmstart, 6, device=env.device)
    velocities[:, :3] = vx.unsqueeze(-1) * forward_world

    asset.write_root_link_velocity_to_sim(velocities, env_ids=warmstart_ids)

    # Spin wheels to match forward velocity — prevents instantaneous no-slip braking.
    # Wheel radius = 0.0175 m (measured).
    # All 4 wheels spin at +ω for forward motion (verified by test_wheel_direction.py).
    _WHEEL_RADIUS = 0.0175
    all_wheel_ids, _ = asset.find_joints(r"^passive_.*")

    if all_wheel_ids:
        joint_pos = asset.data.joint_pos[warmstart_ids].clone()
        joint_vel = asset.data.joint_vel[warmstart_ids].clone()
        omega = vx / _WHEEL_RADIUS  # (n,) rad/s, positive = forward
        joint_vel[:, all_wheel_ids] = omega.unsqueeze(-1).expand(-1, len(all_wheel_ids))
        asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=warmstart_ids)


def reset_action_history(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
    """
    Reset cached action history for environments that are being reset.
    This is critical for action rate and acceleration penalty terms.

    This function should be called in the post_reset callback or at episode termination.

    Args:
        env: The environment
        env_ids: Indices of environments being reset
        asset_cfg: Asset configuration
    """
    if len(env_ids) == 0:
        return

    asset: Entity = env.scene[asset_cfg.name]

    # Reset leg action rate cache
    if hasattr(env, '_prev_leg_actions'):
        # Set to current action (or zero if no action yet)
        if hasattr(env, 'action_manager') and env.action_manager.action is not None:
            leg_joint_indices = list(range(0, 5)) + list(range(9, 14))
            env._prev_leg_actions[env_ids] = env.action_manager.action[env_ids][:, leg_joint_indices]
        else:
            env._prev_leg_actions[env_ids] = 0.0

    # Reset neck action rate cache
    if hasattr(env, '_prev_neck_actions'):
        if hasattr(env, 'action_manager') and env.action_manager.action is not None:
            neck_joint_indices = list(range(5, 9))
            env._prev_neck_actions[env_ids] = env.action_manager.action[env_ids][:, neck_joint_indices]
        else:
            env._prev_neck_actions[env_ids] = 0.0

    # Reset leg action acceleration cache
    if hasattr(env, '_prev_leg_actions_for_acc'):
        if hasattr(env, 'action_manager') and env.action_manager.action is not None:
            leg_joint_indices = list(range(0, 5)) + list(range(9, 14))
            current_action = env.action_manager.action[env_ids][:, leg_joint_indices]
            env._prev_leg_actions_for_acc[env_ids] = current_action
            env._prev_prev_leg_actions_for_acc[env_ids] = current_action
        else:
            env._prev_leg_actions_for_acc[env_ids] = 0.0
            env._prev_prev_leg_actions_for_acc[env_ids] = 0.0

    # Reset neck action acceleration cache
    if hasattr(env, '_prev_neck_actions_for_acc'):
        if hasattr(env, 'action_manager') and env.action_manager.action is not None:
            neck_joint_indices = list(range(5, 9))
            current_action = env.action_manager.action[env_ids][:, neck_joint_indices]
            env._prev_neck_actions_for_acc[env_ids] = current_action
            env._prev_prev_neck_actions_for_acc[env_ids] = current_action
        else:
            env._prev_neck_actions_for_acc[env_ids] = 0.0
            env._prev_prev_neck_actions_for_acc[env_ids] = 0.0

    # Reset joint velocity cache for joint accelerations
    if hasattr(asset.data, '_prev_joint_vel'):
        # Get current joint velocities for reset environments
        joint_vel = asset.data.joint_vel[env_ids, :][:, asset_cfg.joint_ids]
        asset.data._prev_joint_vel[env_ids] = joint_vel

    # Reset contact frequency tracking
    if hasattr(env, '_contact_change_count'):
        env._contact_change_count[env_ids] = 0.0
    if hasattr(env, '_contact_change_timer'):
        env._contact_change_timer[env_ids] = 0.0
    if hasattr(env, '_prev_contacts_for_freq'):
        if "feet_ground_contact" in env.scene.sensors:
            contacts = env.scene.sensors["feet_ground_contact"].data.found[env_ids, :2]
            env._prev_contacts_for_freq[env_ids] = contacts

    # Reset foot force smoothness tracking
    if hasattr(env, '_prev_foot_forces'):
        if "feet_ground_contact" in env.scene.sensors:
            forces = env.scene.sensors["feet_ground_contact"].data.found[env_ids, :2].squeeze(-1)
            env._prev_foot_forces[env_ids] = forces


def joint_accelerations_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """
    Penalize joint accelerations using L2 squared norm.
    Joint accelerations are computed using finite differences of joint velocities.

    Args:
        env: The environment
        asset_cfg: Asset configuration

    Returns:
        Penalty tensor of shape (num_envs,) - sum of squared joint accelerations
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Get current joint velocities
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]

    # Get previous joint velocities (stored in asset data)
    # Note: This assumes the environment stores previous joint velocities
    if not hasattr(asset.data, '_prev_joint_vel'):
        # Initialize on first call
        asset.data._prev_joint_vel = joint_vel.clone()
        return torch.zeros(env.num_envs, device=env.device)

    # Compute joint accelerations using finite differences
    dt = env.step_dt
    joint_acc = (joint_vel - asset.data._prev_joint_vel) / dt

    # Store current velocities for next step
    asset.data._prev_joint_vel = joint_vel.clone()

    # Return L2 squared norm
    return torch.sum(torch.square(joint_acc), dim=1)


def leg_action_rate_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """
    Penalize the rate of change of leg actions (action_t - action_{t-1}).
    Leg joints are indices 0-4 and 9-13 (10 joints total).

    Args:
        env: The environment
        asset_cfg: Asset configuration

    Returns:
        Penalty tensor of shape (num_envs,)
    """
    # Get leg joint indices
    leg_joint_indices = list(range(0, 5)) + list(range(9, 14))

    # Get current and previous actions for leg joints only
    # Actions are stored in env (assuming the action is available)
    if not hasattr(env, 'action_manager'):
        return torch.zeros(env.num_envs, device=env.device)

    # Get the joint position action
    actions = env.action_manager.action
    if actions.shape[1] < 14:
        return torch.zeros(env.num_envs, device=env.device)

    leg_actions = actions[:, leg_joint_indices]

    if not hasattr(env, '_prev_leg_actions'):
        env._prev_leg_actions = leg_actions.clone()
        return torch.zeros(env.num_envs, device=env.device)

    action_rate = leg_actions - env._prev_leg_actions
    env._prev_leg_actions = leg_actions.clone()

    return torch.sum(torch.square(action_rate), dim=1)


def neck_action_rate_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """
    Penalize the rate of change of neck actions (action_t - action_{t-1}).
    Neck joints are indices 5-8 (4 joints total).

    Args:
        env: The environment
        asset_cfg: Asset configuration

    Returns:
        Penalty tensor of shape (num_envs,)
    """
    # Get neck joint indices
    neck_joint_indices = list(range(5, 9))

    # Get current and previous actions for neck joints only
    if not hasattr(env, 'action_manager'):
        return torch.zeros(env.num_envs, device=env.device)

    actions = env.action_manager.action
    if actions.shape[1] < 14:
        return torch.zeros(env.num_envs, device=env.device)

    neck_actions = actions[:, neck_joint_indices]

    if not hasattr(env, '_prev_neck_actions'):
        env._prev_neck_actions = neck_actions.clone()
        return torch.zeros(env.num_envs, device=env.device)

    action_rate = neck_actions - env._prev_neck_actions
    env._prev_neck_actions = neck_actions.clone()

    return torch.sum(torch.square(action_rate), dim=1)


def leg_action_acceleration_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """
    Penalize leg action accelerations (action_t - 2*action_{t-1} + action_{t-2}).
    Leg joints are indices 0-4 and 9-13 (10 joints total).

    Args:
        env: The environment
        asset_cfg: Asset configuration

    Returns:
        Penalty tensor of shape (num_envs,)
    """
    # Get leg joint indices
    leg_joint_indices = list(range(0, 5)) + list(range(9, 14))

    if not hasattr(env, 'action_manager'):
        return torch.zeros(env.num_envs, device=env.device)

    actions = env.action_manager.action
    if actions.shape[1] < 14:
        return torch.zeros(env.num_envs, device=env.device)

    leg_actions = actions[:, leg_joint_indices]

    if not hasattr(env, '_prev_leg_actions_for_acc'):
        env._prev_leg_actions_for_acc = leg_actions.clone()
        env._prev_prev_leg_actions_for_acc = leg_actions.clone()
        return torch.zeros(env.num_envs, device=env.device)

    action_acc = leg_actions - 2 * env._prev_leg_actions_for_acc + env._prev_prev_leg_actions_for_acc

    env._prev_prev_leg_actions_for_acc = env._prev_leg_actions_for_acc.clone()
    env._prev_leg_actions_for_acc = leg_actions.clone()

    return torch.sum(torch.square(action_acc), dim=1)


def neck_action_acceleration_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """
    Penalize neck action accelerations (action_t - 2*action_{t-1} + action_{t-2}).
    Neck joints are indices 5-8 (4 joints total).

    Args:
        env: The environment
        asset_cfg: Asset configuration

    Returns:
        Penalty tensor of shape (num_envs,)
    """
    # Get neck joint indices
    neck_joint_indices = list(range(5, 9))

    if not hasattr(env, 'action_manager'):
        return torch.zeros(env.num_envs, device=env.device)

    actions = env.action_manager.action
    if actions.shape[1] < 14:
        return torch.zeros(env.num_envs, device=env.device)

    neck_actions = actions[:, neck_joint_indices]

    if not hasattr(env, '_prev_neck_actions_for_acc'):
        env._prev_neck_actions_for_acc = neck_actions.clone()
        env._prev_prev_neck_actions_for_acc = neck_actions.clone()
        return torch.zeros(env.num_envs, device=env.device)

    action_acc = neck_actions - 2 * env._prev_neck_actions_for_acc + env._prev_prev_neck_actions_for_acc

    env._prev_prev_neck_actions_for_acc = env._prev_neck_actions_for_acc.clone()
    env._prev_neck_actions_for_acc = neck_actions.clone()

    return torch.sum(torch.square(action_acc), dim=1)


def body_upright_linear(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Linear reward for body uprightness — provides gradient at every tilt angle.

    Returns +1 when fully upright, 0 when horizontal (prone/supine), -1 when inverted.
    Unlike flat_orientation (Gaussian), this has non-zero gradient everywhere, so the
    robot always has a signal to rotate toward upright even when starting from prone.

    Computed as the z-component of the body's local Z-axis expressed in world frame,
    which equals R[2,2] = 1 - 2*(qx² + qy²) for quaternion [w, x, y, z].
    """
    asset: Entity = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w  # (N, 4): [w, x, y, z]
    qx = quat[:, 1]
    qy = quat[:, 2]
    return 1.0 - 2.0 * (qx * qx + qy * qy)


def com_upward_velocity(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    max_height: float = 0.08,
) -> torch.Tensor:
    """Reward upward CoM velocity to incentivize dynamic standup motion.

    Gated by height: only active while the CoM is below `max_height` (the
    standing target). Once standing, the reward is zero so the robot has no
    incentive to keep squatting to farm upward-velocity reward.
    """
    asset: Entity = env.scene[asset_cfg.name]
    # nan_to_num: MuJoCo can produce NaN on contact instability; treat as z=0
    com_z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    vz = torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0)
    below_target = (com_z < max_height).float()
    return torch.clamp(vz, min=0.0) * below_target


def robot_state_is_nan(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Terminate environments where MuJoCo produced NaN joint positions.

    MuJoCo's contact solver can overflow to NaN under extreme penetration or
    impulse (e.g. robot landing at high velocity). A NaN simulation state
    propagates into observations, corrupting the policy network weights.

    Terminating immediately resets the environment before the cascade spreads:
    - The observation returned to the runner is from the valid reset state.
    - NaN rewards are avoided on subsequent steps.

    Note: the reward at THIS terminal step may still be NaN from the simulation;
    mjlab computes rewards before resetting (see manager_based_rl_env.py step()).
    Our custom reward functions guard against NaN internally with nan_to_num,
    but standard mjlab rewards can still be NaN here. One NaN reward is
    tolerable because done=True prevents it propagating backward through GAE.
    """
    asset: Entity = env.scene[asset_cfg.name]
    return torch.any(torch.isnan(asset.data.joint_pos), dim=1)


def is_alive(env: ManagerBasedRlEnv) -> torch.Tensor:
    """
    Reward for staying alive (not terminated)

    Args:
        env: The environment

    Returns:
        Reward tensor of shape (num_envs,) - ones for all envs
    """
    return torch.ones(env.num_envs, device=env.device)


def com_height_target(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    target_height_min: float = 0.1,
    target_height_max: float = 0.15,
) -> torch.Tensor:
    """
    Reward for keeping the center of mass within a target height range.
    Returns positive reward when in range, negative penalty when outside.

    Args:
        env: The environment
        asset_cfg: Asset configuration
        target_height_min: Minimum target height for CoM (meters)
        target_height_max: Maximum target height for CoM (meters)

    Returns:
        Reward tensor of shape (num_envs,)
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Height above terrain spawn origin (world z minus terrain z).
    # env_origins[:, 2] is 0 for flat ground, so this is safe unconditionally.
    # nan_to_num: MuJoCo can produce NaN on contact instability; treat as z=0
    # so the penalty is finite (small, since 0 is near the target range).
    com_height = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )

    # Reward when in range, penalty when outside
    # Use smooth penalty that increases quadratically with distance from range
    below_min = com_height < target_height_min
    above_max = com_height > target_height_max
    in_range = ~(below_min | above_max)

    # Compute penalties for being outside range
    penalty_below = torch.square(com_height - target_height_min) * below_min.float()
    penalty_above = torch.square(com_height - target_height_max) * above_max.float()

    # Reward: +1 when in range, -squared_distance when outside
    reward = in_range.float() - (penalty_below + penalty_above)

    return reward


def neck_joint_vel_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """
    Penalize neck joint velocities to keep head stable.
    Neck joints are indices 5-8 (4 joints total).

    Args:
        env: The environment
        asset_cfg: Asset configuration

    Returns:
        Penalty tensor of shape (num_envs,)
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Get neck joint indices (neck_pitch, head_pitch, head_yaw, head_roll)
    neck_joint_indices = list(range(5, 9))

    # Get joint velocities for neck joints
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    neck_joint_vel = joint_vel[:, neck_joint_indices]

    # Return L2 squared norm of neck joint velocities
    return torch.sum(torch.square(neck_joint_vel), dim=1)


def leg_joint_vel_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """
    Penalize leg joint velocities to encourage smoother, less dynamic motion.
    Leg joints are indices 0-4 and 9-13 (10 joints total).

    Args:
        env: The environment
        asset_cfg: Asset configuration

    Returns:
        Penalty tensor of shape (num_envs,)
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Get leg joint indices (left hip-ankle: 0-4, right hip-ankle: 9-13)
    leg_joint_indices = list(range(0, 5)) + list(range(9, 14))

    # Get joint velocities for leg joints
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    leg_joint_vel = joint_vel[:, leg_joint_indices]

    # Return L2 squared norm of leg joint velocities
    return torch.sum(torch.square(leg_joint_vel), dim=1)

_NECK_JOINT_CFG = SceneEntityCfg("robot", joint_names=(r".*(neck|head).*",))
_HIP_PITCH_KNEE_CFG = SceneEntityCfg("robot", joint_names=(r".*(hip_pitch|knee).*",))
_ROLLER_FEET_SITE_CFG = SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))


def feet_flat_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _ROLLER_FEET_SITE_CFG,
) -> torch.Tensor:
    """Penalize foot sites not being parallel to the ground.

    The foot site frame has Z+ pointing up when flat. We project a unit gravity
    vector (pointing down) into each foot site's local frame. When flat, gravity
    maps to [0,0,-1] in site frame (xy=0, penalty=0). Any tilt rotates Z away
    from world-up, giving nonzero xy components.

    Max value ≈ 2.0 per foot (foot fully sideways), total ≈ 4.0.

    Bug note: must normalize gravity PER ENV with dim=-1. Using torch.norm()
    without dim computes a scalar over all envs × 3 dims, making the vector
    ~1/sqrt(num_envs) in magnitude → penalty ~num_envs times too small.
    """
    from mjlab.utils.lab_api.math import quat_apply_inverse
    import torch.nn.functional as F

    asset: Entity = env.scene[asset_cfg.name]
    gravity_w_n = F.normalize(asset.data.gravity_vec_w, dim=-1)  # (B, 3), unit vector per env

    foot_quats = asset.data.site_quat_w[:, asset_cfg.site_ids, :]  # (B, N_feet, 4)
    total = torch.zeros(env.num_envs, device=env.device)
    for i in range(foot_quats.shape[1]):
        proj = quat_apply_inverse(foot_quats[:, i, :], gravity_w_n)  # (B, 3)
        total += torch.sum(torch.square(proj[:, :2]), dim=1)  # xy² only
    return total


def hip_pitch_knee_vel_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _HIP_PITCH_KNEE_CFG,
) -> torch.Tensor:
    """Penalize hip_pitch and knee joint velocities (L2 squared).

    Walking requires rapid oscillation of these sagittal-plane joints.
    Skating uses hip_roll laterally and glides with minimal sagittal movement.
    This penalizes the oscillation without preventing static balance adjustments.
    """
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def neck_joint_pos_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _NECK_JOINT_CFG,
) -> torch.Tensor:
    """Penalize neck/head joint position deviation from default (L2 squared).

    Uses find_joints() every call to avoid stale cached indices when the same
    SceneEntityCfg singleton is reused across robots with different joint layouts
    (e.g. walk robot vs rollers robot where passive wheels shift neck indices).
    """
    asset: Entity = env.scene[asset_cfg.name]
    joint_ids, _ = asset.find_joints(r".*(neck|head).*")
    error = asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]
    return torch.sum(torch.square(error), dim=1)


def joint_torques_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """
    Penalize actuator forces (torques) to encourage energy-efficient motion.

    Args:
        env: The environment
        asset_cfg: Asset configuration

    Returns:
        Penalty tensor of shape (num_envs,) - sum of squared actuator forces
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Get actuator forces (scalar actuation in actuation space)
    actuator_forces = asset.data.actuator_force

    # Return L2 squared norm
    return torch.sum(torch.square(actuator_forces), dim=1)


def wheel_speed_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    wheel_radius: float = 0.0175,
    vel_scale: float = 0.5,
) -> torch.Tensor:
    """Reward forward wheel spin proportional to commanded speed.

    All 4 wheels spin positive for forward motion (verified visually).
    tanh saturation at vel_scale m/s equivalent prevents runaway.
    Provides gradient at low body speeds when velocity tracking reward is near-zero.
    Only fires for forward commands.
    """
    cmd_x = env.command_manager.get_command(command_name)[:, 0]  # (B,)

    asset: Entity = env.scene["robot"]
    lf_ids, _ = asset.find_joints("passive_LFwheel")
    lr_ids, _ = asset.find_joints("passive_LRwheel")
    rf_ids, _ = asset.find_joints("passive_RFwheel")
    rr_ids, _ = asset.find_joints("passive_RRwheel")

    vel = asset.data.joint_vel
    # All 4 wheels spin positive for forward motion (verified by test_wheel_direction.py)
    forward_omega = (vel[:, lf_ids[0]] + vel[:, lr_ids[0]] + vel[:, rf_ids[0]] + vel[:, rr_ids[0]]) / 4.0

    omega_scale = vel_scale / wheel_radius
    return torch.clamp(cmd_x, min=0.0) * torch.tanh(torch.clamp(forward_omega, min=0.0) / omega_scale)


def coasting_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    vel_std: float = 0.3,
    stillness_std: float = 5.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=(r".*(hip|knee|ankle).*",)),
) -> torch.Tensor:
    """Reward coasting: low leg-joint velocity while at target speed.

    Returns exp(-vel_error / vel_std²) × exp(-sum(joint_vel²) / stillness_std²).
    Both factors must be high simultaneously — robot is rewarded for being at
    target speed AND keeping its legs still (gliding), not for either alone.

    Typical values when coasting well: ~0.7–1.0.  When actively stomping at
    speed the joint_vel term suppresses the reward toward 0.
    """
    cmd = env.command_manager.get_command(command_name)
    vel_b = env.scene["robot"].data.root_link_lin_vel_b[:, :2]
    vel_error = torch.sum(torch.square(cmd[:, :2] - vel_b), dim=1)
    at_speed = torch.exp(-vel_error / vel_std ** 2)

    asset: Entity = env.scene[asset_cfg.name]
    joint_vel_sq = torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
    stillness = torch.exp(-joint_vel_sq / stillness_std ** 2)

    return at_speed * stillness


def braking_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    vel_std: float = 0.3,
) -> torch.Tensor:
    """Reward coming to a stop when cmd_x < 0 (brake commanded).

    Returns clamp(-cmd_x, 0) * exp(-fwd_vel² / vel_std²).
    - Silent when cmd_x ≥ 0 (coast or push).
    - At cmd_x = -1 and vel = 0: reward = 1.0 (full stop achieved).
    - At cmd_x = -1 and vel = vel_std: reward ≈ 0.37 (strong gradient).
    vel_std=0.3 m/s gives meaningful gradient down to walking-pace speeds.
    """
    cmd = env.command_manager.get_command(command_name)
    cmd_x = cmd[:, 0]
    braking_strength = torch.clamp(-cmd_x, min=0.0)
    fwd_vel = env.scene["robot"].data.root_link_lin_vel_b[:, 0]
    stopped = torch.exp(-(fwd_vel.clamp(min=0.0) ** 2) / (vel_std ** 2))
    return braking_strength * stopped


def contact_frequency_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str = "feet_ground_contact",
    max_contact_changes_per_sec: float = 4.0,
    command_threshold: float = 0.01,
) -> torch.Tensor:
    """
    Penalize high frequency of contact changes to encourage slower stepping.
    Tracks the number of contact state changes per second and penalizes when above threshold.

    Args:
        env: The environment
        sensor_name: Name of the contact sensor
        max_contact_changes_per_sec: Maximum allowed contact changes per second
        command_threshold: Minimum command magnitude to apply penalty

    Returns:
        Penalty tensor of shape (num_envs,) - negative when exceeding threshold
    """
    if sensor_name not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)

    # Check if command is above threshold
    if "twist" in env.command_manager._terms:
        cmd = env.command_manager.get_command("twist")
        cmd_vel = cmd[:, :3]
        cmd_norm = torch.linalg.norm(cmd_vel, dim=1)
        active_mask = cmd_norm > command_threshold
    else:
        active_mask = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)

    sensor = env.scene.sensors[sensor_name]
    contacts = sensor.data.found[:, :2]  # (num_envs, 2)

    # Initialize tracking if needed
    if not hasattr(env, '_contact_change_count'):
        env._contact_change_count = torch.zeros(env.num_envs, device=env.device)
        env._contact_change_timer = torch.zeros(env.num_envs, device=env.device)
        env._prev_contacts_for_freq = contacts.clone()
        return torch.zeros(env.num_envs, device=env.device)

    # Detect any contact changes (either foot)
    contact_changed = torch.any(contacts != env._prev_contacts_for_freq, dim=1)

    # Increment change counter
    env._contact_change_count += contact_changed.float()

    # Update timer
    env._contact_change_timer += env.step_dt

    # Calculate current frequency (changes per second)
    # Avoid division by zero
    freq = env._contact_change_count / torch.clamp(env._contact_change_timer, min=0.01)

    # Reset counter and timer every 1 second
    reset_mask = env._contact_change_timer >= 1.0
    env._contact_change_count[reset_mask] = 0.0
    env._contact_change_timer[reset_mask] = 0.0

    # Penalize when frequency exceeds maximum
    # Use quadratic penalty for frequencies above threshold
    excess_freq = torch.clamp(freq - max_contact_changes_per_sec, min=0.0)
    penalty = -torch.square(excess_freq)

    # Update previous contacts
    env._prev_contacts_for_freq = contacts.clone()

    # Apply command threshold mask
    penalty = penalty * active_mask.float()

    return penalty


# ==============================================================================
# Ground Pick Rewards
# ==============================================================================

def mouth_ground_proximity(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=["mouth_tip"]),
    std: float = 0.03,
    target_height: float = 0.0,
    command_name: str = "twist",
) -> torch.Tensor:
    """Reward for mouth tip approaching the ground, weighted by the approach phase.

    The command for the ground pick task is [cos(2π*phase), sin(2π*phase), 0].
    The approach phase is the first half-cycle (sin > 0, phase ∈ [0, 0.5]),
    smoothly weighted by max(0, sin(2π*phase)).

    Args:
        std: Gaussian std on mouth_tip height (m). 0.03 m gives strong gradient.
        target_height: Target z-height for the mouth tip (m). 0 = ground level.
    """
    asset = env.scene[asset_cfg.name]
    mouth_z = asset.data.site_pos_w[:, asset_cfg.site_ids[0], 2]  # (num_envs,)
    proximity = torch.exp(-((mouth_z - target_height) / std) ** 2)

    # Approach weight: max(0, sin(2π*phase)) — peaks at 1 at phase=0.25, zero at 0 and 0.5
    cmd = env.command_manager.get_command(command_name)
    approach_weight = torch.clamp(cmd[:, 1], min=0.0)

    return approach_weight * proximity


def mouth_perpendicular_to_ground(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=["mouth_tip"]),
    command_name: str = "twist",
) -> torch.Tensor:
    """Reward the mouth tip x-axis being vertical (pointing down) during the approach phase.

    A perfectly perpendicular contact gives alignment=1; horizontal gives 0; pointing up gives -1.
    Weighted by max(0, sin(2π*phase)) so it only applies during the descent.
    """
    asset = env.scene[asset_cfg.name]
    # site_quat_w: (num_envs, num_sites, 4) as [w, x, y, z]
    q = asset.data.site_quat_w[:, asset_cfg.site_ids[0], :]  # (num_envs, 4)
    w, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    # z-component of the site x-axis in world frame (first column of rotation matrix)
    x_axis_z = 2.0 * (qx * qz - w * qy)
    # dot with [0, 0, -1]: 1 = perfectly downward, -1 = upward
    alignment = -x_axis_z

    cmd = env.command_manager.get_command(command_name)
    approach_weight = torch.clamp(cmd[:, 1], min=0.0)

    return approach_weight * alignment


def ground_pick_return_pose(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.3,
    command_name: str = "twist",
    joint_indices: Optional[list] = None,
) -> torch.Tensor:
    """Reward for returning to the standing pose after ground pick, weighted by the return phase.

    The return phase is the second half-cycle (sin < 0, phase ∈ [0.5, 1.0]),
    smoothly weighted by max(0, -sin(2π*phase)).

    Args:
        std: Gaussian std per joint (rad).
        joint_indices: Subset of joints to evaluate. Use to apply different stds
            to leg joints vs neck/head joints (call this reward twice).
    """
    asset = env.scene[asset_cfg.name]
    joint_pos  = asset.data.joint_pos        # (num_envs, n_joints)
    default_pos = asset.data.default_joint_pos

    if joint_indices is not None:
        joint_pos   = joint_pos[:, joint_indices]
        default_pos = default_pos[:, joint_indices]

    pose_reward = torch.exp(-((joint_pos - default_pos) / std) ** 2).mean(dim=-1)

    # Return weight: max(0, -sin(2π*phase)) — peaks at 1 at phase=0.75, zero at 0.5 and 1
    cmd = env.command_manager.get_command(command_name)
    return_weight = torch.clamp(-cmd[:, 1], min=0.0)

    return return_weight * pose_reward


# ==============================================================================
# Domain Randomization Events
# ==============================================================================


def randomize_delayed_actuator_gains(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    kp_range: tuple[float, float],
    kd_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    operation: str = "scale",
):
    """Randomize PD gains for DelayedActuator (which wraps XmlPositionActuator).

    Args:
        env: The environment
        env_ids: Environment IDs to randomize (None = all envs)
        kp_range: (min, max) for kp randomization
        kd_range: (min, max) for kd randomization
        asset_cfg: Asset configuration
        operation: "scale" or "abs"
    """
    from mjlab.actuator.delayed_actuator import DelayedActuator
    from mjlab.actuator import XmlPositionActuator

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]

    # Store original gains on first call
    if not hasattr(env, '_original_actuator_gains'):
        env._original_actuator_gains = {}

    # Apply to actuators
    for actuator in asset.actuators:
        # Handle DelayedActuator wrapping XmlPositionActuator
        if isinstance(actuator, DelayedActuator):
            base_actuator = actuator._base_actuator
        else:
            base_actuator = actuator

        # Get control IDs
        ctrl_ids = base_actuator.ctrl_ids

        # Store original values on first call (use tuple of ctrl_ids as key)
        from mjlab_microduck.actuator.bam_actuator import BamM6Actuator
        ctrl_key = tuple(ctrl_ids.tolist())
        if not isinstance(base_actuator, BamM6Actuator):
            if ctrl_key not in env._original_actuator_gains:
                env._original_actuator_gains[ctrl_key] = {
                    'gainprm': env.sim.model.actuator_gainprm[0, ctrl_ids, 0].clone(),
                    'biasprm1': env.sim.model.actuator_biasprm[0, ctrl_ids, 1].clone(),
                    'biasprm2': env.sim.model.actuator_biasprm[0, ctrl_ids, 2].clone(),
                }

        # Reset to original values first (to prevent accumulation)
        if isinstance(base_actuator, BamM6Actuator):
            base_actuator.reset_gains(env_ids)
        else:
            original = env._original_actuator_gains[ctrl_key]
            env.sim.model.actuator_gainprm[env_ids[:, None], ctrl_ids, 0] = original['gainprm'].unsqueeze(0).expand(len(env_ids), -1)
            env.sim.model.actuator_biasprm[env_ids[:, None], ctrl_ids, 1] = original['biasprm1'].unsqueeze(0).expand(len(env_ids), -1)
            env.sim.model.actuator_biasprm[env_ids[:, None], ctrl_ids, 2] = original['biasprm2'].unsqueeze(0).expand(len(env_ids), -1)

        # Sample random gains for each env and each control
        kp_samples = torch.rand(len(env_ids), len(ctrl_ids), device=env.device) * (kp_range[1] - kp_range[0]) + kp_range[0]
        kd_samples = torch.rand(len(env_ids), len(ctrl_ids), device=env.device) * (kd_range[1] - kd_range[0]) + kd_range[0]

        # For XmlPositionActuator, modify MuJoCo model parameters directly
        if isinstance(base_actuator, XmlPositionActuator):
            if operation == "scale":
                # Scale the ORIGINAL (now-reset) values
                env.sim.model.actuator_gainprm[env_ids[:, None], ctrl_ids, 0] *= kp_samples
                env.sim.model.actuator_biasprm[env_ids[:, None], ctrl_ids, 1] *= kp_samples
                env.sim.model.actuator_biasprm[env_ids[:, None], ctrl_ids, 2] *= kd_samples
            elif operation == "abs":
                env.sim.model.actuator_gainprm[env_ids[:, None], ctrl_ids, 0] = kp_samples
                env.sim.model.actuator_biasprm[env_ids[:, None], ctrl_ids, 1] = -kp_samples
                env.sim.model.actuator_biasprm[env_ids[:, None], ctrl_ids, 2] = -kd_samples
        else:
            # For BamM6Actuator (or other custom actuators with set_gains):
            # Use per-env gain scaling instead of modifying MuJoCo model params.
            from mjlab_microduck.actuator.bam_actuator import BamM6Actuator
            if isinstance(base_actuator, BamM6Actuator):
                # kp_samples shape: (num_envs, num_joints) — average across joints for a scalar scale
                kp_mean = kp_samples.mean(dim=1, keepdim=True)
                kd_mean = kd_samples.mean(dim=1, keepdim=True)
                base_actuator.set_gains(env_ids, kp_scale=kp_mean, kd_scale=kd_mean)


def randomize_mass_and_inertia(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    scale_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
    """Randomize body mass and inertia together with the same scaling factor.

    This maintains physical consistency - mass and inertia must scale together
    to avoid creating invalid inertia tensors that cause simulation instability.

    Args:
        env: The environment
        env_ids: Environment IDs to randomize
        scale_range: (min, max) scaling factor applied to both mass and inertia
        asset_cfg: Asset configuration specifying which bodies to randomize
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]

    # Get body indices
    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, slice):
        body_ids = list(range(asset.num_bodies))[body_ids]
    body_indices = asset.indexing.body_ids[body_ids]

    # Sample ONE random scale per environment (applied to both mass and inertia)
    num_envs = len(env_ids)
    num_bodies = len(body_indices)
    scales = torch.rand(num_envs, num_bodies, device=env.device) * (scale_range[1] - scale_range[0]) + scale_range[0]

    # Store original values on first call
    if not hasattr(env, '_original_mass_inertia'):
        env._original_mass_inertia = {
            'mass': env.sim.model.body_mass[0, body_indices].clone(),
            'inertia': env.sim.model.body_inertia[0, body_indices].clone(),
        }

    # Reset to original first (to prevent accumulation)
    original = env._original_mass_inertia
    env.sim.model.body_mass[env_ids[:, None], body_indices] = original['mass'].unsqueeze(0).expand(num_envs, -1)
    env.sim.model.body_inertia[env_ids[:, None], body_indices] = original['inertia'].unsqueeze(0).expand(num_envs, -1, -1)

    # Apply same scale to both mass and inertia
    env.sim.model.body_mass[env_ids[:, None], body_indices] *= scales
    env.sim.model.body_inertia[env_ids[:, None], body_indices] *= scales.unsqueeze(-1)  # Scale all 3 inertia components


def standing_envs_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    standing_stages: list[dict],
) -> torch.Tensor:
    """Update the relative number of standing environments based on training progress.

    Args:
        env: The RL environment
        env_ids: Environment IDs (unused, but required by curriculum interface)
        command_name: Name of the velocity command term
        standing_stages: List of dicts with 'step' and 'rel_standing_envs' keys
            Example: [
                {"step": 0, "rel_standing_envs": 0.02},
                {"step": 1000, "rel_standing_envs": 0.1},
                {"step": 2000, "rel_standing_envs": 0.2},
            ]

    Returns:
        Current rel_standing_envs value as a tensor
    """
    del env_ids  # Unused

    from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
    from typing import cast

    command_term = env.command_manager.get_term(command_name)
    assert command_term is not None, f"Command term '{command_name}' not found"

    cfg = cast(UniformVelocityCommandCfg, command_term.cfg)

    # Update rel_standing_envs based on current step
    for stage in standing_stages:
        if env.common_step_counter > stage["step"]:
            cfg.rel_standing_envs = stage["rel_standing_envs"]

    return torch.tensor([cfg.rel_standing_envs])


def velocity_tracking_std_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    reward_name: str,
    std_stages: list[dict],
) -> torch.Tensor:
    """Update velocity tracking std parameter based on training progress.

    Starts with loose std (easy rewards) to learn basic walking, then gradually
    tightens to improve velocity tracking accuracy.

    Args:
        env: The RL environment
        env_ids: Environment IDs (unused, but required by curriculum interface)
        reward_name: Name of the reward term (e.g., "track_linear_velocity")
        std_stages: List of dicts with 'step' and 'std' keys
            Example: [
                {"step": 0, "std": 0.5},      # Start loose - learn to walk
                {"step": 250, "std": 0.3},     # Moderate - refine gait
                {"step": 500, "std": 0.2},     # Strict - accurate tracking
            ]

    Returns:
        Current std value as a tensor
    """
    del env_ids  # Unused

    # Get reward term configuration
    reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)

    # Update std based on current step
    current_std = std_stages[0]["std"]  # Default to first stage

    for stage in std_stages:
        if env.common_step_counter > stage["step"]:
            current_std = stage["std"]

    # Update the reward term's std parameter
    reward_term_cfg.params["std"] = current_std

    return torch.tensor([current_std])


def push_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    event_name: str,
    push_stages: list[dict],
) -> torch.Tensor:
    """Update push velocity range based on training progress.

    Starts with no/small pushes to learn clean walking, then gradually increases
    to build robustness without disrupting early learning.

    Args:
        env: The RL environment
        env_ids: Environment IDs (unused, but required by curriculum interface)
        event_name: Name of the push event term (e.g., "push_robot")
        push_stages: List of dicts with 'step' and 'velocity_range' keys
            Example: [
                {"step": 0, "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}},
                {"step": 250, "velocity_range": {"x": (-0.15, 0.15), "y": (-0.15, 0.15)}},
                {"step": 500, "velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
            ]

    Returns:
        Current max push magnitude as a tensor
    """
    del env_ids  # Unused

    # Access event configuration directly from environment config
    assert event_name in env.cfg.events, f"Event '{event_name}' not found"
    event_cfg = env.cfg.events[event_name]

    # Update velocity_range based on current step
    current_range = push_stages[0]["velocity_range"]  # Default to first stage

    for stage in push_stages:
        if env.common_step_counter > stage["step"]:
            current_range = stage["velocity_range"]

    # Update the event configuration's velocity_range parameter
    event_cfg.params["velocity_range"] = current_range

    # Return max magnitude for logging
    max_push = max(abs(current_range["x"][0]), abs(current_range["x"][1]))
    return torch.tensor([max_push])


def wheel_friction_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    event_name: str,
    ranges_stages: list[dict],
) -> torch.Tensor:
    """Update wheel friction based on training step stages."""
    del env_ids  # Unused

    current_ranges = ranges_stages[0]["ranges"]
    for stage in ranges_stages:
        if env.common_step_counter > stage["step"]:
            current_ranges = stage["ranges"]

    env.event_manager.get_term_cfg(event_name).params["ranges"] = current_ranges
    return torch.tensor([current_ranges[0]])


def neck_offset_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    event_name: str,
    offset_stages: list[dict],
) -> torch.Tensor:
    """Update neck offset magnitude based on training progress.

    Gradually increases the max random neck offset so the robot first learns
    to walk with no head disturbance, then progressively harder ones.

    Args:
        env: The RL environment
        env_ids: Environment IDs (unused, but required by curriculum interface)
        event_name: Name of the neck offset event (e.g., "randomize_neck_offset_target")
        offset_stages: List of dicts with 'step' and 'max_offset' keys
            Example: [
                {"step": 0,         "max_offset": 0.0},
                {"step": 500 * 24,  "max_offset": 0.1},
                {"step": 1000 * 24, "max_offset": 0.2},
                {"step": 1500 * 24, "max_offset": 0.3},
            ]

    Returns:
        Current max_offset value as a tensor (for logging)
    """
    del env_ids  # Unused

    assert event_name in env.cfg.events, f"Event '{event_name}' not found"
    event_cfg = env.cfg.events[event_name]

    current_offset = offset_stages[0]["max_offset"]
    for stage in offset_stages:
        if env.common_step_counter > stage["step"]:
            current_offset = stage["max_offset"]

    event_cfg.params["max_offset"] = current_offset
    return torch.tensor([current_offset])


def com_range_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    event_name: str,
    range_stages: list[dict],
) -> torch.Tensor:
    """Update CoM randomization range based on training progress.

    Gradually increases the CoM offset range so the robot first learns to walk
    with a small CoM uncertainty, then progressively larger.

    Args:
        env: The RL environment
        env_ids: Environment IDs (unused)
        event_name: Name of the CoM randomization event (e.g., "randomize_com")
        range_stages: List of dicts with 'step' and 'range' keys (range in meters)
            Example: [
                {"step": 0,          "range": 0.003},
                {"step": 1000 * 24,  "range": 0.005},
                {"step": 2000 * 24,  "range": 0.008},
            ]

    Returns:
        Current range value as a tensor (for logging)
    """
    del env_ids

    assert event_name in env.cfg.events, f"Event '{event_name}' not found"
    event_cfg = env.cfg.events[event_name]

    current_range = range_stages[0]["range"]
    for stage in range_stages:
        if env.common_step_counter > stage["step"]:
            current_range = stage["range"]

    event_cfg.params["ranges"] = (-current_range, current_range)
    return torch.tensor([current_range])


def velocity_command_ranges_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    velocity_stages: list[dict],
    update_lin_vel_y: bool = True,
    update_ang_vel_z: bool = True,
    forward_only: bool = False,
) -> torch.Tensor:
    """Update velocity command ranges based on training progress.

    Gradually increases the commanded velocity ranges to allow the robot to learn
    higher speeds progressively. Starts with smaller ranges for stable learning,
    then expands to more challenging velocities.

    Args:
        env: The RL environment
        env_ids: Environment IDs (unused, but required by curriculum interface)
        command_name: Name of the velocity command term (e.g., "twist")
        velocity_stages: List of dicts with 'step', 'lin_vel_range', and 'ang_vel_range' keys
            Example: [
                {"step": 0, "lin_vel_range": 0.3, "ang_vel_range": 1.5},
                {"step": 500 * 24, "lin_vel_range": 0.4, "ang_vel_range": 1.75},
                {"step": 1000 * 24, "lin_vel_range": 0.5, "ang_vel_range": 2.0},
            ]

    Returns:
        Current max linear velocity as a tensor
    """
    del env_ids  # Unused

    from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
    from typing import cast

    command_term = env.command_manager.get_term(command_name)
    assert command_term is not None, f"Command term '{command_name}' not found"

    cfg = cast(UniformVelocityCommandCfg, command_term.cfg)

    # Update velocity ranges based on current step
    current_lin_vel = velocity_stages[0]["lin_vel_range"]
    current_ang_vel = velocity_stages[0]["ang_vel_range"]

    for stage in velocity_stages:
        if env.common_step_counter > stage["step"]:
            current_lin_vel = stage["lin_vel_range"]
            current_ang_vel = stage["ang_vel_range"]

    # Update command ranges
    if forward_only:
        cfg.ranges.lin_vel_x = (0.0, current_lin_vel)
    else:
        cfg.ranges.lin_vel_x = (-current_lin_vel, current_lin_vel)
    if update_lin_vel_y:
        cfg.ranges.lin_vel_y = (-current_lin_vel, current_lin_vel)
    if update_ang_vel_z:
        cfg.ranges.ang_vel_z = (-current_ang_vel, current_ang_vel)

    return torch.tensor([current_lin_vel])


def projected_gravity(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Projected gravity vector in body frame.

    Returns the gravity vector projected into the robot's body frame,
    representing pure orientation without linear acceleration.
    This is simpler than raw accelerometer and only depends on orientation.

    Returns:
        torch.Tensor: Projected gravity in body frame (num_envs, 3)
    """
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b


def raw_accelerometer(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Raw accelerometer reading (includes gravity + linear acceleration).

    Returns normalized raw accelerometer which mimics what a real IMU measures.
    This is different from pure projected_gravity which only reflects orientation.
    Reads from the MuJoCo accelerometer sensor "imu_accel".

    Returns:
        torch.Tensor: Normalized raw accelerometer reading (num_envs, 3)
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Access the model to find the sensor address
    # The accelerometer sensor is the 5th sensor (index 4) in robot.xml
    # Sensors: framequat, gyro, gyro, velocimeter, accelerometer, subtreeangmom
    mj_model = asset.data.model

    # Get sensor address from model arrays (sensor_adr is torch tensor)
    sensor_adr_array = mj_model.sensor_adr  # This is a TorchArray/tensor
    sensor_id = 4  # imu_accel is the 5th sensor (0-indexed)
    sensor_adr = int(sensor_adr_array[sensor_id].item())  # Convert to Python int

    # Read accelerometer data (specific force measured by sensor)
    # Shape: (num_envs, 3)
    accel_raw = asset.data.data.sensordata[:, sensor_adr:sensor_adr+3]

    # MuJoCo accelerometer measures specific force (like real sensor)
    # Negate to match convention: when at rest upright, should point down
    accel_negated = -accel_raw

    # Normalize to unit vector
    accel_norm = torch.norm(accel_negated, dim=-1, keepdim=True)
    accel_normalized = torch.where(
        accel_norm > 0.1,
        accel_negated / accel_norm,
        asset.data.projected_gravity_b  # Fallback to projected gravity
    )

    return accel_normalized

def randomize_imu_orientation(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    max_angle_deg: float = 2.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
    """Randomize IMU sensor mounting orientation by small angles.
    
    Simulates slight mounting errors or calibration offsets in the real robot.
    The IMU orientation is randomized by rotating around random axes by up to max_angle_deg.
    
    Args:
        env: The environment
        env_ids: Environment IDs to randomize
        max_angle_deg: Maximum rotation angle in degrees (default 2.0°)
        asset_cfg: Asset configuration
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)
    
    asset: Entity = env.scene[asset_cfg.name]

    # IMU site is the first site (index 0) in robot.xml
    # Sites: imu (0), left_foot (1), right_foot (2)
    site_id = 0
    
    # Store original orientation on first call
    if not hasattr(env, '_original_imu_quat'):
        env._original_imu_quat = env.sim.model.site_quat[0, site_id].clone()
    
    # Generate random rotations for each environment
    num_envs = len(env_ids)
    max_angle_rad = max_angle_deg * torch.pi / 180.0
    
    # Random rotation angles [-max_angle, +max_angle] for each axis
    angles = (torch.rand(num_envs, 3, device=env.device) * 2 - 1) * max_angle_rad
    
    # Convert Euler angles to quaternions (small angle approximation for efficiency)
    # For small angles: quat ≈ [1, θx/2, θy/2, θz/2]
    half_angles = angles / 2.0
    quats_delta = torch.zeros(num_envs, 4, device=env.device)
    quats_delta[:, 0] = 1.0  # w component
    quats_delta[:, 1:] = half_angles  # x, y, z components
    
    # Normalize the quaternion
    quats_delta = quats_delta / torch.norm(quats_delta, dim=1, keepdim=True)
    
    # Get original quaternion and apply delta rotation
    original_quat = env._original_imu_quat.unsqueeze(0).expand(num_envs, -1)
    
    # Quaternion multiplication: q_new = q_delta * q_original
    # q1 * q2 = [w1*w2 - dot(v1,v2), w1*v2 + w2*v1 + cross(v1,v2)]
    w1, x1, y1, z1 = quats_delta[:, 0], quats_delta[:, 1], quats_delta[:, 2], quats_delta[:, 3]
    w2, x2, y2, z2 = original_quat[:, 0], original_quat[:, 1], original_quat[:, 2], original_quat[:, 3]
    
    new_quat = torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,  # w
        w1*x2 + x1*w2 + y1*z2 - z1*y2,  # x
        w1*y2 - x1*z2 + y1*w2 + z1*x2,  # y
        w1*z2 + x1*y2 - y1*x2 + z1*w2,  # z
    ], dim=1)
    
    # Apply to the selected environments
    env.sim.model.site_quat[env_ids, site_id] = new_quat


def standing_phase(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Simple time-based phase for standing task.

    Returns a scalar phase value that cycles from 0 to 1 based on time.
    This allows the policy to have a sense of time progression even when standing.

    Args:
        env: The RL environment
        asset_cfg: Not used, but kept for API consistency

    Returns:
        Phase value [0, 1] as tensor of shape (num_envs, 1)
    """
    # Simple time-based phase that cycles every 2 seconds
    # This gives the policy a time-varying signal
    phase_period = 2.0  # seconds
    time = env.episode_length_buf * env.step_dt
    phase = (time % phase_period) / phase_period

    return phase.unsqueeze(-1)  # Shape: (num_envs, 1)


def air_time_adaptive(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str = "twist",
    command_threshold: float = 0.01,    # below this: no reward (standing)
    running_threshold: float = 0.5,     # above this: use running air-time window
    walk_threshold_min: float = 0.10,
    walk_threshold_max: float = 0.25,
    run_threshold_min: float = 0.05,
    run_threshold_max: float = 0.25,
) -> torch.Tensor:
    """Air-time reward with separate swing-time windows for walking vs running.

    - command < command_threshold  → 0 (standing, no reward)
    - command_threshold–running_threshold → walk window [walk_min, walk_max]
    - command > running_threshold  → run  window [run_min,  run_max]

    This lets the walking gait keep its deliberate 100–250 ms swing while
    running can use a faster 50–250 ms cadence.
    """
    sensor = env.scene.sensors[sensor_name]
    current_air_time = sensor.data.current_air_time  # (num_envs, num_feet)
    assert current_air_time is not None

    command = env.command_manager.get_command(command_name)
    total_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])

    is_walking = ((total_speed >= command_threshold) & (total_speed < running_threshold)).float()  # (num_envs,)
    is_running = (total_speed >= running_threshold).float()

    # Per-env thresholds broadcast over feet
    tmin = (is_walking * walk_threshold_min + is_running * run_threshold_min).unsqueeze(1)
    tmax = (is_walking * walk_threshold_max + is_running * run_threshold_max).unsqueeze(1)

    in_range = (current_air_time > tmin) & (current_air_time < tmax)
    reward = torch.sum(in_range.float(), dim=1)  # sum over feet

    # Zero reward when standing
    active = (total_speed >= command_threshold).float()
    return reward * active


def stillness_at_zero_command(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_name: str = "twist",
    command_threshold: float = 0.01,
    vel_std: float = 0.1,
) -> torch.Tensor:
    """Reward staying still when command is near zero.

    Returns exp(-body_vel² / vel_std²) when command < threshold, else 0.
    This is monotonically decreasing with body speed — moving faster is always
    less rewarding. There is no threshold the robot can cross to 'escape' it,
    unlike gate-based stepping penalties.
    """
    asset: Entity = env.scene[asset_cfg.name]

    command = env.command_manager.get_command(command_name)
    total_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    is_standing_cmd = (total_speed < command_threshold).float()

    body_vel = torch.norm(asset.data.root_link_vel_w[:, :2], dim=1)
    stillness = torch.exp(-body_vel ** 2 / vel_std ** 2)

    return is_standing_cmd * stillness


def joint_vel_l2_when_standing(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_name: str = "twist",
    command_threshold: float = 0.01,
) -> torch.Tensor:
    """Penalise leg joint velocities only when command is near zero.

    Targets the standing-shake problem: the policy makes rapid oscillating
    corrections around the home pose when standing. Gated on command so it
    does not affect the walking gait at all.
    """
    asset: Entity = env.scene[asset_cfg.name]

    command = env.command_manager.get_command(command_name)
    total_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    is_standing_cmd = (total_speed < command_threshold).float()

    leg_indices = list(range(0, 5)) + list(range(9, 14))
    joint_vel = asset.data.joint_vel[:, leg_indices]
    vel_sq = torch.sum(joint_vel ** 2, dim=-1)

    return is_standing_cmd * vel_sq


def foot_step_penalty_when_standing(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_name: str = "twist",
    command_threshold: float = 0.01,
    body_vel_threshold: float = 0.2,
    air_time_threshold: float = 0.05,
) -> torch.Tensor:
    """Penalise stepping when at zero command and the body is not being pushed.

    Symmetric counterpart to the air_time reward:
    - air_time gives  +reward for stepping when command > threshold  (walk)
    - this gives      -reward for stepping when command < threshold  (stand)

    The body-velocity gate prevents penalising recovery steps after a push:
    if the robot is already moving fast (pushed), no penalty is applied so it
    can still take steps to catch itself.

    Returns a value in [0, 1] (use a negative weight in the config).
    """
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor = env.scene.sensors["feet_ground_contact"]

    # Was either foot recently lifted? (last completed air phase > threshold)
    air_time = contact_sensor.data.last_air_time[:, :2]  # (num_envs, 2)
    any_foot_stepped = (air_time > air_time_threshold).any(dim=1).float()

    # Are we in standing mode? (command near zero)
    command = env.command_manager.get_command(command_name)
    total_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    is_standing = (total_speed < command_threshold).float()

    # Is the body still? (not being pushed)
    body_vel = torch.norm(asset.data.root_link_vel_w[:, :2], dim=1)
    is_still = (body_vel < body_vel_threshold).float()

    return any_foot_stepped * is_standing * is_still


def recovery_stepping_reward(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_name: str = "twist",
    command_threshold: float = 0.01,
    velocity_threshold: float = 0.3,
    air_time_threshold: float = 0.05,
) -> torch.Tensor:
    """Reward foot air time only when at zero command AND robot has high velocity (recovering from push).

    This encourages the robot to take steps to recover balance when pushed,
    but does NOT fire during normal walking (command > threshold).

    Args:
        env: The RL environment
        asset_cfg: Asset configuration (unused but kept for API consistency)
        command_name: Name of the velocity command in the command manager
        command_threshold: Speed below which the robot is considered to be in standing mode
        velocity_threshold: Linear velocity threshold to activate stepping reward (m/s)
        air_time_threshold: Minimum air time to count as a step (seconds)

    Returns:
        Reward tensor of shape (num_envs,)
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Only fire for standing envs (command near zero)
    command = env.command_manager.get_command(command_name)
    total_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    is_standing_cmd = (total_speed < command_threshold).float()

    # Get base linear velocity magnitude
    base_lin_vel = asset.data.root_link_vel_w[:, :3]  # (num_envs, 3)
    vel_magnitude = torch.norm(base_lin_vel[:, :2], dim=1)  # Only XY plane

    # Only reward stepping when velocity is high (being pushed)
    should_step = vel_magnitude > velocity_threshold

    # Get foot air time from contact sensor
    contact_sensor = env.scene.sensors["feet_ground_contact"]
    air_time = contact_sensor.data.last_air_time[:, :2]  # (num_envs, 2) - left and right foot

    # Reward if either foot has been in air recently
    foot_in_air = (air_time > air_time_threshold).any(dim=1)  # (num_envs,)

    # Only give reward when: standing command AND high body velocity AND foot stepped
    reward = is_standing_cmd * should_step.float() * foot_in_air.float()

    return reward


def adaptive_pose_weight(
    env: ManagerBasedRlEnv,
    base_pose_reward: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    velocity_threshold: float = 0.3,
    min_weight: float = 0.3,
) -> torch.Tensor:
    """Reduce pose tracking weight when robot has high velocity (recovering from push).

    This gives the robot freedom to deviate from the standing pose when taking
    recovery steps, while maintaining strict pose tracking when standing still.

    Args:
        env: The RL environment
        base_pose_reward: The original pose reward (before weighting)
        asset_cfg: Asset configuration (unused but kept for API consistency)
        velocity_threshold: Linear velocity threshold to start reducing weight (m/s)
        min_weight: Minimum weight multiplier (0-1) at high velocities

    Returns:
        Weighted reward tensor of shape (num_envs,)
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Get base linear velocity magnitude
    base_lin_vel = asset.data.root_link_vel_w[:, :3]  # (num_envs, 3)
    vel_magnitude = torch.norm(base_lin_vel[:, :2], dim=1)  # Only XY plane

    # Compute weight: 1.0 when stationary, min_weight at high velocity
    # Use smooth transition via sigmoid-like function
    weight = min_weight + (1.0 - min_weight) * torch.exp(
        -((vel_magnitude - velocity_threshold) / velocity_threshold).clamp(min=0.0) ** 2
    )

    return base_pose_reward * weight


def randomize_base_orientation(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    max_pitch_deg: float = 10.0,
    max_roll_deg: float = 5.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
    """Randomize base orientation at episode start to force reactive behavior.

    Adds random pitch and roll to the robot's base orientation at the start of
    each episode. This prevents the policy from memorizing a single initial state
    and forces it to use feedback to adapt to different orientations.

    Args:
        env: The environment
        env_ids: Environment IDs to randomize
        max_pitch_deg: Maximum pitch angle in degrees (forward/backward tilt)
        max_roll_deg: Maximum roll angle in degrees (side-to-side tilt)
        asset_cfg: Asset configuration
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]
    num_envs = len(env_ids)

    # Generate random pitch and roll angles
    max_pitch_rad = max_pitch_deg * torch.pi / 180.0
    max_roll_rad = max_roll_deg * torch.pi / 180.0

    pitch = (torch.rand(num_envs, device=env.device) * 2 - 1) * max_pitch_rad
    roll = (torch.rand(num_envs, device=env.device) * 2 - 1) * max_roll_rad
    yaw = torch.zeros(num_envs, device=env.device)  # Keep yaw at 0

    # Convert Euler angles (roll, pitch, yaw) to quaternion
    # Using the standard aerospace sequence (ZYX)
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

    # Normalize quaternion
    new_quat = new_quat / torch.norm(new_quat, dim=1, keepdim=True)

    # Get root position index (freejoint starts at qpos index 0)
    # Freejoint: [x, y, z, qw, qx, qy, qz]
    root_quat_idx = 3  # Quaternion starts at index 3

    # Apply the randomized orientation to selected environments
    env.sim.data.qpos[env_ids, root_quat_idx:root_quat_idx+4] = new_quat


def set_face_down_orientation(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
    """Set the robot to a prone (belly-down) orientation for stand-up training.

    Rotates the robot 90° forward around the pitch axis (Y) so the front/belly
    faces the ground and legs point upward. Combined with a random yaw.

    Quaternion derivation:
        quat_pitch90 = [s, 0, s, 0]   where s = sqrt(2)/2  (90° around Y)
        quat_yaw     = [cy, 0, 0, sy]
        combined     = quat_yaw * quat_pitch90 = [s*cy, -s*sy, s*cy, s*sy]
    """
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.int)
    num = len(env_ids)

    yaw = torch.rand(num, device=env.device) * 2 * np.pi - np.pi
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    s = 2.0 ** -0.5  # sqrt(2)/2

    new_quat = torch.stack(
        [
            s * cy,   # w
            -s * sy,  # x
            s * cy,   # y
            s * sy,   # z
        ],
        dim=1,
    )

    # Freejoint qpos: [x, y, z, qw, qx, qy, qz, ...]
    env.sim.data.qpos[env_ids, 3:7] = new_quat
    env.sim.data.qvel[env_ids, :6] = 0.0


def set_random_prone_orientation(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
    """Randomly initialize each env as face-down (belly) or face-up (back), with random yaw.

    Face-down:  +90° pitch → quat = [s*cy, -s*sy,  s*cy,  s*sy]
    Face-up:    -90° pitch → quat = [s*cy,  s*sy, -s*cy,  s*sy]
    """
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.int)
    num = len(env_ids)

    yaw = torch.rand(num, device=env.device) * 2 * np.pi - np.pi
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    s = 2.0 ** -0.5  # sqrt(2)/2

    face_down = torch.stack([ s * cy, -s * sy,  s * cy,  s * sy], dim=1)
    face_up   = torch.stack([ s * cy,  s * sy, -s * cy,  s * sy], dim=1)

    # Randomly assign each env to face-down or face-up (50/50)
    mask = torch.rand(num, device=env.device) < 0.5  # True → face-down
    new_quat = torch.where(mask.unsqueeze(1), face_down, face_up)

    env.sim.data.qpos[env_ids, 3:7] = new_quat
    env.sim.data.qvel[env_ids, :6] = 0.0


class VelocityCommandCommandOnly(UniformVelocityCommand):
    """Like UniformVelocityCommand but only draws the command arrows (no actual velocity arrows)."""

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

        # Command linear velocity arrow (blue).
        cmd_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
        cmd_lin_to = local_to_world(
            (np.array([0, 0, z_offset]) + np.array([cmd[0], cmd[1], 0])) * scale
        )
        visualizer.add_arrow(cmd_lin_from, cmd_lin_to, color=(0.2, 0.2, 0.6, 0.6), width=0.015)


class VelocityCommandCommandOnlyCfg(UniformVelocityCommandCfg):
    def build(self, env: ManagerBasedRlEnv) -> "VelocityCommandCommandOnly":
        return VelocityCommandCommandOnly(self, env)


class RelativeHeadingVelocityCommand(VelocityCommandCommandOnly):
    """Velocity command where cmd[2] is the heading error in the robot's body frame.

    cmd[0] = lin_vel_x  (throttle: 0=coast, +push, -brake)
    cmd[1] = lin_vel_y  (unused, 0)
    cmd[2] = heading_error  (+ = target is to the right/CW, - = to the left/CCW)
             0 → go straight, ±max = target is max_angle rad to the right/left

    During training: a random world-frame heading is sampled at each episode reset.
    At every step, cmd[2] = clamp(wrap(current_yaw - target_yaw), ±max_angle).
    Positive when the robot is pointing CCW (left) of the target → needs to turn right.

    At inference: the user feeds cmd[2] directly.  Holding cmd[2] = constant gives
    a proportional heading correction = approximately constant turn rate.

    Set heading_command=False and rel_heading_envs=0.0 in the cfg (we handle
    heading internally).  ang_vel_z range in cfg is used as the clip limit for cmd[2].
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        # Sampled target heading per env, world frame (rad)
        self._target_heading_w = torch.zeros(self.num_envs, device=self.device)
        # Clip limit for cmd[2]: use ang_vel_z[1] from cfg (the positive bound)
        ang_rng = cfg.ranges.ang_vel_z
        self._heading_max = float(ang_rng[1]) if ang_rng else 1.0

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        n = len(env_ids)
        # Sample random world-frame target heading uniformly in [-π, π]
        self._target_heading_w[env_ids] = (
            torch.rand(n, device=self.device) * 2.0 * math.pi - math.pi
        )
        # Zero ang_vel slot; _update_command will fill it each step
        self.vel_command_b[env_ids, 2] = 0.0

    def _update_command(self) -> None:
        # Do NOT call super()._update_command() — it would run the heading
        # proportional controller and overwrite cmd[2] with a yaw rate.
        # Instead recompute heading error from scratch each step.
        quat = self.robot.data.root_link_quat_w  # (N, 4) [w, x, y, z]
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        current_yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        # Positive = target is CCW (left) of robot → turn left. Standard convention.
        delta = self._target_heading_w - current_yaw
        heading_error = torch.atan2(torch.sin(delta), torch.cos(delta))
        self.vel_command_b[:, 2] = heading_error.clamp(-self._heading_max, self._heading_max)

    def _update_metrics(self) -> None:
        pass  # No velocity tracking metrics for heading command


class RelativeHeadingVelocityCommandCfg(UniformVelocityCommandCfg):
    def build(self, env: ManagerBasedRlEnv) -> "RelativeHeadingVelocityCommand":
        return RelativeHeadingVelocityCommand(self, env)


def heading_tracking_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float = 0.5,
) -> torch.Tensor:
    """Reward for reducing heading error when cmd[2] encodes heading error.

    Returns exp(-cmd[2]² / std²).
    - At error = 0 (on heading): reward = 1.0.
    - At error = std: reward ≈ 0.37 (strong gradient).
    - At error = 1.0 rad with std=0.5: reward ≈ 0.018 (nearly zero).

    std=0.5 rad (≈28°) gives a meaningful gradient across the expected range.
    """
    cmd = env.command_manager.get_command(command_name)
    heading_error = cmd[:, 2]
    return torch.exp(-(heading_error ** 2) / (std ** 2))


def skating_air_time_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    threshold_min: float = 0.05,
    threshold_max: float = 0.4,
) -> torch.Tensor:
    """Reward feet air time only when pushing (cmd_x > 0).

    Encourages the robot to lift each foot during the recovery phase of the
    skating stroke rather than dragging it on the ground.
    Scaled by cmd_x so the incentive grows with push intensity.
    """
    from mjlab.sensor import ContactSensor
    sensor: ContactSensor = env.scene[sensor_name]
    current_air_time = sensor.data.current_air_time
    assert current_air_time is not None

    in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
    reward = torch.sum(in_range.float(), dim=1)

    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    return reward * torch.clamp(cmd_x, min=0.0)


def forward_lean_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    target_pitch: float = 0.08,
    std: float = 0.08,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=("trunk_base",)),
) -> torch.Tensor:
    """Reward leaning slightly forward when pushing, to counteract the backward
    torque from skating strokes.

    Uses projected_gravity_b x-component as a pitch proxy:
      forward_lean = -gravity_b[:, 0]  (positive when leaning forward)

    Only fires when cmd_x > 0. Peaks at target_pitch radians of forward lean.
    """
    asset: Entity = env.scene[asset_cfg.name]
    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    forward_lean = asset.data.projected_gravity_b[:, 0]
    push = torch.clamp(cmd_x, min=0.0)
    return push * torch.exp(-((forward_lean - target_pitch) ** 2) / (std ** 2))


class GroundPickPhaseCommand(UniformVelocityCommand):
    """Phase-encoding command for the ground pick task.

    Replaces the velocity command with a cyclic phase signal:
        command = [cos(2π*phase), sin(2π*phase), 0]

    Phase ∈ [0, 0.5]: approach (go down, touch ground with mouth).
    Phase ∈ [0.5, 1.0]: return (stand back up).

    Phase is randomized per environment on episode reset to decorrelate envs.
    Period is 4 seconds by default (2 s down + 2 s up).
    """

    PERIOD: float = 4.0  # seconds per full cycle

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._gp_phase = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.vel_command_b

    def compute(self, dt: float) -> None:
        self._gp_phase = (self._gp_phase + dt / self.PERIOD) % 1.0
        self.vel_command_b[:, 0] = torch.cos(2 * torch.pi * self._gp_phase)
        self.vel_command_b[:, 1] = torch.sin(2 * torch.pi * self._gp_phase)
        self.vel_command_b[:, 2] = 0.0

    def reset(self, env_ids: torch.Tensor | None) -> dict:
        if env_ids is not None and len(env_ids) > 0:
            self._gp_phase[env_ids] = torch.rand(len(env_ids), device=self.device)
        return {}

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        pass  # Phase is continuous; no resampling needed

    def _update_command(self) -> None:
        pass  # Updated in compute()

    def _update_metrics(self) -> None:
        pass  # No velocity tracking metrics for ground pick


class GroundPickPhaseCommandCfg(UniformVelocityCommandCfg):
    class_type: type = GroundPickPhaseCommand

    def build(self, env: ManagerBasedRlEnv) -> "GroundPickPhaseCommand":
        return GroundPickPhaseCommand(self, env)


class BodyPoseCommand(UniformVelocityCommand):
    """Body pose command for standing control: [Δz (m), Δpitch (rad), Δroll (rad)].

    Repurposes the 3-slot velocity command to control body height offset, pitch,
    and roll while the robot is standing. The command is sampled uniformly from the
    configured ranges and resampled at the configured interval.

    Mapping:
        vel_command_b[:, 0] = Δz      (height offset in meters, + = up)
        vel_command_b[:, 1] = Δpitch  (pitch offset in radians, + = forward tilt)
        vel_command_b[:, 2] = Δroll   (roll offset in radians, + = right lean)

    Configure via UniformVelocityCommandCfg ranges:
        ranges.lin_vel_x = (-max_z, max_z)
        ranges.lin_vel_y = (-max_pitch, max_pitch)
        ranges.ang_vel_z = (-max_roll, max_roll)

    Set rel_standing_envs=0.0 and heading_command=False so the parent never zeros
    or heading-adjusts the command.
    """

    def _update_metrics(self) -> None:
        pass  # Body pose has no velocity tracking metrics

    def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
        pass  # No visualization needed


class BodyPoseCommandCfg(UniformVelocityCommandCfg):
    def build(self, env: ManagerBasedRlEnv) -> "BodyPoseCommand":
        return BodyPoseCommand(self, env)


def body_pose_cmd_obs(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    max_z: float = 0.025,
    max_angle: float = math.radians(20),
) -> torch.Tensor:
    """Normalized body pose command observation: [Δz/max_z, Δpitch/max_angle, Δroll/max_angle].

    Returns a 3D vector in [-1, 1] when commands are within their configured ranges,
    preserving the same 3-slot obs shape as the original velocity command.
    """
    cmd = env.command_manager.get_command(command_name)  # (N, 3)
    norm = torch.tensor([max_z, max_angle, max_angle], device=env.device)
    return cmd[:, :3] / norm


def body_pose_tracking(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_name: str = "twist",
    nominal_height: float = 0.095,
    z_std: float = 0.01,
    angle_std: float = math.radians(5),
) -> torch.Tensor:
    """Gaussian reward for tracking commanded body pose (z height, pitch, roll).

    Returns the mean of three independent Gaussian rewards:
        - z:     actual CoM height vs (nominal_height + Δz_cmd)
        - pitch: actual body pitch vs Δpitch_cmd  (ZYX Euler, + = forward tilt)
        - roll:  actual body roll  vs Δroll_cmd   (ZYX Euler, + = right lean)

    Args:
        nominal_height: Nominal standing CoM height in meters.
        z_std:          Std for height tracking Gaussian (meters).
        angle_std:      Std for pitch/roll tracking Gaussians (radians).
    """
    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)  # (N, 3)
    dz_cmd = cmd[:, 0]
    dpitch_cmd = cmd[:, 1]
    droll_cmd = cmd[:, 2]

    # Height above terrain spawn origin (world z minus terrain z).
    # nan_to_num: MuJoCo can produce NaN on contact instability; treat as z=0.
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    z_reward = torch.exp(-((z - (nominal_height + dz_cmd)) / z_std) ** 2)

    # Pitch and roll from quaternion (ZYX Euler angles)
    quat = asset.data.root_link_quat_w  # (N, 4): [w, x, y, z]
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    roll = torch.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx ** 2 + qy ** 2))
    pitch = torch.asin(torch.clamp(2.0 * (qw * qy - qz * qx), -1.0, 1.0))

    pitch_reward = torch.exp(-((pitch - dpitch_cmd) / angle_std) ** 2)
    roll_reward = torch.exp(-((roll - droll_cmd) / angle_std) ** 2)

    return (z_reward + pitch_reward + roll_reward) / 3.0


def body_pose_cmd_range_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    range_stages: list[dict],
) -> torch.Tensor:
    """Update body pose command ranges based on training progress.

    Args:
        command_name: Name of the command term (e.g., "twist").
        range_stages: List of dicts with 'step', 'max_z' (m), 'max_angle' (rad) keys.
            Example: [
                {"step": 0,          "max_z": 0.0,   "max_angle": 0.0},
                {"step": 1000 * 24,  "max_z": 0.01,  "max_angle": 0.087},
                {"step": 3000 * 24,  "max_z": 0.025, "max_angle": 0.349},
            ]
    """
    del env_ids  # Unused

    from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
    from typing import cast

    command_term = env.command_manager.get_term(command_name)
    assert command_term is not None, f"Command term '{command_name}' not found"
    cfg = cast(UniformVelocityCommandCfg, command_term.cfg)

    current_z = range_stages[0]["max_z"]
    current_angle = range_stages[0]["max_angle"]
    for stage in range_stages:
        if env.common_step_counter >= stage["step"]:
            current_z = stage["max_z"]
            current_angle = stage["max_angle"]

    cfg.ranges.lin_vel_x = (-current_z, current_z)
    cfg.ranges.lin_vel_y = (-current_angle, current_angle)
    cfg.ranges.ang_vel_z = (-current_angle, current_angle)

    return torch.tensor(current_z)
