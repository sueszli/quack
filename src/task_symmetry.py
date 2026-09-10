# Actor obs layout (61-D, term insertion order) that the tables below index:
#     [0:3] base_ang_vel  [3:6] projected_gravity  [6:20] joint_pos_rel
#     [20:34] joint_vel_rel  [34:48] last_action  [48:51] twist
#     [51:55] head command  [55:61] body command
# Joint order within each 14-block: 0-4 left leg, 5-8 neck/head, 9-13 right leg.
# hip_pitch/knee/ankle negate because HOME uses opposite signs left vs right
# (left_hip_pitch = +0.6, right_hip_pitch = -0.6).

from dataclasses import dataclass

import torch
from mjlab.rl import RslRlPpoAlgorithmCfg
from tensordict import TensorDict


@dataclass
class PpoWithSymmetryCfg(RslRlPpoAlgorithmCfg):
    symmetry_cfg: dict | None = None


SYMMETRY_CFG = {"use_data_augmentation": False, "use_mirror_loss": True, "mirror_loss_coeff": 0.5, "data_augmentation_func": "src.task_symmetry.microduck_vel_symmetry"}

_JOINT_PERM: list[int] = [9, 10, 11, 12, 13, 5, 6, 7, 8, 0, 1, 2, 3, 4]

# Applied AFTER permutation.
_JOINT_SIGN: list[float] = [-1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1]

_OBS_PERM: list[int] = (
    [0, 1, 2]  # base_ang_vel
    + [3, 4, 5]  # projected_gravity
    + [6 + j for j in _JOINT_PERM]  # joint_pos
    + [20 + j for j in _JOINT_PERM]  # joint_vel
    + [34 + j for j in _JOINT_PERM]  # last_action
    + [48, 49, 50]  # twist command
    + [51, 52, 53, 54]  # head command
    + [55, 56, 57, 58, 59, 60]  # body command
)

_OBS_SIGN: list[float] = (
    [-1.0, 1.0, -1.0]  # base_ang_vel: negate roll, yaw
    + [1.0, -1.0, 1.0]  # projected_gravity: negate gy
    + _JOINT_SIGN  # joint_pos
    + _JOINT_SIGN  # joint_vel
    + _JOINT_SIGN  # last_action
    + [1.0, -1.0, -1.0]  # twist: negate lin_vel_y, ang_vel_z
    + [1.0, 1.0, -1.0, -1.0]  # head: negate head_yaw, head_roll
    + [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]  # body: negate y, roll, yaw
)

_cache: dict[torch.device, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _get_tensors(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if device not in _cache:
        obs_perm = torch.tensor(_OBS_PERM, dtype=torch.long, device=device)
        obs_sign = torch.tensor(_OBS_SIGN, dtype=torch.float32, device=device)
        act_perm = torch.tensor(_JOINT_PERM, dtype=torch.long, device=device)
        act_sign = torch.tensor(_JOINT_SIGN, dtype=torch.float32, device=device)
        _cache[device] = (obs_perm, obs_sign, act_perm, act_sign)
    return _cache[device]


def microduck_vel_symmetry(env, obs: TensorDict | None, actions: torch.Tensor | None) -> tuple[TensorDict | None, torch.Tensor | None]:
    aug_obs: TensorDict | None = None
    aug_actions: torch.Tensor | None = None

    if obs is not None:
        actor_orig: torch.Tensor = obs["actor"]
        obs_perm, obs_sign, _, _ = _get_tensors(actor_orig.device)
        actor_sym = actor_orig[:, obs_perm] * obs_sign

        critic_orig: torch.Tensor = obs["critic"]
        # Critic obs is repeated unmirrored: mirror loss does not need it, and the
        # critic's privileged terms have no actor-side mirror.
        critic_repeated = torch.cat([critic_orig, critic_orig], dim=0)

        aug_obs = TensorDict({"actor": torch.cat([actor_orig, actor_sym], dim=0), "critic": critic_repeated}, batch_size=[actor_orig.shape[0] * 2], device=actor_orig.device)

    if actions is not None:
        _, _, act_perm, act_sign = _get_tensors(actions.device)
        actions_sym = actions[:, act_perm] * act_sign
        aug_actions = torch.cat([actions, actions_sym], dim=0)

    return aug_obs, aug_actions
