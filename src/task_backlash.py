# Pass the robot cfg mirroring the base task's collision model, so backlash A/B
# comparisons are unconfounded. Obs/action dims stay 14 joints (runtime unchanged).

from copy import deepcopy

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from . import task_mdp as microduck_mdp
from .robot import MICRODUCK_BACKLASH_ROBOT_CFG

_SERVO_JOINTS_ONLY = (r"^(?!passive_).*",)


def make_backlash_variant(cfg: ManagerBasedRlEnvCfg, robot_cfg: EntityCfg = MICRODUCK_BACKLASH_ROBOT_CFG) -> ManagerBasedRlEnvCfg:
    cfg.scene.entities = {**cfg.scene.entities, "robot": robot_cfg}

    for group in ("actor", "critic"):
        terms = cfg.observations[group].terms
        for term_name, func in (("joint_pos", microduck_mdp.joint_pos_rel_backlash), ("joint_vel", microduck_mdp.joint_vel_rel_backlash)):
            term = terms.get(term_name)
            if term is None:
                continue
            term.func = func
            # Envs that never narrowed the selection would otherwise feed the
            # backlash joints themselves into the obs (wrong dim + double count).
            if "asset_cfg" not in term.params:
                term.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=_SERVO_JOINTS_ONLY)

    # Backlash joints legitimately ride their hard limits; the default asset_cfg
    # covers every joint and would charge a permanent soft-limit penalty.
    dof_limits = cfg.rewards.get("dof_pos_limits")
    if dof_limits is not None and "asset_cfg" not in dof_limits.params:
        dof_limits.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=_SERVO_JOINTS_ONLY)

    # The pose reward ERRORS on ambiguous std-dict matches: on this model
    # "passive_left_hip_yaw_backlash" matches both ".*hip_yaw.*" and ".*passive_.*".
    pose = cfg.rewards.get("pose")
    if pose is not None and "asset_cfg" in pose.params:
        # Base templates share SceneEntityCfg objects across make() calls; mutating
        # in place would leak into the base tasks.
        ac = deepcopy(pose.params["asset_cfg"])
        ac.joint_names = tuple(p if "_backlash" in p else r"^(?!passive_.*_backlash)" + p.lstrip("^") for p in ac.joint_names)
        pose.params["asset_cfg"] = ac

    return cfg
