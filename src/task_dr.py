from __future__ import annotations

import dataclasses
import math

from mjlab.envs.mdp import dr
from mjlab.managers import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp

from . import task_mdp as microduck_mdp

ARMATURE_ALL_JOINTS = (r".*",)
ARMATURE_SERVO_JOINTS = (r"^(?!passive_).*",)
WHEEL_JOINTS = (r"^passive_.*wheel",)


@dataclasses.dataclass(frozen=True)
class MicroduckDrCfg:
    com: bool = True
    com_range: float = 0.003

    head_com: bool = True
    head_com_range: float = 0.003

    mass_inertia: bool = True
    mass_inertia_range: tuple[float, float] = (0.95, 1.05)

    joint_friction: bool = True
    joint_friction_range: tuple[float, float] = (0.9, 1.1)

    armature: bool = True
    armature_range: tuple[float, float] = (0.9, 1.1)
    armature_joints: tuple[str, ...] = ARMATURE_ALL_JOINTS

    kp: bool = False
    kp_range: tuple[float, float] = (0.85, 1.15)
    kd: bool = False
    kd_range: tuple[float, float] = (0.9, 1.1)

    joint_damping: bool = False
    joint_damping_range: tuple[float, float] = (0.9, 1.1)

    wheel_friction: bool = False
    wheel_friction_range: tuple[float, float] = (0.0, 0.0)

    pushes: bool = True
    push_interval_s: tuple[float, float] = (3.0, 6.0)
    push_play_interval_s: tuple[float, float] | None = (0.5, 1.0)
    push_range: tuple[float, float] = (-0.3, 0.3)

    imu_orientation: bool = True
    imu_orientation_angle_deg: float = 6.0

    encoder_bias: bool = True
    encoder_bias_range: tuple[float, float] = (-0.015, 0.015)

    base_orientation: bool = False
    base_orientation_max_pitch_deg: float = 10.0
    base_orientation_max_roll_deg: float = 5.0


DEFAULT_DR = MicroduckDrCfg()

ROLLER_DR = dataclasses.replace(DEFAULT_DR, armature_joints=ARMATURE_SERVO_JOINTS, wheel_friction=True, push_range=(-0.2, 0.2))


def apply_dr(cfg, dr_cfg: MicroduckDrCfg, head_body_names, play: bool = False) -> None:
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(func=microduck_mdp.expand_bam_friction_fields, mode="startup")

    if dr_cfg.pushes:
        interval = dr_cfg.push_play_interval_s if (play and dr_cfg.push_play_interval_s is not None) else dr_cfg.push_interval_s
        cfg.events["push_robot"] = EventTermCfg(func=mdp.push_by_setting_velocity, mode="interval", interval_range_s=interval, params={"velocity_range": {"x": dr_cfg.push_range, "y": dr_cfg.push_range}, "asset_cfg": SceneEntityCfg("robot")})

    if dr_cfg.com:
        cfg.events["randomize_com"] = EventTermCfg(func=dr.body_ipos, mode="reset", params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)), "operation": "add", "ranges": (-dr_cfg.com_range, dr_cfg.com_range)})

    if dr_cfg.head_com:
        cfg.events["randomize_head_com"] = EventTermCfg(func=dr.body_ipos, mode="reset", params={"asset_cfg": SceneEntityCfg("robot", body_names=head_body_names), "operation": "add", "ranges": (-dr_cfg.head_com_range, dr_cfg.head_com_range)})

    if dr_cfg.kp or dr_cfg.kd:
        kp_range = dr_cfg.kp_range if dr_cfg.kp else (1.0, 1.0)
        kd_range = dr_cfg.kd_range if dr_cfg.kd else (1.0, 1.0)
        cfg.events["randomize_motor_gains"] = EventTermCfg(func=microduck_mdp.randomize_delayed_actuator_gains, mode="reset", params={"asset_cfg": SceneEntityCfg("robot"), "operation": "scale", "kp_range": kp_range, "kd_range": kd_range})

    if dr_cfg.mass_inertia:
        lo, hi = dr_cfg.mass_inertia_range
        cfg.events["randomize_mass_inertia"] = EventTermCfg(func=dr.pseudo_inertia, mode="startup", params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)), "alpha_range": (math.log(lo) / 2.0, math.log(hi) / 2.0)})

    if dr_cfg.joint_friction:
        cfg.events["randomize_joint_friction"] = EventTermCfg(func=microduck_mdp.randomize_bam_friction, mode="reset", params={"asset_cfg": SceneEntityCfg("robot"), "scale_range": dr_cfg.joint_friction_range})

    if dr_cfg.joint_damping:
        cfg.events["randomize_joint_damping"] = EventTermCfg(func=microduck_mdp.randomize_dof_field_scaled, mode="reset", domain_randomization=True, params={"asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)), "field": "dof_damping", "scale_range": dr_cfg.joint_damping_range})

    if dr_cfg.armature:
        cfg.events["randomize_armature"] = EventTermCfg(func=dr.joint_armature, mode="reset", params={"asset_cfg": SceneEntityCfg("robot", joint_names=dr_cfg.armature_joints), "operation": "scale", "ranges": dr_cfg.armature_range})

    if dr_cfg.wheel_friction:
        cfg.events["randomize_wheel_friction"] = EventTermCfg(func=dr.dof_frictionloss, mode="reset", params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINTS), "operation": "abs", "ranges": dr_cfg.wheel_friction_range})

    if dr_cfg.base_orientation:
        cfg.events["randomize_base_orientation"] = EventTermCfg(func=microduck_mdp.randomize_base_orientation, mode="reset", params={"asset_cfg": SceneEntityCfg("robot"), "max_pitch_deg": dr_cfg.base_orientation_max_pitch_deg, "max_roll_deg": dr_cfg.base_orientation_max_roll_deg})
