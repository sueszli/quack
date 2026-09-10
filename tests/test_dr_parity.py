from __future__ import annotations

import dataclasses

import pytest

from src import task_dr

EXPECTED_EVENTS = {"Mjlab-Velocity-Flat-MicroDuck": {"expand_bam_friction_fields", "push_robot", "randomize_com", "randomize_head_com", "randomize_mass_inertia", "randomize_joint_friction", "randomize_armature"}, "Mjlab-StandUp-Flat-MicroDuck": {"expand_bam_friction_fields", "push_robot", "randomize_com", "randomize_head_com", "randomize_mass_inertia", "randomize_joint_friction", "randomize_armature"}, "Mjlab-SitStand-Flat-MicroDuck": {"expand_bam_friction_fields", "push_robot", "randomize_com", "randomize_head_com", "randomize_mass_inertia", "randomize_joint_friction", "randomize_armature"}, "Mjlab-BallKick-Flat-MicroDuck": {"expand_bam_friction_fields", "push_robot", "randomize_com", "randomize_head_com", "randomize_mass_inertia", "randomize_joint_friction", "randomize_armature"}, "Mjlab-GroundPick-Flat-MicroDuck": {"expand_bam_friction_fields", "push_robot", "randomize_com", "randomize_head_com", "randomize_mass_inertia", "randomize_joint_friction", "randomize_armature"}, "Mjlab-Roulade-Flat-MicroDuck": {"expand_bam_friction_fields", "randomize_com", "randomize_head_com", "randomize_mass_inertia", "randomize_joint_friction", "randomize_armature"}, "Mjlab-Velocity-Flat-MicroDuck-Rollers": {"expand_bam_friction_fields", "push_robot", "randomize_com", "randomize_head_com", "randomize_mass_inertia", "randomize_joint_friction", "randomize_armature", "randomize_wheel_friction"}, "Mjlab-Spin-Flat-MicroDuck": {"expand_bam_friction_fields", "push_robot", "randomize_com", "randomize_head_com", "randomize_mass_inertia", "randomize_joint_friction", "randomize_armature", "randomize_wheel_friction"}, "Mjlab-RollerCrouch-Flat-MicroDuck": {"expand_bam_friction_fields", "push_robot", "randomize_com", "randomize_head_com", "randomize_mass_inertia", "randomize_joint_friction", "randomize_armature", "randomize_wheel_friction"}}

WALK_TASKS = ("Mjlab-Velocity-Flat-MicroDuck", "Mjlab-StandUp-Flat-MicroDuck", "Mjlab-SitStand-Flat-MicroDuck", "Mjlab-BallKick-Flat-MicroDuck", "Mjlab-GroundPick-Flat-MicroDuck", "Mjlab-Roulade-Flat-MicroDuck")
ROLLER_TASKS = ("Mjlab-Velocity-Flat-MicroDuck-Rollers", "Mjlab-Spin-Flat-MicroDuck", "Mjlab-RollerCrouch-Flat-MicroDuck")


def _env_cfg(task_id):
    from mjlab.tasks.registry import load_env_cfg

    return load_env_cfg(task_id)


@pytest.mark.parametrize("task_id", sorted(EXPECTED_EVENTS))
def test_dr_events_present(task_id):
    import src.task_registry  # noqa: F401

    events = set(_env_cfg(task_id).events)
    missing = EXPECTED_EVENTS[task_id] - events
    assert not missing, f"{task_id} lost DR events: {sorted(missing)}"


@pytest.mark.parametrize("task_id", WALK_TASKS)
def test_walk_tasks_randomize_all_joint_armature(task_id):
    import src.task_registry  # noqa: F401

    term = _env_cfg(task_id).events["randomize_armature"]
    assert term.params["asset_cfg"].joint_names == task_dr.ARMATURE_ALL_JOINTS
    assert term.params["ranges"] == (0.9, 1.1)


@pytest.mark.parametrize("task_id", ROLLER_TASKS)
def test_roller_tasks_exclude_passive_from_armature(task_id):
    import src.task_registry  # noqa: F401

    term = _env_cfg(task_id).events["randomize_armature"]
    assert term.params["asset_cfg"].joint_names == task_dr.ARMATURE_SERVO_JOINTS


@pytest.mark.parametrize(("task_id", "expected"), [("Mjlab-Velocity-Flat-MicroDuck", (-0.3, 0.3)), ("Mjlab-GroundPick-Flat-MicroDuck", (-0.15, 0.15)), ("Mjlab-Spin-Flat-MicroDuck", (-0.2, 0.2)), ("Mjlab-RollerCrouch-Flat-MicroDuck", (-0.2, 0.2)), ("Mjlab-Velocity-Flat-MicroDuck-Rollers", (-0.2, 0.2))])
def test_push_ranges_preserved(task_id, expected):
    import src.task_registry  # noqa: F401

    vel = _env_cfg(task_id).events["push_robot"].params["velocity_range"]
    assert vel["x"] == expected
    assert vel["y"] == expected


def test_roulade_has_no_pushes():
    import src.task_registry  # noqa: F401

    assert "push_robot" not in _env_cfg("Mjlab-Roulade-Flat-MicroDuck").events


def test_wheel_friction_regex_does_not_match_backlash_joints():
    assert task_dr.WHEEL_JOINTS == (r"^passive_.*wheel",)


def test_every_roller_task_shares_one_roller_dr():
    from src import task_roller_crouch, task_spin, task_velocity_rollers

    assert task_spin.DR is task_dr.ROLLER_DR
    assert task_roller_crouch.DR is task_dr.ROLLER_DR
    assert task_velocity_rollers.DR is task_dr.ROLLER_DR


def test_defaults_match_the_velocity_recipe():
    d = task_dr.DEFAULT_DR
    assert (d.com, d.head_com, d.mass_inertia, d.joint_friction, d.armature) == (True, True, True, True, True)
    assert (d.kp, d.kd, d.joint_damping, d.base_orientation, d.wheel_friction) == (False, False, False, False, False)
    assert d.com_range == 0.003
    assert d.head_com_range == 0.003
    assert d.mass_inertia_range == (0.95, 1.05)
    assert d.joint_friction_range == (0.9, 1.1)
    assert d.armature_range == (0.9, 1.1)
    assert d.encoder_bias_range == (-0.015, 0.015)
    assert d.imu_orientation_angle_deg == 6.0
    assert d.push_interval_s == (3.0, 6.0)
    assert d.push_range == (-0.3, 0.3)


def test_roller_dr_differs_from_default_in_exactly_three_fields():
    changed = {f.name for f in dataclasses.fields(task_dr.MicroduckDrCfg) if getattr(task_dr.DEFAULT_DR, f.name) != getattr(task_dr.ROLLER_DR, f.name)}
    assert changed == {"armature_joints", "wheel_friction", "push_range"}


def test_dr_cfg_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        task_dr.DEFAULT_DR.com = False


BAM_MODEL_FIELDS = ("dof_frictionloss", "dof_damping")

ALL_MICRODUCK_TASKS = ("Mjlab-Velocity-Flat-MicroDuck", "Mjlab-Velocity-Rough-MicroDuck", "Mjlab-VelStand-Flat-MicroDuck", "Mjlab-StandUp-Flat-MicroDuck", "Mjlab-SitStand-Flat-MicroDuck", "Mjlab-GroundPick-Flat-MicroDuck", "Mjlab-BallKick-Flat-MicroDuck", "Mjlab-Roulade-Flat-MicroDuck", "Mjlab-Velocity-Flat-MicroDuck-Rollers", "Mjlab-Velocity-Swizzle-MicroDuck", "Mjlab-Spin-Flat-MicroDuck", "Mjlab-RollerCrouch-Flat-MicroDuck", "Mjlab-RollerSlope-Flat-MicroDuck", "Mjlab-RollerStandUp-Flat-MicroDuck")


def _expanded_model_fields(task_id):
    import src.task_registry  # noqa: F401

    fields = set()
    for term in _env_cfg(task_id).events.values():
        fields |= set(getattr(term.func, "model_fields", ()) or ())
    return fields


@pytest.mark.parametrize("task_id", ALL_MICRODUCK_TASKS)
def test_every_bam_env_expands_the_fields_bam_writes(task_id):
    missing = set(BAM_MODEL_FIELDS) - _expanded_model_fields(task_id)
    assert not missing, f"{task_id} does not expand {sorted(missing)}; BAM writes it per-env every step"


@pytest.mark.parametrize("task_id", ("Mjlab-Spin-Flat-MicroDuck", "Mjlab-RollerCrouch-Flat-MicroDuck", "Mjlab-Velocity-Flat-MicroDuck-Rollers"))
def test_wheel_friction_selects_only_wheels(task_id):
    import src.task_registry  # noqa: F401

    term = _env_cfg(task_id).events["randomize_wheel_friction"]
    assert term.params["asset_cfg"].joint_names == task_dr.WHEEL_JOINTS


def test_the_wheel_regex_excludes_backlash_joints_on_the_backlash_model():
    import re

    import mujoco

    from src.robot import MJCF_DIR

    model = mujoco.MjModel.from_xml_path(str(MJCF_DIR / "robot_groundcontact_rollers_backlash.xml"))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    passive = [n for n in names if n and n.startswith("passive_")]

    pattern = re.compile(task_dr.WHEEL_JOINTS[0])
    matched = [n for n in passive if pattern.match(n)]

    assert len(passive) == 18
    assert sorted(matched) == ["passive_LF_wheel", "passive_LR_wheel", "passive_RF_wheel", "passive_RR_wheel"]
    assert not [n for n in matched if "backlash" in n]
