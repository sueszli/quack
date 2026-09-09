"""infer.py must rehearse on the SAME BAM M6 actuator training uses in warp.

The script's BAM constants are hardcoded rather than imported from robot.py, to
keep it torch/warp-free — so these tests pin them to ``_BAM_ACTUATOR_KWARGS``,
and pin the motor conversion to what ``bam.mjlab.BamActuator.edit_spec`` does.
"""

import mujoco
import numpy as np
import pytest


@pytest.fixture(scope="module")
def ip():
    import src.infer as mod

    return mod


def test_cpu_bam_constants_mirror_training_cfg(ip):
    from bam.mjlab import BamActuator

    from src.robot import _BAM_ACTUATOR_KWARGS as k

    assert ip.BAM_MOTOR_NAME == k["motor_name"]
    assert ip.BAM_MODEL == k["model"]
    assert ip.BAM_KP_FW == k["kp_fw"]
    assert ip.BAM_VIN_RANGE == k["vin_range"]
    assert ip.BAM_VIN_DROP_GAIN_RANGE == k["vin_drop_gain_range"]
    assert ip.BAM_VIN_MIN == k["vin_min"]
    assert ip.BAM_MAX_CURRENT == k.get("max_current")
    assert ip.BAM_STIFF_SOLREF_FRICTION == BamActuator._STIFF_SOLREF_FRICTION
    assert ip.BAM_STIFF_SOLIMP_FRICTION == BamActuator._STIFF_SOLIMP_FRICTION


@pytest.fixture(scope="module")
def bam_sim(ip):
    bam_model = ip.load_bam_model(ip.BAM_KP_FW, 7.4, ip.BAM_MAX_CURRENT)
    model, data, ctrl, names = ip.load_mujoco_with_bam(ip.MICRODUCK_XML, bam_model, 0.005, 0.1, ip.BAM_VIN_MIN)
    return ip, bam_model, model, data, ctrl, names


def test_actuators_converted_like_warp(bam_sim):
    ip, bam_model, model, data, ctrl, names = bam_sim
    kt, R = bam_model.kt.value, bam_model.R.value
    assert len(names) == 14 and model.nu == 14
    assert not any(n.startswith("passive_") for n in names)
    # Torque motors: ctrl is the BAM torque, no MuJoCo PD left over. set_to_motor
    # leaves the old PD biasprm bytes behind, inert under BIAS_NONE, as in warp.
    assert (model.actuator_gaintype == mujoco.mjtGain.mjGAIN_FIXED).all()
    assert (model.actuator_biastype == mujoco.mjtBias.mjBIAS_NONE).all()
    assert np.allclose(model.actuator_gainprm[:, 0], 1.0)
    assert (model.actuator_forcelimited == 1).all()
    assert np.allclose(model.actuator_forcerange[:, 1], 7.4 * kt / R)
    dofs = model.jnt_dofadr[model.actuator_trnid[:, 0]]
    assert np.allclose(model.dof_armature[dofs], bam_model.actuator.get_extra_inertia())
    assert np.allclose(model.dof_solref[dofs], ip.BAM_STIFF_SOLREF_FRICTION)
    assert np.allclose(model.dof_solimp[dofs], ip.BAM_STIFF_SOLIMP_FRICTION)
    assert bam_model.actuator.kp == ip.BAM_KP_FW
    assert bam_model.actuator.max_current is None  # training has no current limiter


def test_bam_step_loop_runs_with_live_friction(bam_sim):
    ip, bam_model, model, data, ctrl, names = bam_sim
    fj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    qa = model.jnt_qposadr[fj]
    mujoco.mj_resetData(model, data)
    data.qpos[qa + 2] = 0.125
    data.qpos[qa + 3 : qa + 7] = [1, 0, 0, 0]
    jq = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    data.qpos[jq] = ip.DEFAULT_POSE
    ctrl.reset(data.qpos)
    ctrl.q_target[:] = ip.DEFAULT_POSE
    mujoco.mj_forward(model, data)
    dofs = model.jnt_dofadr[model.actuator_trnid[:, 0]]
    for _ in range(100):
        ctrl.update()
        mujoco.mj_step(model, data)
    assert not np.isnan(data.qpos).any()
    limit = model.actuator_forcerange[0, 1]
    assert (np.abs(data.ctrl) <= limit + 1e-9).all()  # ctrl IS the motor torque
    assert (model.dof_frictionloss[dofs] > 0).all()  # BAM budget written every step
    assert np.allclose(model.dof_damping[dofs], bam_model.friction_viscous.value)
