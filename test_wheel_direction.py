"""
Test wheel rotation direction.
Hangs the robot in the air (gravity off) and spins all 4 wheels at +omega.
Watch which direction the wheels roll to verify sign convention.

Left wheels at +omega should roll FORWARD.
Right wheels at -omega should roll FORWARD.
If they go backward, flip the sign in wheel_speed_reward.
"""

import mujoco
import mujoco.viewer
import numpy as np
from pathlib import Path

XML = Path(__file__).parent / "src/mjlab_microduck/robot/microduck/robot_walk_rollers.xml"

model = mujoco.MjModel.from_xml_path(str(XML))
data = mujoco.MjData(model)

# Disable gravity so the robot floats
model.opt.gravity[:] = 0.0

# Find free joint qpos address (7 values: x y z qw qx qy qz)
free_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
free_qpos_adr = model.jnt_qposadr[free_jnt_id]

# Find wheel joint qvel addresses
WHEEL_NAMES = ["passive_LFwheel", "passive_LRwheel", "passive_RFwheel", "passive_RRwheel"]
WHEEL_SIGNS = [+1, +1, +1, +1]  # all wheels +omega = forward (verified)

wheel_dofadrs = []
for name in WHEEL_NAMES:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    wheel_dofadrs.append(model.jnt_dofadr[jid])

OMEGA = 20.0  # rad/s

# Set robot floating at a comfortable viewing height
data.qpos[free_qpos_adr + 0] = 0.0   # x
data.qpos[free_qpos_adr + 1] = 0.0   # y
data.qpos[free_qpos_adr + 2] = 0.3   # z (float in air)
data.qpos[free_qpos_adr + 3] = 1.0   # qw (identity quaternion)
data.qpos[free_qpos_adr + 4] = 0.0   # qx
data.qpos[free_qpos_adr + 5] = 0.0   # qy
data.qpos[free_qpos_adr + 6] = 0.0   # qz

mujoco.mj_forward(model, data)

print("Spinning wheels:")
for name, sign, adr in zip(WHEEL_NAMES, WHEEL_SIGNS, wheel_dofadrs):
    print(f"  {name}: omega = {sign * OMEGA:+.1f} rad/s")
print()
print("If wheels roll FORWARD → sign convention is correct.")
print("If wheels roll BACKWARD → set left_outward_negative=False in skating_outward_push")
print("  and flip WHEEL_SIGNS in wheel_speed_reward.")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # Hold root stationary, keep wheels spinning
        data.qpos[free_qpos_adr:free_qpos_adr + 7] = [0, 0, 0.3, 1, 0, 0, 0]
        data.qvel[model.jnt_dofadr[free_jnt_id]:model.jnt_dofadr[free_jnt_id] + 6] = 0.0

        for adr, sign in zip(wheel_dofadrs, WHEEL_SIGNS):
            data.qvel[adr] = sign * OMEGA

        mujoco.mj_step(model, data)
        viewer.sync()
