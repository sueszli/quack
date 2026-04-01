#!/usr/bin/env python3
"""Skating trajectory test — robot hangs in air, plays a keyframed stroke cycle.

Edit KEYFRAMES and DURATIONS at the bottom to tune the motion.
Run with:  uv run python scripts/test_skating_trajectory.py
"""

import math
import os
import time

import mujoco
import mujoco.viewer
import numpy as np

SCENE_XML = os.path.join(
    os.path.dirname(__file__),
    "../src/mjlab_microduck/robot/microduck/scene_roller.xml",
)

# ── Ctrl indices (14 actuated joints, passive wheels excluded) ───────────────
#  0  left_hip_yaw      1  left_hip_roll     2  left_hip_pitch
#  3  left_knee         4  left_ankle
#  5  neck_pitch        6  head_pitch        7  head_yaw    8  head_roll
#  9  right_hip_yaw    10  right_hip_roll   11  right_hip_pitch
# 12  right_knee       13  right_ankle
L_YAW, L_ROLL, L_PITCH, L_KNEE, L_ANKLE = 0, 1, 2, 3, 4
NECK, H_PITCH, H_YAW, H_ROLL             = 5, 6, 7, 8
R_YAW, R_ROLL, R_PITCH, R_KNEE, R_ANKLE = 9, 10, 11, 12, 13

# ── Default (home) pose ──────────────────────────────────────────────────────
HOME = np.array([
     0.0,  # L_YAW
     0.0,  # L_ROLL
     0.6,  # L_PITCH
    -1.2,  # L_KNEE
     0.6,  # L_ANKLE
    -0.5,  # NECK
     0.5,  # H_PITCH
     0.0,  # H_YAW
     0.0,  # H_ROLL
     0.0,  # R_YAW
     0.0,  # R_ROLL
    -0.6,  # R_PITCH
     1.2,  # R_KNEE
    -0.6,  # R_ANKLE
], dtype=float)

_IDX = dict(
    L_YAW=L_YAW, L_ROLL=L_ROLL, L_PITCH=L_PITCH, L_KNEE=L_KNEE, L_ANKLE=L_ANKLE,
    NECK=NECK, H_PITCH=H_PITCH, H_YAW=H_YAW, H_ROLL=H_ROLL,
    R_YAW=R_YAW, R_ROLL=R_ROLL, R_PITCH=R_PITCH, R_KNEE=R_KNEE, R_ANKLE=R_ANKLE,
)

def pose(**joints) -> np.ndarray:
    """Build a ctrl array from HOME, overriding named joints."""
    ctrl = HOME.copy()
    for k, v in joints.items():
        ctrl[_IDX[k]] = v
    return ctrl


# ── Keyframes ────────────────────────────────────────────────────────────────
# Each entry is (ctrl_array, duration_to_next_keyframe_in_seconds).
# The sequence loops: last keyframe transitions back to first.

KEYFRAMES = [

    # 0 — skating stance: feet closer together
    (pose(
        L_ROLL=-0.3,
        R_ROLL= 0.3,
    ), 0.4),

    # 1 — right push: extend right leg backward+outward, shift weight left
    (pose(
        L_ROLL=-0.4,   # lean onto left leg
        L_PITCH= 0.7,
        L_KNEE= -1.3,
        R_YAW=   0.25, # right leg sweeps backward
        R_ROLL=  0.5,  # push outward
        R_PITCH=-0.3,  # leg becomes more upright (less pitched)
        R_KNEE=  0.5,  # knee extends
        R_ANKLE=-0.3,  # ankle extends
    ), 0.7),

    # 2 — right leg lift and swing forward
    (pose(
        L_ROLL=-0.3,
        R_YAW=  -0.15, # swing forward
        R_ROLL=  0.3,  # back to neutral width
        R_PITCH=-1.0,  # swing leg forward
        R_KNEE=  1.7,  # bend knee to lift foot
    ), 0.5),

    # 3 — right leg returns to stance
    (pose(
        L_ROLL=-0.3,
        R_ROLL= 0.3,
    ), 0.3),

    # 4 — left push: mirror of right push
    (pose(
        R_ROLL= 0.4,   # lean onto right leg
        R_PITCH=-0.7,
        R_KNEE=  1.3,
        L_YAW=  -0.25, # left leg sweeps backward
        L_ROLL= -0.5,  # push outward
        L_PITCH= 0.3,  # leg becomes more upright
        L_KNEE= -0.5,  # knee extends
        L_ANKLE= 0.3,  # ankle extends
    ), 0.7),

    # 5 — left leg lift and swing forward
    (pose(
        R_ROLL= 0.3,
        L_YAW=  0.15,  # swing forward
        L_ROLL= -0.3,
        L_PITCH= 1.0,  # swing leg forward
        L_KNEE= -1.7,  # bend knee to lift foot
    ), 0.5),

    # 6 — left leg returns to stance (loops back to 0)
    (pose(
        L_ROLL=-0.3,
        R_ROLL= 0.3,
    ), 0.3),
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def ease(t: float) -> float:
    """Smooth cosine ease-in-out: t ∈ [0,1] → [0,1]."""
    return 0.5 * (1.0 - math.cos(math.pi * t))


def main():
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data  = mujoco.MjData(model)

    # Disable gravity — robot hangs in air
    # model.opt.gravity[:] = 0.0

    mujoco.mj_resetData(model, data)

    # Root position: float upright at a nice viewing height
    ROOT_POS  = np.array([0.0, 0.0, 0.5])
    ROOT_QUAT = np.array([1.0, 0.0, 0.0, 0.0])  # identity = upright

    # Set initial ctrl to first keyframe
    data.ctrl[:] = KEYFRAMES[0][0]

    ctrls     = [kf for kf, _ in KEYFRAMES]
    durations = [dur for _, dur in KEYFRAMES]
    total_dur = sum(durations)
    n         = len(KEYFRAMES)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance  = 0.9
        viewer.cam.elevation = -20
        viewer.cam.lookat[:] = ROOT_POS

        t_start = time.time()

        while viewer.is_running():
            # Elapsed time, looping
            wall = time.time() - t_start
            elapsed = wall % total_dur

            # Find which segment we're in
            seg_start = 0.0
            for i, dur in enumerate(durations):
                if elapsed < seg_start + dur:
                    alpha = (elapsed - seg_start) / dur
                    ctrl = ctrls[i] + (ctrls[(i + 1) % n] - ctrls[i]) * ease(alpha)
                    break
                seg_start += dur
            else:
                ctrl = ctrls[-1]

            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)
            viewer.sync()

            # Real-time pacing
            next_step = t_start + (wall // model.opt.timestep + 1) * model.opt.timestep
            sleep = next_step - time.time()
            if sleep > 0:
                time.sleep(sleep)


if __name__ == "__main__":
    main()
