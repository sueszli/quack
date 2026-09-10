from pathlib import Path

import mujoco
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from .robot_actuator import BacklashEncoderBamActuatorCfg, FrictionDRBamActuatorCfg

MJCF_DIR: Path = Path(__file__).resolve().parents[1] / "assets" / "mjcf"
_ROBOT_DIR: Path = MJCF_DIR

MICRODUCK_WALK_XML: Path = _ROBOT_DIR / "robot_walk.xml"
# Curated floor-contact collision set (soles, legs, trunk/head shells, jaw, battery, hips), not every geom.
MICRODUCK_GROUNDCONTACT_XML: Path = _ROBOT_DIR / "robot_groundcontact.xml"
MICRODUCK_BALL_XML: Path = _ROBOT_DIR / "ball.xml"
MICRODUCK_GROUNDCONTACT_ROLLERS_XML: Path = _ROBOT_DIR / "robot_groundcontact_rollers.xml"
# Backlash models add an unactuated passive_<joint>_backlash hinge in series per servo (±1° play).
MICRODUCK_GROUNDCONTACT_BACKLASH_XML: Path = _ROBOT_DIR / "robot_groundcontact_backlash.xml"
MICRODUCK_WALK_BACKLASH_XML: Path = _ROBOT_DIR / "robot_walk_backlash.xml"
MICRODUCK_GROUNDCONTACT_ROLLERS_BACKLASH_XML: Path = _ROBOT_DIR / "robot_groundcontact_rollers_backlash.xml"

assert MICRODUCK_WALK_XML.exists(), f"XML not found: {MICRODUCK_WALK_XML}"
assert MICRODUCK_GROUNDCONTACT_XML.exists(), f"XML not found: {MICRODUCK_GROUNDCONTACT_XML}"
assert MICRODUCK_BALL_XML.exists(), f"XML not found: {MICRODUCK_BALL_XML}"
assert MICRODUCK_GROUNDCONTACT_ROLLERS_XML.exists(), f"XML not found: {MICRODUCK_GROUNDCONTACT_ROLLERS_XML}"
assert MICRODUCK_GROUNDCONTACT_BACKLASH_XML.exists(), f"XML not found: {MICRODUCK_GROUNDCONTACT_BACKLASH_XML}"
assert MICRODUCK_WALK_BACKLASH_XML.exists(), f"XML not found: {MICRODUCK_WALK_BACKLASH_XML}"
assert MICRODUCK_GROUNDCONTACT_ROLLERS_BACKLASH_XML.exists(), f"XML not found: {MICRODUCK_GROUNDCONTACT_ROLLERS_BACKLASH_XML}"


def get_walk_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_WALK_XML))


def get_standup_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_GROUNDCONTACT_XML))


def get_ground_pick_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_GROUNDCONTACT_XML))


def get_walk_rollers_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_GROUNDCONTACT_ROLLERS_XML))


def get_ball_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_BALL_XML))


def get_backlash_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_GROUNDCONTACT_BACKLASH_XML))


def get_walk_backlash_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_WALK_BACKLASH_XML))


def get_rollers_backlash_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_GROUNDCONTACT_ROLLERS_BACKLASH_XML))


HOME_FRAME = EntityCfg.InitialStateCfg(
    joint_pos={
        # STAND2 pose: trunk ~5mm forward of the old HOME so the CoM sits over the ankle axis.
        # Must match the STAND keyframe in scene.xml / scene_walk.xml.
        r".*hip_yaw.*": 0.0,
        r".*left_hip_roll.*": -0.0873,
        r".*right_hip_roll.*": 0.0873,
        r".*left_hip_pitch.*": -0.4579,
        r".*right_hip_pitch.*": 0.4579,
        r".*left_knee.*": -0.0049,
        r".*right_knee.*": 0.0049,
        r".*left_ankle.*": 0.4530,
        r".*right_ankle.*": -0.4530,
        r".*neck_pitch.*": 0.3491,
        r".*head_pitch.*": 0.3491,
        r".*head_yaw.*": 0.0,
        r".*head_roll.*": 0.0,
    },
    joint_vel={".*": 0.0},
)

FULL_COLLISION = CollisionCfg(geom_names_expr=[".*_collision"], condim={r"^(left|right)_foot_collision$": 3, ".*_collision": 1}, priority={r"^(left|right)_foot_collision$": 1}, friction={r"^(left|right)_foot_collision$": (1.0,)})

_BAM_ACTUATOR_KWARGS = {
    "motor_name": "xl330",
    "model": "m6",
    "target_names_expr": (r"^(?!passive_).*",),
    "kp_fw": 200.0,  # microduck's preserved firmware stiffness (microban uses 125)
    "vin_range": (6.5, 8.2),
    "vin_drop_gain_range": (0.0, 0.2),
    "vin_min": 6.0,
    "delay_min_lag": 3,
    "delay_max_lag": 6,
}
actuators = FrictionDRBamActuatorCfg(**_BAM_ACTUATOR_KWARGS)

backlash_actuators = BacklashEncoderBamActuatorCfg(**_BAM_ACTUATOR_KWARGS)

# The backlash rule must stay FIRST: matching is first-match-wins, and HOME_FRAME's unanchored
# patterns would otherwise initialize passive_*_backlash joints outside their ±1° range.
BACKLASH_HOME_FRAME = EntityCfg.InitialStateCfg(joint_pos={r".*_backlash$": 0.0, **HOME_FRAME.joint_pos}, joint_vel={".*": 0.0})

MICRODUCK_WALK_ROBOT_CFG = EntityCfg(spec_fn=get_walk_spec, init_state=HOME_FRAME, collisions=(FULL_COLLISION,), articulation=EntityArticulationInfoCfg(actuators=(actuators,), soft_joint_pos_limit_factor=0.9))

MICRODUCK_STANDUP_ROBOT_CFG = EntityCfg(spec_fn=get_standup_spec, init_state=HOME_FRAME, collisions=(FULL_COLLISION,), articulation=EntityArticulationInfoCfg(actuators=(actuators,), soft_joint_pos_limit_factor=0.9))

MICRODUCK_GROUND_PICK_ROBOT_CFG = EntityCfg(spec_fn=get_ground_pick_spec, init_state=HOME_FRAME, collisions=(FULL_COLLISION,), articulation=EntityArticulationInfoCfg(actuators=(actuators,), soft_joint_pos_limit_factor=0.9))

# Each backlash robot must mirror its base task's model, or backlash A/B comparisons are confounded.
MICRODUCK_BACKLASH_ROBOT_CFG = EntityCfg(spec_fn=get_backlash_spec, init_state=BACKLASH_HOME_FRAME, collisions=(FULL_COLLISION,), articulation=EntityArticulationInfoCfg(actuators=(backlash_actuators,), soft_joint_pos_limit_factor=0.9))

MICRODUCK_WALK_BACKLASH_ROBOT_CFG = EntityCfg(spec_fn=get_walk_backlash_spec, init_state=BACKLASH_HOME_FRAME, collisions=(FULL_COLLISION,), articulation=EntityArticulationInfoCfg(actuators=(backlash_actuators,), soft_joint_pos_limit_factor=0.9))

MICRODUCK_ROLLERS_BACKLASH_ROBOT_CFG = EntityCfg(spec_fn=get_rollers_backlash_spec, init_state=BACKLASH_HOME_FRAME, collisions=(), articulation=EntityArticulationInfoCfg(actuators=(backlash_actuators,), soft_joint_pos_limit_factor=0.9))

# Position is set each episode by reset_ball_in_front_of_foot; this init pos is pre-first-reset only.
MICRODUCK_BALL_CFG = EntityCfg(spec_fn=get_ball_spec, init_state=EntityCfg.InitialStateCfg(pos=(0.3, 0.0, 0.035)))

MICRODUCK_WALK_ROLLERS_ROBOT_CFG = EntityCfg(
    spec_fn=get_walk_rollers_spec,
    init_state=HOME_FRAME,
    collisions=(),  # roller wheel collision geoms have no explicit names; XML defaults apply
    articulation=EntityArticulationInfoCfg(actuators=(actuators,), soft_joint_pos_limit_factor=0.9),
)

if __name__ == "__main__":
    from mjlab.scene import Scene, SceneCfg
    from mjlab.terrains import TerrainImporterCfg
    from mujoco import viewer

    SCENE_CFG = SceneCfg(terrain=TerrainImporterCfg(terrain_type="plane"), entities={"robot": MICRODUCK_WALK_ROBOT_CFG})

    scene = Scene(SCENE_CFG, device="cuda:0")
    viewer.launch(scene.compile())
