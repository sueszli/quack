import math
import os

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as base_mdp
from mjlab.managers import CurriculumTermCfg, EventTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from . import task_mdp as microduck_mdp
from .task_slope_terrain import RAMP_DEG_MAX, FlatRampTerrainCfg
from .task_symmetry import PpoWithSymmetryCfg
from .task_velocity import LOCAL_CHECKPOINTS_ONLY
from .task_velocity_rollers import make_microduck_velocity_rollers_env_cfg

FLAT_LENGTH = 2.0
RAMP_LENGTH_RANGE = (3.0, 8.0)  # horizontal, drawn per tile
RUNOUT_LENGTH = 4.0
SPAWN_ON_RAMP = 0.3  # gravity -> rolling, no skidding
ENTRY_VELOCITY_X = (0.25, 0.45)
TILE_SIZE = (15.0, 4.0)  # >= flat + ramp_max + runout (= 14) + margin
SPAWN_YAW = (0.0, 0.0)  # facing the descent (+x)

# None = random steepness, as in training; 0..1 forces a slope (1.0 = steepest ~20°).
PLAY_DIFFICULTY = None


def _resolve_play_difficulty():
    raw = os.environ.get("SLOPE_PLAY_DIFFICULTY")
    if raw is None:
        return PLAY_DIFFICULTY
    raw = raw.strip().lower()
    if raw in ("", "none", "random"):
        return None
    try:
        return max(0.0, min(1.0, float(raw)))
    except ValueError:
        print(f"[roller_slope] SLOPE_PLAY_DIFFICULTY='{raw}' invalid -> default {PLAY_DIFFICULTY}")
        return PLAY_DIFFICULTY


# Below the lowest possible exit flat (steepest, longest ramp) plus margin, so this
# never fires during a normal descent.
_MAX_DROP = RAMP_LENGTH_RANGE[1] * math.tan(math.radians(RAMP_DEG_MAX))
VOID_FLOOR = -_MAX_DROP - 0.5


def make_microduck_roller_slope_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_velocity_rollers_env_cfg(play=play)

    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=TILE_SIZE,
            curriculum=True,
            num_rows=10,  # steepness levels
            num_cols=1,
            difficulty_range=(0.0, 1.0),
            sub_terrains={"flat_ramp": FlatRampTerrainCfg(flat_length=FLAT_LENGTH, ramp_length_range=RAMP_LENGTH_RANGE, runout_length=RUNOUT_LENGTH, spawn_on_ramp=SPAWN_ON_RAMP)},
        ),
        max_init_terrain_level=0,  # curriculum: start on the gentlest ramp
    )

    if play:
        play_difficulty = _resolve_play_difficulty()
        if play_difficulty is not None:
            cfg.scene.terrain.terrain_generator.difficulty_range = (play_difficulty, play_difficulty)
        else:
            cfg.scene.terrain.max_init_terrain_level = None

    command = cfg.commands["twist"]
    command.rel_standing_envs = 1.0
    command.rel_heading_envs = 0.0
    command.ranges.lin_vel_x = (0.0, 0.0)
    command.ranges.lin_vel_y = (0.0, 0.0)
    if getattr(command.ranges, "ang_vel_z", None) is not None:
        command.ranges.ang_vel_z = (0.0, 0.0)

    cfg.events["reset_base"].params["pose_range"]["yaw"] = SPAWN_YAW
    # No base push: a moving base with stationary wheels skids on step 1 -> contact
    # spike -> NaN divergence. reset_rolling_entry below spins base and wheels
    # consistently (ω·r = v) instead.
    cfg.events["reset_base"].params["velocity_range"] = {}

    # No fixed-pose reward: dictating the flat-ground posture stopped it flexing and
    # leaning to place its CoM over the slope.
    keep = {"action_rate_l2"}
    for name in list(cfg.rewards.keys()):
        if name not in keep:
            del cfg.rewards[name]

    cfg.rewards["upright"] = RewardTermCfg(func=microduck_mdp.body_upright_gaussian, weight=3.0, params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)), "std": 0.2})
    cfg.rewards["alive"] = RewardTermCfg(func=microduck_mdp.is_alive, weight=1.0)
    # Capped so there is no incentive to push faster, and wheel-based so pushing the
    # base without rolling pays nothing. Without it, standing still is optimal.
    cfg.rewards["wheel_glide"] = RewardTermCfg(func=microduck_mdp.wheel_glide_reward, weight=2.0, params={"cap_speed": 0.35})
    cfg.rewards["heading_hold"] = RewardTermCfg(func=microduck_mdp.heading_hold_reward, weight=1.5, params={"std": 0.4})
    cfg.rewards["feet_flat"] = RewardTermCfg(func=microduck_mdp.feet_flat_penalty, weight=-2.0, params={"asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot")), "sensor_name": "feet_ground_contact"})
    cfg.rewards["neck_action_rate_l2"] = RewardTermCfg(func=microduck_mdp.neck_action_rate_l2, weight=-0.5)
    # Constrains head/neck only: dropping the fixed leg pose left nothing holding the
    # head, and it wandered.
    cfg.rewards["neck_joint_pos_l2"] = RewardTermCfg(func=microduck_mdp.neck_joint_pos_l2, weight=-0.75)
    cfg.rewards["joint_torques_l2"] = RewardTermCfg(func=microduck_mdp.joint_torques_l2, weight=-1e-3)
    cfg.rewards["action_rate_l2"].weight = -1.0

    # The exit flat is solid ground, so edge-termination is unnecessary; it cut long
    # ramps short.
    cfg.terminations["fell_over"] = TerminationTermCfg(func=base_mdp.bad_orientation, params={"limit_angle": 1.0, "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))})
    if "out_of_terrain_bounds" in cfg.terminations:
        del cfg.terminations["out_of_terrain_bounds"]
    cfg.terminations["fell_into_void"] = TerminationTermCfg(func=microduck_mdp.root_height_below, params={"min_height": VOID_FLOOR, "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))})
    cfg.terminations["nan_state"] = TerminationTermCfg(func=microduck_mdp.robot_state_is_nan, time_out=False)

    # A rare contact (~1/25M env-steps) diverges the free joint to NaN. nan_state only
    # catches it the NEXT step, so the NaN would reach this step's obs and rsl_rl's
    # check_nan would kill training.
    for grp in ("actor", "critic"):
        cfg.observations[grp].nan_policy = "sanitize"

    cfg.events["reset_action_history"] = EventTermCfg(func=microduck_mdp.reset_action_history, mode="reset")
    # Must stay ordered after reset_base.
    cfg.events["reset_rolling_entry"] = EventTermCfg(func=microduck_mdp.reset_rolling_entry, mode="reset", params={"speed_range": ENTRY_VELOCITY_X})

    # Promotes on distance descended. Thrown straight onto 20° it just nosedives.
    for name in list(cfg.curriculum.keys()):
        del cfg.curriculum[name]
    cfg.curriculum["terrain_levels"] = CurriculumTermCfg(func=microduck_mdp.terrain_levels_slope)

    return cfg


MicroduckRollerSlopeRlCfg = RslRlOnPolicyRunnerCfg(actor=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True, distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}), critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True), algorithm=PpoWithSymmetryCfg(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.01, num_learning_epochs=5, num_mini_batches=4, learning_rate=1.0e-3, schedule="adaptive", gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0, symmetry_cfg=None), logger=LOCAL_CHECKPOINTS_ONLY, experiment_name="roller_slope", run_name="roller_slope", save_interval=250, num_steps_per_env=24, max_iterations=8_000)
