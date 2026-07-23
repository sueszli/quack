"""Microduck roller slope — descente passive équilibrée.

Le robot spawne sur du plat (impulsion vers l'avant), roule sur une rampe
descendante et se laisse glisser en restant debout. Aucun pilotage : la
commande twist est neutralisée (rel_standing_envs=1.0). Terrain custom
plat+rampe (FlatRampTerrainCfg), curriculum de raideur (terrain_levels_slope).
Obs 61D unifié → interchangeable au runtime (--new-cmd-obs) — hérité tel quel
de make_microduck_velocity_rollers_env_cfg (DR/obs/reset non touchés ici).
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as base_mdp
from mjlab.managers import CurriculumTermCfg, EventTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlModelCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.slope_terrain import FlatRampTerrainCfg, RAMP_DEG_MAX
from mjlab_microduck.tasks.microduck_velocity_rollers_env_cfg import (
    make_microduck_velocity_rollers_env_cfg,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg

# Géométrie du terrain plat+rampe+sortie.
FLAT_LENGTH        = 2.0
RAMP_LENGTH_RANGE  = (3.0, 8.0)   # longueur horizontale de la rampe, tirée au hasard par tuile
RUNOUT_LENGTH      = 4.0          # plat de sortie en bas
SPAWN_ON_RAMP      = 0.3          # spawn ce nb de m sur la rampe (gravité -> roulement, pas de patinage)
TILE_SIZE          = (15.0, 4.0)  # >= flat + ramp_max + runout (= 14) + marge
SPAWN_YAW          = (0.0, 0.0)   # face à la descente (+x), fixe

# Au PLAY (uv run play), difficulté imposée à toutes les tuiles pour choisir la
# pente affichée (1.0 = la plus raide ~20°, 0.5 = moyenne, 0.0 = la plus douce).
# À l'entraînement (play=False), le curriculum gère la difficulté normalement.
PLAY_DIFFICULTY    = 1.0

# Terminaison « tombé dans le vide » : sous le plat de sortie le plus bas
# (rampe la plus raide et la plus longue), avec marge => ne se déclenche jamais
# pendant une descente normale, seulement si le robot quitte le solide.
_MAX_DROP  = RAMP_LENGTH_RANGE[1] * math.tan(math.radians(RAMP_DEG_MAX))
VOID_FLOOR = -_MAX_DROP - 0.5


def make_microduck_roller_slope_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_velocity_rollers_env_cfg(play=play)

    # === TERRAIN : plat + rampe (longueur aléatoire) + plat de sortie ===
    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=TILE_SIZE,
            curriculum=True,
            num_rows=10,          # 10 niveaux de raideur
            num_cols=1,
            difficulty_range=(0.0, 1.0),
            sub_terrains={
                "flat_ramp": FlatRampTerrainCfg(
                    flat_length=FLAT_LENGTH,
                    ramp_length_range=RAMP_LENGTH_RANGE,
                    runout_length=RUNOUT_LENGTH,
                    spawn_on_ramp=SPAWN_ON_RAMP,
                )
            },
        ),
        max_init_terrain_level=0,  # démarrer sur la rampe la plus douce
    )

    # Au play : toutes les tuiles à la même difficulté (PLAY_DIFFICULTY) pour
    # visualiser la pente voulue (par défaut la plus raide) — sinon tout le monde
    # spawn sur la plus douce (max_init_terrain_level=0).
    if play:
        cfg.scene.terrain.terrain_generator.difficulty_range = (PLAY_DIFFICULTY, PLAY_DIFFICULTY)

    # === COMMANDE neutralisée (équilibre pur) ===
    command = cfg.commands["twist"]
    command.rel_standing_envs = 1.0
    command.rel_heading_envs = 0.0
    command.ranges.lin_vel_x = (0.0, 0.0)
    command.ranges.lin_vel_y = (0.0, 0.0)
    if getattr(command.ranges, "ang_vel_z", None) is not None:
        command.ranges.ang_vel_z = (0.0, 0.0)

    # === RESET : toujours face à la descente (+x), PAS de poussée de base ===
    # Le yaw hérité est aléatoire (-180°/+180°) -> on le fixe à 0 (face au bas de
    # la pente). Aucune vitesse de base injectée : le robot spawne sur la rampe
    # (voir spawn_on_ramp), la gravité fait rouler les roues (élan aux roues,
    # sans glissement). L'ancienne poussée de base (base rapide, roues immobiles)
    # patinait -> pic de contact -> divergence NaN, et le robot "marchait pour
    # s'arrêter" au lieu de rouler.
    cfg.events["reset_base"].params["pose_range"]["yaw"] = SPAWN_YAW
    cfg.events["reset_base"].params["velocity_range"] = {}

    # === RÉCOMPENSES : équilibre + posture debout nominale ===
    keep = {"action_rate_l2"}
    for name in list(cfg.rewards.keys()):
        if name not in keep:
            del cfg.rewards[name]

    cfg.rewards["upright"] = RewardTermCfg(
        func=microduck_mdp.body_upright_gaussian,
        weight=3.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)), "std": 0.2},
    )
    cfg.rewards["alive"] = RewardTermCfg(func=microduck_mdp.is_alive, weight=1.0)
    # posture debout nominale (cible fixe = default_joint_pos, aucun override)
    cfg.rewards["standing_pose"] = RewardTermCfg(
        func=microduck_mdp.pose_target_match, weight=3.0, params={"std": 0.4},
    )
    cfg.rewards["standing_pose_l1"] = RewardTermCfg(
        func=microduck_mdp.pose_l1_penalty, weight=1.0,
    )
    cfg.rewards["feet_flat"] = RewardTermCfg(
        func=microduck_mdp.feet_flat_penalty,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
            "sensor_name": "feet_ground_contact",
        },
    )
    cfg.rewards["neck_action_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.neck_action_rate_l2, weight=-0.5,
    )
    cfg.rewards["joint_torques_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torques_l2, weight=-1e-3,
    )
    cfg.rewards["action_rate_l2"].weight = -1.0

    # === TERMINATIONS : chute + tombé dans le vide ===
    # Le plat de sortie donne du solide au bas de la rampe, donc plus besoin de
    # terminer « au bord » (terrain_edge_reached coupait trop tôt les rampes
    # longues). On garde : chute (bad_orientation), NaN, et « tombé dans le vide »
    # (trunk sous le plat de sortie le plus bas) au cas où le robot quitte le solide.
    cfg.terminations["fell_over"] = TerminationTermCfg(
        func=base_mdp.bad_orientation,
        params={"limit_angle": 1.0, "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    if "out_of_terrain_bounds" in cfg.terminations:
        del cfg.terminations["out_of_terrain_bounds"]
    cfg.terminations["fell_into_void"] = TerminationTermCfg(
        func=microduck_mdp.root_height_below,
        params={"min_height": VOID_FLOOR, "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan, time_out=False,
    )

    # === OBS : assainir les NaN/Inf (robustesse aux divergences de contact rares) ===
    # Un contact rare (~1/25M pas-env) fait diverger le free-joint en NaN. À cause
    # du décalage d'un sous-pas, la terminaison nan_state ne l'attrape qu'AU PAS
    # SUIVANT (reset), mais le NaN atteint déjà l'obs du pas courant -> check_nan de
    # rsl_rl tue l'entraînement. nan_policy="sanitize" remplace NaN/Inf par 0 dans
    # l'obs renvoyée (pas de crash) ; nan_state reset ensuite l'env fautif.
    for grp in ("actor", "critic"):
        cfg.observations[grp].nan_policy = "sanitize"

    # === EVENTS ===
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history, mode="reset",
    )

    # === CURRICULUM : raideur de la rampe ===
    for name in list(cfg.curriculum.keys()):
        del cfg.curriculum[name]
    cfg.curriculum["terrain_levels"] = CurriculumTermCfg(func=microduck_mdp.terrain_levels_slope)

    return cfg


MicroduckRollerSlopeRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
        entropy_coef=0.01, num_learning_epochs=5, num_mini_batches=4,
        learning_rate=1.0e-3, schedule="adaptive", gamma=0.99, lam=0.95,
        desired_kl=0.01, max_grad_norm=1.0, symmetry_cfg=None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="roller_slope",
    run_name="roller_slope",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=8_000,
)
