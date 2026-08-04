"""Microduck roller standup — se relever sur rollers.

Policy DÉDIÉE épisodique : le robot démarre au sol (à plat ventre, à plat dos) ou
déjà debout, et doit se remettre debout sur ses rollers puis TENIR la station.
Portage de la recette `standup` (canard marcheur) vers le modèle rollers.

Dérive de l'env roller (`make_microduck_velocity_rollers_env_cfg`) → hérite tel
quel le robot rollers, les capteurs, toute la DR et l'observation 61D, donc
interchangeable au runtime (--new-cmd-obs). C'est le pattern de roller_slope.

Deux différences structurelles avec `standup` :
  - les roues passives sont INTERCALÉES dans l'ordre des joints → indices
    remappés (_LEG_JOINTS ci-dessous), verrouillés par
    tests/test_roller_standup_cfg.py ;
  - pas de commande head_pose : les slots head/body restent zero-paddés
    (convention de la famille roller) et la tête est tenue droite par
    neck_joint_pos_l2, qui résout par NOM.

La pièce nouvelle est le curriculum de friction de roulement, INVERSÉ (roues
freinées → libres) : les roues roulent, donc il n'y a aucune adhérence pour
pousser sur le sol. On bootstrappe avec des roues quasi bloquées puis on rampe
vers la vraie valeur. Si `standing_composite` s'écroule à un palier, le geste
« pieds adhérents » ne transfère pas et il faudra guider une technique de
patineur (appui genou, un patin à la fois).

Déploiement visé : en `--standing` face à la policy roller en `--walking`, avec
la bascule automatique sur la magnitude de la commande de vitesse
(infer_policy.py:262, seuil 0.05) ; le slot twist y est laissé à zéro
(infer_policy.py:239).
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    RewardTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_rollers_env_cfg import (
    make_microduck_velocity_rollers_env_cfg,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg

# ── Hauteurs de tronc (m) ─────────────────────────────────────────────────────
# Mesurées par cinématique exacte (minimum des sommets de maillage des géoms
# collidantes, pose STAND, tronc ramené au contact) sur scene_rollers.xml :
# debout 0.1407, repos à plat ventre 0.0752, repos à plat dos 0.0475.
# Contrôle : le modèle SANS roues donne 0.1172 en cinématique contre STAND_Z=0.115
# mesuré sous charge par standup → ~2 mm d'affaissement, appliqué ici aussi.
# 0.138 tombe dans le reset_base z (0.1335–0.1435) déjà utilisé par l'env roller.
ROLLER_STAND_Z = 0.138
ROLLER_PRONE_Z = 0.075

EPISODE_LENGTH_S  = 6.0   # monter + stabiliser, comme standup
NUM_STEPS_PER_ENV = 24

# ── Indices de joints — les roues passives sont INTERCALÉES ───────────────────
# Ordre réel du modèle rollers (18 joints après le free-joint), vérifié dans
# MuJoCo via get_walk_rollers_spec().compile() :
#   0-4   left_hip_yaw, left_hip_roll, left_hip_pitch, left_knee, left_ankle
#   5-6   passive_LF_wheel, passive_LR_wheel
#   7-10  neck_pitch, head_pitch, head_yaw, head_roll
#   11-15 right_hip_yaw, right_hip_roll, right_hip_pitch, right_knee, right_ankle
#   16-17 passive_RF_wheel, passive_RR_wheel
# Le standup utilise [0-4, 9-13] / [5-8] : ce sont les indices du modèle SANS
# roues, ils ne valent PAS ici. Verrouillé par tests/test_roller_standup_cfg.py.
#
# Seul _LEG_JOINTS est consommé (par les récompenses de pose). _NECK_JOINTS et
# _WHEEL_JOINTS servent à la documentation et au test d'indices : le cou est
# résolu par NOM (neck_joint_pos_l2 appelle find_joints(r".*(neck|head).*") à
# chaque pas) et les roues par la regex ^passive_.*.
_LEG_JOINTS   = [0, 1, 2, 3, 4, 11, 12, 13, 14, 15]
_NECK_JOINTS  = [7, 8, 9, 10]
_WHEEL_JOINTS = [5, 6, 16, 17]

# Récompenses de PATINAGE de l'env roller : aucun sens quand on est par terre.
# feet_flat : les lames ne sont PAS à plat pendant la montée → combattrait le geste.
# hip_roll_neutral : se relever demande d'écarter les jambes.
# pose / com_height_target : remplacés par les cibles pose/hauteur du relevé.
# upright (gaussienne de base) : remplacée par upright_linear + upright_sharp.
_SKATING_REWARDS = (
    "wheel_speed",
    "braking",
    "skating_air_time",
    "glide",
    "single_support",
    "gait_symmetry",
    "forward_lean",
    "heading_hold",
    "feet_flat",
    "hip_roll_neutral",
    "pose",
    "com_height_target",
    "upright",
)


def make_microduck_roller_standup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Env « se relever sur rollers » : départ au sol, cible = debout sur roues."""
    cfg = make_microduck_velocity_rollers_env_cfg(play=play)

    cfg.episode_length_s = EPISODE_LENGTH_S

    # ── Récompenses de patinage retirées ─────────────────────────────────────
    for name in _SKATING_REWARDS:
        cfg.rewards.pop(name, None)

    # ── Commande : slot twist neutralisé (≈ 0) ───────────────────────────────
    # L'env roller installe un RelativeHeadingVelocityCommandCfg (cmd[2] = erreur
    # de cap calculée en interne). Ici on ne pilote rien : on repasse au
    # command-only neutralisé, comme standup. Les slots head_pose (4) et
    # body_pose (6) restent zero-paddés → parité d'obs 61D préservée.
    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs  = 0.0
    command.heading_command   = False
    command.ranges.heading    = None
    command.resampling_time_range = (EPISODE_LENGTH_S, EPISODE_LENGTH_S * 2)
    command.debug_vis = False
    command.ranges.lin_vel_x = (-0.01, 0.01)
    command.ranges.lin_vel_y = (-0.01, 0.01)
    command.ranges.ang_vel_z = (-0.05, 0.05)
    cfg.commands["twist"] = microduck_mdp.VelocityCommandCommandOnlyCfg(**vars(command))

    # ── Robustesse numérique (même choix que roller_slope) ───────────────────
    # Un contact rare (~1/25M pas) fait diverger le free-joint en NaN : on
    # assainit l'obs (→ 0) pour ne pas tuer l'entraînement, l'env fautif se reset
    # au pas suivant.
    for grp in ("actor", "critic"):
        cfg.observations[grp].nan_policy = "sanitize"

    return cfg


# ── Config du runner RL — identique à standup ─────────────────────────────────
MicroduckRollerStandUpRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,  # le normaliseur DOIT être baké dans l'ONNX par export.py
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # Symétrie OFF : SYMMETRY_CFG est câblé pour l'ancien layout 51D et casse
        # sur le 61D (même situation que tous les envs v1.5+).
        symmetry_cfg=None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="roller_standup",
    run_name="roller_standup",
    save_interval=250,
    num_steps_per_env=NUM_STEPS_PER_ENV,
    max_iterations=15_000,
)
