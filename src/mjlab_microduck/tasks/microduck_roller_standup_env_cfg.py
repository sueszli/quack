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

    # ── Récompenses de relevé — transplant du standup, remappé ───────────────
    # Les poids viennent des itérations documentées dans
    # microduck_standup_env_cfg.py : ne les retoucher qu'avec une raison. Seuls
    # les indices de joints et les deux hauteurs changent ici.
    # NB : un SceneEntityCfg NEUF par terme — mjlab les résout et les mute en
    # place, un objet partagé donne des indices périmés.

    # Pose cible = HOME (target_overrides=None), JAMBES seulement : le cou et la
    # tête sont tenus par neck_joint_pos_l2 (hérité), qui résout par NOM.
    cfg.rewards["pose_stand_legs"] = RewardTermCfg(
        func=microduck_mdp.pose_target_match,
        weight=8.0,
        params={
            "std": 0.5,
            "joint_indices": _LEG_JOINTS,
            "target_overrides": None,
        },
    )
    # Bootstrap L1 : gradient constant même loin de HOME (la gaussienne sature).
    cfg.rewards["pose_stand_l1"] = RewardTermCfg(
        func=microduck_mdp.pose_l1_penalty,
        weight=5.0,
        params={
            "joint_indices": _LEG_JOINTS,
            "target_overrides": None,
        },
    )

    # Hauteur en trois couches : gaussienne large (tire depuis le sol),
    # gaussienne étroite (force les derniers cm, là où la large est saturée),
    # et L1 fort qui rend « rester par terre » net NÉGATIF — sans lui, la policy
    # se contente de l'optimum paresseux « immobile au sol ».
    cfg.rewards["height_stand"] = RewardTermCfg(
        func=microduck_mdp.height_target_gaussian,
        weight=4.0,
        params={
            "std": 0.04,
            "target_height": ROLLER_STAND_Z,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    cfg.rewards["height_stand_sharp"] = RewardTermCfg(
        func=microduck_mdp.height_target_gaussian,
        weight=4.0,
        params={
            "std": 0.015,
            "target_height": ROLLER_STAND_Z,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    cfg.rewards["height_stand_l1"] = RewardTermCfg(
        func=microduck_mdp.height_l1_penalty,
        weight=30.0,
        params={
            "target_height": ROLLER_STAND_Z,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # Paye le MOUVEMENT de montée, pas seulement la destination : sans ça,
    # « rester assis en collectant la pose partielle » domine. La coupure est
    # 10 mm AU-DESSUS de la cible, sinon la policy se gare à l'altitude de
    # coupure et ne finit pas la montée.
    cfg.rewards["com_upward_velocity"] = RewardTermCfg(
        func=microduck_mdp.com_upward_velocity,
        weight=3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
            "max_height": ROLLER_STAND_Z + 0.010,
        },
    )
    # Montée douce : pénalise |a_z|. Compatible avec com_upward_velocity — une
    # vitesse verticale constante collecte l'une ET a a_z = 0 → les deux
    # pressions sélectionnent ensemble une montée lisse à vitesse constante.
    cfg.rewards["gentle_rise"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=-0.02,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )

    # Tronc vertical en deux couches : cos(tilt) a un fort gradient quand on est
    # couché mais s'essouffle près de la verticale ; la gaussienne serrée gatée
    # en hauteur prend le relais et tue le penché-arrière (mode d'échec du
    # standup : basculer en arrière en tendant les jambes).
    cfg.rewards["upright_linear"] = RewardTermCfg(
        func=microduck_mdp.body_upright_linear,
        weight=6.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    cfg.rewards["upright_sharp"] = RewardTermCfg(
        func=microduck_mdp.upright_gaussian_at_height,
        weight=6.0,
        params={
            "std": 0.3,
            "height_low": ROLLER_PRONE_Z,
            "height_high": ROLLER_STAND_Z,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # Score MULTIPLICATIF hauteur × verticalité × pose : comme les facteurs se
    # multiplient, être bon sur 2 critères sur 3 ne rapporte rien → casse les
    # compromis « penché à la bonne hauteur » que les récompenses additives
    # laissent passer. Stds volontairement LARGES pour rester visible pendant la
    # montée (des stds serrées donnaient un score ~5e-5, donc zéro gradient).
    cfg.rewards["standing_composite"] = RewardTermCfg(
        func=microduck_mdp.standing_composite_score,
        weight=15.0,
        params={
            "target_height": ROLLER_STAND_Z,
            "height_std": 0.04,
            "upright_std": 0.40,
            "pose_std": 0.40,
            "joint_indices": _LEG_JOINTS,
            "target_overrides": None,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # Anti-jitter : pénalise la VARIATION de couple, pas son amplitude ni la
    # rotation du tronc → amortit la tremblote sans bloquer le retournement.
    # Le standup l'a identifié comme le seul amortisseur qui ne tue pas le
    # relevé depuis le dos.
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torque_rate_l2,
        weight=-2e-3,
    )

    # ── Départ AU SOL : à plat ventre / à plat dos / déjà debout ─────────────
    # Ajouté en DERNIER dans cfg.events : l'ordre d'exécution suit l'ordre
    # d'insertion, et ce terme doit écraser la pose posée par reset_base /
    # reset_robot_joints.
    # Le bucket « déjà debout » n'est pas décoratif : sans lui la policy apprend
    # à monter mais pas à TENIR, et elle retombe juste après s'être relevée.
    # Pas de bucket « assis » → aucun sitting_joint_overrides à remapper (ceux du
    # standup sont des indices du modèle SANS roues).
    # Les probabilités ci-dessous = palier 0 du curriculum ground_state_mix.
    cfg.events["set_ground_state"] = EventTermCfg(
        func=microduck_mdp.set_random_ground_state,
        mode="reset",
        params={
            "face_down_prob": 0.50,   # ventre (+90° de pitch)
            "face_up_prob":   0.00,   # dos — le plus dur, introduit tard
            "sitting_prob":   0.00,
            "standing_prob":  0.50,
            "sitting_joint_overrides": None,
            # Repos au sol : mesuré à 0.075 (ventre) / 0.048 (dos), identique aux
            # deux modèles — c'est la coque du tronc qui touche, pas les pieds.
            "prone_z_min":    0.05,
            "prone_z_max":    0.09,
            # Debout sur roues : ROLLER_STAND_Z = 0.138 (contre 0.11–0.12 sans roues).
            "standing_z_min": 0.134,
            "standing_z_max": 0.144,
            # Bruit de pitch/roll au départ. Attention : dans
            # set_random_ground_state le bucket « debout » réutilise le quaternion
            # du bucket « assis », donc ce bruit s'applique AUSSI aux départs
            # debout — c'est voulu (pas de sur-apprentissage du parfaitement droit).
            "sitting_tilt_max": math.radians(10),
        },
    )

    # Le robot DÉMARRE tombé → la terminaison sur inclinaison n'a aucun sens ici
    # (elle tuerait l'épisode au premier pas). nan_state, hérité, reste.
    cfg.terminations.pop("fell_over", None)

    # Curriculum des poses de départ, easy → hard. Avec un mélange plat dès le
    # départ, la policy optimise la majorité facile et laisse le dos sous-entraîné
    # (leçon du standup : il gelait en « ne rien faire » sur cette pose). On
    # introduit donc debout+ventre d'abord, le dos tard, et on biaise vers les
    # poses dures à la fin pour qu'elles reçoivent le plus d'entraînement.
    cfg.curriculum["ground_state_mix"] = CurriculumTermCfg(
        func=microduck_mdp.event_param_curriculum,
        params={
            "event_name": "set_ground_state",
            "param_stages": [
                {"step": 0, "params": {
                    "standing_prob": 0.50, "sitting_prob": 0.00,
                    "face_down_prob": 0.50, "face_up_prob": 0.00}},
                {"step": 600 * NUM_STEPS_PER_ENV, "params": {
                    "standing_prob": 0.35, "sitting_prob": 0.00,
                    "face_down_prob": 0.45, "face_up_prob": 0.20}},
                {"step": 1500 * NUM_STEPS_PER_ENV, "params": {
                    "standing_prob": 0.25, "sitting_prob": 0.00,
                    "face_down_prob": 0.40, "face_up_prob": 0.35}},
                {"step": 2500 * NUM_STEPS_PER_ENV, "params": {
                    "standing_prob": 0.20, "sitting_prob": 0.00,
                    "face_down_prob": 0.40, "face_up_prob": 0.40}},
            ],
        },
    )

    # ── Friction de roulement INVERSÉE : freinées → libres ───────────────────
    # C'est la seule pièce vraiment nouvelle de cet env, et le cœur de la
    # difficulté : les roues roulent, donc il n'y a AUCUNE adhérence
    # longitudinale pour pousser sur le sol. L'env roller fait MONTER cette
    # friction (0 → 0.0015) ; ici on la fait DESCENDRE, pour bootstrapper le
    # geste sur un problème facile (roues quasi bloquées ≈ des pieds) avant
    # d'imposer la physique réelle du roulement.
    #
    # DIAGNOSTIC à surveiller : si Episode_Reward/standing_composite s'écroule à
    # un palier, le geste « pieds adhérents » ne transfère pas aux roues libres
    # → il faudra guider une technique de patineur (appui genou intermédiaire,
    # un patin à la fois). C'est un résultat exploitable, pas un échec.
    #
    # ATTENTION sim2real : seuls les checkpoints d'APRÈS le dernier palier
    # (iter 4000+) sont candidats au déploiement. Avant, la policy s'appuie sur
    # une friction de roulement qui n'existe pas sur le vrai robot.
    _WHEEL_FRICTION_STAGE0 = (0.0500, 0.0500)
    cfg.curriculum["wheel_friction"] = CurriculumTermCfg(
        func=microduck_mdp.wheel_friction_curriculum,
        params={
            "event_name": "randomize_wheel_friction",
            "ranges_stages": [
                {"step": 0,                        "ranges": _WHEEL_FRICTION_STAGE0},
                {"step": 1000 * NUM_STEPS_PER_ENV, "ranges": (0.0200, 0.0200)},
                {"step": 2000 * NUM_STEPS_PER_ENV, "ranges": (0.0080, 0.0080)},
                {"step": 3000 * NUM_STEPS_PER_ENV, "ranges": (0.0030, 0.0030)},
                {"step": 4000 * NUM_STEPS_PER_ENV, "ranges": (0.0015, 0.0015)},
            ],
        },
    )
    # La valeur de DÉPART de l'événement doit matcher le palier 0 : le curriculum
    # n'est évalué qu'à partir du premier pas, sinon les tout premiers resets
    # utiliseraient le (0, 0) hérité de l'env roller — des roues LIBRES pendant
    # le bootstrap, soit exactement l'inverse du but.
    cfg.events["randomize_wheel_friction"].params["ranges"] = _WHEEL_FRICTION_STAGE0

    # ── action_rate : la rampe du standup, pas celle du roller ───────────────
    # L'env roller monte à -2.0 pour un gait calme. C'est un bloqueur de
    # mouvement : il ralentit l'action rapide dont le relevé depuis le dos a
    # besoin (le standup documente qu'un action_rate trop fort tuait cette
    # récupération). La douceur est portée ici par joint_torque_rate_l2.
    cfg.rewards["action_rate_l2"].weight = -0.6
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0,                       "weight": -0.4},
                {"step": 250 * NUM_STEPS_PER_ENV, "weight": -0.8},
                {"step": 500 * NUM_STEPS_PER_ENV, "weight": -1.0},
            ],
        },
    )

    # ── Poussées rampées ────────────────────────────────────────────────────
    # push_robot est hérité de l'env roller (±0.2 m/s, toutes les 3–6 s) mais
    # sans curriculum. Une bourrade dès le pas 0 parasite le bootstrap du
    # relevé : on la fait monter comme le standup.
    cfg.curriculum["push_magnitude"] = CurriculumTermCfg(
        func=microduck_mdp.push_curriculum,
        params={
            "event_name": "push_robot",
            "push_stages": [
                {"step": 0, "velocity_range": {
                    "x": (0.0, 0.0), "y": (0.0, 0.0)}},
                {"step": 500 * NUM_STEPS_PER_ENV, "velocity_range": {
                    "x": (-0.08, 0.08), "y": (-0.08, 0.08)}},
                {"step": 1000 * NUM_STEPS_PER_ENV, "velocity_range": {
                    "x": (-0.2, 0.2), "y": (-0.2, 0.2)}},
            ],
        },
    )

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
