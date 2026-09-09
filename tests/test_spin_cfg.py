from src import task_mdp as microduck_mdp
from src.task_spin import MicroduckSpinRlCfg, make_microduck_spin_env_cfg


def test_cfg_uses_phase_command_with_runtime_default_period():
    cfg = make_microduck_spin_env_cfg()
    cmd = cfg.commands["twist"]
    assert isinstance(cmd, microduck_mdp.GroundPickPhaseCommandCfg)
    assert cmd.period == 4.0  # the default of --ground-pick-period
    # Every episode starts at phase 0 (standing), like the button at deployment.
    assert cmd.randomize_phase is False


def test_cfg_has_the_spin_rewards():
    cfg = make_microduck_spin_env_cfg()
    for name in ("spin_rate_track", "spin_rate_l1", "spin_stay_in_place", "spin_wheel_differential", "spin_grounded", "leg_antisymmetry"):
        assert name in cfg.rewards, name
    assert cfg.rewards["spin_rate_track"].weight == 6.0
    assert cfg.rewards["spin_stay_in_place"].weight < 0.0


def test_stay_in_place_is_attenuated_during_the_launch_ramp():
    # At -3.0 and full price it would oppose the injection of angular momentum.
    cfg = make_microduck_spin_env_cfg()
    params = cfg.rewards["spin_stay_in_place"].params
    assert 0.0 < params["launch_scale"] < 1.0
    assert params["accel_end"] == microduck_mdp.SPIN_ACCEL_END
    # Positive target = counter-clockwise; the envelope carries the direction.
    assert microduck_mdp.SPIN_RATE_MAX > 0.0


def test_angular_momentum_reward_is_removed():
    # angular_momentum_penalty prices the 3D NORM: it would fight the spin.
    cfg = make_microduck_spin_env_cfg()
    assert "angular_momentum" not in cfg.rewards
    # body_ang_vel prices only x/y, so it stays and tames the wobble.
    assert "body_ang_vel" in cfg.rewards


def test_head_yaw_is_free_to_act_as_a_flywheel():
    cfg = make_microduck_spin_env_cfg()
    pattern = cfg.rewards["neck_joint_pos_l2"].params["pattern"]
    assert "head_yaw" not in pattern


def test_entry_velocity_allows_standstill_and_slow_roll():
    cfg = make_microduck_spin_env_cfg()
    # Never via a reset-mode push: that blew the base free joint up to NaN (crouch).
    assert "entry_velocity" not in cfg.events
    lo, hi = cfg.events["reset_base"].params["velocity_range"]["x"]
    assert lo == 0.0 and hi > 0.0


def test_symmetry_augmentation_is_disabled():
    # The mirror loss would turn a spin to the left into a spin to the right.
    assert MicroduckSpinRlCfg.algorithm.symmetry_cfg is None


def test_leg_antisymmetry_shaping_decays():
    cfg = make_microduck_spin_env_cfg()
    stages = cfg.curriculum["leg_antisym_weight"].params["weight_stages"]
    weights = [s["weight"] for s in stages]
    assert weights[0] == cfg.rewards["leg_antisymmetry"].weight
    assert weights == sorted(weights, reverse=True)
    assert weights[-1] < weights[0]


def test_actor_observation_keeps_the_61d_slot_layout():
    # Structure only; term-order parity is the next test.
    cfg = make_microduck_spin_env_cfg()
    terms = cfg.observations["actor"].terms
    assert "base_lin_vel" not in terms
    assert "height_scan" not in terms
    for padded in ("head_command", "body_command"):
        assert padded in terms
    assert terms["head_command"].params["dim"] == 4
    assert terms["body_command"].params["dim"] == 6


def test_obs_parity_with_roller_crouch():
    # Parity is mandatory: otherwise the ONNX does not load into the runtime slot.
    from src.task_roller_crouch import make_microduck_roller_crouch_env_cfg

    spin = make_microduck_spin_env_cfg()
    crouch = make_microduck_roller_crouch_env_cfg()
    for grp in ("actor", "critic"):
        assert list(spin.observations[grp].terms.keys()) == list(crouch.observations[grp].terms.keys()), f"observation layout diverges on group {grp}"
