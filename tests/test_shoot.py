from mjlab_microduck.tasks.mdp import GroundPickPhaseCommandCfg


def test_phase_cmd_randomize_flag_default_true():
    cfg = GroundPickPhaseCommandCfg(
        entity_name="robot",
        resampling_time_range=(0.1, 0.5),
        ranges=[[0, 0], [0, 0], [0, 0]],
    )
    assert cfg.randomize_phase is True


def test_phase_cmd_randomize_flag_settable_false():
    cfg = GroundPickPhaseCommandCfg(
        entity_name="robot",
        resampling_time_range=(0.1, 0.5),
        ranges=[[0, 0], [0, 0], [0, 0]],
        randomize_phase=False,
    )
    assert cfg.randomize_phase is False
