from mjlab_microduck.tasks.microduck_roller_crouch_env_cfg import (
    make_microduck_roller_crouch_env_cfg,
)
from mjlab_microduck.tasks import mdp as microduck_mdp


def test_cfg_uses_phase_command():
    cfg = make_microduck_roller_crouch_env_cfg()
    assert isinstance(
        cfg.commands["twist"], microduck_mdp.GroundPickPhaseCommandCfg
    )
    assert cfg.commands["twist"].period == 4.0


def test_cfg_has_crouch_and_forward_rewards():
    cfg = make_microduck_roller_crouch_env_cfg()
    assert "crouch_glide_height" in cfg.rewards
    assert "forward_speed" in cfg.rewards
    # rewards de patinage actif retirées (pas de stride pendant le trick)
    for gone in ("braking", "skating_air_time", "single_support", "glide", "wheel_speed"):
        assert gone not in cfg.rewards


def test_cfg_has_entry_velocity_event():
    cfg = make_microduck_roller_crouch_env_cfg()
    assert "entry_velocity" in cfg.events
