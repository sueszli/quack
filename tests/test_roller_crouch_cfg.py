from src import task_mdp as microduck_mdp
from src.task_roller_crouch import make_microduck_roller_crouch_env_cfg


def test_cfg_uses_phase_command():
    cfg = make_microduck_roller_crouch_env_cfg()
    cmd = cfg.commands["twist"]
    assert isinstance(cmd, microduck_mdp.GroundPickPhaseCommandCfg)
    assert cmd.period == 5.0
    assert cmd.randomize_phase is False


def test_cfg_has_crouch_and_forward_rewards():
    cfg = make_microduck_roller_crouch_env_cfg()
    assert "crouch_glide_pose" in cfg.rewards
    assert "crouch_glide_pose_l1" in cfg.rewards
    assert "forward_speed" in cfg.rewards
    assert "crouch_forward_lean" in cfg.rewards
    assert cfg.rewards["crouch_forward_lean"].params["target_pitch"] > 0.0
    cp = cfg.rewards["crouch_glide_pose"].params["crouch_pose"]
    assert "left_knee" in cp and "right_knee" in cp
    for gone in ("braking", "skating_air_time", "single_support", "glide", "wheel_speed"):
        assert gone not in cfg.rewards


def test_entry_velocity_applied_safely_via_reset_base():
    cfg = make_microduck_roller_crouch_env_cfg()
    assert "entry_velocity" not in cfg.events
    vr = cfg.events["reset_base"].params.get("velocity_range")
    assert vr and "x" in vr
    lo, hi = vr["x"]
    assert lo > 0.0 and hi >= lo
