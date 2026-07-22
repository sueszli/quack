from mjlab_microduck.tasks.microduck_roller_slope_env_cfg import (
    make_microduck_roller_slope_env_cfg,
)
from mjlab_microduck.tasks.slope_terrain import FlatRampTerrainCfg


def test_terrain_is_flat_ramp_generator():
    cfg = make_microduck_roller_slope_env_cfg()
    assert cfg.scene.terrain.terrain_type == "generator"
    gen = cfg.scene.terrain.terrain_generator
    assert gen is not None and gen.curriculum is True
    assert any(isinstance(st, FlatRampTerrainCfg) for st in gen.sub_terrains.values())


def test_command_is_neutralised():
    cfg = make_microduck_roller_slope_env_cfg()
    cmd = cfg.commands["twist"]
    assert cmd.rel_standing_envs == 1.0
    assert cmd.rel_heading_envs == 0.0
    assert cmd.ranges.lin_vel_x == (0.0, 0.0)
    assert cmd.ranges.lin_vel_y == (0.0, 0.0)
    if getattr(cmd.ranges, "ang_vel_z", None) is not None:
        assert cmd.ranges.ang_vel_z == (0.0, 0.0)


def test_entry_velocity_set_on_reset_base():
    cfg = make_microduck_roller_slope_env_cfg()
    vr = cfg.events["reset_base"].params["velocity_range"]
    assert vr["x"][0] > 0.0  # impulsion vers l'avant


def test_has_upright_and_pose_rewards():
    cfg = make_microduck_roller_slope_env_cfg()
    for name in ("upright", "alive", "standing_pose", "feet_flat"):
        assert name in cfg.rewards


def test_no_roller_rewards_survive():
    cfg = make_microduck_roller_slope_env_cfg()
    assert "wheel_speed" not in cfg.rewards
    assert "braking" not in cfg.rewards
    assert "heading_hold" not in cfg.rewards


def test_spawn_yaw_faces_downhill():
    # yaw restreint autour de 0 (descente +x), pas le -pi/+pi hérité
    cfg = make_microduck_roller_slope_env_cfg()
    lo, hi = cfg.events["reset_base"].params["pose_range"]["yaw"]
    assert lo > -0.6 and hi < 0.6  # ~±20°, bien plus serré que ±180°


def test_void_termination_present_no_edge_termination():
    cfg = make_microduck_roller_slope_env_cfg()
    assert "fell_into_void" in cfg.terminations
    assert "fell_over" in cfg.terminations
    # plus de terminaison « bord de terrain » (remplacée par le plat de sortie)
    assert "reached_bottom" not in cfg.terminations
    assert "out_of_terrain_bounds" not in cfg.terminations


def test_terrain_tile_fits_geometry():
    # la tuile doit contenir plat + rampe_max + sortie
    cfg = make_microduck_roller_slope_env_cfg()
    gen = cfg.scene.terrain.terrain_generator
    st = next(iter(gen.sub_terrains.values()))
    assert st.flat_length + st.ramp_length_range[1] + st.runout_length <= gen.size[0]
