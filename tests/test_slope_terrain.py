import math

import mujoco
import numpy as np

from src.task_slope_terrain import RAMP_DEG_MAX, RAMP_DEG_MIN, FlatRampTerrainCfg, ramp_angle_by_difficulty


def test_ramp_angle_endpoints():
    assert math.isclose(ramp_angle_by_difficulty(0.0), math.radians(RAMP_DEG_MIN), abs_tol=1e-9)
    assert math.isclose(ramp_angle_by_difficulty(1.0), math.radians(RAMP_DEG_MAX), abs_tol=1e-9)


def test_ramp_angle_midpoint():
    mid_deg = (RAMP_DEG_MIN + RAMP_DEG_MAX) / 2.0
    assert math.isclose(ramp_angle_by_difficulty(0.5), math.radians(mid_deg), abs_tol=1e-9)


def test_ramp_angle_clamps_out_of_range():
    assert math.isclose(ramp_angle_by_difficulty(-1.0), math.radians(RAMP_DEG_MIN), abs_tol=1e-9)
    assert math.isclose(ramp_angle_by_difficulty(2.0), math.radians(RAMP_DEG_MAX), abs_tol=1e-9)


def _empty_terrain_spec():
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    return spec


def test_flat_ramp_builds_geoms_and_origin_on_flat():
    cfg = FlatRampTerrainCfg(flat_length=2.0)
    cfg.size = (15.0, 4.0)  # normally set by the generator
    spec = _empty_terrain_spec()
    out = cfg.function(difficulty=0.5, spec=spec, rng=np.random.default_rng(0))
    assert len(out.geometries) == 3  # start flat + ramp + runout flat
    # The spawn sits ON the ramp, beyond the flat.
    assert out.origin[0] == cfg.flat_length + cfg.spawn_on_ramp
    assert out.origin[2] < 0.0
    angle = ramp_angle_by_difficulty(0.5, cfg.deg_min, cfg.deg_max)
    assert abs(out.origin[2] - (-cfg.spawn_on_ramp * math.tan(angle))) < 1e-9


def test_flat_ramp_steeper_at_higher_difficulty():
    cfg = FlatRampTerrainCfg()
    cfg.size = (15.0, 4.0)
    easy = cfg.function(0.0, _empty_terrain_spec(), np.random.default_rng(0))
    hard = cfg.function(1.0, _empty_terrain_spec(), np.random.default_rng(0))
    # Same rng -> same sampled length, so only the slope changes.
    assert hard.geometries[1].geom.pos[2] < easy.geometries[1].geom.pos[2]


def test_ramp_joins_flat_platform_no_gap():
    # The ramp top must touch the platform edge at x=flat_length, which offsets
    # the ramp center by -(thickness/2)*sin(angle) in x.
    cfg = FlatRampTerrainCfg()
    cfg.size = (15.0, 4.0)
    out = cfg.function(0.5, _empty_terrain_spec(), np.random.default_rng(0))
    ramp = out.geometries[1].geom
    angle = ramp_angle_by_difficulty(0.5, cfg.deg_min, cfg.deg_max)
    surf_half = ramp.size[0]
    ramp_len = surf_half * 2.0 * math.cos(angle)
    expected_cx = cfg.flat_length + ramp_len / 2.0 - (cfg.thickness / 2.0) * math.sin(angle)
    assert abs(ramp.pos[0] - expected_cx) < 1e-6


def test_flat_ramp_runout_at_ramp_bottom():
    cfg = FlatRampTerrainCfg()
    cfg.size = (15.0, 4.0)
    out = cfg.function(1.0, _empty_terrain_spec(), np.random.default_rng(0))
    runout = out.geometries[2].geom
    assert runout.pos[2] < 0.0  # dropped below the start flat
    assert math.isclose(runout.quat[0], 1.0, abs_tol=1e-9)  # flat, not inclined


def test_ramp_length_within_range():
    cfg = FlatRampTerrainCfg(ramp_length_range=(3.0, 8.0))
    cfg.size = (15.0, 4.0)
    # The ramp surface is ramp_length / cos(angle), and angle=2° at difficulty 0.
    for seed in range(20):
        out = cfg.function(0.0, _empty_terrain_spec(), np.random.default_rng(seed))
        surf_half = out.geometries[1].geom.size[0]
        ramp_len = surf_half * 2.0 * math.cos(math.radians(2.0))
        assert 3.0 - 1e-6 <= ramp_len <= 8.0 + 1e-6
