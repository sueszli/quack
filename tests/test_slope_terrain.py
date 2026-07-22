import math

import mujoco
import numpy as np

from mjlab_microduck.tasks.slope_terrain import (
    ramp_angle_by_difficulty,
    RAMP_DEG_MIN,
    RAMP_DEG_MAX,
    FlatRampTerrainCfg,
)


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
    cfg.size = (15.0, 4.0)  # posé normalement par le générateur
    spec = _empty_terrain_spec()
    out = cfg.function(difficulty=0.5, spec=spec, rng=np.random.default_rng(0))
    # trois géométries : plat de départ + rampe + plat de sortie
    assert len(out.geometries) == 3
    # origine sur le plat, PRÈS du bord de la rampe (dans le dernier mètre du plat)
    assert cfg.flat_length - 1.0 <= out.origin[0] <= cfg.flat_length
    assert abs(out.origin[2]) < 1e-6
    # à ~spawn_margin du bord de la rampe (x = flat_length)
    assert abs((cfg.flat_length - out.origin[0]) - cfg.spawn_margin) < 1e-6


def test_flat_ramp_steeper_at_higher_difficulty():
    # à difficulté plus haute, le bout de rampe descend plus bas
    cfg = FlatRampTerrainCfg()
    cfg.size = (15.0, 4.0)
    easy = cfg.function(0.0, _empty_terrain_spec(), np.random.default_rng(0))
    hard = cfg.function(1.0, _empty_terrain_spec(), np.random.default_rng(0))
    # la rampe (2e géométrie) est plus basse (centre z plus négatif) en difficile
    # (même rng -> même longueur tirée -> seule la pente change)
    assert hard.geometries[1].geom.pos[2] < easy.geometries[1].geom.pos[2]


def test_flat_ramp_runout_at_ramp_bottom():
    # le plat de sortie (3e géométrie) est au niveau du bas de la rampe (z<0),
    # et sa surface est plate (box non tourné : quat identité).
    cfg = FlatRampTerrainCfg()
    cfg.size = (15.0, 4.0)
    out = cfg.function(1.0, _empty_terrain_spec(), np.random.default_rng(0))
    runout = out.geometries[2].geom
    assert runout.pos[2] < 0.0  # descendu sous le plat de départ
    # quaternion identité (plat, pas incliné)
    assert math.isclose(runout.quat[0], 1.0, abs_tol=1e-9)


def test_ramp_length_within_range():
    cfg = FlatRampTerrainCfg(ramp_length_range=(3.0, 8.0))
    cfg.size = (15.0, 4.0)
    # surface de rampe = ramp_length / cos(angle) ; à difficulté 0, angle=2°,
    # donc surf_len ~= ramp_length. On vérifie sur plusieurs tirages.
    for seed in range(20):
        out = cfg.function(0.0, _empty_terrain_spec(), np.random.default_rng(seed))
        surf_half = out.geometries[1].geom.size[0]
        ramp_len = surf_half * 2.0 * math.cos(math.radians(2.0))
        assert 3.0 - 1e-6 <= ramp_len <= 8.0 + 1e-6
