from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput

RAMP_DEG_MIN = 2.0
RAMP_DEG_MAX = 20.0


def ramp_angle_by_difficulty(difficulty: float, deg_min: float = RAMP_DEG_MIN, deg_max: float = RAMP_DEG_MAX) -> float:
    d = float(np.clip(difficulty, 0.0, 1.0))
    return math.radians(deg_min + d * (deg_max - deg_min))


@dataclass(kw_only=True)
class FlatRampTerrainCfg(SubTerrainCfg):
    flat_length: float = 2.0
    ramp_length_range: tuple = (3.0, 8.0)  # horizontal, drawn per tile at generation
    runout_length: float = 4.0
    spawn_on_ramp: float = 0.3
    deg_min: float = RAMP_DEG_MIN
    deg_max: float = RAMP_DEG_MAX
    thickness: float = 0.5

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng) -> TerrainOutput:
        total_max = self.flat_length + self.ramp_length_range[1] + self.runout_length
        assert total_max <= self.size[0], f"flat+ramp_max+runout ({total_max}) must fit in size[0] ({self.size[0]})"
        body = spec.body("terrain")
        angle = ramp_angle_by_difficulty(difficulty, self.deg_min, self.deg_max)
        width = self.size[1]
        t = self.thickness
        ramp_length = float(rng.uniform(self.ramp_length_range[0], self.ramp_length_range[1]))
        drop = ramp_length * math.tan(angle)

        flat = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(self.flat_length / 2.0, width / 2.0, t / 2.0), pos=(self.flat_length / 2.0, 0.0, -t / 2.0))

        # The -(t/2)·sin(angle) x-offset closes the gap that would otherwise open
        # between the platform edge and the top of the inclined surface.
        surf_len = ramp_length / math.cos(angle)
        ramp_cx = self.flat_length + ramp_length / 2.0 - (t / 2.0) * math.sin(angle)
        ramp_cz = -(drop / 2.0) - (t / 2.0) * math.cos(angle)
        half = angle / 2.0
        ramp = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(surf_len / 2.0, width / 2.0, t / 2.0), pos=(ramp_cx, 0.0, ramp_cz), quat=(math.cos(half), 0.0, math.sin(half), 0.0))

        runout_cx = self.flat_length + ramp_length + self.runout_length / 2.0
        runout = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(self.runout_length / 2.0, width / 2.0, t / 2.0), pos=(runout_cx, 0.0, -drop - t / 2.0))

        # Spawning on the ramp starts the wheels rolling under gravity; a base push
        # instead would skid.
        spawn_x = self.flat_length + self.spawn_on_ramp
        spawn_z = -self.spawn_on_ramp * math.tan(angle)
        origin = np.array([spawn_x, 0.0, spawn_z])
        return TerrainOutput(origin=origin, geometries=[TerrainGeometry(geom=flat, color=(0.5, 0.5, 0.5, 1.0)), TerrainGeometry(geom=ramp, color=(0.45, 0.55, 0.75, 1.0)), TerrainGeometry(geom=runout, color=(0.5, 0.5, 0.5, 1.0))])
