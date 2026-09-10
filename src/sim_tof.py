# Simulated VL53L5CX. The 8x8 / 45-degree / 4 m shape is fixed by what tofd publishes and maploc
# reprojects. The per-zone status must stay distinct from the distance: maploc treats "no target"
# as empty space to clear and "could not measure" as no information.

from __future__ import annotations

import mujoco
import numpy as np

# As published by tof/src/lib.rs.
ROWS = 8
COLS = 8
ZONES = ROWS * COLS
STATUS_VALID = 5
STATUS_NO_TARGET = 255

FOV_DEG = 45.0
MAX_RANGE = 4.0


class Tof:
    # Rays are cast in the site's frame (+x forward, +y left, +z up), so a turning head carries the
    # sensor with it.

    def __init__(self, model: mujoco.MjModel, site: int, seed: int = 0):
        self.model = model
        self.site = site
        self.random = np.random.default_rng(seed)

        # Row 0 is the top of the frame and column 0 the sensor's LEFT, matching how kinematics::tof
        # reads the real buffer. Column 0 on the right delivers every frame mirrored to the mapper.
        half = np.radians(FOV_DEG) / 2.0
        edges = np.linspace(-half, half, COLS + 1)
        centres = (edges[:-1] + edges[1:]) / 2.0
        self.directions = np.zeros((ZONES, 3))
        for row in range(ROWS):
            elevation = -centres[row]
            for col in range(COLS):
                azimuth = -centres[col]
                self.directions[row * COLS + col] = [np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)]

    def frame(self, data: mujoco.MjData) -> tuple[list[int], list[int]]:
        # Self-hits are reported, not filtered: a real sensor sees the duck's own beak, and skipping
        # own geometry would hide mounting problems.
        origin = data.site_xpos[self.site].copy()
        rotation = data.site_xmat[self.site].reshape(3, 3)
        world = rotation @ self.directions.T  # (3, ZONES)

        distance_mm = [0] * ZONES
        status = [STATUS_NO_TARGET] * ZONES

        # A zero-length direction makes mj_ray abort the whole process. Reachable: before the first
        # forward pass every site orientation is zero, and a client can connect before then.
        if not np.isfinite(world).all() or np.linalg.norm(world) < 1e-9:
            return distance_mm, status

        geom = np.zeros(1, dtype=np.int32)
        for zone in range(ZONES):
            hit = mujoco.mj_ray(self.model, data, origin, np.ascontiguousarray(world[:, zone]), None, 1, -1, geom)
            if hit < 0 or hit > MAX_RANGE:
                continue
            # Datasheet range-dependent noise. Without it the simulated map is crisp and every
            # downstream filter goes untested.
            sigma = 0.003 + 0.02 * (hit / MAX_RANGE)
            measured = max(0.0, hit + self.random.normal(0.0, sigma))
            distance_mm[zone] = int(measured * 1000.0)
            status[zone] = STATUS_VALID
        return distance_mm, status
