"""A simulated VL53L5CX.

8x8 zones, 45-degree square FOV, 4 m, 15 Hz: `tofd` publishes exactly this shape and `maploc`
reprojects assuming it.

The status byte matters as much as the distance. `maploc` treats "nothing out there" (empty space
to clear on the map) differently from "could not measure" (no information), so a distance-only
simulator would hide bugs that hardware finds.
"""

from __future__ import annotations

import mujoco
import numpy as np

# As `tof/src/lib.rs` publishes it.
ROWS = 8
COLS = 8
ZONES = ROWS * COLS
STATUS_VALID = 5
STATUS_NO_TARGET = 255

# 45 degrees per axis (63 on the diagonal).
FOV_DEG = 45.0
MAX_RANGE = 4.0


class Tof:
    """The 8x8 sensor on one duck's `tof` site.

    Rays are cast in the site's frame (+x forward, +y left, +z up), so a head that turns takes the
    sensor with it — what makes `robot.look` able to scan a room.
    """

    def __init__(self, model: mujoco.MjModel, site: int, seed: int = 0):
        self.model = model
        self.site = site
        self.random = np.random.default_rng(seed)

        # Row 0 is the top of the frame and column 0 the sensor's LEFT, as `kinematics::tof` reads
        # the real buffer. Column 0 on the right mirrors every frame reaching the mapper: oblique
        # walls get inked at their mirror image across the head's axis and no loop ever closes.
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
        """One capture: distances in millimetres and a status per zone.

        Self-hits are reported, not filtered: a real sensor sees the duck's own beak, and skipping
        own geometry would hide the mounting problems this exists to catch.
        """
        origin = data.site_xpos[self.site].copy()
        rotation = data.site_xmat[self.site].reshape(3, 3)
        world = rotation @ self.directions.T  # (3, ZONES)

        distance_mm = [0] * ZONES
        status = [STATUS_NO_TARGET] * ZONES

        # WARNING: a zero-length direction makes `mj_ray` ABORT THE PROCESS ("vector length is too
        # small"). Site orientations are all zero before the first forward pass, and a client can
        # connect before then, so this must be checked rather than trusted.
        if not np.isfinite(world).all() or np.linalg.norm(world) < 1e-9:
            return distance_mm, status

        geom = np.zeros(1, dtype=np.int32)
        for zone in range(ZONES):
            hit = mujoco.mj_ray(self.model, data, origin, np.ascontiguousarray(world[:, zone]), None, 1, -1, geom)
            if hit < 0 or hit > MAX_RANGE:
                continue
            # Datasheet noise growth: a few mm up close, a couple of cm at the far end. Without it
            # a simulated map is suspiciously crisp and every downstream filter goes untested.
            sigma = 0.003 + 0.02 * (hit / MAX_RANGE)
            measured = max(0.0, hit + self.random.normal(0.0, sigma))
            distance_mm[zone] = int(measured * 1000.0)
            status[zone] = STATUS_VALID
        return distance_mm, status
