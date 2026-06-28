"""Factory helpers that build official-BAM-backed actuator configs.

Parameters now come from the ``bam`` package's bundled JSON
(``bam/params/xl330/<model>.json``), which is identical to the values that used
to be hardcoded here. These factories simply wire the bundled params into a
:class:`~mjlab_microduck.actuator.bam_actuator.BamActuatorCfg` while preserving
this project's firmware constants (``vin=7.4``, ``kp_fw=200``) and enabling the
XL330 current limit (``max_current=1.75`` A) by default.
"""

from __future__ import annotations

from pathlib import Path

from mjlab_microduck.actuator.bam_actuator import BamActuatorCfg

# Microduck firmware constants (NOT stored in the BAM JSON). Overriding these
# preserves the exact stiffness used to train existing policies; the bam package
# defaults (vin=7.5, kp=400) would make the motors ~2x stiffer.
MICRODUCK_VIN = 7.4
MICRODUCK_KP_FW = 200.0
# XL330 firmware current saturation [A].
XL330_MAX_CURRENT = 1.75


# Kept for backwards-compatible imports (e.g. scripts/testbench_sim2real.py).
# Identical to bam/params/xl330/m6.json (the bundled BAM identification).
DEFAULT_XL330_M6 = {
    "kt": 0.36601349688984386,
    "R": 2.8113923539223227,
    "armature": 0.0018077432831600838,
    "q_offset": 0.0271132870444849,
    "friction_base": 0.004771183165566,
    "friction_stribeck": 0.004676345799486616,
    "load_friction_motor": 0.2667860954283698,
    "load_friction_external": 8.515871897059342e-06,
    "load_friction_motor_stribeck": 1.0722918395099123e-05,
    "load_friction_external_stribeck": 0.08077928978935671,
    "load_friction_motor_quad": 0.009972471242139415,
    "load_friction_external_quad": 0.004902565732332559,
    "dtheta_stribeck": 2.890372094130307,
    "alpha": 8.683259907618984,
    "friction_viscous": 0.005359668274599504,
    "model": "m6",
    "actuator": "xl330",
}


def make_bam_m6_actuator_cfg(
    joint_names_expr: tuple[str, ...] = (r".*",),
    *,
    json_path: str | Path | None = None,
    vin: float = MICRODUCK_VIN,
    kp_fw: float = MICRODUCK_KP_FW,
    max_current: float | None = XL330_MAX_CURRENT,
    vin_range: tuple[float, float] | None = None,
    vin_drop_gain_range: tuple[float, float] | None = None,
    vin_min: float | None = None,
) -> BamActuatorCfg:
    """Create a BAM M6 (xl330) actuator config from the bundled BAM params.

    Set ``vin_range`` for per-env battery-voltage randomization (takes precedence
    over ``vin``), ``vin_drop_gain_range`` for load-dependent voltage sag, and
    ``vin_min`` as the post-sag floor.
    """
    return BamActuatorCfg(
        joint_names_expr=joint_names_expr,
        motor_name="xl330",
        model="m6",
        json_path=str(json_path) if json_path is not None else None,
        vin=vin,
        kp_fw=kp_fw,
        max_current=max_current,
        vin_range=vin_range,
        vin_drop_gain_range=vin_drop_gain_range,
        vin_min=vin_min,
    )


def make_bam_m4_actuator_cfg(
    joint_names_expr: tuple[str, ...] = (r".*",),
    *,
    json_path: str | Path | None = None,
    vin: float = MICRODUCK_VIN,
    kp_fw: float = MICRODUCK_KP_FW,
    max_current: float | None = XL330_MAX_CURRENT,
    vin_range: tuple[float, float] | None = None,
    vin_drop_gain_range: tuple[float, float] | None = None,
    vin_min: float | None = None,
) -> BamActuatorCfg:
    """Create a BAM M4 (xl330) actuator config from the bundled BAM params."""
    return BamActuatorCfg(
        joint_names_expr=joint_names_expr,
        motor_name="xl330",
        model="m4",
        json_path=str(json_path) if json_path is not None else None,
        vin=vin,
        kp_fw=kp_fw,
        max_current=max_current,
        vin_range=vin_range,
        vin_drop_gain_range=vin_drop_gain_range,
        vin_min=vin_min,
    )
