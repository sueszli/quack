"""BAM actuator with per-env friction-magnitude domain randomization.

WARNING: under BAM, ``edit_spec`` zeroes MuJoCo's ``dof_frictionloss`` (BAM
computes friction in ``compute()``), so the stock ``dr.dof_frictionloss`` is a
SILENT NO-OP. Randomize ``friction_scale`` instead.

``friction_scale`` multiplies only the velocity-INDEPENDENT friction budget
(Coulomb + Stribeck + load-dependent), which carries the dominant sim2real
friction uncertainty (stiction / gearbox); the viscous term stays nominal.

Non-accumulating: the ``randomize_bam_friction`` event (task_mdp.py) restores
1.0 before applying a fresh sample.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch
from bam.mjlab import BamActuator, BamActuatorCfg
from mjlab.actuator.actuator import ActuatorCmd


class FrictionDRBamActuator(BamActuator):
    """BamActuator + per-env friction_scale on the BAM friction budget."""

    def initialize(self, mj_model, model, data, device) -> None:
        super().initialize(mj_model, model, data, device)
        # kp_scale is (num_envs, 1); mirror its shape.
        self.friction_scale = torch.ones_like(self.kp_scale)
        self.default_friction_scale = self.friction_scale.clone()

    def _compute_friction_budget(self, motor_torque: torch.Tensor, external_torque: torch.Tensor, stribeck_coeff: torch.Tensor) -> torch.Tensor:
        base = super()._compute_friction_budget(motor_torque, external_torque, stribeck_coeff)
        fs = getattr(self, "friction_scale", None)
        return base if fs is None else base * fs  # (N, J) * (N, 1)

    def set_friction_scale(self, env_ids, friction_scale: torch.Tensor) -> None:
        self.friction_scale[env_ids] = friction_scale

    def reset_friction_scale(self, env_ids) -> None:
        self.friction_scale[env_ids] = self.default_friction_scale[env_ids]


@dataclass(kw_only=True)
class FrictionDRBamActuatorCfg(BamActuatorCfg):
    """Drop-in for BamActuatorCfg that builds a friction-DR-capable actuator."""

    def build(self, entity, target_ids, target_names) -> FrictionDRBamActuator:
        return FrictionDRBamActuator(self, entity, target_ids, target_names)


class BacklashEncoderBamActuator(FrictionDRBamActuator):
    """FrictionDRBamActuator whose firmware PD reads the encoder THROUGH backlash.

    A servo joint (motor output) and its ``passive_<joint>_backlash`` hinge
    (play) are in series; the link angle is their sum. The real magnetic
    encoder sits on the OUTPUT side of the play, so the firmware loop closes on
    main+backlash and the PD error does not change while the servo winds
    through the dead zone — hence ``cmd.pos = qpos[main] + qpos[backlash]``.

    ``cmd.vel`` stays motor-side: in BAM it drives back-EMF and friction, which
    are rotor physics, not an encoder-derived firmware signal.

    Degrades to a plain FrictionDRBamActuator where the mask finds no backlash
    joint, so it is safe on any microduck model.
    """

    def initialize(self, mj_model, model, data, device) -> None:
        super().initialize(mj_model, model, data, device)
        name_to_local = {n: i for i, n in enumerate(self.entity.joint_names)}
        ids, mask = [], []
        for name in self._target_names:
            bl_id = name_to_local.get(f"passive_{name}_backlash")
            ids.append(0 if bl_id is None else bl_id)
            mask.append(0.0 if bl_id is None else 1.0)
        self._backlash_joint_ids = torch.tensor(ids, dtype=torch.long, device=device)
        self._backlash_mask = torch.tensor(mask, dtype=torch.float32, device=device)
        n_backlash = int(self._backlash_mask.sum().item())
        print(f"[BacklashEncoderBamActuator] encoder-through-backlash feedback on {n_backlash}/{len(mask)} joints")

    def get_command(self, data) -> ActuatorCmd:
        cmd = super().get_command(data)
        pos = cmd.pos + data.joint_pos[:, self._backlash_joint_ids] * self._backlash_mask
        return dataclasses.replace(cmd, pos=pos)


@dataclass(kw_only=True)
class BacklashEncoderBamActuatorCfg(FrictionDRBamActuatorCfg):
    """FrictionDRBamActuatorCfg whose PD feedback reads through backlash joints."""

    def build(self, entity, target_ids, target_names) -> BacklashEncoderBamActuator:
        return BacklashEncoderBamActuator(self, entity, target_ids, target_names)
