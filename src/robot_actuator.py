# BAM zeroes dof_frictionloss in edit_spec and computes friction itself, so dr.dof_frictionloss is a
# silent no-op; joint-friction DR must go through friction_scale here. randomize_bam_friction
# restores 1.0 before sampling so the scale never accumulates across resets.

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch
from bam.mjlab import BamActuator, BamActuatorCfg
from mjlab.actuator.actuator import ActuatorCmd


class FrictionDRBamActuator(BamActuator):
    def initialize(self, mj_model, model, data, device) -> None:
        super().initialize(mj_model, model, data, device)
        self.friction_scale = torch.ones_like(self.kp_scale)
        self.default_friction_scale = self.friction_scale.clone()

    def _compute_friction_budget(self, motor_torque: torch.Tensor, external_torque: torch.Tensor, stribeck_coeff: torch.Tensor) -> torch.Tensor:
        # Scales the velocity-INDEPENDENT budget only; the viscous term stays nominal.
        base = super()._compute_friction_budget(motor_torque, external_torque, stribeck_coeff)
        fs = getattr(self, "friction_scale", None)
        return base if fs is None else base * fs

    def set_friction_scale(self, env_ids, friction_scale: torch.Tensor) -> None:
        self.friction_scale[env_ids] = friction_scale

    def reset_friction_scale(self, env_ids) -> None:
        self.friction_scale[env_ids] = self.default_friction_scale[env_ids]


@dataclass(kw_only=True)
class FrictionDRBamActuatorCfg(BamActuatorCfg):
    def build(self, entity, target_ids, target_names) -> FrictionDRBamActuator:
        return FrictionDRBamActuator(self, entity, target_ids, target_names)


class BacklashEncoderBamActuator(FrictionDRBamActuator):
    # The real magnetic encoder sits on the OUTPUT side of the gear play, so the firmware position
    # loop closes on main+backlash: while the servo winds through the dead zone the PD error does not
    # change. cmd.vel stays motor-side on purpose (back-EMF and friction are rotor physics, not an
    # encoder signal). The per-joint mask degrades this to FrictionDRBamActuator on models without
    # backlash joints.

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
    def build(self, entity, target_ids, target_names) -> BacklashEncoderBamActuator:
        return BacklashEncoderBamActuator(self, entity, target_ids, target_names)
