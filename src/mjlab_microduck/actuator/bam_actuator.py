"""BAM actuator for mjlab, backed by the official ``bam`` package.

This adapts the official BAM torque pipeline (:class:`bam.model.Model`) to the
*pinned* mjlab actuator API. The official ``bam.mjlab`` module targets a newer
mjlab (``cmd.pos`` / ``cmd.vel``, ``Actuator.__init__(cfg, ...)``,
``CommandField``); the mjlab version pinned by this project uses
``cmd.joint_pos`` / ``cmd.joint_vel``, ``Actuator.__init__(entity, ...)`` and
``joint_names_expr``. To avoid bumping mjlab we keep the thin mjlab glue here and
delegate *all* physics (friction model, parameters, current limit) to the ``bam``
dependency so the science stays in sync with upstream.

The friction budget and ``compute`` pipeline mirror ``bam/mjlab.py`` (doc branch
of https://github.com/Rhoban/bam) line-for-line apart from the mjlab API
adaptation.

Reference: Duclusaud et al., "Extended Friction Models for the Physics
Simulation of Servo Actuators", 2024. https://arxiv.org/abs/2410.08650
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import mujoco_warp as mjwarp
import torch

from mjlab.actuator.actuator import Actuator, ActuatorCfg, ActuatorCmd
from mjlab.utils.spec import create_motor_actuator

from bam.actuator import VoltageControlledActuator
from bam.model import Model, load_model

if TYPE_CHECKING:
    from mjlab.entity import Entity


@dataclass(kw_only=True)
class BamActuatorCfg(ActuatorCfg):
    """Configuration for a BAM actuator backed by the ``bam`` package.

    Specify the model with **one** of two mutually exclusive approaches:

    * **Bundled motor**: set ``motor_name`` (e.g. ``"xl330"``) and ``model``
      (e.g. ``"m6"``). Params are read from the JSON bundled with the ``bam``
      package (``bam/params/<motor>/<model>.json``).
    * **Custom JSON**: set ``json_path`` to a BAM params JSON produced by
      ``bam.fit``. Takes precedence over ``motor_name`` / ``model``.
    """

    motor_name: str | None = "xl330"
    """Bundled motor name. Ignored if ``json_path`` is set."""
    model: str | None = "m6"
    """Bundled model variant ``"m1"``–``"m6"``. Ignored if ``json_path`` is set."""
    json_path: str | None = None
    """Custom BAM params JSON path. Takes precedence over ``motor_name``/``model``."""

    vin: float | None = None
    """Supply voltage override [V]. ``None`` → use the value from the bam actuator."""
    kp_fw: float | None = None
    """Firmware P-gain override. ``None`` → use the value from the bam actuator."""
    max_current: float | None = None
    """Firmware current limit [A]. The motor current ``I = motor_torque / kt`` is
    clipped to ``±max_current`` (equivalently torque to ``±max_current·kt``),
    reproducing the XL330 firmware current saturation. ``None`` → no clipping."""

    vin_range: tuple[float, float] | None = None
    """If set, a per-env battery voltage is sampled uniformly from this range at
    startup and held constant across resets. Takes precedence over ``vin``."""
    vin_drop_gain_range: tuple[float, float] | None = None
    """If set, a per-env internal-resistance gain [V/Nm] is sampled at startup,
    modelling the voltage drop ``V_drop = gain · Σ|τ|`` from battery + cable
    resistance (gain ≈ resistance / kt). Held constant across resets."""
    vin_min: float | None = None
    """Hard lower bound on the effective supply voltage [V] after the drop."""

    def __post_init__(self) -> None:
        # json_path wins over the bundled selectors (which carry defaults).
        if self.json_path is not None:
            object.__setattr__(self, "motor_name", None)
            object.__setattr__(self, "model", None)

    def build(
        self, entity: "Entity", joint_ids: list[int], joint_names: list[str]
    ) -> "BamActuator":
        return BamActuator(self, entity, joint_ids, joint_names)


class BamActuator(Actuator):
    """BAM actuator (m1–m6) — fully vectorized over all parallel environments.

    Implements the BAM torque pipeline using parameters/flags read from the
    ``bam`` package's :class:`~bam.model.Model`:

    1. **Voltage control law** — firmware P-controller (position error → duty
       cycle → voltage).
    2. **DC motor torque** — back-EMF equation (voltage → torque), with optional
       firmware current clipping.
    3. **Friction budget** — BAM m1–m6 friction model (Coulomb, Stribeck,
       load-dependent, directional, quadratic).
    4. **Static friction clipping** — BAM Algorithm 1.

    Per-environment gain scaling is supported via :meth:`set_gains` (used by the
    domain-randomization events).
    """

    cfg: BamActuatorCfg

    def __init__(
        self,
        cfg: BamActuatorCfg,
        entity: "Entity",
        joint_ids: list[int],
        joint_names: list[str],
    ) -> None:
        super().__init__(entity, joint_ids, joint_names)
        self.cfg = cfg

        # Load the BAM model from the bundled params (or a custom JSON).
        self._bam_model: Model = load_model(
            cfg.json_path, motor_name=cfg.motor_name, model=cfg.model
        )
        if cfg.vin is not None:
            self._bam_model.actuator.vin = cfg.vin
        if cfg.kp_fw is not None:
            self._bam_model.actuator.kp = cfg.kp_fw

        if not isinstance(self._bam_model.actuator, VoltageControlledActuator):
            raise NotImplementedError(
                f"BamActuator only supports VoltageControlledActuator, "
                f"got {type(self._bam_model.actuator).__name__}"
            )

        self._model: mjwarp.Model | None = None
        self._data: mjwarp.Data | None = None
        self._dt: float = 0.0
        self._device: str = "cpu"
        self._dof_ids: torch.Tensor | None = None

        self.vin_tensor: torch.Tensor | None = None
        self.vin_drop_gain: torch.Tensor | None = None
        self._prev_motor_torque: torch.Tensor | None = None

        # Per-env gain tensors (initialized in initialize(), randomized by DR)
        self.kp_scale: torch.Tensor | None = None
        self.kd_scale: torch.Tensor | None = None
        self.default_kp_scale: torch.Tensor | None = None
        self.default_kd_scale: torch.Tensor | None = None

    # ─────────────────────────────────────────────────────────────────────────
    # mjlab interface
    # ─────────────────────────────────────────────────────────────────────────

    def edit_spec(self, spec: mujoco.MjSpec, joint_names: list[str]) -> None:
        """Convert existing XML position actuators to motor (torque) mode and
        zero out MuJoCo's built-in damping/friction. We handle all friction
        ourselves in :meth:`compute`.
        """
        bam = self._bam_model
        act = bam.actuator
        kt = bam.kt.value
        R = bam.R.value
        armature = float(act.get_extra_inertia())
        # Upper bound of vin_range so MuJoCo's forcerange is always a safe ceiling.
        vin_for_limit = (
            max(self.cfg.vin_range) if self.cfg.vin_range is not None else act.vin
        )
        force_limit = vin_for_limit * kt / R

        target_set = set(joint_names)
        converted: set[str] = set()

        for mjact in spec.actuators:
            tgt = mjact.target
            tgt_name = (
                tgt.name if hasattr(tgt, "name") else (str(tgt) if tgt else None)
            )
            if tgt_name in target_set:
                mjact.set_to_motor()
                mjact.forcelimited = True
                mjact.forcerange = (-force_limit, force_limit)
                mjact.gear = [1.0, 0, 0, 0, 0, 0]
                for joint in spec.joints:
                    if joint.name == tgt_name:
                        joint.armature = armature
                        joint.damping = 0.0
                        joint.frictionloss = 0.0
                        break
                self._mjs_actuators.append(mjact)
                converted.add(tgt_name)

        for target_name in joint_names:
            if target_name not in converted:
                actuator = create_motor_actuator(
                    spec,
                    target_name,
                    effort_limit=force_limit,
                    armature=armature,
                    frictionloss=0.0,
                )
                self._mjs_actuators.append(actuator)
                for joint in spec.joints:
                    if joint.name == target_name:
                        joint.damping = 0.0
                        joint.frictionloss = 0.0
                        break

    def initialize(
        self,
        mj_model: mujoco.MjModel,
        model: mjwarp.Model,
        data: mjwarp.Data,
        device: str,
    ) -> None:
        super().initialize(mj_model, model, data, device)
        self._model = model
        self._data = data
        self._dt = mj_model.opt.timestep
        self._device = device

        jnt_dofadr = mj_model.jnt_dofadr
        entity_joint_ids = self.entity.indexing.joint_ids
        dof_ids = [
            jnt_dofadr[entity_joint_ids[tid].item()] for tid in self._joint_ids_list
        ]
        self._dof_ids = torch.tensor(dof_ids, dtype=torch.long, device=device)

        num_envs = data.nworld
        num_joints = len(self._dof_ids)
        self.kp_scale = torch.ones(num_envs, 1, dtype=torch.float32, device=device)
        self.kd_scale = torch.ones(num_envs, 1, dtype=torch.float32, device=device)
        self.default_kp_scale = self.kp_scale.clone()
        self.default_kd_scale = self.kd_scale.clone()

        act = self._bam_model.actuator

        # vin_tensor: (N, 1) — per-env battery voltage, constant across resets.
        if self.cfg.vin_range is not None:
            self.vin_tensor = torch.empty(
                num_envs, 1, dtype=torch.float32, device=device
            ).uniform_(*self.cfg.vin_range)
        else:
            self.vin_tensor = torch.full(
                (num_envs, 1), act.vin, dtype=torch.float32, device=device
            )

        # vin_drop_gain: (N, 1) — per-env resistance gain [V/Nm], constant across resets.
        if self.cfg.vin_drop_gain_range is not None:
            self.vin_drop_gain = torch.empty(
                num_envs, 1, dtype=torch.float32, device=device
            ).uniform_(*self.cfg.vin_drop_gain_range)
        else:
            self.vin_drop_gain = None

        # Previous motor torques for the voltage-drop computation (lagged 1 step).
        self._prev_motor_torque = torch.zeros(
            num_envs, num_joints, dtype=torch.float32, device=device
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        # vin_tensor and vin_drop_gain are startup-randomized: do NOT re-sample.
        # Reset previous motor torques so the voltage-drop model starts clean.
        if self._prev_motor_torque is not None:
            if env_ids is None:
                self._prev_motor_torque.zero_()
            else:
                self._prev_motor_torque[env_ids] = 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # Gain scaling (for domain randomization)
    # ─────────────────────────────────────────────────────────────────────────

    def set_gains(
        self,
        env_ids: torch.Tensor | slice,
        kp_scale: torch.Tensor | None = None,
        kd_scale: torch.Tensor | None = None,
    ) -> None:
        if kp_scale is not None:
            assert self.kp_scale is not None
            self.kp_scale[env_ids] = kp_scale
        if kd_scale is not None:
            assert self.kd_scale is not None
            self.kd_scale[env_ids] = kd_scale

    def reset_gains(self, env_ids: torch.Tensor | slice) -> None:
        assert self.kp_scale is not None and self.default_kp_scale is not None
        assert self.kd_scale is not None and self.default_kd_scale is not None
        self.kp_scale[env_ids] = self.default_kp_scale[env_ids]
        self.kd_scale[env_ids] = self.default_kd_scale[env_ids]

    # ─────────────────────────────────────────────────────────────────────────
    # BAM friction budget (m1–m6 unified, vectorized) — mirrors bam/mjlab.py
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_friction_budget(
        self,
        motor_torque: torch.Tensor,
        external_torque: torch.Tensor,
        stribeck_coeff: torch.Tensor,
    ) -> torch.Tensor:
        """Velocity-independent friction budget — shape ``(N, J)``.

        Covers BAM models m1–m6 by reading flags from the stored Model:

        * **m1**: Coulomb only (``friction_base``)
        * **m2**: + Stribeck (``stribeck``)
        * **m3**: + non-directional load friction (``load_dependent``)
        * **m4**: m3 + Stribeck load friction
        * **m5**: directional load friction (``directional``)
        * **m6**: m5 + quadratic Stribeck load term (``quadratic``)
        """
        bam = self._bam_model
        frictionloss = torch.full_like(motor_torque, bam.friction_base.value)

        if bam.stribeck:
            frictionloss = frictionloss + stribeck_coeff * bam.friction_stribeck.value

        if bam.load_dependent:
            if bam.directional:
                # m5/m6 — directional gearbox torque
                gearbox_torque = torch.abs(
                    external_torque * bam.load_friction_external.value
                    - motor_torque * bam.load_friction_motor.value
                )
                frictionloss = frictionloss + gearbox_torque

                if bam.stribeck:
                    gearbox_torque_stribeck = torch.abs(
                        external_torque * bam.load_friction_external_stribeck.value
                        - motor_torque * bam.load_friction_motor_stribeck.value
                    )
                    frictionloss = (
                        frictionloss + stribeck_coeff * gearbox_torque_stribeck
                    )

                    if bam.quadratic:
                        # m6 — quadratic term; directional: motor-side vs external-side
                        abs_ext = torch.abs(external_torque)
                        abs_mot = torch.abs(motor_torque)
                        drive_mask = (abs_mot > abs_ext).to(motor_torque.dtype)
                        backdrive_mask = 1.0 - drive_mask
                        quad_term = (
                            drive_mask
                            * bam.load_friction_external_quad.value
                            * abs_ext**2
                            + backdrive_mask
                            * bam.load_friction_motor_quad.value
                            * abs_mot**2
                        )
                        frictionloss = frictionloss + stribeck_coeff * quad_term
            else:
                # m3/m4 — non-directional gearbox torque
                gearbox_torque = torch.abs(external_torque - motor_torque)
                frictionloss = (
                    frictionloss + bam.load_friction_base.value * gearbox_torque
                )

                if bam.stribeck:
                    frictionloss = (
                        frictionloss
                        + stribeck_coeff
                        * bam.load_friction_stribeck.value
                        * gearbox_torque
                    )

        return frictionloss

    # ─────────────────────────────────────────────────────────────────────────
    # Main compute — shape (num_envs, num_joints) throughout
    # ─────────────────────────────────────────────────────────────────────────

    def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
        bam = self._bam_model
        act = bam.actuator

        assert self.vin_tensor is not None
        assert self.kp_scale is not None and self.kd_scale is not None
        assert self._data is not None and self._dof_ids is not None
        assert self._model is not None

        # Effective supply voltage, optionally with load-dependent droop.
        vin = self.vin_tensor  # (N, 1)
        if self.vin_drop_gain is not None and self._prev_motor_torque is not None:
            load = self._prev_motor_torque.abs().sum(dim=-1, keepdim=True)  # (N, 1)
            vin = vin - self.vin_drop_gain * load
            if self.cfg.vin_min is not None:
                vin = torch.clamp(vin, min=self.cfg.vin_min)

        kp_fw = act.kp
        kt = bam.kt.value
        R = bam.R.value
        error_gain = act.error_gain
        max_pwm = act.max_pwm
        friction_viscous = bam.friction_viscous.value

        # ── 1. Firmware voltage control law ──
        pos_error = cmd.position_target - cmd.joint_pos
        vel = cmd.joint_vel
        duty_cycle = pos_error * kp_fw * self.kp_scale * error_gain
        duty_cycle = torch.clamp(duty_cycle, -max_pwm, max_pwm)
        voltage = vin * duty_cycle

        # ── 2. DC motor torque with back-EMF ──
        motor_torque = kt * voltage / R - (kt**2) * vel * self.kd_scale / R

        # Firmware current clipping: I = motor_torque / kt capped at ±max_current,
        # i.e. motor torque clipped to ±max_current·kt.
        if self.cfg.max_current is not None:
            torque_limit = self.cfg.max_current * kt
            motor_torque = torch.clamp(motor_torque, -torque_limit, torque_limit)

        # ── 3. External (gravity + Coriolis) torque ──
        qfrc_bias_raw = self._data.qfrc_bias
        if not isinstance(qfrc_bias_raw, torch.Tensor):
            qfrc_bias_raw = torch.as_tensor(qfrc_bias_raw, device=self._device)
        external_torque = -qfrc_bias_raw[:, self._dof_ids]  # (N, J)

        # ── 4. Stribeck coefficient ──
        abs_vel = torch.abs(vel)
        if bam.stribeck:
            stribeck_coeff = torch.exp(
                -torch.pow(abs_vel / bam.dtheta_stribeck.value, bam.alpha.value)
            )
        else:
            stribeck_coeff = torch.zeros_like(vel)

        # ── 5. Friction budget ──
        frictionloss = self._compute_friction_budget(
            motor_torque, external_torque, stribeck_coeff
        )
        friction_budget = frictionloss + friction_viscous * abs_vel

        # ── 6. Static friction clipping — BAM Algorithm 1 ──
        dof_invweight = self._model.dof_invweight0
        if not isinstance(dof_invweight, torch.Tensor):
            dof_invweight = torch.as_tensor(dof_invweight, device=self._device)
        if dof_invweight.ndim == 1:
            eff_inertia = 1.0 / dof_invweight[self._dof_ids].unsqueeze(0)  # (1, J)
        else:
            eff_inertia = 1.0 / dof_invweight[:, self._dof_ids]  # (N, J)

        # tau_stop = (I/dt)·vel + motor_torque + qfrc_bias
        #          = (I/dt)·vel + motor_torque - external_torque
        net_no_friction = motor_torque - external_torque
        tau_stop = (eff_inertia / self._dt) * vel + net_no_friction

        friction_magnitude = torch.minimum(torch.abs(tau_stop), friction_budget)
        friction_torque = -torch.sign(tau_stop) * friction_magnitude

        output = motor_torque + friction_torque

        # Store motor torque for next step's voltage-drop computation.
        if self._prev_motor_torque is not None:
            self._prev_motor_torque = motor_torque.detach()

        return output


# ─────────────────────────────────────────────────────────────────────────────
# Backwards-compatible aliases. The BAM model variant (m4/m6) is now selected via
# BamActuatorCfg.model, and a single class handles all variants by reading flags
# from bam.model.Model. These aliases keep existing imports / isinstance checks
# (e.g. in tasks/mdp.py and the actuator __init__) working unchanged.
# ─────────────────────────────────────────────────────────────────────────────
BamM6Actuator = BamActuator
BamM6ActuatorCfg = BamActuatorCfg
BamM4Actuator = BamActuator
BamM4ActuatorCfg = BamActuatorCfg
