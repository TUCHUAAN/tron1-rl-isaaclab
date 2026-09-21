"""Custom action terms for TRON1A locomotion."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from isaaclab.envs.mdp.actions.actions_cfg import JointActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointAction
from isaaclab.managers import ActionTerm
from isaaclab.utils import configclass


class WheelVelocityPIAction(JointAction):
    """Track wheel-velocity actions with a randomized PI effort controller.

    The policy output remains a wheel-velocity target, so the action schema and
    reward terms that read ``processed_actions`` do not change.  The controller
    runs at the physics rate and applies effort directly.  Conditional
    integration prevents the integral state from winding up against the wheel
    effort limit.
    """

    cfg: WheelVelocityPIActionCfg

    def __init__(self, cfg: WheelVelocityPIActionCfg, env) -> None:
        super().__init__(cfg, env)
        if not math.isfinite(cfg.kp) or cfg.kp <= 0.0:
            raise ValueError("Wheel PI kp must be a positive finite value.")
        if not math.isfinite(cfg.effort_limit) or cfg.effort_limit <= 0.0:
            raise ValueError("Wheel PI effort_limit must be a positive finite value.")
        if (
            len(cfg.kp_scale_range) != 2
            or cfg.kp_scale_range[0] <= 0.0
            or cfg.kp_scale_range[1] < cfg.kp_scale_range[0]
        ):
            raise ValueError(f"Invalid Wheel PI kp_scale_range: {cfg.kp_scale_range}")
        if (
            len(cfg.ki_ratio_range) != 2
            or cfg.ki_ratio_range[0] < 0.0
            or cfg.ki_ratio_range[1] < cfg.ki_ratio_range[0]
        ):
            raise ValueError(f"Invalid Wheel PI ki_ratio_range: {cfg.ki_ratio_range}")

        self._physics_dt = float(env.physics_dt)
        self._integral_error = torch.zeros_like(self.raw_actions)
        self._kp = torch.full_like(self.raw_actions, float(cfg.kp))
        self._ki = torch.full_like(self.raw_actions, float(cfg.kp * cfg.ki_ratio_range[1]))

    @property
    def kp(self) -> torch.Tensor:
        """Per-environment proportional wheel gains."""
        return self._kp

    @property
    def ki(self) -> torch.Tensor:
        """Per-environment integral wheel gains."""
        return self._ki

    @property
    def integral_error(self) -> torch.Tensor:
        """Accumulated per-wheel velocity error."""
        return self._integral_error

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)

        self._processed_actions[env_ids] = 0.0
        self._integral_error[env_ids] = 0.0
        selected_shape = self._kp[env_ids].shape
        gain_shape = (*selected_shape[:-1], 1)
        kp_scale = torch.empty(gain_shape, device=self.device, dtype=self._kp.dtype).uniform_(
            *self.cfg.kp_scale_range
        )
        ki_ratio = torch.empty(gain_shape, device=self.device, dtype=self._ki.dtype).uniform_(
            *self.cfg.ki_ratio_range
        )
        self._kp[env_ids] = float(self.cfg.kp) * kp_scale
        self._ki[env_ids] = self._kp[env_ids] * ki_ratio

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        if self.cfg.fixed_zero_target:
            self._processed_actions.zero_()

    def apply_actions(self) -> None:
        wheel_velocity = self._asset.data.joint_vel[:, self._joint_ids]
        target = torch.zeros_like(wheel_velocity) if self.cfg.fixed_zero_target else self.processed_actions
        error = target - wheel_velocity
        integral_enabled = self._ki > 0.0
        candidate_integral = torch.where(
            integral_enabled,
            self._integral_error + error * self._physics_dt,
            torch.zeros_like(self._integral_error),
        )

        proportional_effort = self._kp * error
        candidate_effort = proportional_effort + self._ki * candidate_integral
        pushes_upper_limit = (candidate_effort > self.cfg.effort_limit) & (error > 0.0)
        pushes_lower_limit = (candidate_effort < -self.cfg.effort_limit) & (error < 0.0)
        accept_integral = ~(pushes_upper_limit | pushes_lower_limit)
        self._integral_error = torch.where(
            accept_integral,
            candidate_integral,
            self._integral_error,
        )
        self._integral_error = torch.where(
            integral_enabled,
            self._integral_error,
            torch.zeros_like(self._integral_error),
        )

        effort = torch.clamp(
            proportional_effort + self._ki * self._integral_error,
            min=-self.cfg.effort_limit,
            max=self.cfg.effort_limit,
        )
        self._asset.set_joint_effort_target(effort, joint_ids=self._joint_ids)


@configclass
class WheelVelocityPIActionCfg(JointActionCfg):
    """Configuration for :class:`WheelVelocityPIAction`."""

    class_type: type[ActionTerm] = WheelVelocityPIAction
    fixed_zero_target: bool = False
    """Ignore policy wheel targets in Foot; retain the action schema."""
    kp: float = 2.0
    """Nominal proportional gain in N m / (rad/s)."""
    kp_scale_range: tuple[float, float] = (0.25, 2.0)
    """Uniform per-reset multiplier range for ``kp``."""
    ki_ratio_range: tuple[float, float] = (0.0, 0.25)
    """Uniform per-reset range for ``ki / kp`` in 1/s."""
    effort_limit: float = 80.0
    """Absolute wheel-effort limit in N m."""
