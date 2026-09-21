"""Deterministic safety supervisor for two wheeled-foot policy experts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


class WheelfootMode(str, Enum):
    WHEEL = "wheel"
    WHEEL_TO_FOOT = "wheel_to_foot"
    FOOT = "foot"
    FOOT_TO_WHEEL = "foot_to_wheel"


@dataclass
class WheelfootModeFSMCfg:
    transition_steps: int = 50
    min_dwell_steps: int = 100
    max_base_speed: float = 0.25
    max_wheel_speed: float = 1.0
    minimum_upright_cosine: float = 0.85


class WheelfootModeFSM:
    """Blend expert actions while enforcing zero wheel commands in foot mode."""

    def __init__(self, cfg: WheelfootModeFSMCfg | None = None, initial_mode: str = "wheel"):
        self.cfg = cfg or WheelfootModeFSMCfg()
        self.mode = WheelfootMode(initial_mode)
        self.requested_mode = self.mode
        self.transition_step = 0
        self.dwell_steps = 0

    @property
    def is_foot_control(self) -> bool:
        return self.mode in (WheelfootMode.FOOT, WheelfootMode.WHEEL_TO_FOOT)

    def request(self, mode: str | WheelfootMode):
        requested = WheelfootMode(mode)
        if requested in (WheelfootMode.WHEEL_TO_FOOT, WheelfootMode.FOOT_TO_WHEEL):
            raise ValueError("Only terminal modes 'wheel' and 'foot' may be requested.")
        self.requested_mode = requested

    def _stable_for_transition(
        self,
        base_speed: float,
        wheel_speed: float,
        both_wheels_contact: bool,
        upright_cosine: float,
    ) -> bool:
        return (
            both_wheels_contact
            and base_speed <= self.cfg.max_base_speed
            and wheel_speed <= self.cfg.max_wheel_speed
            and upright_cosine >= self.cfg.minimum_upright_cosine
        )

    def update(
        self,
        *,
        base_speed: float,
        wheel_speed: float,
        both_wheels_contact: bool,
        upright_cosine: float,
    ) -> WheelfootMode:
        """Advance the supervisor state using measured safety conditions."""
        self.dwell_steps += 1
        stable = self._stable_for_transition(base_speed, wheel_speed, both_wheels_contact, upright_cosine)

        if self.mode == WheelfootMode.WHEEL:
            if self.requested_mode == WheelfootMode.FOOT and self.dwell_steps >= self.cfg.min_dwell_steps:
                self.mode = WheelfootMode.WHEEL_TO_FOOT
                self.transition_step = 0
        elif self.mode == WheelfootMode.FOOT:
            if self.requested_mode == WheelfootMode.WHEEL and self.dwell_steps >= self.cfg.min_dwell_steps:
                self.mode = WheelfootMode.FOOT_TO_WHEEL
                self.transition_step = 0
        elif self.mode == WheelfootMode.WHEEL_TO_FOOT:
            # Fade wheel drive even before the robot is fully settled; otherwise the
            # transition can deadlock while the wheel expert keeps commanding speed.
            self.transition_step = min(self.transition_step + 1, self.cfg.transition_steps)
            if stable and self.transition_step >= self.cfg.transition_steps:
                self.mode = WheelfootMode.FOOT
                self.transition_step = 0
                self.dwell_steps = 0
        elif self.mode == WheelfootMode.FOOT_TO_WHEEL and stable:
            self.transition_step += 1
            if self.transition_step >= self.cfg.transition_steps:
                self.mode = WheelfootMode.WHEEL
                self.transition_step = 0
                self.dwell_steps = 0

        return self.mode

    def route_actions(self, wheel_actions: torch.Tensor, foot_actions: torch.Tensor) -> torch.Tensor:
        """Select or blend all actions, keeping Foot wheel targets within ±1 rad/s."""
        if wheel_actions.shape != foot_actions.shape:
            raise ValueError(f"Expert action shapes must match: {wheel_actions.shape} != {foot_actions.shape}")
        if wheel_actions.shape[-1] < 2:
            raise ValueError("Wheelfoot actions must contain two wheel commands at the end.")

        bounded_foot_actions = foot_actions.clone()
        bounded_foot_actions[..., -2:] = 0.0

        if self.mode == WheelfootMode.WHEEL:
            return wheel_actions
        if self.mode == WheelfootMode.FOOT:
            return bounded_foot_actions

        alpha = min(self.transition_step / max(self.cfg.transition_steps, 1), 1.0)
        if self.mode == WheelfootMode.WHEEL_TO_FOOT:
            actions = (1.0 - alpha) * wheel_actions + alpha * bounded_foot_actions
        else:
            actions = (1.0 - alpha) * bounded_foot_actions + alpha * wheel_actions
        return actions


def terrain_mode_request(ray_hits_w: torch.Tensor, relief_threshold: float = 0.08) -> WheelfootMode:
    """Conservative terrain heuristic: discontinuous/high-relief terrain requests foot mode."""
    height = ray_hits_w[..., 2]
    finite = torch.isfinite(height)
    max_height = torch.where(finite, height, -torch.inf).max(dim=1).values
    min_height = torch.where(finite, height, torch.inf).min(dim=1).values
    relief = torch.where(finite.any(dim=1), max_height - min_height, torch.zeros_like(max_height))
    return WheelfootMode.FOOT if float(torch.max(relief).item()) >= relief_threshold else WheelfootMode.WHEEL
