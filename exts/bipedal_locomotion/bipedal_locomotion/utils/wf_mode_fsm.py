"""Mode manager for the TRON1A wheel/foot experts.

The implementation mirrors the UDMMR B2W planner architecture: a requested
mode is resolved to one active mode, the active mode owns the current control
instance, and a switch recreates/selects that mode.  It is intentionally not
an action blender.  This keeps the mode boundary explicit and makes the
switch deterministic and inspectable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


class WheelfootMode(str, Enum):
    WHEEL = "wheel"
    FOOT = "foot"


@dataclass
class WheelfootModeFSMCfg:
    """Mode-switch safety policy.

    ``min_dwell_steps`` prevents command chatter.  ``transition_steps`` is
    retained as a deprecated compatibility field; B2W-style switching does
    not blend actions over these steps.
    """

    transition_steps: int = 0
    min_dwell_steps: int = 0
    require_stable_switch: bool = False
    max_base_speed: float = 0.25
    max_wheel_speed: float = 1.0
    minimum_upright_cosine: float = 0.85
    defer_foot_to_wheel_until_cycle: bool = True


class WheelfootModeFSM:
    """Resolve one requested mode to one active expert; never blend experts."""

    def __init__(self, cfg: WheelfootModeFSMCfg | None = None, initial_mode: str = "wheel"):
        self.cfg = cfg or WheelfootModeFSMCfg()
        self.mode = WheelfootMode(initial_mode)
        self.requested_mode = self.mode
        self.dwell_steps = 0
        self.switch_count = 0
        self.last_switch_reason = "initialization"

    @property
    def is_foot_control(self) -> bool:
        return self.mode == WheelfootMode.FOOT

    @property
    def transitioning(self) -> bool:
        """B2W changes planner instance atomically; there is no blend state."""
        return False

    def request(self, mode: str | WheelfootMode):
        requested = WheelfootMode(mode)
        self.requested_mode = requested

    def _stable_for_switch(
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
        foot_cycle_complete: bool = True,
    ) -> WheelfootMode:
        """Resolve the active expert, analogous to B2W ``ResolveRequestedMode``.

        Wheel→Foot is a direct mode replacement once the safety gate and dwell
        condition pass.  Foot→Wheel is held until the current foot gait cycle
        is complete, matching B2W's deferred planner switch.  No intermediate
        action is ever generated.
        """
        self.dwell_steps += 1
        if self.requested_mode == self.mode:
            return self.mode
        if self.dwell_steps < self.cfg.min_dwell_steps:
            return self.mode
        if self.cfg.require_stable_switch and not self._stable_for_switch(
            base_speed, wheel_speed, both_wheels_contact, upright_cosine
        ):
            return self.mode
        if (
            self.mode == WheelfootMode.FOOT
            and self.requested_mode == WheelfootMode.WHEEL
            and self.cfg.defer_foot_to_wheel_until_cycle
            and not foot_cycle_complete
        ):
            return self.mode

        previous = self.mode
        self.mode = self.requested_mode
        self.dwell_steps = 0
        self.switch_count += 1
        self.last_switch_reason = f"{previous.value}->{self.mode.value}"
        return self.mode

    def route_actions(self, wheel_actions: torch.Tensor, foot_actions: torch.Tensor) -> torch.Tensor:
        """Select the active expert output; switching is atomic, not blended."""
        if wheel_actions.shape != foot_actions.shape:
            raise ValueError(f"Expert action shapes must match: {wheel_actions.shape} != {foot_actions.shape}")
        if wheel_actions.shape[-1] < 2:
            raise ValueError("Wheelfoot actions must contain two wheel commands at the end.")
        if self.mode == WheelfootMode.WHEEL:
            return wheel_actions
        actions = foot_actions.clone()
        actions[..., -2:] = 0.0
        return actions


def terrain_mode_request(ray_hits_w: torch.Tensor, relief_threshold: float = 0.08) -> WheelfootMode:
    """Conservative terrain heuristic used only to form a mode request."""
    height = ray_hits_w[..., 2]
    finite = torch.isfinite(height)
    max_height = torch.where(finite, height, -torch.inf).max(dim=1).values
    min_height = torch.where(finite, height, torch.inf).min(dim=1).values
    relief = torch.where(finite.any(dim=1), max_height - min_height, torch.zeros_like(max_height))
    return WheelfootMode.FOOT if float(torch.max(relief).item()) >= relief_threshold else WheelfootMode.WHEEL
