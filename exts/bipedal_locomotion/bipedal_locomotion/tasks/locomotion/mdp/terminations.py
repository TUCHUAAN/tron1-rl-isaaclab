"""Custom termination terms for wheeled-foot locomotion."""

from __future__ import annotations

import math

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor

from .reward_math import update_consecutive_contact_steps


class SustainedIllegalContact(ManagerTermBase):
    """Terminate only after an illegal contact persists for a configured time.

    A short force-history maximum catches impacts between policy updates, while
    the per-environment counter prevents a single brief terrain scrape from
    terminating the episode.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        duration_s = float(cfg.params["duration_s"])
        if duration_s <= 0.0:
            raise ValueError(f"duration_s must be positive, got {duration_s}.")
        self._required_steps = max(1, math.ceil(duration_s / env.step_dt))
        sensor_cfg = cfg.params["sensor_cfg"]
        self._consecutive_contact_steps = torch.zeros(
            (env.num_envs, len(sensor_cfg.body_ids)), dtype=torch.long, device=env.device
        )

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._consecutive_contact_steps[env_ids] = 0

    def __call__(
        self,
        env,
        sensor_cfg: SceneEntityCfg,
        force_threshold: float,
        duration_s: float,
    ) -> torch.Tensor:
        del duration_s  # Converted to policy steps during initialization.
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        force_history_w = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
        self._consecutive_contact_steps = update_consecutive_contact_steps(
            self._consecutive_contact_steps,
            force_history_w,
            force_threshold,
        )
        return torch.any(self._consecutive_contact_steps >= self._required_steps, dim=1)
