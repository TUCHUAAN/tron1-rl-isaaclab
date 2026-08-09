"""Body-height command generator for wheeled-foot locomotion modes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import UniformBodyHeightCommandCfg


class BodyHeightCommand(CommandTerm):
    """Generate a smooth base-height command relative to the local support terrain."""

    cfg: UniformBodyHeightCommandCfg

    def __init__(self, cfg: UniformBodyHeightCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        initial_height = 0.5 * (cfg.ranges.height[0] + cfg.ranges.height[1])
        self._command = torch.full((self.num_envs, 1), initial_height, device=self.device)
        self._target = self._command.clone()
        self.metrics = {
            "height_command_error": torch.zeros(self.num_envs, device=self.device),
        }

    @property
    def command(self) -> torch.Tensor:
        """Current rate-limited height command, shaped ``(num_envs, 1)``."""
        return self._command

    @property
    def target(self) -> torch.Tensor:
        """Unfiltered sampled target, shaped ``(num_envs, 1)``."""
        return self._target

    def set_target(self, height: float | torch.Tensor, env_ids: Sequence[int] | None = None):
        """Override the sampled target, primarily for play/deployment control."""
        if env_ids is None:
            env_ids = slice(None)
        value = torch.as_tensor(height, device=self.device, dtype=self._target.dtype)
        value = torch.clamp(value, min=self.cfg.ranges.height[0], max=self.cfg.ranges.height[1])
        self._target[env_ids, 0] = value

    def _update_metrics(self):
        self.metrics["height_command_error"][:] = torch.abs(self._target[:, 0] - self._command[:, 0])

    def _resample_command(self, env_ids: Sequence[int]):
        self._target[env_ids, 0].uniform_(*self.cfg.ranges.height)

    def _update_command(self):
        max_delta = self.cfg.max_rate * self._env.step_dt
        delta = torch.clamp(self._target - self._command, min=-max_delta, max=max_delta)
        self._command += delta

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass
