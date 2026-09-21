"""Sub-module containing command generators for the velocity-based locomotion task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import UniformGaitCommandCfg


class GaitCommand(CommandTerm):
    """Command generator that generates gait frequency, phase offset and contact duration."""

    cfg: UniformGaitCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: UniformGaitCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # create buffers to store the command
        # command format: [frequency, phase offset, contact duration, reserved swing height]
        self.gait_command = torch.zeros(self.num_envs, 4, device=self.device)
        self._phase = torch.zeros(self.num_envs, device=self.device)
        self._phase_step = env.common_step_counter
        # create metrics dictionary for logging
        self.metrics = {}

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "GaitCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The gait command. Shape is (num_envs, 4)."""
        return self.gait_command

    @property
    def phase(self) -> torch.Tensor:
        """Shared phase for observations and rewards, in cycles [0, 1)."""
        if not self.cfg.continuous_phase:
            # Preserve the existing Wheel/PF behavior unless opted in.
            return torch.remainder(
                self._env.episode_length_buf * self._env.step_dt * self.gait_command[:, 0], 1.0
            )
        self._sync_phase()
        return self._phase

    def _sync_phase(self) -> None:
        """Account for elapsed physics before a frequency resample can occur.

        Isaac evaluates rewards before command.compute(). A shared global-step
        guard makes repeated reads and reward/command ordering idempotent.
        """
        if not self.cfg.continuous_phase:
            return
        step = self._env.common_step_counter
        elapsed = step - self._phase_step
        if elapsed > 0:
            self._phase.add_(self.gait_command[:, 0] * (elapsed * self._env.step_dt)).remainder_(1.0)
            self._phase_step = step

    def reset(self, env_ids=None) -> dict[str, float]:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        if len(env_ids) == 0:
            return {}
        self._sync_phase()
        extras = super().reset(env_ids)
        self._phase[env_ids] = 0.0
        return extras

    def _update_metrics(self):
        """Update the metrics based on the current state.

        In this implementation, we don't track any specific metrics.
        """
        pass

    def _resample_command(self, env_ids):
        """Resample the gait command for specified environments."""
        # Advance using the old frequency; the new one applies to future steps.
        self._sync_phase()
        # sample gait parameters
        r = torch.empty(len(env_ids), device=self.device)
        # -- frequency
        self.gait_command[env_ids, 0] = r.uniform_(*self.cfg.ranges.frequencies)
        # -- phase offset
        self.gait_command[env_ids, 1] = r.uniform_(*self.cfg.ranges.offsets)
        # -- contact duration
        self.gait_command[env_ids, 2] = r.uniform_(*self.cfg.ranges.durations)
        # -- swing height
        self.gait_command[env_ids, 3] = r.uniform_(*self.cfg.ranges.swing_height)

    def _update_command(self):
        """Keep the clock current even when no reward reads the phase."""
        self._sync_phase()

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization into visualization objects.

        In this implementation, we don't provide any debug visualization.
        """
        pass

    def _debug_vis_callback(self, event):
        """Callback for debug visualization.

        In this implementation, we don't provide any debug visualization.
        """
        pass
