"""Body-height command generator for wheeled-foot locomotion modes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm

from ..reward_math import base_height_plane_error_l2, fit_height_plane, height_bin_event_statistics

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import UniformBodyHeightCommandCfg


class BodyHeightCommand(CommandTerm):
    """Generate a smooth base-height command relative to the local support terrain."""

    cfg: UniformBodyHeightCommandCfg

    def __init__(self, cfg: UniformBodyHeightCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        minimum, maximum = cfg.ranges.height
        if maximum <= minimum:
            raise ValueError(f"Invalid body-height range [{minimum}, {maximum}].")
        if not 0.0 <= cfg.endpoint_fraction <= 1.0:
            raise ValueError("endpoint_fraction must be in [0, 1].")
        initial_height = 0.5 * (cfg.ranges.height[0] + cfg.ranges.height[1])
        self._command = torch.full((self.num_envs, 1), initial_height, device=self.device)
        self._target = self._command.clone()
        self._asset = env.scene[cfg.asset_name]
        self._height_sensor = env.scene.sensors[cfg.height_sensor_name]
        self._tracking_abs_sum = torch.zeros(self.num_envs, device=self.device)
        self._tracking_sq_sum = torch.zeros(self.num_envs, device=self.device)
        self._tracking_valid_count = torch.zeros(self.num_envs, device=self.device)
        self._slew_abs_sum = torch.zeros(self.num_envs, device=self.device)
        self._sample_count = torch.zeros(self.num_envs, device=self.device)

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

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Log episode-averaged robot tracking metrics, reset buffers and resample targets."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        tracking_count = torch.clamp(self._tracking_valid_count[env_ids], min=1.0)
        sample_count = torch.clamp(self._sample_count[env_ids], min=1.0)
        tracking_mae = torch.mean(self._tracking_abs_sum[env_ids] / tracking_count)
        tracking_rmse = torch.sqrt(torch.mean(self._tracking_sq_sum[env_ids] / tracking_count))
        slew_mae = torch.mean(self._slew_abs_sum[env_ids] / sample_count)
        minimum, maximum = self.cfg.ranges.height
        # reset() runs after the current termination computation and before the
        # termination manager is reset, so get_term() is the current episode's
        # exact base-contact result here.
        base_contact = self._env.termination_manager.get_term("base_contact")[env_ids]
        bin_rates, bin_fractions = height_bin_event_statistics(
            self._command[env_ids, 0],
            base_contact,
            minimum,
            maximum,
        )
        extras = {
            "height_tracking_mae_m": tracking_mae.item(),
            "height_tracking_rmse_m": tracking_rmse.item(),
            "height_command_slew_mae_m": slew_mae.item(),
            "base_contact_rate_low_height": bin_rates[0].item(),
            "base_contact_rate_mid_height": bin_rates[1].item(),
            "base_contact_rate_high_height": bin_rates[2].item(),
            "reset_fraction_low_height": bin_fractions[0].item(),
            "reset_fraction_mid_height": bin_fractions[1].item(),
            "reset_fraction_high_height": bin_fractions[2].item(),
        }

        self._tracking_abs_sum[env_ids] = 0.0
        self._tracking_sq_sum[env_ids] = 0.0
        self._tracking_valid_count[env_ids] = 0.0
        self._slew_abs_sum[env_ids] = 0.0
        self._sample_count[env_ids] = 0.0
        self.command_counter[env_ids] = 0
        self._resample(env_ids)
        return extras

    def _update_metrics(self):
        centroid, normal, valid = fit_height_plane(self._height_sensor.data.ray_hits_w)
        squared_error = base_height_plane_error_l2(
            self._asset.data.root_link_pos_w,
            self._command[:, 0],
            centroid,
            normal,
            valid,
        )
        valid_f = valid.to(squared_error.dtype)
        self._tracking_abs_sum += torch.sqrt(torch.clamp(squared_error, min=0.0))
        self._tracking_sq_sum += squared_error
        self._tracking_valid_count += valid_f
        self._slew_abs_sum += torch.abs(self._target[:, 0] - self._command[:, 0])
        self._sample_count += 1.0

    def _resample_command(self, env_ids: Sequence[int]):
        # Advanced indexing returns a copy: sample separately and assign back.
        sampled = torch.empty(len(env_ids), device=self.device, dtype=self._target.dtype)
        self._target[env_ids, 0] = sampled.uniform_(*self.cfg.ranges.height)
        if self.cfg.endpoint_fraction > 0.0:
            selector = torch.rand(len(env_ids), device=self.device)
            half_fraction = 0.5 * self.cfg.endpoint_fraction
            minimum, maximum = self.cfg.ranges.height
            self._target[env_ids, 0] = torch.where(
                selector < half_fraction,
                torch.as_tensor(minimum, device=self.device, dtype=self._target.dtype),
                self._target[env_ids, 0],
            )
            self._target[env_ids, 0] = torch.where(
                (selector >= half_fraction) & (selector < self.cfg.endpoint_fraction),
                torch.as_tensor(maximum, device=self.device, dtype=self._target.dtype),
                self._target[env_ids, 0],
            )

    def _update_command(self):
        max_delta = self.cfg.max_rate * self._env.step_dt
        delta = torch.clamp(self._target - self._command, min=-max_delta, max=max_delta)
        self._command += delta

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass
