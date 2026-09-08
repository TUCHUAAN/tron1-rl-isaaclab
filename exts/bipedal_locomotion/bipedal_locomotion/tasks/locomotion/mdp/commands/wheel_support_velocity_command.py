"""Velocity command with Wheel Expert support and tracking diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.managers import SceneEntityCfg

from ..reward_math import all_support_confidence, any_support_confidence
from ..rewards import wheel_ground_contact_confidence_components

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import WheelSupportVelocityCommandCfg


class WheelSupportVelocityCommand(UniformVelocityCommand):
    """Mode-balanced velocity command that logs Wheel support-gate behavior."""

    cfg: WheelSupportVelocityCommandCfg

    _DIAGNOSTIC_NAMES = (
        "support_force_L",
        "support_force_R",
        "support_geometry_L",
        "support_geometry_R",
        "support_combined_L",
        "support_combined_R",
        "support_all",
        "support_any",
        "both_support_rate",
        "single_support_rate",
        "no_support_rate",
        "lin_tracking_ungated",
        "lin_tracking_gated",
        "yaw_tracking_ungated",
        "yaw_tracking_gated",
        "moving_command_rate",
        "standing_command_rate",
        "straight_command_rate",
        "yaw_only_command_rate",
        "mixed_command_rate",
        "straight_xy_error",
        "yaw_only_wz_error",
        "yaw_only_wz_acc_rms",
        "mixed_xy_error",
        "mixed_wz_error",
    )
    _CONDITIONAL_DIAGNOSTIC_NAMES = (
        "straight_xy_error",
        "yaw_only_wz_error",
        "yaw_only_wz_acc_rms",
        "mixed_xy_error",
        "mixed_wz_error",
    )

    def __init__(self, cfg: WheelSupportVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        mode_fraction_sum = (
            cfg.rel_standing_envs
            + cfg.rel_straight_envs
            + cfg.rel_yaw_only_envs
            + cfg.rel_mixed_envs
        )
        mode_fractions = (
            cfg.rel_standing_envs,
            cfg.rel_straight_envs,
            cfg.rel_yaw_only_envs,
            cfg.rel_mixed_envs,
        )
        if any(fraction < 0.0 or fraction > 1.0 for fraction in mode_fractions):
            raise ValueError(f"Wheel velocity-command mode fractions must be in [0, 1], got {mode_fractions}.")
        if abs(mode_fraction_sum - 1.0) > 1.0e-6:
            raise ValueError(
                "Wheel velocity-command mode fractions must sum to 1.0, got "
                f"{mode_fraction_sum:.6f}."
            )
        self.is_straight_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_yaw_only_env = torch.zeros_like(self.is_straight_env)
        self.is_mixed_env = torch.zeros_like(self.is_straight_env)
        self._wheel_asset_cfg = SceneEntityCfg(
            cfg.asset_name,
            body_names=list(cfg.wheel_body_names),
            preserve_order=True,
        )
        self._wheel_asset_cfg.resolve(env.scene)
        self._diagnostic_sums = {
            name: torch.zeros(self.num_envs, device=self.device) for name in self._DIAGNOSTIC_NAMES
        }
        self._diagnostic_count = torch.zeros(self.num_envs, device=self.device)
        self._conditional_diagnostic_counts = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in self._CONDITIONAL_DIAGNOSTIC_NAMES
        }
        self._previous_yaw_rate = self.robot.data.root_ang_vel_b[:, 2].clone()
        self._yaw_rate_steps_since_reset = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._steps_since_command_resample = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

    def episode_diagnostic_means(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Return per-environment episode means without clearing the accumulators."""
        if env_ids is None:
            resolved_env_ids: Sequence[int] = torch.arange(self.num_envs, device=self.device)
        else:
            resolved_env_ids = env_ids
        episode_means = {}
        for name, values in self._diagnostic_sums.items():
            if name in self._conditional_diagnostic_counts:
                count = torch.clamp(
                    self._conditional_diagnostic_counts[name][resolved_env_ids], min=1.0
                )
            else:
                count = torch.clamp(self._diagnostic_count[resolved_env_ids], min=1.0)
            mean = values[resolved_env_ids] / count
            if name == "yaw_only_wz_acc_rms":
                mean = torch.sqrt(mean)
            episode_means[name] = mean
        return episode_means

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None:
            resolved_env_ids: Sequence[int] = torch.arange(self.num_envs, device=self.device)
        else:
            resolved_env_ids = env_ids

        episode_means = self.episode_diagnostic_means(resolved_env_ids)
        diagnostics = {}
        for name, values in episode_means.items():
            if name in self._conditional_diagnostic_counts:
                total = torch.sum(self._diagnostic_sums[name][resolved_env_ids])
                count = torch.clamp(
                    torch.sum(self._conditional_diagnostic_counts[name][resolved_env_ids]), min=1.0
                )
                pooled_mean = total / count
                if name == "yaw_only_wz_acc_rms":
                    pooled_mean = torch.sqrt(pooled_mean)
                diagnostics[name] = pooled_mean.item()
            else:
                diagnostics[name] = torch.mean(values).item()
        for values in self._diagnostic_sums.values():
            values[resolved_env_ids] = 0.0
        self._diagnostic_count[resolved_env_ids] = 0.0
        for counts in self._conditional_diagnostic_counts.values():
            counts[resolved_env_ids] = 0.0

        extras = super().reset(resolved_env_ids)
        self._previous_yaw_rate[resolved_env_ids] = self.robot.data.root_ang_vel_b[resolved_env_ids, 2]
        self._yaw_rate_steps_since_reset[resolved_env_ids] = 0
        extras.update(diagnostics)
        return extras

    def _resample_command(self, env_ids: Sequence[int]):
        """Sample one mutually exclusive standing, straight, yaw-only, or mixed mode."""
        super()._resample_command(env_ids)
        mode_env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        mode_sample = torch.rand(len(mode_env_ids), device=self.device)
        standing_edge = self.cfg.rel_standing_envs
        straight_edge = standing_edge + self.cfg.rel_straight_envs
        yaw_only_edge = straight_edge + self.cfg.rel_yaw_only_envs

        is_standing = mode_sample < standing_edge
        is_straight = (mode_sample >= standing_edge) & (mode_sample < straight_edge)
        is_yaw_only = (mode_sample >= straight_edge) & (mode_sample < yaw_only_edge)
        is_mixed = mode_sample >= yaw_only_edge

        self.is_standing_env[mode_env_ids] = is_standing
        self.is_straight_env[mode_env_ids] = is_straight
        self.is_yaw_only_env[mode_env_ids] = is_yaw_only
        self.is_mixed_env[mode_env_ids] = is_mixed
        self._steps_since_command_resample[mode_env_ids] = 0
        # Apply the fixed components immediately so reset observations and the
        # first diagnostic update already see the selected command semantics.
        self.vel_command_b[mode_env_ids[is_standing], :] = 0.0
        self.vel_command_b[mode_env_ids[is_straight], 1:] = 0.0
        self.vel_command_b[mode_env_ids[is_yaw_only], :2] = 0.0
        if self.cfg.heading_command:
            # Heading control belongs only to the mixed mode.  The current Wheel
            # configuration sets rel_heading_envs=0, but keep this invariant if
            # heading commands are enabled in a future experiment.
            self.is_heading_env[mode_env_ids] &= is_mixed

    def _update_command(self):
        """Apply the sampled mode after the base heading/standing post-processing."""
        super()._update_command()
        straight_env_ids = self.is_straight_env.nonzero(as_tuple=False).flatten()
        yaw_only_env_ids = self.is_yaw_only_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[straight_env_ids, 1:] = 0.0
        self.vel_command_b[yaw_only_env_ids, :2] = 0.0

    def _update_metrics(self):
        super()._update_metrics()
        force, geometry, combined = wheel_ground_contact_confidence_components(
            self._env,
            self._wheel_asset_cfg,
            self.cfg.contact_sensor_names,
            self.cfg.terrain_sensor_names,
            self.cfg.wheel_radius,
            self.cfg.force_off,
            self.cfg.force_on,
            self.cfg.geometry_tolerance_on,
            self.cfg.geometry_tolerance_off,
        )
        confidence_all = all_support_confidence(combined)
        confidence_any = any_support_confidence(combined)
        supported = combined >= self.cfg.support_threshold
        support_count = torch.sum(supported, dim=1)

        linear_error_l2 = torch.sum(
            torch.square(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2]),
            dim=1,
        )
        yaw_error_l2 = torch.square(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2])
        linear_error = torch.sqrt(linear_error_l2)
        yaw_error = torch.sqrt(yaw_error_l2)
        linear_tracking = torch.exp(-linear_error_l2 / self.cfg.tracking_std**2)
        yaw_tracking = torch.exp(-yaw_error_l2 / self.cfg.tracking_std**2)
        current_yaw_rate = self.robot.data.root_ang_vel_b[:, 2]
        yaw_acceleration_squared = torch.square(
            (current_yaw_rate - self._previous_yaw_rate) / self._env.step_dt
        )
        yaw_acceleration_valid = (
            (self._yaw_rate_steps_since_reset >= 2)
            & (self._steps_since_command_resample >= 2)
        )

        values = {
            "support_force_L": force[:, 0],
            "support_force_R": force[:, 1],
            "support_geometry_L": geometry[:, 0],
            "support_geometry_R": geometry[:, 1],
            "support_combined_L": combined[:, 0],
            "support_combined_R": combined[:, 1],
            "support_all": confidence_all,
            "support_any": confidence_any,
            "both_support_rate": (support_count == 2).to(linear_tracking.dtype),
            "single_support_rate": (support_count == 1).to(linear_tracking.dtype),
            "no_support_rate": (support_count == 0).to(linear_tracking.dtype),
            "lin_tracking_ungated": linear_tracking,
            "lin_tracking_gated": linear_tracking * confidence_any,
            "yaw_tracking_ungated": yaw_tracking,
            "yaw_tracking_gated": yaw_tracking * confidence_any,
            "moving_command_rate": (
                torch.linalg.vector_norm(self.vel_command_b[:, :2], dim=1) > self.cfg.moving_command_threshold
            ).to(linear_tracking.dtype),
            "standing_command_rate": self.is_standing_env.to(linear_tracking.dtype),
            "straight_command_rate": self.is_straight_env.to(linear_tracking.dtype),
            "yaw_only_command_rate": self.is_yaw_only_env.to(linear_tracking.dtype),
            "mixed_command_rate": self.is_mixed_env.to(linear_tracking.dtype),
        }
        for name, value in values.items():
            self._diagnostic_sums[name] += value

        conditional_values = {
            "straight_xy_error": (linear_error, self.is_straight_env),
            "yaw_only_wz_error": (yaw_error, self.is_yaw_only_env),
            "yaw_only_wz_acc_rms": (
                yaw_acceleration_squared,
                self.is_yaw_only_env & yaw_acceleration_valid,
            ),
            "mixed_xy_error": (linear_error, self.is_mixed_env),
            "mixed_wz_error": (yaw_error, self.is_mixed_env),
        }
        for name, (value, mask) in conditional_values.items():
            mask_float = mask.to(value.dtype)
            self._diagnostic_sums[name] += value * mask_float
            self._conditional_diagnostic_counts[name] += mask_float
        self._diagnostic_count += 1.0
        self._previous_yaw_rate[:] = current_yaw_rate
        self._yaw_rate_steps_since_reset += 1
        self._steps_since_command_resample += 1
