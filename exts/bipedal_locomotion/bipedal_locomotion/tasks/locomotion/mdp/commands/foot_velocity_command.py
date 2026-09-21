"""Foot velocity modes and conditional tracking/clearance diagnostics."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.managers import SceneEntityCfg

from ..rewards import _wheel_terrain_planes

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .commands_cfg import FootVelocityCommandCfg


class FootVelocityCommand(UniformVelocityCommand):
    """Sample five exclusive modes; zero velocity still permits the Foot gait."""

    cfg: FootVelocityCommandCfg
    MODE_NAMES = ("in_place", "straight", "lateral", "yaw_only", "mixed")
    HEIGHT_NAMES = ("low", "middle", "high")
    CYCLE_TERMS = tuple((component, f"pen_cycle_mean_{component}", unit) for component, unit in
                        (("vx", "m_s"), ("vy", "m_s"), ("yaw", "rad_s"),
                         ("height", "m"), ("roll", "rad"), ("pitch", "rad")))

    def __init__(self, cfg: FootVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        fractions = (cfg.rel_standing_envs, cfg.rel_straight_envs, cfg.rel_lateral_envs,
                     cfg.rel_yaw_only_envs, cfg.rel_mixed_envs)
        if any(not math.isfinite(p) or not 0.0 <= p <= 1.0 for p in fractions):
            raise ValueError(f"Foot command fractions must be finite and in [0, 1]: {fractions}")
        if abs(sum(fractions) - 1.0) > 1.0e-6:
            raise ValueError(f"Foot command fractions must sum to 1: {fractions}")
        if cfg.rel_heading_envs != 0.0:
            raise ValueError("Foot velocity modes require rel_heading_envs=0 (direct yaw-rate commands).")
        if len(cfg.height_bin_edges) != 2 or not (
            all(math.isfinite(edge) for edge in cfg.height_bin_edges)
            and cfg.height_bin_edges[0] < cfg.height_bin_edges[1]
        ):
            raise ValueError("Foot height bins require two finite, increasing edges.")
        self._mode_edges = torch.tensor(fractions, device=self.device).cumsum(0)[:-1]
        self._height_edges = torch.tensor(cfg.height_bin_edges, device=self.device)
        self.mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._wheel_cfg = SceneEntityCfg(cfg.asset_name, body_names=list(cfg.wheel_body_names), preserve_order=True)
        self._leg_cfg = SceneEntityCfg(cfg.asset_name, joint_names=list(cfg.leg_joint_names), preserve_order=True)
        self._wheel_cfg.resolve(env.scene)
        self._leg_cfg.resolve(env.scene)
        self._mode_counts = torch.zeros(self.num_envs, 5, device=self.device)
        # Squared XY error, squared yaw error, yaw, squared yaw.
        self._mode_sums = torch.zeros(self.num_envs, 5, 4, device=self.device)
        # Swing-foot samples and control-step samples, conditioned on current height command.
        self._height_counts = torch.zeros(self.num_envs, 3, 2, device=self.device)
        # Actual swing clearance, phase target, shortfall, leg soft-limit excess, peak target.
        self._height_sums = torch.zeros(self.num_envs, 3, 5, device=self.device)
        self._wheel_sums = torch.zeros(self.num_envs, 2, device=self.device)
        # landings, inside, rho sum, planned regions, invalid regions, missed swings.
        self._foothold_sums = torch.zeros(self.num_envs, 6, device=self.device)
        # Complete cycle windows: count, |mean error|, mean error², duration.
        self._cycle_sums = torch.zeros(self.num_envs, len(self.CYCLE_TERMS), 4, device=self.device)

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.mode[ids] = torch.bucketize(torch.rand(len(ids), device=self.device), self._mode_edges, right=True)
        self.is_standing_env[ids] = self.mode[ids] == 0
        self.is_heading_env[ids] = False
        self._apply_modes(ids)

    def _apply_modes(self, ids):
        modes = self.mode[ids]
        self.vel_command_b[ids[modes == 0], :] = 0.0
        self.vel_command_b[ids[modes == 1], 1:] = 0.0
        self.vel_command_b[ids[modes == 2], 0] = 0.0
        self.vel_command_b[ids[modes == 2], 2] = 0.0
        self.vel_command_b[ids[modes == 3], :2] = 0.0

    def _update_command(self):
        super()._update_command()
        self._apply_modes(torch.arange(self.num_envs, device=self.device))

    def _update_metrics(self):
        super()._update_metrics()
        # CommandManager runs after environment resets. Exclude the new episode's
        # reset state, which has not executed an action at its new command yet.
        live = self._env.episode_length_buf > 0
        mode_mask = torch.nn.functional.one_hot(self.mode, 5) * live[:, None]
        yaw = self.robot.data.root_ang_vel_b[:, 2]
        errors = torch.stack((
            (self.command[:, :2] - self.robot.data.root_lin_vel_b[:, :2]).square().sum(1),
            (self.command[:, 2] - yaw).square(), yaw, yaw.square(),
        ), dim=1)
        self._mode_counts += mode_mask
        self._mode_sums += mode_mask[:, :, None] * errors[:, None, :]

        env = self._env
        for i, (_, reward_name, _) in enumerate(self.CYCLE_TERMS):
            if reward_name not in getattr(env.reward_manager, "active_terms", ()):
                continue
            window = env.reward_manager.get_term_cfg(reward_name).func.window
            valid = window.ready & live
            mean_error = window.integral / window.duration.clamp_min(1.e-8)
            values = torch.stack((torch.ones_like(mean_error), mean_error.abs(),
                                  mean_error.square(), window.duration), dim=1)
            self._cycle_sums[:, i] += torch.where(valid[:, None], values, 0.0)
        if "pen_foothold_region" in getattr(env.reward_manager, "active_terms", ()):
            tracker = env.reward_manager.get_term_cfg("pen_foothold_region").func.tracker
            event = tracker.events & live[:, None]
            values = torch.stack((event.sum(1), (event & (tracker.rho <= 1.)).sum(1),
                                  torch.where(event, tracker.rho, 0.).sum(1),
                                  tracker.new_regions.sum(1), tracker.invalid_regions.sum(1),
                                  tracker.missed_swings.sum(1)), dim=1)
            self._foothold_sums += values * live[:, None]
        swing_term = env.reward_manager.get_term_cfg(self.cfg.swing_reward_name).func
        centers, normals, plane_valid = _wheel_terrain_planes(env, self.cfg.terrain_sensor_names)
        positions = self.robot.data.body_link_pos_w[:, self._wheel_cfg.body_ids]
        clearance = ((positions - centers) * normals).sum(-1) - self.cfg.wheel_radius
        gait = env.command_manager.get_command(self.cfg.gait_command_name)
        phase = env.command_manager.get_term(self.cfg.gait_command_name).phase
        foot_phase = torch.stack((phase, (phase + gait[:, 1]).remainder(1.0)), dim=1)
        duration = gait[:, 2:3]
        progress = ((foot_phase - duration) / (1.0 - duration)).clamp(0.0, 1.0)
        target = swing_term.peak_height[:, None] * torch.sin(math.pi * progress).square()
        swing_valid = ((foot_phase > duration) & plane_valid & torch.isfinite(clearance)
                       & swing_term.scan_valid[:, None] & live[:, None])
        # No actual-airborne or opposite-support gate: failed swings must be visible.
        swing_values = torch.stack((clearance, target, (target - clearance).clamp_min(0.0)), dim=-1)
        swing_values = torch.where(swing_valid[:, :, None], swing_values, 0.0).sum(1)
        heights = env.command_manager.get_command(self.cfg.body_height_command_name)[:, 0].contiguous()
        height_bin = torch.bucketize(heights, self._height_edges, right=True)
        height_mask = torch.nn.functional.one_hot(height_bin, 3) * live[:, None]
        q = self.robot.data.joint_pos[:, self._leg_cfg.joint_ids]
        limits = self.robot.data.soft_joint_pos_limits[:, self._leg_cfg.joint_ids]
        excess = ((limits[:, :, 0] - q).clamp_min(0.0) + (q - limits[:, :, 1]).clamp_min(0.0)).sum(1)
        height_values = torch.cat((swing_values, excess[:, None], swing_term.peak_height[:, None]), dim=1)
        counts = torch.stack((swing_valid.sum(1), live), dim=1)
        self._height_counts += height_mask[:, :, None] * counts[:, None, :]
        self._height_sums += height_mask[:, :, None] * height_values[:, None, :]
        raw = env.action_manager.get_term(self.cfg.wheel_action_name).raw_actions
        limit = self.cfg.wheel_target_limit
        clipped = (raw.abs() >= limit).float().mean(1)
        same_limit = (raw >= limit).all(1) | (raw <= -limit).all(1)
        self._wheel_sums += torch.stack((clipped, same_limit), dim=1) * live[:, None]

    def diagnostic_means(self, env_ids=None) -> dict[str, float]:
        """Pool only matching samples, with counts to distinguish empty bins."""
        ids = slice(None) if env_ids is None else env_ids
        counts = self._mode_counts[ids].sum(0)
        sums = self._mode_sums[ids].sum(0)
        means = sums / counts.clamp_min(1.0)[:, None]
        total = counts.sum().clamp_min(1.0)
        diagnostics = {}
        for i, mode in enumerate(self.MODE_NAMES):
            diagnostics[f"{mode}/samples"] = counts[i]
            diagnostics[f"{mode}/fraction"] = counts[i] / total
            diagnostics[f"{mode}/xy_rmse"] = means[i, 0].sqrt()
            diagnostics[f"{mode}/wz_rmse"] = means[i, 1].sqrt()
            diagnostics[f"{mode}/wz_mean"] = means[i, 2]
            diagnostics[f"{mode}/wz_std"] = (means[i, 3] - means[i, 2].square()).clamp_min(0.0).sqrt()
        counts = self._height_counts[ids].sum(0)
        sums = self._height_sums[ids].sum(0)
        for i, name in enumerate(self.HEIGHT_NAMES):
            diagnostics[f"height_{name}/swing_samples"] = counts[i, 0]
            diagnostics[f"height_{name}/step_samples"] = counts[i, 1]
            for j, metric in enumerate(("swing_clearance_m", "swing_target_m", "swing_shortfall_m",
                                        "leg_soft_limit_excess_rad", "peak_target_m")):
                count = counts[i, 0 if j < 3 else 1].clamp_min(1.0)
                diagnostics[f"height_{name}/{metric}"] = sums[i, j] / count
        wheel_means = self._wheel_sums[ids].sum(0) / total
        diagnostics["wheel_raw_output_over_limit_rate"] = wheel_means[0]
        diagnostics["wheel_raw_output_same_sign_over_limit_rate"] = wheel_means[1]
        cycle_sums = self._cycle_sums[ids].sum(0)
        for i, (component, _, unit) in enumerate(self.CYCLE_TERMS):
            count = cycle_sums[i, 0]
            means = cycle_sums[i] / count.clamp_min(1.0)
            prefix = f"cycle_mean_{component}"
            diagnostics[f"{prefix}/samples"] = count
            diagnostics[f"{prefix}/valid_fraction"] = count / total
            diagnostics[f"{prefix}/abs_error_{unit}"] = means[1]
            diagnostics[f"{prefix}/rms_error_{unit}"] = means[2].sqrt()
            diagnostics[f"{prefix}/duration_s"] = means[3]
        landings, inside, rho, plans, invalid, missed = self._foothold_sums[ids].sum(0)
        diagnostics.update({
            "foothold/landings": landings,
            "foothold/inside_rate": inside / landings.clamp_min(1.),
            "foothold/rho_mean": rho / landings.clamp_min(1.),
            "foothold/plans": plans,
            "foothold/invalid_rate": invalid / plans.clamp_min(1.),
            "foothold/missed_swings": missed,
        })
        # Transfer once when logging; avoid a GPU synchronization per scalar.
        values = torch.stack(list(diagnostics.values())).detach().cpu().tolist()
        return dict(zip(diagnostics, values))

    def reset(self, env_ids=None) -> dict[str, float]:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        if len(env_ids) == 0:
            return {}
        diagnostics = self.diagnostic_means(env_ids)
        for buffer in (self._mode_counts, self._mode_sums, self._height_counts, self._height_sums, self._wheel_sums, self._cycle_sums, self._foothold_sums):
            buffer[env_ids] = 0.0
        extras = super().reset(env_ids)
        extras.update(diagnostics)
        return extras
