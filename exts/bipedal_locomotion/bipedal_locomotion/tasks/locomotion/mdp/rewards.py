"""This sub-module contains the reward functions that can be used for LimX Point Foot's locomotion task.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import math
import numpy as np
import torch
from torch import distributions
from typing import TYPE_CHECKING, Optional

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
import isaaclab.utils.math as math_utils

from .reward_math import (
    GaitCycleIntegral,
    FootholdRegionTracker,
    MissedSwingTracker,
    ZeroCommandHoldTracker,
    terrain_obstacle_relief,
    gait_contact_targets,
    all_support_confidence,
    any_support_confidence,
    base_height_plane_error_l2,
    base_height_plane_tracking_exp,
    bounded_acceleration_tracking_penalty,
    charbonnier_acceleration_tracking_penalty,
    contact_confidence_from_force_history,
    contact_force_excess_l2,
    differential_wheel_rolling_error_l2,
    fit_height_plane,
    height_command_transition_scale,
    horizontal_neutral_penalty,
    landing_impact_l2,
    lateral_foot_width_penalty,
    mean_support_confidence,
    phase_swing_clearance_exp,
    swing_min_clearance_shortfall,
    rolling_contact_slip_l2,
    terrain_relative_feet_regulation,
    terrain_relative_landing_velocity_l2,
    terrain_orientation_penalty,
    terrain_swing_peak_height,
    wheel_clearance_confidence,
    wheel_speed_huber,
    wheel_target_deadband_l2,
    wheel_target_symmetry_penalty,
    zero_command_yaw_rate_penalty,
    zero_command_wheel_target_penalty,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg

def stay_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for staying alive."""
    return torch.ones(env.num_envs, device=env.device)


def undesired_contact_force_l2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 10.0,
    force_scale: float = 100.0,
    max_normalized_excess: float = 3.0,
) -> torch.Tensor:
    """Penalize the load carried by selected undesired-contact bodies."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_history_w = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    return contact_force_excess_l2(
        force_history_w,
        force_threshold,
        force_scale,
        max_normalized_excess,
    )


def foot_landing_vel(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        foot_radius: float,
        about_landing_threshold: float,
) -> torch.Tensor:
    """Penalize high foot landing velocities"""
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    z_vels = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2]
    contacts = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > 0.1

    foot_heights = torch.clip(
    asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - foot_radius, 0, 1
    )  # TODO: change to the height relative to the vertical projection of the terrain

    about_to_land = (foot_heights < about_landing_threshold) & (~contacts) & (z_vels < 0.0)
    landing_z_vels = torch.where(about_to_land, z_vels, torch.zeros_like(z_vels))
    reward = torch.sum(torch.square(landing_z_vels), dim=1)
    return reward

def joint_powers_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint powers on the articulation using L1-kernel"""

    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(
        asset.data.applied_torque[:, asset_cfg.joint_ids] * asset.data.joint_vel[:, asset_cfg.joint_ids]
    ), dim=1)


def lin_vel_z_height_command_gated_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    min_scale: float = 0.2,
    full_penalty_gap: float = 0.01,
    reduced_penalty_gap: float = 0.04,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize vertical base velocity less while the height command is slewing."""
    asset: Articulation = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    if not hasattr(command_term, "target"):
        raise AttributeError(f"Command term {command_name!r} does not expose a sampled target.")
    scale = height_command_transition_scale(
        command_term.command[:, 0],
        command_term.target[:, 0],
        min_scale=min_scale,
        full_penalty_gap=full_penalty_gap,
        reduced_penalty_gap=reduced_penalty_gap,
    )
    return torch.square(asset.data.root_lin_vel_b[:, 2]) * scale


def no_fly(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """Reward if only one foot is in contact with the ground."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    latest_contact_forces = contact_sensor.data.net_forces_w_history[:, 0, :, 2]

    contacts = latest_contact_forces > threshold
    single_contact = torch.sum(contacts.float(), dim=1) == 1

    return 1.0 * single_contact


def unbalance_feet_air_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize if the feet air time variance exceeds the balance threshold."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    return torch.var(contact_sensor.data.last_air_time[:, sensor_cfg.body_ids], dim=-1)


def unbalance_feet_height(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize the variance of feet maximum height using sensor positions."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    feet_positions = contact_sensor.data.pos_w[:, sensor_cfg.body_ids]

    if feet_positions is None:
        return torch.zeros(env.num_envs)

    feet_heights = feet_positions[:, :, 2]
    max_feet_heights = torch.max(feet_heights, dim=-1)[0]
    height_variance = torch.var(max_feet_heights, dim=-1)
    return height_variance


# def feet_distance(
#     env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ) -> torch.Tensor:
#     """Penalize if the distance between feet is below a minimum threshold."""

#     asset: Articulation = env.scene[asset_cfg.name]

#     feet_positions = asset.data.joint_pos[sensor_cfg.body_ids]

#     if feet_positions is None:
#         return torch.zeros(env.num_envs)

#     # feet distance on x-y plane
#     feet_distance = torch.norm(feet_positions[0, :2] - feet_positions[1, :2], dim=-1)

#     return torch.clamp(0.1 - feet_distance, min=0.0)


def feet_distance(env: ManagerBasedRLEnv,
                  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                  feet_links_name: list[str]=["foot_[RL]_Link"],
                  min_feet_distance: float = 0.1,
                  max_feet_distance: float = 1.0,)-> torch.Tensor:
    # Penalize base height away from target
    asset: Articulation = env.scene[asset_cfg.name]
    feet_links_idx = asset.find_bodies(feet_links_name)[0]
    feet_pos = asset.data.body_link_pos_w[:,feet_links_idx]
    # feet distance on x-y plane
    feet_distance = torch.norm(feet_pos[:, 0, :2] - feet_pos[:, 1, :2], dim=-1)
    reward = torch.clip(min_feet_distance - feet_distance, 0, 1)
    reward += torch.clip(feet_distance - max_feet_distance, 0, 1)
    return reward

def foot_lateral_width_huber(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    min_width: float = 0.30,
    max_width: float = 0.38,
    error_scale: float = 0.05,
) -> torch.Tensor:
    """Constrain left/right wheel-center width without restricting step length."""
    asset: Articulation = env.scene[asset_cfg.name]
    return lateral_foot_width_penalty(
        asset.data.body_link_pos_w[:, asset_cfg.body_ids],
        asset.data.root_link_quat_w,
        min_width,
        max_width,
        error_scale,
    )


def nominal_foot_position(env: ManagerBasedRLEnv, command_name: str,
                          base_height_target: float,
                           asset_cfg: SceneEntityCfg, std: float) -> torch.Tensor:
    """Compute the nominal foot position"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    base_quat = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, 2, -1)
    # assert (compute_rotation_distance(asset.data.root_com_quat_w, asset.data.root_link_quat_w) < 0.1).all()
    base_pos = asset.data.root_link_state_w[:, :3].unsqueeze(1).expand(-1, 2, -1)
    feet_pos_b = math_utils.quat_rotate_inverse(
        base_quat,
        feet_pos_w - base_pos,
    )
    feet_center_b = torch.mean(feet_pos_b[:, :, :3], dim=1)
    base_height_error = torch.abs((feet_center_b[:, 2] - env._foot_radius + base_height_target))

    reward = torch.exp(-base_height_error / std**2)
    return reward

def leg_symmetry(env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),) -> torch.Tensor:
    """Reward regulate abad joint position."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    base_quat = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, 2, -1)
    # assert (compute_rotation_distance(asset.data.root_com_quat_w, asset.data.root_link_quat_w) < 0.1).all()
    base_pos = asset.data.root_link_state_w[:, :3].unsqueeze(1).expand(-1, 2, -1)
    feet_pos_b = math_utils.quat_rotate_inverse(
        base_quat,
        feet_pos_w - base_pos,
    )
    leg_symmetry_err = torch.abs(feet_pos_b[:, 0, 1]) - torch.abs(feet_pos_b[:, 1, 1])

    return torch.exp(-leg_symmetry_err ** 2 / std**2)

def same_feet_x_position(env: ManagerBasedRLEnv,
                  asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward regulate abad joint position."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    base_quat = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, 2, -1)
    # assert (compute_rotation_distance(asset.data.root_com_quat_w, asset.data.root_link_quat_w) < 0.1).all()
    base_pos = asset.data.root_link_state_w[:, :3].unsqueeze(1).expand(-1, 2, -1)
    feet_pos_b = math_utils.quat_rotate_inverse(
        base_quat,
        feet_pos_w - base_pos,
    )
    feet_x_distance = torch.abs(feet_pos_b[:, 0, 0] - feet_pos_b[:, 1, 0])
    # return torch.exp(-feet_x_distance / 0.2)
    return feet_x_distance

def keep_ankle_pitch_zero_in_air(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor", body_names=["ankle_[LR]_Link"]),
    force_threshold: float = 2.0,
    pitch_scale: float = 0.2
) -> torch.Tensor:
    """Reward for keeping ankle pitch angle close to zero when foot is in the air.
    
    Args:
        env: The environment object.
        asset_cfg: Configuration for the robot asset containing DOF positions.
        sensor_cfg: Configuration for the contact force sensor.
        force_threshold: Threshold value for contact detection (in Newtons).
        pitch_scale: Scaling factor for the exponential reward.
        
    Returns:
        The computed reward tensor.
    """
    asset = env.scene[asset_cfg.name]
    contact_forces_history = env.scene.sensors[sensor_cfg.name].data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    current_contact = torch.norm(contact_forces_history[:, -1], dim=-1) > force_threshold
    last_contact = torch.norm(contact_forces_history[:, -2], dim=-1) > force_threshold
    contact_filt = torch.logical_or(current_contact, last_contact)
    ankle_pitch_left = torch.abs(asset.data.joint_pos[:, 3]) * ~contact_filt[:, 0]
    ankle_pitch_right = torch.abs(asset.data.joint_pos[:, 7]) * ~contact_filt[:, 1]
    weighted_ankle_pitch = ankle_pitch_left + ankle_pitch_right
    return torch.exp(-weighted_ankle_pitch / pitch_scale)

def no_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Penalize if both feet are not in contact with the ground.
    """

    # Access the contact sensor
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Get the latest contact forces in the z direction (upward direction)
    latest_contact_forces = contact_sensor.data.net_forces_w_history[:, 0, :, 2]  # shape: (env_num, 2)

    # Determine if each foot is in contact
    contacts = latest_contact_forces > 1.0  # Returns a boolean tensor where True indicates contact

    return (torch.sum(contacts.float(), dim=1) == 0).float()


def stand_still(
    env, lin_threshold: float = 0.05, ang_threshold: float = 0.05, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    penalizing linear and angular motion when command velocities are near zero.
    """

    asset = env.scene[asset_cfg.name]
    base_lin_vel = asset.data.root_lin_vel_w[:, :2]
    base_ang_vel = asset.data.root_ang_vel_w[:, -1]

    commands = env.command_manager.get_command("base_velocity")

    lin_commands = commands[:, :2]
    ang_commands = commands[:, 2]

    reward_lin = torch.sum(
        torch.abs(base_lin_vel) * (torch.norm(lin_commands, dim=1, keepdim=True) < lin_threshold), dim=-1
    )

    reward_ang = torch.abs(base_ang_vel) * (torch.abs(ang_commands) < ang_threshold)

    total_reward = reward_lin + reward_ang
    return total_reward


# def feet_regulation(
#     env: ManagerBasedRLEnv,
#     sensor_cfg: SceneEntityCfg,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     desired_body_height: float = 0.65,
# ) -> torch.Tensor:
#     """Penalize if the feet are not in contact with the ground.

#     Args:
#         env: The environment object.
#         sensor_cfg: The configuration of the contact sensor.
#         desired_body_height: The desired body height used for normalization.

#     Returns:
#         A tensor representing the feet regulation penalty for each environment.
#     """

#     asset: Articulation = env.scene[asset_cfg.name]

#     feet_positions_z = asset.data.joint_pos[sensor_cfg.body_ids, 2]

#     feet_vel_xy = asset.data.joint_vel[sensor_cfg.body_ids, :2]

#     vel_norms_xy = torch.norm(feet_vel_xy, dim=-1)

#     exp_term = torch.exp(-feet_positions_z / (0.025 * desired_body_height))

#     r_fr = torch.sum(vel_norms_xy**2 * exp_term, dim=-1)

#     return r_fr

def feet_regulation(env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    foot_radius: float,
    base_height_target: float,
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    feet_height = torch.clip(
        asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - foot_radius, 0, 1
    )  # TODO: change to the height relative to the vertical projection of the terrain
    feet_vel_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]

    height_scale = torch.exp(-feet_height / base_height_target)
    reward = torch.sum(height_scale * torch.square(torch.norm(feet_vel_xy, dim=-1)), dim=1)
    return reward


def base_height_rough_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        Currently, it assumes a flat terrain, i.e. the target height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    height = asset.data.root_pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[:, :, 2]
    # sensor.data.ray_hits_w can be inf, so we clip it to avoid NaN
    height = torch.nan_to_num(height, nan=target_height, posinf=target_height, neginf=target_height)
    return torch.square(height.mean(dim=1) - target_height)


def base_com_height(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_target_height = target_height + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    return torch.abs(asset.data.root_pos_w[:, 2] - adjusted_target_height)


class GaitReward(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)

        self.sensor_cfg = cfg.params["sensor_cfg"]
        self.asset_cfg = cfg.params["asset_cfg"]

        # extract the used quantities (to enable type-hinting)
        self.contact_sensor: ContactSensor = env.scene.sensors[self.sensor_cfg.name]
        self.asset: Articulation = env.scene[self.asset_cfg.name]

        # Store configuration parameters
        self.force_scale = float(cfg.params["tracking_contacts_shaped_force"])
        self.vel_scale = float(cfg.params["tracking_contacts_shaped_vel"])
        self.force_sigma = cfg.params["gait_force_sigma"]
        self.force_kernel = cfg.params.get("force_kernel", "exponential")
        self.force_reference = float(cfg.params.get("force_reference", 100.0))
        if self.force_kernel not in ("exponential", "huber"):
            raise ValueError("force_kernel must be exponential or huber.")
        if not math.isfinite(self.force_reference) or self.force_reference <= 0.0:
            raise ValueError("force_reference must be finite and positive (N).")
        if self.force_kernel == "huber" and self.force_scale >= 0.0:
            raise ValueError("Huber gait unloading requires a negative tracking_contacts_shaped_force.")
        self.vel_sigma = cfg.params["gait_vel_sigma"]
        self.kappa_gait_probs = cfg.params["kappa_gait_probs"]
        self.command_name = cfg.params["command_name"]
        self.dt = env.step_dt

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        tracking_contacts_shaped_force,
        tracking_contacts_shaped_vel,
        gait_force_sigma,
        gait_vel_sigma,
        kappa_gait_probs,
        command_name,
        sensor_cfg,
        asset_cfg,
        force_kernel="exponential",
        force_reference=100.0,
    ) -> torch.Tensor:
        """Compute the reward.

        The reward combines force-based and velocity-based terms to encourage desired gait patterns.

        Args:
            env: The RL environment instance.

        Returns:
            The reward value.
        """

        gait_params = env.command_manager.get_command(self.command_name)

        # Update contact targets
        desired_contact_states = self.compute_contact_targets(gait_params)

        # Force-based reward
        foot_forces = torch.norm(self.contact_sensor.data.net_forces_w[:, self.sensor_cfg.body_ids], dim=-1)
        force_reward = self._compute_force_reward(foot_forces, desired_contact_states)

        # Velocity-based reward
        foot_velocities = torch.norm(self.asset.data.body_lin_vel_w[:, self.asset_cfg.body_ids], dim=-1)
        velocity_reward = self._compute_velocity_reward(foot_velocities, desired_contact_states)

        # Combine rewards
        total_reward = force_reward + velocity_reward
        return total_reward

    def compute_contact_targets(self, gait_params):
        return gait_contact_targets(
            self._env.command_manager.get_term(self.command_name).phase,
            gait_params[:, 1], gait_params[:, 2], self.kappa_gait_probs,
        )

    def _compute_force_reward(self, forces: torch.Tensor, desired_contacts: torch.Tensor) -> torch.Tensor:
        """Compute force-based reward component."""
        if self.force_kernel == "huber":
            # Foot: distinguish partial unloading at normal stance loads.
            # Planned stance force is not penalized; the opposite foot may
            # take the load. Actual lift height is handled by clearance.
            normalized_force = forces.abs() / self.force_reference
            cost = torch.where(
                normalized_force <= 1.0,
                0.5 * normalized_force.square(),
                normalized_force - 0.5,
            )
            return self.force_scale * ((1.0 - desired_contacts) * cost).mean(dim=1)
        reward = torch.zeros_like(forces[:, 0])
        if self.force_scale < 0:  # Negative scale means penalize unwanted contact
            for i in range(forces.shape[1]):
                reward += (1 - desired_contacts[:, i]) * (1 - torch.exp(-forces[:, i] ** 2 / self.force_sigma))
        else:  # Positive scale means reward desired contact
            for i in range(forces.shape[1]):
                reward += (1 - desired_contacts[:, i]) * torch.exp(-forces[:, i] ** 2 / self.force_sigma)

        return (reward / forces.shape[1]) * self.force_scale

    def _compute_velocity_reward(self, velocities: torch.Tensor, desired_contacts: torch.Tensor) -> torch.Tensor:
        """Compute velocity-based reward component."""
        reward = torch.zeros_like(velocities[:, 0])
        if self.vel_scale < 0:  # Negative scale means penalize movement during contact
            for i in range(velocities.shape[1]):
                reward += desired_contacts[:, i] * (1 - torch.exp(-velocities[:, i] ** 2 / self.vel_sigma))
        else:  # Positive scale means reward movement during swing
            for i in range(velocities.shape[1]):
                reward += desired_contacts[:, i] * torch.exp(-velocities[:, i] ** 2 / self.vel_sigma)

        return (reward / velocities.shape[1]) * self.vel_scale


def leg_action_rate_l2(env: ManagerBasedRLEnv, action_dim: int = 6) -> torch.Tensor:
    """Foot leg action changes only; ignored wheel outputs do not incur cost."""
    return (env.action_manager.action[:, :action_dim] -
            env.action_manager.prev_action[:, :action_dim]).square().sum(1)


def planned_support_contact_loss(
    env: ManagerBasedRLEnv, command_name: str,
    contact_sensor_names: tuple[str, str], kappa_gait_probs: float = 0.05,
    force_off: float = 5.0, force_on: float = 10.0,
) -> torch.Tensor:
    """Require the planned stance wheel to maintain ground support in all modes."""
    gait = env.command_manager.get_command(command_name)
    phase = env.command_manager.get_term(command_name).phase
    planned = gait_contact_targets(phase, gait[:, 1], gait[:, 2], kappa_gait_probs)
    actual = _wheel_ground_force_confidence(env, contact_sensor_names, force_off, force_on)
    return (planned * (1. - actual).square()).mean(1)


class ActionSmoothnessPenalty(ManagerTermBase):
    """
    A reward term for penalizing large instantaneous changes in the network action output.
    This penalty encourages smoother actions over time.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward term.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.dt = env.step_dt
        self.prev_prev_action = None
        self.prev_action = None
        # self.__name__ = "action_smoothness_penalty"

    def __call__(self, env: ManagerBasedRLEnv, action_dim: int | None = None) -> torch.Tensor:
        """Compute the action smoothness penalty.

        Args:
            env: The RL environment instance.

        Returns:
            The penalty value based on the action smoothness.
        """
        # Get the current action from the environment's action manager
        current_action = env.action_manager.action[:, :action_dim].clone()

        # If this is the first call, initialize the previous actions
        if self.prev_action is None:
            self.prev_action = current_action
            return torch.zeros(current_action.shape[0], device=current_action.device)

        if self.prev_prev_action is None:
            self.prev_prev_action = self.prev_action
            self.prev_action = current_action
            return torch.zeros(current_action.shape[0], device=current_action.device)

        # Compute the smoothness penalty
        penalty = torch.sum(torch.square(current_action - 2 * self.prev_action + self.prev_prev_action), dim=1)

        # Update the previous actions for the next call
        self.prev_prev_action = self.prev_action
        self.prev_action = current_action

        # Apply a condition to ignore penalty during the first few episodes
        startup_env_mask = env.episode_length_buf < 3
        penalty[startup_env_mask] = 0

        # Return the penalty scaled by the configured weight
        return penalty


class GaitCycleMeanTrackingPenalty(ManagerTermBase):
    """Signed tracking error averaged over one complete sliding gait cycle."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        params = cfg.params
        if params["component"] not in ("vx", "vy", "yaw", "height", "roll", "pitch"):
            raise ValueError("Unknown cycle tracking component.")
        scale = params["error_scale"]
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("Cycle mean error_scale must be finite and positive.")
        gait = env.command_manager.get_term(params.get("gait_command_name", "gait_command"))
        if not gait.cfg.continuous_phase:
            raise ValueError("Cycle mean tracking requires the continuous gait clock.")
        f_min, f_max = gait.cfg.ranges.frequencies
        if not (math.isfinite(f_min) and math.isfinite(f_max)
                and 0.0 < f_min <= f_max and f_max * env.step_dt < 1.0):
            raise ValueError("Expected positive gait frequencies below one cycle per control step.")
        capacity = math.ceil(1.0 / (f_min * env.step_dt)) + 2
        self.window = GaitCycleIntegral(env.num_envs, capacity, env.device)
        self._previous_phase = torch.zeros(env.num_envs, device=env.device)
        self._previous_heading = torch.zeros_like(self._previous_phase)
        self._initialized = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self._cost = torch.zeros_like(self._previous_phase)
        self._last_step = None

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.window.reset(ids)
        self._initialized[ids] = False
        self._previous_phase[ids] = 0.0
        self._previous_heading[ids] = 0.0
        self._cost[ids] = 0.0

    def __call__(
        self, env, command_name: str, component: str, error_scale: float,
        asset_cfg: SceneEntityCfg, gait_command_name: str = "gait_command",
        height_command_name: str = "body_height",
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    ) -> torch.Tensor:
        step = env.common_step_counter
        if step == self._last_step:
            return self._cost
        if self._last_step is not None and step != self._last_step + 1:
            self.reset()
        self._last_step = step
        phase = env.command_manager.get_term(gait_command_name).phase
        phase_span = torch.remainder(phase - self._previous_phase, 1.0)
        asset = env.scene[asset_cfg.name]
        command = env.command_manager.get_command(command_name)
        valid = torch.isfinite(phase)
        if component in ("vx", "vy"):
            index = 0 if component == "vx" else 1
            increment = (asset.data.root_lin_vel_b[:, index] - command[:, index]) * env.step_dt
        elif component == "yaw":
            qw, qx, qy, qz = asset.data.root_link_quat_w.unbind(-1)
            x = 1.0 - 2.0 * (qy.square() + qz.square())
            y = 2.0 * (qx * qy + qw * qz)
            heading = torch.atan2(y, x)
            delta = heading - self._previous_heading
            increment = torch.atan2(torch.sin(delta), torch.cos(delta)) - command[:, 2] * env.step_dt
            self._previous_heading[:] = heading
            valid = valid & torch.isfinite(heading) & (x.square() + y.square() > 1.e-8)
        else:
            centroid, normal, plane_valid = _terrain_plane(env.scene.sensors[sensor_cfg.name])
            valid = valid & plane_valid
            if component == "height":
                actual = ((asset.data.root_link_pos_w - centroid) * normal).sum(-1)
                error = actual - env.command_manager.get_command(height_command_name)[:, 0]
            else:
                # Signed tilt relative to the local terrain normal; no independent
                # roll/pitch command exists in this task. Both vanish when aligned.
                normal_b = math_utils.quat_apply_inverse(asset.data.root_link_quat_w, normal)
                nx, ny, nz = normal_b.unbind(-1)
                error = (torch.atan2(ny, nz) if component == "roll" else
                         torch.atan2(-nx, torch.sqrt(ny.square() + nz.square())))
            increment = error * env.step_dt
        valid = valid & torch.isfinite(increment)
        integral = self.window.update(increment, phase_span, valid & self._initialized, env.step_dt)
        mean_error = integral / self.window.duration.clamp_min(1.e-8)
        error = mean_error.abs() / error_scale
        self._cost[:] = torch.where(error <= 1.0, 0.5 * error.square(), error - 0.5)
        self._previous_phase[:] = phase
        self._initialized[:] = valid
        return self._cost


class BaseVelocityAccelerationPenalty(ManagerTermBase):
    """Base velocity-change penalty with configurable tracking and support gates.

    Defaults preserve Wheel behavior. Foot XY uses world-frame differences so
    rotation of the base frame alone cannot be mistaken for acceleration.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._previous_velocity = torch.zeros(env.num_envs, 2, device=env.device)
        self._steps_since_reset = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._previous_velocity[env_ids] = 0.0
        self._steps_since_reset[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        component: str,
        tracking_std: float,
        acceleration_scale: float,
        asset_cfg: SceneEntityCfg,
        contact_sensor_names: tuple[str, str],
        terrain_sensor_names: tuple[str, str],
        kernel: str = "bounded",
        wheel_radius: float = 0.128,
        force_off: float = 5.0,
        force_on: float = 10.0,
        geometry_tolerance_on: float = 0.02,
        geometry_tolerance_off: float = 0.035,
        linear_velocity_frame: str = "body",
        min_tracking_gate: float = 0.0,
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        command = env.command_manager.get_command(command_name)
        if linear_velocity_frame not in ("body", "world"):
            raise ValueError("linear_velocity_frame must be 'body' or 'world'.")
        if component == "xy":
            # Commands are body-frame velocities even when acceleration is
            # evaluated in world coordinates.
            tracking_error_squared = torch.sum(
                torch.square(command[:, :2] - asset.data.root_lin_vel_b[:, :2]), dim=1
            )
            current_velocity = (
                asset.data.root_lin_vel_w[:, :2] if linear_velocity_frame == "world"
                else asset.data.root_lin_vel_b[:, :2]
            )
        elif component == "yaw":
            current_velocity = asset.data.root_ang_vel_b[:, 2:3]
            tracking_error_squared = torch.square(command[:, 2] - current_velocity[:, 0])
        else:
            raise ValueError(f"component must be 'xy' or 'yaw', got {component!r}.")

        acceleration = (current_velocity - self._previous_velocity[:, : current_velocity.shape[1]]) / env.step_dt
        acceleration_squared = torch.sum(torch.square(acceleration), dim=1)
        self._previous_velocity[:, : current_velocity.shape[1]] = current_velocity

        contact = _wheel_ground_contact_confidence(
            env,
            asset_cfg,
            contact_sensor_names,
            terrain_sensor_names,
            wheel_radius,
            force_off,
            force_on,
            geometry_tolerance_on,
            geometry_tolerance_off,
        )
        support_confidence = any_support_confidence(contact)
        if kernel == "bounded":
            penalty = bounded_acceleration_tracking_penalty(
                acceleration_squared,
                tracking_error_squared,
                support_confidence,
                acceleration_scale=acceleration_scale,
                tracking_std=tracking_std,
                min_tracking_gate=min_tracking_gate,
            )
        elif kernel == "charbonnier":
            penalty = charbonnier_acceleration_tracking_penalty(
                acceleration_squared,
                tracking_error_squared,
                support_confidence,
                acceleration_scale=acceleration_scale,
                tracking_std=tracking_std,
                min_tracking_gate=min_tracking_gate,
            )
        else:
            raise ValueError(f"kernel must be 'bounded' or 'charbonnier', got {kernel!r}.")
        # The finite difference is invalid immediately after a reset, and the
        # second step is also suppressed to avoid reset-settling transients.
        penalty = torch.where(self._steps_since_reset >= 2, penalty, torch.zeros_like(penalty))
        self._steps_since_reset += 1
        return penalty


def body_height_command_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Track commanded base height relative to the local terrain using an L2 error."""
    asset: RigidObject = env.scene[asset_cfg.name]
    target_height = env.command_manager.get_command(command_name)[:, 0]
    if sensor_cfg is None:
        ground_height = torch.zeros_like(target_height)
    else:
        sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
        ray_height = sensor.data.ray_hits_w[..., 2]
        ray_height = torch.where(torch.isfinite(ray_height), ray_height, torch.nan)
        ground_height = torch.nanmedian(ray_height, dim=1).values
        ground_height = torch.nan_to_num(ground_height, nan=0.0, posinf=0.0, neginf=0.0)
    actual_height = asset.data.root_pos_w[:, 2] - ground_height
    return torch.square(actual_height - target_height)


def _contact_force_history(sensor: ContactSensor, body_ids) -> torch.Tensor:
    """Return contact-force history with a history dimension on all supported IsaacLab versions."""
    history = sensor.data.net_forces_w_history
    if history is None:
        history = sensor.data.net_forces_w.unsqueeze(1)
    return history[:, :, body_ids]


def _wheel_contact_confidence(
    sensor: ContactSensor,
    body_ids,
    force_off: float,
    force_on: float,
) -> torch.Tensor:
    return contact_confidence_from_force_history(
        _contact_force_history(sensor, body_ids),
        force_off=force_off,
        force_on=force_on,
    )


def _terrain_plane(sensor: RayCaster) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return fit_height_plane(sensor.data.ray_hits_w)


def _filtered_contact_force_history(sensor: ContactSensor) -> torch.Tensor:
    """Return ground-filtered contact-force history shaped ``(N, T, 1, 3)``."""
    history = sensor.data.force_matrix_w_history
    if history is None:
        current = sensor.data.force_matrix_w
        if current is None:
            raise RuntimeError(
                f"Contact sensor '{sensor.cfg.prim_path}' has no filtered force matrix. "
                "Configure filter_prim_paths_expr for the terrain mesh."
            )
        history = current.unsqueeze(1)
    if history.ndim != 5 or history.shape[-1] != 3:
        raise RuntimeError(f"Unexpected filtered contact history shape: {tuple(history.shape)}")
    return torch.sum(history, dim=3)


def _wheel_ground_force_confidence(
    env: ManagerBasedRLEnv,
    contact_sensor_names: tuple[str, str],
    force_off: float,
    force_on: float,
) -> torch.Tensor:
    """Return continuous per-wheel confidence from ground-filtered force history."""
    if len(contact_sensor_names) != 2:
        raise ValueError("TRON1A wheel support requires exactly two ground-contact sensors.")
    force_confidences = []
    for contact_name in contact_sensor_names:
        contact_sensor: ContactSensor = env.scene.sensors[contact_name]
        force_confidence = contact_confidence_from_force_history(
            _filtered_contact_force_history(contact_sensor),
            force_off=force_off,
            force_on=force_on,
        )
        if force_confidence.shape[1] != 1:
            raise RuntimeError(
                f"Ground-contact sensor '{contact_name}' must resolve exactly one wheel body, "
                f"got {force_confidence.shape[1]}."
            )
        force_confidences.append(force_confidence[:, 0])
    return torch.stack(force_confidences, dim=1)


def _wheel_ground_contact_state(
    env: ManagerBasedRLEnv,
    contact_sensor_names: tuple[str, str],
    force_threshold: float,
) -> torch.Tensor:
    """Return current ground-filtered contact state for the two wheel links."""
    if len(contact_sensor_names) != 2:
        raise ValueError("TRON1A wheel support requires exactly two ground-contact sensors.")
    if force_threshold < 0.0:
        raise ValueError(f"force_threshold must be non-negative, got {force_threshold}.")

    contact_states = []
    for contact_name in contact_sensor_names:
        contact_sensor: ContactSensor = env.scene.sensors[contact_name]
        force_matrix = contact_sensor.data.force_matrix_w
        if force_matrix is None:
            history = contact_sensor.data.force_matrix_w_history
            if history is None:
                raise RuntimeError(
                    f"Contact sensor '{contact_name}' has no filtered force matrix. "
                    "Configure filter_prim_paths_expr for the terrain mesh."
                )
            force_matrix = history[:, -1]
        if force_matrix.ndim != 4 or force_matrix.shape[-1] != 3:
            raise RuntimeError(f"Unexpected filtered contact force shape: {tuple(force_matrix.shape)}")
        ground_force = torch.sum(force_matrix, dim=2)
        if ground_force.shape[1] != 1:
            raise RuntimeError(
                f"Ground-contact sensor '{contact_name}' must resolve exactly one wheel body, "
                f"got {ground_force.shape[1]}."
            )
        contact_states.append(torch.linalg.vector_norm(ground_force[:, 0], dim=-1) > force_threshold)
    return torch.stack(contact_states, dim=1)


def _wheel_terrain_planes(
    env: ManagerBasedRLEnv,
    terrain_sensor_names: tuple[str, str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit one local terrain plane below each wheel."""
    if len(terrain_sensor_names) != 2:
        raise ValueError("TRON1A wheel terrain geometry requires exactly two terrain sensors.")

    plane_centroids = []
    plane_normals = []
    plane_validities = []
    for terrain_name in terrain_sensor_names:
        terrain_sensor: RayCaster = env.scene.sensors[terrain_name]
        centroid, normal, valid = _terrain_plane(terrain_sensor)
        plane_centroids.append(centroid)
        plane_normals.append(normal)
        plane_validities.append(valid)
    return (
        torch.stack(plane_centroids, dim=1),
        torch.stack(plane_normals, dim=1),
        torch.stack(plane_validities, dim=1),
    )


def _wheel_ground_contact_confidence(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    contact_sensor_names: tuple[str, str],
    terrain_sensor_names: tuple[str, str],
    wheel_radius: float,
    force_off: float,
    force_on: float,
    geometry_tolerance_on: float,
    geometry_tolerance_off: float,
) -> torch.Tensor:
    """Combine filtered support force and wheel-local ground clearance for both wheels."""
    _, _, combined_confidence = wheel_ground_contact_confidence_components(
        env,
        asset_cfg,
        contact_sensor_names,
        terrain_sensor_names,
        wheel_radius,
        force_off,
        force_on,
        geometry_tolerance_on,
        geometry_tolerance_off,
    )
    return combined_confidence


def wheel_ground_contact_confidence_components(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    contact_sensor_names: tuple[str, str],
    terrain_sensor_names: tuple[str, str],
    wheel_radius: float,
    force_off: float,
    force_on: float,
    geometry_tolerance_on: float,
    geometry_tolerance_off: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-wheel force, geometry, and combined ground confidences."""
    if len(terrain_sensor_names) != 2:
        raise ValueError("TRON1A wheel contact requires exactly two terrain sensors.")

    asset: Articulation = env.scene[asset_cfg.name]
    wheel_positions_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    if wheel_positions_w.shape[1] != 2:
        raise ValueError("TRON1A wheel contact expects exactly two wheel bodies.")

    force_confidence = _wheel_ground_force_confidence(
        env,
        contact_sensor_names,
        force_off,
        force_on,
    )
    plane_centroids, plane_normals, plane_validities = _wheel_terrain_planes(env, terrain_sensor_names)

    geometry_confidence = wheel_clearance_confidence(
        wheel_positions_w,
        plane_centroids,
        plane_normals,
        plane_validities,
        wheel_radius=wheel_radius,
        tolerance_on=geometry_tolerance_on,
        tolerance_off=geometry_tolerance_off,
    )
    return force_confidence, geometry_confidence, force_confidence * geometry_confidence


def wheel_terrain_feet_regulation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    terrain_sensor_names: tuple[str, str],
    wheel_radius: float = 0.128,
    height_scale: float = 0.65,
) -> torch.Tensor:
    """Apply the PF foot-regulation objective relative to each wheel's local terrain."""
    asset: Articulation = env.scene[asset_cfg.name]
    wheel_positions_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    wheel_velocities_w = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids]
    if wheel_positions_w.shape[1] != 2:
        raise ValueError("TRON1A foot regulation expects exactly two wheel bodies.")
    plane_centroids, plane_normals, plane_validities = _wheel_terrain_planes(env, terrain_sensor_names)
    return terrain_relative_feet_regulation(
        wheel_positions_w,
        wheel_velocities_w,
        plane_centroids,
        plane_normals,
        plane_validities,
        foot_radius=wheel_radius,
        height_scale=height_scale,
    )


def lin_vel_xy_tracking_error_huber(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    error_scale: float = 0.4,
) -> torch.Tensor:
    """Instantaneous body-frame planar tracking cost with a linear large-error tail."""
    if not math.isfinite(error_scale) or error_scale <= 0.0:
        raise ValueError("error_scale must be finite and positive.")
    command = env.command_manager.get_command(command_name)[:, :2]
    actual = env.scene[asset_cfg.name].data.root_lin_vel_b[:, :2]
    error = torch.linalg.vector_norm(command - actual, dim=1) / error_scale
    return torch.where(error <= 1.0, 0.5 * error.square(), error - 0.5)


def ang_vel_z_tracking_error_huber(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    error_scale: float = 0.5,
) -> torch.Tensor:
    """Retain a yaw tracking cost beyond the narrow exponential reward kernel."""
    if error_scale <= 0.0:
        raise ValueError("error_scale must be positive.")
    command = env.command_manager.get_command(command_name)[:, 2]
    actual = env.scene[asset_cfg.name].data.root_ang_vel_b[:, 2]
    error = (command - actual).abs() / error_scale
    return torch.where(error <= 1.0, 0.5 * error.square(), error - 0.5)


class TerrainAdaptiveSwingClearanceExp(ManagerTermBase):
    """Always-active exponential Foot swing reward driven by the full terrain scan's Z range.

    The peak follows live scans with fast rise and slower decay. It is not
    latched at takeoff, controlled by body height, or gated by speed commands.
    Local wheel planes still define each wheel's measured rim clearance.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._min_peak_height = float(cfg.params.get("min_peak_height", 0.04))
        self.peak_height = torch.full((env.num_envs,), self._min_peak_height, device=env.device)
        self.raw_peak_height = self.peak_height.clone()
        self.scan_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._initialized = torch.zeros_like(self.scan_valid)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.peak_height[env_ids] = self._min_peak_height
        self.raw_peak_height[env_ids] = self._min_peak_height
        self.scan_valid[env_ids] = False
        self._initialized[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        contact_sensor_names: tuple[str, str],
        terrain_sensor_names: tuple[str, str],
        height_sensor_cfg: SceneEntityCfg,
        gait_command_name: str = "gait_command",
        wheel_radius: float = 0.128,
        min_peak_height: float = 0.04,
        max_peak_height: float = 0.10,
        rise_time_constant: float = 0.04,
        fall_time_constant: float = 0.15,
        std: float = 0.025,
        force_off: float = 5.0,
        force_on: float = 10.0,
        geometry_tolerance_on: float = 0.02,
        geometry_tolerance_off: float = 0.06,
    ) -> torch.Tensor:
        if rise_time_constant <= 0.0 or fall_time_constant <= 0.0:
            raise ValueError("Swing height filter time constants must be positive.")
        hits = env.scene.sensors[height_sensor_cfg.name].data.ray_hits_w
        raw_peak, scan_valid = terrain_swing_peak_height(hits, min_peak_height, max_peak_height)
        alpha = torch.where(
            raw_peak > self.peak_height,
            1.0 - math.exp(-env.step_dt / rise_time_constant),
            1.0 - math.exp(-env.step_dt / fall_time_constant),
        )
        filtered = self.peak_height + alpha * (raw_peak - self.peak_height)
        # The first valid scan initializes without inheriting another episode's
        # terrain. Missing scans hold state, but do not score clearance.
        filtered = torch.where(self._initialized, filtered, raw_peak)
        self.peak_height[:] = torch.where(scan_valid, filtered, self.peak_height).clamp(
            min_peak_height, max_peak_height
        )
        self.raw_peak_height[:] = raw_peak
        self.scan_valid[:] = scan_valid
        self._initialized |= scan_valid

        asset: Articulation = env.scene[asset_cfg.name]
        positions = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
        centroids, normals, valid = _wheel_terrain_planes(env, terrain_sensor_names)
        clearance = torch.sum((positions - centroids) * normals, dim=-1) - wheel_radius
        force = _wheel_ground_force_confidence(env, contact_sensor_names, force_off, force_on)
        geometry = wheel_clearance_confidence(
            positions, centroids, normals, valid,
            wheel_radius=wheel_radius,
            tolerance_on=geometry_tolerance_on,
            tolerance_off=geometry_tolerance_off,
        )
        gait = env.command_manager.get_command(gait_command_name)
        phase = env.command_manager.get_term(gait_command_name).phase
        foot_phase = torch.stack((phase, torch.remainder(phase + gait[:, 1], 1.0)), dim=1)
        return phase_swing_clearance_exp(
            clearance, foot_phase, gait[:, 2], force * geometry,
            valid & scan_valid.unsqueeze(1), self.peak_height,
            std=std,
        )


class FootholdRegionPenalty(ManagerTermBase):
    """Event cost for landing outside a frozen, direction-aware foothold ellipse."""
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.tracker = FootholdRegionTracker(env.num_envs, env.device, **cfg.params.get("tracker_options", {}))
        self._last_step = None
        self._cost = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None):
        self.tracker.reset(env_ids)
        self._cost[slice(None) if env_ids is None else env_ids] = 0.

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg,
                 contact_sensor_names: tuple[str,str], terrain_sensor_names: tuple[str,str],
                 height_sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity",
                 gait_command_name: str = "gait_command", tracker_options: dict | None = None):
        step = env.common_step_counter
        if step == self._last_step:
            return self._cost
        if self._last_step is not None and step != self._last_step + 1:
            self.reset()
        self._last_step = step
        asset = env.scene[asset_cfg.name]
        positions = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
        quat = asset.data.root_link_quat_w
        qw,qx,qy,qz = quat.unbind(-1)
        heading = torch.atan2(2.*(qx*qy+qw*qz), 1.-2.*(qy.square()+qz.square()))
        centroid, normal, valid = _wheel_terrain_planes(env, terrain_sensor_names)
        radius = self.tracker.plan_options.get("wheel_radius", .128)
        clearance = ((positions-centroid)*normal).sum(-1)-radius
        clearance = torch.where(valid, clearance, torch.nan)
        forces = []
        for name in contact_sensor_names:
            matrix = env.scene.sensors[name].data.force_matrix_w
            if matrix is None:
                raise RuntimeError("Foothold events require current terrain-filtered wheel forces.")
            forces.append(torch.linalg.vector_norm(matrix.sum(2)[:,0], dim=-1))
        event_cost = self.tracker.update(
            asset.data.root_link_pos_w, heading, asset.data.root_lin_vel_b,
            env.command_manager.get_command(command_name),
            env.command_manager.get_command(gait_command_name),
            env.command_manager.get_term(gait_command_name).phase,
            env.scene.sensors[height_sensor_cfg.name].data.ray_hits_w,
            positions, torch.stack(forces, dim=1), clearance, env.step_dt,
        )
        # A landing is a discrete event: RewardManager multiplies by dt, so
        # cancel that multiplier to keep the configured cost per landing fixed.
        self._cost[:] = event_cost / env.step_dt
        return self._cost


class TerrainCycleSwingClearanceExp(ManagerTermBase):
    """Select a 5/10 cm peak once per full cycle; score a live cycloidal height."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.low = float(cfg.params.get("min_peak_height", .05))
        self.high = float(cfg.params.get("max_peak_height", .10))
        center = float(cfg.params.get("switch_center", .04))
        band = float(cfg.params.get("switch_band", .01))
        if not all(math.isfinite(v) for v in (self.low, self.high, center, band)) or not (
            0. < self.low < self.high and 0. < band < 2. * center
        ):
            raise ValueError("Invalid two-level swing heights or switch hysteresis.")
        self.peak_height = torch.full((env.num_envs,), self.low, device=env.device)
        self.raw_peak_height = self.peak_height.clone()
        self.terrain_metric_m = torch.zeros_like(self.peak_height)
        self.scan_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._initialized = torch.zeros_like(self.scan_valid)
        self._previous_phase = torch.zeros_like(self.peak_height)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.peak_height[ids] = self.low
        self.raw_peak_height[ids] = self.low
        self.terrain_metric_m[ids] = 0.
        self.scan_valid[ids] = False
        self._initialized[ids] = False
        self._previous_phase[ids] = 0.

    def __call__(
        self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg,
        contact_sensor_names: tuple[str, str], terrain_sensor_names: tuple[str, str],
        height_sensor_cfg: SceneEntityCfg, gait_command_name: str = "gait_command",
        wheel_radius: float = .128, min_peak_height: float = .05,
        max_peak_height: float = .10, switch_center: float = .04,
        switch_band: float = .01, std: float = .025,
        force_off: float = 5., force_on: float = 10.,
        geometry_tolerance_on: float = .02, geometry_tolerance_off: float = .06,
    ) -> torch.Tensor:
        gait = env.command_manager.get_command(gait_command_name)
        phase = env.command_manager.get_term(gait_command_name).phase
        update = (~self._initialized) | (phase < self._previous_phase)
        # Only selected environments' scans are fitted, once at reset/phase wrap.
        hits = env.scene.sensors[height_sensor_cfg.name].data.ray_hits_w[update]
        relief, valid = terrain_obstacle_relief(hits)
        previous_high = self.peak_height[update] > (min_peak_height + max_peak_height) / 2.
        high = torch.where(
            self._initialized[update],
            torch.where(previous_high, relief >= switch_center - switch_band / 2.,
                        relief > switch_center + switch_band / 2.),
            relief > switch_center,
        )
        selected = torch.where(high, max_peak_height, min_peak_height)
        # Invalid scans keep the last selected peak, but disable the height reward
        # for this cycle. A fresh episode starts at the low peak.
        self.peak_height[update] = torch.where(valid, selected, self.peak_height[update])
        self.raw_peak_height[update] = self.peak_height[update]
        self.terrain_metric_m[update] = relief
        self.scan_valid[update] = valid
        self._initialized[:] = True
        self._previous_phase[:] = phase
        asset = env.scene[asset_cfg.name]
        positions = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
        centroids, normals, valid = _wheel_terrain_planes(env, terrain_sensor_names)
        clearance = ((positions - centroids) * normals).sum(-1) - wheel_radius
        force = _wheel_ground_force_confidence(env, contact_sensor_names, force_off, force_on)
        geometry = wheel_clearance_confidence(
            positions, centroids, normals, valid, wheel_radius=wheel_radius,
            tolerance_on=geometry_tolerance_on, tolerance_off=geometry_tolerance_off,
        )
        foot_phase = torch.stack((phase, torch.remainder(phase + gait[:, 1], 1.)), dim=1)
        # H*sin²(pi*u) == H/2*(1-cos(2*pi*u)): same cycloidal vertical profile.
        return phase_swing_clearance_exp(
            clearance, foot_phase, gait[:, 2], force * geometry,
            valid & self.scan_valid[:, None], self.peak_height, std=std,
        )


def wheel_terrain_landing_velocity_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    contact_sensor_names: tuple[str, str],
    terrain_sensor_names: tuple[str, str],
    wheel_radius: float = 0.128,
    about_landing_threshold: float = 0.08,
    contact_force_threshold: float = 0.1,
    allowed_downward_speed: float = 0.0,
) -> torch.Tensor:
    """Apply the PF pre-contact landing-velocity objective along local terrain normals."""
    asset: Articulation = env.scene[asset_cfg.name]
    wheel_positions_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    wheel_velocities_w = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids]
    if wheel_positions_w.shape[1] != 2:
        raise ValueError("TRON1A landing-velocity penalty expects exactly two wheel bodies.")
    plane_centroids, plane_normals, plane_validities = _wheel_terrain_planes(env, terrain_sensor_names)
    in_contact = _wheel_ground_contact_state(env, contact_sensor_names, contact_force_threshold)
    return terrain_relative_landing_velocity_l2(
        wheel_positions_w,
        wheel_velocities_w,
        plane_centroids,
        plane_normals,
        plane_validities,
        in_contact,
        foot_radius=wheel_radius,
        about_landing_threshold=about_landing_threshold,
        allowed_downward_speed=allowed_downward_speed,
    )


def stand_still_grounded(
    env: ManagerBasedRLEnv,
    contact_sensor_names: tuple[str, str],
    force_off: float = 5.0,
    force_on: float = 10.0,
    lin_threshold: float = 0.05,
    ang_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize zero-command motion fully when either wheel provides ground support."""
    penalty = stand_still(env, lin_threshold, ang_threshold, asset_cfg)
    confidence = _wheel_ground_force_confidence(env, contact_sensor_names, force_off, force_on)
    return penalty * any_support_confidence(confidence)


def track_lin_vel_xy_exp_any_wheel_ground(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    contact_sensor_names: tuple[str, str],
    terrain_sensor_names: tuple[str, str],
    wheel_radius: float = 0.128,
    force_off: float = 5.0,
    force_on: float = 10.0,
    geometry_tolerance_on: float = 0.02,
    geometry_tolerance_off: float = 0.035,
) -> torch.Tensor:
    """Track planar velocity while either wheel has valid ground contact."""
    asset: Articulation = env.scene[asset_cfg.name]
    error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    reward = torch.exp(-error / std**2)
    contact = _wheel_ground_contact_confidence(
        env,
        asset_cfg,
        contact_sensor_names,
        terrain_sensor_names,
        wheel_radius,
        force_off,
        force_on,
        geometry_tolerance_on,
        geometry_tolerance_off,
    )
    return reward * any_support_confidence(contact)


def track_ang_vel_z_exp_any_wheel_ground(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    contact_sensor_names: tuple[str, str],
    terrain_sensor_names: tuple[str, str],
    wheel_radius: float = 0.128,
    force_off: float = 5.0,
    force_on: float = 10.0,
    geometry_tolerance_on: float = 0.02,
    geometry_tolerance_off: float = 0.035,
) -> torch.Tensor:
    """Track yaw rate while either wheel has valid ground contact."""
    asset: Articulation = env.scene[asset_cfg.name]
    error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    reward = torch.exp(-error / std**2)
    contact = _wheel_ground_contact_confidence(
        env,
        asset_cfg,
        contact_sensor_names,
        terrain_sensor_names,
        wheel_radius,
        force_off,
        force_on,
        geometry_tolerance_on,
        geometry_tolerance_off,
    )
    return reward * any_support_confidence(contact)


def body_height_command_plane_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track commanded base height along the fitted local terrain normal."""
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    centroid, normal, valid = _terrain_plane(sensor)
    target_height = env.command_manager.get_command(command_name)[:, 0]
    return base_height_plane_error_l2(asset.data.root_link_pos_w, target_height, centroid, normal, valid)


def body_height_command_grounded_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    contact_sensor_names: tuple[str, str],
    force_off: float = 5.0,
    force_on: float = 10.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Track vertical body height with a continuous ground-support gate."""
    penalty = body_height_command_l2(env, command_name, asset_cfg, sensor_cfg)
    confidence = _wheel_ground_force_confidence(env, contact_sensor_names, force_off, force_on)
    return penalty * mean_support_confidence(confidence)


def body_height_command_plane_grounded_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    contact_sensor_names: tuple[str, str],
    force_off: float = 5.0,
    force_on: float = 10.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track plane-relative body height while any wheel provides ground support."""
    penalty = body_height_command_plane_l2(env, command_name, sensor_cfg, asset_cfg)
    confidence = _wheel_ground_force_confidence(env, contact_sensor_names, force_off, force_on)
    return penalty * any_support_confidence(confidence)


def body_height_command_plane_grounded_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    contact_sensor_names: tuple[str, str],
    std: float,
    force_off: float = 5.0,
    force_on: float = 10.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward plane-relative height tracking while any wheel provides ground support."""
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    centroid, normal, valid = _terrain_plane(sensor)
    target_height = env.command_manager.get_command(command_name)[:, 0]
    reward = base_height_plane_tracking_exp(
        asset.data.root_link_pos_w,
        target_height,
        centroid,
        normal,
        valid,
        std,
    )
    confidence = _wheel_ground_force_confidence(env, contact_sensor_names, force_off, force_on)
    return reward * any_support_confidence(confidence)


def wheel_ground_contact_loss(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    contact_sensor_names: tuple[str, str],
    terrain_sensor_names: tuple[str, str],
    wheel_radius: float = 0.128,
    force_off: float = 5.0,
    force_on: float = 10.0,
    geometry_tolerance_on: float = 0.02,
    geometry_tolerance_off: float = 0.035,
) -> torch.Tensor:
    """Return the continuous number of wheels missing valid ground contact."""
    confidence = _wheel_ground_contact_confidence(
        env,
        asset_cfg,
        contact_sensor_names,
        terrain_sensor_names,
        wheel_radius,
        force_off,
        force_on,
        geometry_tolerance_on,
        geometry_tolerance_off,
    )
    return torch.sum(1.0 - confidence, dim=1)


def both_wheels_airborne(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Return the instantaneous AIRBORNE state without implying termination."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = torch.linalg.vector_norm(sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1)
    return torch.all(forces < threshold, dim=1)


def wheel_air_time_l2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    max_air_time: float = 1.0,
) -> torch.Tensor:
    """Penalize current per-wheel air time, capped to keep reward scale bounded."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = sensor.data.current_air_time
    if air_time is None:
        air_time = sensor.data.last_air_time
    if air_time is None:
        return torch.zeros(env.num_envs, device=env.device)
    air_time = torch.clamp(air_time[:, sensor_cfg.body_ids], min=0.0, max=max_air_time)
    return torch.sum(torch.square(air_time), dim=1)


def all_wheels_air_time_l2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    max_air_time: float = 1.0,
) -> torch.Tensor:
    """Penalize only the time for which all configured supports are airborne."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = sensor.data.current_air_time
    if air_time is None:
        air_time = sensor.data.last_air_time
    if air_time is None:
        return torch.zeros(env.num_envs, device=env.device)
    simultaneous_air_time = torch.amin(air_time[:, sensor_cfg.body_ids], dim=1)
    return torch.square(torch.clamp(simultaneous_air_time, min=0.0, max=max_air_time))


def wheel_actual_speed_huber(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    speed_scale: float = 1.0,
) -> torch.Tensor:
    """Penalize actual wheel speed in every contact state."""
    asset: Articulation = env.scene[asset_cfg.name]
    wheel_velocity = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return wheel_speed_huber(wheel_velocity, speed_scale=speed_scale)


def wheel_target_zero_deadband_l2(
    env: ManagerBasedRLEnv,
    action_name: str,
    target_deadband: float = 0.1,
) -> torch.Tensor:
    """Penalize learned wheel-speed targets outside a small braking deadband."""
    wheel_targets = env.action_manager.get_term(action_name).processed_actions
    return wheel_target_deadband_l2(wheel_targets, target_deadband=target_deadband)


def wheel_rolling_velocity_error(
    env: ManagerBasedRLEnv,
    radius: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize per-wheel rolling mismatch while accounting for actual base yaw rate."""
    asset: Articulation = env.scene[asset_cfg.name]
    wheel_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    relative_pos_w = wheel_pos_w - asset.data.root_link_pos_w.unsqueeze(1)
    base_quat_w = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, wheel_pos_w.shape[1], -1)
    wheel_pos_b = math_utils.quat_apply_inverse(base_quat_w, relative_pos_w)
    return differential_wheel_rolling_error_l2(
        asset.data.joint_vel[:, asset_cfg.joint_ids],
        asset.data.root_lin_vel_b[:, 0],
        asset.data.root_ang_vel_b[:, 2],
        wheel_pos_b[:, :, 1],
        radius,
    )


def wheel_target_symmetry_l2(
    env: ManagerBasedRLEnv,
    action_name: str,
    command_name: str,
    yaw_threshold: float = 0.05,
    target_difference_tolerance: float = 0.10,
) -> torch.Tensor:
    """Penalize differential wheel targets when no yaw motion is commanded."""
    wheel_targets = env.action_manager.get_term(action_name).processed_actions
    yaw_commands = env.command_manager.get_command(command_name)[:, 2]
    return wheel_target_symmetry_penalty(
        wheel_targets,
        yaw_commands,
        yaw_threshold,
        target_difference_tolerance,
    )


def zero_command_wheel_target_l2(
    env: ManagerBasedRLEnv,
    action_name: str,
    command_name: str,
    linear_threshold: float = 0.05,
    angular_threshold: float = 0.05,
    target_tolerance: float = 0.15,
) -> torch.Tensor:
    """Penalize wheel targets outside a small dead-zone under a zero velocity command."""
    wheel_targets = env.action_manager.get_term(action_name).processed_actions
    velocity_commands = env.command_manager.get_command(command_name)[:, :3]
    return zero_command_wheel_target_penalty(
        wheel_targets,
        velocity_commands,
        linear_threshold,
        angular_threshold,
        target_tolerance,
    )


def zero_command_yaw_rate_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    linear_threshold: float = 0.05,
    angular_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize actual base yaw rate under a zero planar velocity command."""
    asset: Articulation = env.scene[asset_cfg.name]
    velocity_commands = env.command_manager.get_command(command_name)[:, :3]
    return zero_command_yaw_rate_penalty(
        asset.data.root_ang_vel_b[:, 2],
        velocity_commands,
        linear_threshold,
        angular_threshold,
    )


def zero_command_yaw_rate_grounded_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    contact_sensor_names: tuple[str, str],
    force_off: float = 5.0,
    force_on: float = 10.0,
    linear_threshold: float = 0.05,
    angular_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize zero-command yaw rate only while the wheels support the robot."""
    penalty = zero_command_yaw_rate_l2(
        env,
        command_name,
        linear_threshold,
        angular_threshold,
        asset_cfg,
    )
    confidence = _wheel_ground_force_confidence(env, contact_sensor_names, force_off, force_on)
    return penalty * mean_support_confidence(confidence)


def terrain_aligned_orientation_l2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize base-up misalignment with the fitted local terrain normal."""
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    _, normal, valid = _terrain_plane(sensor)
    base_up_b = torch.zeros(env.num_envs, 3, device=asset.device)
    base_up_b[:, 2] = 1.0
    base_up_w = math_utils.quat_apply(asset.data.root_link_quat_w, base_up_b)
    return terrain_orientation_penalty(base_up_w, normal, valid)


def wheel_horizontal_neutral_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    neutral_xy: tuple[tuple[float, float], tuple[float, float]],
    tolerance_xy: tuple[float, float] = (0.04, 0.03),
    scale_xy: tuple[float, float] = (0.02, 0.02),
) -> torch.Tensor:
    """Penalize wheel centers leaving their horizontal neutral points in the base frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    wheel_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    relative_w = wheel_pos_w - asset.data.root_link_pos_w.unsqueeze(1)
    base_quat = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, wheel_pos_w.shape[1], -1)
    wheel_pos_b = math_utils.quat_apply_inverse(base_quat, relative_w)

    neutral = torch.as_tensor(neutral_xy, device=wheel_pos_b.device, dtype=wheel_pos_b.dtype)
    tolerance = torch.as_tensor(tolerance_xy, device=wheel_pos_b.device, dtype=wheel_pos_b.dtype).expand_as(neutral)
    scale = torch.as_tensor(scale_xy, device=wheel_pos_b.device, dtype=wheel_pos_b.dtype).expand_as(neutral)
    return horizontal_neutral_penalty(wheel_pos_b, neutral, tolerance, scale)


def wheel_landing_impact_l2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 100.0,
    force_scale: float = 100.0,
) -> torch.Tensor:
    """Penalize excessive force when a wheel establishes a new contact."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    force_w = sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    return landing_impact_l2(first_contact, force_w, force_threshold=force_threshold, force_scale=force_scale)


def wheel_stance_slip_l2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    terrain_sensor_cfg: SceneEntityCfg,
    wheel_radius: float = 0.128,
    force_off: float = 5.0,
    force_on: float = 10.0,
) -> torch.Tensor:
    """Penalize estimated wheel contact-point slip while the wheel is supporting."""
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    terrain_sensor: RayCaster = env.scene.sensors[terrain_sensor_cfg.name]
    _, normal, valid = _terrain_plane(terrain_sensor)
    contact = _wheel_contact_confidence(
        contact_sensor,
        sensor_cfg.body_ids,
        force_off=force_off,
        force_on=force_on,
    )
    return rolling_contact_slip_l2(
        asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids],
        asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids],
        normal,
        contact,
        wheel_radius=wheel_radius,
        plane_valid=valid,
    )


def wheel_swing_height_tracking(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    terrain_sensor_cfg: SceneEntityCfg,
    wheel_radius: float = 0.128,
    force_off: float = 5.0,
    force_on: float = 10.0,
) -> torch.Tensor:
    """Track gait-commanded swing clearance for non-contact wheel links."""
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    terrain_sensor: RayCaster = env.scene.sensors[terrain_sensor_cfg.name]

    contact = _wheel_contact_confidence(
        contact_sensor,
        sensor_cfg.body_ids,
        force_off=force_off,
        force_on=force_on,
    )
    swing_mask = contact < 0.5
    commanded_height = env.command_manager.get_command(command_name)[:, 3].unsqueeze(1)

    ray_height = terrain_sensor.data.ray_hits_w[..., 2]
    ray_height = torch.where(torch.isfinite(ray_height), ray_height, torch.nan)
    ground_height = torch.nanmedian(ray_height, dim=1).values
    ground_height = torch.nan_to_num(ground_height, nan=0.0, posinf=0.0, neginf=0.0).unsqueeze(1)

    wheel_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - ground_height - wheel_radius
    error = torch.square(wheel_height - commanded_height) * swing_mask.float()
    active_feet = torch.clamp(torch.sum(swing_mask.float(), dim=1), min=1.0)
    return torch.sum(error, dim=1) / active_feet


class ZeroCommandHoldPenalty(ManagerTermBase):
    """Dense, bounded zero-command hold cost shared by Foot and Wheel."""
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.tracker = ZeroCommandHoldTracker(env.num_envs, env.device, env.step_dt,
                                             **cfg.params.get('tracker_options', {}))
        self._last_step = None
        self._cost = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None):
        self.tracker.reset(env_ids)
        self._cost[slice(None) if env_ids is None else env_ids] = 0.

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg,
                 command_name: str = 'base_velocity', tracker_options: dict | None = None):
        step = env.common_step_counter
        if step == self._last_step:
            return self._cost
        if self._last_step is not None and step != self._last_step + 1:
            self.reset()
        self._last_step = step
        asset = env.scene[asset_cfg.name]
        qw, qx, qy, qz = asset.data.root_link_quat_w.unbind(-1)
        yaw = torch.atan2(2.*(qx*qy+qw*qz), 1.-2.*(qy.square()+qz.square()))
        self._cost[:] = self.tracker.update(asset.data.root_link_pos_w[:, :2], yaw,
                                            env.command_manager.get_command(command_name)[:, :3])
        return self._cost

    def observe(self, env):
        # Rewards run before command resampling in Isaac Lab. Advance the pose
        # once for this physics step, then reconcile the new command without dt.
        params = self.cfg.params
        self(env, **params)
        asset = env.scene[params['asset_cfg'].name]
        qw, qx, qy, qz = asset.data.root_link_quat_w.unbind(-1)
        yaw = torch.atan2(2.*(qx*qy+qw*qz), 1.-2.*(qy.square()+qz.square()))
        self.tracker.prime(asset.data.root_link_pos_w[:, :2], yaw)
        self.tracker.synchronize_command(env.command_manager.get_command(
            params.get('command_name', 'base_velocity'))[:, :3])
        return self.tracker.observation()


class MissedSwingPenalty(ManagerTermBase):
    """Per-foot failure cost per complete planned swing; cancel manager dt."""
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.tracker = MissedSwingTracker(env.num_envs, env.device, **cfg.params.get('tracker_options', {}))
        self._last_step = None
        self._cost = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None):
        self.tracker.reset(env_ids)
        self._cost[slice(None) if env_ids is None else env_ids] = 0.

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg,
                 contact_sensor_names: tuple[str, str], terrain_sensor_names: tuple[str, str],
                 gait_command_name: str = 'gait_command', wheel_radius: float = .128,
                 tracker_options: dict | None = None):
        step = env.common_step_counter
        if step == self._last_step:
            return self._cost
        if self._last_step is not None and step != self._last_step + 1:
            self.reset()
        self._last_step = step
        centroid, normal, valid = _wheel_terrain_planes(env, terrain_sensor_names)
        positions = env.scene[asset_cfg.name].data.body_link_pos_w[:, asset_cfg.body_ids]
        clearance = ((positions-centroid)*normal).sum(-1)-wheel_radius
        forces = []
        for name in contact_sensor_names:
            matrix = env.scene.sensors[name].data.force_matrix_w
            if matrix is None:
                raise RuntimeError('Missed swing detection requires terrain-filtered wheel forces.')
            forces.append(torch.linalg.vector_norm(matrix.sum(2)[:, 0], dim=-1))
        events = self.tracker.update(env.command_manager.get_term(gait_command_name).phase,
            env.command_manager.get_command(gait_command_name), clearance,
            torch.stack(forces, -1), valid, env.step_dt)
        self._cost[:] = events/env.step_dt
        return self._cost


def swing_min_clearance_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg,
    terrain_sensor_names: tuple[str, str], gait_command_name: str = 'gait_command',
    wheel_radius: float = .128, min_clearance: float = .02,
    active_start: float = .20, full_start: float = .35,
) -> torch.Tensor:
    """Continuous local-ground clearance requirement, independent for each foot."""
    gait = env.command_manager.get_command(gait_command_name)
    phase = env.command_manager.get_term(gait_command_name).phase
    foot_phase = torch.stack((phase, torch.remainder(phase+gait[:, 1], 1.)), -1)
    centroid, normal, valid = _wheel_terrain_planes(env, terrain_sensor_names)
    positions = env.scene[asset_cfg.name].data.body_link_pos_w[:, asset_cfg.body_ids]
    clearance = ((positions-centroid)*normal).sum(-1)-wheel_radius
    return swing_min_clearance_shortfall(clearance, foot_phase, gait[:, 2], valid,
                                         min_clearance, active_start, full_start)
