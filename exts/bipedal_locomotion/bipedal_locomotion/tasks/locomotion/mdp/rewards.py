"""This sub-module contains the reward functions that can be used for LimX Point Foot's locomotion task.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import distributions
from typing import TYPE_CHECKING, Optional

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
import isaaclab.utils.math as math_utils

from .reward_math import (
    all_support_confidence,
    base_height_plane_error_l2,
    contact_confidence_from_force_history,
    fit_height_plane,
    height_command_transition_scale,
    horizontal_neutral_penalty,
    landing_impact_l2,
    mean_support_confidence,
    rolling_contact_slip_l2,
    terrain_orientation_penalty,
    wheel_clearance_confidence,
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
    return torch.sum(torch.abs(torch.mul(asset.data.applied_torque, asset.data.joint_vel)), dim=1)


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
        """Calculate desired contact states for the current timestep."""
        frequencies = gait_params[:, 0]
        offsets = gait_params[:, 1]
        durations = torch.cat(
            [
                gait_params[:, 2].view(self.num_envs, 1),
                gait_params[:, 2].view(self.num_envs, 1),
            ],
            dim=1,
        )

        assert torch.all(frequencies > 0), "Frequencies must be positive"
        assert torch.all((offsets >= 0) & (offsets <= 1)), "Offsets must be between 0 and 1"
        assert torch.all((durations > 0) & (durations < 1)), "Durations must be between 0 and 1"

        gait_indices = torch.remainder(self._env.episode_length_buf * self.dt * frequencies, 1.0)

        # Calculate foot indices
        foot_indices = torch.remainder(
            torch.cat(
                [gait_indices.view(self.num_envs, 1), (gait_indices + offsets + 1).view(self.num_envs, 1)],
                dim=1,
            ),
            1.0,
        )

        # Determine stance and swing phases
        stance_idxs = foot_indices < durations
        swing_idxs = foot_indices > durations

        # Adjust foot indices based on phase
        foot_indices[stance_idxs] = torch.remainder(foot_indices[stance_idxs], 1) * (0.5 / durations[stance_idxs])
        foot_indices[swing_idxs] = 0.5 + (torch.remainder(foot_indices[swing_idxs], 1) - durations[swing_idxs]) * (
            0.5 / (1 - durations[swing_idxs])
        )

        # Calculate desired contact states using von mises distribution
        smoothing_cdf_start = distributions.normal.Normal(0, self.kappa_gait_probs).cdf
        desired_contact_states = smoothing_cdf_start(foot_indices) * (
            1 - smoothing_cdf_start(foot_indices - 0.5)
        ) + smoothing_cdf_start(foot_indices - 1) * (1 - smoothing_cdf_start(foot_indices - 1.5))

        return desired_contact_states

    def _compute_force_reward(self, forces: torch.Tensor, desired_contacts: torch.Tensor) -> torch.Tensor:
        """Compute force-based reward component."""
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

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Compute the action smoothness penalty.

        Args:
            env: The RL environment instance.

        Returns:
            The penalty value based on the action smoothness.
        """
        # Get the current action from the environment's action manager
        current_action = env.action_manager.action.clone()

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
    plane_centroids = []
    plane_normals = []
    plane_validities = []
    for terrain_name in terrain_sensor_names:

        terrain_sensor: RayCaster = env.scene.sensors[terrain_name]
        centroid, normal, valid = _terrain_plane(terrain_sensor)
        plane_centroids.append(centroid)
        plane_normals.append(normal)
        plane_validities.append(valid)

    geometry_confidence = wheel_clearance_confidence(
        wheel_positions_w,
        torch.stack(plane_centroids, dim=1),
        torch.stack(plane_normals, dim=1),
        torch.stack(plane_validities, dim=1),
        wheel_radius=wheel_radius,
        tolerance_on=geometry_tolerance_on,
        tolerance_off=geometry_tolerance_off,
    )
    return force_confidence * geometry_confidence


def stand_still_grounded(
    env: ManagerBasedRLEnv,
    contact_sensor_names: tuple[str, str],
    force_off: float = 5.0,
    force_on: float = 10.0,
    lin_threshold: float = 0.05,
    ang_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize zero-command motion only while the wheels provide ground support."""
    penalty = stand_still(env, lin_threshold, ang_threshold, asset_cfg)
    confidence = _wheel_ground_force_confidence(env, contact_sensor_names, force_off, force_on)
    return penalty * mean_support_confidence(confidence)


def track_lin_vel_xy_exp_all_wheels_ground(
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
    """Track planar velocity only while both wheels have valid ground contact."""
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
    return reward * all_support_confidence(contact)


def track_ang_vel_z_exp_all_wheels_ground(
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
    """Track yaw rate only while both wheels have valid ground contact."""
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
    return reward * all_support_confidence(contact)


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
    """Track plane-relative body height with a continuous ground-support gate."""
    penalty = body_height_command_plane_l2(env, command_name, sensor_cfg, asset_cfg)
    confidence = _wheel_ground_force_confidence(env, contact_sensor_names, force_off, force_on)
    return penalty * mean_support_confidence(confidence)


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


def wheel_rolling_velocity_error(
    env: ManagerBasedRLEnv,
    radius: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize mismatch between average wheel surface speed and base forward speed."""
    asset: Articulation = env.scene[asset_cfg.name]
    wheel_speed = torch.mean(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1) * radius
    base_forward_speed = torch.abs(asset.data.root_lin_vel_b[:, 0])
    return torch.square(wheel_speed - base_forward_speed)


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
