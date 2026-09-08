from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def wheel_terrain_progression_masks(
    distance: torch.Tensor,
    failed: torch.Tensor,
    moving_command_rate: torch.Tensor,
    linear_tracking: torch.Tensor,
    support_confidence: torch.Tensor,
    *,
    terrain_length: float,
    distance_fraction_up: float,
    moving_rate_threshold: float,
    tracking_up: float,
    tracking_down: float,
    support_up: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decide Wheel terrain promotion/demotion from episode-level performance."""
    expected_shape = distance.shape
    for name, value in {
        "failed": failed,
        "moving_command_rate": moving_command_rate,
        "linear_tracking": linear_tracking,
        "support_confidence": support_confidence,
    }.items():
        if value.shape != expected_shape:
            raise ValueError(f"{name} must have shape {tuple(expected_shape)}, got {tuple(value.shape)}.")
    if terrain_length <= 0.0 or not 0.0 < distance_fraction_up <= 1.0:
        raise ValueError("terrain_length and distance_fraction_up must define a positive promotion distance.")
    if tracking_down > tracking_up:
        raise ValueError("tracking_down must not exceed tracking_up.")

    was_commanded_to_move = moving_command_rate >= moving_rate_threshold
    move_up = (
        ~failed
        & was_commanded_to_move
        & (distance > terrain_length * distance_fraction_up)
        & (linear_tracking >= tracking_up)
        & (support_confidence >= support_up)
    )
    # A fall is always evidence that the current level is too hard.  For
    # timeout episodes, only demote when the robot was actually commanded to
    # move and tracking was clearly poor; standing episodes otherwise hold.
    move_down = failed | (was_commanded_to_move & (linear_tracking < tracking_down))
    move_down &= ~move_up
    return move_up, move_down


def wheel_terrain_levels_vel_tracking(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "base_velocity",
    asset_name: str = "robot",
    distance_fraction_up: float = 0.5,
    moving_rate_threshold: float = 0.25,
    tracking_up: float = 0.55,
    tracking_down: float = 0.25,
    support_up: float = 0.75,
) -> torch.Tensor:
    """Advance Wheel terrain only after sustained tracking with valid support."""
    asset = env.scene[asset_name]
    terrain = env.scene.terrain
    command_term = env.command_manager.get_term(command_name)
    if not hasattr(command_term, "episode_diagnostic_means"):
        raise TypeError(
            f"Command term '{command_name}' must expose episode_diagnostic_means() for the Wheel curriculum."
        )

    diagnostics = command_term.episode_diagnostic_means(env_ids)
    distance = torch.linalg.vector_norm(
        asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
        dim=1,
    )
    move_up, move_down = wheel_terrain_progression_masks(
        distance,
        env.termination_manager.terminated[env_ids],
        diagnostics["moving_command_rate"],
        diagnostics["lin_tracking_ungated"],
        diagnostics["support_any"],
        terrain_length=terrain.cfg.terrain_generator.size[0],
        distance_fraction_up=distance_fraction_up,
        moving_rate_threshold=moving_rate_threshold,
        tracking_up=tracking_up,
        tracking_down=tracking_down,
        support_up=support_up,
    )
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def modify_event_parameter(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    param_name: str,
    value: Any | SceneEntityCfg,
    num_steps: int,
) -> torch.Tensor:
    """Curriculum that modifies a parameter of an event at a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the event term.
        param_name: The name of the event term parameter.
        value: The new value for the event term parameter.
        num_steps: The number of steps after which the change should be applied.

    Returns:
        torch.Tensor: Whether the parameter has already been modified or not.
    """
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.event_manager.get_term_cfg(term_name)
        # update term settings
        term_cfg.params[param_name] = value
        env.event_manager.set_term_cfg(term_name, term_cfg)
        return torch.ones(1)
    return torch.zeros(1)


def disable_termination(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    num_steps: int,
) -> torch.Tensor:
    """Curriculum that modifies the push velocity range at a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the termination term.
        num_steps: The number of steps after which the change should be applied.

    Returns:
        torch.Tensor: Whether the parameter has already been modified or not.
    """
    env.command_manager.num_envs
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.termination_manager.get_term_cfg(term_name)
        # Remove term settings
        term_cfg.params = dict()
        term_cfg.func = lambda env: torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        env.termination_manager.set_term_cfg(term_name, term_cfg)
        return torch.ones(1)
    return torch.zeros(1)
