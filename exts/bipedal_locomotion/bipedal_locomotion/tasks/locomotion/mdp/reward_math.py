"""Pure tensor helpers for wheeled-foot reward terms.

This module intentionally depends only on PyTorch so that the numerical parts of
the rewards can be tested without starting Isaac Sim.
"""

from __future__ import annotations

import torch


def contact_confidence_from_force_history(
    force_history_w: torch.Tensor,
    force_off: float,
    force_on: float,
) -> torch.Tensor:
    """Return continuous per-body contact confidence from a short force history.

    Args:
        force_history_w: Contact forces shaped ``(N, T, B, 3)``. The history
            maximum provides a short debounce against one-frame force dropouts.
        force_off: Force at or below which confidence is zero.
        force_on: Force at or above which confidence is one.
    """
    if force_on <= force_off:
        raise ValueError(f"force_on ({force_on}) must be greater than force_off ({force_off}).")
    if force_history_w.ndim != 4 or force_history_w.shape[-1] != 3:
        raise ValueError(f"Expected force history shaped (N, T, B, 3), got {tuple(force_history_w.shape)}.")

    force_magnitude = torch.linalg.vector_norm(force_history_w, dim=-1)
    peak_force = torch.amax(force_magnitude, dim=1)
    return torch.clamp((peak_force - force_off) / (force_on - force_off), min=0.0, max=1.0)


def contact_force_excess_l2(
    force_history_w: torch.Tensor,
    force_threshold: float,
    force_scale: float,
    max_normalized_excess: float = 3.0,
) -> torch.Tensor:
    """Return a force-sensitive contact penalty summed over selected bodies.

    The peak force in the short sensor history is used so that a collision
    impulse cannot disappear between policy updates. Forces below the
    threshold are ignored; forces above it grow quadratically after
    normalization by ``force_scale``.
    """
    if force_threshold < 0.0:
        raise ValueError(f"force_threshold must be non-negative, got {force_threshold}.")
    if force_scale <= 0.0:
        raise ValueError(f"force_scale must be positive, got {force_scale}.")
    if max_normalized_excess <= 0.0:
        raise ValueError(f"max_normalized_excess must be positive, got {max_normalized_excess}.")
    if force_history_w.ndim != 4 or force_history_w.shape[-1] != 3:
        raise ValueError(f"Expected force history shaped (N, T, B, 3), got {tuple(force_history_w.shape)}.")

    peak_force = torch.amax(torch.linalg.vector_norm(force_history_w, dim=-1), dim=1)
    normalized_excess = torch.clamp(
        (peak_force - force_threshold) / force_scale,
        min=0.0,
        max=max_normalized_excess,
    )
    return torch.sum(torch.square(normalized_excess), dim=1)


def update_consecutive_contact_steps(
    previous_steps: torch.Tensor,
    force_history_w: torch.Tensor,
    force_threshold: float,
) -> torch.Tensor:
    """Advance per-environment consecutive-contact counters."""
    if force_threshold < 0.0:
        raise ValueError(f"force_threshold must be non-negative, got {force_threshold}.")
    if previous_steps.ndim != 2:
        raise ValueError(f"Expected previous_steps shaped (N, B), got {tuple(previous_steps.shape)}.")
    if force_history_w.ndim != 4 or force_history_w.shape[-1] != 3:
        raise ValueError(f"Expected force history shaped (N, T, B, 3), got {tuple(force_history_w.shape)}.")
    if force_history_w.shape[0] != previous_steps.shape[0] or force_history_w.shape[2] != previous_steps.shape[1]:
        raise ValueError(
            "Force history and previous_steps must have matching environment and body dimensions."
        )

    peak_force = torch.amax(torch.linalg.vector_norm(force_history_w, dim=-1), dim=1)
    has_contact = peak_force > force_threshold
    return torch.where(has_contact, previous_steps + 1, torch.zeros_like(previous_steps))


def all_support_confidence(contact_confidence: torch.Tensor) -> torch.Tensor:
    """Return confidence that every configured support is in contact."""
    if contact_confidence.ndim != 2:
        raise ValueError(
            f"Expected contact confidence shaped (N, B), got {tuple(contact_confidence.shape)}."
        )
    return torch.amin(contact_confidence, dim=-1)


def mean_support_confidence(contact_confidence: torch.Tensor) -> torch.Tensor:
    """Return a soft support gate averaged over the configured contacts.

    With two supports this evaluates to one for double support, one half for a
    single confident support, and zero while both supports are airborne.
    """
    if contact_confidence.ndim != 2 or contact_confidence.shape[1] == 0:
        raise ValueError(
            f"Expected non-empty contact confidence shaped (N, B), got {tuple(contact_confidence.shape)}."
        )
    return torch.mean(contact_confidence, dim=-1)


def any_support_confidence(contact_confidence: torch.Tensor) -> torch.Tensor:
    """Return a soft gate that is fully active when any support is confident.

    With two supports this evaluates to one for either single or double
    support, and zero only while both supports are airborne.
    """
    if contact_confidence.ndim != 2 or contact_confidence.shape[1] == 0:
        raise ValueError(
            f"Expected non-empty contact confidence shaped (N, B), got {tuple(contact_confidence.shape)}."
        )
    return torch.amax(contact_confidence, dim=-1)


def bounded_acceleration_tracking_penalty(
    acceleration_squared: torch.Tensor,
    tracking_error_squared: torch.Tensor,
    support_confidence: torch.Tensor,
    acceleration_scale: float,
    tracking_std: float,
) -> torch.Tensor:
    """Bound a base-acceleration penalty and activate it near the command target.

    The bounded acceleration factor prevents rare impacts from dominating the
    reward.  The tracking gate stays small immediately after a command change,
    so the policy can produce the acceleration needed to follow the new target.
    """
    if acceleration_scale <= 0.0:
        raise ValueError(f"acceleration_scale must be positive, got {acceleration_scale}.")
    if tracking_std <= 0.0:
        raise ValueError(f"tracking_std must be positive, got {tracking_std}.")
    if acceleration_squared.shape != tracking_error_squared.shape:
        raise ValueError(
            "Acceleration and tracking-error tensors must have identical shapes, got "
            f"{acceleration_squared.shape} and {tracking_error_squared.shape}."
        )
    if support_confidence.shape != acceleration_squared.shape:
        raise ValueError(
            "Support confidence must match acceleration tensors, got "
            f"{support_confidence.shape} and {acceleration_squared.shape}."
        )
    bounded_acceleration = acceleration_squared / (acceleration_squared + acceleration_scale**2)
    tracking_gate = torch.exp(-tracking_error_squared / tracking_std**2)
    return bounded_acceleration * tracking_gate * support_confidence


def charbonnier_acceleration_tracking_penalty(
    acceleration_squared: torch.Tensor,
    tracking_error_squared: torch.Tensor,
    support_confidence: torch.Tensor,
    acceleration_scale: float,
    tracking_std: float,
) -> torch.Tensor:
    """Robust acceleration penalty that retains a gradient for large oscillations.

    ``sqrt(1 + a^2 / scale^2) - 1`` is quadratic near zero and grows
    approximately linearly for large acceleration.  Unlike the bounded kernel,
    it does not flatten to a constant under severe high-frequency jitter.
    """
    if acceleration_scale <= 0.0:
        raise ValueError(f"acceleration_scale must be positive, got {acceleration_scale}.")
    if tracking_std <= 0.0:
        raise ValueError(f"tracking_std must be positive, got {tracking_std}.")
    if acceleration_squared.shape != tracking_error_squared.shape:
        raise ValueError(
            "Acceleration and tracking-error tensors must have identical shapes, got "
            f"{acceleration_squared.shape} and {tracking_error_squared.shape}."
        )
    if support_confidence.shape != acceleration_squared.shape:
        raise ValueError(
            "Support confidence must match acceleration tensors, got "
            f"{support_confidence.shape} and {acceleration_squared.shape}."
        )
    robust_acceleration = torch.sqrt(1.0 + acceleration_squared / acceleration_scale**2) - 1.0
    tracking_gate = torch.exp(-tracking_error_squared / tracking_std**2)
    return robust_acceleration * tracking_gate * support_confidence


def height_command_transition_scale(
    command: torch.Tensor,
    target: torch.Tensor,
    min_scale: float,
    full_penalty_gap: float,
    reduced_penalty_gap: float,
) -> torch.Tensor:
    """Scale a damping penalty down while a rate-limited height command is moving.

    The scale is one when the current command is within ``full_penalty_gap`` of
    its sampled target, ``min_scale`` beyond ``reduced_penalty_gap``, and
    linearly interpolated between the two gaps.
    """
    if command.shape != target.shape:
        raise ValueError(f"Command and target shapes must match, got {command.shape} and {target.shape}.")
    if not 0.0 <= min_scale <= 1.0:
        raise ValueError(f"min_scale must be in [0, 1], got {min_scale}.")
    if full_penalty_gap < 0.0 or reduced_penalty_gap <= full_penalty_gap:
        raise ValueError(
            "Require 0 <= full_penalty_gap < reduced_penalty_gap, got "
            f"{full_penalty_gap} and {reduced_penalty_gap}."
        )

    gap = torch.abs(target - command)
    transition = torch.clamp(
        (gap - full_penalty_gap) / (reduced_penalty_gap - full_penalty_gap),
        min=0.0,
        max=1.0,
    )
    return 1.0 - (1.0 - min_scale) * transition


def height_bin_event_statistics(
    commanded_height: torch.Tensor,
    event: torch.Tensor,
    minimum: float,
    maximum: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return event rates and sample fractions for low/mid/high height bins.

    The low and high bins each cover one quarter of the configured range; the
    middle bin covers the remaining half. Empty bins report a zero rate and a
    zero sample fraction so the companion fraction exposes missing samples.
    """
    if commanded_height.ndim != 1 or event.shape != commanded_height.shape:
        raise ValueError(
            "Expected one-dimensional height and event tensors with matching shapes, got "
            f"{commanded_height.shape} and {event.shape}."
        )
    if maximum <= minimum:
        raise ValueError(f"maximum ({maximum}) must be greater than minimum ({minimum}).")

    span = maximum - minimum
    low_edge = minimum + 0.25 * span
    high_edge = maximum - 0.25 * span
    masks = torch.stack(
        (
            commanded_height < low_edge,
            (commanded_height >= low_edge) & (commanded_height <= high_edge),
            commanded_height > high_edge,
        ),
        dim=0,
    )
    counts = torch.sum(masks, dim=1)
    event_f = event.to(commanded_height.dtype)
    event_counts = torch.sum(masks.to(commanded_height.dtype) * event_f.unsqueeze(0), dim=1)
    rates = event_counts / torch.clamp(counts.to(commanded_height.dtype), min=1.0)
    fractions = counts.to(commanded_height.dtype) / max(commanded_height.numel(), 1)
    return rates, fractions


def wheel_clearance_confidence(
    wheel_positions_w: torch.Tensor,
    plane_centroids_w: torch.Tensor,
    plane_normals_w: torch.Tensor,
    plane_valid: torch.Tensor,
    wheel_radius: float,
    tolerance_on: float,
    tolerance_off: float,
) -> torch.Tensor:
    """Return per-wheel confidence that wheel-center clearance matches the radius."""
    if tolerance_off <= tolerance_on:
        raise ValueError(
            f"tolerance_off ({tolerance_off}) must be greater than tolerance_on ({tolerance_on})."
        )
    if (
        wheel_positions_w.shape != plane_centroids_w.shape
        or wheel_positions_w.shape != plane_normals_w.shape
    ):
        raise ValueError("Wheel positions, plane centroids and plane normals must have identical shapes.")
    if wheel_positions_w.ndim != 3 or wheel_positions_w.shape[-1] != 3:
        raise ValueError(f"Expected wheel geometry shaped (N, B, 3), got {tuple(wheel_positions_w.shape)}.")
    if plane_valid.shape != wheel_positions_w.shape[:2]:
        raise ValueError(f"Expected plane validity shaped {wheel_positions_w.shape[:2]}, got {tuple(plane_valid.shape)}.")

    signed_distance = torch.sum((wheel_positions_w - plane_centroids_w) * plane_normals_w, dim=-1)
    clearance_error = torch.abs(torch.abs(signed_distance) - wheel_radius)
    confidence = torch.clamp(
        (tolerance_off - clearance_error) / (tolerance_off - tolerance_on),
        min=0.0,
        max=1.0,
    )
    return confidence * plane_valid.to(confidence.dtype)


def horizontal_neutral_penalty(
    wheel_positions_b: torch.Tensor,
    neutral_xy: torch.Tensor,
    tolerance_xy: torch.Tensor,
    scale_xy: torch.Tensor,
) -> torch.Tensor:
    """Penalize only horizontal wheel-center offsets outside a neutral dead-zone."""
    if wheel_positions_b.ndim != 3 or wheel_positions_b.shape[-1] != 3:
        raise ValueError(f"Expected wheel positions shaped (N, B, 3), got {tuple(wheel_positions_b.shape)}.")
    expected_shape = wheel_positions_b.shape[1:2] + (2,)
    for name, value in (("neutral_xy", neutral_xy), ("tolerance_xy", tolerance_xy), ("scale_xy", scale_xy)):
        if value.shape != expected_shape:
            raise ValueError(f"Expected {name} shaped {expected_shape}, got {tuple(value.shape)}.")
    if torch.any(tolerance_xy < 0.0):
        raise ValueError("Horizontal neutral tolerances must be non-negative.")
    if torch.any(scale_xy <= 0.0):
        raise ValueError("Horizontal neutral scales must be positive.")

    normalized_excess = torch.relu(torch.abs(wheel_positions_b[..., :2] - neutral_xy) - tolerance_xy) / scale_xy
    huber = torch.where(
        normalized_excess < 1.0,
        0.5 * torch.square(normalized_excess),
        normalized_excess - 0.5,
    )
    return torch.sum(huber, dim=(1, 2))


def wheel_target_symmetry_penalty(
    wheel_targets: torch.Tensor,
    yaw_commands: torch.Tensor,
    yaw_threshold: float,
    target_difference_tolerance: float,
) -> torch.Tensor:
    """Penalize unequal left/right wheel targets while the yaw command is near zero."""
    if wheel_targets.ndim != 2 or wheel_targets.shape[1] != 2:
        raise ValueError(f"Expected wheel targets shaped (N, 2), got {tuple(wheel_targets.shape)}.")
    if yaw_commands.shape != wheel_targets.shape[:1]:
        raise ValueError(f"Expected yaw commands shaped {wheel_targets.shape[:1]}, got {tuple(yaw_commands.shape)}.")
    if yaw_threshold < 0.0 or target_difference_tolerance < 0.0:
        raise ValueError("Yaw threshold and wheel-target tolerance must be non-negative.")

    target_difference = torch.abs(wheel_targets[:, 0] - wheel_targets[:, 1])
    excess = torch.relu(target_difference - target_difference_tolerance)
    near_zero_yaw = torch.abs(yaw_commands) <= yaw_threshold
    return torch.square(excess) * near_zero_yaw.to(excess.dtype)


def zero_command_wheel_target_penalty(
    wheel_targets: torch.Tensor,
    velocity_commands: torch.Tensor,
    linear_threshold: float,
    angular_threshold: float,
    target_tolerance: float,
) -> torch.Tensor:
    """Penalize non-zero wheel targets while every planar velocity command is near zero."""
    if wheel_targets.ndim != 2 or wheel_targets.shape[1] != 2:
        raise ValueError(f"Expected wheel targets shaped (N, 2), got {tuple(wheel_targets.shape)}.")
    if velocity_commands.ndim != 2 or velocity_commands.shape != (wheel_targets.shape[0], 3):
        raise ValueError(
            f"Expected velocity commands shaped {(wheel_targets.shape[0], 3)}, got {tuple(velocity_commands.shape)}."
        )
    if linear_threshold < 0.0 or angular_threshold < 0.0 or target_tolerance < 0.0:
        raise ValueError("Command thresholds and wheel-target tolerance must be non-negative.")

    zero_linear = torch.linalg.vector_norm(velocity_commands[:, :2], dim=1) <= linear_threshold
    zero_yaw = torch.abs(velocity_commands[:, 2]) <= angular_threshold
    excess = torch.relu(torch.abs(wheel_targets) - target_tolerance)
    return torch.sum(torch.square(excess), dim=1) * (zero_linear & zero_yaw).to(excess.dtype)


def zero_command_yaw_rate_penalty(
    yaw_rate: torch.Tensor,
    velocity_commands: torch.Tensor,
    linear_threshold: float,
    angular_threshold: float,
) -> torch.Tensor:
    """Penalize actual yaw rate only while all planar velocity commands are near zero."""
    if yaw_rate.ndim != 1:
        raise ValueError(f"Expected yaw rate shaped (N,), got {tuple(yaw_rate.shape)}.")
    if velocity_commands.ndim != 2 or velocity_commands.shape != (yaw_rate.shape[0], 3):
        raise ValueError(
            f"Expected velocity commands shaped {(yaw_rate.shape[0], 3)}, got {tuple(velocity_commands.shape)}."
        )
    if linear_threshold < 0.0 or angular_threshold < 0.0:
        raise ValueError("Command thresholds must be non-negative.")

    zero_linear = torch.linalg.vector_norm(velocity_commands[:, :2], dim=1) <= linear_threshold
    zero_yaw = torch.abs(velocity_commands[:, 2]) <= angular_threshold
    return torch.square(yaw_rate) * (zero_linear & zero_yaw).to(yaw_rate.dtype)


def fit_height_plane(points_w: torch.Tensor, eps: float = 1.0e-6) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit ``z = a*x + b*y + c`` and return centroid, upward normal and validity.

    The closed-form 2-D least-squares fit is inexpensive enough to use in
    vectorized rewards. Invalid and degenerate point sets fall back to an upward
    normal and are marked invalid so callers can mask their penalties.
    """
    if points_w.ndim != 3 or points_w.shape[-1] != 3:
        raise ValueError(f"Expected terrain points shaped (N, P, 3), got {tuple(points_w.shape)}.")

    finite = torch.isfinite(points_w).all(dim=-1)
    safe_points = torch.where(finite.unsqueeze(-1), points_w, torch.zeros_like(points_w))
    count = finite.sum(dim=1)
    denom_count = torch.clamp(count, min=1).to(points_w.dtype).unsqueeze(-1)
    centroid = safe_points.sum(dim=1) / denom_count

    centered = points_w - centroid.unsqueeze(1)
    centered = torch.where(finite.unsqueeze(-1), centered, torch.zeros_like(centered))
    x, y, z = centered.unbind(dim=-1)
    count_f = torch.clamp(count, min=1).to(points_w.dtype)
    s_xx = torch.sum(x * x, dim=1) / count_f
    s_xy = torch.sum(x * y, dim=1) / count_f
    s_yy = torch.sum(y * y, dim=1) / count_f
    s_xz = torch.sum(x * z, dim=1) / count_f
    s_yz = torch.sum(y * z, dim=1) / count_f

    determinant = s_xx * s_yy - s_xy * s_xy
    valid = (count >= 3) & (torch.abs(determinant) > eps)
    safe_determinant = torch.where(valid, determinant, torch.ones_like(determinant))
    slope_x = (s_xz * s_yy - s_yz * s_xy) / safe_determinant
    slope_y = (s_yz * s_xx - s_xz * s_xy) / safe_determinant

    normal = torch.stack((-slope_x, -slope_y, torch.ones_like(slope_x)), dim=-1)
    normal = torch.nn.functional.normalize(normal, dim=-1)
    fallback = torch.zeros_like(normal)
    fallback[:, 2] = 1.0
    normal = torch.where(valid.unsqueeze(-1), normal, fallback)
    centroid = torch.where(valid.unsqueeze(-1), centroid, torch.zeros_like(centroid))
    return centroid, normal, valid


def terrain_orientation_penalty(
    base_up_w: torch.Tensor,
    plane_normal_w: torch.Tensor,
    plane_valid: torch.Tensor,
) -> torch.Tensor:
    """Penalize base-up misalignment with the local terrain normal."""
    cosine = torch.sum(base_up_w * plane_normal_w, dim=-1)
    penalty = 1.0 - torch.square(torch.clamp(cosine, min=-1.0, max=1.0))
    return penalty * plane_valid.to(penalty.dtype)


def base_height_plane_error_l2(
    base_position_w: torch.Tensor,
    target_height: torch.Tensor,
    plane_centroid_w: torch.Tensor,
    plane_normal_w: torch.Tensor,
    plane_valid: torch.Tensor,
) -> torch.Tensor:
    """Return squared base-to-plane normal-distance tracking error."""
    distance = torch.abs(torch.sum((base_position_w - plane_centroid_w) * plane_normal_w, dim=-1))
    error = torch.square(distance - target_height)
    return error * plane_valid.to(error.dtype)


def base_height_plane_tracking_exp(
    base_position_w: torch.Tensor,
    target_height: torch.Tensor,
    plane_centroid_w: torch.Tensor,
    plane_normal_w: torch.Tensor,
    plane_valid: torch.Tensor,
    std: float,
) -> torch.Tensor:
    """Return a Gaussian/RBF-shaped reward for base-to-plane height tracking."""
    if std <= 0.0:
        raise ValueError(f"std must be positive, got {std}.")
    distance = torch.abs(torch.sum((base_position_w - plane_centroid_w) * plane_normal_w, dim=-1))
    squared_error = torch.square(distance - target_height)
    reward = torch.exp(-squared_error / std**2)
    return reward * plane_valid.to(reward.dtype)


def terrain_relative_feet_regulation(
    foot_positions_w: torch.Tensor,
    foot_velocities_w: torch.Tensor,
    plane_centroids_w: torch.Tensor,
    plane_normals_w: torch.Tensor,
    plane_valid: torch.Tensor,
    foot_radius: float,
    height_scale: float,
) -> torch.Tensor:
    """Apply PF-style foot regulation using local terrain-normal clearance.

    Tangential motion is expensive near the local terrain and becomes
    progressively cheaper as the support end is lifted.  This does not impose
    a target swing height, so the policy can choose clearance from terrain
    observations and task outcomes.
    """
    if foot_radius < 0.0:
        raise ValueError(f"foot_radius must be non-negative, got {foot_radius}.")
    if height_scale <= 0.0:
        raise ValueError(f"height_scale must be positive, got {height_scale}.")
    if foot_positions_w.ndim != 3 or foot_positions_w.shape[-1] != 3:
        raise ValueError(f"Expected foot positions shaped (N, B, 3), got {tuple(foot_positions_w.shape)}.")
    if (
        foot_velocities_w.shape != foot_positions_w.shape
        or plane_centroids_w.shape != foot_positions_w.shape
        or plane_normals_w.shape != foot_positions_w.shape
    ):
        raise ValueError("Foot positions/velocities and terrain planes must have identical shapes.")
    if plane_valid.shape != foot_positions_w.shape[:2]:
        raise ValueError(f"Expected plane validity shaped {foot_positions_w.shape[:2]}, got {tuple(plane_valid.shape)}.")

    signed_distance = torch.sum((foot_positions_w - plane_centroids_w) * plane_normals_w, dim=-1)
    clearance = torch.clamp(torch.abs(signed_distance) - foot_radius, min=0.0, max=1.0)
    normal_velocity = torch.sum(foot_velocities_w * plane_normals_w, dim=-1, keepdim=True)
    tangential_velocity = foot_velocities_w - normal_velocity * plane_normals_w
    tangential_speed_l2 = torch.sum(torch.square(tangential_velocity), dim=-1)
    per_foot = torch.exp(-clearance / height_scale) * tangential_speed_l2
    return torch.sum(per_foot * plane_valid.to(per_foot.dtype), dim=1)


def terrain_relative_landing_velocity_l2(
    foot_positions_w: torch.Tensor,
    foot_velocities_w: torch.Tensor,
    plane_centroids_w: torch.Tensor,
    plane_normals_w: torch.Tensor,
    plane_valid: torch.Tensor,
    in_contact: torch.Tensor,
    foot_radius: float,
    about_landing_threshold: float,
) -> torch.Tensor:
    """Penalize downward normal velocity shortly before local-terrain contact."""
    if foot_radius < 0.0:
        raise ValueError(f"foot_radius must be non-negative, got {foot_radius}.")
    if about_landing_threshold <= 0.0:
        raise ValueError(f"about_landing_threshold must be positive, got {about_landing_threshold}.")
    if foot_positions_w.ndim != 3 or foot_positions_w.shape[-1] != 3:
        raise ValueError(f"Expected foot positions shaped (N, B, 3), got {tuple(foot_positions_w.shape)}.")
    if (
        foot_velocities_w.shape != foot_positions_w.shape
        or plane_centroids_w.shape != foot_positions_w.shape
        or plane_normals_w.shape != foot_positions_w.shape
    ):
        raise ValueError("Foot positions/velocities and terrain planes must have identical shapes.")
    expected_mask_shape = foot_positions_w.shape[:2]
    if plane_valid.shape != expected_mask_shape or in_contact.shape != expected_mask_shape:
        raise ValueError(
            f"Expected plane/contact masks shaped {expected_mask_shape}, got "
            f"{tuple(plane_valid.shape)} and {tuple(in_contact.shape)}."
        )

    signed_distance = torch.sum((foot_positions_w - plane_centroids_w) * plane_normals_w, dim=-1)
    clearance = torch.abs(signed_distance) - foot_radius
    normal_velocity = torch.sum(foot_velocities_w * plane_normals_w, dim=-1)
    about_to_land = (
        (clearance < about_landing_threshold)
        & (~in_contact.to(torch.bool))
        & (normal_velocity < 0.0)
        & plane_valid.to(torch.bool)
    )
    return torch.sum(torch.square(normal_velocity) * about_to_land.to(normal_velocity.dtype), dim=1)


def differential_wheel_rolling_error_l2(
    wheel_joint_velocity: torch.Tensor,
    base_forward_velocity: torch.Tensor,
    base_yaw_rate: torch.Tensor,
    wheel_lateral_position_b: torch.Tensor,
    wheel_radius: float,
) -> torch.Tensor:
    """Penalize per-wheel rolling-speed mismatch for differential steering.

    The expected longitudinal velocity of a wheel center at lateral offset
    ``y_i`` is ``v_x - omega_z * y_i``.  Magnitudes are compared to preserve
    the sign-independent behavior of the previous rolling reward while making
    straight, reverse, arc, and in-place turns use the same per-wheel model.
    """
    if wheel_radius <= 0.0:
        raise ValueError(f"wheel_radius must be positive, got {wheel_radius}.")
    if wheel_joint_velocity.ndim != 2 or wheel_joint_velocity.shape[1] == 0:
        raise ValueError(
            f"Expected non-empty wheel joint velocity shaped (N, W), got {tuple(wheel_joint_velocity.shape)}."
        )
    if wheel_lateral_position_b.shape != wheel_joint_velocity.shape:
        raise ValueError(
            "Wheel lateral positions must match wheel joint velocities, got "
            f"{tuple(wheel_lateral_position_b.shape)} and {tuple(wheel_joint_velocity.shape)}."
        )
    expected_batch_shape = wheel_joint_velocity.shape[:1]
    if base_forward_velocity.shape != expected_batch_shape or base_yaw_rate.shape != expected_batch_shape:
        raise ValueError(
            f"Expected base velocities shaped {expected_batch_shape}, got "
            f"{tuple(base_forward_velocity.shape)} and {tuple(base_yaw_rate.shape)}."
        )

    wheel_surface_speed = torch.abs(wheel_joint_velocity) * wheel_radius
    expected_wheel_speed = torch.abs(
        base_forward_velocity.unsqueeze(1) - base_yaw_rate.unsqueeze(1) * wheel_lateral_position_b
    )
    return torch.mean(torch.square(wheel_surface_speed - expected_wheel_speed), dim=1)


def rolling_contact_slip_l2(
    wheel_linear_velocity_w: torch.Tensor,
    wheel_angular_velocity_w: torch.Tensor,
    plane_normal_w: torch.Tensor,
    contact_confidence: torch.Tensor,
    wheel_radius: float,
    plane_valid: torch.Tensor,
) -> torch.Tensor:
    """Estimate tangential contact-point velocity and penalize stance slip."""
    normal = plane_normal_w.unsqueeze(1).expand_as(wheel_linear_velocity_w)
    contact_velocity = wheel_linear_velocity_w - wheel_radius * torch.linalg.cross(
        wheel_angular_velocity_w,
        normal,
        dim=-1,
    )
    normal_velocity = torch.sum(contact_velocity * normal, dim=-1, keepdim=True) * normal
    tangential_velocity = contact_velocity - normal_velocity
    per_wheel = torch.sum(torch.square(tangential_velocity), dim=-1) * contact_confidence
    penalty = torch.sum(per_wheel, dim=1) / torch.clamp(contact_confidence.sum(dim=1), min=1.0)
    return penalty * plane_valid.to(penalty.dtype)


def landing_impact_l2(
    first_contact: torch.Tensor,
    force_w: torch.Tensor,
    force_threshold: float,
    force_scale: float,
) -> torch.Tensor:
    """Penalize excessive normal-force magnitude on newly established contacts."""
    if force_scale <= 0.0:
        raise ValueError("force_scale must be positive.")
    force = torch.linalg.vector_norm(force_w, dim=-1)
    normalized_excess = torch.relu(force - force_threshold) / force_scale
    return torch.sum(torch.square(normalized_excess) * first_contact.to(force.dtype), dim=1)
