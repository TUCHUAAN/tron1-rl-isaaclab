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


def all_support_confidence(contact_confidence: torch.Tensor) -> torch.Tensor:
    """Return confidence that every configured support is in contact."""
    if contact_confidence.ndim != 2:
        raise ValueError(
            f"Expected contact confidence shaped (N, B), got {tuple(contact_confidence.shape)}."
        )
    return torch.amin(contact_confidence, dim=-1)


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
