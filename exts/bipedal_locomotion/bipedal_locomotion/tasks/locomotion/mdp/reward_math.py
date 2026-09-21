"""Pure tensor helpers for wheeled-foot reward terms.

This module intentionally depends only on PyTorch so that the numerical parts of
the rewards can be tested without starting Isaac Sim.
"""

from __future__ import annotations

import math

import torch


class GaitCycleIntegral:
    """Per-environment signed integral over the most recent *one phase cycle*.

    Samples contain a phase span and the signed signal integral over that span.
    The oldest included sample is fractionally weighted at the one-cycle boundary.
    A disabled/invalid sample clears that environment's history, so disjoint zero-
    command intervals can never be joined. Call exactly once per control interval.
    """

    def __init__(self, num_envs: int, capacity: int, device: str):
        if capacity < 2:
            raise ValueError("Cycle history capacity must be at least two samples.")
        self.phase_spans = torch.zeros(num_envs, capacity, device=device)
        self.increments = torch.zeros_like(self.phase_spans)
        self.time_spans = torch.zeros_like(self.phase_spans)
        self.integral = torch.zeros(num_envs, device=device)
        self.duration = torch.zeros_like(self.integral)
        self.ready = torch.zeros(num_envs, device=device, dtype=torch.bool)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.phase_spans[ids] = 0.0
        self.increments[ids] = 0.0
        self.time_spans[ids] = 0.0
        self.integral[ids] = 0.0
        self.duration[ids] = 0.0
        self.ready[ids] = False

    def update(self, increment, phase_span, enabled, dt):
        valid = (enabled & torch.isfinite(increment) & torch.isfinite(phase_span)
                 & (phase_span > 0.0) & (phase_span < 1.0))
        self.phase_spans = self.phase_spans.roll(-1, dims=1)
        self.increments = self.increments.roll(-1, dims=1)
        self.time_spans = self.time_spans.roll(-1, dims=1)
        self.phase_spans[:, -1] = phase_span
        self.increments[:, -1] = increment
        self.time_spans[:, -1] = dt
        # torch.where, rather than multiplication, also removes NaN samples.
        self.phase_spans = torch.where(valid[:, None], self.phase_spans, 0.0)
        self.increments = torch.where(valid[:, None], self.increments, 0.0)
        self.time_spans = torch.where(valid[:, None], self.time_spans, 0.0)
        younger_phase = self.phase_spans.flip(1).cumsum(1).flip(1) - self.phase_spans
        fraction = ((1.0 - younger_phase) / self.phase_spans.clamp_min(1.e-8)).clamp(0.0, 1.0)
        self.ready[:] = valid & (self.phase_spans.sum(1) >= 1.0 - 1.e-6)
        self.integral[:] = torch.where(self.ready, (self.increments * fraction).sum(1), 0.0)
        self.duration[:] = torch.where(self.ready, (self.time_spans * fraction).sum(1), 0.0)
        return self.integral


def lateral_foot_width_penalty(
    foot_positions_w: torch.Tensor,
    base_quat_w: torch.Tensor,
    min_width: float,
    max_width: float,
    error_scale: float,
) -> torch.Tensor:
    """Huber cost outside a heading-frame lateral width band, in left/right order.

    Signed width penalizes crossed legs. Fore-aft separation and unequal foot
    heights cannot satisfy this constraint. Only base heading is used so roll
    does not turn swing height into a spurious width change. Quaternions use
    Isaac Lab's normalized wxyz convention.
    """
    if not 0.0 < min_width < max_width or error_scale <= 0.0:
        raise ValueError("Expected 0 < min_width < max_width and error_scale > 0.")
    if foot_positions_w.ndim != 3 or foot_positions_w.shape[1:] != (2, 3):
        raise ValueError("Expected left/right foot positions shaped (N, 2, 3).")
    if base_quat_w.shape != (foot_positions_w.shape[0], 4):
        raise ValueError("Expected base quaternions shaped (N, 4).")

    qw, qx, qy, qz = base_quat_w.unbind(dim=-1)
    forward_x = 1.0 - 2.0 * (qy.square() + qz.square())
    forward_y = 2.0 * (qx * qy + qw * qz)
    heading_norm = torch.sqrt(forward_x.square() + forward_y.square()).clamp_min(1.0e-6)
    delta = foot_positions_w[:, 0] - foot_positions_w[:, 1]
    width = (-forward_y * delta[:, 0] + forward_x * delta[:, 1]) / heading_norm
    error = (torch.relu(min_width - width) + torch.relu(width - max_width)) / error_scale
    return torch.where(error <= 1.0, 0.5 * error.square(), error - 0.5)


def terrain_swing_peak_height(
    ray_hits_w: torch.Tensor,
    min_height: float = 0.02,
    max_height: float = 0.10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clamp the full scan's finite world-Z range into a swing peak range.

    Missing rays are excluded rather than filled with zero. At least two valid
    points are needed; an invalid scan returns the minimum and a false mask.
    """
    if not 0.0 < min_height <= max_height:
        raise ValueError("Expected 0 < min_height <= max_height.")
    if ray_hits_w.ndim != 3 or ray_hits_w.shape[-1] != 3 or ray_hits_w.shape[1] < 2:
        raise ValueError("Expected ray hits shaped (N, P >= 2, 3).")
    finite = torch.isfinite(ray_hits_w).all(dim=-1)
    valid = finite.sum(dim=1) >= 2
    z = ray_hits_w[..., 2]
    high = torch.where(finite, z, -torch.inf).amax(dim=1)
    low = torch.where(finite, z, torch.inf).amin(dim=1)
    span = torch.where(valid, high - low, torch.zeros_like(high))
    return span.clamp(min_height, max_height), valid


def phase_swing_clearance_penalty(
    clearance: torch.Tensor,
    foot_phase: torch.Tensor,
    stance_fraction: torch.Tensor,
    support_confidence: torch.Tensor,
    plane_valid: torch.Tensor,
    peak_height: torch.Tensor,
    error_scale: float = 0.02,
    upper_tolerance: float = 0.02,
    over_height_weight: float = 0.1,
    under_height_weight: float = 1.0,
) -> torch.Tensor:
    """Penalize insufficient swing clearance strongly and excess height weakly.

    The zero-cost band is [H, H + tolerance] times the sin-squared swing arc.
    Peak height is per environment, not a velocity-command gate. The swing
    foot need not already be airborne to incur a shortfall cost. Normalization
    uses planned swing count, preserving soft opposite-foot support gating.
    """
    if error_scale <= 0.0 or upper_tolerance < 0.0 or not 0.0 <= over_height_weight <= 1.0:
        raise ValueError("Expected positive error_scale, nonnegative tolerance, and excess weight in [0, 1].")
    if not math.isfinite(under_height_weight) or under_height_weight <= 0.0:
        raise ValueError("under_height_weight must be finite and positive.")
    if clearance.ndim != 2 or clearance.shape[1] != 2:
        raise ValueError("Expected left/right clearance shaped (N, 2).")
    if any(value.shape != clearance.shape for value in (foot_phase, support_confidence, plane_valid)):
        raise ValueError("Clearance, phases, support confidence and plane validity must have equal shapes.")
    if stance_fraction.shape != clearance.shape[:1] or peak_height.shape != clearance.shape[:1]:
        raise ValueError("Expected stance_fraction and peak_height shaped (N,).")

    duration = stance_fraction.unsqueeze(1)
    swing = foot_phase > duration
    progress = ((foot_phase - duration) / (1.0 - duration).clamp_min(1.0e-6)).clamp(0.0, 1.0)
    envelope = torch.sin(torch.pi * progress).square() * swing.to(clearance.dtype)
    lower = peak_height.unsqueeze(1) * envelope
    upper = (peak_height.unsqueeze(1) + upper_tolerance) * envelope
    valid = plane_valid & torch.isfinite(clearance)
    safe_clearance = torch.where(valid, clearance, torch.zeros_like(clearance))
    shortfall = torch.relu(lower - safe_clearance) / error_scale
    excess = torch.relu(safe_clearance - upper) / error_scale
    shortfall_cost = torch.where(shortfall <= 1.0, 0.5 * shortfall.square(), shortfall - 0.5)
    excess_cost = torch.where(excess <= 1.0, 0.5 * excess.square(), excess - 0.5)
    support = torch.where(valid, support_confidence.clamp(0.0, 1.0), 0.0)
    opposite_support = support.flip(dims=(1,))
    per_foot = (under_height_weight * shortfall_cost + over_height_weight * excess_cost) * envelope * opposite_support
    per_foot = torch.where(valid, per_foot, 0.0)
    planned_swing_count = swing.sum(dim=1).clamp_min(1)
    return per_foot.sum(dim=1) / planned_swing_count


def phase_swing_clearance_exp(
    clearance: torch.Tensor,
    foot_phase: torch.Tensor,
    stance_fraction: torch.Tensor,
    support_confidence: torch.Tensor,
    plane_valid: torch.Tensor,
    peak_height: torch.Tensor,
    std: float = 0.025,
) -> torch.Tensor:
    """Reward a phase-shaped swing trajectory with opposite-foot support.

    The target is H*sin²(pi*u). The same envelope gates the reward so touching
    down at a zero-height endpoint is not rewarded as a completed swing.
    Normalize by planned swing count, not actual support, to preserve gating.
    Velocity commands do not gate this reward, including at zero command.
    """
    if not math.isfinite(std) or std <= 0.0:
        raise ValueError("Swing clearance std must be finite and positive.")
    if clearance.ndim != 2 or clearance.shape[1] != 2:
        raise ValueError("Expected left/right clearance shaped (N, 2).")
    if any(value.shape != clearance.shape for value in (foot_phase, support_confidence, plane_valid)):
        raise ValueError("Clearance, phases, support confidence and plane validity must have equal shapes.")
    if stance_fraction.shape != clearance.shape[:1] or peak_height.shape != clearance.shape[:1]:
        raise ValueError("Expected stance_fraction and peak_height shaped (N,).")

    duration = stance_fraction.unsqueeze(1)
    swing = foot_phase > duration
    progress = ((foot_phase - duration) / (1.0 - duration).clamp_min(1.0e-6)).clamp(0.0, 1.0)
    envelope = torch.sin(torch.pi * progress).square() * swing.to(clearance.dtype)
    target = peak_height.unsqueeze(1) * envelope
    valid = plane_valid & torch.isfinite(clearance)
    safe_clearance = torch.where(valid, clearance, torch.zeros_like(clearance))
    support = torch.where(valid, support_confidence.clamp(0.0, 1.0), 0.0)
    tracking = torch.exp(-((safe_clearance - target) / std).square())
    per_foot = tracking * envelope * support.flip(dims=(1,))
    per_foot = torch.where(valid, per_foot, 0.0)
    return per_foot.sum(dim=1) / swing.sum(dim=1).clamp_min(1)


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


def wheel_speed_huber(
    wheel_velocity: torch.Tensor,
    speed_scale: float,
) -> torch.Tensor:
    """Penalize wheel speed with a normalized Huber kernel.

    The quadratic region gives a precise zero-speed objective, while the
    linear tail retains a useful gradient without letting large transient
    wheel speeds dominate the locomotion return. The penalty is deliberately
    independent of contact state so that airborne wheels cannot spin for free.
    """
    if speed_scale <= 0.0:
        raise ValueError(f"speed_scale must be positive, got {speed_scale}.")
    if wheel_velocity.ndim != 2 or wheel_velocity.shape[1] == 0:
        raise ValueError(
            f"Expected non-empty wheel_velocity shaped (N, W), got {tuple(wheel_velocity.shape)}."
        )
    normalized_speed = torch.abs(wheel_velocity) / speed_scale
    huber = torch.where(
        normalized_speed <= 1.0,
        0.5 * torch.square(normalized_speed),
        normalized_speed - 0.5,
    )
    return torch.sum(huber, dim=1)


def wheel_target_deadband_l2(
    wheel_target: torch.Tensor,
    target_deadband: float,
) -> torch.Tensor:
    """Penalize wheel-speed targets outside a symmetric zero-speed deadband."""
    if target_deadband < 0.0:
        raise ValueError(f"target_deadband must be non-negative, got {target_deadband}.")
    if wheel_target.ndim != 2 or wheel_target.shape[1] == 0:
        raise ValueError(
            f"Expected non-empty wheel_target shaped (N, W), got {tuple(wheel_target.shape)}."
        )
    excess = torch.relu(torch.abs(wheel_target) - target_deadband)
    return torch.sum(torch.square(excess), dim=1)


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
    min_tracking_gate: float = 0.0,
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
    if not 0.0 <= min_tracking_gate <= 1.0:
        raise ValueError("min_tracking_gate must be in [0, 1].")
    bounded_acceleration = acceleration_squared / (acceleration_squared + acceleration_scale**2)
    tracking_gate = torch.exp(-tracking_error_squared / tracking_std**2).clamp_min(min_tracking_gate)
    return bounded_acceleration * tracking_gate * support_confidence


def charbonnier_acceleration_tracking_penalty(
    acceleration_squared: torch.Tensor,
    tracking_error_squared: torch.Tensor,
    support_confidence: torch.Tensor,
    acceleration_scale: float,
    tracking_std: float,
    min_tracking_gate: float = 0.0,
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
    if not 0.0 <= min_tracking_gate <= 1.0:
        raise ValueError("min_tracking_gate must be in [0, 1].")
    robust_acceleration = torch.sqrt(1.0 + acceleration_squared / acceleration_scale**2) - 1.0
    tracking_gate = torch.exp(-tracking_error_squared / tracking_std**2).clamp_min(min_tracking_gate)
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


def terrain_obstacle_relief(points_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Peak-to-valley normal residual after removing the scan's fitted slope.

    A smooth incline has zero relief. This is a coarse local roughness metric,
    not a guarantee of obstacle clearance or a foothold planner.
    """
    centroid, normal, valid = fit_height_plane(points_w)
    finite = torch.isfinite(points_w).all(-1)
    residual = ((points_w - centroid[:, None]) * normal[:, None]).sum(-1)
    high = torch.where(finite, residual, -torch.inf).amax(1)
    low = torch.where(finite, residual, torch.inf).amin(1)
    return torch.where(valid, (high - low).clamp_min(0.), 0.), valid


def gait_contact_targets(phase, offsets, durations, kappa=0.05):
    """Shared smooth planned contact probabilities for force and support terms."""
    foot_phase = torch.stack((phase, torch.remainder(phase + offsets + 1., 1.)), dim=1)
    duration = durations[:, None]
    normalized = torch.where(
        foot_phase < duration,
        torch.remainder(foot_phase, 1.) * (0.5 / duration),
        torch.where(foot_phase > duration,
                    0.5 + torch.remainder(foot_phase - duration, 1.) * (0.5 / (1. - duration)),
                    foot_phase),
    )
    def cdf(x):
        return 0.5 * (1. + torch.erf(x / (kappa * math.sqrt(2.))))
    return cdf(normalized) * (1. - cdf(normalized - 0.5)) + cdf(normalized - 1.) * (1. - cdf(normalized - 1.5))


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
    allowed_downward_speed: float = 0.0,
) -> torch.Tensor:
    """Penalize downward normal velocity shortly before local-terrain contact."""
    if not math.isfinite(allowed_downward_speed) or allowed_downward_speed < 0.:
        raise ValueError("Allowed downward speed must be finite and nonnegative.")
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
    excess = torch.relu(-normal_velocity - allowed_downward_speed)
    return torch.sum(excess.square() * about_to_land.to(normal_velocity.dtype), dim=1)


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


def foothold_region_plan(base_pos, heading, velocity_b, command, gait, foot_phase, hits,
                         axes=(.04, .03), nominal_width=.34, balance_time=.10,
                         max_balance=.04, max_forward=.25, min_lateral=.12,
                         max_lateral=.28, max_leg_reach=.90, wheel_radius=.128,
                         max_slope_deg=20., max_relief=.015, scan_distance=.085):
    """Conservative flat/smooth-plane foothold regions, left then right.

    Predict constant body-frame commanded twist to touchdown; add half a stance
    of per-foot commanded velocity and bounded velocity-error feedback. This is
    a geometric reach envelope, not IK or a stair-edge support guarantee.
    """
    n_env = base_pos.shape[0]
    remaining = ((1. - foot_phase) / gait[:, 0, None]).clamp_min(0.)
    yaw_step = command[:, 2, None] * remaining
    # Stable exact SE(2) integration for a constant commanded body twist.
    sinc = torch.sinc(yaw_step / math.pi)
    cosc = .5 * yaw_step * torch.sinc(yaw_step / (2. * math.pi)).square()
    vx, vy = command[:, 0, None], command[:, 1, None]
    dx = remaining * (sinc * vx - cosc * vy)
    dy = remaining * (cosc * vx + sinc * vy)
    c, s = heading.cos()[:, None], heading.sin()[:, None]
    predicted_base = base_pos[:, None, :].expand(-1, 2, -1).clone()
    predicted_base[:, :, 0] += c * dx - s * dy
    predicted_base[:, :, 1] += s * dx + c * dy
    touchdown_yaw = heading[:, None] + yaw_step
    nominal_y = torch.tensor([nominal_width / 2., -nominal_width / 2.], device=base_pos.device)
    local_vx = vx - command[:, 2, None] * nominal_y
    stance_half = .5 * gait[:, 2, None] / gait[:, 0, None]
    feedback = ((velocity_b[:, :2] - command[:, :2]) * balance_time).clamp(-max_balance, max_balance)
    offset_x = (stance_half * local_vx + feedback[:, 0, None]).clamp(-max_forward, max_forward)
    offset_y = nominal_y + stance_half * vy + feedback[:, 1, None]
    offset_y = torch.stack((offset_y[:, 0].clamp(min_lateral, max_lateral),
                            offset_y[:, 1].clamp(-max_lateral, -min_lateral)), dim=1)
    candidate = predicted_base.clone()
    ct, st = touchdown_yaw.cos(), touchdown_yaw.sin()
    candidate[:, :, 0] += ct * offset_x - st * offset_y
    candidate[:, :, 1] += st * offset_x + ct * offset_y
    centroid, normal, plane_valid = fit_height_plane(hits)
    normals = normal[:, None].expand(-1, 2, -1)
    center = candidate - ((candidate - centroid[:, None]) * normals).sum(-1, keepdim=True) * normals
    forward = torch.stack((ct, st, torch.zeros_like(ct)), dim=-1)
    tangent_x = torch.nn.functional.normalize(forward - (forward * normals).sum(-1, keepdim=True) * normals, dim=-1)
    tangent_y = torch.cross(normals, tangent_x, dim=-1)
    finite = torch.isfinite(hits).all(-1)
    residual = ((hits - centroid[:, None]) * normal[:, None]).sum(-1)
    relief = torch.where(finite, residual.abs(), 0.).amax(1)
    # Require scan coverage at center and the four ellipse extrema, not just
    # a single nearest point. Coarse rays cannot resolve narrow edges.
    probes = torch.stack((center, center + axes[0]*tangent_x, center - axes[0]*tangent_x,
                          center + axes[1]*tangent_y, center - axes[1]*tangent_y), dim=2)
    distance2 = (probes[:, :, :, None, :] - hits[:, None, None, :, :]).square().sum(-1)
    nearest = torch.where(finite[:, None, None, :], distance2, torch.inf).amin(-1)
    coverage = (nearest <= scan_distance**2).all(-1)
    wheel_center = center + wheel_radius * normals
    # Include the region extent in the reach bound instead of checking only its center.
    reach = torch.linalg.vector_norm(wheel_center - predicted_base, dim=-1) + max(axes)
    valid = (plane_valid[:, None] & (normal[:, 2, None] >= math.cos(math.radians(max_slope_deg)))
             & (relief[:, None] <= max_relief) & coverage & (reach <= max_leg_reach)
             & torch.isfinite(center).all(-1) & torch.isfinite(command).all(-1)[:, None]
             & torch.isfinite(velocity_b).all(-1)[:, None])
    return center, tangent_x, tangent_y, normals, valid


class FootholdRegionTracker:
    """Per-swing frozen regions and one debounced landing event per actual swing.

    Pure tensors shared between Isaac rewards and optional MuJoCo visualization.
    update() is called once per policy interval. It returns an *event* cost;
    the reward adapter cancels RewardManager's dt multiplier for this term only.
    """
    def __init__(self, num_envs, device, axes=(.04,.03), min_air_time=.06,
                 contact_time=.04, min_clearance=.01, force_off=5., force_on=10.,
                 max_event_cost=5., **plan_options):
        values = (*axes, min_air_time, contact_time, min_clearance, force_on, max_event_cost)
        if len(axes) != 2 or not all(math.isfinite(v) and v > 0. for v in values) or not 0 <= force_off < force_on:
            raise ValueError("Invalid foothold region dimensions or event thresholds.")
        self.axes, self.min_air_time, self.contact_time = axes, min_air_time, contact_time
        self.min_clearance, self.force_off, self.force_on = min_clearance, force_off, force_on
        self.max_event_cost, self.plan_options = max_event_cost, plan_options
        shape = (num_envs, 2)
        for name in ('initialized','was_swing','contact','seen_support','active','valid','airborne','scored','candidate_valid'):
            setattr(self, name, torch.zeros(shape, dtype=torch.bool, device=device))
        for name in ('air_time','contact_elapsed','rho'):
            setattr(self, name, torch.zeros(shape, device=device))
        for name in ('center','tangent_x','tangent_y','normal','candidate_landing','last_landing','last_landing_normal'):
            setattr(self, name, torch.zeros((*shape,3), device=device))
        self.last_landing_valid = torch.zeros(shape, dtype=torch.bool, device=device)
        self.events = torch.zeros(shape, dtype=torch.bool, device=device)
        self.new_regions = torch.zeros_like(self.events)
        self.invalid_regions = torch.zeros_like(self.events)
        self.missed_swings = torch.zeros_like(self.events)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for value in vars(self).values():
            if isinstance(value, torch.Tensor):
                value[ids] = 0

    def update(self, base_pos, heading, velocity_b, command, gait, phase, hits,
               wheel_pos, ground_force, clearance, dt):
        if not math.isfinite(dt) or dt <= 0.:
            raise ValueError("Expected positive policy dt.")
        foot_phase = torch.stack((phase, torch.remainder(phase + gait[:,1], 1.)), dim=1)
        swing = foot_phase >= gait[:,2,None]
        previous_contact = self.contact.clone()
        self.contact[:] = torch.where(ground_force >= self.force_on, True,
                                      torch.where(ground_force <= self.force_off, False, self.contact))
        self.seen_support |= self.contact
        start = swing & (~self.was_swing | ~self.initialized)
        self.new_regions[:] = start
        self.missed_swings[:] = start & self.active & ~self.scored
        self.invalid_regions.zero_(); self.events.zero_()
        selected = start.any(1)
        if selected.any():
            planned = foothold_region_plan(base_pos[selected], heading[selected], velocity_b[selected],
                command[selected], gait[selected], foot_phase[selected], hits[selected], axes=self.axes,
                **self.plan_options)
            local_start = start[selected]
            for name, value in zip(('center','tangent_x','tangent_y','normal','valid'), planned):
                old = getattr(self, name)[selected]
                mask = local_start if old.ndim == 2 else local_start[:,:,None]
                getattr(self, name)[selected] = torch.where(mask, value, old)
            self.invalid_regions[:] = start & ~self.valid
        self.active = torch.where(start, self.seen_support, self.active)
        self.scored = torch.where(start, False, self.scored)
        self.airborne = torch.where(start, False, self.airborne)
        self.candidate_valid = torch.where(start, False, self.candidate_valid)
        self.air_time = torch.where(start, 0., self.air_time)
        self.contact_elapsed = torch.where(start, 0., self.contact_elapsed)
        signal_valid = torch.isfinite(wheel_pos).all(-1) & torch.isfinite(ground_force) & torch.isfinite(clearance)
        off = ~self.contact & (clearance >= self.min_clearance) & signal_valid
        self.air_time[:] = torch.where(off & self.active & ~self.scored, self.air_time + dt, 0.)
        self.airborne |= self.air_time >= self.min_air_time - 1.e-6
        # A prolonged flight spanning another planned swing must not reuse
        # ground support from an earlier cycle to arm another landing trial.
        self.seen_support &= ~(self.airborne & ~self.contact)
        self.candidate_valid &= self.contact & signal_valid
        touchdown = self.contact & (~previous_contact | ~self.candidate_valid) & self.airborne & self.active & ~self.scored & signal_valid
        self.candidate_landing[:] = torch.where(touchdown[:,:,None], wheel_pos, self.candidate_landing)
        self.candidate_valid |= touchdown
        self.contact_elapsed[:] = torch.where(self.contact & self.airborne & self.candidate_valid & signal_valid,
                                               self.contact_elapsed + dt, 0.)
        completed = (self.contact_elapsed >= self.contact_time - 1.e-6) & self.active & ~self.scored & self.candidate_valid
        self.events[:] = completed & self.valid & signal_valid
        delta = self.candidate_landing - self.center  # wheel-radius offset is normal to both axes
        dx = (delta * self.tangent_x).sum(-1) / self.axes[0]
        dy = (delta * self.tangent_y).sum(-1) / self.axes[1]
        self.rho[:] = torch.sqrt(dx.square()+dy.square())
        outside = (self.rho - 1.).clamp_min(0.)
        cost = torch.where(outside <= 1., .5*outside.square(), outside-.5).clamp_max(self.max_event_cost)
        cost = torch.where(self.events, cost, 0.)
        self.last_landing[:] = torch.where(self.events[:,:,None], self.candidate_landing, self.last_landing)
        self.last_landing_normal[:] = torch.where(self.events[:,:,None], self.normal, self.last_landing_normal)
        self.last_landing_valid |= self.events
        self.scored |= completed
        self.was_swing[:] = swing; self.initialized[:] = True
        return cost.sum(1)


class MissedSwingTracker:
    """One failure event at the end of each complete, valid planned swing."""
    def __init__(self, num_envs, device, min_clearance=.01, min_air_time=.06,
                 force_off=5., force_on=10.):
        self.min_clearance, self.min_air_time = min_clearance, min_air_time
        self.force_off, self.force_on = force_off, force_on
        for name in ('initialized', 'previous_swing', 'active', 'valid', 'contact', 'completed', 'stance_support', 'supported'):
            setattr(self, name, torch.zeros((num_envs, 2), dtype=torch.bool, device=device))
        self.air_time = torch.zeros((num_envs, 2), device=device)

    def reset(self, env_ids=None):
        for value in vars(self).values():
            if isinstance(value, torch.Tensor):
                value[slice(None) if env_ids is None else env_ids] = 0

    def update(self, phase, gait, clearance, force, plane_valid, dt):
        foot_phase = torch.stack((phase, (phase + gait[:, 1]) % 1.), -1)
        swing = foot_phase >= gait[:, 2, None]
        finite = plane_valid & torch.isfinite(clearance) & torch.isfinite(force)
        self.contact[:] = torch.where(force >= self.force_on, True,
                                      torch.where(force <= self.force_off, False, self.contact))
        # Ignore partial swings at reset; observe a real stance->swing transition.
        self.stance_support |= ~swing & self.contact & finite
        start = self.initialized & ~self.previous_swing & swing
        end = self.initialized & self.previous_swing & ~swing
        failed = end & self.active & self.valid & finite & ~self.completed
        self.supported = torch.where(start, self.stance_support, self.supported)
        self.stance_support &= ~start
        self.active = torch.where(end, False, torch.where(start, True, self.active))
        self.valid = torch.where(start, finite, self.valid & finite)
        self.completed = torch.where(start, False, self.completed)
        self.air_time = torch.where(start, 0., self.air_time)
        off = swing & self.active & self.supported & finite & ~self.contact & (clearance >= self.min_clearance)
        self.air_time = torch.where(off, self.air_time + dt, 0.)
        self.completed |= self.air_time >= self.min_air_time - 1.e-6
        self.previous_swing[:] = swing
        self.initialized[:] = True
        return failed.float().sum(-1)


class ZeroCommandHoldTracker:
    """Independent zero-axis drift, yaw anchor, and all-zero world-position anchor.

    Moving-axis displacement is allowed. Partial-zero translation uses an exact
    finite time window of displacement resolved in the midpoint heading frame.
    All-zero translation replaces this window with a fixed world anchor.
    """
    def __init__(self, num_envs, device, dt, zero_threshold=.02, settle_time=.3,
                 window_s=1., position_deadband=.03, position_scale=.05,
                 yaw_deadband=.03, yaw_scale=.10, max_cost=5.):
        if dt <= 0 or window_s <= 0 or settle_time < 0 or min(position_scale, yaw_scale) <= 0:
            raise ValueError('Invalid zero-command hold timing or scales.')
        self.dt, self.zero_threshold, self.settle_time = dt, zero_threshold, settle_time
        self.position_deadband, self.position_scale = position_deadband, position_scale
        self.yaw_deadband, self.yaw_scale, self.max_cost = yaw_deadband, yaw_scale, max_cost
        self.history = torch.zeros((max(1, math.ceil(window_s/dt)), num_envs, 2), device=device)
        self.cursor = 0
        self.zero_mask = torch.zeros((num_envs, 3), dtype=torch.bool, device=device)
        self.age = torch.zeros((num_envs, 3), device=device)
        self.active = torch.zeros((num_envs, 3), dtype=torch.bool, device=device)
        self.initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.previous_pos = torch.zeros((num_envs, 2), device=device)
        self.previous_yaw = torch.zeros(num_envs, device=device)
        self.yaw_anchor = torch.zeros(num_envs, device=device)
        self.position_anchor = torch.zeros((num_envs, 2), device=device)
        self.full_active = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.history[:, ids] = 0
        for name in ('zero_mask', 'age', 'active', 'initialized', 'previous_pos', 'previous_yaw',
                     'yaw_anchor', 'position_anchor', 'full_active'):
            getattr(self, name)[ids] = 0

    def prime(self, pos, yaw):
        """Initialize reset environments at their current pose without advancing time."""
        valid = torch.isfinite(pos).all(-1) & torch.isfinite(yaw)
        fresh = ~self.initialized & valid
        self.previous_pos[:] = torch.where(fresh[:, None], pos, self.previous_pos)
        self.previous_yaw[:] = torch.where(fresh, yaw, self.previous_yaw)
        self.initialized |= fresh

    def synchronize_command(self, command):
        """Release/rearm after command resampling; never integrate a second dt."""
        zero = (command.abs() <= self.zero_threshold) & torch.isfinite(command)
        keep = zero & self.zero_mask
        self.age *= keep
        self.active &= keep
        self.history *= keep[None, :, :2]
        self.full_active &= self.active.all(-1)
        self.zero_mask[:] = zero

    def observation(self):
        """Signed actual-minus-reference errors / scales, then three active flags.

        Fixed world-anchor XY error is rotated into current heading. Partial
        holds expose the same signed finite-window components used by the cost.
        Errors are clipped to +/-5; disabled components are exactly zero.
        """
        delta = self.previous_pos-self.position_anchor
        c, s = self.previous_yaw.cos(), self.previous_yaw.sin()
        body_delta = torch.stack((c*delta[:, 0]+s*delta[:, 1],
                                  -s*delta[:, 0]+c*delta[:, 1]), -1)
        xy = torch.where(self.full_active[:, None], body_delta, self.history.sum(0))
        yaw = self.previous_yaw-self.yaw_anchor
        yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))
        errors = torch.cat((xy/self.position_scale, yaw[:, None]/self.yaw_scale), -1)
        errors = torch.where(self.active, errors, 0.).clamp(-5., 5.)
        return torch.cat((errors, self.active.to(errors.dtype)), -1)

    def update(self, pos, yaw, command):
        finite = torch.isfinite(pos).all(-1) & torch.isfinite(yaw) & torch.isfinite(command).all(-1)
        safe_pos = torch.where(finite[:, None], pos, 0.)
        safe_yaw = torch.where(finite, yaw, 0.)
        zero = (command.abs() <= self.zero_threshold) & finite[:, None]
        self.age = torch.where(zero & self.initialized[:, None], self.age + self.dt, 0.)
        active = zero & (self.age >= self.settle_time - 1.e-6) & self.initialized[:, None]
        new = active & ~self.active
        yaw_delta = torch.atan2(torch.sin(safe_yaw-self.previous_yaw), torch.cos(safe_yaw-self.previous_yaw))
        mid = self.previous_yaw + .5*yaw_delta
        delta = safe_pos-self.previous_pos
        local = torch.stack((mid.cos()*delta[:, 0]+mid.sin()*delta[:, 1],
                             -mid.sin()*delta[:, 0]+mid.cos()*delta[:, 1]), -1)
        # Clear a component's entire history immediately when released/re-armed.
        keep = active[:, :2] & self.active[:, :2]
        self.history *= keep[None]
        self.history[self.cursor] = torch.where(keep, local, 0.)
        self.cursor = (self.cursor+1) % len(self.history)
        displacement = self.history.sum(0)
        self.yaw_anchor = torch.where(new[:, 2], safe_yaw, self.yaw_anchor)
        full = active.all(-1)
        self.position_anchor = torch.where((full & ~self.full_active)[:, None], safe_pos, self.position_anchor)
        # Radial world error allows a circular sway deadband, independent of heading.
        world_distance = torch.linalg.vector_norm(safe_pos-self.position_anchor, dim=-1)
        xy_excess = (displacement.abs()-self.position_deadband).clamp_min(0.)/self.position_scale
        world_excess = (world_distance-self.position_deadband).clamp_min(0.)/self.position_scale
        angle = torch.atan2(torch.sin(safe_yaw-self.yaw_anchor), torch.cos(safe_yaw-self.yaw_anchor))
        yaw_excess = (angle.abs()-self.yaw_deadband).clamp_min(0.)/self.yaw_scale
        def huber(x):
            return torch.where(x <= 1., .5*x.square(), x-.5).clamp_max(self.max_cost)
        xy_cost = (huber(xy_excess)*active[:, :2]).sum(-1)
        cost = torch.where(full, huber(world_excess), xy_cost) + huber(yaw_excess)*active[:, 2]
        self.previous_pos[:] = safe_pos
        self.previous_yaw[:] = safe_yaw
        self.initialized[:] = finite
        self.active[:] = active
        self.full_active[:] = full
        self.zero_mask[:] = zero
        return torch.where(finite, cost, 0.)


def swing_min_clearance_shortfall(clearance, foot_phase, stance_fraction, plane_valid,
                                 min_clearance=.02, active_start=.20, full_start=.35):
    """Independent per-foot dense cost in the middle of a planned swing.

    No velocity-command or opposite-support gate. Smoothstep ramps on swing
    progress [active_start, full_start], symmetric at touchdown; sum foot costs
    so one successful foot cannot compensate for the other missing its target.
    """
    if not math.isfinite(min_clearance) or min_clearance <= 0:
        raise ValueError('min_clearance must be finite and positive.')
    if not 0 <= active_start < full_start < .5:
        raise ValueError('Expected 0 <= active_start < full_start < 0.5.')
    duration = stance_fraction[:, None]
    progress = ((foot_phase-duration)/(1-duration).clamp_min(1.e-6)).clamp(0., 1.)
    edge = torch.minimum(progress, 1.-progress)
    ramp = ((edge-active_start)/(full_start-active_start)).clamp(0., 1.)
    envelope = ramp.square()*(3.-2.*ramp)
    valid = plane_valid & torch.isfinite(clearance) & torch.isfinite(foot_phase)
    safe = torch.where(valid, clearance, min_clearance)
    shortfall = ((min_clearance-safe)/min_clearance).clamp(0., 1.)
    return torch.where(valid, envelope*shortfall, 0.).sum(-1)
