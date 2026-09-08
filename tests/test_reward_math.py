"""Unit tests for wheeled-foot reward tensor helpers."""

from __future__ import annotations

import importlib.util
import math
import pathlib
import unittest

import torch


_MODULE_PATH = (
    pathlib.Path(__file__).parents[1]
    / "exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/reward_math.py"
)
_SPEC = importlib.util.spec_from_file_location("wf_reward_math", _MODULE_PATH)
reward_math = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(reward_math)


class RewardMathTest(unittest.TestCase):
    def test_contact_force_excess_l2_grows_with_load(self):
        force = torch.zeros(2, 3, 2, 3)
        force[0, 0, 0, 2] = 10.0
        force[0, 1, 1, 2] = 60.0
        force[1, 2, 0, 2] = 110.0
        force[1, 1, 1, 2] = 1000.0

        penalty = reward_math.contact_force_excess_l2(
            force,
            force_threshold=10.0,
            force_scale=100.0,
        )
        torch.testing.assert_close(penalty, torch.tensor([0.25, 10.0]))

    def test_consecutive_contact_counter_resets_after_a_clear_step(self):
        counter = torch.tensor([[2, 0], [0, 2], [1, 1]])
        force = torch.zeros(3, 2, 2, 3)
        force[0, 0, 0, 2] = 21.0
        force[2, 1, 1, 2] = 50.0

        updated = reward_math.update_consecutive_contact_steps(counter, force, force_threshold=20.0)
        torch.testing.assert_close(updated, torch.tensor([[3, 0], [0, 0], [0, 2]]))

        with self.assertRaises(ValueError):
            reward_math.contact_force_excess_l2(force, force_threshold=10.0, force_scale=0.0)

    def test_contact_confidence_uses_history_and_schmitt_band(self):
        force = torch.zeros(2, 4, 2, 3)
        force[0, 0, 0, 2] = 15.0
        force[1, 0, 0, 2] = 7.5
        force[1, 3, 1, 2] = 10.0

        confidence = reward_math.contact_confidence_from_force_history(force, force_off=5.0, force_on=10.0)
        expected = torch.tensor([[1.0, 0.0], [0.5, 1.0]])
        torch.testing.assert_close(confidence, expected)
        torch.testing.assert_close(reward_math.all_support_confidence(confidence), torch.tensor([0.0, 0.5]))

    def test_mean_support_confidence_is_soft_single_support_gate(self):
        confidence = torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.25, 0.75], [0.0, 0.0]])
        gate = reward_math.mean_support_confidence(confidence)
        torch.testing.assert_close(gate, torch.tensor([1.0, 0.5, 0.5, 0.0]))

    def test_any_support_confidence_uses_strongest_support(self):
        confidence = torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.25, 0.75], [0.0, 0.0]])
        gate = reward_math.any_support_confidence(confidence)
        torch.testing.assert_close(gate, torch.tensor([1.0, 1.0, 0.75, 0.0]))

    def test_bounded_acceleration_penalty_uses_tracking_and_support_gates(self):
        penalty = reward_math.bounded_acceleration_tracking_penalty(
            acceleration_squared=torch.tensor([9.0, 16.0, 1.0]),
            tracking_error_squared=torch.tensor([0.0, 0.09, 0.0]),
            support_confidence=torch.tensor([1.0, 1.0, 0.0]),
            acceleration_scale=3.0,
            tracking_std=0.30,
        )
        expected = torch.tensor([0.5, (16.0 / 25.0) * math.exp(-1.0), 0.0])
        torch.testing.assert_close(penalty, expected, atol=1.0e-6, rtol=0.0)

    def test_bounded_acceleration_penalty_validates_scales_and_shapes(self):
        values = torch.ones(2)
        with self.assertRaises(ValueError):
            reward_math.bounded_acceleration_tracking_penalty(values, values, values, 0.0, 0.3)
        with self.assertRaises(ValueError):
            reward_math.bounded_acceleration_tracking_penalty(values, values, values, 3.0, 0.0)
        with self.assertRaises(ValueError):
            reward_math.bounded_acceleration_tracking_penalty(values, torch.ones(3), values, 3.0, 0.3)

    def test_charbonnier_acceleration_penalty_retains_large_error_growth(self):
        penalty = reward_math.charbonnier_acceleration_tracking_penalty(
            acceleration_squared=torch.tensor([0.0, 16.0, 144.0]),
            tracking_error_squared=torch.zeros(3),
            support_confidence=torch.tensor([1.0, 1.0, 0.5]),
            acceleration_scale=4.0,
            tracking_std=0.30,
        )
        expected = torch.tensor([0.0, math.sqrt(2.0) - 1.0, (math.sqrt(10.0) - 1.0) * 0.5])
        torch.testing.assert_close(penalty, expected, atol=1.0e-6, rtol=0.0)

        bounded = reward_math.bounded_acceleration_tracking_penalty(
            acceleration_squared=torch.tensor([16.0, 144.0]),
            tracking_error_squared=torch.zeros(2),
            support_confidence=torch.ones(2),
            acceleration_scale=4.0,
            tracking_std=0.30,
        )
        robust = penalty[1:] / torch.tensor([1.0, 0.5])
        self.assertGreater(float(robust[1] - robust[0]), float(bounded[1] - bounded[0]))

    def test_height_command_transition_scale_weakens_then_recovers(self):
        command = torch.tensor([0.65, 0.65, 0.65, 0.65, 0.65])
        target = torch.tensor([0.65, 0.66, 0.675, 0.69, 0.75])
        scale = reward_math.height_command_transition_scale(
            command,
            target,
            min_scale=0.2,
            full_penalty_gap=0.01,
            reduced_penalty_gap=0.04,
        )
        torch.testing.assert_close(scale, torch.tensor([1.0, 1.0, 0.6, 0.2, 0.2]), atol=2.0e-6, rtol=0.0)

    def test_height_bin_event_statistics_reports_rates_and_coverage(self):
        height = torch.tensor([0.65, 0.69, 0.72, 0.78, 0.81, 0.85])
        base_contact = torch.tensor([True, False, True, False, True, True])
        rates, fractions = reward_math.height_bin_event_statistics(height, base_contact, 0.65, 0.85)
        torch.testing.assert_close(rates, torch.tensor([0.5, 0.5, 1.0]))
        torch.testing.assert_close(fractions, torch.tensor([2.0 / 6.0, 2.0 / 6.0, 2.0 / 6.0]))

    def test_height_plane_fit_and_invalid_fallback(self):
        coordinates = []
        for x in (-1.0, 0.0, 1.0):
            for y in (-1.0, 0.0, 1.0):
                coordinates.append((x, y, 0.2 * x - 0.1 * y + 0.3))
        valid_points = torch.tensor(coordinates)
        invalid_points = torch.full_like(valid_points, torch.nan)

        centroid, normal, valid = reward_math.fit_height_plane(torch.stack((valid_points, invalid_points)))
        expected_normal = torch.tensor([-0.2, 0.1, 1.0])
        expected_normal = expected_normal / torch.linalg.vector_norm(expected_normal)
        torch.testing.assert_close(normal[0], expected_normal, atol=1.0e-5, rtol=1.0e-5)
        torch.testing.assert_close(centroid[0], torch.tensor([0.0, 0.0, 0.3]), atol=1.0e-6, rtol=0.0)
        self.assertTrue(bool(valid[0]))
        self.assertFalse(bool(valid[1]))
        torch.testing.assert_close(normal[1], torch.tensor([0.0, 0.0, 1.0]))

    def test_height_plane_exp_reward_is_gaussian_shaped_and_masks_invalid_plane(self):
        base_position = torch.tensor([[0.0, 0.0, 0.80], [0.0, 0.0, 0.85], [0.0, 0.0, 0.80]])
        target_height = torch.tensor([0.80, 0.80, 0.80])
        centroid = torch.zeros_like(base_position)
        normal = torch.zeros_like(base_position)
        normal[:, 2] = 1.0
        valid = torch.tensor([True, True, False])

        reward = reward_math.base_height_plane_tracking_exp(
            base_position,
            target_height,
            centroid,
            normal,
            valid,
            std=0.05,
        )
        torch.testing.assert_close(reward, torch.tensor([1.0, math.exp(-1.0), 0.0]), atol=1.0e-6, rtol=0.0)

        with self.assertRaises(ValueError):
            reward_math.base_height_plane_tracking_exp(
                base_position,
                target_height,
                centroid,
                normal,
                valid,
                std=0.0,
            )

    def test_terrain_relative_feet_regulation_rewards_self_selected_clearance(self):
        radius = 0.128
        positions = torch.tensor([[[0.0, 0.0, radius], [0.0, 0.0, radius + 0.65]]])
        velocities = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
        centroids = torch.zeros_like(positions)
        normals = torch.zeros_like(positions)
        normals[..., 2] = 1.0
        valid = torch.ones(1, 2, dtype=torch.bool)

        penalty = reward_math.terrain_relative_feet_regulation(
            positions,
            velocities,
            centroids,
            normals,
            valid,
            foot_radius=radius,
            height_scale=0.65,
        )
        torch.testing.assert_close(penalty, torch.tensor([1.0 + math.exp(-1.0)]), atol=1.0e-6, rtol=0.0)

        normal_only_velocity = velocities.clone()
        normal_only_velocity[..., 0] = 0.0
        normal_only_velocity[..., 2] = -1.0
        no_tangential_penalty = reward_math.terrain_relative_feet_regulation(
            positions,
            normal_only_velocity,
            centroids,
            normals,
            valid,
            foot_radius=radius,
            height_scale=0.65,
        )
        torch.testing.assert_close(no_tangential_penalty, torch.zeros(1))

    def test_terrain_relative_landing_velocity_uses_local_clearance_and_contact(self):
        radius = 0.128
        positions = torch.tensor(
            [
                [[0.0, 0.0, radius + 0.04], [0.0, 0.0, radius + 0.04]],
                [[0.0, 0.0, radius + 0.10], [0.0, 0.0, radius + 0.04]],
            ]
        )
        velocities = torch.zeros_like(positions)
        velocities[..., 2] = torch.tensor([[-0.2, -0.3], [-0.4, 0.2]])
        centroids = torch.zeros_like(positions)
        normals = torch.zeros_like(positions)
        normals[..., 2] = 1.0
        valid = torch.ones(2, 2, dtype=torch.bool)
        in_contact = torch.tensor([[False, True], [False, False]])

        penalty = reward_math.terrain_relative_landing_velocity_l2(
            positions,
            velocities,
            centroids,
            normals,
            valid,
            in_contact,
            foot_radius=radius,
            about_landing_threshold=0.08,
        )
        torch.testing.assert_close(penalty, torch.tensor([0.04, 0.0]), atol=1.0e-6, rtol=0.0)

    def test_wheel_clearance_confidence_uses_radius_but_not_wheel_axis(self):
        positions = torch.tensor(
            [
                [[0.0, 0.17, 0.128], [0.0, -0.17, 0.1555]],
                [[0.0, 0.17, 0.163], [0.0, -0.17, 0.128]],
            ]
        )
        centroids = torch.zeros_like(positions)
        normals = torch.zeros_like(positions)
        normals[..., 2] = 1.0
        valid = torch.tensor([[True, True], [True, False]])

        confidence = reward_math.wheel_clearance_confidence(
            positions,
            centroids,
            normals,
            valid,
            wheel_radius=0.128,
            tolerance_on=0.02,
            tolerance_off=0.035,
        )
        torch.testing.assert_close(confidence, torch.tensor([[1.0, 0.5], [0.0, 0.0]]), atol=1.0e-5, rtol=0.0)

    def test_horizontal_neutral_penalty_ignores_vertical_adjustment(self):
        neutral = torch.tensor([[0.0, 0.17], [0.0, -0.17]])
        tolerance = torch.tensor([[0.04, 0.03], [0.04, 0.03]])
        scale = torch.tensor([[0.02, 0.02], [0.02, 0.02]])
        positions = torch.tensor([[[0.0, 0.17, -0.70], [0.0, -0.17, -0.90]]])

        no_penalty = reward_math.horizontal_neutral_penalty(positions, neutral, tolerance, scale)
        torch.testing.assert_close(no_penalty, torch.zeros(1))

        positions[:, 0, 0] = 0.08
        penalty = reward_math.horizontal_neutral_penalty(positions, neutral, tolerance, scale)
        torch.testing.assert_close(penalty, torch.tensor([1.5]))

    def test_wheel_target_symmetry_only_applies_near_zero_yaw(self):
        targets = torch.tensor([[1.0, 1.0], [1.0, -1.0], [1.0, 2.0]])
        yaw_commands = torch.tensor([0.0, 0.2, 0.04])
        penalty = reward_math.wheel_target_symmetry_penalty(
            targets,
            yaw_commands,
            yaw_threshold=0.05,
            target_difference_tolerance=0.1,
        )
        torch.testing.assert_close(penalty, torch.tensor([0.0, 0.0, 0.81]))

    def test_zero_command_wheel_target_penalty_has_dead_zone(self):
        targets = torch.tensor([[0.10, -0.15], [1.0, -1.0], [1.0, -1.0]])
        commands = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.2, 0.0, 0.0]])
        penalty = reward_math.zero_command_wheel_target_penalty(
            targets,
            commands,
            linear_threshold=0.05,
            angular_threshold=0.05,
            target_tolerance=0.15,
        )
        torch.testing.assert_close(penalty, torch.tensor([0.0, 1.445, 0.0]))

    def test_zero_command_yaw_rate_penalty_is_gated_by_all_commands(self):
        yaw_rate = torch.tensor([0.3, 0.3, 0.3])
        commands = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.1]])
        penalty = reward_math.zero_command_yaw_rate_penalty(
            yaw_rate,
            commands,
            linear_threshold=0.05,
            angular_threshold=0.05,
        )
        torch.testing.assert_close(penalty, torch.tensor([0.09, 0.0, 0.0]))

    def test_differential_rolling_error_handles_straight_reverse_and_yaw(self):
        radius = 0.2
        wheel_y = torch.tensor([[0.5, -0.5], [0.5, -0.5], [0.5, -0.5]])
        base_vx = torch.tensor([1.0, -1.0, 0.0])
        base_wz = torch.tensor([0.0, 0.0, 2.0])
        wheel_velocity = torch.tensor(
            [
                [5.0, 5.0],
                [-5.0, -5.0],
                [-5.0, 5.0],
            ]
        )

        penalty = reward_math.differential_wheel_rolling_error_l2(
            wheel_velocity,
            base_vx,
            base_wz,
            wheel_y,
            wheel_radius=radius,
        )
        torch.testing.assert_close(penalty, torch.zeros(3))

    def test_differential_rolling_error_is_per_wheel_for_arc_turns(self):
        radius = 0.2
        wheel_y = torch.tensor([[0.5, -0.5], [0.5, -0.5]])
        base_vx = torch.tensor([1.0, 1.0])
        base_wz = torch.tensor([1.0, 1.0])
        wheel_velocity = torch.tensor([[2.5, 7.5], [0.0, 0.0]])

        penalty = reward_math.differential_wheel_rolling_error_l2(
            wheel_velocity,
            base_vx,
            base_wz,
            wheel_y,
            wheel_radius=radius,
        )
        torch.testing.assert_close(penalty, torch.tensor([0.0, 1.25]))

    def test_rolling_contact_point_has_zero_slip(self):
        radius = 0.2
        linear_velocity = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
        angular_velocity = torch.tensor([[[0.0, 1.0 / radius, 0.0], [0.0, 1.0 / radius, 0.0]]])
        normal = torch.tensor([[0.0, 0.0, 1.0]])
        contact = torch.ones(1, 2)
        valid = torch.tensor([True])

        no_slip = reward_math.rolling_contact_slip_l2(
            linear_velocity,
            angular_velocity,
            normal,
            contact,
            wheel_radius=radius,
            plane_valid=valid,
        )
        torch.testing.assert_close(no_slip, torch.zeros(1), atol=1.0e-6, rtol=0.0)

        slipping = reward_math.rolling_contact_slip_l2(
            linear_velocity,
            torch.zeros_like(angular_velocity),
            normal,
            contact,
            wheel_radius=radius,
            plane_valid=valid,
        )
        torch.testing.assert_close(slipping, torch.ones(1))

    def test_landing_impact_only_counts_new_contacts(self):
        first_contact = torch.tensor([[True, False]])
        force = torch.tensor([[[0.0, 0.0, 150.0], [0.0, 0.0, 300.0]]])
        penalty = reward_math.landing_impact_l2(
            first_contact,
            force,
            force_threshold=100.0,
            force_scale=100.0,
        )
        torch.testing.assert_close(penalty, torch.tensor([0.25]))

    def test_terrain_orientation_penalty(self):
        base_up = torch.tensor([[0.0, 0.0, 1.0], [math.sqrt(0.5), 0.0, math.sqrt(0.5)]])
        normal = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        valid = torch.tensor([True, True])
        penalty = reward_math.terrain_orientation_penalty(base_up, normal, valid)
        torch.testing.assert_close(penalty, torch.tensor([0.0, 0.5]), atol=1.0e-6, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
