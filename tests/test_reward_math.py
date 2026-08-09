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
    def test_contact_confidence_uses_history_and_schmitt_band(self):
        force = torch.zeros(2, 4, 2, 3)
        force[0, 0, 0, 2] = 15.0
        force[1, 0, 0, 2] = 7.5
        force[1, 3, 1, 2] = 10.0

        confidence = reward_math.contact_confidence_from_force_history(force, force_off=5.0, force_on=10.0)
        expected = torch.tensor([[1.0, 0.0], [0.5, 1.0]])
        torch.testing.assert_close(confidence, expected)
        torch.testing.assert_close(reward_math.all_support_confidence(confidence), torch.tensor([0.0, 0.5]))

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
