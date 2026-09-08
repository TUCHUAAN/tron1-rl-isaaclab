"""Unit tests for the Wheel terrain progression decisions."""

from __future__ import annotations

import importlib.util
import pathlib
import unittest

import torch


_MODULE_PATH = (
    pathlib.Path(__file__).parents[1]
    / "exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/curriculums.py"
)
_SPEC = importlib.util.spec_from_file_location("wf_curriculums", _MODULE_PATH)
curriculums = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(curriculums)


class WheelTerrainCurriculumTest(unittest.TestCase):
    def test_promotion_requires_distance_tracking_and_support(self):
        move_up, move_down = curriculums.wheel_terrain_progression_masks(
            distance=torch.tensor([4.1, 4.1, 4.1, 4.1]),
            failed=torch.tensor([False, False, False, False]),
            moving_command_rate=torch.tensor([0.8, 0.8, 0.8, 0.1]),
            linear_tracking=torch.tensor([0.8, 0.4, 0.8, 0.8]),
            support_confidence=torch.tensor([0.9, 0.9, 0.5, 0.9]),
            terrain_length=8.0,
            distance_fraction_up=0.5,
            moving_rate_threshold=0.25,
            tracking_up=0.55,
            tracking_down=0.25,
            support_up=0.75,
        )
        torch.testing.assert_close(move_up, torch.tensor([True, False, False, False]))
        torch.testing.assert_close(move_down, torch.zeros(4, dtype=torch.bool))

    def test_fall_or_poor_moving_tracking_demotes_but_standing_holds(self):
        move_up, move_down = curriculums.wheel_terrain_progression_masks(
            distance=torch.tensor([5.0, 1.0, 0.0, 0.0]),
            failed=torch.tensor([True, False, False, False]),
            moving_command_rate=torch.tensor([0.8, 0.8, 0.1, 0.8]),
            linear_tracking=torch.tensor([0.9, 0.1, 0.1, 0.3]),
            support_confidence=torch.tensor([0.9, 0.9, 0.0, 0.9]),
            terrain_length=8.0,
            distance_fraction_up=0.5,
            moving_rate_threshold=0.25,
            tracking_up=0.55,
            tracking_down=0.25,
            support_up=0.75,
        )
        torch.testing.assert_close(move_up, torch.zeros(4, dtype=torch.bool))
        torch.testing.assert_close(move_down, torch.tensor([True, True, False, False]))


if __name__ == "__main__":
    unittest.main()
