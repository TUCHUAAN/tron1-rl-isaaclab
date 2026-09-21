"""Unit tests for dual-expert action routing."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest

import torch


_MODULE_PATH = (
    pathlib.Path(__file__).parents[1]
    / "exts/bipedal_locomotion/bipedal_locomotion/utils/wf_mode_fsm.py"
)
_SPEC = importlib.util.spec_from_file_location("wf_mode_fsm_test_module", _MODULE_PATH)
fsm_module = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = fsm_module
_SPEC.loader.exec_module(fsm_module)


class WheelfootModeFSMTest(unittest.TestCase):
    def setUp(self) -> None:
        self.wheel = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
        self.foot = torch.tensor([[-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -2.0, 2.0]])
        self.bounded_foot = self.foot.clone()
        self.bounded_foot[..., -2:] = 0.0

    def test_foot_mode_always_routes_zero_wheel_targets(self):
        fsm = fsm_module.WheelfootModeFSM(initial_mode="foot")
        torch.testing.assert_close(fsm.route_actions(self.wheel, self.foot), self.bounded_foot)

    def test_transition_blends_all_eight_dimensions(self):
        fsm = fsm_module.WheelfootModeFSM(
            cfg=fsm_module.WheelfootModeFSMCfg(transition_steps=4),
            initial_mode="wheel",
        )
        fsm.mode = fsm_module.WheelfootMode.WHEEL_TO_FOOT
        fsm.transition_step = 2
        expected = 0.5 * self.wheel + 0.5 * self.bounded_foot
        torch.testing.assert_close(fsm.route_actions(self.wheel, self.foot), expected)


if __name__ == "__main__":
    unittest.main()
