"""Unit tests for B2W-style atomic mode management."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest

import torch


_MODULE_PATH = pathlib.Path(__file__).parents[1] / "exts/bipedal_locomotion/bipedal_locomotion/utils/wf_mode_fsm.py"
_SPEC = importlib.util.spec_from_file_location("wf_mode_fsm_test_module", _MODULE_PATH)
fsm_module = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = fsm_module
_SPEC.loader.exec_module(fsm_module)


class WheelfootModeFSMTest(unittest.TestCase):
    def _stable(self, fsm, *, foot_cycle_complete=True):
        return fsm.update(
            base_speed=0.0,
            wheel_speed=0.0,
            both_wheels_contact=True,
            upright_cosine=1.0,
            foot_cycle_complete=foot_cycle_complete,
        )

    def test_routes_only_active_expert_without_blending(self):
        wheel = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
        foot = torch.tensor([[-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -2.0, 2.0]])
        fsm = fsm_module.WheelfootModeFSM(initial_mode="wheel")
        torch.testing.assert_close(fsm.route_actions(wheel, foot), wheel)
        fsm.request("foot")
        fsm.cfg.min_dwell_steps = 0
        self._stable(fsm)
        expected = foot.clone()
        expected[..., -2:] = 0.0
        torch.testing.assert_close(fsm.route_actions(wheel, foot), expected)

    def test_foot_to_wheel_waits_for_gait_cycle(self):
        fsm = fsm_module.WheelfootModeFSM(
            cfg=fsm_module.WheelfootModeFSMCfg(min_dwell_steps=0), initial_mode="foot"
        )
        fsm.request("wheel")
        self.assertEqual(self._stable(fsm, foot_cycle_complete=False), fsm_module.WheelfootMode.FOOT)
        self.assertEqual(self._stable(fsm, foot_cycle_complete=True), fsm_module.WheelfootMode.WHEEL)


if __name__ == "__main__":
    unittest.main()
