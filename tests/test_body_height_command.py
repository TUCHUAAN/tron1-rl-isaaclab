"""Exercise the actual height sampling methods on CPU without launching Isaac Sim."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import torch

_PATH = (Path(__file__).resolve().parents[1]
         / "exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/commands/body_height_command.py")
_TREE = ast.parse(_PATH.read_text())
_CLASS = next(n for n in _TREE.body if isinstance(n, ast.ClassDef) and n.name == "BodyHeightCommand")
_METHODS = [n for n in _CLASS.body if isinstance(n, ast.FunctionDef)
            and n.name in ("_resample_command", "_update_command")]
_MODULE = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                          *_METHODS], type_ignores=[])
_NAMESPACE = {"torch": torch}
exec(compile(ast.fix_missing_locations(_MODULE), str(_PATH), "exec"), _NAMESPACE)
resample = _NAMESPACE["_resample_command"]
update = _NAMESPACE["_update_command"]


class HeightCommandTest(unittest.TestCase):
    def command(self, count=4096, endpoint_fraction=.2):
        return NS(device="cpu", cfg=NS(ranges=NS(height=(.65, .85)), endpoint_fraction=endpoint_fraction,
                                       max_rate=.06),
                  _target=torch.full((count, 1), .75), _command=torch.full((count, 1), .75),
                  _env=NS(step_dt=.02))

    def setUp(self):
        torch.manual_seed(42)

    def test_indexed_reset_writes_selected_targets_only(self):
        for ids in (torch.tensor([1, 3, 5]), [1, 3, 5]):
            with self.subTest(ids=ids):
                command = self.command(6, 0.)
                resample(command, ids)
                self.assertTrue(torch.all(command._target[ids, 0] != .75))
                self.assertTrue(torch.all((command._target >= .65) & (command._target <= .85)))
                torch.testing.assert_close(command._target[[0, 2, 4]], torch.full((3, 1), .75))
                torch.testing.assert_close(command._command, torch.full((6, 1), .75))

    def test_uniform_targets_cover_the_interior_without_endpoint_sampling(self):
        command = self.command(endpoint_fraction=0.)
        resample(command, torch.arange(4096))
        self.assertGreater(torch.unique(command._target).numel(), 4000)
        self.assertLess(abs(command._target.mean().item() - .75), .003)
        for lo, hi in ((.65, .70), (.70, .75), (.75, .80), (.80, .85)):
            fraction = ((command._target >= lo) & (command._target < hi)).float().mean().item()
            self.assertLess(abs(fraction - .25), .04)

    def test_repeated_sampling_preserves_continuous_and_endpoint_mixture(self):
        command = self.command()
        for _ in range(100):
            resample(command, torch.arange(4096))
        low = (command._target == .65).float().mean().item()
        high = (command._target == .85).float().mean().item()
        self.assertLess(abs(low - .1), .025)
        self.assertLess(abs(high - .1), .025)
        self.assertGreater(1. - low - high, .75)

    def test_endpoint_only_and_empty_resets(self):
        command = self.command(endpoint_fraction=1.)
        resample(command, torch.arange(4096))
        self.assertTrue(torch.all((command._target == .65) | (command._target == .85)))
        before = command._target.clone()
        resample(command, torch.tensor([], dtype=torch.long))
        torch.testing.assert_close(command._target, before)

    def test_smoothing_still_limits_rate_and_stops_at_target(self):
        command = self.command(3)
        command._target[:, 0] = torch.tensor([.65, .85, .7505])
        update(command)
        torch.testing.assert_close(command._command[:, 0], torch.tensor([.7488, .7512, .7505]))
        for _ in range(100):
            update(command)
        torch.testing.assert_close(command._command, command._target)


if __name__ == "__main__":
    unittest.main()
