"""Foot deployment defaults must follow the loaded checkpoint's gait inputs."""
from __future__ import annotations

import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from collections import deque
from types import SimpleNamespace as NS

import numpy as np
import torch

from test_gait_phase import make_gait, advance

_PATH = Path(__file__).resolve().parents[1] / "mujoco/deploy_wheel_policy.py"
_SPEC = importlib.util.spec_from_file_location("mujoco_gait_under_test", _PATH)
deployment = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = deployment
_SPEC.loader.exec_module(deployment)

_SNAPSHOT = """scene:
  robot: !!python/object:unused.training.Class {}
commands:
  gait_command:
    ranges:
      frequencies: !!python/tuple [1.6, 2.0]
      offsets: !!python/tuple [0.5, 0.5]
      durations: !!python/tuple [0.5, 0.5]
      swing_height: !!python/tuple [0.0, 0.0]
"""


class GaitCommandTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.checkpoint = Path(self.temp.name) / "model_20000.pt"
        self.config = self.checkpoint.parent / "params/env.yaml"
        self.config.parent.mkdir()
        self.config.write_text(_SNAPSHOT, encoding="utf-8")

    def resolve(self, *argv):
        args = deployment.parse_args(list(argv))
        return deployment.resolve_gait_command(args, self.checkpoint)

    def test_foot_uses_snapshot_midpoints_and_fixed_values(self):
        np.testing.assert_allclose(self.resolve("--mode", "foot"), [1.8, .5, .5, 0.])

    def test_partial_explicit_override_preserves_other_snapshot_values(self):
        np.testing.assert_allclose(
            self.resolve("--mode", "foot", "--swing-height", "0.12"),
            [1.8, .5, .5, .12],
        )

    def test_all_explicit_values_do_not_require_a_readable_snapshot(self):
        self.config.write_text("invalid: [", encoding="utf-8")
        np.testing.assert_allclose(
            self.resolve("--mode", "foot", "--gait-frequency", "2.1", "--gait-offset", "0.4",
                         "--gait-duration", "0.55", "--swing-height", "0.1"),
            [2.1, .4, .55, .1],
        )

    def test_missing_snapshot_uses_foot_defaults(self):
        self.config.unlink()
        with contextlib.redirect_stdout(io.StringIO()):
            actual = self.resolve("--mode", "foot")
        np.testing.assert_allclose(actual, [1.7, .5, .5, 0.])

    def test_wheel_keeps_its_existing_defaults_and_cli_override(self):
        np.testing.assert_allclose(self.resolve("--mode", "wheel"), [1.7, .5, .525, .14])
        np.testing.assert_allclose(
            self.resolve("--mode", "wheel", "--gait-duration", "0.6"),
            [1.7, .5, .6, .14],
        )

    def test_invalid_snapshot_range_is_not_silently_used(self):
        for replacement in ("[2.0, 1.6]", "[nan, 2.0]", "[1.6]"):
            with self.subTest(replacement=replacement):
                self.config.write_text(_SNAPSHOT.replace("[1.6, 2.0]", replacement), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "步态范围"):
                    self.resolve("--mode", "foot")

    def test_clock_convention_reads_snapshot_and_defaults_to_legacy(self):
        self.assertFalse(deployment.checkpoint_continuous_gait_phase(self.checkpoint))
        for value, expected in (("true", True), ("false", False), ("True", True)):
            self.config.write_text(_SNAPSHOT.replace("    ranges:", f"    continuous_phase: {value}\n    ranges:"))
            self.assertEqual(deployment.checkpoint_continuous_gait_phase(self.checkpoint), expected)
        self.config.unlink()
        self.assertFalse(deployment.checkpoint_continuous_gait_phase(self.checkpoint))

    def test_invalid_clock_metadata_fails_clearly(self):
        for value in ("maybe", "1", "null"):
            self.config.write_text(_SNAPSHOT.replace("    ranges:", f"    continuous_phase: {value}\n    ranges:"))
            with self.assertRaisesRegex(ValueError, "continuous_phase"):
                deployment.checkpoint_continuous_gait_phase(self.checkpoint)


class DeploymentPhaseTest(unittest.TestCase):
    def policy(self, continuous=True):
        # Run the actual act/history/reset path with deterministic networks and
        # sensor values. Clock assertions are independent of learned weights.
        policy = deployment.WheelPolicy.__new__(deployment.WheelPolicy)
        policy.device = torch.device("cpu")
        policy.continuous_gait_phase = continuous
        policy.body_height_min, policy.body_height_max = .65, .85
        policy.last_action = np.zeros(deployment.ACTION_DIM, dtype=np.float32)
        policy.history = deque(maxlen=deployment.HISTORY_LENGTH)
        common_dim = deployment.HISTORY_OBS_DIM - 6
        policy._common_observation = lambda data: np.zeros(common_dim, dtype=np.float32)
        policy.scanner = NS(scan=lambda data: np.zeros(121, dtype=np.float32))
        policy.encoder = lambda history: torch.zeros(1, 4)
        policy.actor = lambda obs: torch.zeros(1, deployment.ACTION_DIM)
        policy.reset(None, np.array([1.2, .5, .5, 0.], dtype=np.float32))
        return policy

    def test_training_and_deployment_agree_across_frequency_changes_and_reset(self):
        for continuous in (False, True):
            env, term = make_gait(n=1, continuous=continuous, dt=deployment.POLICY_DT)
            policy = self.policy(continuous)
            for step in range(150):
                if step in (17, 51, 103):
                    frequency = 2.2 if step == 17 else 1.6
                    term.cfg.ranges.frequencies = (frequency, frequency)
                    term._resample_command([0])
                gait = term.command[0].numpy().copy()
                phase = term.phase.item()
                expected = np.array([np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)])
                # Repeated reads must not move either observation clock.
                for _ in range(3):
                    np.testing.assert_allclose(policy._gait_phase(float(gait[0])), expected, atol=3e-6)
                policy.act(None, np.array([0., 0., .4, .8], dtype=np.float32), gait)
                np.testing.assert_allclose(policy.history[-1][-6:-4], expected, atol=3e-6)
                advance(env)
                term._update_command()
                if step == 80:
                    term.reset()
                    env.episode_length_buf.zero_()
                    policy.reset(None, term.command[0].numpy())
                    np.testing.assert_array_equal(policy._gait_phase(2.), [0., 1.])

    def test_new_frequency_does_not_rewrite_elapsed_deployment_phase(self):
        policy = self.policy()
        gait = np.array([1.2, .5, .5, 0.], dtype=np.float32)
        for _ in range(19):
            policy.act(None, np.array([0., 0., 0., .8], dtype=np.float32), gait)
        np.testing.assert_array_equal(policy._gait_phase(1.2), policy._gait_phase(2.2))


if __name__ == "__main__":
    unittest.main()
