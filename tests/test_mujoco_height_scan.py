"""Terrain observations must match training hit/miss semantics and current state."""
from __future__ import annotations

from collections import defaultdict, deque
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np

_PATH = Path(__file__).resolve().parents[1] / "mujoco/deploy_wheel_policy.py"
_SPEC = importlib.util.spec_from_file_location("mujoco_scan_under_test", _PATH)
deployment = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = deployment
_SPEC.loader.exec_module(deployment)


class HeightScanTest(unittest.TestCase):
    def scene(self, terrain):
        model = mujoco.MjModel.from_xml_string(f'''<mujoco><compiler angle="radian"/>
          <option timestep="0.005" gravity="0 0 0"/><worldbody>{terrain}
          <body name="base_Link" pos="0 0 1"><freejoint/>
            <inertial mass="1" pos="0 0 0" diaginertia="1 1 1"/>
          </body></worldbody></mujoco>''')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return model, data, deployment.TerrainHeightScanner(model, 1)

    def test_all_misses_match_clipped_training_value_and_keep_plane_invalid(self):
        _, data, scanner = self.scene("")
        expected = np.clip(data.qpos[2] - np.full(121, np.inf) - .5, 0., 10.)
        np.testing.assert_array_equal(scanner.scan(data), expected)
        self.assertFalse(scanner.last_plane_valid)
        self.assertTrue(np.isnan(scanner.body_height(data)))

    def test_partial_misses_do_not_enter_plane_fit(self):
        # Only the middle of the scan sees this platform. Its top is z=0.2.
        _, data, scanner = self.scene('<geom type="box" pos="0 0 .1" size=".21 .21 .1"/>')
        actual = scanner.scan(data)
        inside = (np.abs(scanner.offsets[:, 0]) < .21) & (np.abs(scanner.offsets[:, 1]) < .21)
        np.testing.assert_allclose(actual[inside], .3, atol=1e-7)
        np.testing.assert_array_equal(actual[~inside], 0.)
        self.assertTrue(scanner.last_plane_valid)
        np.testing.assert_allclose(scanner.last_plane_centroid[2], .2, atol=1e-12)
        np.testing.assert_allclose(scanner.last_plane_normal, [0, 0, 1], atol=1e-12)
        self.assertAlmostEqual(scanner.body_height(data), .8)

    def test_tilted_terrain_matches_analytic_scan_at_nonzero_orientation(self):
        model, data, scanner = self.scene('<geom type="plane" size="0 0 .1" euler="0 .3 0"/>')
        data.qpos[3:7] = np.array([.8, .2, -.3, .4]) / np.linalg.norm([.8, .2, -.3, .4])
        mujoco.mj_forward(model, data)
        yaw = np.arctan2(data.xmat[1, 3], data.xmat[1, 0])
        world_x = np.cos(yaw) * scanner.offsets[:, 0] - np.sin(yaw) * scanner.offsets[:, 1]
        hit_z = -np.tan(.3) * world_x
        np.testing.assert_allclose(scanner.scan(data), np.clip(1. - hit_z - .5, 0., 10.), atol=1e-7)

    def test_scan_refreshes_post_step_origin(self):
        model, data, scanner = self.scene('<geom type="plane" size="0 0 .1"/>')
        data.qvel[2] = 1.
        mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)
        self.assertNotEqual(data.xpos[1, 2], data.qpos[2])
        np.testing.assert_allclose(scanner.scan(data), data.qpos[2] - .5, atol=1e-7)

    def test_telemetry_refreshes_terrain_and_velocity_without_opening_gui(self):
        model, data, scanner = self.scene('<geom type="plane" size="0 0 .1"/>')
        plot = deployment.TelemetryPlot.__new__(deployment.TelemetryPlot)
        plot._closed = False
        plot.root = SimpleNamespace(winfo_exists=lambda: True)
        plot.model, plot.base_body_id, plot.scanner = model, 1, scanner
        plot.velocity_buffer = np.zeros(6)
        plot.times, plot.values = deque(), defaultdict(deque)
        plot.history_seconds = 20.
        data.qvel[:3] = [.2, -.1, .4]
        mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)
        plot.sample(data, np.array([0., 0., 0., .8]), data.time)
        self.assertAlmostEqual(plot.values['height'][-1], data.qpos[2])
        self.assertAlmostEqual(plot.values['vx'][-1], .2)
        self.assertAlmostEqual(plot.values['vy'][-1], -.1)


if __name__ == "__main__":
    unittest.main()
