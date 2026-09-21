"""Velocity observations must use link axes, independent of inertia orientation."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import mujoco
import numpy as np

_PATH = Path(__file__).resolve().parents[1] / 'mujoco/deploy_wheel_policy.py'
_SPEC = importlib.util.spec_from_file_location('mujoco_velocity_frame_under_test', _PATH)
deployment = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = deployment
_SPEC.loader.exec_module(deployment)


class BaseVelocityFrameTest(unittest.TestCase):
    def make_model(self, inertial_quat='0.972578 -0.00615364 -0.232461 0.00398682'):
        model = mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
          <body name="base_Link" pos="0 0 1">
            <freejoint/>
            <inertial pos="0.045 0 -0.164" quat="{inertial_quat}"
                      mass="10" diaginertia="0.15 0.11 0.08"/>
          </body>
        </worldbody></mujoco>''')
        data = mujoco.MjData(model)
        # A rotated root makes accidental use of world axes observable too.
        quat = np.array([.8, .2, -.3, .4])
        data.qpos[3:7] = quat / np.linalg.norm(quat)
        return model, data, model.body('base_Link').id

    def test_each_angular_axis_is_preserved_despite_inertial_rotation(self):
        model, data, body = self.make_model()
        for axis in range(3):
            with self.subTest(axis=axis):
                data.qvel[:] = 0.
                data.qvel[3 + axis] = 1.
                mujoco.mj_forward(model, data)
                actual = np.zeros(6)
                deployment._base_velocity_in_body_frame(model, data, body, actual)
                np.testing.assert_allclose(actual[:3], np.eye(3)[axis], atol=1e-12)

    def test_linear_velocity_is_at_com_and_expressed_in_link_axes(self):
        model, data, body = self.make_model()
        data.qvel[:] = [.3, -.2, .1, .4, -.5, .6]
        mujoco.mj_forward(model, data)
        actual = np.zeros(6)
        deployment._base_velocity_in_body_frame(model, data, body, actual)
        rotation = data.xmat[body].reshape(3, 3)
        expected_linear = rotation.T @ data.qvel[:3] + np.cross(data.qvel[3:6], model.body_ipos[body])
        np.testing.assert_allclose(actual[:3], data.qvel[3:6], atol=1e-12)
        np.testing.assert_allclose(actual[3:], expected_linear, atol=1e-12)

    def test_same_motion_is_independent_of_principal_inertia_axes(self):
        velocities = []
        for quat in ['1 0 0 0', '0.972578 -0.00615364 -0.232461 0.00398682']:
            model, data, body = self.make_model(quat)
            data.qvel[:] = [.3, -.2, .1, .4, -.5, .6]
            mujoco.mj_forward(model, data)
            result = np.zeros(6)
            deployment._base_velocity_in_body_frame(model, data, body, result)
            velocities.append(result)
        np.testing.assert_allclose(*velocities, atol=1e-12)

    def test_post_step_velocity_matches_current_state_without_solving_dynamics(self):
        model, data, body = self.make_model()
        model.opt.timestep = .005
        data.qvel[:] = [.3, -.2, .1, .4, -.5, .6]
        data.xfrc_applied[body] = [10., -5., 20., 1., -2., 3.]
        mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)
        cached_position = data.xpos[body].copy()
        preserved = {name: getattr(data, name).copy()
                     for name in ('qpos', 'qvel', 'ctrl', 'qacc', 'qacc_warmstart', 'efc_force')}
        simulation_time = data.time

        reference = mujoco.MjData(model)
        mujoco.mj_copyData(reference, model, data)
        mujoco.mj_forward(model, reference)
        expected = np.zeros(6)
        mujoco.mj_objectVelocity(model, reference, mujoco.mjtObj.mjOBJ_BODY, body, expected, 0)
        rotation = reference.xmat[body].reshape(3, 3)
        expected[:3] = rotation.T @ expected[:3]
        expected[3:] = rotation.T @ expected[3:]

        actual = np.zeros(6)
        deployment._base_velocity_in_body_frame(model, data, body, actual)
        self.assertGreater(np.linalg.norm(data.xpos[body] - cached_position), 1e-6)
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        np.testing.assert_allclose(data.xmat, reference.xmat, atol=1e-12)
        self.assertEqual(data.time, simulation_time)
        for name, before in preserved.items():
            np.testing.assert_array_equal(getattr(data, name), before, err_msg=name)


if __name__ == '__main__':
    unittest.main()
