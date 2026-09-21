"""Exercise the actual PI methods in both simulators without launching PhysX."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import torch
import numpy as np
from test_mujoco_velocity_frame import deployment

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/actions.py'

class JointStub:
    def process_actions(self, actions):
        self._processed_actions = actions.clone()
    @property
    def processed_actions(self): return self._processed_actions

node = next(n for n in ast.parse(PATH.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'WheelVelocityPIAction')
ns = dict(torch=torch, JointAction=JointStub)
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])), str(PATH), 'exec'), ns)

class FixedWheelsTest(unittest.TestCase):
    def isaac(self, fixed):
        obj = object.__new__(ns['WheelVelocityPIAction'])
        obj.cfg = NS(fixed_zero_target=fixed, effort_limit=80.)
        obj._joint_ids = [0, 1]
        obj._physics_dt = .005
        obj._integral_error = torch.zeros(1, 2)
        obj._kp = torch.full((1, 2), 2.)
        obj._ki = torch.full((1, 2), .5)
        obj._asset = NS(data=NS(joint_vel=torch.tensor([[1., -1.]])),
                        set_joint_effort_target=lambda value, **kw: setattr(obj, 'effort', value.clone()))
        return obj

    def mujoco(self, mode):
        obj = object.__new__(deployment.TorqueController)
        obj.control_mode = mode
        obj.leg_qpos_addresses = obj.leg_dof_addresses = obj.leg_actuator_ids = np.arange(6)
        obj.wheel_dof_addresses = obj.wheel_actuator_ids = np.arange(6, 8)
        obj.default_leg_positions = np.zeros(6)
        obj.wheel_integral_error = np.zeros(2)
        obj.wheel_kp, obj.wheel_ki, obj.control_dt = 2., .5, .005
        obj.low_pass_enabled = False
        data = NS(qpos=np.zeros(8), qvel=np.array([0.] * 6 + [1., -1.]), ctrl=np.zeros(8))
        return obj, data

    def test_foot_ignores_policy_outputs_and_pi_matches_between_simulators(self):
        for targets in ([10., -20.], [-3., 4.], [0., 0.]):
            isaac = self.isaac(True)
            controller, data = self.mujoco('foot')
            isaac.process_actions(torch.tensor([targets]))
            self.assertEqual(isaac.processed_actions.abs().sum().item(), 0.)
            for _ in range(10):
                isaac.apply_actions()
                controller.apply(data, np.array([0.] * 6 + list(targets)))
                np.testing.assert_allclose(data.ctrl[6:], isaac.effort.numpy()[0], rtol=1.e-6)
            self.assertLess(data.ctrl[6], 0.)
            self.assertGreater(data.ctrl[7], 0.)

    def test_wheel_keeps_policy_targets(self):
        isaac = self.isaac(False)
        isaac.process_actions(torch.tensor([[2., -2.]])); isaac.apply_actions()
        controller, data = self.mujoco('wheel')
        controller.apply(data, np.array([0.] * 6 + [2., -2.]))
        np.testing.assert_allclose(data.ctrl[6:], isaac.effort.numpy()[0], rtol=1.e-6)
        self.assertGreater(data.ctrl[6], 0.)

if __name__ == '__main__': unittest.main()
