"""Sliding gait-cycle integration and the actual manager term, CPU/no Isaac Sim."""
from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import torch

from test_reward_math import reward_math
from test_foot_geometry_rewards import ManagerTermStub

_ROOT = Path(__file__).resolve().parents[1]
_PATH = _ROOT / 'exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/rewards.py'
_CLASS = next(n for n in ast.parse(_PATH.read_text()).body
              if isinstance(n, ast.ClassDef) and n.name == 'GaitCycleMeanTrackingPenalty')
_NS = dict(torch=torch, math=math, ManagerTermBase=ManagerTermStub,
           GaitCycleIntegral=reward_math.GaitCycleIntegral, SceneEntityCfg=lambda name: NS(name=name))
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
                              _CLASS], type_ignores=[])), str(_PATH), 'exec'), _NS)
Term = _NS['GaitCycleMeanTrackingPenalty']


def make_env(n=1, dt=.02):
    robot = NS(data=NS(root_lin_vel_b=torch.zeros(n, 3),
                       root_link_quat_w=torch.tensor([[1., 0., 0., 0.]]).repeat(n, 1),
                       root_ang_vel_b=torch.zeros(n, 3)))
    gait = NS(phase=torch.zeros(n), cfg=NS(continuous_phase=True, ranges=NS(frequencies=(1., 4.))))
    env = NS(num_envs=n, device='cpu', step_dt=dt, common_step_counter=0,
             scene={'robot': robot}, gait=gait, command=torch.zeros(n, 3))
    env.command_manager = NS(get_term=lambda name: gait, get_command=lambda name: env.command)
    return env


def make_term(env, component='vy', **overrides):
    params = dict(command_name='base_velocity', gait_command_name='gait_command',
                  component=component, asset_cfg=NS(name='robot'), error_scale=.2 if component == 'vy' else .3)
    params.update(overrides)
    return Term(NS(params=params), env), params


def advance(env, frequency=2., heading=None):
    env.common_step_counter += 1
    env.gait.phase[:] = (env.gait.phase + torch.as_tensor(frequency) * env.step_dt) % 1.
    if heading is not None:
        heading = torch.as_tensor(heading).expand(env.num_envs)
        quat = env.scene['robot'].data.root_link_quat_w
        quat.zero_(); quat[:, 0] = torch.cos(heading / 2); quat[:, 3] = torch.sin(heading / 2)


class GaitCycleIntegralTest(unittest.TestCase):
    def test_constant_drift_requires_a_full_cycle(self):
        w = reward_math.GaitCycleIntegral(1, 30, 'cpu')
        for i in range(25):
            actual = w.update(torch.tensor([.002]), torch.tensor([.04]), torch.tensor([True]), .02)
            if i < 24:
                self.assertEqual(actual.item(), 0.)
                self.assertFalse(w.ready.item())
        self.assertTrue(w.ready.item())
        torch.testing.assert_close(actual, torch.tensor([.05]))
        torch.testing.assert_close(w.duration, torch.tensor([.5]))

    def test_signed_cycle_cancellation_and_rolling_window(self):
        w = reward_math.GaitCycleIntegral(1, 12, 'cpu')
        for i in range(30):
            value = .01 if i % 10 < 5 else -.01
            w.update(torch.tensor([value]), torch.tensor([.1]), torch.tensor([True]), .02)
            if i >= 9:
                self.assertTrue(w.ready.item())
                self.assertAlmostEqual(w.integral.item(), 0., places=7)

    def test_fractional_oldest_sample_not_integer_rounded_period(self):
        w = reward_math.GaitCycleIntegral(1, 5, 'cpu')
        for _ in range(4):
            w.update(torch.tensor([.006]), torch.tensor([.3]), torch.tensor([True]), .02)
        torch.testing.assert_close(w.integral, torch.tensor([.02]))
        torch.testing.assert_close(w.duration, torch.tensor([.02 / .3]))

    def test_variable_frequency_matches_scalar_phase_window(self):
        w = reward_math.GaitCycleIntegral(2, 60, 'cpu')
        samples = [[], []]
        for i in range(180):
            spans = [.02 * (1.2 if i < 57 else 2.2), .02 * (2.2 if i < 71 else 1.2)]
            inc = [.002 * math.sin(i * .17), .001]
            w.update(torch.tensor(inc), torch.tensor(spans), torch.tensor([True, True]), .02)
            for j in range(2):
                samples[j].append((spans[j], inc[j]))
                remaining, expected, duration = 1., 0., 0.
                for phase, value in reversed(samples[j]):
                    fraction = min(remaining / phase, 1.)
                    expected += value * fraction; duration += .02 * fraction
                    remaining -= phase * fraction
                    if remaining < 1.e-10:
                        break
                self.assertEqual(w.ready[j].item(), remaining < 1.e-6)
                self.assertAlmostEqual(w.integral[j].item(), expected if remaining < 1.e-6 else 0., places=6)
                self.assertAlmostEqual(w.duration[j].item(), duration if remaining < 1.e-6 else 0., places=6)

    def test_disabled_interval_cannot_be_stitched_into_a_cycle(self):
        w = reward_math.GaitCycleIntegral(2, 12, 'cpu')
        for _ in range(10):
            w.update(torch.ones(2) * .001, torch.ones(2) * .1, torch.ones(2, dtype=torch.bool), .02)
        w.update(torch.ones(2) * .001, torch.ones(2) * .1, torch.tensor([False, True]), .02)
        self.assertFalse(w.ready[0]); self.assertTrue(w.ready[1])
        for i in range(10):
            w.update(torch.ones(2) * .001, torch.ones(2) * .1, torch.ones(2, dtype=torch.bool), .02)
            self.assertEqual(w.ready[0].item(), i == 9)

    def test_invalid_input_and_partial_reset_clear_only_affected_environments(self):
        w = reward_math.GaitCycleIntegral(3, 12, 'cpu')
        for _ in range(10):
            w.update(torch.ones(3), torch.ones(3) * .1, torch.ones(3, dtype=torch.bool), .02)
        w.reset([1])
        self.assertEqual(w.ready.tolist(), [True, False, True])
        w.update(torch.tensor([float('nan'), 1., 1.]), torch.tensor([.1, .1, 1.]),
                 torch.ones(3, dtype=torch.bool), .02)
        self.assertFalse(w.ready.any())
        self.assertTrue(torch.isfinite(w.increments).all())
        w.reset(); self.assertEqual(w.phase_spans.sum().item(), 0.)


class CycleDriftManagerTest(unittest.TestCase):
    def test_nonzero_commands_and_historical_command_changes(self):
        env = make_env(); term, p = make_term(env)
        term(env, **p)
        for i in range(25):
            env.command[:, 1] = .2 if i < 10 else -.3
            env.scene['robot'].data.root_lin_vel_b[:, 1] = env.command[:, 1] + .1
            advance(env); cost = term(env, **p)
            self.assertEqual(term.window.ready.item(), i == 24)
        self.assertAlmostEqual(cost.item(), .125, places=5)
        self.assertAlmostEqual((term.window.integral / term.window.duration).item(), .1, places=5)

    def test_mean_error_does_not_depend_on_frequency(self):
        for f in (1., 2., 4.):
            env = make_env(); term, p = make_term(env, 'vx', error_scale=.2)
            env.command[:, 0] = -.4
            env.scene['robot'].data.root_lin_vel_b[:, 0] = -.2
            term(env, **p)
            for _ in range(55):
                advance(env, frequency=f); cost = term(env, **p)
            self.assertAlmostEqual(cost.item(), .5, places=5)

    def test_heading_tracks_nonzero_command_across_wrap(self):
        env = make_env(); term, p = make_term(env, 'yaw')
        env.command[:, 2] = .4
        advance(env, heading=math.pi - .025); term(env, **p)
        for i in range(25):
            advance(env, heading=math.pi - .025 + (i + 1) * .008)
            cost = term(env, **p)
        self.assertAlmostEqual(cost.item(), 0., places=6)

    def test_heading_wrap_uses_heading_not_body_wz(self):
        env = make_env(); term, p = make_term(env, 'yaw')
        advance(env, heading=math.pi - .025); term(env, **p)
        env.scene['robot'].data.root_ang_vel_b[:, 2] = 999.  # Must not be integrated.
        for i in range(25):
            advance(env, heading=math.pi - .025 + (i + 1) * .002)
            term(env, **p)
        self.assertAlmostEqual(term.window.integral.item(), .05, places=5)

    def test_signed_heading_sway_and_body_lateral_sway_cancel(self):
        env = make_env(dt=.025); vy, vp = make_term(env); yaw, yp = make_term(env, 'yaw')
        vy(env, **vp); yaw(env, **yp)
        for i in range(60):
            env.scene['robot'].data.root_lin_vel_b[:, 1] = .1 if i % 20 < 10 else -.1
            advance(env, heading=.03 * math.sin(2 * math.pi * (i + 1) / 20))
            vy(env, **vp); yaw(env, **yp)
            if i >= 19:
                self.assertAlmostEqual(vy.window.integral.item(), 0., places=6)
                self.assertAlmostEqual(yaw.window.integral.item(), 0., places=6)

    def test_repeated_read_is_idempotent_and_partial_reset_discards_old_episode(self):
        env = make_env(2); term, p = make_term(env, 'yaw'); term(env, **p)
        for i in range(25):
            advance(env, heading=(i + 1) * .002); term(env, **p)
        before = term.window.phase_spans.clone(); result = term(env, **p).clone()
        torch.testing.assert_close(term.window.phase_spans, before)
        term.reset([0]); self.assertEqual(term(env, **p)[0].item(), 0.)
        advance(env, heading=torch.tensor([2., .052])); term(env, **p)
        self.assertFalse(term.window.ready[0]); self.assertTrue(term.window.ready[1])
        self.assertEqual(term.window.integral[0].item(), 0.)
        self.assertGreater(result[1].item(), 0.)
        term.reset([])

    def test_missing_steps_and_nonfinite_signals_do_not_bridge_history(self):
        env = make_env(2); term, p = make_term(env); term(env, **p)
        for _ in range(25):
            advance(env); term(env, **p)
        env.scene['robot'].data.root_lin_vel_b[0, 1] = float('nan')
        advance(env); cost = term(env, **p)
        self.assertTrue(torch.isfinite(cost).all())
        self.assertFalse(term.window.ready[0]); self.assertTrue(term.window.ready[1])
        env.common_step_counter += 2; term(env, **p)
        self.assertFalse(term.window.ready.any())

    def test_actual_config_registers_six_terms_and_removes_old_terms(self):
        path = _PATH.parent.parent / 'robots/limx_wheelfoot_mode_env_cfg.py'
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'WFFootAllTerrainEnvCfg')
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__post_init__')
        loop = next(n for n in init.body if isinstance(n, ast.For) and 'pen_cycle_mean_' in ast.unparse(n))
        cfg = NS(rewards=NS())
        scope = dict(self=cfg, RewTerm=lambda **kw: NS(**kw),
                     mdp=NS(GaitCycleMeanTrackingPenalty=Term), SceneEntityCfg=lambda name: NS(name=name))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[])), str(path), 'exec'), scope)
        self.assertEqual(len(vars(cfg.rewards)), 6)
        for component in ('vx', 'vy', 'yaw', 'height', 'roll', 'pitch'):
            term_cfg = getattr(cfg.rewards, 'pen_cycle_mean_' + component)
            self.assertEqual(term_cfg.weight, -1. if component in ('vx', 'vy', 'yaw') else -.1)
            Term(term_cfg, make_env())
        self.assertNotIn('pen_zero_vy_cycle_drift', ast.unparse(cls))
        self.assertNotIn('pen_zero_yaw_cycle_drift', ast.unparse(cls))

    def test_configuration_validation(self):
        env = make_env()
        for overrides in [dict(component='bad'), dict(error_scale=0.), dict(error_scale=float('nan'))]:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                make_term(env, **overrides)
        env.gait.cfg.continuous_phase = False
        with self.assertRaises(ValueError):
            make_term(env)

    def test_height_and_tilt_use_historical_terrain_reference(self):
        env = make_env()
        class Scene(dict):
            @property
            def sensors(self): return self
        env.scene = Scene(env.scene)
        env.scene['height_scanner'] = NS()
        env.scene['robot'].data.root_link_pos_w = torch.tensor([[0., 0., .82]])
        env.height = torch.tensor([[.8]])
        env.command_manager.get_command = lambda name: env.height if name == 'body_height' else env.command
        normal_b = torch.tensor([[0., 0., 1.]])
        _NS['math_utils'] = NS(quat_apply_inverse=lambda q, n: normal_b)
        _NS['_terrain_plane'] = lambda sensor: (torch.zeros(1, 3), torch.tensor([[0., 0., 1.]]), torch.tensor([True]))
        for component, scale in [('height', .02), ('roll', .05), ('pitch', .05)]:
            term, p = make_term(env, component, error_scale=scale)
            if component == 'roll': normal_b[:] = torch.tensor([[0., math.sin(.05), math.cos(.05)]])
            if component == 'pitch': normal_b[:] = torch.tensor([[-math.sin(.05), 0., math.cos(.05)]])
            term(env, **p)
            for _ in range(25):
                advance(env); cost = term(env, **p)
            self.assertAlmostEqual(cost.item(), .5, places=5)
        # A changing height target matched at every sample must not be compared
        # retroactively with just the final target.
        term, p = make_term(env, 'height', error_scale=.02); term(env, **p)
        for i in range(25):
            env.height[:] = .7 + i * .002
            env.scene['robot'].data.root_link_pos_w[:, 2] = env.height[:, 0]
            advance(env); cost = term(env, **p)
        self.assertAlmostEqual(cost.item(), 0., places=6)


if __name__ == '__main__':
    unittest.main()
