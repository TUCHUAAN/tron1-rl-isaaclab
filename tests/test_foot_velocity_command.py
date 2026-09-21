"""CPU checks for the actual Foot sampler and conditional diagnostics."""
from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import torch

import test_foot_geometry_rewards as geometry
from test_gait_phase import CommandTermStub


class UniformVelocityStub(CommandTermStub):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        self.vel_command_b = torch.zeros(env.num_envs, 3)
        self.is_standing_env = torch.zeros(env.num_envs, dtype=torch.bool)
        self.is_heading_env = torch.zeros_like(self.is_standing_env)

    @property
    def command(self):
        return self.vel_command_b

    def _resample_command(self, env_ids):
        for i, limits in enumerate((self.cfg.ranges.lin_vel_x, self.cfg.ranges.lin_vel_y, self.cfg.ranges.ang_vel_z)):
            self.vel_command_b[env_ids, i] = torch.empty(len(env_ids)).uniform_(*limits)
        self.is_standing_env[env_ids] = torch.rand(len(env_ids)) <= self.cfg.rel_standing_envs

    def _update_command(self):
        self.vel_command_b[self.is_standing_env] = 0.

    def _update_metrics(self):
        pass


class EntityStub:
    def __init__(self, name, **kwargs):
        self.name = name
        self.body_ids = [0, 1]
        self.joint_ids = list(range(6))

    def resolve(self, scene):
        pass


_PATH = (Path(__file__).resolve().parents[1]
         / "exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/commands/foot_velocity_command.py")
_CLASS = next(n for n in ast.parse(_PATH.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "FootVelocityCommand")
_MODULE = ast.Module(body=[
    ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), _CLASS,
], type_ignores=[])
_NS = dict(torch=torch, math=math, UniformVelocityCommand=UniformVelocityStub, SceneEntityCfg=EntityStub,
           _wheel_terrain_planes=geometry._NS["_wheel_terrain_planes"])
exec(compile(ast.fix_missing_locations(_MODULE), str(_PATH), "exec"), _NS)
FootVelocityCommand = _NS["FootVelocityCommand"]


def make_env(n=2, **overrides):
    env = geometry.FootGeometryRewardsTest().make_env(n)
    data = env.scene["robot"].data
    data.joint_pos = torch.zeros(n, 6)
    data.soft_joint_pos_limits = torch.tensor([-1., 1.]).repeat(n, 6, 1)
    env.raw_actions = torch.zeros(n, 2)
    env.action_manager = NS(get_term=lambda name: NS(raw_actions=env.raw_actions))
    env.swing_term, params = geometry.FootGeometryRewardsTest().swing_term(env)
    env.swing_term(env, **params)
    env.reward_manager = NS(get_term_cfg=lambda name: NS(func=env.swing_term))
    cfg = NS(asset_name="robot", rel_standing_envs=.15, rel_straight_envs=.25, rel_lateral_envs=.10,
             rel_yaw_only_envs=.20, rel_mixed_envs=.30, rel_heading_envs=0.,
             wheel_body_names=("L", "R"), leg_joint_names=tuple(range(6)),
             terrain_sensor_names=("scan_L", "scan_R"), wheel_radius=.128,
             gait_command_name="gait_command", body_height_command_name="body_height",
             swing_reward_name="rew_swing_clearance", wheel_action_name="joint_vel", wheel_target_limit=1.,
             height_bin_edges=(.7, .8), ranges=NS(lin_vel_x=(-.8, .8), lin_vel_y=(-.4, .4), ang_vel_z=(-.8, .8)))
    vars(cfg).update(overrides)
    term = FootVelocityCommand(cfg, env)
    term.reset()
    return env, term


class FootVelocityCommandTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_five_modes_have_requested_probabilities_and_balanced_signs(self):
        _, term = make_env(20000)
        for i, (fraction, active) in enumerate(zip((.15, .25, .10, .20, .30), ((), (0,), (1,), (2,), (0, 1, 2)))):
            selected = term.mode == i
            self.assertLess(abs(selected.float().mean().item() - fraction), .015)
            commands = term.command[selected]
            for j in range(3):
                if j in active:
                    self.assertLess(abs((commands[:, j] > 0).float().mean().item() - .5), .035)
                else:
                    self.assertTrue((commands[:, j] == 0.).all())
        before = term.command.clone()
        term._update_command()
        torch.testing.assert_close(term.command, before)
        torch.testing.assert_close(term.is_standing_env, term.mode == 0)
        self.assertFalse(term.is_heading_env.any())

    def test_partial_resample_and_reset_preserve_other_environments(self):
        env, term = make_env(6)
        term._update_metrics()
        before = term.command.clone()
        counts = term._mode_counts.clone()
        term.reset([1, 3, 5])
        torch.testing.assert_close(term.command[[0, 2, 4]], before[[0, 2, 4]])
        torch.testing.assert_close(term._mode_counts[[0, 2, 4]], counts[[0, 2, 4]])
        self.assertTrue((term._mode_counts[[1, 3, 5]] == 0).all())
        self.assertEqual(term.reset([]), {})

    def test_cycle_metrics_use_only_complete_live_windows_and_reset_independently(self):
        env, term = make_env(2)
        base_get = env.reward_manager.get_term_cfg
        windows = {
            "pen_cycle_mean_vy": NS(ready=torch.tensor([True, False]),
                integral=torch.tensor([-.02, 99.]), duration=torch.tensor([.5, 0.])),
            "pen_cycle_mean_yaw": NS(ready=torch.tensor([False, True]),
                integral=torch.tensor([99., .03]), duration=torch.tensor([0., .6])),
        }
        env.reward_manager.active_terms = list(windows)
        env.reward_manager.get_term_cfg = lambda name: (
            NS(func=NS(window=windows[name])) if name in windows else base_get(name))
        term._update_metrics()
        stats = term.diagnostic_means()
        self.assertAlmostEqual(stats["cycle_mean_vy/abs_error_m_s"], .04, places=6)
        self.assertAlmostEqual(stats["cycle_mean_yaw/rms_error_rad_s"], .05, places=6)
        self.assertEqual(stats["cycle_mean_vy/valid_fraction"], .5)
        self.assertEqual(stats["cycle_mean_yaw/samples"], 1.)
        term.reset([0])
        self.assertEqual(term.diagnostic_means()["cycle_mean_vy/samples"], 0.)
        self.assertEqual(term.diagnostic_means()["cycle_mean_yaw/samples"], 1.)
        term.reset()
        self.assertEqual(term._cycle_sums.sum().item(), 0.)

    def test_invalid_configuration_is_rejected(self):
        for change in (dict(rel_mixed_envs=.4), dict(rel_mixed_envs=float("nan")),
                       dict(rel_lateral_envs=-.1), dict(rel_heading_envs=.1), dict(height_bin_edges=(.8, .7))):
            with self.subTest(change=change), self.assertRaises(ValueError):
                make_env(**change)

    def test_height_bins_follow_each_step_and_include_dragging_swings(self):
        env, term = make_env(1)
        # At phase .75 the left foot should lift .08 m, but it stays on its local terrain.
        env.scene["robot"].data.body_link_pos_w[:, 0, 2] = .328
        env.scene["robot"].data.joint_pos[:, -1] = 1.1
        for height in (.65, .75, .85):
            env.commands["body_height"][:] = height
            term._update_metrics()
        stats = term.diagnostic_means()
        for name in ("low", "middle", "high"):
            self.assertEqual(stats[f"height_{name}/swing_samples"], 1.)
            self.assertEqual(stats[f"height_{name}/step_samples"], 1.)
            self.assertAlmostEqual(stats[f"height_{name}/swing_shortfall_m"], .08, places=6)
            self.assertAlmostEqual(stats[f"height_{name}/swing_target_m"], .08, places=6)
            self.assertAlmostEqual(stats[f"height_{name}/leg_soft_limit_excess_rad"], .1, places=6)
        for height, expected in ((.7, [0., 1., 0.]), (.8, [0., 0., 1.])):
            term.reset()
            env.commands["body_height"][:] = height
            term._update_metrics()
            torch.testing.assert_close(term._height_counts[0, :, 1], torch.tensor(expected))

    def test_mode_statistics_use_conditional_denominators(self):
        env, term = make_env(1)
        term.mode[:] = 3
        term.vel_command_b[:] = torch.tensor([0., 0., .4])
        for yaw in (.2, .6):
            env.scene["robot"].data.root_ang_vel_b[:, 2] = yaw
            term._update_metrics()
        term.mode[:] = 4
        env.scene["robot"].data.root_ang_vel_b[:, 2] = 10.
        term._update_metrics()
        stats = term.diagnostic_means()
        self.assertAlmostEqual(stats["yaw_only/wz_rmse"], .2, places=6)
        self.assertAlmostEqual(stats["yaw_only/wz_mean"], .4, places=6)
        self.assertAlmostEqual(stats["yaw_only/wz_std"], .2, places=6)
        self.assertAlmostEqual(stats["yaw_only/fraction"], 2 / 3, places=6)
        self.assertEqual(stats["in_place/samples"], 0.)
        self.assertTrue(all(math.isfinite(value) for value in stats.values()))

    def test_reset_states_and_invalid_scans_are_excluded(self):
        env, term = make_env(2)
        env.episode_length_buf[0] = 0
        env.scene["scan_L"].data.ray_hits_w.fill_(torch.inf)
        term._update_metrics()
        self.assertEqual(term._mode_counts[0].sum().item(), 0.)
        self.assertEqual(term._mode_counts[1].sum().item(), 1.)
        stats = term.diagnostic_means()
        self.assertEqual(stats["height_high/swing_samples"], 0.)
        self.assertTrue(all(math.isfinite(value) for value in stats.values()))

    def test_wheel_clipping_metrics_distinguish_same_and_opposite_limits(self):
        env, term = make_env(2)
        env.raw_actions[:] = torch.tensor([[2., 2.], [-2., 2.]])
        term._update_metrics()
        stats = term.reset()
        self.assertEqual(stats["wheel_raw_output_over_limit_rate"], 1.)
        self.assertEqual(stats["wheel_raw_output_same_sign_over_limit_rate"], .5)
        self.assertEqual(term.diagnostic_means()["wheel_raw_output_over_limit_rate"], 0.)

    def test_actual_foot_configuration_installs_sampler_and_continuous_clock(self):
        path = _PATH.parent.parent.parent / "robots/limx_wheelfoot_mode_env_cfg.py"
        cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "WFFootAllTerrainEnvCfg")
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__post_init__")
        nodes = [n for n in init.body if isinstance(n, ast.Assign) and (
            ast.unparse(n.targets[0]).startswith("self.commands.base_velocity")
            or ast.unparse(n.targets[0]) in ("velocity_cfg", "self.commands.gait_command.continuous_phase"))]
        base = NS(ranges=NS(), asset_name="robot", heading_command=True, heading_control_stiffness=.5,
                  resampling_time_range=(3., 15.), debug_vis=False, goal_vel_visualizer_cfg=None,
                  current_vel_visualizer_cfg=None)
        cfg = NS(commands=NS(base_velocity=base, gait_command=NS(continuous_phase=False)))
        namespace = dict(self=cfg, math=math, mdp=NS(FootVelocityCommandCfg=NS),
                         WHEEL_BODY_NAMES=("L", "R"), LEG_JOINTS=tuple(range(6)),
                         WHEEL_GROUND_SCAN_SENSORS=("scan_L", "scan_R"), WHEEL_RADIUS=.128)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
        result = cfg.commands.base_velocity
        self.assertEqual((result.rel_standing_envs, result.rel_straight_envs, result.rel_lateral_envs,
                          result.rel_yaw_only_envs, result.rel_mixed_envs), (.15, .25, .1, .2, .3))
        self.assertEqual(result.ranges.lin_vel_x, (-.8, .8))
        self.assertEqual(result.ranges.lin_vel_y, (-.4, .4))
        self.assertEqual(result.ranges.ang_vel_z, (-.8, .8))
        self.assertTrue(cfg.commands.gait_command.continuous_phase)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
