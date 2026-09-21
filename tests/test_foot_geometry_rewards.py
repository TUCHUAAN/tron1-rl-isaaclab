"""Exercise actual Foot reward terms with CPU sensor tensors (no PhysX)."""
from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import torch

from test_reward_math import reward_math


class ManagerTermStub:
    def __init__(self, cfg, env):
        self.cfg, self._env = cfg, env
        self.num_envs, self.device = env.num_envs, env.device


_PATH = (Path(__file__).resolve().parents[1]
         / "exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/rewards.py")
_NAMES = {
    "foot_lateral_width_huber", "TerrainAdaptiveSwingClearanceExp", "TerrainCycleSwingClearanceExp",
    "planned_support_contact_loss", "leg_action_rate_l2", "ActionSmoothnessPenalty", "BaseVelocityAccelerationPenalty",
    "ang_vel_z_tracking_error_huber", "lin_vel_xy_tracking_error_huber", "_wheel_terrain_planes", "_terrain_plane",
    "_wheel_ground_force_confidence", "_filtered_contact_force_history", "_wheel_ground_contact_confidence",
    "wheel_ground_contact_confidence_components", "GaitReward", "wheel_actual_speed_huber", "wheel_target_zero_deadband_l2",
}
_TREE = ast.parse(_PATH.read_text())
_MODULE = ast.Module(body=[
    ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
    *(node for node in _TREE.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in _NAMES),
], type_ignores=[])
_NS = {**vars(reward_math), "torch": torch, "math": math, "ManagerTermBase": ManagerTermStub,
       "distributions": torch.distributions}
exec(compile(ast.fix_missing_locations(_MODULE), str(_PATH), "exec"), _NS)


class Scene(dict):
    @property
    def sensors(self):
        return self


class FootGeometryRewardsTest(unittest.TestCase):
    def make_env(self, n=1):
        # Each wheel is measured above its own plane; the full scan sets the peak.
        positions = torch.tensor([[[0., .17, .408], [0., -.17, .128]]]).repeat(n, 1, 1)
        robot = NS(data=NS(
            body_link_pos_w=positions,
            root_link_quat_w=torch.tensor([[1., 0., 0., 0.]]).repeat(n, 1),
            root_lin_vel_w=torch.zeros(n, 3), root_lin_vel_b=torch.zeros(n, 3),
            root_ang_vel_b=torch.zeros(n, 3),
        ))
        scene = Scene(robot=robot)
        grid = torch.tensor([[x, y, 0.] for x in (-.04, 0., .04) for y in (-.04, 0., .04)])
        for side, ground_z, force_z in (("L", .2, 0.), ("R", 0., 100.)):
            hits = grid[None].repeat(n, 1, 1)
            hits[:, :, 2] = ground_z
            history = torch.zeros(n, 4, 1, 1, 3)
            history[..., 2] = force_z
            scene[f"scan_{side}"] = NS(data=NS(ray_hits_w=hits))
            scene[f"contact_{side}"] = NS(data=NS(force_matrix_w_history=history))
        full_scan = torch.zeros(n, 121, 3)
        full_scan[:, -1, 2] = .08
        scene["height_scanner"] = NS(data=NS(ray_hits_w=full_scan))
        commands = {
            "gait_command": torch.tensor([[1., .5, .5, 0.]]).repeat(n, 1),
            "base_velocity": torch.zeros(n, 3),
            "body_height": torch.full((n, 1), .8),
        }
        env = NS(scene=scene, commands=commands, num_envs=n, device="cpu", step_dt=.01,
                 episode_length_buf=torch.full((n,), 75))
        env.command_manager = NS(
            get_command=lambda name: commands[name],
            get_term=lambda name: NS(phase=(env.episode_length_buf * env.step_dt * commands[name][:, 0]) % 1.),
        )
        return env

    def test_clearance_consumes_shared_phase_instead_of_episode_time(self):
        env = self.make_env()
        env.episode_length_buf.zero_()
        env.command_manager.get_term = lambda name: NS(phase=torch.full((1,), .75))
        env.scene["robot"].data.body_link_pos_w[:, 0, 2] = .328
        term, params = self.swing_term(env)
        torch.testing.assert_close(term(env, **params), torch.full((1,), math.exp(-(.08 / .025) ** 2)))

    def swing_term(self, env):
        params = dict(asset_cfg=NS(name="robot", body_ids=[0, 1]),
                      contact_sensor_names=("contact_L", "contact_R"),
                      terrain_sensor_names=("scan_L", "scan_R"),
                      height_sensor_cfg=NS(name="height_scanner"))
        return _NS["TerrainAdaptiveSwingClearanceExp"](NS(params=params), env), params

    def acceleration_term(self, env, **overrides):
        params = dict(command_name="base_velocity", component="xy", tracking_std=.3,
                      acceleration_scale=3., asset_cfg=NS(name="robot", body_ids=[0, 1]),
                      contact_sensor_names=("contact_L", "contact_R"),
                      terrain_sensor_names=("scan_L", "scan_R"), kernel="charbonnier",
                      linear_velocity_frame="world", min_tracking_gate=.1)
        params.update(overrides)
        return _NS["BaseVelocityAccelerationPenalty"](NS(params=params), env), params

    def gait_term(self, env, **overrides):
        env.scene["contact_forces"] = NS(data=NS(net_forces_w=torch.zeros(env.num_envs, 2, 3)))
        env.scene["robot"].data.body_lin_vel_w = torch.zeros(env.num_envs, 2, 3)
        params = dict(tracking_contacts_shaped_force=-4., tracking_contacts_shaped_vel=-2.,
                      gait_force_sigma=25., gait_vel_sigma=.25, kappa_gait_probs=.05,
                      command_name="gait_command", force_kernel="huber", force_reference=100.,
                      asset_cfg=NS(name="robot", body_ids=[0, 1]),
                      sensor_cfg=NS(name="contact_forces", body_ids=[0, 1]))
        params.update(overrides)
        return _NS["GaitReward"](NS(params=params), env), params

    def test_gait_huber_distinguishes_partial_unloading_without_penalizing_stance_load(self):
        env = self.make_env(5)
        term, _ = self.gait_term(env)
        forces = torch.tensor([[100., 100.], [50., 150.], [30., 170.], [20., 180.], [0., 200.]])
        desired = torch.tensor([[0., 1.]]).repeat(5, 1)
        reward = term._compute_force_reward(forces, desired)
        torch.testing.assert_close(reward, torch.tensor([-1., -.25, -.09, -.04, 0.]))
        self.assertTrue(torch.all(reward[1:] > reward[:-1]))
        forces[:, 1] = 1000.  # Planned stance leg may carry the robot's load.
        torch.testing.assert_close(term._compute_force_reward(forces, desired), reward)
        torch.testing.assert_close(term._compute_force_reward(forces.flip(1), desired.flip(1)), reward)

    def test_gait_huber_runtime_switches_unloading_leg_at_zero_command(self):
        env = self.make_env()
        term, params = self.gait_term(env)
        forces = env.scene["contact_forces"].data.net_forces_w
        forces[:, 1, 2] = 200.
        # At phase .75, left swings and right supports; at .25 roles switch.
        correct = term(env, **params)
        env.episode_length_buf[:] = 25
        incorrect = term(env, **params)
        self.assertLess(abs(correct.item()), 1.e-4)
        self.assertLess(incorrect.item(), -2.99)

    def test_gait_legacy_default_and_huber_configuration_validation(self):
        env = self.make_env()
        _, params = self.gait_term(env)
        params.pop("force_kernel")
        params.pop("force_reference")
        params["tracking_contacts_shaped_force"] = -2.
        term = _NS["GaitReward"](NS(params=params), env)
        actual = term._compute_force_reward(torch.tensor([[100., 100.]]), torch.tensor([[0., 1.]]))
        torch.testing.assert_close(actual, torch.tensor([-1.]))
        for overrides in (dict(force_kernel="unknown"), dict(force_reference=0.),
                          dict(force_reference=float("nan")), dict(tracking_contacts_shaped_force=1.)):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.gait_term(env, **overrides)

    def test_exponential_clearance_rewards_target_and_discourages_both_error_directions(self):
        env = self.make_env(5)
        env.scene["robot"].data.body_link_pos_w[:, 0, 2] = .328 + torch.tensor([0., .055, .08, .105, .16])
        term, params = self.swing_term(env)
        reward = term(env, **params)
        torch.testing.assert_close(reward, torch.tensor([math.exp(-10.24), math.exp(-1.), 1., math.exp(-1.), math.exp(-10.24)]))
        self.assertTrue(((reward >= 0.) & (reward <= 1.)).all())
        for std in (0., -1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                term(env, **params, std=std)

    def test_flat_target_is_four_cm_and_reward_retains_soft_support_gate(self):
        env = self.make_env(2)
        env.scene["height_scanner"].data.ray_hits_w.zero_()
        env.scene["robot"].data.body_link_pos_w[:, 0, 2] = torch.tensor([.328, .368])
        term, params = self.swing_term(env)
        torch.testing.assert_close(term(env, **params), torch.tensor([math.exp(-2.56), 1.]))
        torch.testing.assert_close(term.peak_height, torch.full((2,), .04))
        env.scene["contact_R"].data.force_matrix_w_history[..., 2] = 7.5
        torch.testing.assert_close(term(env, **params), .5 * torch.tensor([math.exp(-2.56), 1.]))
        env.scene["contact_R"].data.force_matrix_w_history.zero_()
        torch.testing.assert_close(term(env, **params), torch.zeros(2))

    def test_zero_translation_and_yaw_commands_share_the_same_swing_reward(self):
        env = self.make_env()
        term, params = self.swing_term(env)
        env.scene["robot"].data.body_link_pos_w[:, 0, 2] = .328
        for cmd in [(0., 0., 0.), (.4, 0., 0.), (0., 0., .4)]:
            env.commands["base_velocity"][:] = torch.tensor(cmd)
            torch.testing.assert_close(term(env, **params), torch.full((1,), math.exp(-(.08 / .025) ** 2)))

    def test_height_and_reserved_gait_command_do_not_change_peak(self):
        env = self.make_env()
        term, params = self.swing_term(env)
        for h in [.65, .8, .85]:
            env.commands["body_height"][:] = h
            env.commands["gait_command"][:, 3] = h
            torch.testing.assert_close(term(env, **params), torch.ones(1))
            torch.testing.assert_close(term.peak_height, torch.full((1,), .08))

    def test_scan_updates_peak_during_same_swing_with_fast_rise_slow_decay(self):
        env = self.make_env()
        hits = env.scene["height_scanner"].data.ray_hits_w
        hits.zero_()
        term, params = self.swing_term(env)
        term(env, **params)
        torch.testing.assert_close(term.peak_height, torch.full((1,), .04))
        hits[:, -1, 2] = .4
        term(env, **params)
        first_rise = term.peak_height.clone()
        self.assertTrue(((first_rise > .04) & (first_rise < .10)).all())
        term(env, **params)
        before_fall = term.peak_height.clone()
        self.assertTrue((before_fall > first_rise).all())
        hits.zero_()
        term(env, **params)
        self.assertTrue((term.peak_height < before_fall).all())
        self.assertLess(float(before_fall - term.peak_height), float(first_rise - .04))

    def test_peak_initialization_clamps_scan_and_reset_is_per_environment(self):
        env = self.make_env(2)
        env.scene["height_scanner"].data.ray_hits_w[:, -1, 2] = torch.tensor([.3, .06])
        term, params = self.swing_term(env)
        term(env, **params)
        torch.testing.assert_close(term.peak_height, torch.tensor([.1, .06]))
        term.reset(torch.tensor([0]))
        torch.testing.assert_close(term.peak_height, torch.tensor([.04, .06]))
        env.scene["height_scanner"].data.ray_hits_w[0, :, 2] = 0.
        term(env, **params)
        torch.testing.assert_close(term.peak_height, torch.tensor([.04, .06]))

    def test_invalid_full_scan_holds_peak_without_scoring(self):
        env = self.make_env()
        term, params = self.swing_term(env)
        term(env, **params)
        env.scene["robot"].data.body_link_pos_w[:, 0, 2] = .328
        env.scene["height_scanner"].data.ray_hits_w.fill_(torch.inf)
        torch.testing.assert_close(term(env, **params), torch.zeros(1))
        torch.testing.assert_close(term.peak_height, torch.full((1,), .08))
        self.assertFalse(term.scan_valid.any())
        term.reset()
        torch.testing.assert_close(term(env, **params), torch.zeros(1))
        torch.testing.assert_close(term.peak_height, torch.full((1,), .04))

    def test_missing_local_rays_or_opposite_support_suppress_reward(self):
        for key in ["scan_L", "scan_R", "contact_R"]:
            env = self.make_env()
            env.scene["robot"].data.body_link_pos_w[:, 0, 2] = .328
            term, params = self.swing_term(env)
            if key.startswith("scan"):
                env.scene[key].data.ray_hits_w.fill_(torch.inf)
            else:
                env.scene[key].data.force_matrix_w_history.zero_()
            torch.testing.assert_close(term(env, **params), torch.zeros(1))

    def test_local_slope_normal_clearance_is_world_translation_invariant(self):
        env = self.make_env()
        normal = torch.tensor([-.2, .1, 1.])
        normal /= torch.linalg.vector_norm(normal)
        for i, side in enumerate(("L", "R")):
            hits = env.scene[f"scan_{side}"].data.ray_hits_w
            hits[:, :, 2] = (.2 if i == 0 else 0.) + .2 * hits[:, :, 0] - .1 * hits[:, :, 1]
            env.scene["robot"].data.body_link_pos_w[:, i] = hits.mean(1) + normal * (.208 if i == 0 else .128)
        term, params = self.swing_term(env)
        torch.testing.assert_close(term(env, **params), torch.ones(1))
        env.scene["robot"].data.body_link_pos_w[:, :, 2] += 3.
        for key in ["scan_L", "scan_R", "height_scanner"]:
            env.scene[key].data.ray_hits_w[:, :, 2] += 3.
        torch.testing.assert_close(term(env, **params), torch.ones(1))

    def test_half_cycle_and_phase_boundary(self):
        env = self.make_env()
        term, params = self.swing_term(env)
        env.episode_length_buf.zero_()
        torch.testing.assert_close(term(env, **params), torch.zeros(1))
        env.episode_length_buf[:] = 25
        env.scene["robot"].data.body_link_pos_w[:, 0, 2] -= .08
        env.scene["robot"].data.body_link_pos_w[:, 1, 2] += .08
        env.scene["contact_L"].data.force_matrix_w_history[..., 2] = 100.
        env.scene["contact_R"].data.force_matrix_w_history.zero_()
        torch.testing.assert_close(term(env, **params), torch.ones(1))

    def test_world_acceleration_ignores_rotation_of_body_velocity_coordinates(self):
        env = self.make_env()
        term, params = self.acceleration_term(env)
        data = env.scene["robot"].data
        data.root_lin_vel_w[:, 0] = 1.
        data.root_lin_vel_b[:, 0] = 1.
        for _ in range(2):
            torch.testing.assert_close(term(env, **params), torch.zeros(1))
        data.root_lin_vel_b[:] = torch.tensor([[0., -1., 0.]])
        env.commands["base_velocity"][:] = data.root_lin_vel_b
        torch.testing.assert_close(term(env, **params), torch.zeros(1))
        data.root_lin_vel_w[:, 0] += .03
        torch.testing.assert_close(term(env, **params), torch.full((1,), math.sqrt(2.) - 1.))

    def test_acceleration_partial_reset_and_large_yaw_error(self):
        env = self.make_env(2)
        term, params = self.acceleration_term(env, component="yaw", acceleration_scale=4.)
        data = env.scene["robot"].data
        for _ in range(2):
            torch.testing.assert_close(term(env, **params), torch.zeros(2))
        data.root_ang_vel_b[:, 2] = 1.
        cost = term(env, **params)
        torch.testing.assert_close(cost, torch.full((2,), .1 * (math.sqrt(1 + 100**2 / 16) - 1)))
        term.reset(torch.tensor([0]))
        for rate in [2., 3.]:
            data.root_ang_vel_b[:, 2] = rate
            cost = term(env, **params)
            self.assertEqual(float(cost[0]), 0.)
            self.assertGreater(float(cost[1]), 0.)
        data.root_ang_vel_b[:, 2] = 4.
        self.assertTrue((term(env, **params) > 0.).all())

    def test_yaw_huber_retains_large_error_cost_and_both_signs(self):
        env = self.make_env(5)
        env.scene["robot"].data.root_ang_vel_b[:, 2] = torch.tensor([0., .5, 1., 2., -2.])
        cost = _NS["ang_vel_z_tracking_error_huber"](env, "base_velocity", NS(name="robot"))
        torch.testing.assert_close(cost, torch.tensor([0., .5, 1.5, 3.5, 3.5]))

    def test_xy_huber_uses_body_frame_and_is_direction_independent(self):
        env = self.make_env(6)
        data = env.scene["robot"].data
        data.root_lin_vel_b[:, :2] = torch.tensor([[0., 0.], [.4, 0.], [.8, 0.], [-.8, 0.], [0., .8], [.48, .64]])
        data.root_lin_vel_b[:, 2] = 10.  # Vertical motion is not planar tracking error.
        data.root_lin_vel_w[:] = 20.  # Commands and actual velocity must share the body frame.
        cost = _NS["lin_vel_xy_tracking_error_huber"](env, "base_velocity", NS(name="robot"))
        torch.testing.assert_close(cost, torch.tensor([0., .5, 1.5, 1.5, 1.5, 1.5]))

    def test_xy_huber_improves_with_tracking_and_has_no_integral_state(self):
        env = self.make_env(3)
        env.commands["base_velocity"][:, 0] = .8
        env.scene["robot"].data.root_lin_vel_b[:, 0] = torch.tensor([0., .4, .8])
        term = _NS["lin_vel_xy_tracking_error_huber"]
        for _ in range(3):
            torch.testing.assert_close(term(env, "base_velocity", NS(name="robot")), torch.tensor([1.5, .5, 0.]))
        env.commands["base_velocity"][:, :2] = env.scene["robot"].data.root_lin_vel_b[:, :2]
        torch.testing.assert_close(term(env, "base_velocity", NS(name="robot")), torch.zeros(3))
        for scale in (0., -1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                term(env, "base_velocity", NS(name="robot"), error_scale=scale)

    def test_actual_foot_configuration_binds_new_reward_parameters(self):
        # Execute the changed reward declarations from the actual config, then
        # call the real terms. This catches manager signature/parameter drift
        # without requiring Isaac application startup or a GPU.
        path = _PATH.parent.parent / "robots/limx_wheelfoot_mode_env_cfg.py"
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "WFFootAllTerrainEnvCfg")
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__post_init__")
        terms = {"pen_swing_clearance", "rew_swing_clearance", "pen_wheel_target_zero", "pen_yaw_tracking_error", "pen_lin_vel_xy_tracking_error",
                 "pen_base_lin_acc_xy", "pen_base_yaw_acc", "gait_contact_schedule", "pen_wheel_actual_speed"}
        nodes = []
        for node in init.body:
            if not isinstance(node, ast.Assign):
                continue
            target = ast.unparse(node.targets[0])
            if target == "acceleration_params" or target in {f"self.rewards.{name}" for name in terms} or any(
                target.startswith(f"self.rewards.{name}.") for name in ["rew_lin_vel_xy", "rew_ang_vel_z"]
            ):
                nodes.append(node)
        cfg = NS(rewards=NS(rew_lin_vel_xy=NS(params={}), rew_ang_vel_z=NS(params={})))
        namespace = dict(
            self=cfg, math=math, mdp=NS(**{name: _NS[name] for name in [
                "TerrainAdaptiveSwingClearanceExp", "TerrainCycleSwingClearanceExp", "BaseVelocityAccelerationPenalty", "ang_vel_z_tracking_error_huber",
                "lin_vel_xy_tracking_error_huber", "GaitReward", "wheel_actual_speed_huber", "wheel_target_zero_deadband_l2"
            ]}), RewTerm=NS,
            SceneEntityCfg=lambda name, **kwargs: NS(name=name, body_ids=[0, 1], joint_ids=[0, 1]),
            WHEEL_BODY_NAMES=["wheel_L_Link", "wheel_R_Link"], WHEEL_RADIUS=.128,
            WHEEL_JOINTS=["wheel_L_Joint", "wheel_R_Joint"],
            WHEEL_GROUND_CONTACT_SENSORS=("contact_L", "contact_R"),
            WHEEL_GROUND_SCAN_SENSORS=("scan_L", "scan_R"),
        )
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
        self.assertEqual(cfg.rewards.rew_lin_vel_xy.weight, 5.5)
        self.assertEqual(cfg.rewards.rew_ang_vel_z.weight, 3.0)
        self.assertEqual(cfg.rewards.pen_lin_vel_xy_tracking_error.weight, -.5)
        self.assertEqual(cfg.rewards.pen_lin_vel_xy_tracking_error.params["error_scale"], .4)
        self.assertEqual(cfg.rewards.pen_base_lin_acc_xy.weight, 0.)
        self.assertEqual(cfg.rewards.pen_base_yaw_acc.weight, 0.)
        self.assertEqual(cfg.rewards.pen_wheel_actual_speed.weight, 0.)
        self.assertEqual(cfg.rewards.gait_contact_schedule.params["force_kernel"], "huber")
        self.assertEqual(cfg.rewards.gait_contact_schedule.params["tracking_contacts_shaped_force"], -6.)
        self.assertEqual(cfg.rewards.gait_contact_schedule.params["force_reference"], 100.)
        self.assertIsNone(cfg.rewards.pen_swing_clearance)
        self.assertEqual(cfg.rewards.rew_swing_clearance.weight, 2.)
        self.assertEqual(cfg.rewards.rew_swing_clearance.params["min_peak_height"], .05)
        self.assertEqual(cfg.rewards.rew_swing_clearance.params["std"], .025)
        self.assertIsNone(cfg.rewards.pen_wheel_target_zero)
        self.assertEqual(cfg.rewards.rew_lin_vel_xy.params["std"], math.sqrt(.20))
        self.assertEqual(cfg.rewards.rew_ang_vel_z.params["std"], .5)
        env = self.make_env()
        self.gait_term(env)  # Add the whole-robot contact and velocity sensor buffers.
        env.scene["robot"].data.joint_vel = torch.ones(1, 2)
        env.action_manager = NS(get_term=lambda name: NS(processed_actions=torch.zeros(1, 2)))
        for name in terms - {"pen_swing_clearance", "pen_wheel_target_zero"}:
            term_cfg = getattr(cfg.rewards, name)
            func = term_cfg.func(term_cfg, env) if isinstance(term_cfg.func, type) else term_cfg.func
            for _ in range(3):
                result = func(env, **term_cfg.params)
                self.assertEqual(result.shape, (1,))
                self.assertTrue(torch.isfinite(result).all())

    def test_width_wrapper_reads_left_right_link_positions(self):
        env = self.make_env()
        torch.testing.assert_close(_NS["foot_lateral_width_huber"](env, NS(name="robot", body_ids=[0, 1])), torch.zeros(1))
        self.assertGreater(float(_NS["foot_lateral_width_huber"](env, NS(name="robot", body_ids=[1, 0]))), 0.)


if __name__ == "__main__":
    unittest.main()
