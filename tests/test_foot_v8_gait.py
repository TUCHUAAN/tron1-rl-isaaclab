"""v8 terrain latch, support timing, landing and leg-only regularization."""
import math
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import torch
import test_foot_geometry_rewards as geometry
from test_foot_geometry_rewards import _NS
from test_reward_math import reward_math as rm

class FootV8Test(unittest.TestCase):
    def setup_env(self, n=1):
        env = geometry.FootGeometryRewardsTest().make_env(n)
        x, y = torch.meshgrid(torch.linspace(-.5, .5, 11), torch.linspace(-.5, .5, 11), indexing='ij')
        self.grid = torch.stack((x.flatten(), y.flatten(), torch.zeros(121)), -1)
        env.scene['height_scanner'].data.ray_hits_w = self.grid[None].repeat(n, 1, 1)
        env.phase = torch.full((n,), .75)
        env.command_manager.get_term = lambda name: NS(phase=env.phase)
        params = dict(asset_cfg=NS(name='robot', body_ids=[0, 1]),
                      contact_sensor_names=('contact_L', 'contact_R'),
                      terrain_sensor_names=('scan_L', 'scan_R'),
                      height_sensor_cfg=NS(name='height_scanner'))
        term = _NS['TerrainCycleSwingClearanceExp'](NS(params=params), env)
        return env, term, params

    def bump(self, env, value):
        env.scene['height_scanner'].data.ray_hits_w[:, :, 2] = 0.
        env.scene['height_scanner'].data.ray_hits_w[:, 60, 2] = value

    def wrap(self, env, term, params):
        env.phase[:] = .99; term(env, **params)
        env.phase[:] = .01; return term(env, **params)

    def test_two_levels_hysteresis_cycle_latch_and_reset(self):
        env, t, p = self.setup_env(2)
        self.bump(env, .03); t(env, **p)
        torch.testing.assert_close(t.peak_height, torch.full((2,), .05))
        self.bump(env, .08); env.phase[:] = .9; t(env, **p)
        torch.testing.assert_close(t.peak_height, torch.full((2,), .05))
        self.wrap(env, t, p)
        torch.testing.assert_close(t.peak_height, torch.full((2,), .10))
        self.bump(env, .04); self.wrap(env, t, p)
        torch.testing.assert_close(t.peak_height, torch.full((2,), .10))
        self.bump(env, .03); self.wrap(env, t, p)
        torch.testing.assert_close(t.peak_height, torch.full((2,), .05))
        self.bump(env, .04); self.wrap(env, t, p)
        torch.testing.assert_close(t.peak_height, torch.full((2,), .05))
        self.bump(env, .08); self.wrap(env, t, p)
        t.reset([0]); self.bump(env, .03); env.phase[:] = .1; t(env, **p)
        torch.testing.assert_close(t.peak_height, torch.tensor([.05, .10]))

    def test_smooth_slope_does_not_select_high_and_invalid_scan_disables_reward(self):
        env, t, p = self.setup_env()
        hits = env.scene['height_scanner'].data.ray_hits_w
        hits[:, :, 2] = .4 * hits[:, :, 0] + .2 * hits[:, :, 1]
        t(env, **p)
        self.assertLess(t.terrain_metric_m.item(), 1e-6)
        self.assertAlmostEqual(t.peak_height.item(), .05)
        hits[:] = float('nan'); value = self.wrap(env, t, p)
        self.assertFalse(t.scan_valid.item()); self.assertEqual(value.item(), 0.)
        hits[:] = self.grid; hits[:, 60, 2] = .08
        env.phase[:] = .2; t(env, **p)
        self.assertFalse(t.scan_valid.item())  # no mid-cycle reselection
        self.wrap(env, t, p); self.assertTrue(t.scan_valid.item())
        self.assertAlmostEqual(t.peak_height.item(), .10)

    def test_cycloid_target_matches_live_phase_with_latched_peak(self):
        env, t, p = self.setup_env()
        for phase in (.6, .75, .9):
            env.phase[:] = phase
            u = (phase - .5) / .5
            envelope = (1 - math.cos(2 * math.pi * u)) / 2
            env.scene['robot'].data.body_link_pos_w[:, 0, 2] = .328 + .05 * envelope
            value = t(env, **p)
            self.assertAlmostEqual(value.item(), envelope, places=5)

    def test_support_loss_follows_phase_at_zero_and_nonzero_commands(self):
        env, _, _ = self.setup_env()
        args = dict(command_name='gait_command', contact_sensor_names=('contact_L','contact_R'))
        loss = _NS['planned_support_contact_loss']
        self.assertLess(loss(env, **args).item(), 1e-5)
        env.phase[:] = .25
        self.assertAlmostEqual(loss(env, **args).item(), .5, places=5)
        for name in ('contact_L','contact_R'):
            env.scene[name].data.force_matrix_w_history.zero_()
        self.assertAlmostEqual(loss(env, **args).item(), .5, places=5)
        env.commands['base_velocity'][:] = torch.tensor([[-.4,.2,.4]])
        self.assertAlmostEqual(loss(env, **args).item(), .5, places=5)

    def test_landing_only_final_two_cm_and_excess_downward_speed(self):
        n=5
        pos=torch.zeros(n,1,3); pos[:,:,2]=torch.tensor([.138,.138,.178,.138,.138])[:,None]
        vel=torch.zeros_like(pos); vel[:,:,2]=torch.tensor([-.1,-.5,-.5,.5,-.5])[:,None]
        normals=torch.zeros_like(pos); normals[:,:,2]=1.
        actual=rm.terrain_relative_landing_velocity_l2(pos,vel,torch.zeros_like(pos),normals,
            torch.ones(n,1,dtype=torch.bool),torch.tensor([[False],[False],[False],[False],[True]]),
            .128,.02,allowed_downward_speed=.2)
        torch.testing.assert_close(actual,torch.tensor([0.,.09,0.,0.,0.]))

    def test_real_configuration_selects_legs_and_replaces_air_time(self):
        path=Path(__file__).resolve().parents[1]/'exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/robots/limx_wheelfoot_mode_env_cfg.py'
        cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='WFFootAllTerrainEnvCfg')
        init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__post_init__')
        names=('pen_joint_torque','pen_joint_accel','pen_joint_power_l1','pen_action_rate','pen_action_smoothness')
        cfg=NS(rewards=NS(**{name:NS(params={}) for name in names}))
        chosen=[]
        for node in init.body:
            text=ast.unparse(node)
            if isinstance(node,ast.For) and 'joint_names=LEG_JOINTS' in text:
                chosen.append(node)
            elif isinstance(node,ast.Assign) and (
                text.startswith('self.rewards.pen_action_rate.func') or
                text.startswith('self.rewards.pen_action_rate.params') or
                text.startswith('self.rewards.pen_action_smoothness.params') or
                text.startswith('self.rewards.pen_all_wheels_air_time =') or
                text.startswith('self.rewards.pen_planned_support_contact =')):
                chosen.append(node)
        legs=['abad_L_Joint','abad_R_Joint','hip_L_Joint','hip_R_Joint','knee_L_Joint','knee_R_Joint']
        scope=dict(self=cfg,mdp=NS(**_NS),RewTerm=NS,LEG_JOINTS=legs,
                   WHEEL_GROUND_CONTACT_SENSORS=('contact_L','contact_R'),
                   SceneEntityCfg=lambda name,**kw:NS(name=name,**kw))
        exec(compile(ast.Module(body=chosen,type_ignores=[]),str(path),'exec'),scope)
        for name in names[:3]:
            self.assertEqual(getattr(cfg.rewards,name).params['asset_cfg'].joint_names,legs)
        self.assertEqual(cfg.rewards.pen_action_rate.params,{'action_dim':6})
        self.assertEqual(cfg.rewards.pen_action_smoothness.params,{'action_dim':6})
        self.assertIsNone(cfg.rewards.pen_all_wheels_air_time)
        self.assertEqual(cfg.rewards.pen_planned_support_contact.weight,-1.)

    def test_joint_power_respects_selected_leg_ids(self):
        path=Path(geometry._PATH)
        node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='joint_powers_l1')
        scope=dict(torch=torch,SceneEntityCfg=lambda name:NS(name=name,joint_ids=slice(None)))
        module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])
        exec(compile(ast.fix_missing_locations(module),str(path),'exec'),scope)
        env=NS(scene={'robot':NS(data=NS(applied_torque=torch.tensor([[1.]*6+[100.,100.]]),
                                       joint_vel=torch.ones(1,8)))})
        fn=scope['joint_powers_l1']
        self.assertEqual(fn(env,NS(name='robot',joint_ids=list(range(6)))).item(),6.)
        self.assertEqual(fn(env).item(),206.)  # default Wheel/PF callers keep full selection

    def test_ignored_wheel_outputs_do_not_change_action_costs(self):
        env=NS(action_manager=NS(action=torch.zeros(1,8),prev_action=torch.zeros(1,8)),
               num_envs=1,device='cpu',step_dt=.02,episode_length_buf=torch.tensor([10]))
        smooth=_NS['ActionSmoothnessPenalty'](NS(params={}),env)
        for value in (10.,-30.,500.,-200.):
            env.action_manager.prev_action=env.action_manager.action.clone()
            env.action_manager.action[:,-2:]=value
            self.assertEqual(_NS['leg_action_rate_l2'](env).item(),0.)
            self.assertEqual(smooth(env,action_dim=6).item(),0.)
        env.action_manager.action[:,0]=1.
        self.assertEqual(_NS['leg_action_rate_l2'](env).item(),1.)
        self.assertEqual(smooth(env,action_dim=6).item(),1.)

if __name__=='__main__': unittest.main()
