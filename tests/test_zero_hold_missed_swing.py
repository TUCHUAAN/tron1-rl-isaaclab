"""Behavioral tests for v10 zero-axis holds and independent swing completion."""
import ast
import math
from types import SimpleNamespace as NS
import unittest
import torch
from test_reward_math import reward_math as rm
import test_foot_geometry_rewards as geometry


class ZeroHoldTests(unittest.TestCase):
    def make(self, n=1, dt=.02):
        return rm.ZeroCommandHoldTracker(n, 'cpu', dt, settle_time=.04,
                                        position_deadband=.01, yaw_deadband=.01)

    def test_forward_allowed_lateral_and_yaw_held(self):
        t=self.make();cmd=torch.tensor([[.4,0.,0.]])
        for i in range(10):
            cost=t.update(torch.tensor([[i*.02,0.]]),torch.zeros(1),cmd)
            self.assertEqual(cost.item(),0.)
        self.assertGreater(t.update(torch.tensor([[.2,.05]]),torch.tensor([.2]),cmd).item(),0.)
        self.assertEqual(t.update(torch.tensor([[.2,.05]]),torch.tensor([.2]),torch.ones(1,3)).item(),0.)
        self.assertEqual(t.history.abs().sum().item(),0.)

    def test_rotating_body_frame_allows_body_forward_motion(self):
        t=self.make();pos=torch.zeros(1,2);yaw=torch.zeros(1);cmd=torch.tensor([[.4,0.,.5]])
        for _ in range(100):
            old=yaw.clone();yaw+=.01;mid=(old+yaw)/2
            pos+=.008*torch.stack((mid.cos(),mid.sin()),-1)
            self.assertLess(t.update(pos,yaw,cmd).item(),1.e-8)
        # Same curving world path should not create a false lateral hold error.
        self.assertTrue(t.active[0,1]);self.assertFalse(t.active[0,0])

    def test_all_zero_anchor_does_not_follow_slow_drift_and_reset_is_local(self):
        t=self.make(2);cmd=torch.zeros(2,3)
        for i in range(100):
            cost=t.update(torch.tensor([[i*.001,0.],[0.,0.]]),torch.zeros(2),cmd)
        self.assertGreater(cost[0],0.);self.assertEqual(cost[1],0.)
        anchor=t.position_anchor[0].clone();t.reset([1])
        torch.testing.assert_close(t.position_anchor[0],anchor)
        t.reset([0]);self.assertEqual(t.update(torch.ones(2,2)*10,torch.zeros(2),cmd).sum(),0.)

    def test_partial_window_expires_yaw_wrap_and_invalid_inputs(self):
        t=self.make();cmd=torch.tensor([[0.,.3,0.]])
        yaw=torch.tensor([math.pi-.005]);pos=torch.zeros(1,2)
        for _ in range(3):t.update(pos,yaw,cmd)
        self.assertLess(t.update(pos,torch.tensor([-math.pi+.005]),cmd).item(),1.e-9)
        pos[:,0]=.1;self.assertGreater(t.update(pos,yaw,cmd).item(),0.)
        for _ in range(52):cost=t.update(pos,yaw,cmd)
        self.assertEqual(cost.item(),0.)
        self.assertEqual(t.update(pos*float('nan'),yaw,cmd).item(),0.)
        self.assertEqual(t.update(pos,yaw,cmd).item(),0.)

    def test_step_size_keeps_same_drift_cost(self):
        costs=[]
        for dt in (.01,.02):
            t=self.make(dt=dt);cmd=torch.tensor([[0.,.3,0.]])
            for i in range(round(2/dt)):
                c=t.update(torch.tensor([[i*dt*.05,0.]]),torch.zeros(1),cmd)
            costs.append(c)
        torch.testing.assert_close(*costs,atol=1.e-6,rtol=1.e-5)


class MissedSwingTests(unittest.TestCase):
    def run_cycle(self, lift=False, invalid=False, airborne=False, dt=.02):
        t=rm.MissedSwingTracker(1,'cpu');gait=torch.tensor([[1.,.5,.5,0.]])
        total=0.
        for i in range(round(1.02/dt)):
            phase=(i*dt)%1
            clearance=torch.zeros(1,2);force=torch.full((1,2),100.);valid=torch.ones(1,2,dtype=torch.bool)
            if (lift and .6<=phase<.8) or airborne:
                clearance[0,0]=.05;force[0,0]=0.
            if invalid and .6<=phase<.8:valid[0,0]=False
            total+=t.update(torch.tensor([phase]),gait,clearance,force,valid,dt).item()
        return total

    def test_each_leg_must_lift_and_event_is_once(self):
        self.assertEqual(self.run_cycle(),1.) # initial right partial swing ignored
        self.assertEqual(self.run_cycle(lift=True),0.)
        self.assertEqual(self.run_cycle(invalid=True),0.)
        self.assertEqual(self.run_cycle(airborne=True),1.) # flight is not a new takeoff
        self.assertEqual(self.run_cycle(dt=.01),1.)

    def test_contact_chatter_is_not_success(self):
        t=rm.MissedSwingTracker(1,'cpu');g=torch.tensor([[1.,.5,.5,0.]])
        cost=0
        for i in range(51):
            force=torch.tensor([[0. if i%2 else 100.,100.]])
            cost+=t.update(torch.tensor([(i*.02)%1]),g,torch.full((1,2),.02),force,torch.ones(1,2,dtype=torch.bool),.02).item()
        self.assertEqual(cost,1.)

    def test_actual_adapters_and_both_mode_config_bindings(self):
        tree=ast.parse(geometry._PATH.read_text());names={'ZeroCommandHoldPenalty','MissedSwingPenalty'}
        nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in names]
        scope=dict(geometry._NS,ZeroCommandHoldTracker=rm.ZeroCommandHoldTracker,MissedSwingTracker=rm.MissedSwingTracker)
        future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future,*nodes],type_ignores=[])),str(geometry._PATH),'exec'),scope)
        path=geometry._PATH.parent.parent/'robots/limx_wheelfoot_mode_env_cfg.py';tree=ast.parse(path.read_text())
        for mode in ('WFFootAllTerrainEnvCfg','WFWheelModeEnvCfg'):
            cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==mode)
            init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__post_init__')
            nodes=[n for n in init.body if isinstance(n,ast.Assign) and ast.unparse(n.targets[0]) in
                   ('self.rewards.pen_zero_command_hold','self.rewards.pen_missed_swing')]
            cfg=NS(rewards=NS());ns=dict(self=cfg,RewTerm=NS,mdp=NS(**{n:scope[n] for n in names}),
                SceneEntityCfg=lambda name,**kw:NS(name=name,body_ids=[0,1]),WHEEL_BODY_NAMES=[],
                WHEEL_GROUND_CONTACT_SENSORS=('contact_L','contact_R'),WHEEL_GROUND_SCAN_SENSORS=('scan_L','scan_R'),WHEEL_RADIUS=.128)
            exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
            self.assertTrue(hasattr(cfg.rewards,'pen_zero_command_hold'))
            self.assertEqual(hasattr(cfg.rewards,'pen_missed_swing'),mode=='WFFootAllTerrainEnvCfg')
            for name,rcfg in vars(cfg.rewards).items():
                env=geometry.FootGeometryRewardsTest().make_env();env.common_step_counter=0
                env.scene['robot'].data.root_link_pos_w=torch.tensor([[0.,0.,.8]])
                for sensor in ('contact_L','contact_R'):
                    data=env.scene[sensor].data;data.force_matrix_w=data.force_matrix_w_history[:,-1].clone()
                term=rcfg.func(rcfg,env);calls=[]
                term.tracker.update=lambda *a:(calls.append(1) or torch.tensor([1.]))
                result=term(env,**rcfg.params).clone()
                expected=1/env.step_dt if name=='pen_missed_swing' else 1.
                self.assertAlmostEqual(result.item(),expected)
                term(env,**rcfg.params);self.assertEqual(len(calls),1)
                term.reset([0]);self.assertEqual(term(env,**rcfg.params).item(),0.)

if __name__=='__main__':unittest.main()
