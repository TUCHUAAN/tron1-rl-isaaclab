"""Independent dense swing shortfall, phase transitions and actual config binding."""
import ast
import unittest
from types import SimpleNamespace as NS
import torch
from test_reward_math import reward_math as rm
import test_foot_geometry_rewards as geometry


class MinimumClearanceTests(unittest.TestCase):
    def cost(self, height, phase=None, valid=None):
        n=len(height)
        return rm.swing_min_clearance_shortfall(torch.tensor(height),
            torch.tensor([[.75,.25]]*n) if phase is None else torch.tensor(phase),
            torch.full((n,),.5),torch.ones(n,2,dtype=torch.bool) if valid is None else torch.tensor(valid))

    def test_continuous_progress_saturates_and_does_not_reward_overheight(self):
        torch.testing.assert_close(self.cost([[0.,0.],[.01,0.],[.02,0.],[.08,0.],[-.01,0.]]),
                                   torch.tensor([1.,.5,0.,0.,1.]))

    def test_other_foot_height_cannot_compensate_and_right_is_symmetric(self):
        self.assertEqual(self.cost([[0.,.10]]).item(),1.)
        self.assertEqual(self.cost([[.10,0.]],[[.25,.75]]).item(),1.)
        # With overlapping swing phases, costs sum independently.
        self.assertEqual(self.cost([[0.,0.]],[[.75,.75]]).item(),2.)

    def test_phase_boundaries_and_smooth_ramps(self):
        phases=[[p,.25] for p in (.25,.5,.6,.6375,.675,.75,.825,.8625,.9,.99)]
        torch.testing.assert_close(self.cost([[0.,0.]]*len(phases),phases),
            torch.tensor([0.,0.,0.,.5,1.,1.,1.,.5,0.,0.]),atol=1.e-6,rtol=1.e-5)

    def test_invalid_foot_does_not_disable_the_other_foot(self):
        torch.testing.assert_close(self.cost([[float('nan'),0.],[0.,0.]],[[.75,.75]]*2,
                                             [[True,True],[False,True]]),torch.ones(2))

    def test_real_adapter_and_foot_only_configuration(self):
        path=geometry._PATH
        node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='swing_min_clearance_penalty')
        scope=dict(geometry._NS,swing_min_clearance_shortfall=rm.swing_min_clearance_shortfall)
        future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future,node],type_ignores=[])),str(path),'exec'),scope)
        cfgpath=path.parent.parent/'robots/limx_wheelfoot_mode_env_cfg.py';tree=ast.parse(cfgpath.read_text())
        configs={}
        for mode in ('WFFootAllTerrainEnvCfg','WFWheelModeEnvCfg'):
            cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==mode)
            init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__post_init__')
            nodes=[n for n in init.body if isinstance(n,ast.Assign) and ast.unparse(n.targets[0]) in
                   ('self.rewards.pen_swing_min_clearance','self.rewards.pen_missed_swing')]
            cfg=NS(rewards=NS());namespace=dict(self=cfg,RewTerm=NS,
                mdp=NS(swing_min_clearance_penalty=scope['swing_min_clearance_penalty'],MissedSwingPenalty=object),
                SceneEntityCfg=lambda name,**kw:NS(name=name,body_ids=[0,1]),WHEEL_BODY_NAMES=[],
                WHEEL_GROUND_SCAN_SENSORS=('scan_L','scan_R'),WHEEL_GROUND_CONTACT_SENSORS=('contact_L','contact_R'),WHEEL_RADIUS=.128)
            exec(compile(ast.Module(body=nodes,type_ignores=[]),str(cfgpath),'exec'),namespace);configs[mode]=cfg
        self.assertFalse(vars(configs['WFWheelModeEnvCfg'].rewards))
        rewards=configs['WFFootAllTerrainEnvCfg'].rewards
        self.assertEqual(rewards.pen_missed_swing.weight,-.2)
        term=rewards.pen_swing_min_clearance;self.assertEqual(term.weight,-1.)
        env=geometry.FootGeometryRewardsTest().make_env()
        env.command_manager.get_term=lambda name:NS(phase=torch.tensor([.75]))
        env.scene['robot'].data.body_link_pos_w[:,:,2]=torch.tensor([.328,.128])
        # Opposite contact loss and nonzero/zero commands must not gate this term.
        for name in ('contact_L','contact_R'):
            env.scene[name].data.force_matrix_w_history.zero_()
        self.assertAlmostEqual(term.func(env,**term.params).item(),1.,places=5)
        env.scene['robot'].data.body_link_pos_w[0,0,2]+=.01
        self.assertAlmostEqual(term.func(env,**term.params).item(),.5,places=5)

if __name__=='__main__':unittest.main()
