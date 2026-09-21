"""Shared hold observations: command timing, resets and real MuJoCo input schemas."""
import ast
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import torch
import mujoco
import yaml
import test_foot_geometry_rewards as geometry
from test_reward_math import reward_math as rm
from test_mujoco_gait_command import deployment as d


def actual_term():
    node=next(n for n in ast.parse(geometry._PATH.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='ZeroCommandHoldPenalty')
    scope=dict(geometry._NS,ZeroCommandHoldTracker=rm.ZeroCommandHoldTracker)
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future,node],type_ignores=[])),str(geometry._PATH),'exec'),scope)
    return scope['ZeroCommandHoldPenalty']


def tiny_model():
    children=''.join(f'<body pos="0 0 {0.02*i}"><joint name="{name}"/><geom type="sphere" size=".01" mass=".1" group="1"/></body>' for i,name in enumerate(d.ALL_JOINT_NAMES))
    return mujoco.MjModel.from_xml_string(f'<mujoco><worldbody><geom type="plane" size="20 20 .1" group="0"/><body name="base_Link" pos="0 0 .8"><freejoint/><geom type="sphere" size=".05" mass="1" group="1"/>{children}</body></worldbody></mujoco>')


class HoldObservationTests(unittest.TestCase):
    def test_normalization_frame_clipping_disabled_and_nonfinite(self):
        t=rm.ZeroCommandHoldTracker(1,'cpu',.02,settle_time=.02)
        pos=torch.zeros(1,2);yaw=torch.tensor([math.pi/2]);cmd=torch.zeros(1,3)
        t.update(pos,yaw,cmd);t.update(pos,yaw,cmd)
        t.update(torch.tensor([[0.,.10]]),yaw+.2,cmd)
        obs=t.observation()
        self.assertGreater(obs[0,0],1.9);self.assertAlmostEqual(obs[0,2].item(),2.,places=4)
        torch.testing.assert_close(obs[0,3:],torch.ones(3))
        t.update(torch.tensor([[0.,100.]]),yaw,cmd)
        self.assertLessEqual(t.observation()[:,:3].abs().max().item(),5.)
        t.synchronize_command(torch.ones(1,3));torch.testing.assert_close(t.observation(),torch.zeros(1,6))
        t.update(pos*float('nan'),yaw,cmd);self.assertTrue(torch.isfinite(t.observation()).all())

    def test_observation_init_probe_before_reward_manager_exists(self):
        path=geometry._PATH.with_name('observations.py')
        node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='zero_command_hold_observation')
        module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])
        ns={'torch':torch};exec(compile(ast.fix_missing_locations(module),str(path),'exec'),ns)
        torch.testing.assert_close(ns[node.name](NS(num_envs=3,device='cpu')),torch.zeros(3,6))

    def test_training_reward_command_observation_order_matches_mujoco(self):
        model=tiny_model();data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        robot=NS(data=NS(root_link_pos_w=torch.tensor([[0.,0.,.8]]),root_link_quat_w=torch.tensor([[1.,0.,0.,0.]])))
        command=torch.zeros(1,3)
        env=NS(num_envs=1,device='cpu',step_dt=.02,common_step_counter=0,scene={'robot':robot},command_manager=NS(get_command=lambda _:command))
        params=dict(asset_cfg=NS(name='robot'),tracker_options={})
        term=actual_term()(NS(params=params),env)
        policy=d.WheelPolicy.__new__(d.WheelPolicy);policy.base_body_id=model.body('base_Link').id
        policy.hold_tracker=rm.ZeroCommandHoldTracker(1,'cpu',.02);policy._hold_started=False;policy._hold_previous_command=np.zeros(3,dtype=np.float32)
        for step in range(100):
            pos=np.array([step*.003,step*.001]);yaw=3.+step*.004
            data.qpos[:2]=pos;data.qpos[3:7]=[math.cos(yaw/2),0,0,math.sin(yaw/2)];mujoco.mj_forward(model,data)
            robot.data.root_link_pos_w[0,:2]=torch.tensor(pos);robot.data.root_link_quat_w[0]=torch.tensor(data.qpos[3:7])
            env.common_step_counter=step
            if step:term(env,**params) # physics ended, reward uses previous command
            if step in (25,65):command[:]=torch.tensor([[.4,0.,.4]])
            if step in (40,80):command.zero_()
            if step==55:
                term.reset([0]);policy.hold_tracker.reset();policy._hold_started=False;policy._hold_previous_command.fill(0.)
            actual=term.observe(env).clone();age=term.tracker.age.clone()
            torch.testing.assert_close(term.observe(env),actual);torch.testing.assert_close(term.tracker.age,age)
            expected=policy._hold_observation(data,np.array([*command[0].tolist(),.8],dtype=np.float32))
            np.testing.assert_allclose(actual.numpy()[0],expected,atol=2.e-5,rtol=1.e-5)

    def checkpoint(self, root, new):
        path=root/'model_0.pt';(root/'params').mkdir(exist_ok=True)
        dim=168 if new else 162
        weight=torch.zeros(8,dim)
        if new:weight[:6,158:164]=torch.eye(6)
        torch.save({'iter':0,'model_state_dict':{'actor.0.weight':weight,'actor.0.bias':torch.zeros(8)},
                    'encoder_state_dict':{'encoder.0.weight':torch.zeros(3,340),'encoder.0.bias':torch.zeros(3)}},path)
        config={'commands':{'gait_command':{'continuous_phase':True}},'observations':{'policy':{}},'rewards':{}}
        if new:
            config['observations']['policy']['zero_command_hold']={'func':'module:zero_command_hold_observation'}
            config['rewards']['pen_zero_command_hold']={'weight':-.5,'params':{'tracker_options':{'settle_time':.04,'position_scale':.05,'yaw_scale':.1}}}
        (root/'params/env.yaml').write_text(yaml.safe_dump(config));return path

    def test_real_new_actor_receives_six_features_and_legacy_still_loads(self):
        model=tiny_model();data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        gait=np.array([1.7,.5,.5,0.],dtype=np.float32);command=np.array([0,0,0,.8],dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for new in (False,True):
                path=self.checkpoint(root,new);scanner=d.TerrainHeightScanner(model,model.body('base_Link').id)
                policy=d.WheelPolicy(path,torch.device('cpu'),model,scanner,(.65,.85));policy.reset(data,gait)
                for _ in range(4):out=policy.act(data,command,gait)
                if new:
                    np.testing.assert_allclose(out[3:6],np.ones(3))
                    data.qpos[0]+=.1;mujoco.mj_forward(model,data)
                    out=policy.act(data,command,gait);self.assertAlmostEqual(float(out[0]),2.,places=4)
                    out=policy.act(data,np.array([.4,0,0,.8],dtype=np.float32),gait)
                    self.assertEqual(out[0],0.);self.assertEqual(out[3],0.)
                else:np.testing.assert_allclose(out,np.zeros(8))
                self.assertEqual(len(policy.history[-1]),34)
                policy.reset(data,gait);np.testing.assert_allclose(policy.act(data,command,gait),np.zeros(8))
            (root/'params/env.yaml').unlink()
            with self.assertRaisesRegex(ValueError,'配置快照'):
                d.WheelPolicy(path,torch.device('cpu'),model,scanner,(.65,.85))

    def test_common_schema_appends_only_actor_and_critic(self):
        path=geometry._PATH.parent.parent/'robots/limx_wheelfoot_mode_env_cfg.py'
        fn=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_configure_common_mode')
        node=next(n for n in fn.body if isinstance(n,ast.For) and 'group.zero_command_hold' in ast.unparse(n))
        cfg=NS(observations=NS(policy=NS(),critic=NS(),obsHistory=NS()))
        ns={'cfg':cfg,'ObsTerm':NS,'mdp':NS(zero_command_hold_observation=object)}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
        self.assertTrue(hasattr(cfg.observations.policy,'zero_command_hold'))
        self.assertTrue(hasattr(cfg.observations.critic,'zero_command_hold'))
        self.assertFalse(hasattr(cfg.observations.obsHistory,'zero_command_hold'))

if __name__=='__main__':unittest.main()
