"""Directional planning and fixed-per-landing costs, without Isaac Sim."""
from pathlib import Path
from types import SimpleNamespace as NS
import ast
import math
import unittest
import torch
from test_reward_math import reward_math as rm
import test_foot_geometry_rewards as geometry


def inputs(n=1):
    x,y=torch.meshgrid(torch.linspace(-.5,.5,11),torch.linspace(-.5,.5,11),indexing='ij')
    return dict(base_pos=torch.tensor([[0.,0.,.8]]).repeat(n,1),heading=torch.zeros(n),
        velocity_b=torch.zeros(n,3),command=torch.zeros(n,3),gait=torch.tensor([[2.,.5,.5,0.]]).repeat(n,1),
        phase=torch.full((n,),.49),hits=torch.stack((x.flatten(),y.flatten(),torch.zeros(121)),-1)[None].repeat(n,1,1),
        wheel_pos=torch.tensor([[[0.,.17,.128],[0.,-.17,.128]]]).repeat(n,1,1),
        ground_force=torch.full((n,2),100.),clearance=torch.zeros(n,2))


def plan(data):
    return rm.foothold_region_plan(*(data[k] for k in ('base_pos','heading','velocity_b','command','gait')),
                                   torch.full((len(data['phase']),2),.5),data['hits'])


class FootholdRegionTest(unittest.TestCase):
    def test_direction_and_turning_targets_are_not_forced_mirrors(self):
        d=inputs(7)
        d['command'][:]=torch.tensor([[0.,0.,0.],[.4,0.,0.],[-.4,0.,0.],[0.,.2,0.],[0.,-.2,0.],[0.,0.,.4],[0.,0.,-.4]])
        d['velocity_b'][:]=d['command']
        center,_,_,_,valid=plan(d)
        self.assertTrue(valid.all())
        self.assertTrue((center[1,:,0]>center[0,:,0]).all())
        self.assertTrue((center[2,:,0]<center[0,:,0]).all())
        self.assertTrue((center[3,:,1]>center[0,:,1]).all())
        self.assertTrue((center[4,:,1]<center[0,:,1]).all())
        self.assertLess(center[5,0,0],0.);self.assertGreater(center[5,1,0],0.)
        self.assertGreater(center[6,0,0],0.);self.assertLess(center[6,1,0],0.)
        torch.testing.assert_close(center[0,:,:2],torch.tensor([[0.,.17],[0.,-.17]]))

    def test_world_translation_and_yaw_equivariance(self):
        d=inputs();d['command'][:]=torch.tensor([[.4,.1,.2]])
        original=plan(d)
        rotation=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        shift=torch.tensor([4.,-3.,.2])
        d['base_pos']=d['base_pos']@rotation.T+shift
        d['hits']=d['hits']@rotation.T+shift
        d['heading'][:]=math.pi/2
        transformed=plan(d)
        torch.testing.assert_close(transformed[0],original[0]@rotation.T+shift,atol=1e-5,rtol=1e-5)
        self.assertTrue(transformed[-1].all())

    def test_invalid_scans_rough_terrain_and_unreachable_height_skip(self):
        d=inputs(4)
        d['hits'][0]=float('nan')
        d['hits'][1,:,2]=d['hits'][1,:,0]*2.  # too steep
        d['hits'][2,60,2]=.1  # coarse step/obstacle => first version skips
        d['base_pos'][3,2]=2.
        self.assertFalse(plan(d)[-1].any())

    def simulate_landing(self, dt=.02, distance=.08):
        d=inputs();tracker=rm.FootholdRegionTracker(1,'cpu')
        tracker.update(**d,dt=dt)
        d['phase'][:]=.51;tracker.update(**d,dt=dt)
        target=tracker.center.clone()
        d['ground_force'][0,0]=0.;d['clearance'][0,0]=.05
        for _ in range(round(.06/dt)):
            d['phase']+=2*dt;tracker.update(**d,dt=dt)
        d['wheel_pos'][0,0]=target[0,0]+torch.tensor([distance,0.,.128])
        d['ground_force'][0,0]=100.;d['clearance'][0,0]=0.
        total=0.; events=0
        for _ in range(round(.04/dt)+2):
            d['phase']+=2*dt
            total+=float(tracker.update(**d,dt=dt));events+=int(tracker.events.sum())
        return total,events,tracker,d

    def test_one_landing_one_cost_and_event_cost_is_dt_independent(self):
        for dt in (.01,.02):
            total,events,_,_=self.simulate_landing(dt)
            self.assertEqual(events,1)
            self.assertAlmostEqual(total,.5,places=5)  # rho=2, Huber(rho-1)=.5
        self.assertAlmostEqual(self.simulate_landing(distance=.02)[0],0.)

    def test_contact_chatter_and_no_liftoff_do_not_create_extra_events(self):
        total,events,t,d=self.simulate_landing()
        for force in (0.,100.,0.,100.,100.,100.):
            d['ground_force'][0,0]=force
            self.assertEqual(t.update(**d,dt=.02).item(),0.)
            self.assertFalse(t.events.any())
        t.reset();d=inputs()
        for _ in range(50):
            d['phase']=(d['phase']+.04)%1
            self.assertEqual(t.update(**d,dt=.02).item(),0.)
        self.assertFalse(t.events.any())

    def test_target_freezes_during_swing_and_partial_reset_clears_only_one_env(self):
        d=inputs(2);t=rm.FootholdRegionTracker(2,'cpu')
        t.update(**d,dt=.02);d['phase'][:]=.51;t.update(**d,dt=.02)
        old=t.center.clone()
        d['base_pos'][:,0]+=.2;d['command'][:,0]=-.4
        d['phase'][:]=.6;t.update(**d,dt=.02)
        torch.testing.assert_close(t.center,old)
        t.reset([0]);self.assertFalse(t.initialized[0].any());self.assertTrue(t.initialized[1].all())
        torch.testing.assert_close(t.center[1],old[1])

    def test_no_startup_airborne_credit(self):
        d=inputs();d['ground_force'].zero_();d['clearance'][:]=.1
        t=rm.FootholdRegionTracker(1,'cpu')
        for _ in range(4):t.update(**d,dt=.02)
        d['ground_force'][:]=100.;d['clearance'].zero_()
        for _ in range(4):
            self.assertEqual(t.update(**d,dt=.02).item(),0.)
            self.assertFalse(t.events.any())

    def test_two_contact_confirmations_use_first_touch_position(self):
        d=inputs();t=rm.FootholdRegionTracker(1,'cpu')
        t.update(**d,dt=.02);d['phase'][:]=.51;t.update(**d,dt=.02)
        d['ground_force'][0,0]=0.;d['clearance'][0,0]=.05
        for _ in range(3):
            d['phase']+=.04;t.update(**d,dt=.02)
        center=t.center[0,0].clone()
        d['ground_force'][0,0]=100.;d['clearance'][0,0]=0.
        d['wheel_pos'][0,0]=center+torch.tensor([.02,0.,.128])
        self.assertEqual(t.update(**d,dt=.02).item(),0.)
        self.assertFalse(t.events.any())
        d['wheel_pos'][0,0,0]+=.2  # post-touch sliding must not move the scored landing
        self.assertEqual(t.update(**d,dt=.02).item(),0.)
        self.assertTrue(t.events[0,0]);self.assertAlmostEqual(t.rho[0,0].item(),.5,places=5)

    def test_diagnostic_counts_use_only_valid_events_and_clear_on_reset(self):
        import test_foot_velocity_command as cmdtest
        env,term=cmdtest.make_env(2)
        base_get=env.reward_manager.get_term_cfg
        tracker=NS(events=torch.tensor([[True,False],[False,False]]),
            rho=torch.tensor([[.5,float('nan')],[float('nan'),float('nan')]]),
            new_regions=torch.tensor([[True,False],[True,False]]),
            invalid_regions=torch.tensor([[False,False],[True,False]]),
            missed_swings=torch.zeros(2,2,dtype=torch.bool))
        env.reward_manager.active_terms=['pen_foothold_region']
        env.reward_manager.get_term_cfg=lambda name:NS(func=NS(tracker=tracker)) if name=='pen_foothold_region' else base_get(name)
        term._update_metrics();stats=term.diagnostic_means()
        self.assertEqual(stats['foothold/landings'],1.)
        self.assertEqual(stats['foothold/inside_rate'],1.)
        self.assertEqual(stats['foothold/rho_mean'],.5)
        self.assertEqual(stats['foothold/invalid_rate'],.5)
        term.reset([0]);self.assertEqual(term.diagnostic_means()['foothold/landings'],0.)

    def test_actual_manager_cancels_dt_and_is_idempotent(self):
        path=geometry._PATH
        node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='FootholdRegionPenalty')
        scope=dict(geometry._NS,FootholdRegionTracker=rm.FootholdRegionTracker)
        module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])
        exec(compile(ast.fix_missing_locations(module),str(path),'exec'),scope)
        env=geometry.FootGeometryRewardsTest().make_env()
        env.common_step_counter=0
        env.scene['robot'].data.root_link_pos_w=torch.tensor([[0.,0.,.8]])
        for name in ('contact_L','contact_R'):
            sensor=env.scene[name].data
            sensor.force_matrix_w=sensor.force_matrix_w_history[:,-1].clone()
        args=dict(asset_cfg=NS(name='robot',body_ids=[0,1]),contact_sensor_names=('contact_L','contact_R'),
                  terrain_sensor_names=('scan_L','scan_R'),height_sensor_cfg=NS(name='height_scanner'))
        t=scope['FootholdRegionPenalty'](NS(params=args),env)
        # Control the pure event cost to verify the framework adapter itself.
        calls=[]
        t.tracker.update=lambda *a,**kw:(calls.append(1) or torch.tensor([.5]))
        result=t(env,**args).clone()
        self.assertAlmostEqual((result*env.step_dt).item(),.5)
        torch.testing.assert_close(t(env,**args),result);self.assertEqual(len(calls),1)
        t.reset([0]);self.assertEqual(t(env,**args).item(),0.)

if __name__=='__main__':unittest.main()
