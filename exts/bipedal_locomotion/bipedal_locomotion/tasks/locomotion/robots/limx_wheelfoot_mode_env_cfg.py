"""Dual-expert environments for the TRON1A wheeled-foot robot."""

import math

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as GaussianNoise

from bipedal_locomotion.tasks.locomotion import mdp
from bipedal_locomotion.tasks.locomotion.cfg.WF.terrains_cfg import (
    FOOT_ALL_TERRAINS_CFG,
    FOOT_ALL_TERRAINS_PLAY_CFG,
    WHEEL_HEIGHT_PRETRAIN_TERRAINS_CFG,
    WHEEL_MODE_TERRAINS_CFG,
    WHEEL_MODE_TERRAINS_PLAY_CFG,
)

from .limx_wheelfoot_env_cfg import WFBaseEnvCfg, WFBaseEnvCfg_PLAY


WHEEL_LINKS = "wheel_[LR]_Link"
WHEEL_BODY_NAMES = ["wheel_L_Link", "wheel_R_Link"]
WHEEL_JOINTS = ["wheel_L_Joint", "wheel_R_Joint"]
LEG_JOINTS = [
    "abad_L_Joint",
    "abad_R_Joint",
    "hip_L_Joint",
    "hip_R_Joint",
    "knee_L_Joint",
    "knee_R_Joint",
]
WHEEL_RADIUS = 0.128
WHEEL_GROUND_CONTACT_SENSORS = ("wheel_L_ground_contact", "wheel_R_ground_contact")
WHEEL_GROUND_SCAN_SENSORS = ("wheel_L_ground_scan", "wheel_R_ground_scan")
WHEEL_NEUTRAL_XY = ((0.0, 0.17), (0.0, -0.17))
BODY_HEIGHT_RANGE = (0.65, 0.85)



def _configure_wheel_ground_sensors(cfg, *, play: bool):
    """Add ground-filtered wheel support sensors shared by both experts."""
    ground_mesh_path = "/World/ground/terrain/mesh"
    for side in ("L", "R"):
        setattr(
            cfg.scene,
            f"wheel_{side}_ground_contact",
            ContactSensorCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/wheel_{side}_Link",
                history_length=4,
                track_air_time=False,
                update_period=0.0,
                filter_prim_paths_expr=[ground_mesh_path],
            ),
        )
        setattr(
            cfg.scene,
            f"wheel_{side}_ground_scan",
            RayCasterCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/wheel_{side}_Link",
                offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.25)),
                attach_yaw_only=True,
                pattern_cfg=patterns.GridPatternCfg(resolution=0.04, size=(0.12, 0.08)),
                debug_vis=play,
                mesh_prim_paths=["/World/ground"],
                update_period=cfg.decimation * cfg.sim.dt,
            ),
        )

def _configure_common_mode(cfg, *, play: bool):
    """Install the common observation/command schema used by both experts."""
    _configure_wheel_ground_sensors(cfg, play=play)
    cfg.scene.height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_Link",
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.0, 1.0]),
        debug_vis=play,
        mesh_prim_paths=["/World/ground"],
    )
    cfg.scene.height_scanner.update_period = cfg.decimation * cfg.sim.dt

    height_term = ObsTerm(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        clip=(0.0, 10.0),
    )
    cfg.observations.policy.heights = height_term
    cfg.observations.critic.heights = ObsTerm(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        clip=(0.0, 10.0),
    )

    cfg.commands.body_height = mdp.UniformBodyHeightCommandCfg(
        resampling_time_range=(5.0, 8.0),
        max_rate=0.08,
        endpoint_fraction=0.20,
        debug_vis=False,
        ranges=mdp.UniformBodyHeightCommandCfg.Ranges(height=BODY_HEIGHT_RANGE),
    )
    # Keep identical gait inputs for both experts so their checkpoint schemas match.
    cfg.commands.gait_command = mdp.UniformGaitCommandCfg(
        resampling_time_range=(5.0, 8.0),
        debug_vis=False,
        ranges=mdp.UniformGaitCommandCfg.Ranges(
            frequencies=(1.2, 2.2),
            offsets=(0.5, 0.5),
            durations=(0.45, 0.60),
            swing_height=(0.08, 0.20),
        ),
    )

    cfg.observations.commands.body_height_command = ObsTerm(
        func=mdp.normalized_body_height_command, params={"command_name": "body_height"}
    )
    cfg.observations.policy.gait_phase = ObsTerm(func=mdp.get_gait_phase)
    cfg.observations.policy.gait_command = ObsTerm(
        func=mdp.get_gait_command, params={"command_name": "gait_command"}
    )
    cfg.observations.obsHistory.gait_phase = ObsTerm(func=mdp.get_gait_phase)
    cfg.observations.obsHistory.gait_command = ObsTerm(
        func=mdp.get_gait_command, params={"command_name": "gait_command"}
    )
    cfg.observations.critic.gait_phase = ObsTerm(func=mdp.get_gait_phase)
    cfg.observations.critic.gait_command = ObsTerm(
        func=mdp.get_gait_command, params={"command_name": "gait_command"}
    )

    # Append identical hold state after the existing gait inputs. History stays
    # proprioceptive: the actor and critic receive the current reference error.
    for group in (cfg.observations.policy, cfg.observations.critic):
        group.zero_command_hold = ObsTerm(func=mdp.zero_command_hold_observation)

    support_params = {
        "contact_sensor_names": WHEEL_GROUND_CONTACT_SENSORS,
        "force_off": 5.0,
        "force_on": 10.0,
    }
    cfg.rewards.stand_still = RewTerm(
        func=mdp.stand_still_grounded,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot"), **support_params},
    )
    cfg.rewards.pen_base_height = RewTerm(
        func=mdp.body_height_command_grounded_l2,
        weight=-30.0,
        params={
            "command_name": "body_height",
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            **support_params,
        },
    )
    cfg.rewards.pen_base_contact_termination = RewTerm(
        # is_terminated_term() reads the manager's persistent per-term history,
        # so a base-contact flag can remain true after reset and be penalized on
        # every step of the next episode. is_terminated() is the current-step,
        # non-timeout termination signal. Wheel currently has base_contact;
        # Foot additionally installs sustained_knee_contact below.
        func=mdp.is_terminated,
        weight=-500.0,
    )

    # Start the new experts without stochastic pushes; re-enable after locomotion converges.
    cfg.events.push_robot = None
    # Training does not need remote marker assets. Keeping this disabled also
    # allows fully offline headless environment creation.
    cfg.commands.base_velocity.debug_vis = play

    if play:
        cfg.scene.num_envs = 1
        cfg.observations.policy.enable_corruption = False


@configclass
class WFWheelModeEnvCfg(WFBaseEnvCfg):
    """Wheel expert: continuous rolling terrain, no intentional stepping."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.pen_zero_command_hold = RewTerm(
            func=mdp.ZeroCommandHoldPenalty, weight=-0.5,
            params={
                "asset_cfg": SceneEntityCfg("robot"), "command_name": "base_velocity",
                "tracker_options": {
                    "zero_threshold": 0.02, "settle_time": 0.3, "window_s": 1.0,
                    "position_deadband": 0.015, "position_scale": 0.05,
                    "yaw_deadband": 0.03, "yaw_scale": 0.10, "max_cost": 5.0,
                },
            },
        )

        _configure_common_mode(self, play=False)
        self.rewards.stand_still.weight = -7.0

        # The Wheel expert uses an explicit PI wheel-velocity controller.  Keep
        # the PhysX wheel drives passive so implicit gains are not added on top
        # of the PI effort, and leave the existing actuator-gain randomization
        # scoped to the six leg joints.
        self.scene.robot.actuators["wheels"].stiffness = 0.0
        self.scene.robot.actuators["wheels"].damping = 0.0
        self.events.robot_joint_stiffness_and_damping.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=LEG_JOINTS
        )
        self.events.randomize_actuator_gains.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=LEG_JOINTS
        )
        self.actions.joint_vel = mdp.WheelVelocityPIActionCfg(
            asset_name="robot",
            joint_names=WHEEL_JOINTS,
            scale=1.0,
            preserve_order=True,
            kp=2.0,
            kp_scale_range=(0.25, 2.0),
            ki_ratio_range=(0.0, 0.25),
            effort_limit=80.0,
        )

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = WHEEL_MODE_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = 0
        self.curriculum.terrain_levels = CurrTerm(
            func=mdp.wheel_terrain_levels_vel_tracking,
            params={
                "command_name": "base_velocity",
                "asset_name": "robot",
                "distance_fraction_up": 0.5,
                "moving_rate_threshold": 0.25,
                "tracking_up": 0.55,
                "tracking_down": 0.25,
                "support_up": 0.75,
            },
        )

        self.commands.body_height.ranges.height = BODY_HEIGHT_RANGE
        self.commands.base_velocity.rel_standing_envs = 0.25
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)

        # Wheel locomotion rewards use filtered ground force and wheel-center
        # clearance consistent with the wheel radius. Tracking-related terms
        # are fully active when either wheel has valid support (C_any).
        contact_params = {
            "asset_cfg": SceneEntityCfg("robot", body_names=WHEEL_BODY_NAMES),
            "contact_sensor_names": WHEEL_GROUND_CONTACT_SENSORS,
            "terrain_sensor_names": WHEEL_GROUND_SCAN_SENSORS,
            "wheel_radius": WHEEL_RADIUS,
            "force_off": 5.0,
            "force_on": 10.0,
            "geometry_tolerance_on": 0.02,
            "geometry_tolerance_off": 0.035,
        }
        self.rewards.rew_lin_vel_xy = RewTerm(
            func=mdp.track_lin_vel_xy_exp_any_wheel_ground,
            weight=3.5,
            params={"command_name": "base_velocity", "std": math.sqrt(0.12), **contact_params},
        )
        self.rewards.rew_ang_vel_z = RewTerm(
            func=mdp.track_ang_vel_z_exp_any_wheel_ground,
            weight=1.5,
            params={"command_name": "base_velocity", "std": math.sqrt(0.12), **contact_params},
        )
        self.rewards.pen_base_lin_acc_xy = RewTerm(
            func=mdp.BaseVelocityAccelerationPenalty,
            weight=0.0,
            params={
                "command_name": "base_velocity",
                "component": "xy",
                "tracking_std": 0.30,
                "acceleration_scale": 3.0,
                "kernel": "bounded",
                **contact_params,
            },
        )
        self.rewards.pen_base_yaw_acc = RewTerm(
            func=mdp.BaseVelocityAccelerationPenalty,
            weight=-0.025,
            params={
                "command_name": "base_velocity",
                "component": "yaw",
                "tracking_std": 0.30,
                "acceleration_scale": 4.0,
                "kernel": "charbonnier",
                **contact_params,
            },
        )
        self.rewards.pen_base_height = RewTerm(
            func=mdp.body_height_command_plane_grounded_l2,
            weight=-60.0,
            params={
                "command_name": "body_height",
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "contact_sensor_names": WHEEL_GROUND_CONTACT_SENSORS,
                "force_off": 5.0,
                "force_on": 10.0,
            },
        )
        self.rewards.rew_base_height_exp = RewTerm(
            func=mdp.body_height_command_plane_grounded_exp,
            weight=1.0,
            params={
                "command_name": "body_height",
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "contact_sensor_names": WHEEL_GROUND_CONTACT_SENSORS,
                "std": 0.05,
                "force_off": 5.0,
                "force_on": 10.0,
            },
        )

        # Replace the previous x/z symmetry, wheel-spacing and relative-velocity
        # terms with one direct horizontal neutral-position penalty.
        self.rewards.rew_same_foot_x_position = None
        self.rewards.pen_feet_distance = None
        self.rewards.pen_joint_vel_wheel_l2 = None
        self.rewards.pen_flat_orientation_l2 = None
        self.rewards.pen_vel_non_wheel_l2.weight = -0.08
        self.rewards.pen_action_rate.weight = -0.15
        self.rewards.pen_action_smoothness.weight = -0.08
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.pen_lin_vel_z = RewTerm(
            func=mdp.lin_vel_z_height_command_gated_l2,
            weight=-0.3,
            params={
                "command_name": "body_height",
                "min_scale": 0.2,
                "full_penalty_gap": 0.01,
                "reduced_penalty_gap": 0.04,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.pen_wheel_contact = RewTerm(
            func=mdp.wheel_ground_contact_loss,
            weight=-4.0,
            params=contact_params,
        )
        self.rewards.pen_wheel_horizontal_neutral = RewTerm(
            func=mdp.wheel_horizontal_neutral_l2,
            weight=-4.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=WHEEL_BODY_NAMES),
                "neutral_xy": WHEEL_NEUTRAL_XY,
                "tolerance_xy": (0.04, 0.03),
                "scale_xy": (0.02, 0.02),
            },
        )
        self.rewards.pen_rolling_error = RewTerm(
            func=mdp.wheel_rolling_velocity_error,
            weight=-2.0,
            params={
                "radius": WHEEL_RADIUS,
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=WHEEL_JOINTS,
                    body_names=WHEEL_BODY_NAMES,
                    preserve_order=True,
                ),
            },
        )
        self.rewards.pen_wheel_target_symmetry = RewTerm(
            func=mdp.wheel_target_symmetry_l2,
            weight=-0.5,
            params={
                "action_name": "joint_vel",
                "command_name": "base_velocity",
                "yaw_threshold": 0.05,
                "target_difference_tolerance": 0.10,
            },
        )
        self.rewards.pen_zero_command_wheel_target = RewTerm(
            func=mdp.zero_command_wheel_target_l2,
            weight=-0.10,
            params={
                "action_name": "joint_vel",
                "command_name": "base_velocity",
                "linear_threshold": 0.05,
                "angular_threshold": 0.05,
                "target_tolerance": 0.15,
            },
        )
        self.rewards.pen_terrain_orientation = RewTerm(
            func=mdp.terrain_aligned_orientation_l2,
            weight=-10.0,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        )
        self.rewards.pen_wheel_landing_impact = RewTerm(
            func=mdp.wheel_landing_impact_l2,
            weight=-0.02,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=WHEEL_LINKS),
                "force_threshold": 100.0,
                "force_scale": 100.0,
            },
        )
        self.rewards.pen_wheel_stance_slip = RewTerm(
            func=mdp.wheel_stance_slip_l2,
            weight=-0.5,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=WHEEL_LINKS),
                "asset_cfg": SceneEntityCfg("robot", body_names=WHEEL_LINKS),
                "terrain_sensor_cfg": SceneEntityCfg("height_scanner"),
                "wheel_radius": WHEEL_RADIUS,
                "force_off": 5.0,
                "force_on": 10.0,
            },
        )

        # Explicitly sample standing, straight, yaw-only and mixed commands,
        # while logging support-gate, tracking and command-mode diagnostics.
        velocity_cfg = self.commands.base_velocity
        self.commands.base_velocity = mdp.WheelSupportVelocityCommandCfg(
            asset_name=velocity_cfg.asset_name,
            heading_command=velocity_cfg.heading_command,
            heading_control_stiffness=velocity_cfg.heading_control_stiffness,
            rel_standing_envs=velocity_cfg.rel_standing_envs,
            rel_heading_envs=velocity_cfg.rel_heading_envs,
            ranges=velocity_cfg.ranges,
            resampling_time_range=velocity_cfg.resampling_time_range,
            debug_vis=velocity_cfg.debug_vis,
            goal_vel_visualizer_cfg=velocity_cfg.goal_vel_visualizer_cfg,
            current_vel_visualizer_cfg=velocity_cfg.current_vel_visualizer_cfg,
            contact_sensor_names=WHEEL_GROUND_CONTACT_SENSORS,
            terrain_sensor_names=WHEEL_GROUND_SCAN_SENSORS,
            wheel_body_names=tuple(WHEEL_BODY_NAMES),
            wheel_radius=WHEEL_RADIUS,
            force_off=contact_params["force_off"],
            force_on=contact_params["force_on"],
            geometry_tolerance_on=contact_params["geometry_tolerance_on"],
            geometry_tolerance_off=contact_params["geometry_tolerance_off"],
            support_threshold=0.5,
            tracking_std=math.sqrt(0.12),
            moving_command_threshold=0.1,
            rel_straight_envs=0.30,
            rel_yaw_only_envs=0.10,
            rel_mixed_envs=0.35,
        )


@configclass
class WFWheelModeEnvCfg_PLAY(WFWheelModeEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        # Follow the robot during headless video capture.  The default
        # environment-relative camera is too far away on generated terrain
        # and can end up recording only the terrain tile.
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.eye = (3.0, 3.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.3)
        self.scene.height_scanner.debug_vis = True
        self.scene.wheel_L_ground_scan.debug_vis = True
        self.scene.wheel_R_ground_scan.debug_vis = True
        self.observations.policy.enable_corruption = False
        # Evaluation uses the nominal PI controller deterministically.
        self.actions.joint_vel.kp_scale_range = (1.0, 1.0)
        self.actions.joint_vel.ki_ratio_range = (0.25, 0.25)
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.scene.terrain.terrain_generator = WHEEL_MODE_TERRAINS_PLAY_CFG
        self.scene.terrain.max_init_terrain_level = None


@configclass
class WFWheelHeightPretrainEnvCfg(WFWheelModeEnvCfg):
    """Flat-terrain first stage for learning commanded height and zero-command stability."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = WHEEL_HEIGHT_PRETRAIN_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = None
        self.curriculum.terrain_levels = None


@configclass
class WFFootAllTerrainEnvCfg(WFBaseEnvCfg):
    """Foot expert with fixed-zero PI wheel references and cycle-latched swing peaks."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.pen_zero_command_hold = RewTerm(
            func=mdp.ZeroCommandHoldPenalty, weight=-0.5,
            params={
                "asset_cfg": SceneEntityCfg("robot"), "command_name": "base_velocity",
                "tracker_options": {
                    "zero_threshold": 0.02, "settle_time": 0.3, "window_s": 1.0,
                    "position_deadband": 0.03, "position_scale": 0.05,
                    "yaw_deadband": 0.03, "yaw_scale": 0.10, "max_cost": 5.0,
                },
            },
        )

        _configure_common_mode(self, play=False)

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = FOOT_ALL_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = 0

        # Fix Foot wheel-speed targets to zero and brake with explicit PI,
        # shared with deployment. Implicit wheel
        # drives are disabled and their gains are excluded from the generic
        # actuator randomization to avoid stacking two controllers.
        self.scene.robot.actuators["wheels"].stiffness = 0.0
        self.scene.robot.actuators["wheels"].damping = 0.0
        self.events.robot_joint_stiffness_and_damping.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=LEG_JOINTS
        )
        self.events.randomize_actuator_gains.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=LEG_JOINTS
        )
        self.actions.joint_vel = mdp.WheelVelocityPIActionCfg(
            asset_name="robot",
            joint_names=WHEEL_JOINTS,
            scale=1.0,
            clip={"wheel_.*": (0.0, 0.0)},
            fixed_zero_target=True,
            preserve_order=True,
            kp=2.0,
            kp_scale_range=(0.25, 2.0),
            ki_ratio_range=(0.0, 0.25),
            effort_limit=80.0,
        )

        self.commands.body_height.ranges.height = BODY_HEIGHT_RANGE
        self.commands.body_height.max_rate = 0.06
        self.commands.base_velocity.rel_standing_envs = 0.15
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (-0.8, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.4, 0.4)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)
        self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)
        velocity_cfg = self.commands.base_velocity
        self.commands.base_velocity = mdp.FootVelocityCommandCfg(
            asset_name=velocity_cfg.asset_name,
            heading_command=velocity_cfg.heading_command,
            heading_control_stiffness=velocity_cfg.heading_control_stiffness,
            rel_standing_envs=0.15,
            rel_heading_envs=0.0,
            rel_straight_envs=0.25,
            rel_lateral_envs=0.10,
            rel_yaw_only_envs=0.20,
            rel_mixed_envs=0.30,
            ranges=velocity_cfg.ranges,
            resampling_time_range=velocity_cfg.resampling_time_range,
            debug_vis=velocity_cfg.debug_vis,
            goal_vel_visualizer_cfg=velocity_cfg.goal_vel_visualizer_cfg,
            current_vel_visualizer_cfg=velocity_cfg.current_vel_visualizer_cfg,
            wheel_body_names=tuple(WHEEL_BODY_NAMES),
            leg_joint_names=tuple(LEG_JOINTS),
            terrain_sensor_names=WHEEL_GROUND_SCAN_SENSORS,
            wheel_radius=WHEEL_RADIUS,
        )
        # Preserve the shared four-dimensional checkpoint schema. The swing
        # clearance target is a reward parameter, not an observation command.
        self.commands.gait_command.continuous_phase = True
        self.commands.gait_command.ranges.durations = (0.5, 0.5)
        self.commands.gait_command.ranges.swing_height = (0.0, 0.0)

        # Follow the PF-style locomotion regularization while retaining the
        # wheel-specific zero-speed and terrain-aware gait constraints.
        self.rewards.stand_still = None
        self.rewards.pen_joint_torque.weight = -8.0e-5
        self.rewards.pen_joint_accel.weight = -2.5e-7
        self.rewards.pen_action_rate.weight = -0.03
        self.rewards.pen_action_smoothness.weight = -0.04
        self.rewards.pen_joint_power_l1.weight = -5.0e-4
        self.rewards.pen_vel_non_wheel_l2.weight = -1.0e-3
        for name in ("pen_joint_torque", "pen_joint_accel", "pen_joint_power_l1"):
            getattr(self.rewards, name).params["asset_cfg"] = SceneEntityCfg(
                "robot", joint_names=LEG_JOINTS, preserve_order=True,
            )
        self.rewards.pen_action_rate.func = mdp.leg_action_rate_l2
        self.rewards.pen_action_rate.params = {"action_dim": 6}
        self.rewards.pen_action_smoothness.params = {"action_dim": 6}

        # Foot must keep stepping even at zero command. Track base motion
        # independently of gait, restoring the wider h08 tracking kernels.
        self.rewards.rew_lin_vel_xy.weight = 5.5
        self.rewards.rew_lin_vel_xy.params["std"] = math.sqrt(0.20)
        self.rewards.rew_ang_vel_z.weight = 3.0
        self.rewards.rew_ang_vel_z.params["std"] = 0.5
        self.rewards.pen_lin_vel_xy_tracking_error = RewTerm(
            func=mdp.lin_vel_xy_tracking_error_huber,
            weight=-0.5,
            params={
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot"),
                "error_scale": 0.4,
            },
        )
        self.rewards.pen_yaw_tracking_error = RewTerm(
            func=mdp.ang_vel_z_tracking_error_huber,
            weight=-1.0,
            params={
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot"),
                "error_scale": 0.5,
            },
        )
        acceleration_params = {
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names=WHEEL_BODY_NAMES, preserve_order=True),
            "contact_sensor_names": WHEEL_GROUND_CONTACT_SENSORS,
            "terrain_sensor_names": WHEEL_GROUND_SCAN_SENSORS,
            "wheel_radius": WHEEL_RADIUS,
            "force_off": 5.0,
            "force_on": 10.0,
            "geometry_tolerance_on": 0.02,
            "geometry_tolerance_off": 0.035,
            "tracking_std": 0.30,
            "min_tracking_gate": 0.1,
            "kernel": "charbonnier",
        }
        self.rewards.pen_base_lin_acc_xy = RewTerm(
            func=mdp.BaseVelocityAccelerationPenalty,
            weight=0.0,
            params={
                **acceleration_params, "component": "xy", "acceleration_scale": 3.0,
                "linear_velocity_frame": "world",
            },
        )
        self.rewards.pen_base_yaw_acc = RewTerm(
            func=mdp.BaseVelocityAccelerationPenalty,
            weight=0.0,
            params={**acceleration_params, "component": "yaw", "acceleration_scale": 4.0},
        )

        # v8: mean signed tracking errors over a full gait cycle, alongside
        # unchanged instantaneous tracking. Weights/scales are initial values.
        for component, weight, scale in (
            ("vx", -1.0, 0.2), ("vy", -1.0, 0.2), ("yaw", -1.0, 0.3),
            ("height", -0.1, 0.02), ("roll", -0.1, 0.05235987756),
            ("pitch", -0.1, 0.05235987756),
        ):
            setattr(self.rewards, f"pen_cycle_mean_{component}", RewTerm(
                func=mdp.GaitCycleMeanTrackingPenalty, weight=weight,
                params={
                    "command_name": "base_velocity", "gait_command_name": "gait_command",
                    "asset_cfg": SceneEntityCfg("robot"), "component": component,
                    "error_scale": scale,
                },
            ))

        # Use the same local-plane height definition and any-wheel support gate
        # as Wheel Expert, while retaining Foot's existing L2 weight.
        height_params = {
            "command_name": "body_height",
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "contact_sensor_names": WHEEL_GROUND_CONTACT_SENSORS,
            "force_off": 5.0,
            "force_on": 10.0,
        }
        self.rewards.pen_base_height = RewTerm(
            func=mdp.body_height_command_plane_grounded_l2,
            weight=-30.0,
            params=height_params,
        )
        self.rewards.rew_base_height_exp = RewTerm(
            func=mdp.body_height_command_plane_grounded_exp,
            weight=1.5,
            params={**height_params, "std": 0.05},
        )

        self.rewards.rew_same_foot_x_position = None
        # Constrain signed lateral width in the heading frame. A long forward
        # step cannot disguise narrow/crossed legs as it could with XY distance.
        # The band surrounds the shared nominal wheel-center width of 0.34 m.
        self.rewards.pen_feet_distance = RewTerm(
            func=mdp.foot_lateral_width_huber,
            weight=-1.0,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=WHEEL_BODY_NAMES, preserve_order=True
                ),
                "min_width": 0.30,
                "max_width": 0.38,
                "error_scale": 0.05,
            },
        )
        # Foot locomotion should follow the local support slope instead of
        # being pulled toward a world-horizontal base orientation.
        self.rewards.pen_flat_orientation_l2 = None
        self.rewards.pen_terrain_orientation = RewTerm(
            func=mdp.terrain_aligned_orientation_l2,
            weight=-10.0,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        )
        # Penalize sustained actual wheel rotation in every contact state.
        # Foot uses a fixed zero target; PI still supplies braking effort.
        self.rewards.pen_joint_vel_wheel_l2 = None
        self.rewards.pen_wheel_actual_speed = RewTerm(
            func=mdp.wheel_actual_speed_huber,
            weight=0.0,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=WHEEL_JOINTS, preserve_order=True
                ),
                "speed_scale": 1.0,
            },
        )
        self.rewards.pen_wheel_target_zero = None
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.pen_knee_contact_force = RewTerm(
            func=mdp.undesired_contact_force_l2,
            weight=-2.0,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=["knee_L_Link", "knee_R_Link"]
                ),
                "force_threshold": 10.0,
                "force_scale": 100.0,
                "max_normalized_excess": 3.0,
            },
        )
        self.terminations.sustained_knee_contact = DoneTerm(
            func=mdp.SustainedIllegalContact,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=["knee_L_Link", "knee_R_Link"]
                ),
                "force_threshold": 20.0,
                "duration_s": 0.08,
            },
        )
        self.rewards.gait_contact_schedule = RewTerm(
            func=mdp.GaitReward,
            weight=1.0,
            params={
                "tracking_contacts_shaped_force": -6.0,
                "tracking_contacts_shaped_vel": -2.0,
                "force_kernel": "huber",
                "force_reference": 100.0,
                "gait_force_sigma": 25.0,
                "gait_vel_sigma": 0.25,
                "kappa_gait_probs": 0.05,
                "command_name": "gait_command",
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=WHEEL_BODY_NAMES, preserve_order=True
                ),
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=WHEEL_BODY_NAMES, preserve_order=True
                ),
            },
        )
        self.rewards.pen_feet_regulation = RewTerm(
            func=mdp.wheel_terrain_feet_regulation,
            weight=-0.1,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=WHEEL_BODY_NAMES,
                    preserve_order=True,
                ),
                "terrain_sensor_names": WHEEL_GROUND_SCAN_SENSORS,
                "wheel_radius": WHEEL_RADIUS,
                "height_scale": 0.05,
            },
        )
        self.rewards.pen_swing_clearance = None
        self.rewards.rew_swing_clearance = RewTerm(
            func=mdp.TerrainCycleSwingClearanceExp,
            weight=2.0,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=WHEEL_BODY_NAMES, preserve_order=True
                ),
                "contact_sensor_names": WHEEL_GROUND_CONTACT_SENSORS,
                "terrain_sensor_names": WHEEL_GROUND_SCAN_SENSORS,
                "height_sensor_cfg": SceneEntityCfg("height_scanner"),
                "gait_command_name": "gait_command",
                "wheel_radius": WHEEL_RADIUS,
                "min_peak_height": 0.05,
                "max_peak_height": 0.10,
                "switch_center": 0.04,
                "switch_band": 0.01,
                "std": 0.025,
            },
        )
        self.rewards.pen_all_wheels_air_time = None
        self.rewards.pen_planned_support_contact = RewTerm(
            func=mdp.planned_support_contact_loss, weight=-1.0,
            params={
                "command_name": "gait_command",
                "contact_sensor_names": WHEEL_GROUND_CONTACT_SENSORS,
                "kappa_gait_probs": 0.05, "force_off": 5.0, "force_on": 10.0,
            },
        )
        self.rewards.pen_swing_min_clearance = RewTerm(
            func=mdp.swing_min_clearance_penalty, weight=-1.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=WHEEL_BODY_NAMES, preserve_order=True),
                "terrain_sensor_names": WHEEL_GROUND_SCAN_SENSORS,
                "gait_command_name": "gait_command", "wheel_radius": WHEEL_RADIUS,
                "min_clearance": 0.02, "active_start": 0.20, "full_start": 0.35,
            },
        )
        self.rewards.pen_missed_swing = RewTerm(
            func=mdp.MissedSwingPenalty, weight=-0.2,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=WHEEL_BODY_NAMES, preserve_order=True),
                "contact_sensor_names": WHEEL_GROUND_CONTACT_SENSORS,
                "terrain_sensor_names": WHEEL_GROUND_SCAN_SENSORS,
                "gait_command_name": "gait_command", "wheel_radius": WHEEL_RADIUS,
                "tracker_options": {"min_clearance": 0.01, "min_air_time": 0.06,
                                    "force_off": 5.0, "force_on": 10.0},
            },
        )
        self.rewards.pen_foothold_region = RewTerm(
            func=mdp.FootholdRegionPenalty, weight=-0.2,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=WHEEL_BODY_NAMES, preserve_order=True),
                "contact_sensor_names": WHEEL_GROUND_CONTACT_SENSORS,
                "terrain_sensor_names": WHEEL_GROUND_SCAN_SENSORS,
                "height_sensor_cfg": SceneEntityCfg("height_scanner"),
                "command_name": "base_velocity", "gait_command_name": "gait_command",
                "tracker_options": {
                    "axes": (0.04, 0.03), "nominal_width": 0.34,
                    "balance_time": 0.10, "max_balance": 0.04,
                    "max_forward": 0.25, "min_lateral": 0.12, "max_lateral": 0.28,
                    "max_leg_reach": 0.90, "wheel_radius": WHEEL_RADIUS,
                    "max_slope_deg": 20.0, "max_relief": 0.015, "scan_distance": 0.085,
                    "min_air_time": 0.06, "contact_time": 0.04, "min_clearance": 0.01,
                    "force_off": 5.0, "force_on": 10.0, "max_event_cost": 5.0,
                },
            },
        )
        self.rewards.foot_landing_vel = RewTerm(
            func=mdp.wheel_terrain_landing_velocity_l2,
            weight=-0.5,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=WHEEL_BODY_NAMES,
                    preserve_order=True,
                ),
                "contact_sensor_names": WHEEL_GROUND_CONTACT_SENSORS,
                "terrain_sensor_names": WHEEL_GROUND_SCAN_SENSORS,
                "wheel_radius": WHEEL_RADIUS,
                "about_landing_threshold": 0.02,
                "allowed_downward_speed": 0.2,
                "contact_force_threshold": 0.1,
            },
        )
        self.rewards.pen_wheel_stance_slip = RewTerm(
            func=mdp.wheel_stance_slip_l2,
            weight=-0.75,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=WHEEL_LINKS),
                "asset_cfg": SceneEntityCfg("robot", body_names=WHEEL_LINKS),
                "terrain_sensor_cfg": SceneEntityCfg("height_scanner"),
                "wheel_radius": WHEEL_RADIUS,
                "force_off": 5.0,
                "force_on": 10.0,
            },
        )


@configclass
class WFFootAllTerrainEnvCfg_PLAY(WFFootAllTerrainEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.eye = (3.0, 3.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.3)
        self.scene.height_scanner.debug_vis = True
        self.observations.policy.enable_corruption = False
        self.actions.joint_vel.kp_scale_range = (1.0, 1.0)
        self.actions.joint_vel.ki_ratio_range = (0.25, 0.25)
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.scene.terrain.terrain_generator = FOOT_ALL_TERRAINS_PLAY_CFG
        self.scene.terrain.max_init_terrain_level = None


@configclass
class WFDualModeEnvCfg_PLAY(WFWheelModeEnvCfg_PLAY):
    """Union-schema play environment used to run both expert checkpoints."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = FOOT_ALL_TERRAINS_PLAY_CFG
        # The FSM zeroes Foot wheel targets and blends toward Wheel targets
        # during transitions; this union environment keeps the wheel PI enabled.
        self.actions.joint_vel.scale = 1.0
