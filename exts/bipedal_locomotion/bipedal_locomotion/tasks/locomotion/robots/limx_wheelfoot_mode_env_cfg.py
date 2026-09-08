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
        _configure_common_mode(self, play=False)
        self.rewards.stand_still.weight = -7.0

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
            weight=-0.02,
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
            weight=-0.02,
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
    """Foot expert: wheel-speed targets are hard-masked to zero on all terrain."""

    def __post_init__(self):
        super().__post_init__()
        _configure_common_mode(self, play=False)

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = FOOT_ALL_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = 0

        # Hard action-layer constraint: processed wheel velocity target is always zero.
        self.actions.joint_vel.scale = 0.0
        self.actions.joint_vel.offset = 0.0
        self.actions.joint_vel.use_default_offset = True

        self.commands.body_height.ranges.height = BODY_HEIGHT_RANGE
        self.commands.body_height.max_rate = 0.06
        self.commands.base_velocity.rel_standing_envs = 0.15
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (-0.8, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.4, 0.4)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)
        self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)
        # Preserve the shared four-dimensional checkpoint schema, but remove
        # the unused target-height signal now that swing clearance is learned
        # from terrain observations instead of explicitly commanded.
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
            weight=1.0,
            params={**height_params, "std": 0.05},
        )

        self.rewards.rew_same_foot_x_position = None
        # Match PF: prevent the support ends from becoming too close without
        # constraining their fore-aft separation during a normal step.
        self.rewards.pen_feet_distance.params["min_feet_distance"] = 0.115
        self.rewards.pen_feet_distance.params["max_feet_distance"] = 1.0
        # Foot locomotion should follow the local support slope instead of
        # being pulled toward a world-horizontal base orientation.
        self.rewards.pen_flat_orientation_l2 = None
        self.rewards.pen_terrain_orientation = RewTerm(
            func=mdp.terrain_aligned_orientation_l2,
            weight=-10.0,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        )
        self.rewards.pen_joint_vel_wheel_l2.weight = -0.10
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
                "tracking_contacts_shaped_force": -2.0,
                "tracking_contacts_shaped_vel": -2.0,
                "gait_force_sigma": 25.0,
                "gait_vel_sigma": 0.25,
                "kappa_gait_probs": 0.05,
                "command_name": "gait_command",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=WHEEL_LINKS),
                "asset_cfg": SceneEntityCfg("robot", body_names=WHEEL_LINKS),
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
                "height_scale": 0.65,
            },
        )
        self.rewards.pen_all_wheels_air_time = RewTerm(
            func=mdp.all_wheels_air_time_l2,
            weight=-4.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=WHEEL_LINKS),
                "max_air_time": 1.0,
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
                "about_landing_threshold": 0.08,
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
        # The external FSM masks wheel actions in FOOT mode; keep wheel control active here.
        self.actions.joint_vel.scale = 1.0
