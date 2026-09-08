"""IsaacLab smoke test for WF reward configuration and runtime tensors.

Run through the IsaacLab environment, for example:

    python tests/smoke_wf_reward_env.py --task Isaac-Limx-WF-Wheel-Mode-v0 --headless --device cpu
"""

from __future__ import annotations

import argparse
import math
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Smoke-test WF reward terms in a real IsaacLab environment.")
parser.add_argument("--task", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import bipedal_locomotion  # noqa: F401  # registers project Gym environments
from isaaclab_tasks.utils import parse_env_cfg


def main() -> None:
    is_wheel_task = "WF-Wheel-" in args.task
    is_foot_task = "WF-Foot-" in args.task
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env_cfg.commands.base_velocity.debug_vis = False
    env = gym.make(args.task, cfg=env_cfg)
    print("[SMOKE] Environment created.", flush=True)
    unwrapped = env.unwrapped

    env.reset()
    print("[SMOKE] Environment reset.", flush=True)
    action = torch.zeros_like(unwrapped.action_manager.action)
    acceleration_startup_values = {
        name: [] for name in ("pen_base_lin_acc_xy", "pen_base_yaw_acc")
        if name in unwrapped.reward_manager.active_terms
    }
    for _ in range(3):
        _, reward, _, _, _ = env.step(action)
        if not torch.isfinite(reward).all():
            raise AssertionError(f"Non-finite total reward: {reward}")
        if not torch.isfinite(unwrapped.reward_manager._step_reward).all():
            bad_terms = [
                name
                for index, name in enumerate(unwrapped.reward_manager.active_terms)
                if not torch.isfinite(unwrapped.reward_manager._step_reward[:, index]).all()
            ]
            raise AssertionError(f"Non-finite reward terms: {bad_terms}")
        for name, values in acceleration_startup_values.items():
            term_index = unwrapped.reward_manager.active_terms.index(name)
            values.append(unwrapped.reward_manager._step_reward[:, term_index].clone())

    for name, values in acceleration_startup_values.items():
        first_two_steps = torch.stack(values[:2], dim=0)
        if not torch.all(first_two_steps == 0.0):
            raise AssertionError(f"{name} must be zero for the first two steps after reset: {first_two_steps}")

    if "both_wheels_airborne" in unwrapped.termination_manager.active_terms:
        raise AssertionError("AIRBORNE must not be registered as a termination.")

    active_rewards = set(unwrapped.reward_manager.active_terms)
    if "pen_base_contact_termination" not in active_rewards:
        raise AssertionError("Missing base-contact termination penalty.")
    termination_reward_cfg = unwrapped.reward_manager.get_term_cfg("pen_base_contact_termination")
    if termination_reward_cfg.func.__name__ != "is_terminated":
        raise AssertionError(
            "Termination penalty must use the current-step non-timeout signal, got "
            f"{termination_reward_cfg.func.__name__}."
        )
    if "term_keys" in termination_reward_cfg.params:
        raise AssertionError("Termination penalty must not read persistent per-term termination history.")
    if termination_reward_cfg.weight != -500.0:
        raise AssertionError(f"Unexpected termination penalty weight: {termination_reward_cfg.weight}")
    expected_non_timeout_terms = {"base_contact"}
    if is_foot_task:
        expected_non_timeout_terms.add("sustained_knee_contact")
    unexpected_non_timeout_terms = {
        name
        for name in unwrapped.termination_manager.active_terms
        if not unwrapped.termination_manager.get_term_cfg(name).time_out
        and name not in expected_non_timeout_terms
    }
    if unexpected_non_timeout_terms:
        raise AssertionError(
            "is_terminated would also penalize unexpected non-timeout terms: "
            f"{sorted(unexpected_non_timeout_terms)}"
        )
    if is_foot_task:
        if "pen_knee_contact_force" not in active_rewards:
            raise AssertionError("Foot task is missing the force-sensitive knee-contact penalty.")
        knee_reward_cfg = unwrapped.reward_manager.get_term_cfg("pen_knee_contact_force")
        if knee_reward_cfg.func.__name__ != "undesired_contact_force_l2":
            raise AssertionError(f"Unexpected knee-contact reward function: {knee_reward_cfg.func.__name__}")
        if knee_reward_cfg.weight != -2.0:
            raise AssertionError(f"Unexpected knee-contact reward weight: {knee_reward_cfg.weight}")
        if (
            knee_reward_cfg.params["force_threshold"] != 10.0
            or knee_reward_cfg.params["force_scale"] != 100.0
            or knee_reward_cfg.params["max_normalized_excess"] != 3.0
        ):
            raise AssertionError("Unexpected force-sensitive knee-contact reward parameters.")
        knee_termination_cfg = unwrapped.termination_manager.get_term_cfg("sustained_knee_contact")
        if knee_termination_cfg.params["force_threshold"] != 20.0:
            raise AssertionError("Unexpected sustained knee-contact force threshold.")
        if knee_termination_cfg.params["duration_s"] != 0.08:
            raise AssertionError("Unexpected sustained knee-contact duration.")
        if knee_termination_cfg.func._required_steps != 4:
            raise AssertionError(
                f"Unexpected sustained knee-contact step count: {knee_termination_cfg.func._required_steps}"
            )

    height_cfg = unwrapped.reward_manager.get_term_cfg("pen_base_height")
    if "grounded" not in height_cfg.func.__name__:
        raise AssertionError(f"Height penalty is not support-gated: {height_cfg.func.__name__}")
    if "pen_flat_orientation_l2" in active_rewards:
        raise AssertionError("Mode experts must not retain the world-horizontal orientation penalty.")
    if "pen_terrain_orientation" not in active_rewards:
        raise AssertionError("Missing local-terrain orientation penalty.")
    terrain_orientation_cfg = unwrapped.reward_manager.get_term_cfg("pen_terrain_orientation")
    if terrain_orientation_cfg.func.__name__ != "terrain_aligned_orientation_l2":
        raise AssertionError(
            f"Orientation penalty does not track local terrain: {terrain_orientation_cfg.func.__name__}"
        )
    expected_orientation_weight = -10.0
    if terrain_orientation_cfg.weight != expected_orientation_weight:
        raise AssertionError(f"Unexpected local-terrain orientation weight: {terrain_orientation_cfg.weight}")

    height_command_cfg = unwrapped.command_manager.get_term("body_height").cfg
    if height_command_cfg.ranges.height != (0.65, 0.85) or height_command_cfg.endpoint_fraction != 0.20:
        raise AssertionError(
            f"Unexpected height sampling: range={height_command_cfg.ranges.height}, "
            f"endpoint_fraction={height_command_cfg.endpoint_fraction}"
        )

    if is_wheel_task:
        stand_cfg = unwrapped.reward_manager.get_term_cfg("stand_still")
        if stand_cfg.func.__name__ != "stand_still_grounded":
            raise AssertionError(f"Stand-still penalty is not support-gated: {stand_cfg.func.__name__}")
        linear_tracking_cfg = unwrapped.reward_manager.get_term_cfg("rew_lin_vel_xy")
        yaw_tracking_cfg = unwrapped.reward_manager.get_term_cfg("rew_ang_vel_z")
        if linear_tracking_cfg.func.__name__ != "track_lin_vel_xy_exp_any_wheel_ground":
            raise AssertionError(
                f"Linear tracking does not use the any-wheel gate: {linear_tracking_cfg.func.__name__}"
            )
        if yaw_tracking_cfg.func.__name__ != "track_ang_vel_z_exp_any_wheel_ground":
            raise AssertionError(f"Yaw tracking does not use the any-wheel gate: {yaw_tracking_cfg.func.__name__}")

        velocity_term = unwrapped.command_manager.get_term("base_velocity")
        if velocity_term.__class__.__name__ != "WheelSupportVelocityCommand":
            raise AssertionError(f"Wheel support diagnostics are inactive: {velocity_term.__class__.__name__}")
        expected_diagnostics = {
            "support_force_L",
            "support_force_R",
            "support_geometry_L",
            "support_geometry_R",
            "support_combined_L",
            "support_combined_R",
            "support_all",
            "support_any",
            "both_support_rate",
            "single_support_rate",
            "no_support_rate",
            "lin_tracking_ungated",
            "lin_tracking_gated",
            "yaw_tracking_ungated",
            "yaw_tracking_gated",
            "moving_command_rate",
            "standing_command_rate",
            "straight_command_rate",
            "yaw_only_command_rate",
            "mixed_command_rate",
            "straight_xy_error",
            "yaw_only_wz_error",
            "yaw_only_wz_acc_rms",
            "mixed_xy_error",
            "mixed_wz_error",
        }
        if set(velocity_term._diagnostic_sums) != expected_diagnostics:
            raise AssertionError(
                "Unexpected Wheel diagnostic set: "
                f"{sorted(set(velocity_term._diagnostic_sums) ^ expected_diagnostics)}"
            )
        if not all(torch.isfinite(value).all() for value in velocity_term._diagnostic_sums.values()):
            raise AssertionError("Non-finite Wheel support diagnostics.")
        expected_mode_fractions = {
            "rel_standing_envs": 0.25,
            "rel_straight_envs": 0.30,
            "rel_yaw_only_envs": 0.10,
            "rel_mixed_envs": 0.35,
        }
        for name, expected_fraction in expected_mode_fractions.items():
            actual_fraction = getattr(velocity_term.cfg, name)
            if actual_fraction != expected_fraction:
                raise AssertionError(f"Unexpected Wheel command fraction {name}: {actual_fraction}")
        mode_mask = torch.stack(
            (
                velocity_term.is_standing_env,
                velocity_term.is_straight_env,
                velocity_term.is_yaw_only_env,
                velocity_term.is_mixed_env,
            ),
            dim=1,
        )
        if not torch.all(torch.sum(mode_mask, dim=1) == 1):
            raise AssertionError(f"Wheel command modes are not mutually exclusive: {mode_mask}")
        if velocity_term.is_standing_env.any() and not torch.all(
            velocity_term.command[velocity_term.is_standing_env] == 0.0
        ):
            raise AssertionError("Standing mode contains a nonzero velocity command.")
        if velocity_term.is_straight_env.any() and not torch.all(
            velocity_term.command[velocity_term.is_straight_env, 1:] == 0.0
        ):
            raise AssertionError("Straight mode contains lateral or yaw command.")
        if velocity_term.is_yaw_only_env.any() and not torch.all(
            velocity_term.command[velocity_term.is_yaw_only_env, :2] == 0.0
        ):
            raise AssertionError("Yaw-only mode contains a planar velocity command.")
        terrain_generator_cfg = unwrapped.cfg.scene.terrain.terrain_generator
        if terrain_generator_cfg.num_rows != 12:
            raise AssertionError(f"Unexpected Wheel terrain level count: {terrain_generator_cfg.num_rows}")
        expected_terrain_proportions = {
            "flat": 0.20,
            "slope_up": 0.25,
            "slope_down": 0.15,
            "waves": 0.10,
            "random_rough": 0.10,
            "stairs_up": 0.10,
            "stairs_down": 0.10,
        }
        actual_terrain_proportions = {
            name: cfg.proportion for name, cfg in terrain_generator_cfg.sub_terrains.items()
        }
        if actual_terrain_proportions != expected_terrain_proportions:
            raise AssertionError(f"Unexpected Wheel terrain proportions: {actual_terrain_proportions}")
        expected_terrain_classes = {
            "slope_up": "HfInvertedPyramidSlopedTerrainCfg",
            "slope_down": "HfPyramidSlopedTerrainCfg",
            "stairs_up": "MeshInvertedPyramidStairsTerrainCfg",
            "stairs_down": "MeshPyramidStairsTerrainCfg",
        }
        for name, expected_class in expected_terrain_classes.items():
            actual_class = terrain_generator_cfg.sub_terrains[name].__class__.__name__
            if actual_class != expected_class:
                raise AssertionError(f"Unexpected {name} terrain direction: {actual_class}")
        if terrain_generator_cfg.sub_terrains["stairs_up"].step_height_range != (0.005, 0.04):
            raise AssertionError("Wheel uphill stairs must retain the low 0.5--4 cm range.")
        curriculum_cfg = unwrapped.cfg.curriculum.terrain_levels
        if curriculum_cfg.func.__name__ != "wheel_terrain_levels_vel_tracking":
            raise AssertionError(f"Unexpected Wheel terrain curriculum: {curriculum_cfg.func.__name__}")
        expected_curriculum_params = {
            "command_name": "base_velocity",
            "asset_name": "robot",
            "distance_fraction_up": 0.5,
            "moving_rate_threshold": 0.25,
            "tracking_up": 0.55,
            "tracking_down": 0.25,
            "support_up": 0.75,
        }
        if curriculum_cfg.params != expected_curriculum_params:
            raise AssertionError(f"Unexpected Wheel curriculum thresholds: {curriculum_cfg.params}")
        curriculum_state = curriculum_cfg.func(
            unwrapped,
            torch.arange(unwrapped.num_envs, device=unwrapped.device),
            **curriculum_cfg.params,
        )
        if not torch.isfinite(curriculum_state):
            raise AssertionError(f"Non-finite Wheel terrain curriculum state: {curriculum_state}")

        reported_diagnostics = velocity_term.reset()
        missing_reported_diagnostics = expected_diagnostics - set(reported_diagnostics)
        if missing_reported_diagnostics:
            raise AssertionError(
                f"Wheel diagnostics are not emitted on command reset: {sorted(missing_reported_diagnostics)}"
            )
        if not all(math.isfinite(reported_diagnostics[name]) for name in expected_diagnostics):
            raise AssertionError("Non-finite reported Wheel support diagnostics.")
        expected_rewards = {
            "rew_base_height_exp",
            "pen_wheel_contact",
            "pen_wheel_horizontal_neutral",
            "pen_rolling_error",
            "pen_wheel_stance_slip",
            "pen_wheel_target_symmetry",
            "pen_zero_command_wheel_target",
            "pen_base_lin_acc_xy",
            "pen_base_yaw_acc",
        }
        missing_rewards = expected_rewards - active_rewards
        if missing_rewards:
            raise AssertionError(f"Missing new Wheel rewards: {sorted(missing_rewards)}")

        zero_target_cfg = unwrapped.reward_manager.get_term_cfg("pen_zero_command_wheel_target")
        if zero_target_cfg.weight != -0.10 or zero_target_cfg.params["target_tolerance"] != 0.15:
            raise AssertionError(
                "Unexpected zero-command wheel-target constraint: "
                f"weight={zero_target_cfg.weight}, tolerance={zero_target_cfg.params['target_tolerance']}"
            )
        if stand_cfg.weight != -7.0:
            raise AssertionError(f"Unexpected Wheel stand-still weight: {stand_cfg.weight}")
        rolling_cfg = unwrapped.reward_manager.get_term_cfg("pen_rolling_error")
        if rolling_cfg.func.__name__ != "wheel_rolling_velocity_error":
            raise AssertionError(f"Unexpected rolling-error function: {rolling_cfg.func.__name__}")
        if rolling_cfg.weight != -2.0:
            raise AssertionError(f"Unexpected rolling-error weight: {rolling_cfg.weight}")
        if not rolling_cfg.params["asset_cfg"].preserve_order:
            raise AssertionError("Rolling-error wheel body/joint matching must preserve left/right order.")
        wheel_height_cfg = unwrapped.reward_manager.get_term_cfg("pen_base_height")
        if wheel_height_cfg.weight != -60.0:
            raise AssertionError(f"Unexpected Wheel height penalty weight: {wheel_height_cfg.weight}")
        height_exp_cfg = unwrapped.reward_manager.get_term_cfg("rew_base_height_exp")
        if height_exp_cfg.func.__name__ != "body_height_command_plane_grounded_exp":
            raise AssertionError(f"Unexpected Wheel exponential height reward: {height_exp_cfg.func.__name__}")
        if height_exp_cfg.weight != 1.0 or height_exp_cfg.params["std"] != 0.05:
            raise AssertionError(
                "Unexpected Wheel exponential height reward configuration: "
                f"weight={height_exp_cfg.weight}, std={height_exp_cfg.params['std']}"
            )
        if linear_tracking_cfg.weight != 3.5 or abs(linear_tracking_cfg.params["std"] ** 2 - 0.12) > 1.0e-9:
            raise AssertionError(
                "Unexpected Wheel linear-velocity tracking configuration: "
                f"weight={linear_tracking_cfg.weight}, std={linear_tracking_cfg.params['std']}"
            )
        if yaw_tracking_cfg.weight != 1.5 or abs(yaw_tracking_cfg.params["std"] ** 2 - 0.12) > 1.0e-9:
            raise AssertionError(
                "Unexpected Wheel yaw-rate tracking configuration: "
                f"weight={yaw_tracking_cfg.weight}, std={yaw_tracking_cfg.params['std']}"
            )
        acceleration_terms = {
            "pen_base_lin_acc_xy": ("xy", 3.0, -0.02, "bounded"),
            "pen_base_yaw_acc": ("yaw", 4.0, -0.02, "charbonnier"),
        }
        for name, (component, scale, weight, kernel) in acceleration_terms.items():
            acceleration_cfg = unwrapped.reward_manager.get_term_cfg(name)
            if acceleration_cfg.func.__name__ != "BaseVelocityAccelerationPenalty":
                raise AssertionError(f"Unexpected acceleration reward function for {name}: {acceleration_cfg.func}")
            if acceleration_cfg.weight != weight:
                raise AssertionError(f"Unexpected acceleration reward weight for {name}: {acceleration_cfg.weight}")
            if acceleration_cfg.params["component"] != component:
                raise AssertionError(f"Unexpected acceleration component for {name}: {acceleration_cfg.params['component']}")
            if acceleration_cfg.params["acceleration_scale"] != scale:
                raise AssertionError(f"Unexpected acceleration scale for {name}: {acceleration_cfg.params['acceleration_scale']}")
            if acceleration_cfg.params["tracking_std"] != 0.30:
                raise AssertionError(f"Unexpected acceleration tracking std for {name}.")
            if acceleration_cfg.params["kernel"] != kernel:
                raise AssertionError(f"Unexpected acceleration kernel for {name}: {acceleration_cfg.params['kernel']}")
            if not torch.all(acceleration_cfg.func._steps_since_reset == 3):
                raise AssertionError(f"Unexpected acceleration history length for {name}.")
        if unwrapped.reward_manager.get_term_cfg("pen_action_rate").weight != -0.15:
            raise AssertionError("Unexpected Wheel action-rate penalty weight.")
        if unwrapped.reward_manager.get_term_cfg("pen_action_smoothness").weight != -0.08:
            raise AssertionError("Unexpected Wheel action-smoothness penalty weight.")
        vertical_cfg = unwrapped.reward_manager.get_term_cfg("pen_lin_vel_z")
        if vertical_cfg.func.__name__ != "lin_vel_z_height_command_gated_l2":
            raise AssertionError(f"Vertical velocity penalty is not height-transition-gated: {vertical_cfg.func.__name__}")
        expected_vertical_params = {
            "min_scale": 0.2,
            "full_penalty_gap": 0.01,
            "reduced_penalty_gap": 0.04,
        }
        for name, expected in expected_vertical_params.items():
            if vertical_cfg.params[name] != expected:
                raise AssertionError(f"Unexpected vertical gate {name}: {vertical_cfg.params[name]}")
        removed_rewards = {
            "rew_same_foot_x_position",
            "pen_feet_distance",
            "pen_wheel_vertical_velocity",
            "pen_wheel_terrain_geometry",
            "pen_wheel_configuration",
            "pen_wheel_relative_velocity",
            "pen_wheel_air_time",
            "pen_zero_command_yaw_rate",
        }
        unexpected_rewards = removed_rewards & active_rewards
        if unexpected_rewards:
            raise AssertionError(f"Removed Wheel rewards are still active: {sorted(unexpected_rewards)}")

        for sensor_name in ("wheel_L_ground_contact", "wheel_R_ground_contact"):
            sensor = unwrapped.scene.sensors[sensor_name]
            if sensor.data.force_matrix_w is None:
                raise AssertionError(f"{sensor_name} has no ground-filtered force matrix.")
        for sensor_name in ("wheel_L_ground_scan", "wheel_R_ground_scan"):
            hits = unwrapped.scene.sensors[sensor_name].data.ray_hits_w
            if not torch.isfinite(hits).any():
                raise AssertionError(f"{sensor_name} has no finite terrain ray hits.")
    else:
        if "stand_still" in active_rewards:
            raise AssertionError("Foot must not retain the WF stand-still penalty.")
        expected_foot_weights = {
            "rew_leg_symmetry": 0.5,
            "pen_joint_torque": -8.0e-5,
            "pen_joint_accel": -2.5e-7,
            "pen_action_rate": -0.03,
            "pen_action_smoothness": -0.04,
            "pen_joint_power_l1": -5.0e-4,
            "pen_vel_non_wheel_l2": -1.0e-3,
            "pen_joint_vel_wheel_l2": -0.10,
            "pen_terrain_orientation": -10.0,
            "pen_base_height": -30.0,
            "rew_base_height_exp": 1.0,
            "gait_contact_schedule": 1.0,
            "pen_feet_regulation": -0.1,
            "foot_landing_vel": -0.5,
        }
        for name, expected_weight in expected_foot_weights.items():
            actual_weight = unwrapped.reward_manager.get_term_cfg(name).weight
            if actual_weight != expected_weight:
                raise AssertionError(
                    f"Unexpected Foot reward weight for {name}: {actual_weight}, expected {expected_weight}"
                )
        if height_cfg.func.__name__ != "body_height_command_plane_grounded_l2":
            raise AssertionError(f"Foot height penalty is not local-plane based: {height_cfg.func.__name__}")
        height_exp_cfg = unwrapped.reward_manager.get_term_cfg("rew_base_height_exp")
        if height_exp_cfg.func.__name__ != "body_height_command_plane_grounded_exp":
            raise AssertionError(f"Unexpected Foot exponential height reward: {height_exp_cfg.func.__name__}")
        if height_exp_cfg.params["std"] != 0.05:
            raise AssertionError(f"Unexpected Foot exponential height std: {height_exp_cfg.params['std']}")
        removed_foot_rewards = {"pen_swing_height", "pen_wheel_landing_impact"} & active_rewards
        if removed_foot_rewards:
            raise AssertionError(f"Foot retained replaced swing/landing rewards: {sorted(removed_foot_rewards)}")
        regulation_cfg = unwrapped.reward_manager.get_term_cfg("pen_feet_regulation")
        if regulation_cfg.func.__name__ != "wheel_terrain_feet_regulation":
            raise AssertionError(f"Unexpected Foot regulation function: {regulation_cfg.func.__name__}")
        if regulation_cfg.params["height_scale"] != 0.65:
            raise AssertionError(f"Unexpected Foot regulation height scale: {regulation_cfg.params['height_scale']}")
        landing_cfg = unwrapped.reward_manager.get_term_cfg("foot_landing_vel")
        if landing_cfg.func.__name__ != "wheel_terrain_landing_velocity_l2":
            raise AssertionError(f"Unexpected Foot landing function: {landing_cfg.func.__name__}")
        if landing_cfg.params["about_landing_threshold"] != 0.08:
            raise AssertionError(
                f"Unexpected Foot pre-landing threshold: {landing_cfg.params['about_landing_threshold']}"
            )
        gait_command_cfg = unwrapped.command_manager.get_term("gait_command").cfg
        if gait_command_cfg.ranges.swing_height != (0.0, 0.0):
            raise AssertionError(
                f"Foot swing-height command must be neutralized: {gait_command_cfg.ranges.swing_height}"
            )

    print(unwrapped.reward_manager, flush=True)
    print("[SMOKE] Last-step reward terms:", flush=True)
    for name, value in unwrapped.reward_manager.get_active_iterable_terms(0):
        print(f"  {name}: {value[0]:.6f}", flush=True)
    print(f"[SMOKE] {args.task}: PASS", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
