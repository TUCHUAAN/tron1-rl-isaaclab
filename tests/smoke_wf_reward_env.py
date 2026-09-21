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
    if is_wheel_task or is_foot_task:
        groups = unwrapped.observation_manager.compute(update_history=False)
        assert groups["policy"].shape[-1] == 161, groups["policy"].shape
        assert groups["obsHistory"].flatten(start_dim=1).shape[-1] == 340
        assert unwrapped.observation_manager.active_terms["policy"][-1] == "zero_command_hold"
        hold = unwrapped.reward_manager.get_term_cfg("pen_zero_command_hold").func
        age = hold.tracker.age.clone()
        groups_again = unwrapped.observation_manager.compute(update_history=False)
        torch.testing.assert_close(hold.tracker.age, age)
        torch.testing.assert_close(groups_again["policy"][:, -6:], groups_again["critic"][:, -6:])
        assert torch.isfinite(groups_again["policy"][:, -6:]).all()
        assert (groups_again["policy"][:, -3:] == 0).all()

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
        velocity_term = unwrapped.command_manager.get_term("base_velocity")
        if velocity_term.__class__.__name__ != "FootVelocityCommand":
            raise AssertionError("Foot must use its five-mode velocity sampler.")
        fractions = tuple(getattr(velocity_term.cfg, name) for name in (
            "rel_standing_envs", "rel_straight_envs", "rel_lateral_envs", "rel_yaw_only_envs", "rel_mixed_envs"
        ))
        if fractions != (0.15, 0.25, 0.10, 0.20, 0.30):
            raise AssertionError(f"Unexpected Foot mode fractions: {fractions}")
        gait_term = unwrapped.command_manager.get_term("gait_command")
        if not gait_term.cfg.continuous_phase:
            raise AssertionError("Foot gait must use continuous phase.")
        phase_before = gait_term.phase.clone()
        gait_term._resample_command(torch.arange(unwrapped.num_envs, device=unwrapped.device))
        torch.testing.assert_close(gait_term.phase, phase_before)
        diagnostics = velocity_term.diagnostic_means()
        if not all(math.isfinite(value) for value in diagnostics.values()):
            raise AssertionError(f"Non-finite Foot diagnostics: {diagnostics}")
        if sum(diagnostics[f"{mode}/samples"] for mode in velocity_term.MODE_NAMES) <= 0:
            raise AssertionError("Foot diagnostics did not record any executed steps.")
        wheel_action = unwrapped.action_manager.get_term("joint_vel")
        if wheel_action.__class__.__name__ != "WheelVelocityPIAction":
            raise AssertionError(
                f"Foot task is not using PI wheel-speed control: {wheel_action.__class__.__name__}"
            )
        expected_pi_cfg = {
            "kp": 2.0,
            "kp_scale_range": (0.25, 2.0),
            "ki_ratio_range": (0.0, 0.25),
            "effort_limit": 80.0,
        }
        for name, expected in expected_pi_cfg.items():
            actual = getattr(wheel_action.cfg, name)
            if actual != expected:
                raise AssertionError(f"Unexpected Foot PI setting {name}: {actual}, expected {expected}")
        if wheel_action.cfg.clip != {"wheel_.*": (0.0, 0.0)} or not wheel_action.cfg.fixed_zero_target:
            raise AssertionError(f"Unexpected Foot wheel-target clip: {wheel_action.cfg.clip}")
        wheel_action.process_actions(
            torch.tensor([[2.0, -2.0]], device=unwrapped.device, dtype=wheel_action.raw_actions.dtype)
        )
        torch.testing.assert_close(
            wheel_action.processed_actions,
            torch.tensor([[0.0, 0.0]], device=unwrapped.device, dtype=wheel_action.raw_actions.dtype),
        )
        wheel_action.process_actions(torch.zeros_like(wheel_action.raw_actions))
        if not torch.all((wheel_action.kp >= 0.5) & (wheel_action.kp <= 4.0)):
            raise AssertionError(f"Foot Kp samples are outside [0.5, 4.0]: {wheel_action.kp}")
        ki_ratio = wheel_action.ki / wheel_action.kp
        if not torch.all((ki_ratio >= 0.0) & (ki_ratio <= 0.25)):
            raise AssertionError(f"Foot Ki/Kp samples are outside [0, 0.25]: {ki_ratio}")
        wheel_actuator = unwrapped.scene["robot"].actuators["wheels"]
        if not torch.all(wheel_actuator.stiffness == 0.0) or not torch.all(wheel_actuator.damping == 0.0):
            raise AssertionError(
                "Foot implicit wheel gains must be zero under explicit PI control: "
                f"stiffness={wheel_actuator.stiffness}, damping={wheel_actuator.damping}"
            )
        for event_name in ("robot_joint_stiffness_and_damping", "randomize_actuator_gains"):
            event_cfg = getattr(unwrapped.cfg.events, event_name)
            randomized_joints = set(event_cfg.params["asset_cfg"].joint_names)
            if randomized_joints & {"wheel_L_Joint", "wheel_R_Joint"}:
                raise AssertionError(f"{event_name} still randomizes implicit Foot wheel gains.")

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
        wheel_action = unwrapped.action_manager.get_term("joint_vel")
        if wheel_action.__class__.__name__ != "WheelVelocityPIAction":
            raise AssertionError(f"Wheel task is not using PI velocity control: {wheel_action.__class__.__name__}")
        expected_pi_cfg = {
            "kp": 2.0,
            "kp_scale_range": (0.25, 2.0),
            "ki_ratio_range": (0.0, 0.25),
            "effort_limit": 80.0,
        }
        for name, expected in expected_pi_cfg.items():
            actual = getattr(wheel_action.cfg, name)
            if actual != expected:
                raise AssertionError(f"Unexpected Wheel PI setting {name}: {actual}, expected {expected}")
        if not torch.all((wheel_action.kp >= 0.5) & (wheel_action.kp <= 4.0)):
            raise AssertionError(f"Wheel Kp samples are outside [0.5, 4.0]: {wheel_action.kp}")
        ki_ratio = wheel_action.ki / wheel_action.kp
        if not torch.all((ki_ratio >= 0.0) & (ki_ratio <= 0.25)):
            raise AssertionError(f"Wheel Ki/Kp samples are outside [0, 0.25]: {ki_ratio}")
        if not torch.all(wheel_action.kp[:, 0] == wheel_action.kp[:, 1]):
            raise AssertionError(f"Left/right Wheel Kp samples must match: {wheel_action.kp}")
        if not torch.all(wheel_action.ki[:, 0] == wheel_action.ki[:, 1]):
            raise AssertionError(f"Left/right Wheel Ki samples must match: {wheel_action.ki}")
        if not torch.isfinite(wheel_action.integral_error).all():
            raise AssertionError("Wheel PI integral contains non-finite values.")
        wheel_actuator = unwrapped.scene["robot"].actuators["wheels"]
        if not torch.all(wheel_actuator.stiffness == 0.0) or not torch.all(wheel_actuator.damping == 0.0):
            raise AssertionError(
                "Wheel implicit gains must be zero when explicit PI control is active: "
                f"stiffness={wheel_actuator.stiffness}, damping={wheel_actuator.damping}"
            )
        for event_name in ("robot_joint_stiffness_and_damping", "randomize_actuator_gains"):
            event_cfg = getattr(unwrapped.cfg.events, event_name)
            randomized_joints = set(event_cfg.params["asset_cfg"].joint_names)
            if randomized_joints & {"wheel_L_Joint", "wheel_R_Joint"}:
                raise AssertionError(f"{event_name} still randomizes implicit Wheel gains.")

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
            "pen_base_lin_acc_xy": ("xy", 3.0, 0.0, "bounded"),
            "pen_base_yaw_acc": ("yaw", 4.0, -0.025, "charbonnier"),
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
            expected_steps = 0 if weight == 0.0 else 3
            if not torch.all(acceleration_cfg.func._steps_since_reset == expected_steps):
                raise AssertionError(
                    f"Unexpected acceleration history length for {name}: "
                    f"{acceleration_cfg.func._steps_since_reset}, expected {expected_steps}."
                )
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
            "rew_lin_vel_xy": 5.5,
            "rew_ang_vel_z": 3.0,
            "pen_yaw_tracking_error": -1.0,
            "pen_lin_vel_xy_tracking_error": -0.5,
            "pen_base_lin_acc_xy": 0.0,
            "pen_base_yaw_acc": 0.0,
            **{f"pen_cycle_mean_{c}": (-1.0 if c in ("vx", "vy", "yaw") else -0.1)
               for c in ("vx", "vy", "yaw", "height", "roll", "pitch")},
            "rew_leg_symmetry": 0.5,
            "pen_joint_torque": -8.0e-5,
            "pen_joint_accel": -2.5e-7,
            "pen_action_rate": -0.03,
            "pen_action_smoothness": -0.04,
            "pen_joint_power_l1": -5.0e-4,
            "pen_vel_non_wheel_l2": -1.0e-3,
            "pen_wheel_actual_speed": 0.0,
            "pen_terrain_orientation": -10.0,
            "pen_base_height": -30.0,
            "rew_base_height_exp": 1.5,
            "gait_contact_schedule": 1.0,
            "pen_feet_distance": -1.0,
            "rew_swing_clearance": 2.0,
            "pen_planned_support_contact": -1.0,
            "pen_foothold_region": -0.2,
            "pen_missed_swing": -0.2,
            "pen_swing_min_clearance": -1.0,
            "pen_zero_command_hold": -0.5,
            "pen_feet_regulation": -0.1,
            "foot_landing_vel": -0.5,
        }
        for name, expected_weight in expected_foot_weights.items():
            actual_weight = getattr(unwrapped.cfg.rewards, name).weight
            if actual_weight != expected_weight:
                raise AssertionError(
                    f"Unexpected Foot reward weight for {name}: {actual_weight}, expected {expected_weight}"
                )
        for component, scale in (("vx", .2), ("vy", .2), ("yaw", .3),
                                 ("height", .02), ("roll", .05235987756), ("pitch", .05235987756)):
            name = f"pen_cycle_mean_{component}"
            term_cfg = unwrapped.reward_manager.get_term_cfg(name)
            if term_cfg.func.__class__.__name__ != "GaitCycleMeanTrackingPenalty":
                raise AssertionError(f"Unexpected cycle mean term: {name}")
            if term_cfg.params["component"] != component or term_cfg.params["error_scale"] != scale:
                raise AssertionError(f"Unexpected cycle mean parameters: {name}")
            if term_cfg.func.window.ready.any():
                raise AssertionError("Cycle tracking must not score an incomplete startup window.")
        if {"pen_zero_vy_cycle_drift", "pen_zero_yaw_cycle_drift", "pen_wheel_target_zero"} & active_rewards:
            raise AssertionError("Foot retained replaced cycle/target penalties.")
        for name, std in (("rew_lin_vel_xy", math.sqrt(0.20)), ("rew_ang_vel_z", 0.5)):
            if unwrapped.reward_manager.get_term_cfg(name).params["std"] != std:
                raise AssertionError(f"Foot {name} must restore the h08 kernel width.")
        if "pen_swing_clearance" in active_rewards:
            raise AssertionError("Foot must not combine the replaced swing penalty with its positive reward.")
        for name, scale in (("pen_base_lin_acc_xy", 3.0), ("pen_base_yaw_acc", 4.0)):
            params = getattr(unwrapped.cfg.rewards, name).params
            if params["kernel"] != "charbonnier" or params["acceleration_scale"] != scale:
                raise AssertionError(f"Unexpected Foot acceleration kernel: {name}, {params}")
            if params["min_tracking_gate"] != 0.1 or params["tracking_std"] != 0.30:
                raise AssertionError(f"Unexpected Foot acceleration tracking gate: {name}")
        if unwrapped.cfg.rewards.pen_base_lin_acc_xy.params["linear_velocity_frame"] != "world":
            raise AssertionError("Foot XY acceleration must be evaluated in world coordinates.")
        if "pen_joint_vel_wheel_l2" in active_rewards:
            raise AssertionError("Foot retained the replaced global wheel-speed L2 penalty.")
        wheel_speed_cfg = unwrapped.cfg.rewards.pen_wheel_actual_speed
        if wheel_speed_cfg.func.__name__ != "wheel_actual_speed_huber":
            raise AssertionError(f"Unexpected Foot actual-wheel-speed function: {wheel_speed_cfg.func.__name__}")
        if wheel_speed_cfg.params["speed_scale"] != 1.0:
            raise AssertionError(f"Unexpected Foot wheel-speed Huber scale: {wheel_speed_cfg.params['speed_scale']}")
        unexpected_gate_params = {"contact_sensor_names", "force_off", "force_on"} & wheel_speed_cfg.params.keys()
        if unexpected_gate_params:
            raise AssertionError(f"Foot wheel-speed reward retained contact-gate params: {unexpected_gate_params}")
        if height_cfg.func.__name__ != "body_height_command_plane_grounded_l2":
            raise AssertionError(f"Foot height penalty is not local-plane based: {height_cfg.func.__name__}")
        height_exp_cfg = unwrapped.reward_manager.get_term_cfg("rew_base_height_exp")
        if height_exp_cfg.func.__name__ != "body_height_command_plane_grounded_exp":
            raise AssertionError(f"Unexpected Foot exponential height reward: {height_exp_cfg.func.__name__}")
        if height_exp_cfg.params["std"] != 0.05:
            raise AssertionError(f"Unexpected Foot exponential height std: {height_exp_cfg.params['std']}")
        removed_foot_rewards = {"pen_swing_height", "pen_wheel_landing_impact", "pen_swing_clearance"} & active_rewards
        if removed_foot_rewards:
            raise AssertionError(f"Foot retained replaced swing/landing rewards: {sorted(removed_foot_rewards)}")
        regulation_cfg = unwrapped.reward_manager.get_term_cfg("pen_feet_regulation")
        if regulation_cfg.func.__name__ != "wheel_terrain_feet_regulation":
            raise AssertionError(f"Unexpected Foot regulation function: {regulation_cfg.func.__name__}")
        if regulation_cfg.params["height_scale"] != 0.05:
            raise AssertionError(f"Unexpected Foot regulation height scale: {regulation_cfg.params['height_scale']}")
        width_cfg = unwrapped.reward_manager.get_term_cfg("pen_feet_distance")
        if width_cfg.func.__name__ != "foot_lateral_width_huber":
            raise AssertionError("Foot width must use signed lateral distance, not total XY distance.")
        for key, expected in (("min_width", 0.30), ("max_width", 0.38), ("error_scale", 0.05)):
            if width_cfg.params[key] != expected:
                raise AssertionError(f"Unexpected Foot width parameter {key}: {width_cfg.params[key]}")
        swing_cfg = unwrapped.reward_manager.get_term_cfg("rew_swing_clearance")
        if swing_cfg.func.__class__.__name__ != "TerrainCycleSwingClearanceExp":
            raise AssertionError("Foot swing clearance must adapt to the full terrain height scan.")
        for key, expected in (("min_peak_height", 0.05), ("max_peak_height", 0.10),
                              ("switch_center", 0.04), ("switch_band", 0.01),
                              ("std", 0.025)):
            if swing_cfg.params[key] != expected:
                raise AssertionError(f"Unexpected Foot swing parameter {key}: {swing_cfg.params[key]}")
        if {"velocity_command_name", "body_height_command_name", "lin_threshold", "ang_threshold"} & swing_cfg.params.keys():
            raise AssertionError("Foot swing clearance must remain active at zero command and independent of base height.")
        swing_term = swing_cfg.func
        if not torch.isfinite(swing_term.peak_height).all():
            raise AssertionError("Non-finite adaptive swing peak.")
        if not torch.all((swing_term.peak_height >= 0.05) & (swing_term.peak_height <= 0.10)):
            raise AssertionError("Adaptive swing peak is outside 5-10 cm.")
        gait_cfg = unwrapped.reward_manager.get_term_cfg("gait_contact_schedule")
        for key, expected in (("force_kernel", "huber"), ("force_reference", 100.0),
                              ("tracking_contacts_shaped_force", -6.0), ("tracking_contacts_shaped_vel", -2.0)):
            if gait_cfg.params[key] != expected:
                raise AssertionError(f"Unexpected Foot gait parameter {key}: {gait_cfg.params[key]}")
        for cfg in (width_cfg, swing_cfg, gait_cfg):
            names = unwrapped.scene["robot"].body_names
            resolved_names = [names[index] for index in cfg.params["asset_cfg"].body_ids]
            if resolved_names != ["wheel_L_Link", "wheel_R_Link"]:
                raise AssertionError(f"Foot geometry rewards require left/right order: {resolved_names}")
        contact_names = unwrapped.scene.sensors["contact_forces"].body_names
        gait_contact_names = [contact_names[index] for index in gait_cfg.params["sensor_cfg"].body_ids]
        if gait_contact_names != ["wheel_L_Link", "wheel_R_Link"]:
            raise AssertionError(f"Gait force and swing phases must share left/right order: {gait_contact_names}")
        landing_cfg = unwrapped.reward_manager.get_term_cfg("foot_landing_vel")
        if landing_cfg.func.__name__ != "wheel_terrain_landing_velocity_l2":
            raise AssertionError(f"Unexpected Foot landing function: {landing_cfg.func.__name__}")
        if landing_cfg.params["about_landing_threshold"] != 0.02:
            raise AssertionError(
                f"Unexpected Foot pre-landing threshold: {landing_cfg.params['about_landing_threshold']}"
            )
        if "pen_all_wheels_air_time" in active_rewards:
            raise AssertionError("Old simultaneous-air-time cost must be removed.")
        if landing_cfg.params["allowed_downward_speed"] != 0.2:
            raise AssertionError("Expected a 0.2 m/s landing speed allowance.")
        for name in ("pen_joint_torque", "pen_joint_accel", "pen_joint_power_l1"):
            selected = unwrapped.reward_manager.get_term_cfg(name).params["asset_cfg"]
            names = [unwrapped.scene["robot"].joint_names[i] for i in selected.joint_ids]
            if len(names) != 6 or any("wheel" in name for name in names):
                raise AssertionError(f"Foot regularizer must select six leg joints: {name}: {names}")
        for name in ("pen_action_rate", "pen_action_smoothness"):
            if unwrapped.reward_manager.get_term_cfg(name).params["action_dim"] != 6:
                raise AssertionError(f"Foot action regularizer must exclude wheel outputs: {name}")
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
