"""IsaacLab smoke test for WF reward configuration and runtime tensors.

Run through the IsaacLab environment, for example:

    python tests/smoke_wf_reward_env.py --task Isaac-Limx-WF-Wheel-Mode-v0 --headless --device cpu
"""

from __future__ import annotations

import argparse
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
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env_cfg.commands.base_velocity.debug_vis = False
    env = gym.make(args.task, cfg=env_cfg)
    print("[SMOKE] Environment created.", flush=True)
    unwrapped = env.unwrapped

    env.reset()
    print("[SMOKE] Environment reset.", flush=True)
    action = torch.zeros_like(unwrapped.action_manager.action)
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
    unexpected_non_timeout_terms = {
        name
        for name in unwrapped.termination_manager.active_terms
        if not unwrapped.termination_manager.get_term_cfg(name).time_out and name != "base_contact"
    }
    if unexpected_non_timeout_terms:
        raise AssertionError(
            "is_terminated would also penalize unexpected non-timeout terms: "
            f"{sorted(unexpected_non_timeout_terms)}"
        )

    height_cfg = unwrapped.reward_manager.get_term_cfg("pen_base_height")
    if "grounded" not in height_cfg.func.__name__:
        raise AssertionError(f"Height penalty is not support-gated: {height_cfg.func.__name__}")
    stand_cfg = unwrapped.reward_manager.get_term_cfg("stand_still")
    if stand_cfg.func.__name__ != "stand_still_grounded":
        raise AssertionError(f"Stand-still penalty is not support-gated: {stand_cfg.func.__name__}")

    height_cfg = unwrapped.command_manager.get_term("body_height").cfg
    if height_cfg.ranges.height != (0.65, 0.85) or height_cfg.endpoint_fraction != 0.20:
        raise AssertionError(
            f"Unexpected height sampling: range={height_cfg.ranges.height}, "
            f"endpoint_fraction={height_cfg.endpoint_fraction}"
        )

    if "Wheel-Mode" in args.task:
        expected_rewards = {
            "pen_wheel_contact",
            "pen_wheel_horizontal_neutral",
            "pen_wheel_target_symmetry",
            "pen_zero_command_wheel_target",
            "pen_zero_command_yaw_rate",
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
        zero_yaw_cfg = unwrapped.reward_manager.get_term_cfg("pen_zero_command_yaw_rate")
        if zero_yaw_cfg.func.__name__ != "zero_command_yaw_rate_grounded_l2":
            raise AssertionError(f"Zero-command yaw penalty is not support-gated: {zero_yaw_cfg.func.__name__}")
        wheel_height_cfg = unwrapped.reward_manager.get_term_cfg("pen_base_height")
        if wheel_height_cfg.weight != -60.0:
            raise AssertionError(f"Unexpected Wheel height penalty weight: {wheel_height_cfg.weight}")
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
