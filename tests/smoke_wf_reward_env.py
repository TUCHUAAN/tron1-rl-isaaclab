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

    if "Wheel-Mode" in args.task:
        active_rewards = set(unwrapped.reward_manager.active_terms)
        expected_rewards = {"pen_wheel_contact", "pen_wheel_horizontal_neutral"}
        missing_rewards = expected_rewards - active_rewards
        if missing_rewards:
            raise AssertionError(f"Missing new Wheel rewards: {sorted(missing_rewards)}")
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
