"""Run the TRON1A wheelfoot wheel/foot expert checkpoints with a safety FSM."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play two WF expert policies with deterministic mode switching.")
parser.add_argument("--task", type=str, default="Isaac-Limx-WF-Dual-Mode-Play-v0")
parser.add_argument("--wheel_checkpoint", type=str, required=True)
parser.add_argument("--foot_checkpoint", type=str, required=True)
parser.add_argument("--mode", choices=("wheel", "foot", "auto"), default="wheel")
parser.add_argument("--body_height", type=float, default=None)
parser.add_argument("--terrain_relief_threshold", type=float, default=0.08)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_steps", type=int, default=None, help="Optional finite step count for smoke tests.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from rsl_rl.runner import OnPolicyRunner

from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import bipedal_locomotion  # noqa: F401
from bipedal_locomotion.tasks.locomotion.agents.limx_rsl_rl_ppo_cfg import (
    WF_TRON1AFootAllTerrainPPORunnerCfg,
    WF_TRON1AWheelModePPORunnerCfg,
)
from bipedal_locomotion.utils.wf_mode_fsm import WheelfootModeFSM, terrain_mode_request


def unpack_observations(result, infos=None):
    """Normalize Isaac Lab <=2.2 and >=2.3 RSL wrapper observation APIs."""
    if isinstance(result, tuple):
        policy_obs, extras = result
        return policy_obs, extras["observations"]
    if hasattr(result, "keys") and "policy" in result.keys():
        return result["policy"], result
    if infos is None:
        raise TypeError(f"Unsupported observation output: {type(result)!r}")
    return result, infos["observations"]


def policy_action(policy, encoder, obs, obs_history, commands):
    latent = encoder(obs_history)
    return policy(torch.cat((latent, obs, commands), dim=-1).detach())


def main():
    env_cfg = parse_env_cfg(
        task_name=args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    wheel_cfg = WF_TRON1AWheelModePPORunnerCfg()
    foot_cfg = WF_TRON1AFootAllTerrainPPORunnerCfg()
    wheel_cfg.device = args_cli.device
    foot_cfg.device = args_cli.device

    wheel_runner = OnPolicyRunner(env, wheel_cfg.to_dict(), log_dir=None, device=args_cli.device)
    wheel_runner.load(args_cli.wheel_checkpoint)
    foot_runner = OnPolicyRunner(env, foot_cfg.to_dict(), log_dir=None, device=args_cli.device)
    foot_runner.load(args_cli.foot_checkpoint)

    wheel_policy = wheel_runner.get_inference_policy(device=env.unwrapped.device)
    wheel_encoder = wheel_runner.get_inference_encoder(device=env.unwrapped.device)
    foot_policy = foot_runner.get_inference_policy(device=env.unwrapped.device)
    foot_encoder = foot_runner.get_inference_encoder(device=env.unwrapped.device)

    obs, groups = unpack_observations(env.get_observations())
    obs_history = groups["obsHistory"].flatten(start_dim=1)
    commands = groups["commands"]

    initial_mode = "foot" if args_cli.mode == "foot" else "wheel"
    fsm = WheelfootModeFSM(initial_mode=initial_mode)
    robot = env.unwrapped.scene["robot"]
    contact_sensor = env.unwrapped.scene.sensors["contact_forces"]
    height_scanner = env.unwrapped.scene.sensors["height_scanner"]
    wheel_joint_ids = robot.find_joints(["wheel_L_Joint", "wheel_R_Joint"])[0]
    wheel_body_ids = contact_sensor.find_bodies("wheel_[LR]_Link")[0]
    previous_mode = fsm.mode

    step_count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            if args_cli.body_height is not None:
                env.unwrapped.command_manager.get_term("body_height").set_target(args_cli.body_height)

            if args_cli.mode == "auto":
                fsm.request(
                    terrain_mode_request(
                        height_scanner.data.ray_hits_w,
                        relief_threshold=args_cli.terrain_relief_threshold,
                    )
                )
            else:
                fsm.request(args_cli.mode)

            base_speed = float(torch.linalg.vector_norm(robot.data.root_lin_vel_b[0, :2]).item())
            wheel_speed = float(torch.mean(torch.abs(robot.data.joint_vel[0, wheel_joint_ids])).item())
            wheel_forces = torch.linalg.vector_norm(
                contact_sensor.data.net_forces_w[0, wheel_body_ids], dim=-1
            )
            both_contact = bool(torch.all(wheel_forces > 1.0).item())
            upright_cosine = float((-robot.data.projected_gravity_b[0, 2]).item())
            fsm.update(
                base_speed=base_speed,
                wheel_speed=wheel_speed,
                both_wheels_contact=both_contact,
                upright_cosine=upright_cosine,
            )

            # A mode transition is a zero-velocity manoeuvre. Override both the
            # policy command tensor and the live command term while blending.
            if fsm.mode.value in ("wheel_to_foot", "foot_to_wheel"):
                commands = commands.clone()
                commands[:, :3] = 0.0
                velocity_term = env.unwrapped.command_manager.get_term("base_velocity")
                if hasattr(velocity_term, "vel_command_b"):
                    velocity_term.vel_command_b[:] = 0.0

            # The FSM can override commands after env.step produced obs. Refresh
            # the shared hold flags/errors without advancing their clock twice.
            if "zero_command_hold" in env.unwrapped.observation_manager.active_terms["policy"]:
                hold = env.unwrapped.reward_manager.get_term_cfg("pen_zero_command_hold").func
                obs = obs.clone()
                obs[:, -6:] = hold.observe(env.unwrapped)

            wheel_actions = policy_action(wheel_policy, wheel_encoder, obs, obs_history, commands)
            foot_actions = policy_action(foot_policy, foot_encoder, obs, obs_history, commands)
            actions = fsm.route_actions(wheel_actions, foot_actions)

            if fsm.mode != previous_mode:
                print(f"[INFO] WF mode: {previous_mode.value} -> {fsm.mode.value}")
                previous_mode = fsm.mode

            step_result, _, _, infos = env.step(actions)
            obs, groups = unpack_observations(step_result, infos)
            obs_history = groups["obsHistory"].flatten(start_dim=1)
            commands = groups["commands"]
            step_count += 1
            if args_cli.max_steps is not None and step_count >= args_cli.max_steps:
                break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
