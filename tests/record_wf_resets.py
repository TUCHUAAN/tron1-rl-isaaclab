"""Record repeated WF resets with contact and height diagnostics overlaid."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record WF reset validity clips.")
parser.add_argument("--task", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--num_resets", type=int, default=8)
parser.add_argument("--steps_per_reset", type=int, default=50)
parser.add_argument("--capture_stride", type=int, default=2)
parser.add_argument("--fps", type=int, default=25)
parser.add_argument("--contact_threshold", type=float, default=5.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import bipedal_locomotion  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

WHEEL_RADIUS = 0.128


def _finite_median(values: torch.Tensor) -> float:
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return float("nan")
    return float(torch.median(values).item())


def _draw_overlay(frame: np.ndarray, lines: list[str], alert: bool) -> np.ndarray:
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    image = Image.fromarray(frame.astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    line_height = 18
    box_height = 12 + line_height * len(lines)
    color = (120, 0, 0, 190) if alert else (0, 0, 0, 175)
    draw.rectangle((8, 8, 620, 8 + box_height), fill=color)
    for index, line in enumerate(lines):
        draw.text((16, 14 + index * line_height), line, fill=(255, 255, 255, 255), font=font)
    return np.asarray(image)


def _separator(width: int, height: int, reset_index: int, frames: int = 8) -> list[np.ndarray]:
    output = []
    for _ in range(frames):
        image = Image.new("RGB", (width, height), (15, 15, 15))
        draw = ImageDraw.Draw(image)
        draw.text((30, 30), f"RESET {reset_index + 1}", fill=(255, 255, 255), font=ImageFont.load_default())
        output.append(np.asarray(image))
    return output


def main() -> None:
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env_cfg.scene.num_envs = 1
    env_cfg.commands.base_velocity.debug_vis = False
    if env_cfg.scene.height_scanner is not None:
        env_cfg.scene.height_scanner.debug_vis = False
    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")
    unwrapped = env.unwrapped

    robot = unwrapped.scene["robot"]
    contact_sensor = unwrapped.scene.sensors["contact_forces"]
    height_scanner = unwrapped.scene.sensors["height_scanner"]
    wheel_body_ids = contact_sensor.find_bodies("wheel_[LR]_Link")[0]
    robot_wheel_body_ids = robot.find_bodies("wheel_[LR]_Link")[0]
    wheel_joint_ids = robot.find_joints("wheel_[LR]_Joint")[0]
    action = torch.zeros((1, unwrapped.action_manager.total_action_dim), device=unwrapped.device)

    writer = None
    summaries = []
    frame_width = frame_height = None

    try:
        for reset_index in range(args.num_resets):
            env.reset()
            reset_stats = {
                "reset": reset_index + 1,
                "left_contact_frames": 0,
                "right_contact_frames": 0,
                "any_contact_frames": 0,
                "both_contact_frames": 0,
                "done_events": 0,
                "captured_steps": args.steps_per_reset,
                "base_height_samples": [],
                "left_clearance_samples": [],
                "right_clearance_samples": [],
            }

            for step in range(args.steps_per_reset):
                _, _, terminated, truncated, _ = env.step(action)

                wheel_forces = torch.linalg.vector_norm(
                    contact_sensor.data.net_forces_w[0, wheel_body_ids], dim=-1
                )
                contacts = wheel_forces > args.contact_threshold
                left_contact = bool(contacts[0].item())
                right_contact = bool(contacts[1].item())
                reset_stats["left_contact_frames"] += int(left_contact)
                reset_stats["right_contact_frames"] += int(right_contact)
                reset_stats["any_contact_frames"] += int(left_contact or right_contact)
                reset_stats["both_contact_frames"] += int(left_contact and right_contact)

                done = bool(torch.any(terminated | truncated).item())
                reset_stats["done_events"] += int(done)

                ground_height = _finite_median(height_scanner.data.ray_hits_w[0, :, 2])
                base_height = float(robot.data.root_pos_w[0, 2].item() - ground_height)
                wheel_z = robot.data.body_pos_w[0, robot_wheel_body_ids, 2]
                clearances = wheel_z - ground_height - WHEEL_RADIUS
                wheel_speed = robot.data.joint_vel[0, wheel_joint_ids]
                height_command = float(unwrapped.command_manager.get_command("body_height")[0, 0].item())

                reset_stats["base_height_samples"].append(base_height)
                reset_stats["left_clearance_samples"].append(float(clearances[0].item()))
                reset_stats["right_clearance_samples"].append(float(clearances[1].item()))

                if step % args.capture_stride != 0:
                    continue

                frame = env.render()
                if frame is None:
                    raise RuntimeError("env.render() returned None; camera rendering is not available.")
                frame = np.asarray(frame)
                if frame_width is None:
                    frame_height, frame_width = frame.shape[:2]
                    writer = imageio.get_writer(
                        output_path,
                        fps=args.fps,
                        codec="libx264",
                        quality=8,
                        macro_block_size=None,
                    )

                lines = [
                    f"task={args.task}",
                    f"reset={reset_index + 1}/{args.num_resets} step={step + 1}/{args.steps_per_reset}",
                    f"contact L/R={int(left_contact)}/{int(right_contact)} force={wheel_forces[0]:.1f}/{wheel_forces[1]:.1f} N",
                    f"base_rel_height={base_height:.3f} m command={height_command:.3f} m",
                    f"wheel_clearance L/R={clearances[0]:+.3f}/{clearances[1]:+.3f} m",
                    f"wheel_speed L/R={wheel_speed[0]:+.3f}/{wheel_speed[1]:+.3f} rad/s",
                    f"terminated={bool(torch.any(terminated).item())} truncated={bool(torch.any(truncated).item())}",
                ]
                alert = (not left_contact and not right_contact) or done
                writer.append_data(_draw_overlay(frame, lines, alert))

            count = max(args.steps_per_reset, 1)
            summary = {
                "reset": reset_index + 1,
                "left_contact_ratio": reset_stats["left_contact_frames"] / count,
                "right_contact_ratio": reset_stats["right_contact_frames"] / count,
                "any_contact_ratio": reset_stats["any_contact_frames"] / count,
                "both_contact_ratio": reset_stats["both_contact_frames"] / count,
                "done_events": reset_stats["done_events"],
                "mean_base_relative_height_m": float(np.nanmean(reset_stats["base_height_samples"])),
                "mean_left_wheel_clearance_m": float(np.nanmean(reset_stats["left_clearance_samples"])),
                "mean_right_wheel_clearance_m": float(np.nanmean(reset_stats["right_clearance_samples"])),
            }
            summaries.append(summary)
            print(f"[RESET VIDEO] {summary}", flush=True)

            if writer is not None and reset_index + 1 < args.num_resets:
                for separator_frame in _separator(frame_width, frame_height, reset_index + 1):
                    writer.append_data(separator_frame)
    finally:
        if writer is not None:
            writer.close()
        env.close()

    summary_path = output_path.with_suffix(".json")
    aggregate = {
        "task": args.task,
        "num_resets": args.num_resets,
        "steps_per_reset": args.steps_per_reset,
        "contact_threshold_n": args.contact_threshold,
        "resets": summaries,
        "aggregate": {
            "mean_any_contact_ratio": float(np.mean([item["any_contact_ratio"] for item in summaries])),
            "mean_both_contact_ratio": float(np.mean([item["both_contact_ratio"] for item in summaries])),
            "total_done_events": int(sum(item["done_events"] for item in summaries)),
        },
    }
    summary_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False))
    print(f"[RESET VIDEO] Video: {output_path}", flush=True)
    print(f"[RESET VIDEO] Summary: {summary_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
