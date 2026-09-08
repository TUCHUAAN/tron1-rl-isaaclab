#!/usr/bin/env python3
"""Run an original PF_TRON1A Isaac Lab policy in MuJoCo."""

from __future__ import annotations

import argparse
import math
import re
import time
from collections import deque
from pathlib import Path
from typing import Sequence

import glfw
import mujoco
import numpy as np
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_ROOT = REPO_ROOT / "logs" / "rsl_rl" / "pf_tron_1a_flat"
DEFAULT_ROBOT_XML = Path(
    "/home/tuchuaan/tron1-mujoco-sim/robot-description/pointfoot/PF_TRON1A/xml/robot.xml"
)

PHYSICS_DT = 0.005
POLICY_DECIMATION = 4
POLICY_DT = PHYSICS_DT * POLICY_DECIMATION
HISTORY_LENGTH = 10

# Exact schema of Isaac-Limx-PF-Blind-Flat-v0.  The two foot joints in the
# source URDF are fixed, so Isaac's articulation exposes six, not eight,
# movable joint states.
POLICY_OBS_DIM = 30
HISTORY_OBS_DIM = 30
HISTORY_INPUT_DIM = HISTORY_LENGTH * HISTORY_OBS_DIM
VELOCITY_COMMAND_DIM = 3
GAIT_COMMAND_DIM = 4
ENCODER_OUTPUT_DIM = 3
ACTOR_INPUT_DIM = ENCODER_OUTPUT_DIM + POLICY_OBS_DIM + VELOCITY_COMMAND_DIM
ACTION_DIM = 6

# Isaac/PhysX order used by the PF policy.  MuJoCo stores the left chain before
# the right chain, so every state and torque access below is resolved by name.
JOINT_NAMES = (
    "abad_L_Joint",
    "abad_R_Joint",
    "hip_L_Joint",
    "hip_R_Joint",
    "knee_L_Joint",
    "knee_R_Joint",
)


def _checkpoint_iteration(path: Path) -> int:
    match = re.fullmatch(r"model_(\d+)\.pt", path.name)
    return int(match.group(1)) if match else -1


def discover_checkpoint(checkpoint_root: Path, iteration: int) -> Path:
    """Return the newest checkpoint with the requested file iteration."""
    candidates = tuple(checkpoint_root.glob(f"**/model_{iteration}.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"在 {checkpoint_root} 下没有找到 model_{iteration}.pt。"
        )
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, str(path)))


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"请求了 {device}，但当前 PyTorch 无法使用 CUDA。")
    return device


def _model_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"MuJoCo 模型缺少 {object_type.name}: {name}")
    return object_id


def load_robot_model(robot_xml: Path, torque_limit: float) -> mujoco.MjModel:
    if not robot_xml.is_file():
        raise FileNotFoundError(f"PF 机器人 XML 不存在：{robot_xml}")
    model = mujoco.MjModel.from_xml_path(str(robot_xml))
    model.opt.timestep = PHYSICS_DT
    for joint_name in JOINT_NAMES:
        actuator_id = _model_id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name
        )
        model.actuator_ctrllimited[actuator_id] = 1
        model.actuator_ctrlrange[actuator_id] = (-torque_limit, torque_limit)
    return model


def _build_mlp(
    state_dict: dict[str, torch.Tensor], prefix: str
) -> nn.Sequential:
    weight_pattern = re.compile(rf"^{re.escape(prefix)}(\d+)\.weight$")
    layers_by_index: list[tuple[int, torch.Tensor]] = []
    for name, value in state_dict.items():
        match = weight_pattern.match(name)
        if match:
            layers_by_index.append((int(match.group(1)), value))
    layers_by_index.sort()
    if not layers_by_index:
        raise ValueError(f"checkpoint 中没有网络层：{prefix}*.weight")

    last_linear_index = layers_by_index[-1][0]
    modules: list[nn.Module] = []
    for layer_index, weight in layers_by_index:
        modules.append(nn.Linear(int(weight.shape[1]), int(weight.shape[0])))
        if layer_index != last_linear_index:
            modules.append(nn.ELU())
    network = nn.Sequential(*modules)
    network_state = {
        name.removeprefix(prefix): value
        for name, value in state_dict.items()
        if name.startswith(prefix)
    }
    network.load_state_dict(network_state, strict=True)
    return network


def _require_finite_state(
    state_dict: dict[str, torch.Tensor], label: str
) -> None:
    invalid = [
        name
        for name, value in state_dict.items()
        if torch.is_floating_point(value) and not torch.isfinite(value).all()
    ]
    if invalid:
        raise ValueError(
            f"checkpoint 的 {label} 包含 NaN/Inf：{', '.join(invalid[:5])}"
        )


class PFPolicy:
    """Load the PF actor/encoder and reproduce its deployment observations."""

    def __init__(
        self,
        checkpoint_path: Path,
        device: torch.device,
        model: mujoco.MjModel,
        action_clip: float | None,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model = model
        self.action_clip = action_clip
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        actor_state = checkpoint["model_state_dict"]
        encoder_state = checkpoint["encoder_state_dict"]
        _require_finite_state(actor_state, "Actor/Critic 参数")
        _require_finite_state(encoder_state, "Encoder 参数")
        self.actor = _build_mlp(actor_state, "actor.").to(device).eval()
        self.encoder = _build_mlp(encoder_state, "encoder.").to(device).eval()

        encoder_input = int(self.encoder[0].in_features)
        encoder_output = int(self.encoder[-1].out_features)
        actor_input = int(self.actor[0].in_features)
        actor_output = int(self.actor[-1].out_features)
        if (encoder_input, encoder_output) != (
            HISTORY_INPUT_DIM,
            ENCODER_OUTPUT_DIM,
        ):
            raise ValueError(
                f"Encoder 维度为 {encoder_input}->{encoder_output}，"
                f"原版 PF 应为 {HISTORY_INPUT_DIM}->{ENCODER_OUTPUT_DIM}。"
            )
        if (actor_input, actor_output) != (ACTOR_INPUT_DIM, ACTION_DIM):
            raise ValueError(
                f"Actor 维度为 {actor_input}->{actor_output}，"
                f"原版 PF 应为 {ACTOR_INPUT_DIM}->{ACTION_DIM}。"
            )

        self.base_body_id = _model_id(
            model, mujoco.mjtObj.mjOBJ_BODY, "base_Link"
        )
        self.joint_qpos_addresses = np.asarray(
            [
                model.jnt_qposadr[
                    _model_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
                for name in JOINT_NAMES
            ],
            dtype=np.int32,
        )
        self.joint_dof_addresses = np.asarray(
            [
                model.jnt_dofadr[
                    _model_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
                for name in JOINT_NAMES
            ],
            dtype=np.int32,
        )
        self.default_joint_positions = np.asarray(
            model.qpos0[self.joint_qpos_addresses], dtype=np.float64
        ).copy()
        self.velocity_buffer = np.zeros(6, dtype=np.float64)
        self.last_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self.history: deque[np.ndarray] = deque(maxlen=HISTORY_LENGTH)
        self.policy_step = 0

    def _gait_phase(self, frequency: float) -> np.ndarray:
        phase = (self.policy_step * POLICY_DT * frequency) % 1.0
        angle = 2.0 * math.pi * phase
        return np.asarray((math.sin(angle), math.cos(angle)), dtype=np.float32)

    def _observation(
        self, data: mujoco.MjData, gait_command: np.ndarray
    ) -> np.ndarray:
        mujoco.mj_objectVelocity(
            self.model,
            data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.base_body_id,
            self.velocity_buffer,
            1,
        )
        base_angular_velocity = (
            np.clip(self.velocity_buffer[:3], -100.0, 100.0) * 0.25
        )
        rotation = np.asarray(
            data.xmat[self.base_body_id], dtype=np.float64
        ).reshape(3, 3)
        projected_gravity = np.clip(
            rotation.T @ np.asarray((0.0, 0.0, -1.0)), -100.0, 100.0
        )
        joint_positions = np.clip(
            np.asarray(data.qpos[self.joint_qpos_addresses])
            - self.default_joint_positions,
            -100.0,
            100.0,
        )
        joint_velocities = (
            np.clip(
                np.asarray(data.qvel[self.joint_dof_addresses]),
                -100.0,
                100.0,
            )
            * 0.05
        )
        observation = np.concatenate(
            (
                base_angular_velocity,
                projected_gravity,
                joint_positions,
                joint_velocities,
                self.last_action,
                self._gait_phase(float(gait_command[0])),
                gait_command,
            )
        ).astype(np.float32, copy=False)
        if observation.size != POLICY_OBS_DIM:
            raise RuntimeError(
                f"PF 当前观测为 {observation.size} 维，预期 {POLICY_OBS_DIM} 维。"
            )
        if not np.all(np.isfinite(observation)):
            raise RuntimeError("PF 当前观测包含 NaN 或无穷值。")
        return observation

    def reset(self, data: mujoco.MjData, gait_command: np.ndarray) -> None:
        self.last_action.fill(0.0)
        self.policy_step = 0
        observation = self._observation(data, gait_command)
        self.history.clear()
        self.history.extend(observation.copy() for _ in range(HISTORY_LENGTH))

    def act(
        self,
        data: mujoco.MjData,
        velocity_command: np.ndarray,
        gait_command: np.ndarray,
    ) -> np.ndarray:
        if velocity_command.shape != (VELOCITY_COMMAND_DIM,):
            raise ValueError("PF 速度命令必须为 [vx, vy, wz] 三维。")
        if gait_command.shape != (GAIT_COMMAND_DIM,):
            raise ValueError(
                "PF gait 命令必须为 [frequency, offset, duration, swing_height] 四维。"
            )
        observation = self._observation(data, gait_command)
        self.history.append(observation)
        history = np.concatenate(tuple(self.history)).astype(
            np.float32, copy=False
        )
        if history.size != HISTORY_INPUT_DIM:
            raise RuntimeError(
                f"PF 历史观测为 {history.size} 维，预期 {HISTORY_INPUT_DIM} 维。"
            )

        with torch.inference_mode():
            history_tensor = torch.from_numpy(history).unsqueeze(0).to(self.device)
            observation_tensor = (
                torch.from_numpy(observation).unsqueeze(0).to(self.device)
            )
            command_tensor = (
                torch.from_numpy(velocity_command).unsqueeze(0).to(self.device)
            )
            latent = self.encoder(history_tensor)
            action = self.actor(
                torch.cat((latent, observation_tensor, command_tensor), dim=-1)
            )
        result = action[0].detach().cpu().numpy().astype(np.float32, copy=False)
        if not np.all(np.isfinite(result)):
            raise RuntimeError("PF 策略输出包含 NaN 或无穷值。")
        if self.action_clip is not None:
            result = np.clip(result, -self.action_clip, self.action_clip)
        self.last_action[:] = result
        self.policy_step += 1
        return result


class TorqueController:
    """Apply the PF position action through the Isaac Lab PD law."""

    def __init__(
        self,
        model: mujoco.MjModel,
        torque_limit: float,
        *,
        low_pass_enabled: bool,
        cutoff_hz: float,
    ) -> None:
        self.torque_limit = torque_limit
        self.low_pass_enabled = low_pass_enabled
        self.low_pass_alpha = 1.0 - math.exp(
            -2.0 * math.pi * cutoff_hz * PHYSICS_DT
        )
        self.filtered_torque = np.zeros(ACTION_DIM, dtype=np.float64)
        self.joint_qpos_addresses = np.asarray(
            [
                model.jnt_qposadr[
                    _model_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
                for name in JOINT_NAMES
            ],
            dtype=np.int32,
        )
        self.joint_dof_addresses = np.asarray(
            [
                model.jnt_dofadr[
                    _model_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
                for name in JOINT_NAMES
            ],
            dtype=np.int32,
        )
        self.actuator_ids = np.asarray(
            [
                _model_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in JOINT_NAMES
            ],
            dtype=np.int32,
        )
        self.default_joint_positions = np.asarray(
            model.qpos0[self.joint_qpos_addresses], dtype=np.float64
        ).copy()

    def reset(self) -> None:
        self.filtered_torque.fill(0.0)

    def apply(self, data: mujoco.MjData, action: np.ndarray) -> float:
        targets = self.default_joint_positions + 0.25 * action
        raw_torque = 40.0 * (
            targets - data.qpos[self.joint_qpos_addresses]
        ) - 2.5 * data.qvel[self.joint_dof_addresses]
        raw_torque = np.clip(raw_torque, -self.torque_limit, self.torque_limit)
        if self.low_pass_enabled:
            self.filtered_torque += self.low_pass_alpha * (
                raw_torque - self.filtered_torque
            )
            torque = self.filtered_torque
        else:
            torque = raw_torque
        data.ctrl.fill(0.0)
        data.ctrl[self.actuator_ids] = torque
        return float(np.linalg.norm(torque))


class KeyboardCommandState:
    _MOTION_KEYS = {
        glfw.KEY_W,
        glfw.KEY_S,
        glfw.KEY_Q,
        glfw.KEY_E,
        glfw.KEY_A,
        glfw.KEY_D,
    }

    def __init__(self, base_command: np.ndarray) -> None:
        self.pressed: set[int] = set()
        self.base_command = base_command.copy()

    def handle_key(self, key: int, action: int) -> None:
        if action not in (glfw.PRESS, glfw.RELEASE, glfw.REPEAT):
            return
        if key in self._MOTION_KEYS:
            if action == glfw.RELEASE:
                self.pressed.discard(key)
            else:
                self.pressed.add(key)
        elif key == glfw.KEY_SPACE and action == glfw.PRESS:
            self.clear_motion()
            self.base_command.fill(0.0)

    def clear_motion(self) -> None:
        self.pressed.clear()

    def sync_from_window(self, window: glfw._GLFWwindow) -> None:
        self.pressed = {
            key
            for key in self._MOTION_KEYS
            if glfw.get_key(window, key) == glfw.PRESS
        }

    def command(
        self, forward_speed: float, lateral_speed: float, yaw_speed: float
    ) -> np.ndarray:
        command = self.base_command.copy()
        x_direction = int(glfw.KEY_W in self.pressed) - int(
            glfw.KEY_S in self.pressed
        )
        y_direction = int(glfw.KEY_Q in self.pressed) - int(
            glfw.KEY_E in self.pressed
        )
        yaw_direction = int(glfw.KEY_A in self.pressed) - int(
            glfw.KEY_D in self.pressed
        )
        if x_direction:
            command[0] = x_direction * forward_speed
        if y_direction:
            command[1] = y_direction * lateral_speed
        if yaw_direction:
            command[2] = yaw_direction * yaw_speed
        return command.astype(np.float32, copy=False)


def reset_robot(model: mujoco.MjModel, data: mujoco.MjData, z_offset: float) -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[2] = model.qpos0[2] + z_offset
    mujoco.mj_forward(model, data)


def _camera_yaw(data: mujoco.MjData, base_body_id: int) -> float:
    rotation = np.asarray(data.xmat[base_body_id], dtype=np.float64).reshape(3, 3)
    return math.atan2(rotation[1, 0], rotation[0, 0])


def base_velocity_body(
    model: mujoco.MjModel, data: mujoco.MjData, base_body_id: int
) -> np.ndarray:
    """Return ``[vx, vy, wz]`` in the base frame."""
    spatial_velocity = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(
        model,
        data,
        mujoco.mjtObj.mjOBJ_BODY,
        base_body_id,
        spatial_velocity,
        1,
    )
    return np.asarray(
        (spatial_velocity[3], spatial_velocity[4], spatial_velocity[2]),
        dtype=np.float64,
    )


def render(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    window: glfw._GLFWwindow,
    camera: mujoco.MjvCamera,
    option: mujoco.MjvOption,
    scene: mujoco.MjvScene,
    context: mujoco.MjrContext,
    base_body_id: int,
    command: np.ndarray,
    gait_command: np.ndarray,
    checkpoint_iteration: int,
    torque_norm: float,
) -> None:
    yaw = _camera_yaw(data, base_body_id)
    actual_velocity = base_velocity_body(model, data, base_body_id)
    camera.lookat[:] = data.xpos[base_body_id]
    camera.lookat[2] += 0.05
    camera.azimuth = math.degrees(yaw) + 45.0
    width, height = glfw.get_framebuffer_size(window)
    viewport = mujoco.MjrRect(0, 0, width, height)
    mujoco.mjv_updateScene(
        model,
        data,
        option,
        None,
        camera,
        mujoco.mjtCatBit.mjCAT_ALL.value,
        scene,
    )
    mujoco.mjr_render(viewport, scene, context)
    left_text = (
        "W/S forward/back   Q/E left/right\n"
        "A/D yaw   Space stop   R reset   Esc quit"
    )
    right_text = (
        f"PF checkpoint model_{checkpoint_iteration}.pt\n"
        f"cmd [{command[0]:+.2f}, {command[1]:+.2f}, {command[2]:+.2f}]\n"
        f"actual [{actual_velocity[0]:+.2f}, {actual_velocity[1]:+.2f}, "
        f"{actual_velocity[2]:+.2f}]\n"
        f"gait [{gait_command[0]:.2f}, {gait_command[1]:.2f}, "
        f"{gait_command[2]:.2f}, {gait_command[3]:.2f}]\n"
        f"t {data.time:.2f}  base_z {data.qpos[2]:.3f}  "
        f"torque_norm {torque_norm:.2f}"
    )
    mujoco.mjr_overlay(
        mujoco.mjtFont.mjFONT_NORMAL.value,
        mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
        viewport,
        left_text,
        right_text,
        context,
    )
    glfw.swap_buffers(window)
    glfw.poll_events()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test an original Isaac-Limx-PF-Blind-Flat policy in MuJoCo."
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=2000,
        help="Checkpoint file iteration used for automatic discovery (default: 2000).",
    )
    parser.add_argument("--robot-xml", type=Path, default=DEFAULT_ROBOT_XML)
    parser.add_argument("--device", default="auto", help="cpu, cuda:0, or auto")
    parser.add_argument("--command-vx", type=float, default=0.0)
    parser.add_argument("--command-vy", type=float, default=0.0)
    parser.add_argument("--command-wz", type=float, default=0.0)
    parser.add_argument("--forward-speed", type=float, default=0.8)
    parser.add_argument("--lateral-speed", type=float, default=0.3)
    parser.add_argument("--yaw-speed", type=float, default=0.5)
    parser.add_argument("--gait-frequency", type=float, default=2.0)
    parser.add_argument("--gait-offset", type=float, default=0.5)
    parser.add_argument("--gait-duration", type=float, default=0.5)
    parser.add_argument("--swing-height", type=float, default=0.15)
    parser.add_argument("--spawn-z-offset", type=float, default=0.0)
    parser.add_argument("--torque-limit", type=float, default=80.0)
    parser.add_argument(
        "--action-clip",
        type=float,
        default=None,
        help="Optional symmetric raw-action clip. Off by default to match training.",
    )
    parser.add_argument("--torque-low-pass", action="store_true")
    parser.add_argument("--torque-cutoff-hz", type=float, default=20.0)
    parser.add_argument("--render-hz", type=float, default=60.0)
    parser.add_argument("--camera-distance", type=float, default=3.5)
    parser.add_argument("--camera-elevation", type=float, default=-18.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args(argv)

    if args.iteration < 0:
        parser.error("--iteration 必须大于或等于 0。")
    if args.gait_frequency <= 0.0:
        parser.error("--gait-frequency 必须大于 0。")
    if not 0.0 <= args.gait_offset <= 1.0:
        parser.error("--gait-offset 必须在 [0, 1]。")
    if not 0.0 < args.gait_duration < 1.0:
        parser.error("--gait-duration 必须在 (0, 1)。")
    if not 0.1 <= args.swing_height <= 0.2:
        parser.error("--swing-height 必须在 PF 训练范围 [0.1, 0.2] m。")
    if args.torque_limit <= 0.0:
        parser.error("--torque-limit 必须大于 0。")
    if args.action_clip is not None and args.action_clip <= 0.0:
        parser.error("--action-clip 必须大于 0。")
    if args.torque_cutoff_hz <= 0.0:
        parser.error("--torque-cutoff-hz 必须大于 0。")
    if args.render_hz <= 0.0:
        parser.error("--render-hz 必须大于 0。")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps 必须大于 0。")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoint = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else discover_checkpoint(
            args.checkpoint_root.expanduser().resolve(), args.iteration
        )
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint 不存在：{checkpoint}")
    device = resolve_device(args.device)
    model = load_robot_model(
        args.robot_xml.expanduser().resolve(), args.torque_limit
    )
    data = mujoco.MjData(model)
    base_body_id = _model_id(model, mujoco.mjtObj.mjOBJ_BODY, "base_Link")
    policy = PFPolicy(checkpoint, device, model, args.action_clip)
    controller = TorqueController(
        model,
        args.torque_limit,
        low_pass_enabled=args.torque_low_pass,
        cutoff_hz=args.torque_cutoff_hz,
    )
    gait_command = np.asarray(
        (
            args.gait_frequency,
            args.gait_offset,
            args.gait_duration,
            args.swing_height,
        ),
        dtype=np.float32,
    )
    initial_command = np.asarray(
        (args.command_vx, args.command_vy, args.command_wz), dtype=np.float32
    )
    command_state = KeyboardCommandState(initial_command)
    reset_robot(model, data, args.spawn_z_offset)
    policy.reset(data, gait_command)
    controller.reset()

    checkpoint_iteration = _checkpoint_iteration(checkpoint)
    print(f"[policy] checkpoint={checkpoint}")
    print(
        f"[network] encoder={HISTORY_INPUT_DIM}->{ENCODER_OUTPUT_DIM}, "
        f"actor={ACTOR_INPUT_DIM}->{ACTION_DIM}, device={device}"
    )
    print(
        f"[runtime] physics_dt={PHYSICS_DT}, policy_dt={POLICY_DT}, "
        f"torque_limit={args.torque_limit:g}, "
        f"action_clip={args.action_clip}, "
        f"torque_low_pass={'on' if args.torque_low_pass else 'off'}"
    )
    print("[keys] W/S 前后，Q/E 左右，A/D 转向，Space 停止，R 重置，Esc 退出")

    window = None
    camera = None
    option = None
    scene = None
    context = None
    reset_requested = False
    if not args.headless:
        if not glfw.init():
            raise RuntimeError("GLFW 初始化失败；无显示环境可使用 --headless。")
        window = glfw.create_window(1200, 900, "Original PF MuJoCo Test", None, None)
        if window is None:
            glfw.terminate()
            raise RuntimeError("GLFW 窗口创建失败；无显示环境可使用 --headless。")
        glfw.make_context_current(window)
        glfw.swap_interval(1)
        camera = mujoco.MjvCamera()
        option = mujoco.MjvOption()
        scene = mujoco.MjvScene(model, maxgeom=max(1000, model.ngeom + 100))
        context = mujoco.MjrContext(
            model, mujoco.mjtFontScale.mjFONTSCALE_150.value
        )
        mujoco.mjv_defaultCamera(camera)
        mujoco.mjv_defaultOption(option)
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.distance = args.camera_distance
        camera.elevation = args.camera_elevation

        def key_callback(
            callback_window: glfw._GLFWwindow,
            key: int,
            scancode: int,
            action: int,
            mods: int,
        ) -> None:
            nonlocal reset_requested
            del scancode, mods
            if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
                glfw.set_window_should_close(callback_window, True)
            elif key == glfw.KEY_R and action == glfw.PRESS:
                reset_requested = True
            else:
                command_state.handle_key(key, action)

        def focus_callback(
            callback_window: glfw._GLFWwindow, focused: bool
        ) -> None:
            del callback_window
            if not focused:
                command_state.clear_motion()

        glfw.set_key_callback(window, key_callback)
        glfw.set_window_focus_callback(window, focus_callback)

    physics_steps = 0
    episode_physics_steps = 0
    action = np.zeros(ACTION_DIM, dtype=np.float32)
    command = initial_command.copy()
    torque_norm = 0.0
    next_render_time = float(data.time)
    next_status_time = float(data.time)
    wall_start = time.perf_counter()
    simulation_start = float(data.time)
    try:
        while True:
            if window is not None and glfw.window_should_close(window):
                break
            if args.max_steps is not None and physics_steps >= args.max_steps:
                break
            if reset_requested:
                command_state.clear_motion()
                reset_robot(model, data, args.spawn_z_offset)
                policy.reset(data, gait_command)
                controller.reset()
                action.fill(0.0)
                episode_physics_steps = 0
                reset_requested = False
                next_render_time = float(data.time)
                next_status_time = float(data.time)
                wall_start = time.perf_counter()
                simulation_start = float(data.time)
                print("[reset] robot reset to the PF XML initial state")

            if episode_physics_steps % POLICY_DECIMATION == 0:
                command = command_state.command(
                    args.forward_speed, args.lateral_speed, args.yaw_speed
                )
                action = policy.act(data, command, gait_command)
            torque_norm = controller.apply(data, action)
            mujoco.mj_step(model, data)
            if not np.all(np.isfinite(data.qpos)) or not np.all(
                np.isfinite(data.qvel)
            ):
                raise RuntimeError("MuJoCo 状态包含 NaN/Inf。")
            physics_steps += 1
            episode_physics_steps += 1

            if data.time + 1.0e-12 >= next_status_time:
                actual_velocity = base_velocity_body(model, data, base_body_id)
                print(
                    f"[status] t={data.time:7.3f} base_z={data.qpos[2]:.3f} "
                    f"cmd=({command[0]:+.2f},{command[1]:+.2f},{command[2]:+.2f}) "
                    f"actual=({actual_velocity[0]:+.2f},{actual_velocity[1]:+.2f},"
                    f"{actual_velocity[2]:+.2f}) "
                    f"torque_norm={torque_norm:.2f}"
                )
                next_status_time += 1.0

            if window is not None and data.time + 1.0e-12 >= next_render_time:
                assert camera is not None
                assert option is not None
                assert scene is not None
                assert context is not None
                render(
                    model,
                    data,
                    window,
                    camera,
                    option,
                    scene,
                    context,
                    base_body_id,
                    command,
                    gait_command,
                    checkpoint_iteration,
                    torque_norm,
                )
                command_state.sync_from_window(window)
                while next_render_time <= data.time + 1.0e-12:
                    next_render_time += 1.0 / args.render_hz

            if not args.no_realtime:
                target_wall_time = wall_start + (
                    float(data.time) - simulation_start
                )
                sleep_time = target_wall_time - time.perf_counter()
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
    finally:
        command_state.clear_motion()
        if context is not None:
            context.free()
        if window is not None:
            glfw.destroy_window(window)
            glfw.terminate()

    print(
        f"[done] physics_steps={physics_steps}, sim_time={data.time:.3f}, "
        f"base=({data.qpos[0]:.3f}, {data.qpos[1]:.3f}, {data.qpos[2]:.3f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
