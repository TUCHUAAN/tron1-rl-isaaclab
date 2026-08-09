#!/usr/bin/env python3
"""Deploy the latest TRON1A wheel-mode policy in MuJoCo."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import glfw
import mujoco
import numpy as np
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TERRAIN_XML = Path("/home/tuchuaan/UDMMR/models/terrains/training_terrain.xml")
DEFAULT_TERRAIN_MANIFEST = Path("/home/tuchuaan/UDMMR/models/terrains/training_terrain_manifest.json")
DEFAULT_ROBOT_XML = Path(
    "/home/tuchuaan/tron1-mujoco-sim/robot-description/pointfoot/WF_TRON1A/xml/robot.xml"
)
DEFAULT_CHECKPOINT_ROOT = REPO_ROOT / "logs" / "rsl_rl" / "wf_tron_1a_wheel_mode"

PHYSICS_DT = 0.005
POLICY_DECIMATION = 4
POLICY_DT = PHYSICS_DT * POLICY_DECIMATION
HISTORY_LENGTH = 10
POLICY_OBS_DIM = 155
HISTORY_OBS_DIM = 34
COMMAND_DIM = 4
ACTION_DIM = 8

# This is the PhysX tensor order recorded by the training environment.  It is
# intentionally different from the left-chain/right-chain order in the MJCF.
# All MuJoCo state and torque access below is name-based so the policy sees the
# exact Isaac Lab order.
ALL_JOINT_NAMES = (
    "abad_L_Joint",
    "abad_R_Joint",
    "hip_L_Joint",
    "hip_R_Joint",
    "knee_L_Joint",
    "knee_R_Joint",
    "wheel_L_Joint",
    "wheel_R_Joint",
)
LEG_JOINT_NAMES = (
    "abad_L_Joint",
    "abad_R_Joint",
    "hip_L_Joint",
    "hip_R_Joint",
    "knee_L_Joint",
    "knee_R_Joint",
)
WHEEL_JOINT_NAMES = ("wheel_L_Joint", "wheel_R_Joint")

TERRAIN_DISPLAY_NAMES = (
    "平地、轻微起伏与粗糙地面",
    "坡面",
    "单级高度突变",
    "台阶",
    "低矮障碍",
    "低矮隧洞",
    "固定综合地形",
    "随机综合地形",
)


@dataclass(frozen=True)
class TerrainRegion:
    terrain_type: int
    terrain_level: int
    number: int
    path_type: int
    center_xy_w: tuple[float, float]


def _read_choice(label: str, minimum: int, maximum: int, default: int) -> int:
    while True:
        try:
            raw_value = input(
                f"请选择{label} [{minimum}-{maximum}]（直接回车默认 {default}）："
            ).strip()
        except EOFError:
            raw_value = ""
        if not raw_value:
            return default
        try:
            value = int(raw_value)
        except ValueError:
            print(f"输入无效：{label}必须是 {minimum}-{maximum} 范围内的整数。")
            continue
        if minimum <= value <= maximum:
            return value
        print(f"输入超出范围：{label}必须在 {minimum}-{maximum} 之间。")


def load_terrain_regions(manifest_path: Path) -> tuple[TerrainRegion, ...]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"地形清单不存在：{manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    regions = []
    for entry in manifest.get("regions", []):
        center = entry["center_xy_w"]
        regions.append(
            TerrainRegion(
                terrain_type=int(entry["terrain_type"]),
                terrain_level=int(entry["terrain_level"]),
                number=int(entry["number"]),
                path_type=int(entry["path_type"]),
                center_xy_w=(float(center[0]), float(center[1])),
            )
        )
    if not regions:
        raise ValueError(f"地形清单中没有 regions：{manifest_path}")
    return tuple(regions)


def find_terrain_region(
    regions: Sequence[TerrainRegion], terrain_type: int, terrain_level: int, number: int
) -> TerrainRegion:
    for region in regions:
        if (
            region.terrain_type == terrain_type
            and region.terrain_level == terrain_level
            and region.number == number
        ):
            return region
    raise ValueError(
        "地形区域不存在："
        f"type={terrain_type}, level={terrain_level}, number={number}"
    )


def _print_selected_region(region: TerrainRegion) -> None:
    print(
        "已选择地形区域："
        f"类型={region.terrain_type}（{TERRAIN_DISPLAY_NAMES[region.terrain_type - 1]}），"
        f"等级={region.terrain_level}，号码={region.number}，"
        f"路径类型={region.path_type}，基准点={region.center_xy_w}"
    )
    print("==================================\n")


def select_terrain_region_from_terminal(
    regions: Sequence[TerrainRegion],
) -> TerrainRegion:
    """Use the same type/level/number prompts as the UDMMR keyboard test."""
    print("\n========== 地形区域选择 ==========")
    print("地形类型：")
    for terrain_id, terrain_name in enumerate(TERRAIN_DISPLAY_NAMES, start=1):
        print(f"  {terrain_id}: {terrain_name}")
    terrain_type = _read_choice("地形类型", 1, 8, 4)
    if terrain_type <= 6:
        terrain_level = _read_choice("地形等级", 1, 4, 1)
        terrain_number = 1
        print("平地和单体地形固定使用号码 1。")
    else:
        terrain_level = 4
        terrain_number = _read_choice("综合地形号码", 1, 4, 1)
        print("综合地形固定使用等级 4。")
    region = find_terrain_region(regions, terrain_type, terrain_level, terrain_number)
    _print_selected_region(region)
    return region


def select_or_reuse_terrain_region_on_reset(
    current_region: TerrainRegion, regions: Sequence[TerrainRegion]
) -> TerrainRegion:
    print(
        "\n当前地形："
        f"类型={current_region.terrain_type}（{TERRAIN_DISPLAY_NAMES[current_region.terrain_type - 1]}），"
        f"等级={current_region.terrain_level}，号码={current_region.number}，"
        f"基准点={current_region.center_xy_w}"
    )
    while True:
        try:
            choice = input(
                "重置时选择地形：1=复用当前地形（默认），2=重新选择地形："
            ).strip().lower()
        except EOFError:
            choice = ""
        if choice in ("", "1", "r", "reuse", "复用"):
            print("本次 reset 复用当前地形区域。\n")
            return current_region
        if choice in ("2", "s", "select", "重新选择"):
            return select_terrain_region_from_terminal(regions)
        print("输入无效：请输入 1 复用当前地形，或输入 2 重新选择地形。")


def discover_latest_checkpoint(checkpoint_root: Path) -> Path:
    candidates = tuple(checkpoint_root.glob("**/model_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"没有在 {checkpoint_root} 中找到 model_*.pt")
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, str(path)))


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"请求了 {device}，但当前 PyTorch 无法使用 CUDA。")
    return device


def build_scene_model(terrain_xml: Path, robot_xml: Path, physics_dt: float) -> mujoco.MjModel:
    """Attach the supplied standalone robot MJCF to the supplied terrain MJCF."""
    if not terrain_xml.is_file():
        raise FileNotFoundError(f"地形 XML 不存在：{terrain_xml}")
    if not robot_xml.is_file():
        raise FileNotFoundError(f"机器人 XML 不存在：{robot_xml}")

    terrain_spec = mujoco.MjSpec.from_file(str(terrain_xml))
    robot_spec = mujoco.MjSpec.from_file(str(robot_xml))

    # The robot file is a complete standalone scene. Its floor would otherwise
    # duplicate the training terrain's floor and create duplicate contacts.
    robot_floor = robot_spec.geom("floor")
    if robot_floor is not None:
        robot_spec.delete(robot_floor)

    terrain_spec.option.timestep = physics_dt
    robot_spec.option.timestep = physics_dt
    robot_spec.njmax = terrain_spec.njmax
    robot_spec.nconmax = terrain_spec.nconmax
    attach_frame = terrain_spec.worldbody.add_frame()
    terrain_spec.attach(robot_spec, prefix="", suffix="", frame=attach_frame)
    model = terrain_spec.compile()
    model.opt.timestep = physics_dt

    # Keep robot geoms out of the terrain-only height scanner's geom group.
    model.geom_group[model.geom_bodyid != 0] = 2

    for actuator_name in ALL_JOINT_NAMES:
        actuator_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
        )
        if actuator_id < 0:
            raise ValueError(f"机器人模型缺少执行器：{actuator_name}")
        model.actuator_ctrllimited[actuator_id] = 1
        model.actuator_ctrlrange[actuator_id] = (-80.0, 80.0)
    return model


def _model_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"MuJoCo 模型缺少 {object_type.name}: {name}")
    return object_id


class TerrainHeightScanner:
    """Reproduce Isaac Lab's 1 m x 1 m yaw-attached 0.1 m height scan."""

    def __init__(self, model: mujoco.MjModel, base_body_id: int) -> None:
        self.model = model
        self.base_body_id = base_body_id
        axis = np.arange(-0.5, 0.5 + 1.0e-9, 0.1, dtype=np.float64)
        grid_x, grid_y = np.meshgrid(axis, axis, indexing="xy")
        self.offsets = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        self.geom_group = np.zeros(6, dtype=np.uint8)
        self.geom_group[0] = 1
        self.ray_direction = np.array((0.0, 0.0, -1.0), dtype=np.float64)
        self.geom_id = np.empty(1, dtype=np.int32)

    def scan(self, data: mujoco.MjData) -> np.ndarray:
        base_position = np.asarray(data.xpos[self.base_body_id], dtype=np.float64)
        rotation = np.asarray(data.xmat[self.base_body_id], dtype=np.float64).reshape(3, 3)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        heights = np.empty(self.offsets.shape[0], dtype=np.float32)
        ray_origin = base_position.copy()
        for index, (offset_x, offset_y) in enumerate(self.offsets):
            ray_origin[0] = base_position[0] + cosine * offset_x - sine * offset_y
            ray_origin[1] = base_position[1] + sine * offset_x + cosine * offset_y
            distance = mujoco.mj_ray(
                self.model,
                data,
                ray_origin,
                self.ray_direction,
                self.geom_group,
                True,
                -1,
                self.geom_id,
            )
            heights[index] = 10.0 if distance < 0.0 else np.clip(distance - 0.5, 0.0, 10.0)
        return heights


def _build_mlp(state_dict: dict[str, torch.Tensor], prefix: str) -> nn.Sequential:
    weight_pattern = re.compile(rf"^{re.escape(prefix)}(\d+)\.weight$")
    layers_by_index = []
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


class WheelPolicy:
    """Checkpoint loader plus the exact actor and encoder observation schema."""

    def __init__(
        self,
        checkpoint_path: Path,
        device: torch.device,
        model: mujoco.MjModel,
        scanner: TerrainHeightScanner,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model = model
        self.scanner = scanner
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        self.iteration = int(checkpoint.get("iter", -1))
        self.actor = _build_mlp(checkpoint["model_state_dict"], "actor.").to(device).eval()
        self.encoder = _build_mlp(checkpoint["encoder_state_dict"], "encoder.").to(device).eval()

        encoder_input_dim = int(self.encoder[0].in_features)
        encoder_output_dim = int(self.encoder[-1].out_features)
        actor_input_dim = int(self.actor[0].in_features)
        actor_output_dim = int(self.actor[-1].out_features)
        expected_actor_input = encoder_output_dim + POLICY_OBS_DIM + COMMAND_DIM
        if encoder_input_dim != HISTORY_LENGTH * HISTORY_OBS_DIM:
            raise ValueError(
                f"encoder 输入维度为 {encoder_input_dim}，预期 "
                f"{HISTORY_LENGTH * HISTORY_OBS_DIM}。"
            )
        if actor_input_dim != expected_actor_input or actor_output_dim != ACTION_DIM:
            raise ValueError(
                f"actor 维度为 {actor_input_dim}->{actor_output_dim}，"
                f"预期 {expected_actor_input}->{ACTION_DIM}。"
            )

        self.base_body_id = _model_id(model, mujoco.mjtObj.mjOBJ_BODY, "base_Link")
        self.joint_qpos_addresses = np.asarray(
            [
                model.jnt_qposadr[
                    _model_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
                for name in ALL_JOINT_NAMES
            ],
            dtype=np.int32,
        )
        self.joint_dof_addresses = np.asarray(
            [
                model.jnt_dofadr[
                    _model_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
                for name in ALL_JOINT_NAMES
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

    def reset(self, data: mujoco.MjData, gait_command: np.ndarray) -> None:
        self.last_action.fill(0.0)
        self.policy_step = 0
        history_row = self._history_observation(data, gait_command)
        self.history.clear()
        self.history.extend(history_row.copy() for _ in range(HISTORY_LENGTH))

    def _gait_phase(self, frequency: float) -> np.ndarray:
        phase = (self.policy_step * POLICY_DT * frequency) % 1.0
        angle = 2.0 * math.pi * phase
        return np.asarray((math.sin(angle), math.cos(angle)), dtype=np.float32)

    def _common_observation(self, data: mujoco.MjData) -> np.ndarray:
        mujoco.mj_objectVelocity(
            self.model,
            data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.base_body_id,
            self.velocity_buffer,
            1,
        )
        base_angular_velocity = np.clip(self.velocity_buffer[:3], -100.0, 100.0) * 0.25
        rotation = np.asarray(data.xmat[self.base_body_id], dtype=np.float64).reshape(3, 3)
        projected_gravity = rotation.T @ np.asarray((0.0, 0.0, -1.0))
        joint_positions = (
            np.asarray(data.qpos[self.joint_qpos_addresses]) - self.default_joint_positions
        )
        leg_joint_positions = joint_positions[:6]
        joint_velocities = (
            np.clip(np.asarray(data.qvel[self.joint_dof_addresses]), -100.0, 100.0)
            * 0.05
        )
        return np.concatenate(
            (
                base_angular_velocity,
                projected_gravity,
                leg_joint_positions,
                joint_velocities,
                np.clip(self.last_action, -100.0, 100.0),
            )
        ).astype(np.float32, copy=False)

    def _history_observation(
        self, data: mujoco.MjData, gait_command: np.ndarray
    ) -> np.ndarray:
        history_observation = np.concatenate(
            (
                self._common_observation(data),
                self._gait_phase(float(gait_command[0])),
                gait_command,
            )
        ).astype(np.float32, copy=False)
        if history_observation.size != HISTORY_OBS_DIM:
            raise RuntimeError(
                f"历史观测维度为 {history_observation.size}，预期 {HISTORY_OBS_DIM}。"
            )
        return history_observation

    def act(
        self, data: mujoco.MjData, command: np.ndarray, gait_command: np.ndarray
    ) -> np.ndarray:
        common_observation = self._common_observation(data)
        gait_phase = self._gait_phase(float(gait_command[0]))
        history_observation = np.concatenate(
            (common_observation, gait_phase, gait_command)
        ).astype(np.float32, copy=False)
        self.history.append(history_observation)
        history = np.concatenate(tuple(self.history)).astype(np.float32, copy=False)
        policy_observation = np.concatenate(
            (
                common_observation,
                self.scanner.scan(data),
                gait_phase,
                gait_command,
            )
        ).astype(np.float32, copy=False)
        if policy_observation.size != POLICY_OBS_DIM:
            raise RuntimeError(
                f"策略观测维度为 {policy_observation.size}，预期 {POLICY_OBS_DIM}。"
            )

        with torch.inference_mode():
            history_tensor = torch.from_numpy(history).unsqueeze(0).to(self.device)
            policy_tensor = torch.from_numpy(policy_observation).unsqueeze(0).to(self.device)
            command_tensor = torch.from_numpy(command).unsqueeze(0).to(self.device)
            latent = self.encoder(history_tensor)
            action = self.actor(torch.cat((latent, policy_tensor, command_tensor), dim=-1))
        result = action[0].detach().cpu().numpy().astype(np.float32, copy=False)
        if not np.all(np.isfinite(result)):
            raise RuntimeError("策略输出包含 NaN 或无穷值。")
        self.last_action[:] = result
        self.policy_step += 1
        return result


class TorqueController:
    """Convert Isaac Lab position/velocity actions to direct MuJoCo torques."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        self.leg_qpos_addresses = np.asarray(
            [
                model.jnt_qposadr[
                    _model_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
                for name in LEG_JOINT_NAMES
            ]
        )
        self.leg_dof_addresses = np.asarray(
            [
                model.jnt_dofadr[
                    _model_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
                for name in LEG_JOINT_NAMES
            ]
        )
        self.wheel_dof_addresses = np.asarray(
            [
                model.jnt_dofadr[
                    _model_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
                for name in WHEEL_JOINT_NAMES
            ]
        )
        self.leg_actuator_ids = np.asarray(
            [
                _model_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in LEG_JOINT_NAMES
            ]
        )
        self.wheel_actuator_ids = np.asarray(
            [
                _model_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in WHEEL_JOINT_NAMES
            ]
        )
        self.default_leg_positions = np.asarray(
            model.qpos0[self.leg_qpos_addresses], dtype=np.float64
        ).copy()

    def apply(self, data: mujoco.MjData, action: np.ndarray) -> float:
        leg_targets = self.default_leg_positions + 0.25 * action[:6]
        leg_torque = 40.0 * (leg_targets - data.qpos[self.leg_qpos_addresses])
        leg_torque -= 2.5 * data.qvel[self.leg_dof_addresses]
        wheel_torque = 0.8 * (action[6:] - data.qvel[self.wheel_dof_addresses])
        torque = np.clip(np.concatenate((leg_torque, wheel_torque)), -80.0, 80.0)
        data.ctrl.fill(0.0)
        data.ctrl[self.leg_actuator_ids] = torque[:6]
        data.ctrl[self.wheel_actuator_ids] = torque[6:]
        return float(np.linalg.norm(torque))


class KeyboardCommandState:
    _MOTION_KEYS = {
        glfw.KEY_W,
        glfw.KEY_S,
        glfw.KEY_Q,
        glfw.KEY_E,
        glfw.KEY_Z,
        glfw.KEY_C,
        glfw.KEY_A,
        glfw.KEY_D,
    }

    def __init__(self, body_height: float) -> None:
        self.pressed: set[int] = set()
        self.body_height = body_height

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

    def clear_motion(self) -> None:
        self.pressed.clear()

    def sync_from_window(self, window: glfw._GLFWwindow) -> None:
        self.pressed = {
            key for key in self._MOTION_KEYS if glfw.get_key(window, key) == glfw.PRESS
        }

    def command(
        self,
        forward_speed: float,
        lateral_speed: float,
        yaw_speed: float,
        height_rate: float,
    ) -> np.ndarray:
        x_semantic = int(glfw.KEY_W in self.pressed) - int(glfw.KEY_S in self.pressed)
        y_semantic = int(glfw.KEY_Q in self.pressed) - int(glfw.KEY_E in self.pressed)
        z_semantic = int(glfw.KEY_Z in self.pressed) - int(glfw.KEY_C in self.pressed)
        yaw_semantic = int(glfw.KEY_A in self.pressed) - int(glfw.KEY_D in self.pressed)
        self.body_height = float(
            np.clip(
                self.body_height + z_semantic * height_rate * POLICY_DT,
                0.75,
                0.85,
            )
        )
        return np.asarray(
            (
                x_semantic * forward_speed,
                y_semantic * lateral_speed,
                yaw_semantic * yaw_speed,
                self.body_height,
            ),
            dtype=np.float32,
        )


def reset_robot(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    region: TerrainRegion,
    rng: np.random.Generator,
    spawn_z_offset: float,
) -> tuple[np.ndarray, float]:
    mujoco.mj_resetData(model, data)
    spawn_local = np.asarray(
        (rng.uniform(-5.0, -4.0), rng.uniform(-5.0, 5.0)), dtype=np.float64
    )
    spawn_xy = np.asarray(region.center_xy_w, dtype=np.float64) + spawn_local
    yaw = float(rng.uniform(-0.35, 0.35))
    half_yaw = 0.5 * yaw
    data.qpos[0:2] = spawn_xy
    data.qpos[2] = model.qpos0[2] + spawn_z_offset
    data.qpos[3:7] = (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))
    mujoco.mj_forward(model, data)
    return spawn_xy, yaw


def _print_help(
    checkpoint: Path, region: TerrainRegion, spawn_xy: np.ndarray, yaw: float
) -> None:
    print(f"[policy] checkpoint={checkpoint}")
    print(
        f"[terrain] type={region.terrain_type}, level={region.terrain_level}, "
        f"number={region.number}, center={region.center_xy_w}"
    )
    print(
        f"[spawn] position=({spawn_xy[0]:.3f}, {spawn_xy[1]:.3f}), "
        f"yaw={yaw:+.3f} rad"
    )
    print("  W/S : 前进/后退")
    print("  Q/E : 左移/右移（该轮式策略训练时横向指令为 0，效果可能不稳定）")
    print("  Z/C : 升高/降低机身")
    print("  A/D : 左转/右转")
    print("  Space : 清除运动指令")
    print("  R : 重置，并在终端选择复用或重新选择地形")
    print("  Esc : 退出")


def _camera_yaw(data: mujoco.MjData, base_body_id: int) -> float:
    rotation = np.asarray(data.xmat[base_body_id], dtype=np.float64).reshape(3, 3)
    return math.atan2(rotation[1, 0], rotation[0, 0])


def _render(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    window: glfw._GLFWwindow,
    camera: mujoco.MjvCamera,
    option: mujoco.MjvOption,
    scene: mujoco.MjvScene,
    context: mujoco.MjrContext,
    base_body_id: int,
    command: np.ndarray,
    region: TerrainRegion,
    torque_norm: float,
) -> None:
    # MuJoCo azimuth 45 deg places the camera at local (-x, -y): right rear.
    yaw = _camera_yaw(data, base_body_id)
    camera.lookat[:] = data.xpos[base_body_id]
    camera.lookat[2] += 0.10
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
        "A/D yaw            Z/C body height\n"
        "Space stop         R reset/select terrain   Esc quit"
    )
    right_text = (
        f"terrain {region.terrain_type}/{region.terrain_level}/{region.number}\n"
        f"cmd [{command[0]:+.2f}, {command[1]:+.2f}, {command[2]:+.2f}]  "
        f"height {command[3]:.3f}\n"
        f"t {data.time:.2f}  torque norm {torque_norm:.2f}"
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
        description="Deploy the latest Isaac Lab TRON1A wheel-mode policy in MuJoCo."
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--terrain-xml", type=Path, default=DEFAULT_TERRAIN_XML)
    parser.add_argument("--terrain-manifest", type=Path, default=DEFAULT_TERRAIN_MANIFEST)
    parser.add_argument("--robot-xml", type=Path, default=DEFAULT_ROBOT_XML)
    parser.add_argument("--terrain-type", type=int, choices=range(1, 9), default=None)
    parser.add_argument("--terrain-level", type=int, choices=range(1, 5), default=None)
    parser.add_argument("--terrain-number", type=int, choices=range(1, 5), default=None)
    parser.add_argument("--device", default="auto", help="cpu, cuda:0, or auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--forward-speed", type=float, default=0.8)
    parser.add_argument("--lateral-speed", type=float, default=0.3)
    parser.add_argument("--yaw-speed", type=float, default=0.6)
    parser.add_argument("--body-height", type=float, default=0.80)
    parser.add_argument("--height-rate", type=float, default=0.08)
    parser.add_argument("--gait-frequency", type=float, default=1.7)
    parser.add_argument("--gait-offset", type=float, default=0.5)
    parser.add_argument("--gait-duration", type=float, default=0.525)
    parser.add_argument("--swing-height", type=float, default=0.14)
    parser.add_argument("--spawn-z-offset", type=float, default=0.1)
    parser.add_argument("--render-hz", type=float, default=60.0)
    parser.add_argument("--camera-distance", type=float, default=3.5)
    parser.add_argument("--camera-elevation", type=float, default=-18.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument(
        "--max-steps", type=int, default=None, help="Optional number of physics steps."
    )
    args = parser.parse_args(argv)
    if not 0.75 <= args.body_height <= 0.85:
        parser.error("--body-height 必须在 [0.75, 0.85] 内。")
    if args.render_hz <= 0.0:
        parser.error("--render-hz 必须大于 0。")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps 必须大于 0。")
    if args.terrain_type is None and (
        args.terrain_level is not None or args.terrain_number is not None
    ):
        parser.error("使用 --terrain-level/--terrain-number 时必须同时提供 --terrain-type。")
    return args


def _select_region_from_args(
    args: argparse.Namespace, regions: Sequence[TerrainRegion]
) -> TerrainRegion:
    if args.terrain_type is None:
        return select_terrain_region_from_terminal(regions)
    terrain_level = (
        args.terrain_level
        if args.terrain_level is not None
        else (4 if args.terrain_type >= 7 else 1)
    )
    terrain_number = args.terrain_number if args.terrain_number is not None else 1
    region = find_terrain_region(
        regions, args.terrain_type, terrain_level, terrain_number
    )
    _print_selected_region(region)
    return region


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    regions = load_terrain_regions(args.terrain_manifest.expanduser().resolve())
    region = _select_region_from_args(args, regions)
    checkpoint = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else discover_latest_checkpoint(args.checkpoint_root.expanduser().resolve())
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint 不存在：{checkpoint}")
    device = resolve_device(args.device)

    model = build_scene_model(
        args.terrain_xml.expanduser().resolve(),
        args.robot_xml.expanduser().resolve(),
        PHYSICS_DT,
    )
    data = mujoco.MjData(model)
    base_body_id = _model_id(model, mujoco.mjtObj.mjOBJ_BODY, "base_Link")
    scanner = TerrainHeightScanner(model, base_body_id)
    policy = WheelPolicy(checkpoint, device, model, scanner)
    torque_controller = TorqueController(model)
    rng = np.random.default_rng(args.seed)
    gait_command = np.asarray(
        (
            args.gait_frequency,
            args.gait_offset,
            args.gait_duration,
            args.swing_height,
        ),
        dtype=np.float32,
    )
    command_state = KeyboardCommandState(args.body_height)
    spawn_xy, spawn_yaw = reset_robot(
        model, data, region, rng, args.spawn_z_offset
    )
    policy.reset(data, gait_command)
    _print_help(checkpoint, region, spawn_xy, spawn_yaw)
    print(
        f"[runtime] checkpoint_iter={policy.iteration}, device={device}, "
        f"physics_dt={PHYSICS_DT}, policy_dt={POLICY_DT}"
    )

    window = None
    camera = None
    option = None
    scene = None
    context = None
    reset_requested = False
    if not args.headless:
        if not glfw.init():
            raise RuntimeError("GLFW 初始化失败；无显示环境可使用 --headless。")
        window = glfw.create_window(
            1200, 900, "TRON1A MuJoCo Wheel Policy", None, None
        )
        if window is None:
            glfw.terminate()
            raise RuntimeError("GLFW 窗口创建失败；无显示环境可使用 --headless。")
        glfw.make_context_current(window)
        glfw.swap_interval(1)
        camera = mujoco.MjvCamera()
        option = mujoco.MjvOption()
        scene = mujoco.MjvScene(model, maxgeom=max(2000, model.ngeom + 100))
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
    command = np.asarray((0.0, 0.0, 0.0, args.body_height), dtype=np.float32)
    torque_norm = 0.0
    next_render_time = float(data.time)
    next_status_time = float(data.time)
    wall_start = time.perf_counter()
    sim_start = float(data.time)
    try:
        while True:
            if window is not None and glfw.window_should_close(window):
                break
            if args.max_steps is not None and physics_steps >= args.max_steps:
                break

            if reset_requested:
                command_state.clear_motion()
                region = select_or_reuse_terrain_region_on_reset(region, regions)
                command_state.body_height = args.body_height
                spawn_xy, spawn_yaw = reset_robot(
                    model, data, region, rng, args.spawn_z_offset
                )
                policy.reset(data, gait_command)
                action.fill(0.0)
                episode_physics_steps = 0
                reset_requested = False
                next_render_time = float(data.time)
                next_status_time = float(data.time)
                wall_start = time.perf_counter()
                sim_start = float(data.time)
                print(
                    f"[reset] terrain={region.terrain_type}/{region.terrain_level}/{region.number}, "
                    f"spawn=({spawn_xy[0]:.3f}, {spawn_xy[1]:.3f}), "
                    f"yaw={spawn_yaw:+.3f}"
                )

            if episode_physics_steps % POLICY_DECIMATION == 0:
                command = command_state.command(
                    args.forward_speed,
                    args.lateral_speed,
                    args.yaw_speed,
                    args.height_rate,
                )
                action = policy.act(data, command, gait_command)
            torque_norm = torque_controller.apply(data, action)
            mujoco.mj_step(model, data)
            physics_steps += 1
            episode_physics_steps += 1

            if data.time + 1.0e-12 >= next_status_time:
                print(
                    f"[status] t={data.time:7.3f} base_z={data.qpos[2]:.3f} "
                    f"cmd=({command[0]:+.2f},{command[1]:+.2f},{command[2]:+.2f},"
                    f"h={command[3]:.3f}) torque_norm={torque_norm:.2f}"
                )
                next_status_time += 1.0

            if window is not None and data.time + 1.0e-12 >= next_render_time:
                assert camera is not None
                assert option is not None
                assert scene is not None
                assert context is not None
                _render(
                    model,
                    data,
                    window,
                    camera,
                    option,
                    scene,
                    context,
                    base_body_id,
                    command,
                    region,
                    torque_norm,
                )
                command_state.sync_from_window(window)
                while next_render_time <= data.time + 1.0e-12:
                    next_render_time += 1.0 / args.render_hz

            if not args.no_realtime:
                target_wall_time = wall_start + (float(data.time) - sim_start)
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
