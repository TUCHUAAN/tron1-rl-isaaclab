#!/usr/bin/env python3
"""Deploy a TRON1A wheel or foot expert policy in MuJoCo."""

from __future__ import annotations

import argparse
import importlib.util
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
import yaml
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TERRAIN_XML = Path("/home/tuchuaan/UDMMR/models/terrains/training_terrain.xml")
DEFAULT_TERRAIN_MANIFEST = Path("/home/tuchuaan/UDMMR/models/terrains/training_terrain_manifest.json")
DEFAULT_ROBOT_XML = Path(
    "/home/tuchuaan/tron1-mujoco-sim/robot-description/pointfoot/WF_TRON1A/xml/robot.xml"
)
DEFAULT_WHEEL_CHECKPOINT_ROOT = REPO_ROOT / "logs" / "rsl_rl" / "wf_tron_1a_wheel_mode"
DEFAULT_FOOT_CHECKPOINT_ROOT = (
    REPO_ROOT / "logs" / "rsl_rl" / "wf_tron_1a_foot_all_terrain"
)
STABILITY_SELECTION_FILE = "selected_stability_checkpoint.txt"
CONTROL_MODES = ("wheel", "foot", "dual")

PHYSICS_DT = 0.005
POLICY_DECIMATION = 4
POLICY_DT = PHYSICS_DT * POLICY_DECIMATION
HISTORY_LENGTH = 10
POLICY_OBS_DIM = 155
HISTORY_OBS_DIM = 34
COMMAND_DIM = 4
ACTION_DIM = 8
BODY_HEIGHT_MIN = 0.65
BODY_HEIGHT_MAX = 0.85

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


def _fit_height_plane_numpy(
    points_w: np.ndarray, eps: float = 1.0e-6
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Fit the same ``z=a*x+b*y+c`` local terrain plane used in training."""
    finite = np.isfinite(points_w).all(axis=1)
    valid_points = points_w[finite]
    fallback_centroid = np.zeros(3, dtype=np.float64)
    fallback_normal = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    if valid_points.shape[0] < 3:
        return fallback_centroid, fallback_normal, False

    centroid = valid_points.mean(axis=0)
    centered = valid_points - centroid
    x, y, z = centered.T
    s_xx = float(np.mean(x * x))
    s_xy = float(np.mean(x * y))
    s_yy = float(np.mean(y * y))
    s_xz = float(np.mean(x * z))
    s_yz = float(np.mean(y * z))
    determinant = s_xx * s_yy - s_xy * s_xy
    if not math.isfinite(determinant) or abs(determinant) <= eps:
        return fallback_centroid, fallback_normal, False

    slope_x = (s_xz * s_yy - s_yz * s_xy) / determinant
    slope_y = (s_yz * s_xx - s_xz * s_xy) / determinant
    normal = np.asarray((-slope_x, -slope_y, 1.0), dtype=np.float64)
    normal_norm = float(np.linalg.norm(normal))
    if not np.isfinite(normal).all() or normal_norm <= eps:
        return fallback_centroid, fallback_normal, False
    return centroid, normal / normal_norm, True


def _roll_pitch_degrees(rotation_wb: np.ndarray) -> tuple[float, float]:
    """Return world-frame ZYX roll and pitch from a body-to-world matrix."""
    pitch = math.atan2(
        -float(rotation_wb[2, 0]),
        math.hypot(float(rotation_wb[0, 0]), float(rotation_wb[1, 0])),
    )
    roll = math.atan2(float(rotation_wb[2, 1]), float(rotation_wb[2, 2]))
    return math.degrees(roll), math.degrees(pitch)


def _refresh_body_kinematics(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Refresh poses and COM velocities from qpos/qvel without solving dynamics.

    mj_step integrates qpos/qvel after computing these derived quantities.
    Refresh only the kinematic stages so observation queries do not advance
    time or overwrite accelerations and the constraint solver's warm start.
    """
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_comVel(model, data)


def _base_velocity_in_body_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    base_body_id: int,
    result: np.ndarray,
) -> None:
    """Write COM angular/linear velocity expressed in the base link's axes.

    mjOBJ_BODY with flg_local=1 uses the principal inertia axes (ximat),
    which can be rotated relative to the link frame used by Isaac Lab.
    Keep the COM reference point and rotate world velocities with xmat.
    """
    _refresh_body_kinematics(model, data)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, base_body_id, result, 0)
    rotation_wb = np.asarray(data.xmat[base_body_id]).reshape(3, 3)
    result[:3] = rotation_wb.T @ result[:3]
    result[3:] = rotation_wb.T @ result[3:]


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
    """Return the newest checkpoint file by filesystem modification time.

    Dual-mode MuJoCo debugging intentionally uses the newest Wheel and Foot
    snapshots, rather than a manually written stability-selection file.
    """
    candidates = tuple(checkpoint_root.glob("**/model_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"没有在 {checkpoint_root} 中找到 model_*.pt")
    selected = max(candidates, key=lambda path: (path.stat().st_mtime_ns, str(path)))
    print(f"使用最新 checkpoint：{selected}")
    return selected


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


def checkpoint_body_height_range(checkpoint_path: Path) -> tuple[float, float]:
    """Read the training height range saved beside a checkpoint when available."""
    env_config = checkpoint_path.parent / "params" / "env.yaml"
    if not env_config.is_file():
        return BODY_HEIGHT_MIN, BODY_HEIGHT_MAX
    try:
        text = env_config.read_text(encoding="utf-8")
    except OSError:
        return BODY_HEIGHT_MIN, BODY_HEIGHT_MAX

    commands_index = text.find("\ncommands:\n")
    body_height_index = text.find("\n  body_height:\n", max(commands_index, 0))
    if body_height_index < 0:
        return BODY_HEIGHT_MIN, BODY_HEIGHT_MAX
    gait_index = text.find("\n  gait_command:\n", body_height_index)
    section = text[body_height_index : gait_index if gait_index >= 0 else None]
    match = re.search(
        r"\n\s+ranges:\n\s+height:.*?\n\s+-\s+([-+0-9.eE]+)\n\s+-\s+([-+0-9.eE]+)",
        section,
        flags=re.DOTALL,
    )
    if match is None:
        return BODY_HEIGHT_MIN, BODY_HEIGHT_MAX
    minimum, maximum = (float(match.group(1)), float(match.group(2)))
    if not math.isfinite(minimum) or not math.isfinite(maximum) or maximum <= minimum:
        raise ValueError(f"checkpoint 高度范围无效：[{minimum}, {maximum}]")
    return minimum, maximum


def checkpoint_continuous_gait_phase(checkpoint_path: Path) -> bool:
    """Read the clock convention saved with the model; older models use time*f."""
    env_config = checkpoint_path.parent / "params" / "env.yaml"
    if not env_config.is_file():
        return False
    try:
        config = yaml.load(env_config.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
        value = config.get("commands", {}).get("gait_command", {}).get("continuous_phase", "false")
    except (OSError, yaml.YAMLError, AttributeError) as error:
        raise ValueError(f"无法读取 checkpoint 步态时钟配置：{env_config}") from error
    if str(value).lower() not in ("true", "false"):
        raise ValueError(f"continuous_phase 必须为 true/false，实际为 {value!r}")
    return str(value).lower() == "true"


def resolve_gait_command(args: argparse.Namespace, checkpoint_path: Path) -> np.ndarray:
    """Resolve Foot defaults from its training snapshot, preserving CLI overrides."""
    fields = (
        ("gait_frequency", "frequencies"),
        ("gait_offset", "offsets"),
        ("gait_duration", "durations"),
        ("swing_height", "swing_height"),
    )
    defaults = [1.7, 0.5, 0.5, 0.0] if args.mode == "foot" else [1.7, 0.5, 0.525, 0.14]
    env_config = checkpoint_path.parent / "params" / "env.yaml"
    use_snapshot = args.mode == "foot" and any(
        getattr(args, field) is None for field, _ in fields
    )
    if use_snapshot and env_config.is_file():
        try:
            # BaseLoader reads Isaac Lab's Python-tagged YAML as plain data;
            # loading gait ranges must not instantiate training classes.
            config = yaml.load(env_config.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
            ranges = config["commands"]["gait_command"]["ranges"]
            for index, (field, range_name) in enumerate(fields):
                if getattr(args, field) is not None:
                    continue
                bounds = ranges[range_name]
                if not isinstance(bounds, list) or len(bounds) != 2:
                    raise ValueError(f"{range_name} 必须包含两个边界值")
                minimum, maximum = map(float, bounds)
                if not math.isfinite(minimum) or not math.isfinite(maximum) or maximum < minimum:
                    raise ValueError(f"{range_name} 范围无效：{bounds}")
                defaults[index] = 0.5 * minimum + 0.5 * maximum
        except (OSError, yaml.YAMLError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"无法读取 Foot checkpoint 的步态范围：{env_config}：{exc}") from exc
    elif use_snapshot:
        print("[gait] 未找到 params/env.yaml，使用 Foot 默认步态 (1.7, 0.5, 0.5, 0.0)。")

    values = [
        default if getattr(args, field) is None else getattr(args, field)
        for (field, _), default in zip(fields, defaults)
    ]
    if not all(math.isfinite(value) for value in values) or values[0] <= 0.0:
        raise ValueError("步态参数必须是有限数，且 gait frequency 必须大于 0。")
    return np.asarray(values, dtype=np.float32)


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
        self.last_plane_centroid = np.zeros(3, dtype=np.float64)
        self.last_plane_normal = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
        self.last_plane_valid = False
        self.last_terrain_points = np.full((len(self.offsets), 3), np.nan)

    def scan(self, data: mujoco.MjData) -> np.ndarray:
        _refresh_body_kinematics(self.model, data)
        base_position = np.asarray(data.xpos[self.base_body_id], dtype=np.float64)
        rotation = np.asarray(data.xmat[self.base_body_id], dtype=np.float64).reshape(3, 3)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        heights = np.empty(self.offsets.shape[0], dtype=np.float32)
        terrain_points = np.full((self.offsets.shape[0], 3), np.nan, dtype=np.float64)
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
            if distance < 0.0:
                # Isaac Lab encodes missing hits as +inf positions, so its
                # clipped (sensor_z - hit_z - 0.5) observation becomes zero.
                # Leave terrain_points invalid for the plane fit below.
                heights[index] = 0.0
            else:
                heights[index] = np.clip(distance - 0.5, 0.0, 10.0)
                terrain_points[index] = ray_origin + distance * self.ray_direction
        (
            self.last_plane_centroid,
            self.last_plane_normal,
            self.last_plane_valid,
        ) = _fit_height_plane_numpy(terrain_points)
        self.last_terrain_points = terrain_points
        return heights

    def body_height(self, data: mujoco.MjData) -> float:
        """Return base distance along the latest fitted local terrain normal."""
        if not self.last_plane_valid:
            return math.nan
        base_position = np.asarray(data.xpos[self.base_body_id], dtype=np.float64)
        return abs(
            float(
                np.dot(
                    base_position - self.last_plane_centroid,
                    self.last_plane_normal,
                )
            )
        )

    def attitude_reference(self, data: mujoco.MjData) -> tuple[float, float]:
        """Return roll/pitch that align base-up with the local terrain normal."""
        if not self.last_plane_valid:
            return math.nan, math.nan
        rotation = np.asarray(data.xmat[self.base_body_id], dtype=np.float64).reshape(3, 3)
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
        normal = self.last_plane_normal
        horizontal_heading = np.asarray((math.cos(yaw), math.sin(yaw), 0.0))
        forward = horizontal_heading - np.dot(horizontal_heading, normal) * normal
        forward_norm = float(np.linalg.norm(forward))
        if forward_norm <= 1.0e-6:
            return math.nan, math.nan
        forward /= forward_norm
        left = np.cross(normal, forward)
        left_norm = float(np.linalg.norm(left))
        if left_norm <= 1.0e-6:
            return math.nan, math.nan
        left /= left_norm
        reference_rotation = np.column_stack((forward, left, normal))
        return _roll_pitch_degrees(reference_rotation)


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


def _checkpoint_hold_tracker(checkpoint_path: Path):
    """New 161D policies require the saved hold configuration; no guessed defaults."""
    config_path = checkpoint_path.parent / "params" / "env.yaml"
    if not config_path.is_file():
        raise ValueError(f"零命令保持观测模型需要配置快照：{config_path}")
    config = yaml.load(config_path.read_text(), Loader=yaml.BaseLoader)
    observation = config.get('observations', {}).get('policy', {}).get('zero_command_hold')
    if not isinstance(observation, dict) or not str(observation.get('func', '')).endswith('zero_command_hold_observation'):
        raise ValueError('161D policy 缺少有效的 zero_command_hold 观测声明。')
    reward = config.get('rewards', {}).get('pen_zero_command_hold')
    if not isinstance(reward, dict) or not float(reward.get('weight', 0)):
        raise ValueError('零命令保持观测需要启用 pen_zero_command_hold。')
    options = {key: float(value) for key, value in reward.get('params', {}).get('tracker_options', {}).items()}
    path = Path(__file__).resolve().parents[1] / 'exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/reward_math.py'
    spec = importlib.util.spec_from_file_location('mujoco_zero_hold_math', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ZeroCommandHoldTracker(1, 'cpu', POLICY_DT, **options)


class WheelPolicy:
    """Checkpoint loader plus the exact actor and encoder observation schema."""

    def __init__(
        self,
        checkpoint_path: Path,
        device: torch.device,
        model: mujoco.MjModel,
        scanner: TerrainHeightScanner,
        body_height_range: tuple[float, float],
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model = model
        self.scanner = scanner
        self.body_height_min, self.body_height_max = body_height_range
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
        self.policy_obs_dim = actor_input_dim - encoder_output_dim - COMMAND_DIM
        if self.policy_obs_dim not in (POLICY_OBS_DIM, POLICY_OBS_DIM + 6):
            raise ValueError(f"不支持的策略观测维度：{self.policy_obs_dim}，预期 155 或 161。")
        self.hold_tracker = (_checkpoint_hold_tracker(checkpoint_path)
                             if self.policy_obs_dim == POLICY_OBS_DIM + 6 else None)
        self._hold_started = False
        self._hold_previous_command = np.zeros(3, dtype=np.float32)
        expected_actor_input = encoder_output_dim + self.policy_obs_dim + COMMAND_DIM
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
        self.continuous_gait_phase = checkpoint_continuous_gait_phase(checkpoint_path)
        self._phase_cycles = 0.0

    def reset(self, data: mujoco.MjData, gait_command: np.ndarray) -> None:
        self.last_action.fill(0.0)
        if getattr(self, 'hold_tracker', None) is not None:
            self.hold_tracker.reset()
            self._hold_started = False
            self._hold_previous_command.fill(0.)
        self.policy_step = 0
        self._phase_cycles = 0.0
        history_row = self._history_observation(data, gait_command)
        self.history.clear()
        self.history.extend(history_row.copy() for _ in range(HISTORY_LENGTH))

    def _gait_phase(self, frequency: float) -> np.ndarray:
        phase = (
            self._phase_cycles if self.continuous_gait_phase
            else (self.policy_step * POLICY_DT * frequency) % 1.0
        )
        angle = 2.0 * math.pi * phase
        return np.asarray((math.sin(angle), math.cos(angle)), dtype=np.float32)

    def _common_observation(self, data: mujoco.MjData) -> np.ndarray:
        _base_velocity_in_body_frame(
            self.model,
            data,
            self.base_body_id,
            self.velocity_buffer,
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

    def _hold_observation(self, data, command):
        tracker = self.hold_tracker
        pos = torch.tensor(np.asarray(data.xpos[self.base_body_id, :2]).copy(), dtype=torch.float32)[None]
        yaw = torch.tensor([_camera_yaw(data, self.base_body_id)], dtype=torch.float32)
        if self._hold_started:
            tracker.update(pos, yaw, torch.from_numpy(self._hold_previous_command.copy())[None])
        else:
            tracker.prime(pos, yaw)
            self._hold_started = True
        # The just-finished interval used the previous command. Apply the new
        # keyboard command after that update, matching Isaac reward->command->obs.
        tracker.synchronize_command(torch.tensor(command[:3], dtype=torch.float32)[None])
        self._hold_previous_command[:] = command[:3]
        return tracker.observation()[0].numpy().copy()

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
        if getattr(self, 'hold_tracker', None) is not None:
            policy_observation = np.concatenate((policy_observation, self._hold_observation(data, command)))
        expected_dim = getattr(self, 'policy_obs_dim', POLICY_OBS_DIM)
        if policy_observation.size != expected_dim:
            raise RuntimeError(
                f"策略观测维度为 {policy_observation.size}，预期 {expected_dim}。"
            )

        network_command = command.copy()
        network_command[3] = np.clip(
            (network_command[3] - self.body_height_min)
            / (self.body_height_max - self.body_height_min),
            0.0,
            1.0,
        )
        with torch.inference_mode():
            history_tensor = torch.from_numpy(history).unsqueeze(0).to(self.device)
            policy_tensor = torch.from_numpy(policy_observation).unsqueeze(0).to(self.device)
            command_tensor = torch.from_numpy(network_command).unsqueeze(0).to(self.device)
            latent = self.encoder(history_tensor)
            action = self.actor(torch.cat((latent, policy_tensor, command_tensor), dim=-1))
        result = action[0].detach().cpu().numpy().astype(np.float32, copy=False)
        if not np.all(np.isfinite(result)):
            raise RuntimeError("策略输出包含 NaN 或无穷值。")
        self.last_action[:] = result
        self.policy_step += 1
        if self.continuous_gait_phase:
            # This action will execute for one interval at the supplied
            # frequency. A new frequency on the next call cannot rewrite it.
            self._phase_cycles = (self._phase_cycles + POLICY_DT * float(gait_command[0])) % 1.0
        return result


class FootholdDebug:
    """Optional view of the same frozen-region/event algorithm used in training."""
    def __init__(self, model, base_body_id, scanner, checkpoint):
        source = REPO_ROOT / "exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/reward_math.py"
        spec = importlib.util.spec_from_file_location("foothold_reward_math", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.model, self.base_body_id, self.scanner = model, base_body_id, scanner
        self.body_ids = [_model_id(model, mujoco.mjtObj.mjOBJ_BODY, f"wheel_{side}_Link") for side in ("L", "R")]
        options = {}
        path = checkpoint.parent / "params/env.yaml"
        if path.is_file():
            cfg = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
            term = cfg.get("rewards", {}).get("pen_foothold_region")
            if isinstance(term, dict):
                saved = term.get("params", {}).get("tracker_options", {})
                options = {k: tuple(map(float,v)) if k == "axes" else float(v) for k,v in saved.items()}
        self.tracker = module.FootholdRegionTracker(1, "cpu", **options)
        self.radius = self.tracker.plan_options.get("wheel_radius", .128)
        self.velocity = np.zeros(6)
        self.last_rho = np.zeros(2)
        print("[foothold] cyan=L, orange=R ellipse; green/red=last landing inside/outside. "
              + ("Using checkpoint region parameters." if options else "No saved region config; previewing current defaults."))

    def reset(self):
        self.tracker.reset()
        self.last_rho.fill(0.)

    def update(self, data, command, gait, phase):
        self.scanner.scan(data)
        force_vectors = np.zeros((2,3))
        force = np.zeros(6)
        for index in range(data.ncon):
            contact = data.contact[index]
            geoms = (contact.geom1, contact.geom2)
            for side, body_id in enumerate(self.body_ids):
                for which in (0,1):
                    if (self.model.geom_bodyid[geoms[which]] == body_id and
                            self.model.geom_group[geoms[1-which]] == 0):
                        mujoco.mj_contactForce(self.model, data, index, force)
                        force_vectors[side] += (1. if which == 1 else -1.) * contact.frame.reshape(3,3).T @ force[:3]
        positions = np.asarray(data.xpos[self.body_ids]).copy()
        # Fit the same 3x3 wheel-local patches used by the training foot geometry.
        clearance = np.full(2, np.nan)
        for side, position in enumerate(positions):
            hits=[]
            for dx in (-.04,0.,.04):
                for dy in (-.04,0.,.04):
                    origin=np.array([position[0]+dx,position[1]+dy,position[2]+.3])
                    distance=mujoco.mj_ray(self.model,data,origin,self.scanner.ray_direction,
                        self.scanner.geom_group,True,-1,self.scanner.geom_id)
                    hits.append(origin+distance*self.scanner.ray_direction if distance >= 0 else [np.nan]*3)
            centroid, normal, valid = _fit_height_plane_numpy(np.asarray(hits))
            if valid:
                clearance[side] = (position-centroid)@normal-self.radius
        _base_velocity_in_body_frame(self.model,data,self.base_body_id,self.velocity)
        tensor = lambda value: torch.as_tensor(np.asarray(value).copy(), dtype=torch.float32).unsqueeze(0)
        self.tracker.update(tensor(data.xpos[self.base_body_id]), tensor(_camera_yaw(data,self.base_body_id)),
            tensor(self.velocity[3:]), tensor(command[:3]), tensor(gait), tensor(phase),
            tensor(self.scanner.last_terrain_points), tensor(positions),
            tensor(np.linalg.norm(force_vectors,axis=1)), tensor(clearance), POLICY_DT)
        for side in range(2):
            if self.tracker.events[0,side]:
                self.last_rho[side] = float(self.tracker.rho[0,side])
                print(f"[foothold] t={data.time:.2f} side={'LR'[side]} rho={self.last_rho[side]:.3f} "
                      f"inside={self.last_rho[side] <= 1.}")

    def add_geoms(self, scene):
        tracker = self.tracker
        def sphere(position, color, size=.008):
            if scene.ngeom >= scene.maxgeom: return
            geom=scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([size]*3),position,np.eye(3).ravel(),np.array(color,dtype=np.float32))
            scene.ngeom += 1
        for side,color in enumerate(((.1,.8,1.,1.),(1.,.55,.1,1.))):
            if tracker.valid[0,side] and tracker.initialized[0,side]:
                center=tracker.center[0,side].numpy()
                normal=tracker.normal[0,side].numpy()
                x=tracker.tangent_x[0,side].numpy(); y=tracker.tangent_y[0,side].numpy()
                theta=np.linspace(0.,2*np.pi,33)
                points=center+.004*normal+tracker.axes[0]*np.cos(theta)[:,None]*x+tracker.axes[1]*np.sin(theta)[:,None]*y
                sphere(center+.004*normal,color)
                for start,end in zip(points[:-1],points[1:]):
                    if scene.ngeom >= scene.maxgeom: break
                    geom=scene.geoms[scene.ngeom]
                    mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array(color,dtype=np.float32))
                    mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,.002,start,end)
                    scene.ngeom += 1
            if tracker.last_landing_valid[0,side]:
                position=tracker.last_landing[0,side].numpy()-self.radius*tracker.last_landing_normal[0,side].numpy()
                sphere(position, (0.,1.,.2,1.) if self.last_rho[side] <= 1. else (1.,0.,0.,1.), .012)


class TorqueController:
    """Convert Isaac Lab actions to MuJoCo torques using wheel PI control."""

    def __init__(
        self,
        model: mujoco.MjModel,
        control_mode: str,
        *,
        wheel_kp: float = 2.0,
        wheel_ki: float = 0.5,
        low_pass_enabled: bool = False,
        cutoff_hz: float = 20.0,
        control_dt: float = PHYSICS_DT,
    ) -> None:
        if control_mode not in CONTROL_MODES:
            raise ValueError(f"不支持的控制模式：{control_mode}")
        if not math.isfinite(wheel_kp) or wheel_kp <= 0.0:
            raise ValueError("轮速比例增益 Kp 必须是大于 0 的有限数。")
        if not math.isfinite(wheel_ki) or wheel_ki < 0.0:
            raise ValueError("轮速积分增益 Ki 必须是大于等于 0 的有限数。")
        if cutoff_hz <= 0.0:
            raise ValueError("力矩低通滤波截止频率必须大于 0 Hz。")
        if control_dt <= 0.0:
            raise ValueError("力矩控制周期必须大于 0 s。")
        self.model = model
        self.control_mode = control_mode
        self.wheel_kp = wheel_kp
        self.wheel_ki = wheel_ki
        self.control_dt = control_dt
        self.wheel_integral_error = np.zeros(2, dtype=np.float64)
        self.low_pass_enabled = low_pass_enabled
        self.cutoff_hz = cutoff_hz
        self.low_pass_alpha = 1.0 - math.exp(
            -2.0 * math.pi * cutoff_hz * control_dt
        )
        self.filtered_torque = np.zeros(ACTION_DIM, dtype=np.float64)
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

    def set_mode(self, control_mode: str) -> None:
        """Switch the active wheel/foot actuator semantics without rebuilding MuJoCo."""
        if control_mode not in ("wheel", "foot"):
            raise ValueError(f"TorqueController mode must be wheel or foot, got {control_mode!r}")
        self.control_mode = control_mode
        self.wheel_integral_error.fill(0.0)

    def reset(self) -> None:
        """Reset controller memory after a robot reset."""
        self.wheel_integral_error.fill(0.0)
        self.filtered_torque.fill(0.0)

    def apply(self, data: mujoco.MjData, action: np.ndarray) -> float:
        leg_targets = self.default_leg_positions + 0.25 * action[:6]
        leg_torque = 40.0 * (leg_targets - data.qpos[self.leg_qpos_addresses])
        leg_torque -= 2.5 * data.qvel[self.leg_dof_addresses]
        # Foot always brakes toward zero wheel velocity, matching Isaac Lab.
        # Wheel mode continues to use the policy targets.
        wheel_targets = np.asarray(action[6:], dtype=np.float64)
        if self.control_mode == "foot":
            wheel_targets = np.zeros_like(wheel_targets)
        wheel_error = wheel_targets - data.qvel[self.wheel_dof_addresses]
        candidate_integral = (
            self.wheel_integral_error + wheel_error * self.control_dt
            if self.wheel_ki > 0.0
            else np.zeros_like(self.wheel_integral_error)
        )
        proportional_torque = self.wheel_kp * wheel_error
        candidate_torque = proportional_torque + self.wheel_ki * candidate_integral
        pushes_upper_limit = (candidate_torque > 80.0) & (wheel_error > 0.0)
        pushes_lower_limit = (candidate_torque < -80.0) & (wheel_error < 0.0)
        accept_integral = ~(pushes_upper_limit | pushes_lower_limit)
        self.wheel_integral_error = np.where(
            accept_integral,
            candidate_integral,
            self.wheel_integral_error,
        )
        if self.wheel_ki == 0.0:
            self.wheel_integral_error.fill(0.0)
        wheel_torque = np.clip(
            proportional_torque + self.wheel_ki * self.wheel_integral_error,
            -80.0,
            80.0,
        )
        raw_torque = np.clip(
            np.concatenate((leg_torque, wheel_torque)), -80.0, 80.0
        )
        if self.low_pass_enabled:
            self.filtered_torque += self.low_pass_alpha * (
                raw_torque - self.filtered_torque
            )
            torque = self.filtered_torque
        else:
            torque = raw_torque
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

    def __init__(self, body_height: float, body_height_range: tuple[float, float]) -> None:
        self.pressed: set[int] = set()
        self.requested_mode: str | None = None
        self.body_height = body_height
        self.body_height_min, self.body_height_max = body_height_range

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
        elif key == glfw.KEY_1 and action == glfw.PRESS:
            self.requested_mode = "wheel"
        elif key == glfw.KEY_2 and action == glfw.PRESS:
            self.requested_mode = "foot"

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
                self.body_height_min,
                self.body_height_max,
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


class TelemetryPlot:
    """Lightweight live command tracking and attitude curves in a Tk window."""

    _PLOT_SPECS = (
        ("vx command vs actual", "vx", "vx_cmd", "m/s", "command", (-1.2, 1.2)),
        ("vy command vs actual", "vy", "vy_cmd", "m/s", "command", (-0.8, 0.8)),
        ("wz command vs actual", "wz", "wz_cmd", "rad/s", "command", (-1.2, 1.2)),
        ("height command vs actual", "height", "height_cmd", "m", "command", None),
        ("roll vs terrain reference", "roll", "roll_ref", "deg", "terrain ref", (-35.0, 35.0)),
        ("pitch vs terrain reference", "pitch", "pitch_ref", "deg", "terrain ref", (-35.0, 35.0)),
    )

    def __init__(
        self,
        model: mujoco.MjModel,
        base_body_id: int,
        scanner: TerrainHeightScanner,
        history_seconds: float,
        body_height_range: tuple[float, float],
    ) -> None:
        try:
            import tkinter as tk
        except Exception as error:
            raise RuntimeError(
                "实时曲线窗口初始化失败；可临时使用 --no-plot 关闭。"
            ) from error

        self.model = model
        self.base_body_id = base_body_id
        self.scanner = scanner
        self.history_seconds = history_seconds
        self.velocity_buffer = np.zeros(6, dtype=np.float64)
        self.times: deque[float] = deque()
        self.values = {
            key: deque()
            for key in (
                "vx",
                "vy",
                "wz",
                "height",
                "roll",
                "pitch",
                "roll_ref",
                "pitch_ref",
                "vx_cmd",
                "vy_cmd",
                "wz_cmd",
                "height_cmd",
            )
        }
        self._tk = tk
        self._closed = False
        self.root = tk.Tk()
        self.root.title("TRON1A command tracking and attitude")
        self.root.geometry("1050x850+1240+40")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.canvas = tk.Canvas(
            self.root,
            background="#0b1020",
            highlightthickness=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        height_limits = (body_height_range[0] - 0.10, body_height_range[1] + 0.10)
        self.y_limits = [
            height_limits if limits is None else limits
            for *_, limits in self._PLOT_SPECS
        ]
        self.plot_bounds: list[tuple[float, float, float, float]] = []
        self.line_items: list[tuple[int, int]] = []
        self._last_canvas_size = (0, 0)
        self.root.update_idletasks()
        self._layout()
        self.root.update()

    def _on_close(self) -> None:
        self._closed = True
        try:
            self.root.destroy()
        except self._tk.TclError:
            pass

    def _layout(self) -> None:
        width = max(int(self.canvas.winfo_width()), 800)
        height = max(int(self.canvas.winfo_height()), 650)
        self._last_canvas_size = (width, height)
        self.canvas.delete("all")
        self.plot_bounds.clear()
        self.line_items.clear()
        gap = 12.0
        panel_width = (width - 3.0 * gap) / 2.0
        panel_height = (height - 4.0 * gap) / 3.0
        for index, spec in enumerate(self._PLOT_SPECS):
            title, _, _, unit, reference_label, _ = spec
            column = index % 2
            row = index // 2
            panel_x0 = gap + column * (panel_width + gap)
            panel_y0 = gap + row * (panel_height + gap)
            panel_x1 = panel_x0 + panel_width
            panel_y1 = panel_y0 + panel_height
            self.canvas.create_rectangle(
                panel_x0,
                panel_y0,
                panel_x1,
                panel_y1,
                fill="#111827",
                outline="#334155",
            )
            self.canvas.create_text(
                0.5 * (panel_x0 + panel_x1),
                panel_y0 + 16.0,
                text=title,
                fill="#f8fafc",
                font=("TkDefaultFont", 10, "bold"),
            )
            x0 = panel_x0 + 58.0
            y0 = panel_y0 + 36.0
            x1 = panel_x1 - 16.0
            y1 = panel_y1 - 34.0
            self.plot_bounds.append((x0, y0, x1, y1))
            lower, upper = self.y_limits[index]
            for fraction in (0.0, 0.5, 1.0):
                y = y1 - fraction * (y1 - y0)
                self.canvas.create_line(x0, y, x1, y, fill="#263247")
                value = lower + fraction * (upper - lower)
                self.canvas.create_text(
                    x0 - 7.0,
                    y,
                    text=f"{value:.2f}",
                    anchor="e",
                    fill="#94a3b8",
                    font=("TkDefaultFont", 8),
                )
            self.canvas.create_rectangle(x0, y0, x1, y1, outline="#64748b")
            self.canvas.create_text(
                panel_x0 + 14.0,
                0.5 * (y0 + y1),
                text=unit,
                angle=90,
                fill="#cbd5e1",
                font=("TkDefaultFont", 8),
            )
            for fraction, label in (
                (0.0, f"-{self.history_seconds:g}"),
                (0.5, f"-{0.5 * self.history_seconds:g}"),
                (1.0, "0"),
            ):
                x = x0 + fraction * (x1 - x0)
                self.canvas.create_text(
                    x,
                    y1 + 13.0,
                    text=label,
                    fill="#94a3b8",
                    font=("TkDefaultFont", 8),
                )
            legend_y = panel_y1 - 12.0
            self.canvas.create_line(
                panel_x0 + 75.0,
                legend_y,
                panel_x0 + 97.0,
                legend_y,
                fill="#38bdf8",
                width=2,
            )
            self.canvas.create_text(
                panel_x0 + 102.0,
                legend_y,
                text="actual",
                anchor="w",
                fill="#cbd5e1",
                font=("TkDefaultFont", 8),
            )
            self.canvas.create_line(
                panel_x0 + 160.0,
                legend_y,
                panel_x0 + 182.0,
                legend_y,
                fill="#fb923c",
                width=2,
                dash=(5, 3),
            )
            self.canvas.create_text(
                panel_x0 + 187.0,
                legend_y,
                text=reference_label,
                anchor="w",
                fill="#cbd5e1",
                font=("TkDefaultFont", 8),
            )
            actual_item = self.canvas.create_line(
                0.0, 0.0, 0.0, 0.0, fill="#38bdf8", width=2
            )
            reference_item = self.canvas.create_line(
                0.0,
                0.0,
                0.0,
                0.0,
                fill="#fb923c",
                width=2,
                dash=(5, 3),
            )
            self.line_items.append((actual_item, reference_item))

    @property
    def is_open(self) -> bool:
        if self._closed:
            return False
        try:
            return bool(self.root.winfo_exists())
        except self._tk.TclError:
            self._closed = True
            return False

    def reset(self) -> None:
        if not self.is_open:
            return
        self.times.clear()
        for series in self.values.values():
            series.clear()
        for actual_item, reference_item in self.line_items:
            self.canvas.coords(actual_item, 0.0, 0.0, 0.0, 0.0)
            self.canvas.coords(reference_item, 0.0, 0.0, 0.0, 0.0)

    def sample(
        self, data: mujoco.MjData, command: np.ndarray, simulation_time: float
    ) -> None:
        if not self.is_open:
            self._closed = True
            return
        if self.times and simulation_time < self.times[-1]:
            self.reset()

        _base_velocity_in_body_frame(
            self.model,
            data,
            self.base_body_id,
            self.velocity_buffer,
        )
        rotation = np.asarray(
            data.xmat[self.base_body_id], dtype=np.float64
        ).reshape(3, 3)
        roll, pitch = _roll_pitch_degrees(rotation)
        self.scanner.scan(data)
        roll_ref, pitch_ref = self.scanner.attitude_reference(data)
        sample = {
            "vx": float(self.velocity_buffer[3]),
            "vy": float(self.velocity_buffer[4]),
            "wz": float(self.velocity_buffer[2]),
            "height": self.scanner.body_height(data),
            "roll": roll,
            "pitch": pitch,
            "roll_ref": roll_ref,
            "pitch_ref": pitch_ref,
            "vx_cmd": float(command[0]),
            "vy_cmd": float(command[1]),
            "wz_cmd": float(command[2]),
            "height_cmd": float(command[3]),
        }
        self.times.append(simulation_time)
        for key, value in sample.items():
            self.values[key].append(value)

        while self.times and simulation_time - self.times[0] > self.history_seconds:
            self.times.popleft()
            for series in self.values.values():
                series.popleft()

    def draw(self) -> None:
        if not self.is_open or not self.times:
            return
        try:
            self.root.update_idletasks()
        except self._tk.TclError:
            self._closed = True
            return
        canvas_size = (int(self.canvas.winfo_width()), int(self.canvas.winfo_height()))
        relative_times = np.asarray(self.times, dtype=np.float64) - self.times[-1]
        limits_changed = False
        plot_values = []
        for index, (_, actual_key, reference_key, _, _, _) in enumerate(self._PLOT_SPECS):
            actual_values = np.asarray(self.values[actual_key], dtype=np.float64)
            reference_values = np.asarray(self.values[reference_key], dtype=np.float64)
            plot_values.append((actual_values, reference_values))
            finite_values = np.concatenate(
                (actual_values[np.isfinite(actual_values)], reference_values[np.isfinite(reference_values)])
            )
            if finite_values.size:
                lower, upper = self.y_limits[index]
                data_min = float(np.min(finite_values))
                data_max = float(np.max(finite_values))
                if data_min < lower or data_max > upper:
                    margin = max(0.05 * (data_max - data_min), 1.0e-3)
                    self.y_limits[index] = (
                        min(lower, data_min - margin),
                        max(upper, data_max + margin),
                    )
                    limits_changed = True
        if limits_changed or canvas_size != self._last_canvas_size:
            self._layout()

        for index, ((actual_values, reference_values), bounds, items) in enumerate(
            zip(plot_values, self.plot_bounds, self.line_items)
        ):
            lower, upper = self.y_limits[index]
            x0, y0, x1, y1 = bounds
            time_fraction = np.clip(
                (relative_times + self.history_seconds) / self.history_seconds,
                0.0,
                1.0,
            )
            x_coordinates = x0 + time_fraction * (x1 - x0)
            for values, item in zip((actual_values, reference_values), items):
                finite = np.isfinite(values)
                if not np.any(finite):
                    self.canvas.coords(item, 0.0, 0.0, 0.0, 0.0)
                    continue
                y_fraction = np.clip((values[finite] - lower) / (upper - lower), 0.0, 1.0)
                y_coordinates = y1 - y_fraction * (y1 - y0)
                coordinates = np.column_stack(
                    (x_coordinates[finite], y_coordinates)
                ).ravel()
                if coordinates.size == 2:
                    coordinates = np.tile(coordinates, 2)
                self.canvas.coords(item, *coordinates.tolist())
        try:
            self.root.update()
        except self._tk.TclError:
            self._closed = True

    def close(self) -> None:
        if self.is_open:
            self._on_close()


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
    checkpoint: Path,
    control_mode: str,
    region: TerrainRegion,
    spawn_xy: np.ndarray,
    yaw: float,
) -> None:
    print(f"[policy] mode={control_mode}, checkpoint={checkpoint}")
    print(
        f"[terrain] type={region.terrain_type}, level={region.terrain_level}, "
        f"number={region.number}, center={region.center_xy_w}"
    )
    print(
        f"[spawn] position=({spawn_xy[0]:.3f}, {spawn_xy[1]:.3f}), "
        f"yaw={yaw:+.3f} rad"
    )
    print("  W/S : 前进/后退")
    if control_mode == "wheel":
        print("  Q/E : 左移/右移（Wheel 训练时横向指令为 0，效果可能不稳定）")
    else:
        print("  Q/E : 左移/右移")
    if control_mode == "dual":
        print("  1 : Wheel 模式")
        print("  2 : Foot 模式（Foot→Wheel 等待当前步态周期结束）")
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
    foothold_debug: FootholdDebug | None = None,
) -> None:
    # MuJoCo azimuth 45 deg places the camera at local (-x, -y): right rear.
    _refresh_body_kinematics(model, data)
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
    if foothold_debug is not None:
        foothold_debug.add_geoms(scene)
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
        description="Deploy an Isaac Lab TRON1A wheel or foot expert in MuJoCo."
    )
    parser.add_argument("--mode", choices=CONTROL_MODES, default="wheel")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--wheel-checkpoint", type=Path, default=None,
                        help="Dual 模式使用的 Wheel checkpoint。")
    parser.add_argument("--foot-checkpoint", type=Path, default=None,
                        help="Dual 模式使用的 Foot checkpoint。")
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=None,
        help="Checkpoint search root; defaults to the selected mode's log directory.",
    )
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
    parser.add_argument(
        "--gait-frequency", type=float, default=None,
        help="Foot: training-range midpoint; Wheel: 1.7. Explicit values override the default.",
    )
    parser.add_argument(
        "--gait-offset", type=float, default=None,
        help="Foot: training-range midpoint; Wheel: 0.5. Explicit values override the default.",
    )
    parser.add_argument(
        "--gait-duration", type=float, default=None,
        help="Foot: training-range midpoint (fallback 0.5); Wheel: 0.525.",
    )
    parser.add_argument(
        "--swing-height", type=float, default=None,
        help="Foot: training-range midpoint (fallback 0 m); Wheel: 0.14 m.",
    )
    parser.add_argument("--spawn-z-offset", type=float, default=0.1)
    parser.add_argument("--render-hz", type=float, default=60.0)
    parser.add_argument("--plot-hz", type=float, default=5.0)
    parser.add_argument("--plot-sample-hz", type=float, default=20.0)
    parser.add_argument("--plot-history", type=float, default=20.0)
    parser.add_argument("--camera-distance", type=float, default=3.5)
    parser.add_argument("--camera-elevation", type=float, default=-18.0)
    parser.add_argument(
        "--wheel-kp",
        "--wheel-kv",
        dest="wheel_kp",
        type=float,
        default=2.0,
        help=(
            "Wheel velocity-error proportional gain in N·m/(rad/s) "
            "(default: 2.0; --wheel-kv is a compatibility alias)."
        ),
    )
    parser.add_argument(
        "--wheel-ki",
        type=float,
        default=0.5,
        help="Wheel velocity-error integral gain in N·m/rad (default: 0.5).",
    )
    parser.add_argument(
        "--torque-low-pass",
        action="store_true",
        help="Enable a first-order low-pass filter on all eight applied joint torques.",
    )
    parser.add_argument(
        "--torque-cutoff-hz",
        type=float,
        default=20.0,
        help="Torque low-pass cutoff frequency in Hz (default: 20; filter is off unless enabled).",
    )
    parser.add_argument("--show-footholds", action="store_true", help="Foot: preview frozen landing regions and actual landing events.")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--no-plot", action="store_true", help="Disable the live telemetry window."
    )
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument(
        "--max-steps", type=int, default=None, help="Optional number of physics steps."
    )
    args = parser.parse_args(argv)
    if args.render_hz <= 0.0:
        parser.error("--render-hz 必须大于 0。")
    if args.plot_hz <= 0.0:
        parser.error("--plot-hz 必须大于 0。")
    if args.plot_sample_hz <= 0.0:
        parser.error("--plot-sample-hz 必须大于 0。")
    if args.plot_history <= 0.0:
        parser.error("--plot-history 必须大于 0。")
    if not math.isfinite(args.wheel_kp) or args.wheel_kp <= 0.0:
        parser.error("--wheel-kp 必须是大于 0 的有限数。")
    if not math.isfinite(args.wheel_ki) or args.wheel_ki < 0.0:
        parser.error("--wheel-ki 必须是大于等于 0 的有限数。")
    if args.torque_cutoff_hz <= 0.0:
        parser.error("--torque-cutoff-hz 必须大于 0。")
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
    def resolve_checkpoint(mode: str, explicit: Path | None) -> Path:
        if explicit is not None:
            result = explicit.expanduser().resolve()
        elif args.checkpoint is not None and args.mode != "dual":
            result = args.checkpoint.expanduser().resolve()
        else:
            root = args.checkpoint_root if args.checkpoint_root is not None else (
                DEFAULT_FOOT_CHECKPOINT_ROOT if mode == "foot" else DEFAULT_WHEEL_CHECKPOINT_ROOT
            )
            result = discover_latest_checkpoint(root.expanduser().resolve())
        if not result.is_file():
            raise FileNotFoundError(f"{mode} checkpoint 不存在：{result}")
        return result

    if args.mode == "dual":
        wheel_checkpoint = resolve_checkpoint("wheel", args.wheel_checkpoint)
        foot_checkpoint = resolve_checkpoint("foot", args.foot_checkpoint)
        checkpoints = {"wheel": wheel_checkpoint, "foot": foot_checkpoint}
    else:
        checkpoint = resolve_checkpoint(args.mode, None)
        checkpoints = {args.mode: checkpoint}

    def mode_args(mode: str) -> argparse.Namespace:
        copied = argparse.Namespace(**vars(args))
        copied.mode = mode
        return copied

    gait_commands = {
        mode: resolve_gait_command(mode_args(mode), path)
        for mode, path in checkpoints.items()
    }
    body_height_ranges = {mode: checkpoint_body_height_range(path) for mode, path in checkpoints.items()}
    body_height_range = body_height_ranges["wheel" if args.mode == "dual" else args.mode]
    for mode, height_range in body_height_ranges.items():
        if not height_range[0] <= args.body_height <= height_range[1]:
            raise ValueError(
                f"--body-height 不在 {mode} checkpoint 的训练范围 "
                f"[{height_range[0]}, {height_range[1]}] 内。"
            )
    device = resolve_device(args.device)

    model = build_scene_model(
        args.terrain_xml.expanduser().resolve(),
        args.robot_xml.expanduser().resolve(),
        PHYSICS_DT,
    )
    data = mujoco.MjData(model)
    base_body_id = _model_id(model, mujoco.mjtObj.mjOBJ_BODY, "base_Link")
    scanner = TerrainHeightScanner(model, base_body_id)
    policies = {
        mode: WheelPolicy(path, device, model, scanner, body_height_ranges[mode])
        for mode, path in checkpoints.items()
    }
    if args.mode == "dual":
        print(f"[dual] Wheel checkpoint: {checkpoints['wheel']}")
        print(f"[dual] Foot checkpoint:  {checkpoints['foot']}")
    active_mode = "wheel" if args.mode == "dual" else args.mode
    requested_mode = active_mode
    policy = policies[active_mode]
    gait_command = gait_commands[active_mode]
    torque_controller = TorqueController(
        model,
        active_mode,
        wheel_kp=args.wheel_kp,
        wheel_ki=args.wheel_ki,
        low_pass_enabled=args.torque_low_pass,
        cutoff_hz=args.torque_cutoff_hz,
    )
    rng = np.random.default_rng(args.seed)
    command_state = KeyboardCommandState(args.body_height, body_height_range)
    spawn_xy, spawn_yaw = reset_robot(
        model, data, region, rng, args.spawn_z_offset
    )
    policy.reset(data, gait_command)
    torque_controller.reset()
    foothold_debug = FootholdDebug(model, base_body_id, scanner, checkpoints[active_mode]) if args.show_footholds and active_mode == "foot" else None
    _print_help(checkpoints[active_mode], args.mode, region, spawn_xy, spawn_yaw)
    print(
        f"[runtime] mode={args.mode}, active_mode={active_mode}, checkpoint_iter={policy.iteration}, "
        f"wheel_target={'fixed_zero' if args.mode == 'foot' else 'policy'}, "
        f"device={device}, "
        f"physics_dt={PHYSICS_DT}, policy_dt={POLICY_DT}, "
        f"wheel_kp={args.wheel_kp:g}, wheel_ki={args.wheel_ki:g}, "
        f"torque_low_pass={'on' if args.torque_low_pass else 'off'}, "
        f"torque_cutoff_hz={args.torque_cutoff_hz:g}, "
        f"height_range={body_height_range}, "
        f"gait_clock={'continuous' if policy.continuous_gait_phase else 'legacy'}, "
        f"gait=({', '.join(f'{value:g}' for value in gait_command)})"
    )

    window = None
    camera = None
    option = None
    scene = None
    context = None
    telemetry_plot = None
    reset_requested = False
    mode_switch_pending = False
    if not args.headless:
        if not glfw.init():
            raise RuntimeError("GLFW 初始化失败；无显示环境可使用 --headless。")
        window = glfw.create_window(
            1200, 900, f"TRON1A MuJoCo {args.mode.title()} Policy", None, None
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
        if not args.no_plot:
            telemetry_plot = TelemetryPlot(
                model,
                base_body_id,
                scanner,
                args.plot_history,
                body_height_range,
            )
            glfw.focus_window(window)

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
    next_plot_sample_time = float(data.time)
    next_plot_draw_time = float(data.time)
    next_status_time = float(data.time)
    wall_start = time.perf_counter()
    sim_start = float(data.time)
    try:
        while True:
            if window is not None and glfw.window_should_close(window):
                break
            if args.max_steps is not None and physics_steps >= args.max_steps:
                break

            if args.mode == "dual" and command_state.requested_mode is not None:
                requested_mode = command_state.requested_mode
                command_state.requested_mode = None
                mode_switch_pending = True

            if args.mode == "dual" and requested_mode != active_mode:
                # Match B2W: do not blend two controllers. Foot->Wheel waits
                # for the gait cycle boundary, then recreates the active mode.
                phase = (policy._phase_cycles if policy.continuous_gait_phase
                         else (policy.policy_step * POLICY_DT * float(gait_command[0])) % 1.0)
                can_switch = active_mode == "wheel" or phase < 0.05
                if can_switch:
                    active_mode = requested_mode
                    policy = policies[active_mode]
                    gait_command = gait_commands[active_mode]
                    policy.reset(data, gait_command)
                    torque_controller.set_mode(active_mode)
                    action.fill(0.0)
                    mode_switch_pending = True
                    print(f"[mode] active mode switched to {active_mode}; checkpoint={checkpoints[active_mode]}")

            if reset_requested:
                command_state.clear_motion()
                region = select_or_reuse_terrain_region_on_reset(region, regions)
                command_state.body_height = args.body_height
                spawn_xy, spawn_yaw = reset_robot(
                    model, data, region, rng, args.spawn_z_offset
                )
                policy.reset(data, gait_command)
                torque_controller.reset()
                if foothold_debug is not None:
                    foothold_debug.reset()
                action.fill(0.0)
                episode_physics_steps = 0
                reset_requested = False
                next_render_time = float(data.time)
                next_plot_sample_time = float(data.time)
                next_plot_draw_time = float(data.time)
                next_status_time = float(data.time)
                if telemetry_plot is not None:
                    telemetry_plot.reset()
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
                if mode_switch_pending:
                    command[:3] = 0.0
                if foothold_debug is not None:
                    phase = policy._phase_cycles if policy.continuous_gait_phase else (policy.policy_step * POLICY_DT * float(gait_command[0])) % 1.
                    foothold_debug.update(data, command, gait_command, phase)
                action = policy.act(data, command, gait_command)
                mode_switch_pending = False
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

            if (
                telemetry_plot is not None
                and data.time + 1.0e-12 >= next_plot_sample_time
            ):
                telemetry_plot.sample(data, command, float(data.time))
                while next_plot_sample_time <= data.time + 1.0e-12:
                    next_plot_sample_time += 1.0 / args.plot_sample_hz

            if (
                telemetry_plot is not None
                and data.time + 1.0e-12 >= next_plot_draw_time
            ):
                telemetry_plot.draw()
                while next_plot_draw_time <= data.time + 1.0e-12:
                    next_plot_draw_time += 1.0 / args.plot_hz

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
                    foothold_debug,
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
        if telemetry_plot is not None:
            telemetry_plot.close()
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
