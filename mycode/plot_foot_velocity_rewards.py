#!/usr/bin/env python3
"""交互绘制 Foot 模式的 vx、vy、wz 速度跟踪奖励曲线。

横轴是有符号速度误差 ``actual - command``。vx/vy 图是二维 XY 误差曲面在
另一个分量误差为零时的一维切片。周期平均项无法只由瞬时误差唯一确定，因此
图中采用“一个完整步态周期内误差恒定”的等效切片，即周期平均误差等于横轴。

直接运行后，可用窗口底部的滑块实时调整七个奖励权重：

    python mycode/plot_foot_velocity_rewards.py

也可以在无图形界面的环境中保存初始曲线：

    MPLBACKEND=Agg python mycode/plot_foot_velocity_rewards.py \
        --save mycode/foot_velocity_rewards.png --no-show
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.widgets import Button, Slider


# 与 Foot v7 环境配置保持一致。这里的 EXP_* 是公式分母 std^2，
# HUBER_* 是将绝对误差归一化到 1 的物理尺度。
EXP_XY_STD_SQUARED = 0.20
EXP_YAW_STD_SQUARED = 0.25
HUBER_XY_SCALE = 0.40
HUBER_YAW_SCALE = 0.50
HUBER_CYCLE_XY_SCALE = 0.20
HUBER_CYCLE_YAW_SCALE = 0.30


@dataclass(frozen=True)
class Weights:
    """Foot 速度奖励权重；默认值来自当前 Foot v7 配置。"""

    xy_exp: float = 8.0
    yaw_exp: float = 4.0
    xy_current_huber: float = -0.5
    yaw_current_huber: float = -0.2
    vx_cycle_huber: float = -0.2
    vy_cycle_huber: float = -0.2
    yaw_cycle_huber: float = -0.2


@dataclass(frozen=True)
class CurveSet:
    """一条总曲线及组成它的三个加权分量。"""

    exponential: np.ndarray
    current_huber: np.ndarray
    cycle_huber: np.ndarray
    total: np.ndarray


def normalized_huber(error: np.ndarray, scale: float) -> np.ndarray:
    """复现环境中的 ``Huber(abs(error) / scale)``。"""

    normalized_error = np.abs(np.asarray(error, dtype=float)) / scale
    return np.where(
        normalized_error <= 1.0,
        0.5 * normalized_error**2,
        normalized_error - 0.5,
    )


def velocity_curves(error: np.ndarray, component: str, weights: Weights) -> CurveSet:
    """计算指定 Foot 速度分量的各加权奖励项和总和。"""

    error = np.asarray(error, dtype=float)
    if component in ("vx", "vy"):
        exponential = weights.xy_exp * np.exp(-(error**2) / EXP_XY_STD_SQUARED)
        current_huber = weights.xy_current_huber * normalized_huber(error, HUBER_XY_SCALE)
        cycle_weight = weights.vx_cycle_huber if component == "vx" else weights.vy_cycle_huber
        cycle_huber = cycle_weight * normalized_huber(error, HUBER_CYCLE_XY_SCALE)
    elif component == "wz":
        exponential = weights.yaw_exp * np.exp(-(error**2) / EXP_YAW_STD_SQUARED)
        current_huber = weights.yaw_current_huber * normalized_huber(error, HUBER_YAW_SCALE)
        cycle_huber = weights.yaw_cycle_huber * normalized_huber(error, HUBER_CYCLE_YAW_SCALE)
    else:
        raise ValueError(f"Unsupported component: {component!r}")

    return CurveSet(
        exponential=exponential,
        current_huber=current_huber,
        cycle_huber=cycle_huber,
        total=exponential + current_huber + cycle_huber,
    )


class InteractivePlot:
    """Matplotlib 图窗及其权重滑块。"""

    _COMPONENTS = ("vx", "vy", "wz")

    def __init__(self, linear_error_max: float, yaw_error_max: float, points: int) -> None:
        self.errors = {
            "vx": np.linspace(-linear_error_max, linear_error_max, points),
            "vy": np.linspace(-linear_error_max, linear_error_max, points),
            "wz": np.linspace(-yaw_error_max, yaw_error_max, points),
        }
        self.figure, axes = plt.subplots(1, 3, figsize=(15.5, 9.0), sharey=False)
        self.axes: dict[str, Axes] = dict(zip(self._COMPONENTS, axes))
        self.figure.subplots_adjust(left=0.065, right=0.985, top=0.86, bottom=0.42, wspace=0.25)
        try:
            self.figure.canvas.manager.set_window_title("Foot velocity reward curves")
        except AttributeError:
            pass

        self.lines: dict[str, dict[str, object]] = {}
        self._make_plots()
        self.sliders = self._make_sliders()
        self._make_reset_button()
        self.update()

    def _make_plots(self) -> None:
        initial = Weights()
        titles = {
            "vx": r"Foot $v_x$ tracking",
            "vy": r"Foot $v_y$ tracking",
            "wz": r"Foot $\omega_z$ tracking",
        }
        units = {"vx": "m/s", "vy": "m/s", "wz": "rad/s"}

        for component in self._COMPONENTS:
            axis = self.axes[component]
            error = self.errors[component]
            curves = velocity_curves(error, component, initial)
            (exp_line,) = axis.plot(error, curves.exponential, linewidth=1.8, label="exponential tracking")
            (current_line,) = axis.plot(error, curves.current_huber, linewidth=1.6, label="instant Huber")
            (cycle_line,) = axis.plot(error, curves.cycle_huber, linewidth=1.6, label="cycle-mean Huber")
            (total_line,) = axis.plot(error, curves.total, color="black", linewidth=2.7, label="total")
            axis.axhline(0.0, color="0.45", linewidth=0.8)
            axis.axvline(0.0, color="0.65", linewidth=0.8, linestyle=":")
            axis.set_title(titles[component])
            axis.set_xlabel(f"signed tracking error ({units[component]})")
            axis.set_ylabel("weighted reward / penalty")
            axis.grid(True, alpha=0.25)
            axis.legend(loc="lower center", fontsize=8)
            self.lines[component] = {
                "exponential": exp_line,
                "current_huber": current_line,
                "cycle_huber": cycle_line,
                "total": total_line,
            }

        self.figure.suptitle(
            "Foot velocity reward slices (actual - command)", fontsize=15, fontweight="bold"
        )
        self.figure.text(
            0.5,
            0.895,
            r"XY: $e_{other}=0$; cycle term: constant error over one complete gait cycle",
            ha="center",
            fontsize=10,
            color="0.3",
        )

    def _make_sliders(self) -> dict[str, Slider]:
        specifications = (
            ("xy_exp", "XY exp", 0.0, 16.0, 8.0),
            ("yaw_exp", "yaw exp", 0.0, 8.0, 4.0),
            ("xy_current_huber", "XY instant Huber", -2.0, 0.0, -0.5),
            ("yaw_current_huber", "yaw instant Huber", -1.0, 0.0, -0.2),
            ("vx_cycle_huber", "vx cycle Huber", -1.0, 0.0, -0.2),
            ("vy_cycle_huber", "vy cycle Huber", -1.0, 0.0, -0.2),
            ("yaw_cycle_huber", "yaw cycle Huber", -1.0, 0.0, -0.2),
        )
        sliders: dict[str, Slider] = {}
        for row, (name, label, minimum, maximum, initial) in enumerate(specifications):
            slider_axis = self.figure.add_axes((0.21, 0.335 - row * 0.042, 0.64, 0.022))
            slider = Slider(
                slider_axis,
                label,
                minimum,
                maximum,
                valinit=initial,
                valstep=0.01,
                valfmt="%1.2f",
            )
            slider.on_changed(self.update)
            sliders[name] = slider
        return sliders

    def _make_reset_button(self) -> None:
        button_axis = self.figure.add_axes((0.88, 0.035, 0.075, 0.035))
        self.reset_button = Button(button_axis, "Reset")
        self.reset_button.on_clicked(self.reset)

    def current_weights(self) -> Weights:
        return Weights(**{name: slider.val for name, slider in self.sliders.items()})

    def update(self, _value: float | None = None) -> None:
        weights = self.current_weights()
        for component in self._COMPONENTS:
            curves = velocity_curves(self.errors[component], component, weights)
            component_lines = self.lines[component]
            component_lines["exponential"].set_ydata(curves.exponential)
            component_lines["current_huber"].set_ydata(curves.current_huber)
            component_lines["cycle_huber"].set_ydata(curves.cycle_huber)
            component_lines["total"].set_ydata(curves.total)
            axis = self.axes[component]
            axis.relim()
            axis.autoscale_view(scalex=False, scaley=True)
        self.figure.canvas.draw_idle()

    def reset(self, _event: object = None) -> None:
        for slider in self.sliders.values():
            slider.reset()


def positive_float(text: str) -> float:
    value = float(text)
    if not np.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return value


def at_least_two(text: str) -> int:
    value = int(text)
    if value < 2:
        raise argparse.ArgumentTypeError("must be at least 2")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linear-error-max", type=positive_float, default=2.0, help="vx/vy 横轴绝对上限")
    parser.add_argument("--yaw-error-max", type=positive_float, default=3.0, help="wz 横轴绝对上限")
    parser.add_argument("--points", type=at_least_two, default=1001, help="每条曲线的采样点数")
    parser.add_argument("--save", type=Path, help="将初始权重下的图保存到该路径")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="不打开交互窗口（通常与 --save 一起使用）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot = InteractivePlot(args.linear_error_max, args.yaw_error_max, args.points)
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        plot.figure.savefig(args.save, dpi=180)
        print(f"Saved figure to {args.save}")
    if not args.no_show:
        plt.show()
    else:
        plt.close(plot.figure)


if __name__ == "__main__":
    main()
