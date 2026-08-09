# WF 双专家方案概览

## 目标

为 `WF_TRON1A` 分别训练两个策略，由外部 FSM 负责切换：

| 专家 | 任务 | 作用 |
|---|---|---|
| Wheel | `Isaac-Limx-WF-Wheel-Mode-v0` | 双轮持续接触，完成滚动、转向和高度调节 |
| Foot | `Isaac-Limx-WF-Foot-AllTerrain-v0` | 轮速目标锁零，通过步态处理复杂地形 |
| Dual Play | `Isaac-Limx-WF-Dual-Mode-Play-v0` | 同时加载两个 checkpoint 并平滑切换 |

## 统一接口

- 动作：8 维 = 6 个腿关节位置 + 2 个轮关节速度。
- 命令：`[vx, vy, wz, body_height]`。
- 两个专家保持相同 observation/action schema。
- Foot 环境和 FSM 都会将最后两维轮速动作置零。

## 专家差异

| 项目 | Wheel | Foot |
|---|---|---|
| 速度范围 | `vx ±1.5`，`vy=0`，`wz ±1.0` | `vx ±0.8`，`vy ±0.4`，`wz ±0.8` |
| 高度范围 | `0.75～0.85 m` | `0.65～0.82 m` |
| 地形 | 平地、坡面、粗糙、低下楼梯 | 平地、坡面、粗糙、上下楼梯 |
| 轮动作 | 正常生效 | 硬置零 |
| 重点约束 | 轮地接触、滚动、固定水平构型 | 步态时序、摆动高度、低轮速 |

## 模式切换

控制状态：

```text
wheel → wheel_to_foot → foot → foot_to_wheel → wheel
```

切换期间：

- 速度命令归零；
- 两个专家动作渐变；
- Foot 侧轮速始终为零；
- Foot→Wheel 需要低机身速度、低轮速、双轮接触和直立状态。

当前 `auto` 模式按 height-scan relief 请求模式，后续计划改为基于双专家实测结果的地形可行性表。

## 当前状态（2026-08-05）

- Wheel 已完成 20,000 iterations GPU 训练：
  - checkpoint：`logs/rsl_rl/wf_tron_1a_wheel_mode/2026-08-04_20-21-08/model_20000.pt`
  - 平均 episode 长度：约 `958/1000`
  - timeout：约 `94.7%`
  - base contact：约 `5.3%`
- Foot 正式大规模训练仍待完成。
- Dual Play、统一网络接口和 Foot 轮速 mask 已通过冒烟验证。
- Wheel 新版“双轮有效接触 + 水平中性点”奖励已完成代码和 CPU 冒烟验证；正式 GPU 重训待执行。

## 文档导航

- 训练与运行：`wf_dual_mode_training_process.md`
- 当前奖励表：`wf_dual_mode_reward_penalty_table.md`
- 下一版轮态判断修改：`wf_new_mode_judgment_modification_plan.md`
