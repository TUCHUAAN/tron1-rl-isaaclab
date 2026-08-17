# WF 双专家方案概览

## 目标

为 `WF_TRON1A` 分别训练两个策略，由外部 FSM 负责切换：

| 专家 | 任务 | 作用 |
|---|---|---|
| Wheel Height Pretrain（可选） | `Isaac-Limx-WF-Wheel-Height-Pretrain-v0` | 仅用于平地诊断，不属于正式训练流程 |
| Wheel | `Isaac-Limx-WF-Wheel-Mode-v0` | 双轮持续接触，完成滚动、转向和高度调节 |
| Foot | `Isaac-Limx-WF-Foot-AllTerrain-v0` | 轮速目标锁零，通过步态处理复杂地形 |
| Dual Play | `Isaac-Limx-WF-Dual-Mode-Play-v0` | 同时加载两个 checkpoint 并平滑切换 |

## 统一接口

- 动作：8 维 = 6 个腿关节位置 + 2 个轮关节速度。
- 命令：`[vx, vy, wz, body_height_norm]`；高度在命令生成器和奖励中保持米制，仅网络输入归一化到 `[0, 1]`。
- 两个专家保持相同 observation/action schema。
- Foot 环境和 FSM 都会将最后两维轮速动作置零。

## 专家差异

| 项目 | Wheel | Foot |
|---|---|---|
| 速度范围 | `vx ±1.5`，`vy=0`，`wz ±1.0` | `vx ±0.8`，`vy ±0.4`，`wz ±0.8` |
| 高度范围 | `0.65～0.85 m` | `0.65～0.85 m` |
| 地形 | 平地、坡面、粗糙、低上/下楼梯 | 平地、坡面、粗糙、上下楼梯 |
| 轮动作 | 正常生效 | 硬置零 |
| 重点约束 | 轮地接触、滚动、固定水平构型 | 步态时序、摆动高度、低轮速 |

Wheel 在近零转向命令下软约束左右轮目标一致，在全部速度命令近零时软约束双轮目标趋零并直接惩罚实际 `wz²`；轮目标保留 `0.15 rad/s` 死区以允许平衡修正。高度、静止和实际 yaw 惩罚都使用轮地支撑软门控，腾空阶段不施加这些状态约束。正式训练采用单阶段全地形 curriculum，从头同时学习高度、静止稳定和地形运动。

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

## 当前状态（2026-08-16）

- 2026-08-12 单阶段 Wheel 训练完成，但实际高度 RMS 误差约 `7.8 cm`，MuJoCo 端点命令响应不足。
- 2026-08-13 的加强版单阶段训练高度 `MAE 7.26 cm / RMSE 11.78 cm`，base contact 终止率 `57.5%`；根因是未触地阶段强高度负奖励和过紧静止约束。
- 已将高度范围改为 `0.65～0.85 m`、端点比例降到 `20%`，为高度/静止/实际 yaw 加入支撑门控，并加入 `base_contact` 终止惩罚。
- `2026-08-15_00-26-20_wheel_single_stage_h065_085_supportgate` 因误用持久化的 `is_terminated_term`，复位后仍逐步扣除终止惩罚，约 4020 iterations 的结果无效；现已改为当前步 `mdp.is_terminated`，必须重新从头训练。
- 修复后的 20k 训练在 `model_6500` 达到 `92.2%` timeout，最终 `model_20000` 降至 `76.6%`，证明最后一个 checkpoint 不是最佳模型；已增加基于高度/速度/地形门槛和 base contact 的选择工具。
- 当前 Wheel 高度权重为 `-60`、倒地权重 `-500`、零命令轮目标 `-0.10`；竖直速度惩罚在高度命令变化时平滑降至 `20%`，接近目标后恢复。等待从头进行新的单阶段全地形训练。`Wheel-Height-Pretrain` 仅作为可选诊断任务保留。
- 新配置已通过真实 Isaac Lab Wheel CPU 1-env / 3-step 冒烟，27 个奖励项均为有限值；MuJoCo 无窗口冒烟确认会优先加载稳定性选择文件指定的 checkpoint。
- Foot 正式大规模训练仍待完成。
- Dual Play、统一网络接口和 Foot 轮速 mask 已通过既有冒烟验证。

## 文档导航

- 训练与运行：`wf_dual_mode_training_process.md`
- 当前奖励表：`wf_dual_mode_reward_penalty_table.md`
- 下一版轮态判断修改：`wf_new_mode_judgment_modification_plan.md`
