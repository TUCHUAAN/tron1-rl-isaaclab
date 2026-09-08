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
| 重点约束 | 轮地接触、滚动、固定水平构型 | 步态时序、地形自适应摆腿、低轮速 |

Wheel 在近零转向命令下软约束左右轮目标一致，在全部速度命令近零时软约束双轮目标趋零；轮目标保留 `0.15 rad/s` 死区以允许平衡修正。独立的零命令实际 `wz²` 惩罚已经删除，由 `stand_still=-7.0` 覆盖静止约束。高度和静止项使用轮地支撑软门控，腾空阶段不施加这些状态约束。正式训练采用单阶段全地形 curriculum，从头同时学习高度、静止稳定和地形运动。

Wheel 速度命令当前按站立/直行/纯转向/混合 `25%/30%/10%/35%` 互斥采样，平移覆盖为 `65%` 并保留显式纯 `wz`。实际 base 平面加速度使用 `-0.02` 的有界惩罚；偏航加速度使用 `-0.02` 的 Charbonnier 鲁棒核。两者都保留近目标跟踪门控和联合 `C_any`。

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

## 当前状态（2026-08-29）

- 2026-08-12 单阶段 Wheel 训练完成，但实际高度 RMS 误差约 `7.8 cm`，MuJoCo 端点命令响应不足。
- 2026-08-13 的加强版单阶段训练高度 `MAE 7.26 cm / RMSE 11.78 cm`，base contact 终止率 `57.5%`；根因是未触地阶段强高度负奖励和过紧静止约束。
- 已将高度范围改为 `0.65～0.85 m`、端点比例降到 `20%`，为高度/静止/实际 yaw 加入支撑门控，并加入 `base_contact` 终止惩罚。
- `2026-08-15_00-26-20_wheel_single_stage_h065_085_supportgate` 因误用持久化的 `is_terminated_term`，复位后仍逐步扣除终止惩罚，约 4020 iterations 的结果无效；现已改为当前步 `mdp.is_terminated`，必须重新从头训练。
- 修复后的 20k 训练在 `model_6500` 达到 `92.2%` timeout，最终 `model_20000` 降至 `76.6%`，证明最后一个 checkpoint 不是最佳模型；已增加基于高度/速度/地形门槛和 base contact 的选择工具。
- 当前 Wheel 高度权重为 `-60`、倒地权重 `-500`、零命令轮目标 `-0.10`；竖直速度惩罚在高度命令变化时平滑降至 `20%`，接近目标后恢复。等待从头进行新的单阶段全地形训练。`Wheel-Height-Pretrain` 仅作为可选诊断任务保留。
- 2026-08-29 Wheel 保持高度 L2 权重 `-60`，新增局部地形与支撑门控的高斯核/RBF 高度奖励 `+1.0`（`std=0.05 m`）；两个 Wheel 高度项均改用 `max(C_L,C_R)` 门控，任意一轮有效支撑时即完整生效；线速度/Yaw 跟踪调为 `+3.5/+1.25`，地形姿态调为 `-10`，动作变化/二阶平滑调为 `-0.15/-0.08`。
- 2026-08-30 针对 Wheel 的速度稳态误差和 yaw 超调，将 XY/Yaw 指数跟踪的 `std²` 均收紧为 `0.12`，Yaw 权重调为 `+1.5`，`stand_still` 调为 `-7.0`；Foot 配置不变。
- 2026-08-31 Wheel 的 XY/Yaw 跟踪由 `C_all=min(C_L,C_R)` 改为联合支撑 `C_any=max(C_L,C_R)`，`stand_still` 也改为任意轮地形过滤力支撑即完整生效；权重与网络输入不变。TensorBoard 新增左右轮力/几何/联合置信度、`C_all/C_any`、支撑比例与门控前后跟踪值。
- 2026-08-31 Wheel 地形课程从 `10` 级加密为 `12` 级，实际上坡比例提高到 `25%`；修正了从中心平台向外运动时 Pyramid/InvertedPyramid 的上下坡和上下楼梯方向，真正的上楼梯范围为 `0.005～0.04 m`。升级除了 `4 m` 位移还要求 XY 跟踪 `>=0.55`、`C_any>=0.75` 且不倒地；倒地或移动期跟踪 `<0.25` 会降级。
- 2026-09-01 Wheel 命令改为站立/直行/纯转向/混合 `25%/20%/20%/35%`；新增 `pen_base_lin_acc_xy=-0.05`（饱和尺度 `3.0 m/s²`）和 `pen_base_yaw_acc=-0.05`（饱和尺度 `4.0 rad/s²`），均用 `std=0.30` 跟踪门控及联合 `C_any`，reset 后前两步为零。四类命令比例和两个新增奖励均进入 TensorBoard；Foot 与网络 schema 不变。
- 2026-09-02 根据上一轮 14k 后坡面能力退化的日志，命令比例调整为 `25/30/10/35%`；`pen_base_lin_acc_xy` 降为 `-0.02`，`pen_base_yaw_acc` 降为 `-0.02` 并改用不会完全饱和的 Charbonnier 核。TensorBoard 新增直行、纯转向和混合模式的条件误差及纯转向 `wz` 加速度 RMS；Foot 与网络 schema 不变。
- 2026-09-06 Foot 的 `pen_feet_distance` 按原有 PF 改为 `0.115～1.0 m`，解除原有 WF `0.32～0.35 m` 上限对正常前后步幅的限制；gait contact duration 同样按 PF 固定为 `0.5`，避免计划双摆动区间与 `pen_all_wheels_air_time` 冲突。速度 command 仍从连续实数范围均匀采样，但每 `3～15 s` 直接重采样一次，没有时间平滑。
- 2026-09-07 新 Foot 日志表明 `pen_feet_distance` 已接近零，但 `undesired_contacts` 从约 500 iterations 起持续恶化并在 4k 后稳定接近 `-1`，策略已把一条小腿作为长期支撑。Foot 因此新增 `pen_knee_contact_force=-2.0`：超过 `10 N` 的小腿力峰值按 `min((F-10)/100,3)^2` 逐腿惩罚，封顶防止冲击尖峰造成数值不稳定；任一小腿接触力 `>20 N` 连续 `0.08 s` 时终止 episode。原有二值非法接触惩罚继续保留，用于覆盖其他非轮部件及轻微接触。
- 2026-08-28 Wheel 奖励去重后已通过真实 Isaac Lab CPU 1-env / 3-step 冒烟，25 个奖励项均为有限值：删除 `pen_zero_command_yaw_rate` 和 `pen_wheel_air_time`，`stand_still` 调为 `-6.0`，滚动误差改为逐轮考虑实际 `wz`；支撑滑移保留。MuJoCo 无窗口冒烟确认会优先加载稳定性选择文件指定的 checkpoint。
- 2026-08-29 Foot 删除 `stand_still`、保留 `rew_leg_symmetry=+0.5`，并将力矩、加速度、动作变化/平滑、功率和腿关节速度正则权重对齐原有 PF；局部地形姿态调为 `-10.0`。高度改用与 Wheel 相同的局部平面法向定义和 `max(C_L,C_R)` 门控，保留 L2 `-30.0` 并新增指数正奖励 `+1.0`。步态三项采用 PF 的目的和权重：保留 `GaitReward=+1.0`，以逐轮局部地形版 `pen_feet_regulation=-0.1` 和 `foot_landing_vel=-0.5` 替换固定摆高和接触后冲击；第四维摆高输入固定为零。Foot 真实 Isaac Lab CPU 1-env 冒烟通过。
- Foot 正式训练已生成 `2026-08-17_21-27-09_foot_local_terrain_orientation_v1/model_20000.pt`；MuJoCo `--mode foot` 自动加载该模型并在执行器层硬锁左右轮目标速度，CPU 无窗口 200 步冒烟通过。
- Dual Play、统一网络接口和 Foot 轮速 mask 已通过既有冒烟验证。

## 文档导航

- 训练与运行：`wf_dual_mode_training_process.md`
- 当前奖励表：`wf_dual_mode_reward_penalty_table.md`
- 下一版轮态判断修改：`wf_new_mode_judgment_modification_plan.md`
