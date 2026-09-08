# WF 当前奖励与惩罚

> 本文描述当前代码。下一版 Wheel 简化方案见 `wf_new_mode_judgment_modification_plan.md`。

## 计算方式

```text
step_reward = Σ(term × weight × 0.02 s)
```

正权重通常是奖励，负权重是惩罚。

## 通用项

| 项目 | 权重 | 目的 |
|---|---:|---|
| 存活 `keep_balance` | `+1.0` | 延长 episode |
| 静止 `stand_still` | Wheel `-7.0`；Foot `--` | Wheel 在近零速度命令且轮地支撑时保持静止；Foot 已删除该项，与原有 PF 一致 |
| 线速度跟踪 | Wheel `+3.5`，`std²=0.12`；Foot `+3.0`，`std²=0.20` | 跟踪 `vx/vy`；Wheel 使用任意轮有效接地 `C_any` 门控并收紧误差核宽度 |
| yaw 跟踪 | Wheel `+1.5`，`std²=0.12`；Foot `+1.0`，`std²=0.25` | 跟踪 `wz`；Wheel 使用任意轮有效接地 `C_any` 门控并加强超调代价 |
| base 平面加速度 `pen_base_lin_acc_xy` | Wheel `-0.02`；Foot `--` | 弱抑制实际 `vx/vy` 高频波动；有界、跟踪误差门控、`C_any` 门控 |
| base yaw 加速度 `pen_base_yaw_acc` | Wheel `-0.02`；Foot `--` | 抑制实际 `wz` 高频波动；Charbonnier 鲁棒核、跟踪误差门控、`C_any` 门控 |
| 腿对称 | `+0.5` | 抑制明显不对称 |
| 竖直速度 | `-0.3` | Foot 全量生效；Wheel 在高度命令变化时平滑减弱，接近目标后恢复 |
| roll/pitch 角速度 | `-0.3` | 抑制摇晃 |
| 关节力矩 | Wheel `-1.6e-4`；Foot `-8e-5` | Foot 权重与原有 PF 一致 |
| 关节加速度 | Wheel `-1.5e-7`；Foot `-2.5e-7` | Foot 权重与原有 PF 一致 |
| 关节限位 | `-2.0` | 避免腿关节越界 |
| 功率 | Wheel `-2e-5`；Foot `-5e-4` | Foot 权重与原有 PF 一致 |
| base 接触 | termination | 机身触地立即结束 |
| 非超时终止惩罚 | `-500.0` | 使用当前步 `mdp.is_terminated`；Wheel 对应 `base_contact`，Foot 还包含持续小腿接触终止；乘 `0.02 s` 后一次约 `-10` |
| time out | termination | 20 秒正常结束 |

## Wheel 专用项

| 项目 | 权重 | 说明 |
|---|---:|---|
| 高度跟踪 | `-60.0` | 按轮地支撑置信度门控，跟踪局部地面相对高度 |
| 高度指数跟踪 | `+1.0` | 高斯核/RBF 奖励，`exp(-(h-h_cmd)^2/0.05^2)`；局部地形平面无效或腾空时不奖励 |
| 非轮关节速度 | `-0.08` | 抑制腿部快速运动 |
| 动作变化 | `-0.15` | 抑制动作突变 |
| 动作平滑 | `-0.08` | 抑制二阶抖动 |
| 非法接触 | `-1.0` | 惩罚腿部/base 碰撞 |
| 双轮有效接触缺失 | `-4.0` | 地面过滤力与轮下几何联合判断 |
| 水平中性点 | `-4.0` | 只惩罚 base 系 x/y 偏离 |
| 滚动误差 | `-2.0` | 按左右轮分别约束轮面速度与 `vx-wz×y_i`，支持差速转向 |
| 近零 yaw 轮目标对称 | `-0.5` | 左右轮目标差超过 `0.10 rad/s` 后惩罚 |
| 零命令轮目标 | `-0.10` | 全部速度命令近零时，轮目标超过 `0.15 rad/s` 后惩罚，允许平衡修正 |
| 地形姿态 | `-10.0` | 机身姿态跟随局部地形 |
| 落地冲击 | `-0.02` | 惩罚过大落地力 |
| 支撑滑移 | `-0.5` | 惩罚轮地接触点滑移 |

Wheel 的两个实际 base 加速度惩罚按 `50 Hz` policy 状态差分计算：

```text
a_xy(t)   = (v_b_xy(t) - v_b_xy(t-1)) / 0.02
alpha_z(t)= (omega_b_z(t) - omega_b_z(t-1)) / 0.02

G_xy = exp(-||c_xy-v_b_xy||^2 / 0.30^2)
G_wz = exp(-(c_yaw-omega_b_z)^2 / 0.30^2)

pen_base_lin_acc_xy = ||a_xy||^2 / (||a_xy||^2 + 3.0^2) × G_xy × C_any
pen_base_yaw_acc     = (sqrt(1 + alpha_z^2 / 4.0^2) - 1) × G_wz × C_any
```

XY 有界因子把未加权输出限制在 `[0,1)`；Yaw 的 Charbonnier 核在小加速度附近近似二次、在大加速度时近似线性，不会像旧饱和核那样在严重抖动区趋于常数。跟踪门控使速度指令刚变化、误差仍大时允许必要加速，接近目标后才重点压制振荡。episode/reset 后前两个 policy step 固定输出零。它们约束的是机器人实际 base 状态，不等同于已有的 action 一阶/二阶平滑。

速度和 yaw 正奖励使用每轮的地形过滤力与轮心几何联合置信度：

```text
max(left_contact_confidence, right_contact_confidence)
```

高度和静止项使用地面过滤力的软支撑门控。Wheel 高度 L2 与高度指数奖励使用：

```text
max(left_contact_confidence, right_contact_confidence)
```

因此双轮支撑、单轮支撑和腾空时的 Wheel 高度与 `stand_still` 力门控分别约为 `1.0`、`1.0` 和 `0.0`。速度/Yaw 使用力与几何联合后的 `C_any`，避免单轮短暂低置信度就撤掉整个跟踪信号。墙面等非地形碰撞不会开启门控。

TensorBoard 在 `Metrics/base_velocity/` 下额外记录左右轮的 `support_force_*`、`support_geometry_*`、`support_combined_*`，以及 `support_all/support_any`、双轮/单轮/无支撑比例、四类命令比例、`moving_command_rate` 和 XY/Yaw 门控前后的跟踪奖励。分模式指标新增 `straight_xy_error`、`yaw_only_wz_error`、`yaw_only_wz_acc_rms`、`mixed_xy_error` 和 `mixed_wz_error`；它们按对应模式的条件样本数归一化，纯转向加速度 RMS 跳过 episode reset 和命令重采样后的前两个 step，避免把正常阶跃响应当成稳态抖动。`Episode_Reward/pen_base_lin_acc_xy` 与 `Episode_Reward/pen_base_yaw_acc` 记录两个加速度惩罚。这些诊断不进入 policy/critic 观测；Wheel 地形课程仍只使用移动命令占比、未门控 XY 跟踪和 `support_any` 作为升降级依据。

Wheel 高度同时使用原有 L2 惩罚与新增指数奖励。L2 在大误差区仍提供恢复信号；指数项是基于平方误差的高斯核（RBF）跟踪奖励，在目标附近提供更密集的正向塑形。`std=0.05 m` 时，高度误差为 `0/5/10 cm` 的未加权奖励约为 `1.000/0.368/0.018`。

Wheel 的竖直速度惩罚根据采样目标与当前限速高度命令之间的差值缩放：差值不超过 `0.01 m` 时为 `100%`，达到 `0.04 m` 时降到 `20%`，中间线性过渡。因此换高度时允许必要的竖直运动，命令稳定后继续抑制上下振荡。

Wheel 的滚动误差使用轮心在 base 系中的实际横向位置 `y_i`：

```text
v_expected_i = abs(v_b_x - omega_b_z × y_i)
v_surface_i  = 0.128 × abs(qdot_i)
pen_rolling_error = mean_i((v_surface_i-v_expected_i)^2)
```

该项提供逐轮、密集的纵向滚动一致性约束；`pen_wheel_stance_slip=-0.5` 继续作为接地时的真实切向滑移约束，两项分工不同并同时保留。

Wheel 已关闭：

- 正常轮速 L2 和固定世界水平姿态惩罚；
- 独立的 `pen_zero_command_yaw_rate`，其静止约束由 `stand_still=-7.0` 覆盖；
- 逐轮 `pen_wheel_air_time`，离地仍由 `pen_wheel_contact` 和速度正奖励的任意轮有效接地 `C_any` 门控处理；
- 双轮腾空 termination；
- 旧 x/z 构型、轮距、轮轴法向、轮端竖直速度和 xyz 相对速度项。

Wheel 的速度命令采用互斥的四类采样：`25%` 站立（`vx=vy=wz=0`）、`30%` 直行（只采样 `vx`）、`10%` 纯转向（只采样 `wz`）、`35%` 混合（同时采样 `vx/wz`）；`vy` 始终为零。平移覆盖由上一版理论 `55%` 恢复到 `65%`，同时保留显式纯 `wz` 样本。高度训练范围为 `0.65～0.85 m`，目标采样中 `10%` 固定最小值、`10%` 固定最大值、`80%` 区间均匀采样。

## Foot 专用项

| 项目 | 权重 | 说明 |
|---|---:|---|
| 动作变化 | `-0.03` | 与原有 PF 一致，抑制一阶动作突变 |
| 动作平滑 | `-0.04` | 与原有 PF 一致，抑制二阶动作抖动 |
| 非轮关节速度 | `-0.001` | 与原有 PF 的全部关节速度 L2 权重一致；Foot 只选腿关节 |
| 轮关节速度 | `-0.10` | 抑制实际轮速 |
| 非法接触 | `-1.0` | 抑制非轮部件触地 |
| 小腿承重力 `pen_knee_contact_force` | `-2.0` | 每条小腿最近 4 帧接触力峰值超过 `10 N` 后，按 `min((F-10)/100,3)^2` 增长；封顶避免冲击尖峰破坏 PPO |
| 小腿持续接触终止 | termination | 任一小腿接触力连续 `0.08 s` 超过 `20 N` 时结束 episode；短暂擦碰不会终止 |
| 支撑端距离 | `-100` | 按 PF 使用 `0.115～1.0 m`，主要防止两轮过近，不限制正常前后步幅 |
| gait 接触时序 | `+1.0` | 内部返回接触/速度偏差 |
| 近地切向运动 `pen_feet_regulation` | `-0.1` | PF 式指数衰减正则；逐轮按局部地形法向间隙和切向速度计算，不指定摆动高度 |
| 双轮同时腾空 | `-4.0` | 保留至少一个支撑 |
| 接触前落地速度 `foot_landing_vel` | `-0.5` | 轮端接近自己的局部地面且尚未接触时，惩罚沿地形法向的向下速度 |
| 支撑滑移 | `-0.75` | 减小 stance 滑动 |
| 高度跟踪 | `-30.0` | 使用 121 点拟合局部地形平面，沿平面法向计算机身高度，并以 `max(C_L,C_R)` 门控 |
| 高度指数跟踪 | `+1.0` | 与 Wheel 相同的 `exp(-(h-h_cmd)^2/0.05^2)`，同样使用局部平面有效性和 `max(C_L,C_R)` 门控 |
| 局部地形姿态 | `-10.0` | 关闭世界水平约束，使 base-up 跟踪 `height_scanner` 拟合的局部地形法向 |

硬约束：

```text
Foot wheel action scale = 0
Foot wheel action offset = 0
```

因此轮速为零不只依赖 reward。

Foot 保留 `rew_leg_symmetry=+0.5`。`stand_still` 已删除；`pen_joint_torque`、`pen_joint_accel`、`pen_action_rate`、`pen_action_smoothness`、`pen_joint_power_l1` 和腿关节的 `pen_vel_non_wheel_l2` 均已对齐原有 PF。`pen_feet_distance` 使用 PF 的 `0.115～1.0 m` 范围，gait contact duration 固定为 PF 的 `0.5`。步态三项采用 PF 的目的和权重：`gait_contact_schedule=+1.0` 保持同一个 `GaitReward`，`pen_feet_regulation=-0.1` 和 `foot_landing_vel=-0.5` 使用适配坡面、高台的逐轮局部地形版本。

第四维 `swing_height` 输入固定为 `0.0`，仅为保持两个专家现有四维 gait schema；没有奖励把它当作目标。策略根据 121 维 base 地形扫描自行选择摆高，高抬腿的代价主要来自功率、力矩、关节速度和动作平滑正则。
