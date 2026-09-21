# WF 当前奖励与惩罚

> **当前配置：v12，文档更新于 2026-09-21。** 当前表格与执行器说明已同步；明确标注历史的 v5/v6/v7 数据仅用于版本对照。

> 本文描述截至 2026-09-21 的当前代码。Wheel 合规奖励的历史设计记录见 `wf_new_mode_judgment_modification_plan.md`；其参数不代表当前配置。

当前为 **v12**：Foot/Wheel 的 policy 和 critic 均追加六维零命令保持误差/启用标记，与奖励共用同一 tracker；MuJoCo 同步支持。policy 161D、Actor 168D，history/Encoder 仍 340D。Foot 保留 v11 的 2 cm 中段最低净空成本 −1.0、漏迈 −0.2/次及此前跟踪权重。**从头训练，不微调**；详见 [v12 观测与训练说明](wf_hold_observation_v12.md)。

## 计算方式

```text
step_reward = Σ(term × weight × 0.02 s)
```

正权重通常是奖励，负权重是惩罚。

事件项 `pen_foothold_region` 与 `pen_missed_swing` 返回值除以 dt，使配置权重按每次事件计；其余项仍按上述公式。

## 通用项

| 项目 | 权重 | 目的 |
|---|---:|---|
| 分量零命令保持 `pen_zero_command_hold` | Wheel/Foot 均 `−0.5` | 分量独立启用；全零锁世界位置，wz=0 锁航向；[细节](wf_zero_hold_swing_v10.md) |
| 存活 `keep_balance` | `+1.0` | 延长 episode |
| 静止 `stand_still` | Wheel `-7.0`；Foot `--` | Wheel 在近零速度命令且轮地支撑时保持静止；Foot 已删除该项，与原有 PF 一致 |
| 线速度跟踪 | Wheel `+3.5`，`std²=0.12`；Foot `+5.5`，`std²=0.20` | 跟踪 `vx/vy`；Wheel 使用任意轮有效接地 `C_any` 门控并收紧误差核宽度 |
| yaw 跟踪 | Wheel `+1.5`，`std²=0.12`；Foot `+3.0`，`std²=0.25` | 跟踪 `wz`；Wheel 使用任意轮有效接地 `C_any` 门控并加强超调代价 |
| base 平面加速度 `pen_base_lin_acc_xy` | Wheel / Foot `0.0` | 均关闭；配置保留，RewardManager 跳过计算 |
| base yaw 加速度 `pen_base_yaw_acc` | Wheel `-0.025`；Foot `0.0` | Wheel 保持 Charbonnier 与支撑/跟踪门控；Foot 关闭 |
| vx 周期平均误差 `pen_cycle_mean_vx` | Foot `-1.0` | 满一个实际步态周期后，对历史跟踪误差均值做 Huber；尺度 `0.2 m/s`；无零命令门控 |
| vy 周期平均误差 `pen_cycle_mean_vy` | Foot `-1.0` | 满一个实际步态周期后，对历史跟踪误差均值做 Huber；尺度 `0.2 m/s`；无零命令门控 |
| yaw 周期平均误差 `pen_cycle_mean_yaw` | Foot `-1.0` | 满一个实际步态周期后，对历史跟踪误差均值做 Huber；尺度 `0.3 rad/s`；无零命令门控 |
| height 周期平均误差 `pen_cycle_mean_height` | Foot `-0.1` | 满一个实际步态周期后，对历史跟踪误差均值做 Huber；尺度 `0.02 m`；无零命令门控 |
| roll 周期平均误差 `pen_cycle_mean_roll` | Foot `-0.1` | 满一个实际步态周期后，对历史跟踪误差均值做 Huber；尺度 `3°`；无零命令门控 |
| pitch 周期平均误差 `pen_cycle_mean_pitch` | Foot `-0.1` | 满一个实际步态周期后，对历史跟踪误差均值做 Huber；尺度 `3°`；无零命令门控 |
| yaw 大误差 `pen_yaw_tracking_error` | Foot `-1.0` | 当前偏航速度误差的 Huber cost，尺度 `0.5 rad/s`；无积分状态 |
| XY 大误差 `pen_lin_vel_xy_tracking_error` | Foot `-0.5` | 当前机体系平面速度误差范数的 Huber cost，尺度 `0.4 m/s`；无积分状态，零命令也生效 |
| 腿对称 | `+0.5` | 抑制明显不对称 |
| 竖直速度 | `-0.3` | Foot 全量生效；Wheel 在高度命令变化时平滑减弱，接近目标后恢复 |
| roll/pitch 角速度 | `-0.3` | 抑制摇晃 |
| 关节力矩 | Wheel `-1.6e-4`；Foot `-8e-5` | Foot 只统计六腿，Wheel 保持原范围 |
| 关节加速度 | Wheel `-1.5e-7`；Foot `-2.5e-7` | Foot 只统计六腿，Wheel 保持原范围 |
| 关节限位 | `-2.0` | 避免腿关节越界 |
| 功率 | Wheel `-2e-5`；Foot `-5e-4` | Foot 只对齐原有 PF 的权重；当前 `joint_powers_l1` 对全 8 关节求和，传入的关节选择不会缩小统计范围 |
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

Wheel 保留两个实际 base 加速度函数配置，按 `50 Hz` policy 状态差分计算：

```text
a_xy(t)   = (v_b_xy(t) - v_b_xy(t-1)) / 0.02
alpha_z(t)= (omega_b_z(t) - omega_b_z(t-1)) / 0.02

G_xy = exp(-||c_xy-v_b_xy||^2 / 0.30^2)
G_wz = exp(-(c_yaw-omega_b_z)^2 / 0.30^2)

pen_base_lin_acc_xy = ||a_xy||^2 / (||a_xy||^2 + 3.0^2) × G_xy × C_any
pen_base_yaw_acc     = (sqrt(1 + alpha_z^2 / 4.0^2) - 1) × G_wz × C_any
```

XY 有界因子把未加权输出限制在 `[0,1)`，但当前权重为 `0.0`，RewardManager 会直接跳过计算并记录为零。Yaw 的 Charbonnier 核保持启用，权重为 `-0.025`；它在小加速度附近近似二次、在大加速度时近似线性。跟踪门控使速度指令刚变化、误差仍大时允许必要加速，接近目标后才重点压制振荡。激活的 Yaw 项在 episode/reset 后前两个 policy step 固定输出零。

Wheel 的网络后两维仍是左右轮目标速度，但训练执行器改为 `200 Hz` 显式 PI 力矩控制：

```text
e_i(t) = omega_target_i(t) - omega_i(t)
E_i(t) = E_i(t-1) + e_i(t) × 0.005
tau_i  = clip(Kp × e_i + Ki × E_i, -80, 80)
```

同一环境的左右轮共用一组增益。每次 reset 采样 `Kp=2.0×Uniform(0.25,2.0)`，即 `0.5～4.0 N·m/(rad/s)`；再采样 `Ki/Kp=Uniform(0,0.25) 1/s`。积分状态在 reset 时清零，力矩饱和且误差继续推向饱和方向时暂停积分。Wheel 的 PhysX 隐式 stiffness/damping 置零，通用执行器增益随机化仅作用于 6 个腿关节，避免与 PI 重复叠加。

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

## Foot 专用项（v12，奖励沿用 v11）

| 项目 | 权重 | 说明 |
|---|---:|---|
| 动作变化 / 二阶平滑 | −0.03 / −0.04 | 只计算六维腿动作，忽略已失效的两维轮输出 |
| 腿关节速度 | −0.001 | 仅六腿 |
| 轮实际速度 | 0 | 停用，轮速参考仍固定 0，由 PI 控制 |
| 非法接触 | −1 | 抑制非轮部件触地 |
| 小腿承重力 | −2 | 超过 10 N 的力按有上限的二次项处罚 |
| 小腿持续承重终止 | termination | 超过 20 N 持续 0.08 s 终止 |
| 横向站宽 | −1 | 0.30～0.38 m 区间外按 0.05 m 尺度做 Huber |
| gait 接触时序 | +1 外层 | 摆动脚承重内部 −6、Huber 尺度 100 N；支撑轮心速度内部 −2 |
| 近地切向运动 | −0.1 | 净空衰减尺度 0.05 m |
| 摆动净空 | +2 | 每周期固定 5/10 cm 峰值，实时摆线高度，指数核 std=0.025 m |
| 落脚区域事件成本 | −0.2/次 | 起脚锁定世界系椭圆，实际离地后首次稳定落地时结算；平地/缓坡、扫描及粗略可达性过滤 |
| 摆动中段最低净空 `pen_swing_min_clearance` | −1.0 | 每脚独立的 2 cm 短缺成本，摆动中段启用、边界平滑，不受另一脚支撑门控 |
| 每脚漏迈 `pen_missed_swing` | −0.2/次 | 完整摆动结束仍未有效离地才处罚；左右独立，实际支撑后净空≥1 cm、无接触≥60 ms |
| 计划支撑接触缺失 | −1 | `mean(planned_contact*(1-contact_confidence)^2)`，无 moving 门控 |
| 接触前落地速度 | −0.5 | 仅净空<2 cm、未接触且下降时，处罚超过 0.2 m/s 的部分平方 |
| 支撑滑移 | −0.75 | 轮地接触点切向滑移 |
| 高度 L2 / 指数跟踪 | −30 / +1.5 | 局部地面法向高度；指数 std=0.05 m；任意轮支撑门控 |
| 地形姿态 | −10 | base-up 对齐局部地面法向 |
| 腿部横向对称 | +0.5 | 不要求两脚前后位置对称 |

轮参考恒为 0。训练 PI 增益随机化与 Play/MuJoCo 的 Kp=2、Ki=0.5 保持，实际轮速不保证瞬时严格为零。删除轮目标惩罚和双脚腾空时间平方，轮端滑移与支撑轮心速度项继续保留。

### 周期两档净空与实时高度

从 121 点扫描拟合局部坡面，取相对平面的法向残差极差 D。初始 D>4 cm 选高档，否则选低档；后续仅在完整周期边界判断：低档 D>4.5 cm 升档，高档 D<3.5 cm 降档，中间保持。左右脚共享该周期峰值，不做周期内快升慢降滤波。

```text
H = 0.05 or 0.10 m  # latched per full gait cycle
E = sin²(pi*u)
h_ref = H*E = H/2*(1-cos(2*pi*u))
reward = 2 * sum(exp(-((h-h_ref)/0.025)^2)*E*C_opposite*valid) / max(planned_swing_count,1)
```

H 的确定与高度轨迹不同：峰值一周期选一次，u 和高度目标每步更新。无效扫描保持峰值、停用该周期净空评分；下周期重试。D 只是一种粗略起伏度，不是精确台阶高度或摆脚路径的最大障碍高度。

### 接触与落地

计划接触与 gait 项共用相位和平滑函数，换脚边界软交接。实际支撑使用地形过滤接触力历史，5～10 N 连续置信度。全腾空时支撑缺失成本通常约 0.5，乘权重 −1 后再乘统一 dt，不依赖腾空时间积累。

单脚计划摆动、另一脚计划支撑时，摆动脚承重 100/50/30/20/0 N 对力项的贡献为 −1.5/−0.375/−0.135/−0.06/0，尚未乘 dt。该项与新增支撑缺失互补。

落地成本为 `sum(relu(-v_normal-0.2)^2)`，仅在局部净空<0.02 m、未接触、下降且扫描有效时计算。原来低于 8 cm 就处罚全部下降速度的 Foot 规则已替换；原 PF 保持原规则。

### 命令与版本状态

Foot 仍为原地 15%、直行 25%、侧移 10%、纯转向 20%、混合 30%；连续相位、占空比 0.5、左右相差半周期保持，零命令也迈步。v9 已训练，用户反馈改善但仍有单腿漏迈、偏航及漂移；v10 尚未训练验收。新增落脚区域见 [v9 说明](wf_foot_foothold_v9.md)。历史配置和结果见奖励对比第 4.1 节；实现与验证见 [v8 说明](wf_foot_gait_v8.md)。
