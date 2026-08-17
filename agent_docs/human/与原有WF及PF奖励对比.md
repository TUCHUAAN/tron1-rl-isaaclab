# 当前双专家方案与原有 WF 及 PF 奖励对比

> 对比时间：2026-08-16。
>
> 原有方案以 `/home/tuchuaan/open_source/tron1-rl-isaaclab-origin/RL_NETWORK_REWARD_RESET.md` 中的 `Isaac-Limx-WF-Blind-Flat-v0` 和 `Isaac-Limx-PF-Blind-Flat-v0` 为基准。
>
> 当前方案指新增的 `Isaac-Limx-WF-Wheel-Mode-v0` 和 `Isaac-Limx-WF-Foot-AllTerrain-v0`。原有 WF/PF Blind-Flat 任务在当前仓库中仍然保留；PF 作为纯点足步态参考，用于解释当前 Foot Expert 与原有足式训练的联系和差异。
>
> `Isaac-Limx-WF-Wheel-Height-Pretrain-v0` 直接继承 Wheel Expert 的全套奖励变量、权重和公式，只替换为平地预训练地形并关闭地形课程，因此不再单独列一套奖励。

## 1. 核心结论

| 项目 | 原有 WF Blind-Flat | 原有 PF Blind-Flat | 当前 Wheel Expert | 当前 Foot Expert |
|---|---|---|---|---|
| 策略定位 | 通用轮足策略 | 纯点足步态策略 | 独立轮式策略 | 独立轮足足式策略 |
| 主要目标 | 平地上跟踪速度并保持固定构型 | 通过 gait 交替摆动点足 | 双轮有效着地、滚动、转向、跟随地形与高度 | 轮目标速度锁零，按步态摆腿通过复杂地形 |
| 高度目标 | 固定 `0.80 m` | 固定 `0.65 m` | 可变 `0.65～0.85 m` | 可变 `0.65～0.85 m` |
| 速度正奖励 | `+3.0 / +1.0` | `+1.0 / +0.5` | `+3.0 / +1.0`，双轮接地门控 | `+3.0 / +1.0` |
| 支撑端约束 | 轮速和固定轮距 | 步态、足端轨迹和落脚速度 | 接地、滚动、滑移和水平中性点 | 轮动作硬置零、步态、摆动高度和滑移 |
| 世界水平姿态 | `-12.0` | `-10.0` | 删除，改为局部地形法向 `-8.0` | 保留 `-12.0` |
| 终止条件 | 20 s 超时或 base 触地 | 相同 | 相同 | 相同 |

这次改动的本质不是简单调整原有奖励权重，而是把原来的单一折中目标拆成两个互补的专家目标。

## 2. 奖励计算方式

四种任务都使用 Isaac Lab 的逐步加权求和：

```text
step_reward = Σ(term_value × weight × policy_dt)
policy_dt = 0.005 s × 4 = 0.02 s
```

正权重通常是奖励，负权重是惩罚。因为当前两个专家的地形、命令分布和奖励门控都已改变，不应只用 episode 总奖励直接与原有 Blind-Flat 模型排名。

### 2.1 总奖励的实际聚合

设第 `i` 个奖励函数的未加权输出为 `f_i(s, a)`，配置权重为 `w_i`，则环境每个 policy step 返回：

```text
R_step = 0.02 × Σ [w_i × f_i(s, a)]
```

RewardManager 先计算函数原值，再乘权重和 `dt`。因此：

- `pen_base_contact_termination=-500` 在一次当前步非 timeout 终止上对 `R_step` 的贡献是 `-500 × 1 × 0.02 = -10`。
- 表中权重不是“每步直接加的数”，还要乘 term value 和 `0.02`。
- `GaitReward` 是特例：外层权重是 `+1.0`，但函数内部的 force/velocity scale 都是 `-2.0`，所以 term value 本身可为负。

### 2.2 记号与当前接地门控

| 记号 | 含义 |
|---|---|
| `c_xy, c_yaw` | 平面线速度命令和 Yaw 角速度命令 |
| `v_b, omega_b` | base 坐标系线速度和角速度 |
| `a_t` | 当前原始 action |
| `u_L, u_R` | `joint_vel` action term 的左右轮 processed target |
| `relu(x)` | `max(x, 0)` |
| `I(condition)` | 条件成立为 `1`，否则为 `0` |

当前 Wheel/Foot 都有左右轮独立的地形过滤 ContactSensor。设最近 4 帧对地形 mesh 的力幅值峰值为 `F_peak_i`，则力接地置信度为：

```text
C_force_i = clip((F_peak_i - 5) / (10 - 5), 0, 1)
C_mean    = (C_force_L + C_force_R) / 2
```

`stand_still`、当前高度惩罚和 Wheel 零命令 Yaw 惩罚乘 `C_mean`：双轮支撑时完全生效，单轮支撑时约半强度，双轮腾空时为零。这三项只用地形过滤力，不乘轮心间隙几何置信度。

Wheel 速度正奖励和 `pen_wheel_contact` 还会增加几何检查。设轮心到轮下局部拟合地面的法向距离为 `d_i`，轮半径为 `0.128 m`：

```text
e_clear_i = abs(abs(d_i) - 0.128)
C_geom_i  = clip((0.035 - e_clear_i) / (0.035 - 0.020), 0, 1) × plane_valid
C_i       = C_force_i × C_geom_i
C_all     = min(C_L, C_R)
```

### 2.3 原有项和当前共享项的计算公式

下表给出未乘文档权重和 `dt` 前的 term value。

| 奖励变量 | 未加权计算方式 |
|---|---|
| `keep_balance` | 恒为 `1` |
| 原有 WF `stand_still` | `sum(abs(v_world_xy)) × I(norm(c_xy)<0.05) + abs(omega_world_z) × I(abs(c_yaw)<0.05)` |
| 当前 Wheel/Foot `stand_still` | 原有 `stand_still` 的值再乘 `C_mean` |
| 普通 `rew_lin_vel_xy` | `exp(-sum((c_xy-v_b_xy)^2) / std^2)`；WF/Foot `std^2=0.2`，PF `std^2=0.2` |
| 普通 `rew_ang_vel_z` | `exp(-(c_yaw-omega_b_z)^2 / std^2)`；WF/Foot `std^2=0.25`，PF `std^2=0.2` |
| `rew_leg_symmetry` | 先把左右支撑端位置转到 base 系；`exp(-((abs(y_L)-abs(y_R))^2)/0.5)` |
| `rew_same_foot_x_position` | `abs(x_L-x_R)`；其配置权重为负，所以实际是惩罚 |
| 普通 `pen_lin_vel_z` | `v_b_z^2` |
| `pen_ang_vel_xy` | `omega_b_x^2 + omega_b_y^2` |
| `pen_joint_torque` | `sum(tau_j^2)` |
| `pen_joint_accel` | `sum(qddot_j^2)` |
| `pen_action_rate` | `sum((a_t-a_(t-1))^2)` |
| `pen_action_smoothness` | `sum((a_t-2a_(t-1)+a_(t-2))^2)`；episode 前 3 步置零 |
| 关节位置限位 | `sum(relu(q_min-q) + relu(q-q_max))` |
| 非期望接触 | 计数：所选 body 中最近 4 帧力幅值峰值超过 `10 N` 的 body 数 |
| 世界水平姿态 | `projected_gravity_b_x^2 + projected_gravity_b_y^2` |
| `pen_feet_distance` | `clip(d_min-d_xy,0,1) + clip(d_xy-d_max,0,1)`；PF 未显式设 `d_max` 时使用默认 `1.0 m` |
| 原有 WF/PF `pen_base_height` | `abs(root_world_z-target_height)`；注意当前 `base_com_height` 实现是 L1 绝对值，不是平方 |
| `pen_joint_power_l1` / `pen_joint_powers` | `sum(abs(tau_j × qdot_j))` |
| 关节速度 L2 | `sum(qdot_j^2)`，具体关节由各配置的 `joint_names` 决定 |
| `pen_base_contact_termination` | `I(current_step_non_timeout_terminated)`；当前唯一非 timeout 终止是 `base_contact` |

### 2.4 Wheel Expert 的当前计算方式

| 奖励变量 | 未加权计算方式 |
|---|---|
| `rew_lin_vel_xy` | `exp(-sum((c_xy-v_b_xy)^2)/0.2) × C_all` |
| `rew_ang_vel_z` | `exp(-(c_yaw-omega_b_z)^2/0.25) × C_all` |
| `pen_base_height` | 121 个 base 扫描点拟合平面中心 `p_bar` 和单位法向 `n`；`(abs((p_base-p_bar) dot n)-h_cmd)^2 × plane_valid × C_mean` |
| `pen_lin_vel_z` | `v_b_z^2 × s_height`；设 `gap=abs(h_target-h_cmd)`，`gap<=0.01` 时 `s_height=1`，`gap>=0.04` 时为 `0.2`，中间线性插值 |
| `pen_wheel_contact` | `(1-C_L)+(1-C_R)`，这里 `C_i=C_force_i×C_geom_i` |
| `pen_wheel_horizontal_neutral` | 轮心转到 base 系，对 X/Y 超出中性点死区的归一化误差求 Huber 和；中性点 `(0,±0.17)`，死区 `(0.04,0.03)`，scale `(0.02,0.02)` |
| `pen_wheel_air_time` | `clip(t_air_L,0,1)^2 + clip(t_air_R,0,1)^2` |
| `pen_rolling_error` | `(0.128×mean(abs(qdot_wheel))-abs(v_b_x))^2` |
| `pen_wheel_target_symmetry` | `relu(abs(u_L-u_R)-0.10)^2 × I(abs(c_yaw)<=0.05)` |
| `pen_zero_command_wheel_target` | `[relu(abs(u_L)-0.15)^2+relu(abs(u_R)-0.15)^2] × I(norm(c_xy)<=0.05 and abs(c_yaw)<=0.05)` |
| `pen_zero_command_yaw_rate` | `omega_b_z^2 × I(norm(c_xy)<=0.05 and abs(c_yaw)<=0.05) × C_mean` |
| `pen_terrain_orientation` | `[1-(base_up dot terrain_normal)^2] × plane_valid` |
| `pen_wheel_landing_impact` | `sum_i I(first_contact_i) × [relu(norm(F_i)-100)/100]^2` |
| `pen_wheel_stance_slip` | 估计接触点速度 `v_contact=v_wheel-0.128×cross(omega_wheel,n)`，去掉法向分量后，对切向速度平方按接地置信度加权平均，再乘 `plane_valid` |

Wheel 的 `stand_still`、高度和零命令 Yaw 项使用 `C_mean`，允许腾空时不用这些项惩罚无支撑运动；腾空由 `pen_wheel_contact`、`pen_wheel_air_time`和终止风险另行处理。

### 2.5 Foot Expert 和原有 PF 步态项的计算方式

当前 Foot 的基础高度为：

```text
ground_z     = nanmedian(base_height_scanner 的 121 个命中点 Z)
actual_height = root_world_z - ground_z
pen_base_height = (actual_height-h_cmd)^2 × C_mean
```

Foot 的世界水平姿态、固定支撑端间距、普通竖直速度惩罚仍从 WF 继承，没有 Wheel 的局部地形法向替换。

| 奖励变量 | 未加权计算方式 |
|---|---|
| PF `pen_feet_regulation` | `sum_i exp(-foot_height_i/0.65) × norm(v_foot_xy_i)^2`，其中 `foot_height=clip(foot_world_z-0.03,0,1)` |
| PF `foot_landing_vel` | 足端未接触、高度低于 `0.08 m`且 Z 速度向下时，求 `sum(v_foot_z^2)` |
| PF/Foot gait term | 先根据 frequency、offset、contact duration 生成平滑期望接触 `d_i`；`force_part=(-2/N)×sum((1-d_i)×(1-exp(-F_i^2/25)))`，`velocity_part=(-2/N)×sum(d_i×(1-exp(-V_i^2/0.25)))`，term value 为两者之和 |
| Foot `pen_swing_height` | 用力置信度 `<0.5` 选摆动端；`wheel_clearance=wheel_world_z-median_ground_z-0.128`，对活跃摆动端平均 `(wheel_clearance-swing_height_cmd)^2` |
| Foot `pen_all_wheels_air_time` | `clip(min(t_air_L,t_air_R),0,1)^2`，因此只累计两个支撑端同时腾空的时间 |
| Foot `pen_wheel_landing_impact` | 与 Wheel 公式相同：只在首次接触时惩罚超过 `100 N` 的力 |
| Foot `pen_wheel_stance_slip` | 与 Wheel 公式相同，但配置权重是 `-0.75` |

需要区分两类接触数据：`stand_still`、高度门控、Wheel 零命令 Yaw 和 Wheel 有效接地使用每轮独立、只过滤地形 mesh 的 `wheel_L/R_ground_contact`；gait、摆动高度、air time、落地冲击和滑移使用全身 `contact_forces` 中选出的 `wheel_L/R_Link`，后者没有做地形 mesh 过滤。

## 3. 原有奖励项的逐项对比

`--` 表示该项在对应专家中已删除。

| 奖励/惩罚项 | 配置变量名（WF / PF / 当前） | 原有 WF | 原有 PF | Wheel | Foot | 主要变化 |
|---|---|---:|---:|---:|---:|---|
| 存活 | `keep_balance` | `+1.0` | `+1.0` | `+1.0` | `+1.0` | 全部相同 |
| 零命令静止 | `stand_still` | `-5.0` | -- | `-5.0` | `-5.0` | PF 无独立静止项；当前两专家增加平均地面支撑置信度门控 |
| XY 速度跟踪 | `rew_lin_vel_xy` | `+3.0` | `+1.0` | `+3.0` | `+3.0` | Wheel 另加双轮有效接地门控 |
| Yaw 速度跟踪 | `rew_ang_vel_z` | `+1.0` | `+0.5` | `+1.0` | `+1.0` | PF 跟踪权重较低 |
| 腿部对称 | `rew_leg_symmetry` | `+0.5` | -- | `+0.5` | `+0.5` | PF 无该项 |
| 左右轮 X 位置一致 | `rew_same_foot_x_position` | `-50.0` | -- | -- | -- | 两个专家均删除 |
| 竖直线速度 | `pen_lin_vel_z` | `-0.3` | `-0.5` | `-0.3` | `-0.3` | Wheel 在高度命令过渡期将实际惩罚缩放到 `0.2～1.0`；Foot/PF 仍为普通 L2 |
| Roll/Pitch 角速度 | `pen_ang_vel_xy` | `-0.3` | `-0.05` | `-0.3` | `-0.3` | PF 权重更低 |
| 关节力矩 | `pen_joint_torque` | `-1.6e-4` | `-8e-5` | `-1.6e-4` | `-1.6e-4` | PF 为 WF 系列的一半 |
| 关节加速度 | `pen_joint_accel` | `-1.5e-7` | `-2.5e-7` | `-1.5e-7` | `-1.5e-7` | PF 权重更大 |
| 动作变化率 | `pen_action_rate` | `-0.3` | `-0.03` | `-0.10` | `-0.3` | PF 最弱，Wheel 居中 |
| 动作二阶平滑 | `pen_action_smoothness` | `-0.03` | `-0.04` | `-0.05` | `-0.03` | Wheel 最强 |
| 非轮/全部关节位置限位 | `pen_non_wheel_pos_limits` / `pen_joint_pos_limits` | `-2.0` | `-2.0` | `-2.0` | `-2.0` | PF 约束其全部关节 |
| 非期望接触 | `undesired_contacts` / `pen_undesired_contacts` / `undesired_contacts` | `-0.25` | `-0.5` | `-1.0` | `-1.0` | 新专家权重最大 |
| 世界系水平姿态 | `pen_flat_orientation_l2` / `pen_flat_orientation` | `-12.0` | `-10.0` | -- | `-12.0` | Wheel 改用 `pen_terrain_orientation` |
| 左右支撑端间距 | `pen_feet_distance` | `-100` | `-100` | -- | `-100` | PF 仅设最小 `0.115 m`；WF/Foot 为 `0.32～0.35 m` |
| base 高度 | `pen_base_height` | `-30.0` | `-20.0` | `-60.0` | `-30.0` | PF 固定 `0.65 m`；新专家改为局部地面相对命令并加接地门控 |
| 关节功率 L1 | `pen_joint_power_l1` / `pen_joint_powers` | `-2e-5` | `-5e-4` | `-2e-5` | `-2e-5` | PF 功率惩罚大 25 倍 |
| 轮关节速度 L2 | `pen_joint_vel_wheel_l2` | `-0.005` | 不适用 | -- | `-0.10` | Foot 加强实际轮速惩罚 |
| 非轮/全部关节速度 L2 | `pen_vel_non_wheel_l2` / `pen_joint_vel_l2` | `-0.03` | `-0.001` | `-0.08` | `-0.03` | PF 项作用于模型全部 8 个关节，其中 6 个受控 |
| base 接触终止 | `pen_base_contact_termination` | -- | -- | `-500.0` | `-500.0` | 只在当前步非 timeout 终止时取 `1`；现配置中即 base 接触 |

### 3.1 高度奖励的变化

- 原有 WF：惩罚 base 质心高度偏离固定 `0.80 m`。
- 原有 PF：惩罚 base 质心高度偏离固定 `0.65 m`，不读取地形扫描。
- Wheel：从 base 高度扫描拟合局部地形平面，沿平面法向跟踪 `0.65～0.85 m` 的平滑命令，并乘以左右轮平均接地力置信度。
- Foot：使用高度扫描中位地面高度，跟踪 `0.65～0.85 m` 的平滑命令，同样乘以左右轮平均接地力置信度。

高度命令每 `5～8 s` 重采样；Wheel 最大变化速率为 `0.08 m/s`，Foot 为 `0.06 m/s`。每次重采样有 `10%` 概率精确取最小高度、`10%` 概率精确取最大高度，余下 `80%` 在全范围均匀采样。

### 3.2 奖励变量名与实现函数

“奖励变量名”是 `RewardsCfg` 中的配置字段，也是训练日志区分各奖励项时使用的 term name；`func` 才是计算该项数值的实现函数。

| 奖励变量名 | `func` | 使用任务 |
|---|---|---|
| `keep_balance` | `mdp.stay_alive` | WF、PF、Wheel、Foot |
| `stand_still` | `mdp.stand_still` | 原有 WF |
| `stand_still` | `mdp.stand_still_grounded` | Wheel、Foot |
| `rew_lin_vel_xy` | `mdp.track_lin_vel_xy_exp` | WF、PF、Foot |
| `rew_lin_vel_xy` | `mdp.track_lin_vel_xy_exp_all_wheels_ground` | Wheel |
| `rew_ang_vel_z` | `mdp.track_ang_vel_z_exp` | WF、PF、Foot |
| `rew_ang_vel_z` | `mdp.track_ang_vel_z_exp_all_wheels_ground` | Wheel |
| `rew_leg_symmetry` | `mdp.leg_symmetry` | WF、Wheel、Foot |
| `rew_same_foot_x_position` | `mdp.same_feet_x_position` | 仅原有 WF；Wheel/Foot 置为 `None` |
| `pen_lin_vel_z` | `mdp.lin_vel_z_l2` | WF、PF、Foot |
| `pen_lin_vel_z` | `mdp.lin_vel_z_height_command_gated_l2` | Wheel |
| `pen_ang_vel_xy` | `mdp.ang_vel_xy_l2` | 全部 |
| `pen_joint_torque` | `mdp.joint_torques_l2` | 全部 |
| `pen_joint_accel` | `mdp.joint_acc_l2` | 全部 |
| `pen_action_rate` | `mdp.action_rate_l2` | 全部 |
| `pen_action_smoothness` | `mdp.ActionSmoothnessPenalty` | 全部 |
| `pen_non_wheel_pos_limits` | `mdp.joint_pos_limits` | WF、Wheel、Foot |
| `pen_joint_pos_limits` | `mdp.joint_pos_limits` | PF |
| `undesired_contacts` | `mdp.undesired_contacts` | WF、Wheel、Foot |
| `pen_undesired_contacts` | `mdp.undesired_contacts` | PF |
| `pen_flat_orientation_l2` | `mdp.flat_orientation_l2` | WF、Foot；Wheel 置为 `None` |
| `pen_flat_orientation` | `mdp.flat_orientation_l2` | PF |
| `pen_feet_distance` | `mdp.feet_distance` | WF、PF、Foot；Wheel 置为 `None` |
| `pen_base_height` | `mdp.base_com_height` | WF、PF |
| `pen_base_height` | `mdp.body_height_command_plane_grounded_l2` | Wheel |
| `pen_base_height` | `mdp.body_height_command_grounded_l2` | Foot |
| `pen_base_contact_termination` | `mdp.is_terminated` | Wheel、Foot |
| `pen_joint_power_l1` | `mdp.joint_powers_l1` | WF、Wheel、Foot |
| `pen_joint_powers` | `mdp.joint_powers_l1` | PF |
| `pen_joint_vel_wheel_l2` | `mdp.joint_vel_l2` | WF、Foot；Wheel 置为 `None` |
| `pen_vel_non_wheel_l2` | `mdp.joint_vel_l2` | WF、Wheel、Foot |
| `pen_joint_vel_l2` | `mdp.joint_vel_l2` | PF |
| `pen_feet_regulation` | `mdp.feet_regulation` | PF |
| `foot_landing_vel` | `mdp.foot_landing_vel` | PF |
| `test_gait_reward` | `mdp.GaitReward` | PF |
| `gait_contact_schedule` | `mdp.GaitReward` | Foot |
| `pen_wheel_contact` | `mdp.wheel_ground_contact_loss` | Wheel |
| `pen_wheel_horizontal_neutral` | `mdp.wheel_horizontal_neutral_l2` | Wheel |
| `pen_wheel_air_time` | `mdp.wheel_air_time_l2` | Wheel |
| `pen_rolling_error` | `mdp.wheel_rolling_velocity_error` | Wheel |
| `pen_wheel_target_symmetry` | `mdp.wheel_target_symmetry_l2` | Wheel |
| `pen_zero_command_wheel_target` | `mdp.zero_command_wheel_target_l2` | Wheel |
| `pen_zero_command_yaw_rate` | `mdp.zero_command_yaw_rate_grounded_l2` | Wheel |
| `pen_terrain_orientation` | `mdp.terrain_aligned_orientation_l2` | Wheel |
| `pen_swing_height` | `mdp.wheel_swing_height_tracking` | Foot |
| `pen_all_wheels_air_time` | `mdp.all_wheels_air_time_l2` | Foot |
| `pen_wheel_landing_impact` | `mdp.wheel_landing_impact_l2` | Wheel、Foot |
| `pen_wheel_stance_slip` | `mdp.wheel_stance_slip_l2` | Wheel、Foot |

## 4. PF 与 Foot Expert 的步态奖励对比

| 项目 | 原有 PF 变量名 | 原有 PF | 当前 Foot 变量名 | 当前 Foot | 差异 |
|---|---|---:|---|---:|---|
| 步态接触/摆动塑形 | `test_gait_reward` | `+1.0` | `gait_contact_schedule` | `+1.0` | 两者都使用 `GaitReward` |
| 足端/轮端轨迹约束 | `pen_feet_regulation` | `-0.1` | `pen_swing_height` | `-4.0` | PF 用 `feet_regulation`；Foot 直接跟踪局部地面离地高度 |
| 落脚约束 | `foot_landing_vel` | `-0.5` | `pen_wheel_landing_impact` | `-0.02` | PF 惩罚着地前速度；Foot 惩罚新接触的大力 |
| 双支撑端同时腾空 | -- | -- | `pen_all_wheels_air_time` | `-4.0` | Foot 额外要求至少保留一个支撑 |
| 支撑期滑移 | -- | -- | `pen_wheel_stance_slip` | `-0.75` | Foot 针对轮形支撑端增加滑移约束 |
| 实际轮速 | -- | -- | `pen_joint_vel_wheel_l2` | `-0.10` | Foot 需要抑制未受动作驱动但可被接触带动的轮子 |

PF 和 Foot 都使用相位差 `0.5` 的交替步态，但它们不是同一个动作问题：PF 是无轮关节的点足机器人，Foot Expert 仍使用轮足机构，必须同时处理轮子转动、滑移和更复杂的地形。

| Gait command | 原有 PF | 当前 Foot |
|---|---:|---:|
| 重采样间隔 | 固定 `5 s` | `5～8 s` |
| 频率 | `1.5～2.5 Hz` | `1.2～2.2 Hz` |
| 相位差 | `0.5` | `0.5` |
| 接触持续比例 | 固定 `0.5` | `0.45～0.60` |
| 摆动高度 | `0.10～0.20 m` | `0.08～0.20 m` |

## 5. Wheel Expert 新增奖励

| 项目 | 奖励变量名 | 权重 | 含义 |
|---|---|---:|---|
| 双轮有效接地缺失 | `pen_wheel_contact` | `-4.0` | 按左右轮缺失的连续接地置信度求和 |
| 水平中性点 | `pen_wheel_horizontal_neutral` | `-4.0` | 约束 base 坐标系中轮心 X/Y，不约束 Z |
| 单轮腾空时间 | `pen_wheel_air_time` | `-8.0` | 分别惩罚每个轮子的当前腾空时间，上限 `1 s` |
| 滚动速度误差 | `pen_rolling_error` | `-2.0` | 惩罚 `mean(abs(wheel_speed)) × 0.128` 与 base 前向速度不匹配 |
| 无 Yaw 命令时左右轮目标对称 | `pen_wheel_target_symmetry` | `-0.5` | `abs(wz_cmd) ≤ 0.05` 时，惩罚左右轮目标差超过 `0.10` |
| 零速度命令时的轮目标 | `pen_zero_command_wheel_target` | `-0.10` | 线速度和 Yaw 命令均接近零时，惩罚轮目标超出 `0.15` |
| 零速度命令时的实际 Yaw 速度 | `pen_zero_command_yaw_rate` | `-2.0` | 命令全部接近零时惩罚 `omega_b_z^2`，并乘平均地面支撑置信度 |
| 局部地形姿态 | `pen_terrain_orientation` | `-8.0` | 惩罚 base-up 方向偏离局部地形法向 |
| 落地冲击 | `pen_wheel_landing_impact` | `-0.02` | 新建立接触时，惩罚超过 `100 N` 的力 |
| 支撑期滑移 | `pen_wheel_stance_slip` | `-0.5` | 惩罚轮地接触点的切向滑动 |

### 5.1 双轮有效接地的判定

每个轮子的有效接地置信度由两部分相乘：

```text
contact_confidence = ground_filtered_force_confidence
                   × wheel_clearance_geometry_confidence
all_contact_confidence = min(left_confidence, right_confidence)
```

- 接触力只过滤地形 mesh，避免把自碰撞当成地面支撑。
- 力阈值为 `5 N` 关闭、`10 N` 完全开启，使用 4 帧历史最大值抑制单帧掉点。
- 轮心至局部地面距离应接近轮半径 `0.128 m`；几何容差为 `0.020/0.035 m`。
- XY 速度和 Yaw 跟踪正奖励都乘以 `all_contact_confidence`。任意一轮失联都会同时降低正奖励并产生接地缺失惩罚。

### 5.2 水平中性点

```text
left neutral  = (0.0,  0.17) m
right neutral = (0.0, -0.17) m
tolerance x/y = (0.04, 0.03) m
scale x/y     = (0.02, 0.02) m
```

超出容差死区后使用 Huber 形式惩罚。这一项替代了原有的左右轮 X 位置一致和固定轮距约束，同时允许腿长沿 Z 方向调节。

## 6. Foot Expert 相对原有 WF 新增的奖励

| 项目 | 奖励变量名 | 权重 | 含义 |
|---|---|---:|---|
| 步态接触时序 | `gait_contact_schedule` | `+1.0` | 按步态相位塑形轮端接触力和摆动速度 |
| 摆动高度 | `pen_swing_height` | `-4.0` | 摆动期跟踪 gait command 中的离地高度 |
| 双轮同时腾空 | `pen_all_wheels_air_time` | `-4.0` | 只惩罚左右轮同时无支撑的持续时间 |
| 落地冲击 | `pen_wheel_landing_impact` | `-0.02` | 减小新建立接触时的冲击 |
| 支撑期滑移 | `pen_wheel_stance_slip` | `-0.75` | 减小支撑轮端滑动，比 Wheel 的权重更大 |

Foot 的 gait command 每 `5～8 s` 重采样：

```text
frequency   = 1.2～2.2 Hz
phase offset = 0.5
contact duration = 0.45～0.60
swing height = 0.08～0.20 m
```

Wheel 也保留相同的 gait observation schema，但 Wheel 奖励不使用 gait 时序项。这是为了让两个专家 checkpoint 拥有一致的输入形状。

## 7. 奖励所依赖的网络接口变化

| 项目 | 原有 WF | 原有 PF | Wheel / Foot |
|---|---:|---:|---:|
| 当前 policy observation | `28` | `34` | `155 = 28 + 121 高度扫描 + 2 步态相位 + 4 步态命令` |
| 单帧 history observation | `28` | `34` | `34 = 28 + 2 步态相位 + 4 步态命令` |
| 10 帧历史编码器输入 | `280` | `340` | `340` |
| 命令组 | `3` 维速度 | `3` 维速度 | `4` 维：`vx, vy, wz, body_height` |
| Encoder latent | `3` | `3` | `3` |
| Actor 总输入 | `34` | `40` | `162` |
| Actor 输出 | `8` | `6` | `8` |

当前 RSL-RL runner 不再假定 `history_dim = 10 × policy_obs_dim`，而是从实际 `obsHistory` 组读取 `340` 维。高度扫描只进入当前 policy/critic observation，不进入 10 帧历史。

因输入尺寸已改变，原有 `WF-Blind-Flat` 的 34 维 Actor checkpoint 不能直接当作当前双专家 checkpoint 加载。PF 虽然与当前 Foot 共享 340 维历史输入，但 Actor 输入分别为 40/162 维且动作输出为 6/8 维，PF checkpoint 同样不能直接加载到 Foot Expert。当前文档中提到的 Wheel `model_20000.pt` 是新 Wheel task/schema 在旧版 Wheel reward 下训练的 baseline，不是原仓库 `WF-Blind-Flat` checkpoint。

## 8. 动作硬约束与终止条件

### 8.1 动作接口

原有 WF、Wheel 和 Foot 输出 8 维动作：

```text
6 个腿关节位置目标 + 2 个轮关节速度目标
```

- 原有 WF 和 Wheel：轮动作 `scale=1.0`，正常生效。
- 原有 PF：只输出 6 个腿关节位置目标，没有轮关节和轮速动作。
- Foot 训练环境：轮动作 `scale=0.0, offset=0.0`，经处理的轮速目标始终为零。
- Dual Play 中：Foot 模式和过渡阶段由外部 FSM 再次 mask 最后两维，因此 Foot 的零轮速不只依赖惩罚项学出。

### 8.2 终止条件

| 终止项 | 原有 WF | 原有 PF | Wheel | Foot |
|---|:---:|:---:|:---:|:---:|
| 20 s `time_out` | ✓ | ✓ | ✓ | ✓ |
| `base_Link` 接触力超过 `1.0` | ✓ | ✓ | ✓ | ✓ |
| 支撑端腾空终止 | -- | -- | -- | -- |

双轮腾空在当前两个专家中都只是惩罚，不会直接结束 episode。

## 9. 设计影响

1. **Wheel 从“跑得快”改为“双轮有效支撑时跑得准”。** 接地门控防止策略通过腾空或非法接触仍获得大量速度正奖励。
2. **Wheel 的构型约束从多个间接项改为可解释的直接条件。** 中性点只约束 X/Y，高度可由腿长命令独立调节。
3. **Foot 从“慢轮式”变为真正的足式控制。** 轮目标速度的硬 mask 和 gait/swing reward 一起强制策略依靠腿部运动。
4. **两个专家不再用同一奖励函数折中。** 代价是需要分别训练两个 checkpoint，并在推理时由 FSM 负责切换。
5. **Foot 保留了世界水平姿态和固定轮距惩罚。** 它们在斜坡或大步态场景中可能与地形跟随目标产生竞争，分析 Foot 训练结果时应单独观察这两项。
6. **PF 提供了 Foot gait 的基础参考，但不能直接代替 Foot Expert。** Foot 在 PF 式交替步态之上，还需处理轮形支撑端、实际轮速、地形高度和楼梯。

## 10. 主要代码位置

| 内容 | 文件 |
|---|---|
| 原有 WF 共享奖励和终止 | `exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/cfg/WF/limx_base_env_cfg.py` |
| 原有 PF 奖励、观测和终止 | `exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/cfg/PF/limx_base_env_cfg.py` |
| Wheel / Foot 最终生效权重 | `exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/robots/limx_wheelfoot_mode_env_cfg.py` |
| 新奖励 adapter | `exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/rewards.py` |
| 新奖励纯张量数学 | `exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/reward_math.py` |
| 高度命令 | `exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/commands/body_height_command.py` |
| 网络和历史编码器 | `rsl_rl/rsl_rl/runner/on_policy_runner.py` |
| 双专家切换和动作 mask | `exts/bipedal_locomotion/bipedal_locomotion/utils/wf_mode_fsm.py` |
