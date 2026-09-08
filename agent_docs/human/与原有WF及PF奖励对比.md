# 当前双专家方案与原有 WF 及 PF 奖励对比

> 对比时间：2026-09-02。
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
| 速度正奖励 | `+3.0 / +1.0` | `+1.0 / +0.5` | `+3.5 / +1.5`，任意轮有效接地 `C_any` 门控 | `+3.0 / +1.0` |
| 实际 base 加速度抑制 | -- | -- | XY/Yaw 各 `-0.02`；XY 有界核，Yaw Charbonnier 核，均带近目标＋`C_any` 门控 | -- |
| 支撑端约束 | 轮速和固定轮距 | 步态、足端轨迹和落脚速度 | 接地、滚动、滑移和水平中性点 | 轮动作硬置零、步态、地形自适应摆腿和滑移 |
| 世界水平姿态 | `-12.0` | `-10.0` | 删除，改为局部地形法向 `-10.0` | 删除，改为局部地形法向 `-10.0` |
| 终止条件 | 20 s 超时或 base 触地 | 相同 | 相同 | Foot 另加持续小腿接触终止 |

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
C_force_any = max(C_force_L, C_force_R)
```

Wheel 的 `stand_still` 与 Wheel/Foot 的高度 L2、高度指数正奖励都乘 `C_force_any`：只要任意一轮有效支撑就按最强支撑置信度完整生效，双轮都腾空时才为零。Foot 已删除 `stand_still`。这些门控只用地形过滤力，不乘轮心间隙几何置信度。

Wheel 速度正奖励和 `pen_wheel_contact` 还会增加几何检查。设轮心到轮下局部拟合地面的法向距离为 `d_i`，轮半径为 `0.128 m`：

```text
e_clear_i = abs(abs(d_i) - 0.128)
C_geom_i  = clip((0.035 - e_clear_i) / (0.035 - 0.020), 0, 1) × plane_valid
C_i       = C_force_i × C_geom_i
C_all     = min(C_L, C_R)
C_any     = max(C_L, C_R)
```

`pen_wheel_contact` 仍逐轮使用 `C_L/C_R`。Wheel 的 XY 速度和 Yaw 跟踪现在使用组合置信度 `C_any`：任意一轮有效落地即保留跟踪信号，仅双轮均无有效支撑时关闭。`C_all` 不再参与这两项奖励，仅作训练诊断指标保留。

### 2.3 原有项和当前共享项的计算公式

下表给出未乘文档权重和 `dt` 前的 term value。

| 奖励变量 | 未加权计算方式 |
|---|---|
| `keep_balance` | 恒为 `1` |
| 原有 WF `stand_still` | `sum(abs(v_world_xy)) × I(norm(c_xy)<0.05) + abs(omega_world_z) × I(abs(c_yaw)<0.05)` |
| 当前 Wheel `stand_still` | 原有 `stand_still` 的值再乘 `C_force_any`；Foot 已删除该项 |
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
| `pen_base_contact_termination` | `I(current_step_non_timeout_terminated)`；Wheel 对应 `base_contact`，Foot 还包含持续小腿接触终止 |
| Wheel `pen_base_lin_acc_xy` | `a_xy=(v_b_xy(t)-v_b_xy(t-1))/0.02`；`[||a_xy||²/(||a_xy||²+3²)] × exp(-||c_xy-v_b_xy||²/0.30²) × C_any` |
| Wheel `pen_base_yaw_acc` | `alpha_z=(omega_b_z(t)-omega_b_z(t-1))/0.02`；`[sqrt(1+alpha_z²/4²)-1] × exp(-(c_yaw-omega_b_z)²/0.30²) × C_any` |

### 2.4 Wheel Expert 的当前计算方式

| 奖励变量 | 未加权计算方式 |
|---|---|
| `rew_lin_vel_xy` | `exp(-sum((c_xy-v_b_xy)^2)/0.12) × C_any` |
| `rew_ang_vel_z` | `exp(-(c_yaw-omega_b_z)^2/0.12) × C_any` |
| `pen_base_lin_acc_xy` | 实际 base 平面速度的 50 Hz 差分平方经过 `3.0 m/s²` 饱和核，再乘 `std=0.30` 的 XY 跟踪门控和 `C_any`；reset 后前两步为零 |
| `pen_base_yaw_acc` | 实际 base yaw rate 的 50 Hz 差分使用尺度 `4.0 rad/s²` 的 Charbonnier 核，再乘 `std=0.30` 的 Yaw 跟踪门控和 `C_any`；reset 后前两步为零 |
| `pen_base_height` | 121 个 base 扫描点拟合平面中心 `p_bar` 和单位法向 `n`；`(abs((p_base-p_bar) dot n)-h_cmd)^2 × plane_valid × C_any` |
| `rew_base_height_exp` | `exp(-(abs((p_base-p_bar) dot n)-h_cmd)^2/0.05^2) × plane_valid × C_any` |
| `pen_lin_vel_z` | `v_b_z^2 × s_height`；设 `gap=abs(h_target-h_cmd)`，`gap<=0.01` 时 `s_height=1`，`gap>=0.04` 时为 `0.2`，中间线性插值 |
| `pen_wheel_contact` | `(1-C_L)+(1-C_R)`，这里 `C_i=C_force_i×C_geom_i` |
| `pen_wheel_horizontal_neutral` | 轮心转到 base 系，对 X/Y 超出中性点死区的归一化误差求 Huber 和；中性点 `(0,±0.17)`，死区 `(0.04,0.03)`，scale `(0.02,0.02)` |
| `pen_rolling_error` | 设轮心 base 系横向位置为 `y_i`；`mean_i([0.128×abs(qdot_i)-abs(v_b_x-omega_b_z×y_i)]^2)` |
| `pen_wheel_target_symmetry` | `relu(abs(u_L-u_R)-0.10)^2 × I(abs(c_yaw)<=0.05)` |
| `pen_zero_command_wheel_target` | `[relu(abs(u_L)-0.15)^2+relu(abs(u_R)-0.15)^2] × I(norm(c_xy)<=0.05 and abs(c_yaw)<=0.05)` |
| `pen_terrain_orientation` | `[1-(base_up dot terrain_normal)^2] × plane_valid` |
| `pen_wheel_landing_impact` | `sum_i I(first_contact_i) × [relu(norm(F_i)-100)/100]^2` |
| `pen_wheel_stance_slip` | 估计接触点速度 `v_contact=v_wheel-0.128×cross(omega_wheel,n)`，去掉法向分量后，对切向速度平方按接地置信度加权平均，再乘 `plane_valid` |

Wheel 的 `stand_still`、高度 L2 和高度指数项使用力支撑 `C_force_any`；XY/Yaw 速度跟踪以及两个实际 base 加速度项使用力与几何联合的 `C_any`。单轮有效支撑时这些项即完整生效，双轮腾空时不施加跟踪、高度或加速度平滑信号。XY 项输出有界；Yaw Charbonnier 项不封顶但只近似线性增长，避免旧核在严重抖动时失去梯度。两项权重均为 `-0.02`，不替代 action rate/smoothness。逐轮 `pen_wheel_air_time` 已删除；腾空仍会触发 `pen_wheel_contact`。`pen_zero_command_yaw_rate` 也已删除，其功能由权重提高到 `-7.0` 的 `stand_still` 覆盖。

### 2.5 Foot Expert 和原有 PF 步态项的计算方式

当前 Foot 已与 Wheel 使用相同的局部平面法向高度：

```text
p_bar, n, plane_valid = 对 base height_scanner 的 121 个命中点拟合局部平面
actual_height = abs((p_base-p_bar) dot n)
C_any = max(C_force_L, C_force_R)
pen_base_height = (actual_height-h_cmd)^2 × plane_valid × C_any
rew_base_height_exp = exp(-(actual_height-h_cmd)^2/0.05^2) × plane_valid × C_any
```

Foot 的固定支撑端间距和普通竖直速度惩罚仍从 WF 继承；世界水平姿态项已关闭，并与 Wheel 一样以 `-10.0` 跟踪局部地形法向。Foot 的高度 L2 权重保留 `-30.0`，并新增 `+1.0` 的指数高度正奖励。

| 奖励变量 | 未加权计算方式 |
|---|---|
| PF `pen_feet_regulation` | `sum_i exp(-foot_height_i/0.65) × norm(v_foot_xy_i)^2`，其中 `foot_height=clip(foot_world_z-0.03,0,1)` |
| PF `foot_landing_vel` | 足端未接触、高度低于 `0.08 m`且 Z 速度向下时，求 `sum(v_foot_z^2)` |
| PF/Foot gait term | 先根据 frequency、offset、contact duration 生成平滑期望接触 `d_i`；`force_part=(-2/N)×sum((1-d_i)×(1-exp(-F_i^2/25)))`，`velocity_part=(-2/N)×sum(d_i×(1-exp(-V_i^2/0.25)))`，term value 为两者之和 |
| Foot `pen_feet_regulation` | 每个轮端用自己的局部扫描拟合 `p_ground_i,n_i`；`h_i=clip(abs((p_wheel_i-p_ground_i) dot n_i)-0.128,0,1)`，`v_tan_i=v_i-(v_i dot n_i)n_i`；求 `sum_i exp(-h_i/0.65)×norm(v_tan_i)^2×plane_valid_i` |
| Foot `foot_landing_vel` | 用同一逐轮局部平面；轮端未接触过滤后的地形、`h_i<0.08 m` 且 `v_i dot n_i<0` 时，求 `sum_i(v_i dot n_i)^2×plane_valid_i` |
| Foot `pen_all_wheels_air_time` | `clip(min(t_air_L,t_air_R),0,1)^2`，因此只累计两个支撑端同时腾空的时间 |
| Foot `pen_wheel_stance_slip` | 与 Wheel 公式相同，但配置权重是 `-0.75` |

需要区分两类接触数据：Wheel 的 `stand_still`、两个专家的高度门控、Foot 的接触前落地速度以及 Wheel 有效接地使用每轮独立、只过滤地形 mesh 的 `wheel_L/R_ground_contact`；gait、Foot 双轮同时腾空时间和滑移使用全身 `contact_forces` 中选出的 `wheel_L/R_Link`，后者没有做地形 mesh 过滤。Foot `pen_feet_regulation` 本身不读取接触力，只读取逐轮局部地形扫描和轮心速度。

## 3. 原有奖励项的逐项对比

`--` 表示该项在对应专家中已删除。

| 奖励/惩罚项 | 配置变量名（WF / PF / 当前） | 原有 WF | 原有 PF | Wheel | Foot | 主要变化 |
|---|---|---:|---:|---:|---:|---|
| 存活 | `keep_balance` | `+1.0` | `+1.0` | `+1.0` | `+1.0` | 全部相同 |
| 零命令静止 | `stand_still` | `-5.0` | -- | `-7.0` | -- | PF 和 Foot 无独立静止项；Wheel 增加平均地面支撑置信度门控，并覆盖已删除的独立零命令 yaw 项 |
| XY 速度跟踪 | `rew_lin_vel_xy` | `+3.0` | `+1.0` | `+3.5` | `+3.0` | Wheel 提高跟踪权重并加任意轮有效接地 `C_any` 门控 |
| Yaw 速度跟踪 | `rew_ang_vel_z` | `+1.0` | `+0.5` | `+1.5` | `+1.0` | Wheel 提高权重并将 `std²` 收紧为 `0.12`，加强转向跟踪与超调代价 |
| base 平面加速度 | `pen_base_lin_acc_xy` | -- | -- | `-0.02` | -- | Wheel 使用弱有界惩罚，近目标且任意轮有效支撑时抑制 `vx/vy` 波动 |
| base Yaw 加速度 | `pen_base_yaw_acc` | -- | -- | `-0.02` | -- | Wheel 使用大抖动区仍有梯度的 Charbonnier 核抑制 `wz` 波动 |
| 腿部对称 | `rew_leg_symmetry` | `+0.5` | -- | `+0.5` | `+0.5` | PF 无该项 |
| 左右轮 X 位置一致 | `rew_same_foot_x_position` | `-50.0` | -- | -- | -- | 两个专家均删除 |
| 竖直线速度 | `pen_lin_vel_z` | `-0.3` | `-0.5` | `-0.3` | `-0.3` | Wheel 在高度命令过渡期将实际惩罚缩放到 `0.2～1.0`；Foot/PF 仍为普通 L2 |
| Roll/Pitch 角速度 | `pen_ang_vel_xy` | `-0.3` | `-0.05` | `-0.3` | `-0.3` | PF 权重更低 |
| 关节力矩 | `pen_joint_torque` | `-1.6e-4` | `-8e-5` | `-1.6e-4` | `-8e-5` | Foot 按 PF 权重减弱力矩惩罚 |
| 关节加速度 | `pen_joint_accel` | `-1.5e-7` | `-2.5e-7` | `-1.5e-7` | `-2.5e-7` | Foot 按 PF 权重加强加速度惩罚 |
| 动作变化率 | `pen_action_rate` | `-0.3` | `-0.03` | `-0.15` | `-0.03` | Foot 与 PF 一致；Wheel 居中 |
| 动作二阶平滑 | `pen_action_smoothness` | `-0.03` | `-0.04` | `-0.08` | `-0.04` | Foot 与 PF 一致；Wheel 最强 |
| 非轮/全部关节位置限位 | `pen_non_wheel_pos_limits` / `pen_joint_pos_limits` | `-2.0` | `-2.0` | `-2.0` | `-2.0` | PF 约束其全部关节 |
| 非期望接触 | `undesired_contacts` / `pen_undesired_contacts` / `undesired_contacts` | `-0.25` | `-0.5` | `-1.0` | `-1.0` | 新专家权重最大 |
| 小腿承重力 L2 | `pen_knee_contact_force` | -- | -- | -- | `-2.0` | 超过 `10 N` 的峰值力按 `min((F-10)/100,3)^2` 逐腿累加，区分轻微触碰和持续承重并限制冲击尖峰 |
| 世界系水平姿态 | `pen_flat_orientation_l2` / `pen_flat_orientation` | `-12.0` | `-10.0` | -- | -- | Wheel/Foot 均改用 `pen_terrain_orientation` |
| 左右支撑端间距 | `pen_feet_distance` | `-100` | `-100` | -- | `-100` | PF/Foot 为 `0.115～1.0 m`；原有 WF 为 `0.32～0.35 m`，Wheel 删除该项 |
| base 高度 L2 | `pen_base_height` | `-30.0` | `-20.0` | `-60.0` | `-30.0` | PF 固定 `0.65 m`；新专家改为局部地面相对命令并加接地门控 |
| base 高度指数跟踪 | `rew_base_height_exp` | -- | -- | `+1.0` | `+1.0` | 两个专家都新增 `exp(-(h-h_cmd)^2/0.05^2)` 高斯核/RBF 奖励，保留 L2 负责大误差恢复 |
| 关节功率 L1 | `pen_joint_power_l1` / `pen_joint_powers` | `-2e-5` | `-5e-4` | `-2e-5` | `-5e-4` | Foot 与 PF 一致；是 WF/Wheel 的 25 倍 |
| 轮关节速度 L2 | `pen_joint_vel_wheel_l2` | `-0.005` | 不适用 | -- | `-0.10` | Foot 加强实际轮速惩罚 |
| 非轮/全部关节速度 L2 | `pen_vel_non_wheel_l2` / `pen_joint_vel_l2` | `-0.03` | `-0.001` | `-0.08` | `-0.001` | Foot 的腿关节部分按 PF 权重；轮关节另由 `pen_joint_vel_wheel_l2=-0.10` 约束 |
| 非超时终止 | `pen_base_contact_termination` | -- | -- | `-500.0` | `-500.0` | 只在当前步非 timeout 终止时取 `1`；Foot 的持续小腿接触终止也会触发该项 |

### 3.1 高度奖励的变化

- 原有 WF：惩罚 base 质心高度偏离固定 `0.80 m`。
- 原有 PF：惩罚 base 质心高度偏离固定 `0.65 m`，不读取地形扫描。
- Wheel：从 base 高度扫描拟合局部地形平面，沿平面法向跟踪 `0.65～0.85 m` 的平滑命令，并乘以左右轮接地力置信度的最大值，任意一轮支撑即生效。
- Foot：与 Wheel 一样拟合局部地形平面、沿平面法向跟踪 `0.65～0.85 m` 的平滑命令，并乘左右轮接地力置信度最大值；L2 与指数项均使用该高度和门控。

高度命令每 `5～8 s` 重采样；Wheel 最大变化速率为 `0.08 m/s`，Foot 为 `0.06 m/s`。每次重采样有 `10%` 概率精确取最小高度、`10%` 概率精确取最大高度，余下 `80%` 在全范围均匀采样。

### 3.2 奖励变量名与实现函数

“奖励变量名”是 `RewardsCfg` 中的配置字段，也是训练日志区分各奖励项时使用的 term name；`func` 才是计算该项数值的实现函数。

| 奖励变量名 | `func` | 使用任务 |
|---|---|---|
| `keep_balance` | `mdp.stay_alive` | WF、PF、Wheel、Foot |
| `stand_still` | `mdp.stand_still` | 原有 WF |
| `stand_still` | `mdp.stand_still_grounded` | Wheel；Foot 置为 `None` |
| `rew_lin_vel_xy` | `mdp.track_lin_vel_xy_exp` | WF、PF、Foot |
| `rew_lin_vel_xy` | `mdp.track_lin_vel_xy_exp_any_wheel_ground` | Wheel |
| `rew_ang_vel_z` | `mdp.track_ang_vel_z_exp` | WF、PF、Foot |
| `rew_ang_vel_z` | `mdp.track_ang_vel_z_exp_any_wheel_ground` | Wheel |
| `pen_base_lin_acc_xy` | `mdp.BaseVelocityAccelerationPenalty(component="xy")` | Wheel |
| `pen_base_yaw_acc` | `mdp.BaseVelocityAccelerationPenalty(component="yaw", kernel="charbonnier")` | Wheel |
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
| `pen_flat_orientation_l2` | `mdp.flat_orientation_l2` | 仅原有 WF；Wheel/Foot 置为 `None` |
| `pen_flat_orientation` | `mdp.flat_orientation_l2` | PF |
| `pen_feet_distance` | `mdp.feet_distance` | WF、PF、Foot；Wheel 置为 `None` |
| `pen_base_height` | `mdp.base_com_height` | WF、PF |
| `pen_base_height` | `mdp.body_height_command_plane_grounded_l2` | Wheel、Foot |
| `rew_base_height_exp` | `mdp.body_height_command_plane_grounded_exp` | Wheel、Foot |
| `pen_base_contact_termination` | `mdp.is_terminated` | Wheel、Foot |
| `pen_joint_power_l1` | `mdp.joint_powers_l1` | WF、Wheel、Foot |
| `pen_joint_powers` | `mdp.joint_powers_l1` | PF |
| `pen_joint_vel_wheel_l2` | `mdp.joint_vel_l2` | WF、Foot；Wheel 置为 `None` |
| `pen_vel_non_wheel_l2` | `mdp.joint_vel_l2` | WF、Wheel、Foot |
| `pen_joint_vel_l2` | `mdp.joint_vel_l2` | PF |
| `pen_feet_regulation` | `mdp.feet_regulation` | PF |
| `pen_feet_regulation` | `mdp.wheel_terrain_feet_regulation` | Foot，PF 目的/权重的局部地形版本 |
| `foot_landing_vel` | `mdp.foot_landing_vel` | PF |
| `foot_landing_vel` | `mdp.wheel_terrain_landing_velocity_l2` | Foot，PF 目的/权重的局部地形版本 |
| `test_gait_reward` | `mdp.GaitReward` | PF |
| `gait_contact_schedule` | `mdp.GaitReward` | Foot |
| `pen_wheel_contact` | `mdp.wheel_ground_contact_loss` | Wheel |
| `pen_wheel_horizontal_neutral` | `mdp.wheel_horizontal_neutral_l2` | Wheel |
| `pen_rolling_error` | `mdp.wheel_rolling_velocity_error` | Wheel |
| `pen_wheel_target_symmetry` | `mdp.wheel_target_symmetry_l2` | Wheel |
| `pen_zero_command_wheel_target` | `mdp.zero_command_wheel_target_l2` | Wheel |
| `pen_terrain_orientation` | `mdp.terrain_aligned_orientation_l2` | Wheel、Foot |
| `pen_all_wheels_air_time` | `mdp.all_wheels_air_time_l2` | Foot |
| `pen_wheel_landing_impact` | `mdp.wheel_landing_impact_l2` | Wheel |
| `pen_wheel_stance_slip` | `mdp.wheel_stance_slip_l2` | Wheel、Foot |

## 4. PF 与 Foot Expert 的步态奖励对比

| 项目 | 原有 PF 变量名 | 原有 PF | 当前 Foot 变量名 | 当前 Foot | 差异 |
|---|---|---:|---|---:|---|
| 步态接触/摆动塑形 | `test_gait_reward` | `+1.0` | `gait_contact_schedule` | `+1.0` | 两者都使用 `GaitReward` |
| 足端/轮端轨迹约束 | `pen_feet_regulation` | `-0.1` | `pen_feet_regulation` | `-0.1` | 两者都让近地切向运动更昂贵、不指定目标摆高；Foot 将世界 XY/世界 Z 改成逐轮局部地形切向速度/法向间隙 |
| 落脚约束 | `foot_landing_vel` | `-0.5` | `foot_landing_vel` | `-0.5` | 两者都惩罚接触前向下速度；Foot 使用局部地形法向速度和地形过滤接触，而非固定世界 Z |
| 双支撑端同时腾空 | -- | -- | `pen_all_wheels_air_time` | `-4.0` | Foot 额外要求至少保留一个支撑 |
| 支撑期滑移 | -- | -- | `pen_wheel_stance_slip` | `-0.75` | Foot 针对轮形支撑端增加滑移约束 |
| 实际轮速 | -- | -- | `pen_joint_vel_wheel_l2` | `-0.10` | Foot 需要抑制未受动作驱动但可被接触带动的轮子 |

PF 和 Foot 都使用相位差 `0.5` 的交替步态，但它们不是同一个动作问题：PF 是无轮关节的点足机器人，Foot Expert 仍使用轮足机构，必须同时处理轮子转动、滑移和更复杂的地形。

| Gait command | 原有 PF | 当前 Foot |
|---|---:|---:|
| 重采样间隔 | 固定 `5 s` | `5～8 s` |
| 频率 | `1.5～2.5 Hz` | `1.2～2.2 Hz` |
| 相位差 | `0.5` | `0.5` |
| 接触持续比例 | 固定 `0.5` | 固定 `0.5` |
| 第四维摆动高度输入 | `0.10～0.20 m`，但原 PF 奖励未读取 | 固定 `0.0`，保留四维 schema 但不指定摆动高度 |

## 5. Wheel Expert 新增奖励

| 项目 | 奖励变量名 | 权重 | 含义 |
|---|---|---:|---|
| 双轮有效接地缺失 | `pen_wheel_contact` | `-4.0` | 按左右轮缺失的连续接地置信度求和 |
| 水平中性点 | `pen_wheel_horizontal_neutral` | `-4.0` | 约束 base 坐标系中轮心 X/Y，不约束 Z |
| 滚动速度误差 | `pen_rolling_error` | `-2.0` | 按左右轮分别比较 `0.128×abs(qdot_i)` 与 `abs(v_b_x-omega_b_z×y_i)`，正确处理差速转向 |
| 无 Yaw 命令时左右轮目标对称 | `pen_wheel_target_symmetry` | `-0.5` | `abs(wz_cmd) ≤ 0.05` 时，惩罚左右轮目标差超过 `0.10` |
| 零速度命令时的轮目标 | `pen_zero_command_wheel_target` | `-0.10` | 线速度和 Yaw 命令均接近零时，惩罚轮目标超出 `0.15` |
| 高度指数跟踪 | `rew_base_height_exp` | `+1.0` | 高斯核/RBF 形式，`std=0.05 m`，按局部平面有效性和最大轮地支撑置信度门控 |
| 局部地形姿态 | `pen_terrain_orientation` | `-10.0` | 惩罚 base-up 方向偏离局部地形法向 |
| 落地冲击 | `pen_wheel_landing_impact` | `-0.02` | 新建立接触时，惩罚超过 `100 N` 的力 |
| 支撑期滑移 | `pen_wheel_stance_slip` | `-0.5` | 惩罚轮地接触点的切向滑动 |

当前 Wheel 已删除 `pen_wheel_air_time` 和 `pen_zero_command_yaw_rate`。前者与 `pen_wheel_contact`、速度奖励接地门控重复较多；后者的主要功能已由 `stand_still` 覆盖。滚动误差和支撑滑移同时保留：前者提供逐轮纵向轮速一致性的密集信号，后者负责有效接地时的真实切向滑移。

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

## 6. Foot Expert 相对原有 WF 和 PF 的奖励变化

下表中 `--` 表示该专家没有对应的激活奖励项。PF 和 Foot 的部分奖励目的相近，但计算对象分别是点足和轮端，不能只根据权重大小判断强弱。第 4 节给出了 PF 与 Foot 的具体公式差异，本节补充 WF 列后做三方汇总。

| 项目 | 原有 WF | 原有 PF | 当前 Foot Expert | 对比结论 |
|---|---|---|---|---|
| 步态接触时序 | `--` | `test_gait_reward` `+1.0` | `gait_contact_schedule` `+1.0` | PF 和 Foot 都用 `GaitReward`。PF 塑形足端接触/摆动时序，Foot 改为轮端并使用自己的 gait command 范围 |
| 摆动末端轨迹 | `--` | `pen_feet_regulation` `-0.1` | `pen_feet_regulation` `-0.1` | Foot 对齐 PF 的目的、公式结构和权重，但按逐轮局部地形法向计算间隙和切向速度，使摆高由地形与能耗权衡自行学习 |
| 落地约束 | `--` | `foot_landing_vel` `-0.5` | `foot_landing_vel` `-0.5` | Foot 对齐 PF 的接触前降速目的和权重，但把世界 Z 速度改成局部地形法向速度，适用于坡面和高台 |
| 双侧同时腾空 | `--` | `--` | `pen_all_wheels_air_time` `-4.0` | Foot 独有；只在左右轮同时无接触时按持续时间惩罚，不惩罚正常单轮摆动 |
| 支撑期滑移 | 无激活的轮端滑移项 | `--` | `pen_wheel_stance_slip` `-0.75` | Foot 独有；根据支撑轮接触点的实际切向速度惩罚打滑 |
| 轮关节实际速度 | `pen_joint_vel_wheel_l2` `-0.005` | 不适用 | `pen_joint_vel_wheel_l2` `-0.10` | Foot 保留 WF 的轮速正则项但权重放大为 `20` 倍，与 gait 行为共同限制实际轮速 |
| 机身姿态 | `pen_flat_orientation_l2` `-12.0` | `pen_flat_orientation` `-10.0` | `pen_terrain_orientation` `-10.0` | WF/PF 惩罚偏离世界系水平；Foot 改为跟随局部地形法向，允许机身随坡面倾斜 |

### 6.1 原有 WF/PF 有、当前 Foot Expert 没有原样保留的奖励

| 来源 | 原有奖励变量名 | 原有权重 | 当前 Foot 状态 | 去向或替代方式 |
|---|---|---:|---|---|
| WF | `stand_still` | `-5.0` | 删除 | Foot 依靠速度跟踪、步态和其他稳定性项，不再额外惩罚零命令时的世界系 XY 速度与 Yaw 角速度；与原有 PF 一致没有独立静止项 |
| WF | `rew_same_foot_x_position` | `-50.0` | 删除 | Foot 不再直接惩罚左右轮的 X 位置差；`pen_feet_distance=-100` 已按 PF 改为 `0.115～1.0 m`，正常步态的前后错位不会触发上限惩罚 |
| WF | `pen_flat_orientation_l2` | `-12.0` | 删除原公式 | 用 `pen_terrain_orientation=-10.0` 替代，由“跟踪世界水平”改为“跟踪局部地形法向” |
| PF | `pen_flat_orientation` | `-10.0` | 删除原公式 | 同样由 `pen_terrain_orientation=-10.0` 取代，不再强制机身保持世界系水平 |

下列 PF 基础功能在 Foot 中通过改名、拆分或局部地形适配继续保留，因此不应认为已删除：

| 原有 PF 变量名 | 当前 Foot 变量名 | 关系 |
|---|---|---|
| `pen_joint_pos_limits` | `pen_non_wheel_pos_limits` | 都调用 `joint_pos_limits`；Foot 只选择非轮关节 |
| `pen_undesired_contacts` | `undesired_contacts` | 都调用 `undesired_contacts`，Foot 将权重从 PF 的 `-0.5` 增大到 `-1.0` |
| -- | `pen_knee_contact_force` | Foot 新增小腿承重力连续惩罚；原有接触计数项超过阈值后不再随接触力增长，此项补足该缺口 |
| `pen_joint_vel_l2` | `pen_vel_non_wheel_l2` + `pen_joint_vel_wheel_l2` | PF 用一项约束全部关节；Foot 将腿关节和轮关节拆开计算 |
| `pen_joint_powers` | `pen_joint_power_l1` | 变量名不同，都调用 `joint_powers_l1` |
| `test_gait_reward` | `gait_contact_schedule` | 变量名不同，都调用 `GaitReward`，但支撑端和 gait command 范围不同 |
| `pen_feet_regulation` | `pen_feet_regulation` | 变量名、权重和指数衰减结构相同；Foot 使用逐轮局部地形法向几何 |
| `foot_landing_vel` | `foot_landing_vel` | 变量名和权重相同；Foot 使用局部法向下降速度与地形过滤接触 |

除上表所列的删除、替代和改名项外，其他原有 WF/PF 基础奖励在 Foot 中都有对应功能，但部分权重、选择的关节或高度计算参考已改变，详见第 3 节逐项表。

Foot 的 gait command 每 `5～8 s` 重采样：

```text
frequency   = 1.2～2.2 Hz
phase offset = 0.5
contact duration = 0.5
swing height input = 0.0（保留 schema，不作为摆动高度目标）
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

因输入尺寸已改变，原有 `WF-Blind-Flat` 的 34 维 Actor checkpoint 不能直接当作当前双专家 checkpoint 加载。原有 PF 使用 300 维历史输入和 36 维 Actor 输入，当前 Foot 使用 340 维历史输入和 162 维 Actor 输入；两者动作输出也分别为 6/8 维，因此 PF checkpoint 同样不能直接加载到 Foot Expert。当前文档中提到的 Wheel `model_20000.pt` 是新 Wheel task/schema 在旧版 Wheel reward 下训练的 baseline，不是原仓库 `WF-Blind-Flat` checkpoint。

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
| 小腿持续承重终止 | -- | -- | -- | `F_knee>20 N` 连续 `0.08 s` |

双轮腾空在当前两个专家中都只是惩罚，不会直接结束 episode。

## 9. 设计影响

1. **Wheel 从“跑得快”改为“双轮有效支撑时跑得准”。** 接地门控防止策略通过腾空或非法接触仍获得大量速度正奖励。
2. **Wheel 的构型约束从多个间接项改为可解释的直接条件。** 中性点只约束 X/Y，高度可由腿长命令独立调节。
3. **Foot 从“慢轮式”变为真正的足式控制。** 轮目标速度的硬 mask 和 gait/swing reward 一起强制策略依靠腿部运动。
4. **两个专家不再用同一奖励函数折中。** 代价是需要分别训练两个 checkpoint，并在推理时由 FSM 负责切换。
5. **Foot 保留固定轮距，但不再追求世界水平姿态。** Foot 与 Wheel 都跟踪局部地形法向；固定轮距在大步态场景中仍可能限制落脚范围，分析 Foot 训练结果时应单独观察。
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
