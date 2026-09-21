# Foot 训练日志各项含义

> 本文数值来自 v5 历史日志，不代表当前 v7。旧轮目标惩罚与旧轮输出裁剪诊断已停用；后续 TensorBoard 短标签及 v7 新指标见 [v7 说明](wf_foot_cycle_mean_v7.md)。


> 整理日期：2026-09-16  
> 本文专门解释所贴出的 Foot 训练控制台日志。该日志对应的配置快照可在：  
> `logs/foot_finetune/2026-09-16_21-27-26_h08_to_exp_clearance_h04_v5/params/`  
> 中查看。**分析历史训练时应以 run 目录中的 `env.yaml`、`agent.yaml` 为准，不要直接拿后来修改过的工作区代码反推旧日志。**

## 1. 先看懂日志分成哪几类

控制台中的项目大致分为五组：

| 前缀/名称 | 表示什么 | 一般怎么看 |
|---|---|---|
| `Computation`、各种 loss、噪声、学习率 | PPO 训练器状态 | 判断训练速度、更新是否异常 |
| `Mean reward`、`Mean episode length` | 最近完成 episode 的总体结果 | 看整体回报和是否容易提前摔倒 |
| `Episode_Reward/...` | 每一个奖励/惩罚项对总 reward 的实际贡献 | 找出谁在主导训练目标 |
| `Metrics/...` | 不参与 reward 的诊断指标 | 直接判断速度、高度、抬脚等真实性能 |
| `Episode_Termination/...` | episode 结束原因占比 | 判断是正常超时，还是摔倒/膝碰地 |
| `Curriculum/...` | 课程学习状态 | 看当前环境难度推进到哪里 |

最重要的原则是：

1. **reward 是训练目标，不完全等于真实性能。**
2. **`Metrics/...` 通常比总 reward 更适合判断机器人到底走得好不好。**
3. 正奖励通常越大越好；负惩罚通常越接近 `0` 越好。
4. 不能只看一帧打印值，应看 TensorBoard 中一段时间的趋势。

---

## 2. 这些数字是怎么统计出来的

### 2.1 当前 run 的时间尺度

该 run 配置为：

```text
仿真 dt          = 0.005 s
控制 decimation  = 4
策略/环境 step_dt = 0.02 s，即 50 Hz
episode 上限      = 20 s = 1000 个策略步
每轮 rollout      = 每个环境 24 步
```

奖励管理器每一步计算：

```text
该项本步贡献 = 原始函数值 × weight × 0.02 s
总本步 reward = 所有项本步贡献之和
```

乘 `dt` 的目的是让 reward 的尺度尽量不随控制频率改变。

### 2.2 `Episode_Reward/...` 不是原始函数值

`Episode_Reward/<name>` 已经包含：

- reward 函数的原始输出；
- 配置的正/负权重；
- 每步 `dt=0.02 s`；
- 从 episode 开始到结束的累加；
- 最后再除以固定的 `max_episode_length_s=20 s`。

近似公式为：

```text
Episode_Reward/name
= 本批结束 episode 的该项累计贡献均值 / 20 s
```

因此它更像“按最大 episode 时长归一化后的贡献速率”。例如：

```text
Episode_Reward/keep_balance = 0.8913
```

`keep_balance` 的理论满值为 `1.0`。`0.8913` 表示这批日志对应的 episode 平均拿到了约 89.13% 的满时长存活奖励；提前结束会把它拉低。

### 2.3 为什么不能把所有 `Episode_Reward` 相加后直接对比 `Mean reward`

二者统计窗口不同：

- `Mean reward`：runner 保存的**最近 100 个完成 episode**的总回报均值；
- `Episode_Reward/...`：本轮 rollout 中每次发生 reset 时产生的日志，再按 reset 批次做平均；
- `Episode_Reward/...` 还额外除以固定的 `20 s`；
- 提前终止 episode 和满 20 秒 episode 的归一化分母相同。

所以：

- 不应要求“所有 `Episode_Reward` 之和 × 20”严格等于 `Mean reward`；
- `Mean episode length` 和 `keep_balance` 也只应大致对应，不必逐位相等；
- 这些值最适合分别看趋势，而不是在单次打印里做严格恒等式检查。

---

## 3. PPO/运行性能项目

### `Computation: 27925 steps/s`

每秒处理的环境步数：

```text
fps = num_envs × num_steps_per_env / (collection_time + learning_time)
```

这里的 step 是所有并行环境的策略步，不是单台机器人的真实时间步。

- 越高表示训练吞吐越高；
- 它主要受环境数、GPU、PhysX、网络规模、日志和显存影响；
- 它不是策略质量指标。

### `collection: 3.318s`

本轮采集 rollout 的时间，包括：

- 策略前向推理；
- PhysX 仿真；
- observation/reward/termination 计算；
- 收集每个环境的 24 个训练步。

### `learning: 0.202s`

PPO 使用刚采集的数据做反向传播和参数更新所花的时间。

若 `collection` 很大，通常瓶颈在仿真或环境计算；若 `learning` 很大，通常瓶颈在网络训练、batch 设置或 GPU。

### `Value function loss: 1.1310`

Critic 预测的 state value 与 PPO return 之间的均方误差；当前代码还使用 clipped value loss。

怎么看：

- 它反映 Critic 对未来累计回报的拟合误差；
- 受 reward 总尺度直接影响，不能脱离本任务的历史曲线判断“大还是小”；
- 不要求单调下降，因为 policy、数据分布和 curriculum 一直在变化；
- 如果突然扩大很多倍并持续发散，才值得重点检查 reward 尺度、学习率、归一化和训练稳定性。

### `Surrogate loss: -0.0038`

PPO 的 clipped policy surrogate loss，即 Actor 更新目标。

怎么看：

- 正负号本身不代表策略好坏；
- PPO 每轮更新后，它在 `0` 附近是常见现象；
- 单独看这个值不能判断机器人是否学会走路；
- 应结合 `mean_kl`、entropy、reward、成功率和实际回放判断。

### `Mean action noise std: 0.6977`

高斯策略分布的平均标准差：

```text
std = mean(exp(logstd))
```

含义是训练时 Actor 采样动作的探索噪声大小。

注意：

- 这是策略动作空间中的标准差，不等于关节角、轮速或力矩的物理误差；
- 还要经过各 action term 的 scale、clip 和执行器处理；
- 太高可能动作很随机、难以收敛；
- 太低可能过早停止探索；
- 该值是从 h08 checkpoint 一起加载后继续训练的，不是这次微调重新初始化的。

### `Learning rate: 0.0001`

PPO optimizer 当前学习率。本次微调使用固定调度，因此保持 `1e-4`；若使用 adaptive schedule，它会根据 KL 自动升降。

### `Mean reward: 32.51`

最近 100 个完成 episode 的总 reward 均值，即每个 episode 内所有步、所有 reward term 的最终加总。

它适合看总体趋势，但不能单独代表策略更好，因为：

- 修改 reward 权重后，不同 run 的绝对值不可直接比较；
- 策略可能通过某个漏洞提高 reward，却没有提高实际运动性能；
- episode 更长通常也会带来更多存活和跟踪 reward。

### `Mean episode length: 930.88`

最近 100 个完成 episode 的平均策略步数。

当前 `step_dt=0.02 s`，所以：

```text
930.88 × 0.02 s = 18.6176 s
```

最大值是 1000 步，即 20 秒。该值已经较接近上限，但仍应结合 termination 比例判断提前结束原因。

---

## 4. `Episode_Reward/...` 各项含义

下表按该 v5 run 的配置快照解释。表中的“日志值”是所贴出的示例值，不是固定参数。

### 4.1 存活、速度跟踪与姿态稳定

| 日志项 | 日志值 | 实际含义 | 怎么看 |
|---|---:|---|---|
| `keep_balance` | `+0.8913` | 每存活一步给常数正奖励，权重 `+1.0` | 越接近 `1` 越好；主要反映 episode 是否活满 20 秒 |
| `rew_lin_vel_xy` | `+2.2736` | 跟踪机体系 `vx/vy` 命令，`4·exp(-||e_xy||²/0.20)` | 越大越好；理论贡献速率上限约 `4` |
| `rew_ang_vel_z` | `+1.1049` | 跟踪机体系 yaw 角速度命令，`2·exp(-e_wz²/0.25)` | 越大越好；理论贡献速率上限约 `2` |
| `rew_leg_symmetry` | `+0.4427` | 奖励左右轮/脚相对机身横向展开较对称，权重 `+0.5` | 已接近上限 `0.5`；只表示横向对称，不表示步态一定正确 |
| `pen_lin_vel_z` | `-0.0090` | 惩罚机身竖直速度平方，减少上下弹跳 | 越接近 `0` 越好 |
| `pen_ang_vel_xy` | `-0.3652` | 惩罚机身 roll/pitch 角速度平方，减少左右/前后摇晃 | 当前是较大的常规负项，需结合姿态视频看是否明显晃动 |
| `pen_terrain_orientation` | `-0.0870` | 惩罚机身 up 方向偏离局部地形法向 | 越接近 `0` 越好；斜坡上要求跟随坡面，不是强行世界水平 |

这里同时存在两类速度目标：

- `rew_lin_vel_xy`、`rew_ang_vel_z`：指数正奖励，误差较小时奖励高；
- `pen_lin_vel_xy_tracking_error`、`pen_yaw_tracking_error`：Huber 大误差惩罚，避免误差很大后指数奖励几乎无梯度。

### 4.2 执行器、动作与能耗

| 日志项 | 日志值 | 实际含义 | 怎么看 |
|---|---:|---|---|
| `pen_joint_torque` | `-0.1737` | 所有关节 applied torque 的平方和，权重 `-8e-5` | 越接近 `0` 越省力；过度追求它可能导致腿发软 |
| `pen_joint_accel` | `-0.4182` | 所有关节加速度平方和，权重 `-2.5e-7` | 抑制关节高频猛动；当前贡献较大 |
| `pen_action_rate` | `-0.2944` | 一阶动作变化 `Σ(a_t-a_{t-1})²`，权重 `-0.03` | 抑制相邻控制步突变 |
| `pen_action_smoothness` | `-0.7565` | 二阶动作差分 `Σ(a_t-2a_{t-1}+a_{t-2})²`，权重 `-0.04` | 抑制动作“拐点”和高频抖动；本日志中是最大的常规负项之一 |
| `pen_joint_power_l1` | `-0.0490` | `Σ|torque × joint_velocity|`，权重 `-5e-4` | 近似机械功率/能耗正则，越接近 `0` 越好 |
| `pen_vel_non_wheel_l2` | `-0.0099` | 六个腿关节速度平方和，权重 `-0.001` | 抑制腿关节过快运动，不统计为轮速锁定目标 |
| `pen_non_wheel_pos_limits` | `-0.0041` | 六个非轮关节超出 soft joint limit 的总量 | 最好长期接近 `0`；持续变大表示姿态在顶软限位 |

`pen_action_rate` 和 `pen_action_smoothness` 不重复：

```text
一阶项：限制“这一步相对上一步变了多少”
二阶项：限制“动作变化速度本身又变了多少”
```

当前样例中 `pen_action_smoothness=-0.7565`、`pen_joint_accel=-0.4182`、`pen_action_rate=-0.2944` 都不小，说明“动作/关节动态比较激烈”是主要扣分来源之一。但是否要加大惩罚不能只看数字，还要确认抬脚、跟踪是否会因此被压死。

### 4.3 接触、腿宽与高度

| 日志项 | 日志值 | 实际含义 | 怎么看 |
|---|---:|---|---|
| `undesired_contacts` | `-0.0004` | abad、hip、knee、base 等不希望接触的 body，接触力超过阈值就按个数惩罚 | 已很小，但它是离散计数型项 |
| `pen_knee_contact_force` | `-0.0000` | 膝部超过 `10 N` 后的超额接触力惩罚 | 接近 `0` 很好；仍要结合膝接触 termination 看 |
| `pen_feet_distance` | `-0.0311` | 机身航向坐标系下，左右轮心有符号横向宽度应在 `0.30～0.38 m` | 越接近 `0` 越好；防止过窄、过宽和交叉腿，不限制前后迈步距离 |
| `pen_base_height` | `-0.0243` | 机身相对局部地形平面的高度平方误差，权重 `-30`，有轮支撑时启用 | 越接近 `0` 越好 |
| `rew_base_height_exp` | `+0.7404` | 同一高度误差的指数正奖励，`std=0.05 m`，有轮支撑时启用 | 越接近理论上限 `1` 越好 |
| `pen_base_contact_termination` | `-0.0726` | 非 timeout 终止的一次性惩罚，配置权重 `-500`，乘 `dt` 后每个失败终止约 `-10` | 越接近 `0` 越好；用于强烈区分摔倒和正常活满时间 |

高度同时使用 L2 负项和 RBF 正项：

- L2 项在偏差变大时持续增加代价；
- RBF 项在高度很准时提供明确正奖励；
- `Metrics/body_height/...` 才是直接以米为单位的实际误差，应优先用它评价高度性能。

### 4.4 Foot 轮锁定与速度大误差

| 日志项 | 日志值 | 实际含义 | 怎么看 |
|---|---:|---|---|
| `pen_wheel_actual_speed` | `-0.7896` | 左右轮**实际关节角速度**的 Huber 惩罚，接地和腾空都生效，权重 `-0.5` | 本日志中很大，说明实际轮速仍是主要矛盾之一 |
| `pen_wheel_target_zero` | `-0.0121` | 策略给出的轮目标速度超过 `±0.1 rad/s` 死区后的 L2 惩罚，权重 `-0.01` | 权重较弱，允许 PI 用小轮目标主动制动；不要和实际轮速混为一谈 |
| `pen_lin_vel_xy_tracking_error` | `-0.1737` | 平面速度误差范数的 Huber cost，尺度 `0.4 m/s`，权重 `-0.5` | 越接近 `0` 越好；对大误差保留线性梯度 |
| `pen_yaw_tracking_error` | `-0.0917` | yaw 角速度绝对误差的 Huber cost，尺度 `0.5 rad/s`，权重 `-0.2` | 越接近 `0` 越好 |
| `pen_base_lin_acc_xy` | `0.0000` | 平面加速度惩罚配置保留，但本 run 权重为 `0.0` | **固定为零不代表平面加速度完美，而是该项被关闭** |
| `pen_base_yaw_acc` | `-0.0195` | yaw 加速度 Charbonnier 惩罚；速度接近命令、且有轮支撑时更强 | 抑制跟踪到目标后的持续摇摆，不应妨碍命令刚变化时必要的加速 |

这里必须区分：

```text
wheel target = policy 希望 PI 控制器达到的轮速
wheel actual = PhysX 中轮关节实际测得的轮速
```

因此：

- `pen_wheel_target_zero` 很小，不代表轮真的没转；
- `pen_wheel_actual_speed` 很大，可能表示制动力不足、腿运动带动轮转、落地冲击或策略在利用轮子；
- 要同时看 `wheel_target_clip_rate`、`wheel_same_limit_rate` 和实际回放。

### 4.5 步态、摆动和落地

| 日志项 | 日志值 | 实际含义 | 怎么看 |
|---|---:|---|---|
| `gait_contact_schedule` | `-0.9534` | 按步态相位约束接触：计划摆动时不应承重，计划支撑时轮/脚不应高速移动 | 名字没有 `pen_`，但函数内部返回负贡献；越接近 `0` 越好 |
| `pen_feet_regulation` | `-0.0190` | 轮/脚越靠近地面，切向运动越受罚；抬高后该限制快速衰减 | 抑制贴地拖脚，但不直接规定目标抬脚高度 |
| `rew_swing_clearance` | `+0.0595` | 摆动期轮缘净空跟踪 `H·sin²(πu)` 目标的指数奖励，峰值 `H` 随地形在 `0.04～0.10 m` 变化 | 越大越好；当前相对理论上限 `0.5` 很低，说明摆动轨迹/离地高度明显不足 |
| `pen_all_wheels_air_time` | `-0.0001` | 两个轮/脚同时离地过久的惩罚 | 已接近 `0`，说明没有明显长时间整体腾空 |
| `foot_landing_vel` | `-0.0222` | 距局部地面小于 `0.08 m`、尚未接触且向下运动时，惩罚法向落地速度平方 | 越接近 `0` 越好；过大表示砸地 |
| `pen_wheel_stance_slip` | `-0.0388` | 有支撑时估计轮-地接触点的切向滑移速度平方 | 越接近 `0` 越好；Foot 模式希望支撑轮不要滚/滑 |

`gait_contact_schedule=-0.9534` 是当前最大负项；同时 `rew_swing_clearance=0.0595` 很低。这两项合起来说明：计划进入摆动相位的脚可能仍在承重，或者离地不充分；支撑期也可能存在较大脚端速度。它们比单纯看总 reward 更直接地暴露“步态有没有真正执行出来”。

---

## 5. Curriculum

### `Curriculum/terrain_levels: 0.0076`

所有环境当前地形难度 level 的均值。

该 run 从 level 0 开始，标准 curriculum 大致根据 episode 内移动距离决定升/降级。因此 `0.0076` 表示绝大多数环境仍在 level 0，只有极少数环境升到了更高一级。

注意：

- 它不是成功率，也不是百分比；
- 值接近 0 表示 curriculum 几乎还没推进；
- 如果它长期不升，应检查机器人移动距离、提前终止率和课程晋级条件；
- 该 v5 run 使用的是 Isaac Lab 标准 `terrain_levels_vel`，不要和工作区后来新增的 Wheel 专用 curriculum 混淆。

---

## 6. 基础速度 Metrics

### 6.1 总体误差

| 日志项 | 值 | 含义 |
|---|---:|---|
| `error_vel_xy` | `0.3650` | Isaac Lab 原生命令指标累计出的平面速度误差，单位近似 `m/s` |
| `error_vel_yaw` | `0.5098` | Isaac Lab 原生命令指标累计出的 yaw 速度误差，单位近似 `rad/s` |

这两个原生指标按“最大命令持续时间”归一化累计，不是下面五类模式严格按样本数计算的 RMSE。做精细比较时，优先看分模式 `xy_rmse`、`wz_rmse`。

### 6.2 五类速度命令是什么

| 模式 | 命令形式 |
|---|---|
| `in_place` | `vx=0, vy=0, wz=0`，原地踏步 |
| `straight` | 只有 `vx`，`vy=0, wz=0` |
| `lateral` | 只有 `vy`，`vx=0, wz=0` |
| `yaw_only` | 只有 `wz`，`vx=0, vy=0` |
| `mixed` | 平移和转向混合 |

配置的采样概率为：

```text
in_place 15%
straight 25%
lateral  10%
yaw_only 20%
mixed    30%
```

实际日志的 `fraction` 不一定精确等于配置概率，因为存在有限样本、不同 episode 长度和提前终止。

### 6.3 每类模式字段

以 `straight` 为例：

| 字段 | 含义 |
|---|---|
| `straight/samples` | 该类模式累计到的有效控制步样本数；它不是 episode 数 |
| `straight/fraction` | 该类样本占所有五类有效控制步样本的比例 |
| `straight/xy_rmse` | 平面速度向量误差 RMSE，单位 `m/s`，越小越好 |
| `straight/wz_rmse` | yaw 角速度误差 RMSE，单位 `rad/s`，越小越好 |
| `straight/wz_mean` | 机器人实际 yaw 角速度均值，单位 `rad/s` |
| `straight/wz_std` | 机器人实际 yaw 角速度总体标准差，单位 `rad/s` |

`wz_mean/std` 统计的是实际 yaw，不是 yaw error；而且样本中命令会变化。因此 `wz_std` 大不一定全是抖动，也可能包含不同正负 yaw 命令。对于理论上 `wz=0` 的 `in_place/straight/lateral`，它才更接近“非期望转动”的指标。

### 6.4 本次五类模式结果怎么读

| 模式 | fraction | XY RMSE | yaw RMSE | 直观结论 |
|---|---:|---:|---:|---|
| `in_place` | `0.0897` | `0.1197 m/s` | `0.2544 rad/s` | 原地仍有明显漂移和自转 |
| `straight` | `0.2839` | `0.3487 m/s` | `0.5189 rad/s` | 直行误差较大，而且不该有的 yaw 很明显 |
| `lateral` | `0.1072` | `0.1786 m/s` | `0.3527 rad/s` | 横移 XY 相对较好，但仍有 yaw 串扰 |
| `yaw_only` | `0.2571` | `0.1869 m/s` | `0.6078 rad/s` | 原地转向时有平移漂移，yaw 跟踪误差大 |
| `mixed` | `0.2621` | `0.3847 m/s` | `0.6507 rad/s` | 最难，XY/yaw 都是当前最差一档 |

额外观察：

- `straight/wz_mean=0.0107` 接近 0，但 `wz_std=0.5128` 很大；均值接近 0 可能只是左右偏转互相抵消，不能据此说直行很稳。
- `yaw_only/wz_mean=0.0402` 也不能直接评价跟踪，因为该模式包含正、负 yaw 命令；应看 `wz_rmse=0.6078`。
- 五类 `fraction` 应大致相加为 1；本样例确实基本如此。

---

## 7. 分高度的摆动指标

高度按当前平滑后的 body-height command，用 `0.70 m`、`0.80 m` 两条边界分成：

- `height_low`：低高度；
- `height_middle`：中高度；
- `height_high`：高高度。

### 各字段含义

| 字段 | 含义 |
|---|---|
| `swing_samples` | 计划处于摆动相位、且局部地形平面有效的脚样本数；按脚计数，一个控制步最多可贡献两个 |
| `step_samples` | 该高度档累计的有效控制步数 |
| `swing_clearance_m` | 计划摆动脚轮缘相对局部地形平面的实际平均净空，单位 `m` |
| `swing_target_m` | 当前相位下 `H·sin²(πu)` 的平均目标净空，不是峰值 |
| `swing_shortfall_m` | `max(target-clearance, 0)` 的均值，只统计抬得不够的部分 |
| `leg_soft_limit_excess_rad` | 六个腿关节超出 soft limit 的总角度，单位 `rad` |
| `peak_target_m` | 地形扫描产生并滤波后的摆动峰值 `H`，范围 `0.04～0.10 m` |

重要区别：

```text
swing_target_m = 当前整个摆动相位上目标曲线的平均值
peak_target_m  = 目标曲线最高点 H
```

所以 `swing_target_m` 本来就会小于 `peak_target_m`。

### 本次结果

| 高度档 | 实际净空 | 相位目标 | 不足量 | 峰值目标 | 结论 |
|---|---:|---:|---:|---:|---|
| low | `0.0047 m` | `0.0240 m` | `0.0210 m` | `0.0482 m` | 平均只抬约 4.7 mm，明显不足 |
| middle | `0.0067 m` | `0.0270 m` | `0.0235 m` | `0.0535 m` | 三档里稍高，但仍远低于目标 |
| high | `0.0054 m` | `0.0251 m` | `0.0219 m` | `0.0502 m` | 同样只有毫米级净空 |

这是本次日志最明确的问题之一：三档 body height 下，实际摆动净空都只有约 `4.7～6.7 mm`，而相位平均目标约 `24～27 mm`，shortfall 约 `21～24 mm`。这和：

- `rew_swing_clearance=0.0595` 很低；
- `gait_contact_schedule=-0.9534` 很大；

是相互一致的，说明策略尚未形成充分、稳定的离地摆动。

`leg_soft_limit_excess_rad`：

- low：`0.0060 rad`，约 `0.34°` 的合计超限；
- middle：`0.0010 rad`；
- high：`0.0002 rad`。

低机身高度更容易逼近腿部软限位，符合几何直觉。

---

## 8. 轮目标饱和 Metrics

### `wheel_target_clip_rate: 0.7659`

左右两个原始轮速动作达到或超过 `±1 rad/s` 裁剪阈值的**单轮比例**。

`0.7659` 表示约 76.59% 的轮动作样本撞到了裁剪边界，数值非常高。

它说明 Actor 经常想输出比允许范围更大的轮速目标。可能原因包括：

- 用饱和轮目标主动制动；
- 在利用轮动作补偿腿部运动；
- 策略动作分布/旧 checkpoint 的轮维输出尚未适应当前软锁轮动力学；
- 轮 PI 控制能力不足，策略只能不断顶限位。

### `wheel_same_limit_rate: 0.5322`

左右轮同时达到**同号** `+1 rad/s` 或同时达到同号 `-1 rad/s` 的控制步比例。

`0.5322` 表示超过一半的控制步两轮一起顶同向边界。这通常不是轻微的左右差动制动，而是强烈的共同模式轮目标，值得重点检查。

这两个指标需要和下面几项一起看：

- `pen_wheel_actual_speed=-0.7896`：实际轮速是否也很大；
- `pen_wheel_target_zero=-0.0121`：目标正则本身较弱；
- MuJoCo/PhysX 回放：轮是在主动刹车、滚动，还是高频反复顶限位；
- action 原始曲线和 PI 输出力矩是否长期饱和。

---

## 9. Body-height Metrics

| 日志项 | 值 | 含义 |
|---|---:|---|
| `height_tracking_mae_m` | `0.0266` | 机身相对局部地形平面的高度平均绝对误差，约 `2.66 cm` |
| `height_tracking_rmse_m` | `0.0460` | 高度误差 RMSE，约 `4.60 cm`；对大误差更敏感 |
| `height_command_slew_mae_m` | `0.0184` | 平滑后当前高度命令与本次随机采样目标之间的平均差，约 `1.84 cm` |

`height_command_slew_mae_m` 不是机器人跟踪误差。它只表示高度命令用了 `0.06 m/s` 的限速，当前 command 还没完全追到随机 target。

`RMSE` 明显大于 `MAE`，说明除了多数普通误差外，还存在少量更大的高度误差/失败状态。

### 分高度 base contact 率

| 日志项 | 值 | 含义 |
|---|---:|---|
| `base_contact_rate_low_height` | `0.1042` | 在低高度档结束的 episode 中，base 接触终止占比 |
| `base_contact_rate_mid_height` | `0.1493` | 中高度档的 base 接触终止占比 |
| `base_contact_rate_high_height` | `0.0208` | 高高度档的 base 接触终止占比 |

这里的结果显示中高度档 base 接触率反而最高，不能简单理解为“越低越容易触地”。还可能受到：

- 各高度档样本数量不同；
- episode 结束时所处命令高度与 episode 全程高度不同；
- 速度模式和地形难度没有按高度完全平衡；
- 统计窗口较小造成波动。

### `reset_fraction_*`

| 日志项 | 值 | 含义 |
|---|---:|---|
| `reset_fraction_low_height` | `0.1897` | 本批 reset 中约 18.97% 结束在低高度档 |
| `reset_fraction_mid_height` | `0.5012` | 约 50.12% 结束在中高度档 |
| `reset_fraction_high_height` | `0.3091` | 约 30.91% 结束在高高度档 |

三者应大致相加为 1。它们是各档的 reset 样本占比，用于判断上面的 contact rate 是否有足够样本支撑；不是低/中/高高度的失败率。

---

## 10. Episode termination

| 日志项 | 值 | 含义 |
|---|---:|---|
| `time_out` | `0.9415` | 94.15% 的 episode 正常活到 20 秒上限 |
| `base_contact` | `0.0529` | 5.29% 因 base 接触地面提前结束 |
| `sustained_knee_contact` | `0.0056` | 0.56% 因膝部接触力超过 `20 N` 并持续 `0.08 s` 而结束 |

三项相加为 1，说明这批 episode 的结束原因统计是闭合的。

直观结论：

- 生存稳定性总体不错，约 94% 能活满；
- 主要失败原因是 base contact；
- 持续膝接触终止较少，但不能仅凭 `0.56%` 判断膝完全没问题，因为短暂膝碰地可能只进入 `undesired_contacts`/`pen_knee_contact_force`，未达到持续终止条件。

---

## 11. 对这份样例日志的快速结论

### 相对好的部分

1. `time_out=94.15%`，绝大多数 episode 能活满 20 秒。
2. `Mean episode length=930.88`，平均约 18.62 秒。
3. 高度 MAE 约 2.66 cm，基础高度跟踪已能工作。
4. 非法接触、持续膝接触、双脚同时长时间腾空都很少。
5. 左右腿横向对称 reward 接近上限。

### 当前最突出的问题

1. **抬脚高度明显不足**：实际净空只有 `4.7～6.7 mm`，相位平均目标约 `24～27 mm`。
2. **步态接触时序惩罚很大**：`gait_contact_schedule=-0.9534`。
3. **轮控制处于严重饱和**：`wheel_target_clip_rate=76.59%`，两轮同号一起顶限位达 `53.22%`。
4. **实际轮速惩罚很大**：`pen_wheel_actual_speed=-0.7896`，软锁轮还没有真正稳定。
5. **动作平滑相关扣分很大**：二阶动作平滑、关节加速度和 action rate 都是主要负项。
6. **速度跟踪仍较差**：尤其 mixed 与 yaw-only，yaw RMSE 分别约 `0.651` 和 `0.608 rad/s`。
7. curriculum 均值仅 `0.0076`，地形难度基本没有推进。

### 建议排查顺序

```text
第一优先：看视频，确认计划摆动脚是否真的离地
第二优先：画左右轮 raw target、clipped target、actual speed、PI torque
第三优先：按五类命令分别回放，重点看 straight 的自转和 yaw_only 的平移漂移
第四优先：检查 action_smoothness 大，是正常迈步需要，还是高频抖动
第五优先：最后再调整 reward 权重，避免只根据单次打印盲调
```

---

## 12. 日常看训练时可用的简化检查表

每隔一段训练，不必逐项全看，可以先看下面这些：

### 是否容易摔

```text
Mean episode length
Episode_Termination/time_out
Episode_Termination/base_contact
Episode_Termination/sustained_knee_contact
```

### 速度是否跟得上

```text
Metrics/base_velocity/{五种模式}/xy_rmse
Metrics/base_velocity/{五种模式}/wz_rmse
```

### 是否真的在迈步

```text
Episode_Reward/gait_contact_schedule
Episode_Reward/rew_swing_clearance
Metrics/base_velocity/height_*/swing_clearance_m
Metrics/base_velocity/height_*/swing_shortfall_m
```

### 轮是否真的锁住

```text
Episode_Reward/pen_wheel_actual_speed
Episode_Reward/pen_wheel_target_zero
Metrics/base_velocity/wheel_target_clip_rate
Metrics/base_velocity/wheel_same_limit_rate
```

### 动作是否过于激烈

```text
Episode_Reward/pen_action_rate
Episode_Reward/pen_action_smoothness
Episode_Reward/pen_joint_accel
Episode_Reward/pen_joint_torque
Episode_Reward/pen_joint_power_l1
```

### 高度是否正常

```text
Metrics/body_height/height_tracking_mae_m
Metrics/body_height/height_tracking_rmse_m
Metrics/body_height/base_contact_rate_*_height
```

---

## 13. 对应代码位置

需要进一步核对公式时，主要看：

```text
# PPO 控制台打印与 Mean reward/length
rsl_rl/rsl_rl/runner/on_policy_runner.py

# PPO value/surrogate loss
rsl_rl/rsl_rl/algorithm/ppo.py

# 该次历史 run 的真实配置快照
logs/foot_finetune/2026-09-16_21-27-26_h08_to_exp_clearance_h04_v5/params/env.yaml
logs/foot_finetune/2026-09-16_21-27-26_h08_to_exp_clearance_h04_v5/params/agent.yaml

# Foot 模式配置
exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/robots/limx_wheelfoot_mode_env_cfg.py

# 自定义 reward 实现
exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/rewards.py
exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/reward_math.py

# 五类速度、分高度、轮目标饱和诊断
exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/commands/foot_velocity_command.py

# 高度 MAE/RMSE 与分高度 reset/contact 统计
exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/commands/body_height_command.py
```
