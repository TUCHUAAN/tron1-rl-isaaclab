# 当前双专家方案与原有 WF 及 PF 网络输入对比

> 对比时间：2026-09-04。
>
> 原有方案指 `Isaac-Limx-WF-Blind-Flat-v0` 和 `Isaac-Limx-PF-Blind-Flat-v0`；当前方案指 `Isaac-Limx-WF-Wheel-Mode-v0` 和 `Isaac-Limx-WF-Foot-AllTerrain-v0`。
>
> 本文以当前工作树中的最终生效配置为准，重点说明 Actor、历史编码器、Commands 和 Critic 的输入维度、数据来源及拼接方式。原有 PF 的维度同时使用已训练的 `model_2000.pt` 权重形状复核。

## 1. 总体输入链路

```text
                                   ├─ policy observation ─────────┐
仿真状态 / 传感器 / 命令 / 上次动作 → ObservationManager ├─ obsHistory → MLP Encoder → 3D latent ├→ Actor → action
                                   ├─ commands ─────────────────┘
                                   └─ critic observation + commands ─────→ Critic → V(s)
```

Actor 的真正输入为：

```text
actor_input = concat(encoder(obsHistory), policy_obs, commands)
```

Critic 的默认输入为：

```text
critic_input = concat(critic_obs, commands)
```

`critic_take_latent=False`，因此默认情况下 Critic 不再拼接 3 维 latent。Critic 只在训练时使用，部署 Actor 时不需要 Critic 的特权信息。

## 2. 网络输入尺寸总表

| 项目 | 原有 WF | 原有 PF | 当前 Wheel | 当前 Foot |
|---|---:|---:|---:|---:|
| 当前 `policy_obs` | `28` | `30` | `155` | `155` |
| `obsHistory` 单帧 | `28` | `30` | `34` | `34` |
| 历史帧数 | `10` | `10` | `10` | `10` |
| Encoder 输入 | `280` | `300` | `340` | `340` |
| Encoder 输出 latent | `3` | `3` | `3` | `3` |
| `commands` | `3` | `3` | `4` | `4` |
| Actor 总输入 | `34` | `36` | `162` | `162` |
| Actor 输出 | `8` | `6` | `8` | `8` |
| Actor / Critic 隐藏层 | `[512, 256, 128]` | 相同 | 相同 | 相同 |
| Encoder 隐藏层 | `[256, 128]` | 相同 | 相同 | 相同 |
| 全局 running normalization | 关闭 | 关闭 | 关闭 | 关闭 |

当前 Wheel 和 Foot 故意使用完全相同的 `155 + 340 + 4` schema，这是 Dual Play 能同时加载并切换两个 checkpoint 的前提。

## 3. Actor 当前帧输入对比

### 3.1 原有 WF：28 维

| `policy_obs` 切片 | 维度 | 内容 |
|---|---:|---|
| `[0:3]` | 3 | base 角速度 |
| `[3:6]` | 3 | base 坐标系投影重力 |
| `[6:12]` | 6 | 排除轮关节后的腿关节相对位置 |
| `[12:20]` | 8 | 6 个腿关节 + 2 个轮关节速度 |
| `[20:28]` | 8 | 上一次原始动作 |

原有 WF Blind-Flat 关闭高度扫描，也没有 gait phase/gait command，因此是纯本体感知输入。

### 3.2 原有 PF：30 维

| `policy_obs` 切片 | 维度 | 内容 |
|---|---:|---|
| `[0:3]` | 3 | base 角速度 |
| `[3:6]` | 3 | base 坐标系投影重力 |
| `[6:12]` | 6 | PF 的 6 个可动腿关节相对位置 |
| `[12:18]` | 6 | PF 的 6 个可动腿关节速度 |
| `[18:24]` | 6 | 上一次腿关节动作 |
| `[24:26]` | 2 | gait phase：`sin(phase), cos(phase)` |
| `[26:30]` | 4 | gait command：频率、相位差、接触持续比例、摆动高度 |

PF URDF 中左右足端关节是 fixed joint，不会出现在 Isaac Lab articulation 的关节状态里。因此 PF 的 `joint_pos`、`joint_vel`、`last_action` 和 Actor 输出均为 6 维。这个结论也与 `model_2000.pt` 中 Encoder `300 → 3`、Actor `36 → 6` 的权重形状一致。

### 3.3 当前 Wheel / Foot：155 维

| `policy_obs` 切片 | 维度 | 内容 |
|---|---:|---|
| `[0:3]` | 3 | base 角速度 |
| `[3:6]` | 3 | base 坐标系投影重力 |
| `[6:12]` | 6 | 排除轮关节后的腿关节相对位置 |
| `[12:20]` | 8 | 6 个腿关节 + 2 个轮关节速度 |
| `[20:28]` | 8 | 上一次原始动作 |
| `[28:149]` | 121 | base 下方局部地形高度扫描 |
| `[149:151]` | 2 | gait phase：`sin(phase), cos(phase)` |
| `[151:155]` | 4 | gait command |

与原有 WF 相比，新增的 `127` 维为：

```text
121 维高度扫描 + 2 维 gait phase + 4 维 gait command
```

Wheel 并不使用 gait reward，但仍保留 gait 输入，目的是与 Foot checkpoint 保持完全相同的网络形状。

## 4. Actor 最终拼接位置

以 Actor 真正收到的向量为准，前 3 维始终是 Encoder latent：

| 任务 | Actor 输入切片 | 内容 |
|---|---|---|
| 原有 WF | `[0:3]` | 3D history latent |
|  | `[3:31]` | 28D `policy_obs` |
|  | `[31:34]` | 3D 速度命令 |
| 原有 PF | `[0:3]` | 3D history latent |
|  | `[3:33]` | 30D `policy_obs` |
|  | `[33:36]` | 3D 速度命令 |
| Wheel / Foot | `[0:3]` | 3D history latent |
|  | `[3:158]` | 155D `policy_obs` |
|  | `[158:162]` | 3D 速度 + 1D 机身高度命令 |

`gait_command` 被放在 `policy_obs` 和 `obsHistory` 中；它不在独立 `commands` 组中。当前 `commands` 组的第 4 维只是机身高度命令。

## 5. 各项输入如何获得

### 5.1 Base 角速度

```text
source = robot.data.root_ang_vel_b
dimension = 3
```

这是已表示在 base 坐标系中的 Roll/Pitch/Yaw 角速度，对应真机上的 IMU 陀螺仪输出。

### 5.2 投影重力

```text
source = robot.data.projected_gravity_b
dimension = 3
```

它是世界重力方向旋转到 base 坐标系后的向量，用于表达机身倾斜，对应真机 IMU 姿态/重力方向估计。

### 5.3 关节位置

```text
relative_position = current_joint_position - default_joint_position
```

| 任务 | 取值范围 |
|---|---|
| 原有 WF / Wheel / Foot | 只取 6 个非轮腿关节，排除 `wheel_L/R_Joint` |
| 原有 PF | 取 PF 模型的 6 个可动腿关节；左右足端 fixed joint 不进入 articulation 状态 |

Wheel/Foot 排除轮关节位置的原因是轮关节可连续旋转，绝对角度不是稳定的姿态特征；轮速度仍保留。

### 5.4 关节速度

| 任务 | 函数 | 数值 |
|---|---|---|
| 原有 PF | `joint_vel` | 当前关节速度 |
| 原有 WF / Wheel / Foot | `joint_vel_rel` | 当前速度减默认速度 |

当前资产的默认关节速度为零，因此两种写法在现配置下数值等价。三类配置都对关节速度乘以 `0.05`。

### 5.5 上一次动作

```text
source = env.action_manager.action
```

这是上一次输入环境的原始 action，不是执行器处理后的关节目标。

对 Foot 尤其需要注意：

- Foot 将轮速动作的 `scale=0`，所以实际轮速目标始终为零；
- `last_action` 返回的仍是 8 维原始 action；
- 因此其最后两维在训练中仍可能非零，即使它们不会改变实际轮速目标。

### 5.6 Gait phase

Gait phase 不是传感器量，而是根据 episode 时间和当前步态频率计算：

```text
phase_01 = remainder(episode_step × policy_dt × gait_frequency, 1.0)
gait_phase = [sin(2π × phase_01), cos(2π × phase_01)]
```

Sin/Cos 表达避免相位从 1 回到 0 时在网络输入上产生大跳变。

当前实现用“当前 frequency × episode 累计时间”重算相位，而不是逐步积分相位。因此 frequency 重采样时，计算出的 gait phase 可能发生瞬时跳变。

### 5.7 Gait command

Gait command 由 `GaitCommand` 在命令管理器中采样：

```text
[frequency, phase_offset, contact_duration, swing_height]
```

| 任务 | 重采样时间 | 频率 | 相位差 | 接触持续比例 | 摆动高度 |
|---|---:|---:|---:|---:|---:|
| 原有 PF | `5 s` | `1.5～2.5 Hz` | `0.5` | `0.5` | `0.10～0.20 m` |
| Wheel | `5～8 s` | `1.2～2.2 Hz` | `0.5` | `0.45～0.60` | `0.08～0.20 m`（奖励未读取） |
| Foot | `5～8 s` | `1.2～2.2 Hz` | `0.5` | 固定 `0.5` | 固定 `0.0`，不作为目标 |

Foot 保留第四维只是为了不改变当前 4 维 gait schema；其摆动高度由 121 维地形扫描、步态接触、近地切向运动、落地速度和能耗约束共同决定。

### 5.8 高度扫描

高度扫描只在当前 Wheel/Foot 中进入 Actor：

```text
scanner attachment = base_Link, yaw-only
grid size          = 1.0 m × 1.0 m
resolution         = 0.1 m
ray count          = 11 × 11 = 121
update period      = 0.02 s
```

每条射线的观测值为：

```text
height_value = scanner_world_z - ray_hit_world_z - 0.5
height_value = clip(height_value, 0.0, 10.0)
```

它不是直接的世界系地面 Z 坐标，而是射线传感器相对击中点的高度差，再减去默认 `0.5 m` offset。

当前的轮下局部射线 `wheel_L/R_ground_scan` 不进入 Actor 或 Critic，它们只用于 Wheel 接地奖励。

### 5.9 速度和高度 Commands

```text
velocity_commands = command_manager.get_command("base_velocity")
                   = [vx, vy, wz]
```

Wheel 的这 3 维输入形状没有变化，但命令生成方式为互斥四类：`25%` 全零站立、`30%` 仅 `vx`、`10%` 仅 `wz`、`35%` 同时采样 `vx/wz`，且 `vy` 恒为零。四类采样标签只用于命令生成和 TensorBoard 诊断，不作为额外网络输入；Actor 只看到最终的 `[vx,0,wz]`。

当前 Wheel/Foot 还增加：

```text
body_height_command = command_manager.get_command("body_height")
```

当前工作树中 Wheel 和 Foot 共用 `0.65～0.85 m` 的高度目标范围，但它始终只占 Actor 输入的 1 维。

## 6. 观测后处理和噪声

ObservationManager 对每一项的处理顺序为：

```text
计算原始值 → modifier → 加噪声 → clip → scale → 拼接/写入历史
```

| 观测 | 原有 WF | 原有 PF | Wheel / Foot | Scale |
|---|---:|---:|---:|---:|
| base 角速度 | Gaussian `std=0.05` | `0.05` | `0.05` | `0.25` |
| 投影重力 | `0.025` | `0.025` | `0.025` | `1.0` |
| 关节位置 | `0.01` | `0.01` | `0.01` | `1.0` |
| 关节速度 | `0.01` | `0.01` | `0.01` | `0.05` |
| 上一次动作 | `0.01` | 无显式噪声 | `0.01` | `1.0` |
| gait phase/command | 不适用 | 无 | 无 | `1.0` |
| 高度扫描 | 不适用 | 不适用 | 无显式高斯噪声 | `1.0` |

表中是加在 scale 之前的原始噪声标准差。例如 base 角速度进入网络后的噪声标准差实际为 `0.05 × 0.25 = 0.0125`；关节速度为 `0.01 × 0.05 = 0.0005`。

训练时 `policy` 和 `obsHistory` 组都开启 corruption，Critic 关闭 corruption。Play 环境会关闭 policy corruption。所有 PPO runner 的 `empirical_normalization=False`，所以没有额外的 running mean/std 全局归一化。

## 7. 10 帧历史和 3 维 Latent 如何获得

### 7.1 历史内容

| 任务 | 历史单帧内容 | 单帧维度 |
|---|---|---:|
| 原有 WF | 角速度、投影重力、6D 腿位置、8D 关节速度、8D last action | `28` |
| 原有 PF | 角速度、投影重力、6D 位置、6D 速度、6D last action、2D phase、4D gait | `30` |
| Wheel / Foot | 原有 WF 的 28D + 2D phase + 4D gait | `34` |

`obsHistory` 保留最近 10 个 policy samples。在 50 Hz 策略频率下，它相当于约 `0.2 s` 的 10 帧窗口。Runner 将 `(num_envs, 10, frame_dim)` 展平：原有 WF 为 280 维、原有 PF 为 300 维，当前 Wheel/Foot 为 340 维。

当前 Wheel/Foot 的 121 维高度扫描不写入历史。否则 10 帧高度扫描单独就需要 1210 维；当前设计只让 Actor 看当前地形，历史编码器专注于机器人动力学状态。

### 7.2 Encoder 结构

```text
原有 WF: 280 → 256 → 128 → 3
原有 PF: 300 → 256 → 128 → 3
Wheel / Foot: 340 → 256 → 128 → 3
activation: ELU
```

Runner 现在从实际 `obsHistory.flatten(start_dim=1).shape[1]` 推导 Encoder 输入尺寸，不再假定：

```text
history_dim = 10 × policy_obs_dim
```

这对当前双专家必不可少，因为当前帧是 155 维，历史单帧却只有 34 维。

### 7.3 Latent 的训练目标

Encoder 配置 `output_detach=True`，Actor 的 PPO loss 不通过 latent 反向传播到 Encoder。Encoder 使用单独的 Adam optimizer 和辅助 loss 训练：

```text
encoder_loss = MSE(encoder(history)[0:3], critic_obs[0:3])
```

所有这些任务的 Critic 前 3 维都是无噪声的 `base_lin_vel`，因此 3 维 latent 的明确目标是：

```text
从最近 10 帧本体观测估计 base 坐标系线速度 [vx, vy, vz]
```

Actor 不直接获得 base 线速度，而是依靠这个 history encoder 的 3 维估计值。

## 8. Critic 输入对比

Critic 使用非对称特权观测。它的总维度由 ObservationManager 在运行时生成，Runner 按下式读取，而不在 PPO 配置中手写：

```text
num_critic_input = observation_groups["critic"].shape[1]
                 + observation_groups["commands"].shape[1]
```

| Critic 输入类别 | 原有 WF | 原有 PF | Wheel / Foot |
|---|:---:|:---:|:---:|
| base 线速度 | ✓ | ✓ | ✓ |
| base 角速度、投影重力 | ✓ | ✓ | ✓ |
| 全部关节位置/速度 | ✓ | ✓ | ✓ |
| last action | 8D | 6D | 8D |
| gait phase + gait command | -- | ✓ | ✓ |
| 121D base 高度扫描 | -- | -- | ✓ |
| 关节力矩、关节加速度 | ✓ | ✓ | ✓ |
| 轮端/足端接触信息 | ✓ | ✓ | ✓ |
| 轮端线速度 | ✓ | -- | ✓ |
| 质量、惯量 | ✓ | ✓ | ✓ |
| 默认关节位置 | ✓ | -- | ✓ |
| 关节刚度、阻尼 | ✓ | ✓ | ✓ |
| root 位置、root 速度 | ✓ | ✓ | ✓ |
| 物理材质属性 | ✓ | ✓ | ✓ |
| 速度 commands | 3D | 3D | 3D |
| 高度 command | -- | -- | 1D |

与原有 WF 相比，Wheel/Foot 的 Critic observation 增加 `121 + 2 + 4 = 127` 维，Critic 网络又多拼接 1 维高度 command，因此在其他 WF 特权项不变时，Critic 总输入比原有 WF 多 128 维。

### 8.1 Critic 特权值的实际来源

| 配置名 | 实际数据来源 |
|---|---|
| `robot_joint_torque` | `asset.data.applied_torque` |
| `robot_joint_acc` | `asset.data.joint_acc` |
| `robot_mass` | `asset.data.default_mass` |
| `robot_inertia` | `asset.data.default_inertia` 展平 |
| `robot_joint_pos` | `asset.data.default_joint_pos`，注意不是当前关节位置 |
| `robot_joint_stiffness/damping` | `asset.data.default_joint_stiffness/default_joint_damping` |
| `robot_pos` | `asset.data.root_pos_w` |
| `robot_vel` | `asset.data.root_vel_w` |
| `robot_material_properties` | PhysX rigid-body material properties 展平 |
| WF `feet_contact_force` | 按轮 body ID 取当前世界系接触力 |
| WF `feet_lin_vel` | 按轮 body ID 取世界系线速度 |

PF 的 `robot_pos` 和 `robot_base_pose` 当前都直接返回 `root_pos_w`，因此 PF Critic 实际上重复包含一次 root world position。

PF 的 `robot_feet_contact_force` 也有一个需注意的实现细节：虽然配置传入了足端 body names，当前 helper 直接展平 `contact_sensor.data.net_forces_w_history`，没有按 `sensor_cfg.body_ids` 切片。因此按现代码理解，PF Critic 获得的是该 contact sensor 的完整历史张量，而不一定只有左右足端接触力。

## 9. 哪些传感器不是网络输入

| 数据 | 用途 | 是否进入 Actor |
|---|---|:---:|
| `wheel_L/R_ground_contact` | Wheel 有效接地奖励 | -- |
| `wheel_L/R_ground_scan` | Wheel 轮心至局部地面几何检查 | -- |
| 全身 `contact_forces` | 奖励、终止、Critic 特权信息 | -- |
| base `height_scanner` | Actor/Critic 地形输入、高度/姿态奖励、FSM auto 判断 | ✓ |
| FSM 读取的 base speed、wheel speed、双轮接地、直立度 | 切换安全判断 | 不作为 Actor 额外输入 |

Wheel 的“双轮有效接地”不是一个直接喂给 Actor 的 bool。Actor 只能通过本体状态、高度扫描和奖励反馈间接学会维持接地。

## 10. 部署时各输入如何对应到真机

| 网络输入 | 真机可能来源 | 部署风险 |
|---|---|---|
| base 角速度 | IMU 陀螺仪 | 偏置、噪声、坐标系对齐 |
| 投影重力 | IMU 姿态或重力方向估计 | 加速运动时的姿态估计误差 |
| 关节位置/速度 | 关节编码器 | 速度差分噪声、延迟 |
| last action | 控制软件中的上次原始命令缓冲 | 必须与训练时的 raw/processed 语义一致 |
| gait phase/command | 控制器内部时钟和命令生成器 | frequency 切换时相位连续性 |
| 速度/高度 command | 上层控制器或遥控器 | 范围、限速和单位必须一致 |
| 121D 高度扫描 | LiDAR/深度相机生成的 base 周围局部高程图 | 当前仿真是理想 ray cast，真机缺失点、遮挡、时延和标定误差未显式加噪 |

原有 WF/PF Blind-Flat Actor 不依赖外感知地形，部署接口更简单。当前 Wheel/Foot 的 Actor 直接依赖 121 维高度扫描，因此高程图生成、坐标系、offset 和更新频率必须与训练配置对齐。

## 11. 关键结论

1. **原有 WF 是最简单的 blind 输入。** 28D 当前本体观测 + 280D 历史 + 3D 速度命令。
2. **PF 在 blind 本体感知上加入 gait。** 它使用 30D 当前帧和 300D 历史；当前 Foot 因保留 8D WF 动作/状态 schema，使用 34D 历史单帧和 340D 历史。
3. **当前双专家的主要增量是 121D 当前地形扫描。** 它不进入 history，所以 Encoder 仍为 340D 输入。
4. **3D latent 有明确物理意义。** 它被单独训练为最近 10 帧观测对 base 线速度的估计。
5. **Wheel 和 Foot 的网络输入完全一致。** 两者的差异主要在地形分布、奖励、命令范围和 Foot 轮动作 mask，不在网络 schema。
6. **高度扫描是当前最大的 Sim-to-Real 输入风险。** 它现在没有显式高斯噪声，却占当前 `policy_obs` 的 121/155。

## 12. 主要代码位置

| 内容 | 文件 |
|---|---|
| 原有 WF 观测组 | `exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/cfg/WF/limx_base_env_cfg.py` |
| 原有 PF 观测组 | `exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/cfg/PF/limx_base_env_cfg.py` |
| Wheel / Foot 共享 schema 和高度扫描 | `exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/robots/limx_wheelfoot_mode_env_cfg.py` |
| gait phase、gait command、非轮关节位置 | `exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/observations.py` |
| Gait command 采样 | `exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/commands/gait_command.py` |
| Body-height command 采样和限速 | `exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/commands/body_height_command.py` |
| 观测分组、尺寸推导和历史展平 | `rsl_rl/rsl_rl/runner/on_policy_runner.py` |
| Actor/Critic 拼接与 Encoder 辅助 loss | `rsl_rl/rsl_rl/algorithm/ppo.py` |
| Encoder MLP | `rsl_rl/rsl_rl/modules/mlp_encoder.py` |
| Actor/Critic MLP | `rsl_rl/rsl_rl/modules/actor_critic.py` |
