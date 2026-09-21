# 当前双专家方案与原有 WF 及 PF 训练对比

> 对比时间：2026-09-18。
>
> 原有方案以 `/home/tuchuaan/open_source/tron1-rl-isaaclab-origin/TRAINING_ENV_CHANGES.md` 中的 `Isaac-Limx-WF-Blind-Flat-v0` 和 `Isaac-Limx-PF-Blind-Flat-v0` 为基准。
>
> 当前方案指 `Isaac-Limx-WF-Wheel-Mode-v0` 和 `Isaac-Limx-WF-Foot-AllTerrain-v0`。当前仓库仍保留原有 WF/PF Blind-Flat 任务，新任务是并行增加，不是将原任务原地覆盖。

当前为 **v12**：Foot/Wheel 的 policy 和 critic 均追加六维零命令保持误差/启用标记，与奖励共用同一 tracker；MuJoCo 同步支持。policy 161D、Actor 168D，history/Encoder 仍 340D。Foot 保留 v11 的 2 cm 中段最低净空成本 −1.0、漏迈 −0.2/次及此前跟踪权重。**从头训练，不微调**；详见 [v12 观测与训练说明](wf_hold_observation_v12.md)。

## 1. 总体流程变化

### 原有 WF

```text
固定平地 + 单套奖励
        ↓
训练 1 个 WF Blind-Flat 策略
        ↓
同一策略同时输出腿位置和轮速度
```

### 原有 PF

```text
固定平地 + 周期步态命令
        ↓
训练 1 个 PF Blind-Flat 策略
        ↓
输出 6 个腿关节位置，用点足交替支撑
```

### 当前双专家

```text
Wheel 地形课程 → 独立训练 Wheel checkpoint ─
                                                  ├→ Dual Play + 外部 FSM 切换
Foot 全地形课程  → 独立训练 Foot checkpoint  ─
```

当前不是联合训练或 mixture-of-experts 端到端训练：Wheel 和 Foot 使用两次独立 PPO 训练，只在推理时同时加载两个 checkpoint。PF 不参与这个 FSM，它是对 Foot 足式步态设计的独立参考组。

## 2. 训练基础参数对比

| 项目 | 原有 WF | 原有 PF | Wheel | Foot |
|---|---:|---:|---:|---:|
| 并行环境数 | `4096` | `4096` | `4096` | `4096` |
| Physics dt | `0.005 s` | `0.005 s` | `0.005 s` | `0.005 s` |
| Decimation | `4` | `4` | `4` | `4` |
| Policy dt | `0.02 s / 50 Hz` | 相同 | 相同 | 相同 |
| Episode 时长 | `20 s / 1000 steps` | 相同 | 相同 | 相同 |
| PPO rollout | `24 steps/env` | 相同 | 相同 | 相同 |
| PPO epochs / mini-batches | `5 / 4` | 相同 | 相同 | 相同 |
| Learning rate | `1e-3 adaptive` | 相同 | 相同 | 相同 |
| Gamma / Lambda | `0.99 / 0.95` | 相同 | 相同 | 相同 |
| Actor / Critic 隐藏层 | `[512, 256, 128]` | 相同 | 相同 | 相同 |
| 历史编码器隐藏层 | `[256, 128]` | 相同 | 相同 | 相同 |
| 历史长度 | `10` | `10` | `10` | `10` |
| 保存间隔 | `500 iterations` | `200` | `500` | `500` |
| 当前推荐 iterations | `10000` | `2000` | `20000` | `15000` |

Foot 推荐 `15000 iterations`，比原有 WF 多 `50%`，与其 gait 和更广地形分布相对应。Wheel 当前实验统一使用 `20000 iterations` 观察中后期稳定性回归；原有 PF 默认只有 `2000 iterations`，因此默认训练量不能与 Foot 直接视为等预算对照。CLI 均可用 `--max_iterations` 覆盖。

修改前的 `2026-09-15_00-48-24_foot_terrain_phase_modes_v3/model_20000.pt` 虽然完成了 20000 iterations，但并未学成 Foot 步态：策略长期保持双轮着地，以小轮速滚动和差速转向，地形课程几乎停在 level 0。因而训练轮数达到或超过推荐值不等于成功，必须同时验收分模式速度误差、摆动净空、轮目标裁剪率和视频中的真实接触时序。

2026-09-16 的 v5（宽核、指数净空、4 cm 下限）随后已完成约 600 轮微调对照并再次出现退化；v6 已有后期退化的评估；v7 已有训练 run，尚未验收。v4 也已复测为持续双轮接地；h08 已有平移交替迈腿，可作为微调起点。当前建议先运行短训练并检查保存的中间 checkpoint，再决定是否完成默认长训练；训练长度本身不是验收标准。奖励前后参数见[奖励对比](与原有WF及PF奖励对比.md#41-当前-foot-修订对照)。

## 3. 训练环境的核心变化

| 项目 | 原有 WF Blind-Flat | 原有 PF Blind-Flat | Wheel Expert | Foot Expert |
|---|---|---|---|---|
| 地形类型 | 始终为平面 | 始终为平面 | 地形生成器 | 地形生成器 |
| 地形用途 | 平地轮式运动 | 平地点足步态 | 轮式可通过的连续或低冲击地形 | 需要抬腿的全地形 |
| 高度扫描 | 关闭 | 关闭 | Actor 和 Critic 开启 | Actor 和 Critic 开启 |
| 高度扫描范围 | -- | -- | `1.0 m × 1.0 m`，分辨率 `0.1 m` | 相同 |
| 地形课程 | 关闭 | 关闭 | 开启 | 开启 |
| 初始地形等级 | -- | -- | `0` | `0` |
| 训练地形网格 | -- | -- | `12 rows × 16 cols` | `10 rows × 16 cols` |
| 地形难度范围 | -- | -- | `0.0～1.0` | `0.0～1.0` |
| 随机外力 | 开启 | 开启 | 关闭 | 关闭 |

原有 WF/PF Blind-Flat 都明确把 `terrain_levels = None`。Foot 使用 Isaac Lab 原有 `terrain_levels_vel`。Wheel 改用 `wheel_terrain_levels_vel_tracking`：不再只要走过地形长度的一半就升级，还要求该 episode 有足够的移动命令、速度跟踪指数均值 `>=0.55` 且联合 `C_any` 均值 `>=0.75`。机身触地立即降一级；有移动命令但跟踪均值 `<0.25` 也降级，纯站立 episode 保持当前等级。

### 3.1 Wheel 地形分布

| 地形 | 比例 | 参数 |
|---|---:|---|
| 平地 | `20%` | 平面 |
| 上坡 | `25%` | 坡度 `0.0～0.4` |
| 下坡 | `15%` | 坡度 `0.0～0.4` |
| 波浪 | `10%` | 振幅 `0.01～0.06 m`，10 个波 |
| 随机粗糙 | `10%` | 噪声 `0.01～0.06 m` |
| 上楼梯 | `10%` | 阶高 `0.005～0.04 m`，阶宽 `0.35 m` |
| 下楼梯 | `10%` | 阶高 `0.03～0.12 m`，阶宽 `0.35 m` |

Wheel 的环境原点在金字塔/倒金字塔中心平台，因此本文的上下坡/楼梯按“从中心向外运动”的实际方向命名。旧配置的 Pyramid/InvertedPyramid 类型与这个名称相反，已修正；现在真正的上楼梯限于 `0.5～4 cm`。

### 3.2 Foot 地形分布

| 地形 | 比例 | 参数 |
|---|---:|---|
| 平地 | `10%` | 平面 |
| 波浪 | `10%` | 振幅 `0.01～0.08 m`，10 个波 |
| 随机粗糙 | `15%` | 噪声 `0.01～0.08 m` |
| 上楼梯 | `25%` | 阶高 `0.05～0.20 m`，阶宽 `0.30 m` |
| 下楼梯 | `25%` | 阶高 `0.05～0.20 m`，阶宽 `0.30 m` |
| 上坡 | `7.5%` | 坡度 `0.0～0.4` |
| 下坡 | `7.5%` | 坡度 `0.0～0.4` |

Foot 将 `50%` 的地形分配给上/下楼梯，且粗糙和波浪幅度上限高于 Wheel。

### 3.3 Play 地形

Wheel Play 和 Foot/Dual Play 使用 `4 × 4` 地形网格，关闭课程并固定为 `0.6` 难度，以便在可重复的中等难度地形上评估。Dual Play 使用 Foot 的全地形集合。

## 4. 命令分布对比

### 4.1 速度命令

| 项目 | 原有 WF | 原有 PF | Wheel | Foot |
|---|---:|---:|---:|---:|
| 重采样间隔 | `3～15 s` | `0～5 s` | `3～15 s` | `3～15 s` |
| `vx` | `[-0.7, 0.7] m/s` | `[-1.5, 1.5] m/s` | `[-1.5, 1.5] m/s` | `[-0.8, 0.8] m/s` |
| `vy` | `[-0.5, 0.5] m/s` | `[-1.0, 1.0] m/s` | `[0.0, 0.0] m/s` | `[-0.4, 0.4] m/s` |
| `wz` | `[-π, π] rad/s` | `[-0.5, 0.5] rad/s` | `[-1.0, 1.0] rad/s` | `[-0.8, 0.8] rad/s` |
| 零速度命令比例 | `2%` | `20%` | `25%` | `15%`，仍原地踏步 |
| 纯前后专用比例 | -- | -- | `30%` | `25%` |
| 纯侧移专用比例 | -- | -- | -- | `10%` |
| 纯转向专用比例 | -- | -- | `10%` | `20%` |
| 混合比例 | -- | -- | `35%`，`vx+wz` | `30%`，`vx+vy+wz` |
| heading-control 环境比例 | `100%` | `0%` | `0%` | `0%` |

Wheel 取消了侧向速度命令，其前向范围与 PF 相同；当前 Wheel 的四类模式互斥：站立全部速度为零，直行只采样 `vx`，纯转向只采样 `wz`，混合同时采样 `vx/wz`。当前 `25/30/10/35%` 分配使 `65%` 命令包含平移，同时解决连续独立采样几乎不会出现严格 `vx=0,wz!=0` 的覆盖缺口。Foot 保留侧向移动，但整体速度范围更保守。PF 的速度命令变化最频繁，侧向范围也最大。PF 和当前两个专家都将 `rel_heading_envs` 设为零，直接训练 yaw-rate 跟踪。

### 4.2 新增高度命令

| 项目 | 原有 WF | 原有 PF | Wheel | Foot |
|---|---|---|---|---|
| 高度命令 | 无 | 无 | `0.65～0.85 m` | `0.65～0.85 m` |
| 固定奖励目标 | `0.80 m` | `0.65 m` | -- | -- |
| 重采样间隔 | -- | -- | `5～8 s` | `5～8 s` |
| 最大变化速率 | -- | -- | `0.08 m/s` | `0.06 m/s` |
| 高度基准 | 固定世界高度 | 固定世界高度 | 局部拟合地形平面的法向高度 | 与 Wheel 相同的局部拟合地形平面法向高度 |

高度目标会限速渐变，而不是在重采样时立即跳变。

2026-09-10 修复了高度目标均匀采样的张量写回。旧实现对高级索引产生的副本调用 `uniform_()`，导致计划中的 `80%` 连续均匀分支实际沿用上一次目标（初始为区间中点）；只有两个端点分支能可靠写回。当前实现先采样临时张量再赋回 `_target`，新训练才会真正按 `10%` 最低端点、`10%` 最高端点、`80%` 全区间均匀值覆盖高度命令。该修复不会改变已经训练好的 checkpoint。

### 4.3 步态命令

原有 WF 无 gait command，原有 PF 与当前双专家的 gait 对比如下：

| 项目 | 原有 PF | Wheel | Foot |
|---|---:|---:|---:|
| 重采样间隔 | 固定 `5 s` | `5～8 s` | `5～8 s` |
| frequency | `1.5～2.5 Hz` | `1.2～2.2 Hz` | `1.2～2.2 Hz` |
| left/right offset | `0.5` | `0.5` | `0.5` |
| contact duration | 固定 `0.5` | `0.45～0.60` | 固定 `0.5` |
| 第四维 swing-height 输入 | `0.10～0.20 m`，奖励未读取 | `0.08～0.20 m`，奖励未读取 | 固定 `0.0`，只保留输入维数 |

PF 和 Foot 均使用 gait 相位与 0.5 接触持续比例。Foot 当前在所有命令下启用交替步态，每周期根据去坡面扫描起伏选择 5/10 cm 峰值，实时计算摆线高度，净空权重 +2。新增计划支撑脚接触缺失 −1 替代双脚腾空时间平方；落地仅在净空<2 cm 时处罚超过 0.2 m/s 的下降速度。连续相位和四维 gait 输入不变。Wheel 保持原控制与奖励。

Foot 的接触塑形与 PF 已有明确公式差异：PF 摆动力项使用指数成本、内部系数 `-2`；Foot 改为 `100 N` 尺度 Huber 成本、内部系数 `-6`，让承重逐步减少时就能减罚。支撑速度项均保持原指数核和 `-2` 系数。接触持续比例只描述计划时序，不保证实际始终至少一脚支撑。

Foot 当前 XY/yaw 指数跟踪核恢复为 `std²=0.20/0.25`，权重为 `5.5/3.0`；Wheel 的核仍为 `std²=0.12`。

## 5. 观测与网络输入变化

| 项目 | 原有 WF | 原有 PF | Wheel / Foot |
|---|---:|---:|---:|
| Policy 当前观测 | `28` | `30` | `155` |
| 其中高度扫描 | `0` | `0` | `121` |
| 其中 gait phase + command | `0` | `2 + 4` | `2 + 4` |
| History 单帧 | `28` | `30` | `34` |
| History 编码器输入 | `280` | `300` | `340` |
| 命令组 | `3` | `3` | `4` |
| Encoder latent | `3` | `3` | `3` |
| Actor 总输入 | `34` | `36` | `162` |
| Actor 输出 | `8` | `6` | `8` |

2026-09-16 的奖励修改没有增加观测或改变网络形状，也没有加入速度命令或 yaw 的积分观测。当前 policy observation 包含当前高度扫描，history observation 包含 gait 但不包含高度扫描。为此，RSL-RL runner 已改为从实际 `obsHistory` 张量推导编码器输入维度，不再用 `10 × policy_obs_dim` 猜测。

由于 Actor 从 WF 的 34 维或 PF 的 36 维输入变为 162 维，且 PF 只输出 6 维动作，原有 WF/PF checkpoint 都不能无结构转换地直接加载到 Wheel/Foot 新任务。

## 6. 启动时域随机化对比

新两个专家继承了原有 WF 的训练域随机化。PF 大部分范围相同，但质心偏移范围不同：

| 随机项 | 原有 WF | 原有 PF | Wheel / Foot |
|---|---|---|---|
| base 质量 | 增加 `[-1, 2] kg` | 相同 | 相同 |
| 腿部连杆质量 | 乘 `[0.8, 1.2]` | 相同 | 相同 |
| 全身质量和惯量 | 乘 `[0.8, 1.2]` | 相同 | 相同 |
| 静摩擦系数 | `[0.4, 1.2]` | 相同 | 相同 |
| 动摩擦系数 | `[0.7, 0.9]` | 相同 | 相同 |
| 恢复系数 | `[0.0, 1.0]` | 相同 | 相同 |
| 关节刚度 | `[32, 48]` | 相同 | Wheel/Foot 仅 6 个腿关节沿用该范围；轮关节隐式刚度为 `0` |
| 关节阻尼 | `[2.0, 3.0]` | 相同 | Wheel/Foot 仅 6 个腿关节沿用该范围；轮关节改用显式 PI |
| 质心 X 偏移 | `±0.075 m` | `±0.075 m` | `±0.075 m` |
| 质心 Y 偏移 | `±0.075 m` | `[-0.05, 0.06] m` | `±0.075 m` |
| 质心 Z 偏移 | `±0.075 m` | `±0.05 m` | `±0.075 m` |

Play 配置中会关闭 base 质量增量随机化和 observation corruption；这不代表训练配置也关闭了域随机化。

## 7. 每次 episode reset 对比

| Reset 项 | 原有 WF | 原有 PF | Wheel | Foot |
|---|---|---|---|---|
| base XY 位置 | `[-0.5, 0.5] m` | 相同 | 相同 | 相同 |
| base Yaw | `[-3.14, 3.14] rad` | 相同 | 相同 | 相同 |
| base XYZ 线速度 | 各维 `[-0.5, 0.5] m/s` | 相同 | 相同 | 相同 |
| base R/P/Y 角速度 | 各维 `[-0.5, 0.5] rad/s` | 相同 | 相同 | 相同 |
| 关节位置 | 默认值加 `[-0.2, 0.2] rad` | 默认值按比例 `[-0.5, 0.5]` | 同原有 WF | 同原有 WF |
| 关节速度 | `[-0.5, 0.5] rad/s` | 固定 `0` | 同原有 WF | 同原有 WF |
| Reset 级执行器刚度/阻尼 | log-uniform 比例 `[0.5, 2.0]` | 无 | 腿关节同原有 WF；轮关节 `Kp∈[0.5,4.0]`、`Ki/Kp∈[0,0.25] 1/s` | 与 Wheel 相同的 PI 增益随机化，但轮目标恒为 `0` |

新双专家方案没有增加倾角、高度或轮接触状态的专用机器人姿态 reset，机器人初始状态沿用原有 WF。区别是 Wheel/Foot 的显式轮 PI 积分在 reset 清零，并重新采样同一环境左右轮共用的 `Kp` 与 `Ki/Kp`；通用刚度/阻尼随机化只作用于 6 个腿关节。PF 更保守：关节初速度为零，且不在每次 reset 重新随机化执行器增益。

## 8. 训练期间动态变化

| 发生时机 | 原有 WF | 原有 PF | Wheel | Foot |
|---|---|---|---|---|
| 速度命令重采样 | `3～15 s` | `0～5 s` | `3～15 s` | `3～15 s` |
| 高度命令重采样 | -- | -- | `5～8 s` | `5～8 s` |
| 步态命令重采样 | -- | 固定 `5 s` | `5～8 s`，仅保持 schema | `5～8 s`，用于 gait reward |
| 观测高斯噪声 | 开启 | 开启 | 开启 | 开启 |
| 随机外力/力矩 | `0.2%` 高频触发 | 相同 | 关闭 | 关闭 |
| 地形等级课程 | 关闭 | 关闭 | 开启 | 开启 |

Wheel 当前将 `pen_base_lin_acc_xy` 权重置为 `0.0`，仅保留 `pen_base_yaw_acc=-0.025` 的 Charbonnier 核抑制 `wz` 抖动。Yaw 项以 50 Hz 状态差分计算，并乘近目标跟踪门控和联合 `C_any`。Wheel 轮速执行器以 200 Hz 更新 PI 力矩，每次 reset 随机化 `Kp` 与 `Ki/Kp`，并使用条件积分 anti-windup。执行器动力学与奖励变化后必须从头训练，不能 resume 旧 Wheel optimizer。

Foot 当前 XY/yaw 加速度均关闭（权重 0）；新增独立的零 vy 周期积分与零 yaw 周期净偏航 Huber 成本，各 -0.2、尺度 0.02 m / 0.03 rad。完整周期后才评分，不把两个零命令开关绑在一起。瞬时 XY/yaw Huber 误差、轮 PI、裁剪及其他奖励保持；原执行器 PI 积分与新增奖励历史是两种独立状态。

原有 WF 和 PF 的外力范围都为 XY `±500 N`、Roll/Pitch `±50 N·m`。当前新专家在公共模式配置中直接将 `push_robot = None`，目的是先让新运动形态收敛；这是相对原训练鲁棒性来源的减少，后续若恢复随机推力应作为独立训练阶段记录。

## 9. 终止、超时与正常完成

| 条件 | 原有 WF | 原有 PF | Wheel | Foot |
|---|:---:|:---:|:---:|:---:|
| 20 s 超时 | ✓ | ✓ | ✓ | ✓ |
| `base_Link` 接触力 > `1.0` | ✓ | ✓ | ✓ | ✓ |
| 倾角阈值终止 | -- | -- | -- | -- |
| 高度阈值终止 | -- | -- | -- | -- |
| 支撑端腾空终止 | -- | -- | -- | -- |
| 小腿持续承重终止 | -- | -- | -- | 接触力 `>20 N` 连续 `0.08 s` |

`time_out` 仍是可进行 value bootstrap 的截断；`base_contact` 仍是失败终止。Foot 另外把持续小腿承重设为失败终止：短暂擦碰不会结束 episode，但任一 `knee_[LR]_Link` 的最近力峰值连续 4 个 policy step（`0.08 s`）超过 `20 N` 时终止。Wheel 没有该项。

## 10. 训练和运行入口变化

### 10.1 原有 WF/PF 单策略

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Limx-WF-Blind-Flat-v0 \
  --num_envs 4096 \
  --headless \
  --device cuda:0
```

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Limx-PF-Blind-Flat-v0 \
  --num_envs 4096 \
  --headless \
  --device cuda:0
```

WF 和 PF 彼此也是独立策略；PF checkpoint 不会被 Dual Play 加载。

### 10.2 当前两个专家分别训练

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Limx-WF-Wheel-Mode-v0 \
  --num_envs 4096 \
  --headless \
  --device cuda:0
```

```bash
# 先完成训练说明中的环境准备，在仓库根目录执行。
export PYTHONPATH="$PWD/exts/bipedal_locomotion:$PWD/rsl_rl${PYTHONPATH:+:$PYTHONPATH}"
PYTHONWARNINGS=ignore python scripts/rsl_rl/train.py \
  --task Isaac-Limx-WF-Foot-AllTerrain-v0 \
  --num_envs 4096 \
  --max_iterations 20000 \
  --save_interval 500 \
  --run_name foot_hold_observation_v12 \
  --resume false \
  --headless \
  --device cuda:1 \
  --kit_args="--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error"
```

两个任务分别写入：

```text
logs/rsl_rl/wf_tron_1a_wheel_mode/
logs/rsl_rl/wf_tron_1a_foot_all_terrain/
```

当前 `--resume` 需要显式布尔值，支持 `true/false`、`yes/no`、`1/0`（大小写均可）。从头训练建议省略该参数或写 `--resume false`；恢复训练写 `--resume true --checkpoint_path <model.pt>`。解析器已避免旧 `type=bool` 将字符串 `False` 当作真的问题，因此命令文本现在可以可靠覆盖任务配置中的默认恢复状态。

#### 历史参考：Foot h08 → v10 微调（独立日志目录）

> 当前要求从头训练；不使用下面的历史微调命令。

先完成[环境准备](wf_dual_mode_training_process.md#1-环境准备)。下面从 h08 的 `model_20000.pt` 加载 Actor/Critic/Encoder 及策略噪声，使用当前 v10 环境与奖励；不恢复旧环境配置，PPO 与独立 Encoder 优化器均重新初始化。

```bash
# 在已激活 isaaclab4.5 的仓库根目录执行；可先进入 tmux new -s foot_finetune_v10。
export PYTHONPATH="$PWD/exts/bipedal_locomotion:$PWD/rsl_rl${PYTHONPATH:+:$PYTHONPATH}"
PYTHONWARNINGS=ignore python scripts/rsl_rl/train.py \
  --task Isaac-Limx-WF-Foot-AllTerrain-v0 \
  --num_envs 4096 \
  --resume true \
  --checkpoint_path logs/rsl_rl/wf_tron_1a_foot_all_terrain/2026-09-11_14-32-20_foot_width_clearance_h08_v1/model_20000.pt \
  --log_root logs/foot_finetune_v10 \
  --run_name h08_to_zero_hold_swing_v10 \
  --learning_rate 1e-4 \
  --encoder_learning_rate 1e-4 \
  --learning_rate_schedule fixed \
  --max_iterations 1000 \
  --save_interval 100 \
  --headless \
  --device cuda:1 \
  --kit_args="--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error"
```

| 对比项 | 上面的 Foot 新训练 | h08 → v10 微调 |
|---|---|---|
| 初始化 | 随机初始化，`--resume false` | h08 权重，`--resume true` + 显式 checkpoint 路径 |
| PPO / Encoder 学习率 | `1e-3 adaptive` / `1e-3` | `1e-4 fixed` / `1e-4`，关闭 PPO 线性退火 |
| 训练与保存 | 15000 次，每 500 次保存 | 追加 1000 次，每 100 次保存 |
| 日志父目录 | `logs/rsl_rl/wf_tron_1a_foot_all_terrain/` | `logs/foot_finetune_v10/` |
| 最终 checkpoint | `model_15000.pt` | 内部 iteration 从 20000 起，最终 `model_21000.pt` |

微调输出为 `logs/foot_finetune_v10/<时间戳>_h08_to_zero_hold_swing_v10/`；模型、TensorBoard 与 `params/` 同目录保存，不覆盖原 h08 run。`--log_root` 不再追加 experiment_name；不传它时仍走原默认日志目录。应同时保留 `--checkpoint_path`，否则自动查找会在新的日志根目录中进行，找不到原 h08 模型。

两条 Foot 命令均用 `cuda:1`，建议分别运行；只有一张可见显卡时改成 `cuda:0`，显存不足可降低 `--num_envs`。MuJoCo 默认 Foot 搜索不包含微调目录，部署需显式指定微调 checkpoint 或 `--checkpoint-root logs/foot_finetune_v10`，并保留同一 run 的 `params/env.yaml`。

本次同步命令与目录约定，未启动训练；8 项 CLI 测试通过不代表 v7 策略已验收。完整说明见[训练说明](wf_dual_mode_training_process.md#foot-从-h08-模型初始化)。

### 10.3 双策略推理

```bash
python scripts/rsl_rl/play_wf_dual.py \
  --wheel_checkpoint <wheel_model.pt> \
  --foot_checkpoint <foot_model.pt> \
  --mode auto \
  --device cuda:0
```

Dual Play 的状态机为：

```text
wheel → wheel_to_foot → foot → foot_to_wheel → wheel
```

- 过渡时速度命令置零，并在 50 个 policy steps（约 1 s）内渐变两个专家动作。
- 最小模式驻留时间为 100 steps（约 2 s）。
- Foot 模式最后两维 PI 参考固定为零；模式过渡将 Wheel 轮目标与 Foot 零参考插值，腿动作仍混合。
- Foot→Wheel 要求 base 平面速度 `≤0.25 m/s`、平均轮速 `≤1.0 rad/s`、双轮接地且直立余弦 `≥0.85`。
- `auto` 当前根据高度扫描的最大地形起伏判定，默认阈值 `0.08 m`。

MuJoCo 单专家部署还需要区分“模型结构兼容”和“训练/部署语义兼容”：

| checkpoint | 当前 MuJoCo 处理 | 是否适合当前结论 |
|---|---|---|
| 原有 WF/PF Blind-Flat | Actor 维度不同，不能作为当前 Wheel/Foot 直接加载 | 否 |
| 显式 PI 之前的 Wheel | 可能可以加载，但控制器动力学不同 | 仅历史对照 |
| Foot 轮目标硬锁零时期模型 | 当前会把旧模型未训练的后两维作为有效轮目标 | 否 |
| 新 Foot 且保留 `params/env.yaml` | 可读取 gait 范围与 `continuous_phase`，并使用固定零轮速参考和 PI | 是，但仍需行为验收 |

当前 MuJoCo 还同步了 `base_Link` 速度坐标系、读状态前的运动学缓存刷新和扫描未命中为 `0` 的编码。旧脚本产生的结果若缺少这些修复，应单独标记版本，不能直接并入当前训练对比。

## 11. 建议的对比评估方法

原有 WF/PF 是平地单策略，当前任务是地形课程下的双策略，不建议只比较 `mean_reward`。至少应分别统计：

| 类别 | 建议指标 |
|---|---|
| 生存性 | mean episode length、time-out 比例、base-contact 比例 |
| 跟踪 | XY 速度 RMSE、Yaw 速度 RMSE、高度 RMSE |
| Wheel 合规 | 双轮接地率、单轮腾空时间、滚动误差、滑移速度、中性点偏差 |
| PF 步态 | 足端接触时序、落脚向下速度、足间距和 gait 跟踪 |
| Foot 合规 | 计划摆动脚卸载曲线、实际交替接触时序、净空及不足量、落地速度、双轮同时腾空比例/持续时间、实际轮速、轮目标与原始动作裁剪率 |
| 地形能力 | 按平地/坡面/粗糙/上阶/下阶分开统计成功率 |
| 切换安全 | 切换成功率、切换时间、切换后摔倒率、切换期间动作跳变 |

为了做公平的原方案对比，应额外在固定平地、相同速度命令交集、关闭观测噪声与域随机化的评估环境中同时测试。PF 与 Foot 的步态比较还应将 gait frequency、contact duration 和 swing height 统一，否则差异同时来自命令分布和机械结构。

本次 Foot 短训练应分别测试原地踏步、前进、后退、侧移、纯转向和混合命令，并固定相同的地形、高度、步频、控制器增益与随机种子对比 checkpoint。重点确认摆动脚是否先卸载再获得净空、支撑脚是否接替承重，以及平移改善是否仍依赖轮子滚动。先确认平地步态，再分地形检查；总奖励上升、timeout 比例高或轮目标很小，都不能单独证明成功。

104 项 CPU 单元测试已通过，训练 Python 环境中的 21 项 Foot 几何/奖励测试也通过；这些验证公式和接口，不是新的 PhysX 训练或 MuJoCo 行为验证。新版本没有可报告的通过率、速度误差或抬脚高度实测结果。

### 11.1 PF 与 Foot 的可比边界

| 可直接参考 | 不可直接等同 |
|---|---|
| 交替步态相位、支撑/摆动时序、腿关节平滑性 | PF 无轮关节，Foot 必须处理实际轮速和滑移 |
| 平地上的 gait 接触质量和 base 稳定性 | PF 固定平地，Foot 在楼梯、坡面和粗糙地形上训练 |
| 相同 6 个腿关节的动作幅度和力矩趋势 | PF 是 6 维动作，Foot checkpoint 仍保留 8 维输出和 162 维 Actor 输入；Foot 的通用力矩/加速度/功率项当前还会统计两个轮关节 |

## 12. 主要代码位置

| 内容 | 文件 |
|---|---|
| 原有 WF 环境、reset、随机化 | `exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/cfg/WF/limx_base_env_cfg.py` |
| 原有 WF Blind-Flat 覆盖 | `exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/robots/limx_wheelfoot_env_cfg.py` |
| 原有 PF 环境、步态、reset 和随机化 | `exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/cfg/PF/limx_base_env_cfg.py` |
| 原有 PF Blind-Flat 覆盖 | `exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/robots/limx_pointfoot_env_cfg.py` |
| Wheel / Foot 最终训练配置 | `exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/robots/limx_wheelfoot_mode_env_cfg.py` |
| Wheel / Foot 地形分布 | `exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/cfg/WF/terrains_cfg.py` |
| PPO 配置 | `exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/agents/limx_rsl_rl_ppo_cfg.py` |
| 高度命令 | `exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/commands/body_height_command.py` |
| Foot 五类速度命令与分模式/分高度诊断 | `exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/commands/foot_velocity_command.py` |
| gait 连续相位 | `exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/commands/gait_command.py` |
| 历史观测维度适配 | `rsl_rl/rsl_rl/runner/on_policy_runner.py` |
| 双策略运行脚本 | `scripts/rsl_rl/play_wf_dual.py` |
| 切换 FSM | `exts/bipedal_locomotion/bipedal_locomotion/utils/wf_mode_fsm.py` |
