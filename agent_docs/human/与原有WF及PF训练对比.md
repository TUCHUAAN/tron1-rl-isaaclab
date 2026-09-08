# 当前双专家方案与原有 WF 及 PF 训练对比

> 对比时间：2026-09-02。
>
> 原有方案以 `/home/tuchuaan/open_source/tron1-rl-isaaclab-origin/TRAINING_ENV_CHANGES.md` 中的 `Isaac-Limx-WF-Blind-Flat-v0` 和 `Isaac-Limx-PF-Blind-Flat-v0` 为基准。
>
> 当前方案指 `Isaac-Limx-WF-Wheel-Mode-v0` 和 `Isaac-Limx-WF-Foot-AllTerrain-v0`。当前仓库仍保留原有 WF/PF Blind-Flat 任务，新任务是并行增加，不是将原任务原地覆盖。

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
| 站立环境比例 | `2%` | `20%` | `25%` | `15%` |
| 直行专用比例 | -- | -- | `30%` | -- |
| 纯转向专用比例 | -- | -- | `10%` | -- |
| `vx+wz` 混合比例 | -- | -- | `35%` | -- |
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

### 4.3 步态命令

原有 WF 无 gait command，原有 PF 与当前双专家的 gait 对比如下：

| 项目 | 原有 PF | Wheel | Foot |
|---|---:|---:|---:|
| 重采样间隔 | 固定 `5 s` | `5～8 s` | `5～8 s` |
| frequency | `1.5～2.5 Hz` | `1.2～2.2 Hz` | `1.2～2.2 Hz` |
| left/right offset | `0.5` | `0.5` | `0.5` |
| contact duration | 固定 `0.5` | `0.45～0.60` | 固定 `0.5` |
| 第四维 swing-height 输入 | `0.10～0.20 m`，奖励未读取 | `0.08～0.20 m`，奖励未读取 | 固定 `0.0`，只保留输入维数 |

PF 和 Foot 奖励都使用 gait 相位和接触塑形，并固定使用 `0.5` 接触持续比例；这避免 `duration<0.5` 产生计划双摆动区间后与 Foot 的双轮离地惩罚冲突。Foot 不再跟踪指定摆动高度，而是使用 PF 式 `feet_regulation` 与接触前落地速度，并把两者改成逐轮局部地形计算；121 维地形扫描为策略选择摆高提供输入。Wheel 只保留这些输入以统一 schema，不用 gait reward 驱动滚动。

## 5. 观测与网络输入变化

| 项目 | 原有 WF | 原有 PF | Wheel / Foot |
|---|---:|---:|---:|
| Policy 当前观测 | `28` | `34` | `155` |
| 其中高度扫描 | `0` | `0` | `121` |
| 其中 gait phase + command | `0` | `2 + 4` | `2 + 4` |
| History 单帧 | `28` | `34` | `34` |
| History 编码器输入 | `280` | `340` | `340` |
| 命令组 | `3` | `3` | `4` |
| Encoder latent | `3` | `3` | `3` |
| Actor 总输入 | `34` | `40` | `162` |
| Actor 输出 | `8` | `6` | `8` |

当前 policy observation 包含当前高度扫描，history observation 包含 gait 但不包含高度扫描。为此，RSL-RL runner 已改为从实际 `obsHistory` 张量推导编码器输入维度，不再用 `10 × policy_obs_dim` 猜测。

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
| 关节刚度 | `[32, 48]` | 相同 | 相同 |
| 关节阻尼 | `[2.0, 3.0]` | 相同 | 相同 |
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
| Reset 级执行器刚度/阻尼 | log-uniform 比例 `[0.5, 2.0]` | 无 | 同原有 WF | 同原有 WF |

新双专家方案没有增加倾角、高度或轮接触状态的专用 reset 逻辑，并完全沿用原有 WF reset。PF 则更保守：关节初速度为零，且不在每次 reset 重新随机化执行器增益。

## 8. 训练期间动态变化

| 发生时机 | 原有 WF | 原有 PF | Wheel | Foot |
|---|---|---|---|---|
| 速度命令重采样 | `3～15 s` | `0～5 s` | `3～15 s` | `3～15 s` |
| 高度命令重采样 | -- | -- | `5～8 s` | `5～8 s` |
| 步态命令重采样 | -- | 固定 `5 s` | `5～8 s`，仅保持 schema | `5～8 s`，用于 gait reward |
| 观测高斯噪声 | 开启 | 开启 | 开启 | 开启 |
| 随机外力/力矩 | `0.2%` 高频触发 | 相同 | 关闭 | 关闭 |
| 地形等级课程 | 关闭 | 关闭 | 开启 | 开启 |

Wheel 还使用两个实际 base 状态平滑项：`pen_base_lin_acc_xy=-0.02` 与 `pen_base_yaw_acc=-0.02`。二者以 50 Hz 状态差分计算；XY 使用有界核，Yaw 使用大抖动区不完全饱和的 Charbonnier 核，并都乘近目标跟踪门控和联合 `C_any`。它们与直接抑制 action 变化的 `pen_action_rate/pen_action_smoothness` 分工不同。命令分布与奖励公式改变后必须从头训练，不能 resume 旧 Wheel optimizer。

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
python scripts/rsl_rl/train.py \
  --task Isaac-Limx-WF-Foot-AllTerrain-v0 \
  --num_envs 4096 \
  --headless \
  --device cuda:0
```

两个任务分别写入：

```text
logs/rsl_rl/wf_tron_1a_wheel_mode/
logs/rsl_rl/wf_tron_1a_foot_all_terrain/
```

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
- Foot 模式始终将最后两维轮速动作置零。
- Foot→Wheel 要求 base 平面速度 `≤0.25 m/s`、平均轮速 `≤1.0 rad/s`、双轮接地且直立余弦 `≥0.85`。
- `auto` 当前根据高度扫描的最大地形起伏判定，默认阈值 `0.08 m`。

## 11. 建议的对比评估方法

原有 WF/PF 是平地单策略，当前任务是地形课程下的双策略，不建议只比较 `mean_reward`。至少应分别统计：

| 类别 | 建议指标 |
|---|---|
| 生存性 | mean episode length、time-out 比例、base-contact 比例 |
| 跟踪 | XY 速度 RMSE、Yaw 速度 RMSE、高度 RMSE |
| Wheel 合规 | 双轮接地率、单轮腾空时间、滚动误差、滑移速度、中性点偏差 |
| PF 步态 | 足端接触时序、落脚向下速度、足间距和 gait 跟踪 |
| Foot 合规 | gait 接触准确率、摆动期碰撞率、`pen_feet_regulation`、`foot_landing_vel`、双轮同时腾空时间、实际轮速 |
| 地形能力 | 按平地/坡面/粗糙/上阶/下阶分开统计成功率 |
| 切换安全 | 切换成功率、切换时间、切换后摔倒率、切换期间动作跳变 |

为了做公平的原方案对比，应额外在固定平地、相同速度命令交集、关闭观测噪声与域随机化的评估环境中同时测试。PF 与 Foot 的步态比较还应将 gait frequency、contact duration 和 swing height 统一，否则差异同时来自命令分布和机械结构。

### 11.1 PF 与 Foot 的可比边界

| 可直接参考 | 不可直接等同 |
|---|---|
| 交替步态相位、支撑/摆动时序、腿关节平滑性 | PF 无轮关节，Foot 必须处理实际轮速和滑移 |
| 平地上的 gait 接触质量和 base 稳定性 | PF 固定平地，Foot 在楼梯、坡面和粗糙地形上训练 |
| 相同 6 个腿关节的动作幅度和力矩趋势 | PF 是 6 维动作，Foot checkpoint 仍保留 8 维输出和 162 维 Actor 输入 |

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
| 历史观测维度适配 | `rsl_rl/rsl_rl/runner/on_policy_runner.py` |
| 双策略运行脚本 | `scripts/rsl_rl/play_wf_dual.py` |
| 切换 FSM | `exts/bipedal_locomotion/bipedal_locomotion/utils/wf_mode_fsm.py` |
