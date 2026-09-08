# WF 双专家训练与运行

## 1. 环境准备

```bash
cd /home/tuchuaan/tron1-rl-isaaclab
conda activate isaaclab4.5
export PYTHONPATH="$PWD/exts/bipedal_locomotion:$PWD/rsl_rl:$PYTHONPATH"
```

检查 GPU：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
nvidia-smi
```

## 2. 关键配置

| 项目 | 值 |
|---|---|
| Physics dt | `0.005 s` / 200 Hz |
| Decimation | `4` |
| Policy dt | `0.02 s` / 50 Hz |
| Episode | `20 s` / 1000 policy steps |
| PPO rollout | `24 steps/env` |
| PPO epochs / mini-batches | `5 / 4` |
| Learning rate | `1e-3 adaptive` |
| Gamma / Lambda | `0.99 / 0.95` |
| Actor | `162 → 512 → 256 → 128 → 8` |
| History encoder | `340 → 256 → 128 → 3` |

默认训练：

| 专家 | Iterations | 保存间隔 | 日志目录 |
|---|---:|---:|---|
| Wheel 单阶段全地形训练 | `20000` | `500` | `logs/rsl_rl/wf_tron_1a_wheel_mode/` |
| Foot | `15000` | `500` | `logs/rsl_rl/wf_tron_1a_foot_all_terrain/` |

CLI 可用 `--max_iterations` 覆盖默认值。

## 3. 先做冒烟

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Limx-WF-Wheel-Mode-v0 \
  --num_envs 64 \
  --max_iterations 20 \
  --save_interval 10 \
  --run_name wheel_smoke \
  --headless \
  --device cuda:0
```

确认：无 NaN、能保存 checkpoint、episode 不在首步结束。

## 4. 正式训练

### Wheel：单阶段全地形训练

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Limx-WF-Wheel-Mode-v0 \
  --num_envs 4096 \
  --max_iterations 20000 \
  --save_interval 500 \
  --run_name wheel_translation_yaw_robust_v3 \
  --headless \
  --device cuda:1 \
  --kit_args="--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error"
```

该命令不加载旧 checkpoint，从头同时学习 `0.65～0.85 m` 高度控制、零命令稳定和全地形运动。Wheel 速度命令按 `25%` 站立、`30%` 直行、`10%` 纯转向、`35%` 混合四类互斥采样，`vy` 始终为零；平移覆盖为 `65%`，同时保留显式纯 `wz` 场景。高度目标中 `20%` 为端点采样：`10%` 精确取 `0.65 m`，`10%` 精确取 `0.85 m`，其余 `80%` 均匀采样。Wheel 地形从 level 0 开始，共 `12` 级；实际上坡样本占 `25%`，低上台阶占 `10%` 且阶高为 `0.005～0.04 m`。

本版使用 `pen_base_lin_acc_xy=-0.02` 的有界核和 `pen_base_yaw_acc=-0.02` 的 Charbonnier 核。两项都保留跟踪误差门控和联合 `C_any`；Yaw 在大抖动区不再完全饱和。由于命令分布和 Yaw 奖励公式均改变，不要 resume `wheel_modebalanced_accel_v2` 或更早 run 的 optimizer 状态。

Wheel 课程升级必须同时满足：位移超过 `4 m`、episode 中移动命令占比 `>=0.25`、未加门控的 XY 跟踪均值 `>=0.55`、`C_any` 均值 `>=0.75` 且未触发 base contact。机身触地或移动期跟踪均值 `<0.25` 时降一级；其余情况保持等级。这防止策略在坡上仅保持向前移动、却已经严重掉速时仍继续升难度。

`Isaac-Limx-WF-Wheel-Height-Pretrain-v0` 仅保留为可选诊断任务，正式训练不使用它，也不再分阶段。

### Foot

Foot 将小腿持续承重作为失败：`knee_[LR]_Link` 接触力超过 `20 N` 连续 `0.08 s` 时终止，同时使用 `pen_knee_contact_force=-2.0` 对超过 `10 N` 的承重力提供有上限的二次增长训练信号。该奖励和终止条件改变后必须从头训练，不能 resume 旧 Foot checkpoint。

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Limx-WF-Foot-AllTerrain-v0 \
  --num_envs 4096 \
  --max_iterations 15000 \
  --save_interval 500 \
  --headless \
  --device cuda:0
```

显存不足时按以下顺序减小：

```text
4096 → 2048 → 1024 → 512
```

关闭大部分 Isaac warning：

```bash
--kit_args="--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error"
```

## 5. 恢复训练

`--resume` 当前是 bool 参数，必须写值：

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Limx-WF-Wheel-Mode-v0 \
  --resume True \
  --checkpoint_path <model.pt> \
  --num_envs 4096 \
  --max_iterations 5000 \
  --headless \
  --device cuda:0
```

## 6. 查看曲线

```bash
tensorboard \
  --logdir logs/rsl_rl \
  --port 6006
```

重点指标：

```text
Train/mean_reward
Train/mean_episode_length
Episode/Episode_Termination/base_contact
Episode/Episode_Termination/time_out
Episode/Curriculum/terrain_levels
Episode/Metrics/base_velocity/error_vel_xy
Episode/Metrics/base_velocity/error_vel_yaw
Episode/Metrics/base_velocity/support_any
Episode/Metrics/base_velocity/support_all
Episode/Metrics/base_velocity/lin_tracking_ungated
Episode/Metrics/base_velocity/lin_tracking_gated
Episode/Metrics/base_velocity/moving_command_rate
Episode/Metrics/base_velocity/standing_command_rate
Episode/Metrics/base_velocity/straight_command_rate
Episode/Metrics/base_velocity/yaw_only_command_rate
Episode/Metrics/base_velocity/mixed_command_rate
Episode/Metrics/base_velocity/straight_xy_error
Episode/Metrics/base_velocity/yaw_only_wz_error
Episode/Metrics/base_velocity/yaw_only_wz_acc_rms
Episode/Metrics/base_velocity/mixed_xy_error
Episode/Metrics/base_velocity/mixed_wz_error
Episode/Metrics/body_height/height_tracking_mae_m
Episode/Metrics/body_height/height_tracking_rmse_m
Episode/Metrics/body_height/height_command_slew_mae_m
Episode/Metrics/body_height/base_contact_rate_low_height
Episode/Metrics/body_height/base_contact_rate_mid_height
Episode/Metrics/body_height/base_contact_rate_high_height
Episode/Metrics/body_height/reset_fraction_low_height
Episode/Metrics/body_height/reset_fraction_mid_height
Episode/Metrics/body_height/reset_fraction_high_height
Episode/Episode_Reward/pen_wheel_contact
Episode/Episode_Reward/pen_base_height
Episode/Episode_Reward/stand_still
Episode/Episode_Reward/pen_rolling_error
Episode/Episode_Reward/pen_wheel_stance_slip
Episode/Episode_Reward/pen_wheel_target_symmetry
Episode/Episode_Reward/pen_zero_command_wheel_target
Episode/Episode_Reward/pen_base_lin_acc_xy
Episode/Episode_Reward/pen_base_yaw_acc
Episode/Episode_Reward/pen_base_contact_termination
Episode/Episode_Reward/pen_terrain_orientation
```

`height_tracking_mae_m` 和 `height_tracking_rmse_m` 是机器人相对局部地形平面的实际高度跟踪误差；它们保留腾空和复位阶段的有效平面样本，用于暴露失败 episode。高度 reward 本身使用支撑门控。`height_command_slew_mae_m` 只描述限速后的命令与采样目标之间的差，不代表机器人跟踪效果。

高度分段为低 `0.65～0.70 m`、中 `0.70～0.80 m`、高 `0.80～0.85 m`。`base_contact_rate_*` 用于识别哪个高度段更容易倒地；对应的 `reset_fraction_*` 必须同时查看，避免把样本不足误判为低倒地率。

训练结束后不要默认选择最后一个 checkpoint。使用 100 iterations 滚动均值，在满足高度、速度和地形门槛后按最低 base contact 排序：

```bash
conda run -n isaaclab4.5 python scripts/rsl_rl/select_wf_checkpoint.py \
  logs/rsl_rl/wf_tron_1a_wheel_mode/<run_dir> \
  --window 100 --top 10 --write_selection
```

默认门槛为高度 RMSE `≤0.08 m`、XY 速度误差 `≤0.35 m/s`、yaw 误差 `≤0.40 rad/s`、terrain level `≥3.5`。`--write_selection` 会写入 `selected_stability_checkpoint.txt`，MuJoCo 自动部署将优先使用该 checkpoint。

## 7. 单策略运行

```bash
python scripts/rsl_rl/play.py \
  --task Isaac-Limx-WF-Wheel-Mode-Play-v0 \
  --num_envs 1 \
  --checkpoint_path <wheel_model.pt> \
  --device cuda:0
```

录制视频：

```bash
python scripts/rsl_rl/play.py \
  --task Isaac-Limx-WF-Wheel-Mode-Play-v0 \
  --checkpoint_path <wheel_model.pt> \
  --video --video_length 1000 \
  --headless \
  --device cuda:0
```

视频位置：checkpoint 目录下的 `videos/play/`。

## 8. 双策略运行

```bash
python scripts/rsl_rl/play_wf_dual.py \
  --wheel_checkpoint <wheel_model.pt> \
  --foot_checkpoint <foot_model.pt> \
  --mode auto \
  --body_height 0.78 \
  --device cuda:0
```

`--mode`：`wheel`、`foot`、`auto`。

MuJoCo 单专家部署使用 `mujoco/deploy_wheel_policy.py --mode wheel|foot`。Foot 模式会自动搜索 `wf_tron_1a_foot_all_terrain` 下最新 checkpoint，并在执行器层将左右轮目标速度硬锁为零；网络原始 8 维动作仍保留给 `last_action` 观测。

可视化 MuJoCo 测试还会打开独立遥测窗口：对比机身系 `vx/vy/wz` 命令和实际值、局部地形平面相对高度命令和实际值，并对比世界系 roll/pitch 与局部地形法向参考。轻量 Tk Canvas 默认保留最近 `20 s`、以 `20 Hz` 采样和 `5 Hz` 刷新，可用 `--plot-history`、`--plot-sample-hz`、`--plot-hz` 调整或用 `--no-plot` 关闭。

## 9. 2026-08-12 单阶段训练诊断

以下 checkpoint 不再作为下一轮起点：

```text
run: logs/rsl_rl/wf_tron_1a_wheel_mode/2026-08-12_20-23-06_wheel_hnorm_h062_085_stairs004
checkpoint: model_20000.pt
```

| 指标 | 末 100 iterations 均值 |
|---|---:|
| Mean reward | `29.11` |
| Mean episode length | `982.8/1000` |
| 高度惩罚 | `-0.1504`，对应 RMS 误差约 `7.8 cm` |
| Yaw 速度误差 | `0.303 rad/s` |
| 零命令轮目标惩罚 | `-0.3827` |

结论：这次旧配置下的单阶段训练使策略优先保证存活与接触而忽略高度命令，因此不应从该 checkpoint resume。该记录仅作为历史基线。

## 10. 2026-08-14 稳定性诊断与修订

以下加强高度和静止约束后的 checkpoint 同样不作为下一轮起点：

```text
run: logs/rsl_rl/wf_tron_1a_wheel_mode/2026-08-13_18-19-28_wheel_single_stage_h062_085_stairs004
checkpoint: model_20000.pt
```

末100次迭代：高度 `MAE 7.26 cm / RMSE 11.78 cm`，episode length `608.5/1000`，base contact 终止率 `57.5%`。根因是复位高度 `0.966 m` 时，未触地阶段的 `-100` 高度惩罚压过 `+1` 存活奖励；40%端点采样和过紧轮目标约束进一步减少稳定余量。

2026-08-15 修订后仍保持单阶段训练：

- 高度范围改为 `0.65～0.85 m`，端点采样降为 `20%`；
- 高度、`stand_still`、零命令实际 yaw 使用左右轮地面过滤支撑置信度的均值门控；
- 双轮/单轮/腾空对应门控约为 `1.0/0.5/0.0`；
- 当前步发生非 timeout 终止时增加 `-200` 配置权重，即 policy dt 下每次倒地约 `-4`；当前唯一非 timeout 终止项是 `base_contact`，正常 timeout 不惩罚；
- 零命令轮目标回调为 `-0.25`，死区恢复到 `0.15 rad/s`，实际 yaw 惩罚保留 `-2.0`；
- 新配置必须从头训练，不要从上述两个旧 checkpoint resume。

### 10.1 2026-08-15 终止惩罚实现修复

以下运行无效，不得继续 resume，也不得用于部署：

```text
run: logs/rsl_rl/wf_tron_1a_wheel_mode/2026-08-15_00-26-20_wheel_single_stage_h065_085_supportgate
observed iteration: 4020
```

末 100 次迭代的 mean reward 为 `-80.99`，episode length 仅 `15.3/1000`，`base_contact=100%`；`pen_base_contact_termination=-3.05` 与 `keep_balance=0.0153` 的比值约为 `-200`，说明终止惩罚错误地出现在 episode 的每一步。

根因是 `mdp.is_terminated_term` 读取 Isaac Lab 2.2.1 中持久保存的 `_term_dones`，该值不会在单个环境 reset 时清零。现已改为 `mdp.is_terminated`，只读取当前步的非 timeout 终止信号。按当前日志归一化方式，一次倒地的 TensorBoard `pen_base_contact_termination` 应约为 `-0.2`，不应再稳定在 `keep_balance` 的 `-200` 倍。修复后必须使用新的 run name 从头训练。

### 10.2 2026-08-16 修复后训练的稳定性回归

```text
run: logs/rsl_rl/wf_tron_1a_wheel_mode/2026-08-15_03-15-51_wheel_single_stage_h065_085_supportgate_termfix
```

该运行确认终止奖励实现已正确，但最终策略不是最稳定 checkpoint：

| checkpoint | base contact | timeout | episode length | 高度 RMSE |
|---|---:|---:|---:|---:|
| `model_6500.pt` | `7.8%` | `92.2%` | `937.2/1000` | `7.32 cm` |
| `model_20000.pt` | `23.4%` | `76.6%` | `840.8/1000` | `7.39 cm` |

20k 策略改善了速度跟踪，却没有继续改善高度，并牺牲了稳定性。新配置将 Wheel 高度权重从 `-100` 回调到 `-60`，倒地权重从 `-200` 提升到 `-500`，零命令轮目标从 `-0.25` 降到 `-0.10`；竖直速度惩罚在高度命令变化时降至最低 `20%`，目标差进入 `0.01 m` 后恢复到 `100%`。下一轮必须从头训练，并通过 checkpoint 排序工具选择稳定模型。

## 11. 2026-08-28 Wheel 奖励去重与滚动公式修订

下一轮 Wheel 训练使用以下新配置，必须从头训练，不能继续使用旧奖励下的 optimizer 状态：

- 删除 `pen_zero_command_yaw_rate=-2.0`，Wheel 的 `stand_still` 从 `-5.0` 调整为 `-6.0`；这是当时的 Wheel 配置记录，Foot 后续改动见第 13 节；
- 删除逐轮 `pen_wheel_air_time=-8.0`，保留 `pen_wheel_contact=-4.0` 和速度正奖励的双轮有效接地门控；
- 保留 `pen_rolling_error=-2.0`，改为分别比较左右轮的 `0.128×abs(qdot_i)` 与 `abs(v_b_x-omega_b_z×y_i)`；`y_i` 使用轮心在 base 系中的实时横向位置；
- 保留 `pen_wheel_stance_slip=-0.5`，作为有效接地时的真实切向滑移约束。

TensorBoard 的 `Episode/Episode_Reward/<term>` 已包含奖励权重和 `dt`，并用最大 episode 时长归一化。滚动项用于观察密集轮速一致性；滑移项均值较小并不自动表示无效，还应检查它是否只在少量严重滑移时出现尖峰。

## 12. 2026-08-29 Wheel 跟踪与动作平滑再平衡

下一轮 Wheel 训练继续从头训练，不能复用旧奖励下的 optimizer 状态：

- 保持 `pen_base_height=-60.0`，新增 `rew_base_height_exp=+1.0`；其公式为 `exp(-(h-h_cmd)^2/0.05^2)`，使用局部地形平面高度、平面有效性与最大轮地支撑置信度 `max(C_L,C_R)` 门控；双轮/单轮/腾空时高度门控约为 `1/1/0`；
- 该项严谨称为“基于平方误差的指数型跟踪奖励”，也可称“高斯核/RBF 形状奖励”；它未包含概率密度归一化系数，不应称为高斯概率密度；
- `rew_lin_vel_xy: +3.0 -> +3.5`，`rew_ang_vel_z: +1.0 -> +1.25`；Yaw 仅小幅上调，以免放大阶跃命令下的转向超调；
- `pen_terrain_orientation: -8.0 -> -10.0`；这一阶段先只调整 Wheel，Foot 的后续对齐见第 13 节；
- `pen_action_rate: -0.10 -> -0.15`，`pen_action_smoothness: -0.05 -> -0.08`，用于减少动作突变与二阶抖动。

## 13. 2026-08-29 Foot 奖励与 PF 正则对齐

下一轮 Foot 训练使用以下新配置，奖励结构和权重均已变化，必须从头训练：

- 删除 `stand_still=-5.0`；保留 `rew_leg_symmetry=+0.5`；
- `pen_joint_torque: -1.6e-4 -> -8e-5`；
- `pen_joint_accel: -1.5e-7 -> -2.5e-7`；
- `pen_action_rate: -0.3 -> -0.03`；
- `pen_action_smoothness: -0.03 -> -0.04`；
- `pen_joint_power_l1: -2e-5 -> -5e-4`；
- 腿关节 `pen_vel_non_wheel_l2: -0.03 -> -0.001`；轮关节 `pen_joint_vel_wheel_l2=-0.10` 保留；
- `pen_terrain_orientation: -8.0 -> -10.0`，继续使用局部地形法向，不恢复世界系水平姿态项；
- Foot 的 `pen_base_height=-30.0` 改为与 Wheel 相同的 121 点局部平面拟合和法向高度，门控改为 `max(C_L,C_R)`；
- 新增 `rew_base_height_exp=+1.0`，`std=0.05 m`，与高度 L2 共用局部平面有效性和 `max(C_L,C_R)` 门控；
- `gait_contact_schedule=+1.0` 保持 PF 的 `GaitReward` 公式和权重；
- 删除固定高度跟踪 `pen_swing_height=-4.0`，改为 `pen_feet_regulation=-0.1`：沿每个轮端自己的局部地形法向计算间隙，沿局部切平面计算速度，沿用 PF 的指数衰减结构但不指定摆动高度；
- 删除接触后冲击 `pen_wheel_landing_impact=-0.02`，改为 `foot_landing_vel=-0.5`：轮端距离局部地形小于 `0.08 m`、尚未接触且沿局部法向下降时惩罚速度平方；
- gait command 第四维 `swing_height` 固定为 `0.0`，只用于保持四维网络 schema，不再作为奖励目标。策略依靠 121 维地形扫描和碰撞/能耗结果自行学习所需摆高。

当前 Foot 仍保留固定支撑端间距、双轮同时腾空、轮端滑移、实际轮速等轮足专用约束。真实 Isaac Lab CPU 1-env 冒烟已验证环境能够创建、奖励项生效且配置断言通过。

## 14. 2026-08-30 Wheel 速度核宽度与零命令稳定性调整

根据 `model_20000.pt` 的 MuJoCo 曲线，Wheel 综合性能良好，但 yaw 阶跃存在明显超调，最高高度直行时仍有残余 `wz`。下一轮 Wheel 从头训练使用：

- `rew_lin_vel_xy` 保持 `+3.5`，将 `std²: 0.20 -> 0.12`；
- `rew_ang_vel_z` 将权重 `+1.25 -> +1.5`，并将 `std²: 0.25 -> 0.12`；
- `stand_still: -6.0 -> -7.0`；其角速度分支只检查近零 yaw 命令，即使线速度命令非零也会惩罚实际世界系 yaw 角速度；
- Foot 的速度奖励和正则配置保持不变。

## 15. 2026-08-31 Wheel 任意轮支撑门控与诊断

- `rew_lin_vel_xy` 和 `rew_ang_vel_z` 由联合接地置信度 `min(C_L,C_R)` 改为 `max(C_L,C_R)`；任意一轮有效支撑即保留完整跟踪信号，仅双轮都无有效支撑时关闭。
- Wheel `stand_still` 由左右轮地形过滤力置信度平均值改为最大值；高度门控保持最大值不变。
- 奖励权重、跟踪误差核宽和 policy/critic 观测维度均不改变。
- TensorBoard 的 `Metrics/base_velocity/` 新增左右轮力/几何/联合置信度、`C_all/C_any`、双轮/单轮/无支撑比例，以及 XY/Yaw 门控前后跟踪值。这些指标用于判断上坡掉速是否由门控撤掉跟踪信号引起。

## 16. 2026-09-01 Wheel 命令覆盖与实际速度振荡抑制

针对最新 Wheel 模型中 `vx/vy/wz` 抖动，以及 20k 策略纯 `wz` 旋转一段时间后倾倒的问题，本版做两类调整：

- 速度命令改为互斥模式采样：站立 `25%`、直行 `20%`、纯转向 `20%`、混合 `35%`；`vy=0`。旧的独立连续采样几乎不会产生严格的纯转向，本版保证训练中持续出现原地旋转 episode。
- 新增 `pen_base_lin_acc_xy=-0.05` 与 `pen_base_yaw_acc=-0.05`。实际 base 速度按 `0.02 s` 差分；XY/Yaw 分别以 `3.0 m/s²`、`4.0 rad/s²` 为饱和尺度，均乘 `std=0.30` 的跟踪门控与力＋几何联合 `C_any`。reset 后前两步置零。
- TensorBoard 增加四类 `*_command_rate` 和两个 `Episode_Reward/pen_base_*_acc*`，应先确认命令比例接近配置，再比较纯转向存活率、`wz` 波动幅度和上坡速度。
- Foot 奖励、Foot 命令、网络输入维度、既有 Wheel 奖励权重和地形课程阈值不变。新 Wheel 必须从头训练，推荐 run name 为 `wheel_modebalanced_accel_v2`。

## 17. 2026-09-02 Wheel 平移覆盖恢复与 Yaw 鲁棒加速度核

对 `wheel_modebalanced_accel_v2` 的 TensorBoard 分析显示：`pen_base_lin_acc_xy` 在 9.5k～20k 始终约为 `-0.0046～-0.0051`，没有在 14k 后增大；同期 XY 误差恶化而 Yaw 误差改善。上一版只有 `55%` 命令包含平移，相比旧采样的约 `75%` 明显减少。

- 命令比例从站立/直行/纯转向/混合 `25/20/20/35%` 改为 `25/30/10/35%`；平移覆盖提高到 `65%`。
- `pen_base_lin_acc_xy: -0.05 -> -0.02`，继续使用 `a²/(a²+3²)` 有界核。
- `pen_base_yaw_acc: -0.05 -> -0.02`，将 `alpha²/(alpha²+4²)` 改为 `sqrt(1+alpha²/4²)-1` Charbonnier 核；严重抖动时仍保留非零梯度。
- 新增按模式条件归一化的 `straight_xy_error`、`yaw_only_wz_error`、`yaw_only_wz_acc_rms`、`mixed_xy_error` 和 `mixed_wz_error`。
- Foot、网络 schema、其余 Wheel 权重和地形课程不变。新 run 必须从头训练，推荐名称 `wheel_translation_yaw_robust_v3`。
