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

## 3. 先做单元验证和冒烟

不启动 Isaac Sim 的回归测试覆盖高度采样、Foot 几何奖励、五类命令、连续相位、CLI 布尔参数和 FSM 动作路由。当前 Isaac Lab 环境不含 MuJoCo 窗口依赖，因此训练侧与部署侧分别执行：

```bash
cd tests

conda run -n isaaclab4.5 python -m unittest \
  test_body_height_command test_foot_geometry_rewards \
  test_foot_velocity_command test_gait_phase test_reward_math \
  test_rsl_rl_cli_args test_wf_mode_fsm test_wheel_terrain_curriculum

/home/tuchuaan/miniconda3/envs/UDMMR/bin/python -m unittest \
  test_mujoco_gait_command test_mujoco_height_scan test_mujoco_velocity_frame

cd ..
```

2026-09-16 v6 修改后，`UDMMR` 环境的 `123` 项 CPU 回归通过；训练 Python 环境另通过 `14` 项周期积分专项测试。覆盖独立门控、满周期等待、变步频、reset、符号抵消、航向跨界与当前配置绑定。本次真实 PhysX smoke 在 Isaac Sim 启动前要求交互确认 EULA，因非交互输入终止，未进入环境测试；没有代用户接受许可。单元测试不替代 PhysX 或新策略验收。随后可运行 Wheel 冒烟：

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
  --run_name wheel_pi_drive_yaw_v4 \
  --headless \
  --device cuda:1 \
  --kit_args="--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error"
```

该命令不加载旧 checkpoint，从头同时学习 `0.65～0.85 m` 高度控制、零命令稳定和全地形运动。Wheel 速度命令按 `25%` 站立、`30%` 直行、`10%` 纯转向、`35%` 混合四类互斥采样，`vy` 始终为零；平移覆盖为 `65%`，同时保留显式纯 `wz` 场景。高度目标中 `20%` 为端点采样：`10%` 精确取 `0.65 m`，`10%` 精确取 `0.85 m`，其余 `80%` 均匀采样。2026-09-10 已修复高级索引导致的均匀样本未写回问题；这套比例只对修复后新启动的训练真正成立。Wheel 地形从 level 0 开始，共 `12` 级；实际上坡样本占 `25%`，低上台阶占 `10%` 且阶高为 `0.005～0.04 m`。

本版将 `pen_base_lin_acc_xy` 权重置为 `0.0`，不再压制坡上持续平移驱动；`pen_base_yaw_acc=-0.025` 使用 Charbonnier 核，保留跟踪误差门控和联合 `C_any`。Wheel 轮速改为 `200 Hz` 显式 PI 力矩控制，每次 reset 在同一环境的左右轮上共享随机的 `Kp∈[0.5,4.0]`和 `Ki/Kp∈[0,0.25] 1/s`。因为执行器动力学和奖励都已改变，不要 resume `wheel_translation_yaw_robust_v3` 或更早 run 的 optimizer 状态。

Wheel 课程升级必须同时满足：位移超过 `4 m`、episode 中移动命令占比 `>=0.25`、未加门控的 XY 跟踪均值 `>=0.55`、`C_any` 均值 `>=0.75` 且未触发 base contact。机身触地或移动期跟踪均值 `<0.25` 时降一级；其余情况保持等级。这防止策略在坡上仅保持向前移动、却已经严重掉速时仍继续升难度。

`Isaac-Limx-WF-Wheel-Height-Pretrain-v0` 仅保留为可选诊断任务，正式训练不使用它，也不再分阶段。

### Foot

当前为 **v12**：Foot/Wheel 的 policy 和 critic 均追加六维零命令保持误差/启用标记，与奖励共用同一 tracker；MuJoCo 同步支持。policy 161D、Actor 168D，history/Encoder 仍 340D。Foot 保留 v11 的 2 cm 中段最低净空成本 −1.0、漏迈 −0.2/次及此前跟踪权重。**从头训练，不微调**；详见 [v12 观测与训练说明](wf_hold_observation_v12.md)。


Foot 将小腿持续承重作为失败：力超过 `20 N` 连续 `0.08 s` 时终止，另有 `pen_knee_contact_force=-2.0`。v7 将轮速参考固定为零；h08 可作为网络初始化，但控制语义已改变，需重新训练和验收。

Foot 的力矩、关节加速度、功率、腿速度正则均只统计六腿关节；动作变化/平滑只统计前六维。v8 剥离轮关节后，统计范围也与六关节 PF 对齐；策略输出与历史仍保留 8 维。

历史演变（以下旧净空参数已被上面的 v5 替换）：2026-09-14 起 Foot 在零速度命令下也要求持续原地迈步。速度跟踪权重改为 XY `4.0`、yaw `2.0`，两者 `std=sqrt(0.12)`，补充 `pen_yaw_tracking_error=-0.2`（Huber 尺度 `0.5 rad/s`）。摆动项改为 `pen_swing_clearance=-0.5`：从 121 个有效地形点计算世界 Z 极差并裁剪到 `2～10 cm`，每步更新，上升/下降时间常数 `0.04/0.15 s`；以该峰值生成 sin² 容差带，容差 `2 cm`，短缺/过高的 Huber 比例最初为 `1/0.1`（2026-09-16 改为 `4/0.1`）。不再按机身高度设定固定 `5～8 cm` 峰值，不再由 `moving` 开关关闭零命令抬脚。

2026-09-14 新增 XY/yaw 加速度惩罚 `-0.005/-0.025`（2026-09-16 关闭 XY 项，yaw 保留），均为 Charbonnier 核，尺度 `3 m/s²` 和 `4 rad/s²`；Foot 跟踪门控保留 `0.1` 下限，XY 从世界系速度差分。横向宽度、膝保护、轮 PI、旧 `pen_feet_regulation` 保持；yaw 积分尚未启用。旧 checkpoint 不会因奖励修改自动改变行为；随后训练的 v3 已完成，但结果仍退化，见下文状态说明。

2026-09-15 起 Foot 的速度采样为五种互斥模式：原地踏步 `15%`、纯前后 `25%`、纯侧移 `10%`、纯转向 `20%`、混合 `30%`。纯模式的其他速度分量严格为零；零命令继续迈步。有效分量仍在原有对称范围内均匀采样，每 `3～15 s` 重采样。

Foot 的相位改为累积 `frequency×dt`，改频率时保持连续，episode reset 时从零开始。接触时序、净空奖励和各组观测共用该时钟。训练快照会保存 `commands.gait_command.continuous_phase=true`；MuJoCo 加载时自动读取，部署时保留 checkpoint 旁的 `params/env.yaml`。

```bash
# 先完成第 1 节环境准备，在仓库根目录执行。
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

新训练不加载 checkpoint，输出为 `logs/rsl_rl/wf_tron_1a_foot_all_terrain/<时间戳>_foot_hold_observation_v12/`，不会覆盖已有 run。PPO 保持默认 `1e-3 adaptive`，Encoder 保持 `1e-3`。

显存不足时按以下顺序减小：

```text
4096 → 2048 → 1024 → 512
```

关闭大部分 Isaac warning：

```bash
--kit_args="--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error"
```

## 5. 恢复训练与独立目录微调

`--resume` 必须写值：`true/false`、`yes/no` 或 `1/0`，大小写均可。当前解析器不会再把字符串 `False` 误当成真；从头训练可省略该参数，也可显式使用 `--resume false`。恢复训练时使用 `--resume true`，并通过 `--checkpoint_path` 指定完整 checkpoint 路径：

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Limx-WF-Wheel-Mode-v0 \
  --resume true \
  --checkpoint_path <model.pt> \
  --num_envs 4096 \
  --max_iterations 5000 \
  --headless \
  --device cuda:0
```

### 历史参考：Foot 从 h08 模型初始化

> 当前用户要求从头训练，以下微调命令仅作历史参考，当前请使用 [v12 从头训练命令](wf_hold_observation_v12.md#从头训练-foot)。

已验证 `2026-09-11_14-32-20_foot_width_clearance_h08_v1/model_20000.pt` 与当前网络 state_dict 键和形状一致，Actor/Critic/Encoder 可加载。动作与轮 PI 配置相同。该模型在平地前进已有交替离地，但原地与纯左转仍基本不抬脚。

加载参数为 `--resume true --checkpoint_path <该 model_20000.pt 路径>`。当前 runner 默认 `load_optimizer=False`，加载网络（含策略噪声）与 encoder，保留内部 iteration，不恢复 PPO optimizer，独立 Encoder optimizer 也重新初始化；`--max_iterations 1000` 表示追加 1000 次。奖励和命令使用当前 v10 task，**不会自动恢复旧 env.yaml**。新 run 会保存自己的配置快照；部署时保留同目录的 `params/env.yaml`。

微调命令现已支持 `--learning_rate`、`--encoder_learning_rate`、`--learning_rate_schedule` 和 `--log_root`。以下命令将 PPO 与独立 Encoder 优化器都设为 `1e-4`，PPO 使用 fixed 调度并关闭线性退火。旧 checkpoint 的 logstd 原样加载，不重置策略噪声。先追加训练 1000 次，每 100 次保存；加载的最终 h08 checkpoint 内部 iteration 为 20000，因此最终 checkpoint 为 `model_21000.pt`。

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

微调输出独立存于 `logs/foot_finetune_v10/<时间戳>_h08_to_zero_hold_swing_v10/`，包括模型、TensorBoard 和 `params/`。`--log_root` 是时间戳目录的直接父目录，不会再追加 experiment_name；未指定时仍使用 `logs/rsl_rl/<experiment_name>/`。`--experiment_name` 现已正确覆盖任务默认值。从头训练使用第 4 节的 `--resume false` 命令，保留默认 PPO `1e-3 adaptive` 与 Encoder `1e-3`。

MuJoCo 默认 Foot 自动搜索不包含新的微调目录。测试微调模型时应显式传 `--checkpoint logs/foot_finetune_v10/<run>/model_21000.pt`，或将 `--checkpoint-root` 指向 `logs/foot_finetune_v10`。两种训练命令均使用 `cuda:1`，建议分别运行；只有一张可见显卡时改为 `cuda:0`。

### Foot 新训练与微调速查（2026-09-19）

| 项目 | 从头新训练 | h08 → v10 微调 |
|---|---|---|
| 任务/环境/奖励 | 当前 `Isaac-Limx-WF-Foot-AllTerrain-v0` v10 | 同左，不读取旧 `env.yaml` 恢复训练配置 |
| 初始化 | `--resume false`，随机初始化网络 | `--resume true`，加载 h08 `model_20000.pt` 的 Actor/Critic/Encoder 和策略噪声 |
| 优化器状态 | 全新 | 全新，不恢复旧优化器状态 |
| PPO 学习率 | 默认 `1e-3`，adaptive | `1e-4`，fixed，并关闭线性退火 |
| Encoder 学习率 | 默认 `1e-3` | 独立设置为 `1e-4` |
| 本次训练次数 | `15000` | 追加 `1000`，保留加载的内部 iteration |
| 保存间隔 | `500` | `100` |
| 最终模型 | `model_15000.pt` | h08 内部 iteration 为 20000 时得到 `model_21000.pt` |
| 日志父目录 | `logs/rsl_rl/wf_tron_1a_foot_all_terrain/` | `logs/foot_finetune_v10/` |
| run 后缀 | `foot_zero_hold_swing_v10` | `h08_to_zero_hold_swing_v10` |

每次启动都在父目录下新建带时间戳的 run，模型、TensorBoard 和 `params/` 一起保存。微调必须保留显式 `--checkpoint_path`：`--log_root` 同时决定未指定 checkpoint 路径时的自动查找根目录，新微调目录下不会自动找到旧 h08 模型。

v7 已有训练目录 `logs/rsl_rl/wf_tron_1a_foot_all_terrain/2026-09-17_21-39-55_foot_cycle_mean_zero_wheel_v7/`。127 项 CPU 回归通过；用户反馈步态相位和落脚仍不理想，不能认定已完成策略验收。

TensorBoard 短标题仅在重新启动/恢复训练后写入；历史事件与已有进程仍使用长标题。完整映射见 [v7 说明](wf_foot_cycle_mean_v7.md#tensorboard-短标题2026-09-18)。

## 6. 查看曲线

Foot 新增以下日志。训练器通常显示为 `Episode/Metrics/base_velocity/<下列后缀>`：

| 后缀 | 含义 |
|---|---|
| `{in_place,straight,lateral,yaw_only,mixed}/fraction`、`samples` | 各模式的实际执行样本占比和样本数；与重采样概率存在有限样本和提前终止造成的差别 |
| 各模式下的 `xy_rmse`、`wz_rmse` | 只按该模式样本归一化的跟踪误差，单位 `m/s`、`rad/s` |
| 各模式下的 `wz_mean`、`wz_std` | 实际 yaw 均值和总体标准差；包含命令变化，不等于固定命令下的稳态抖动 |
| `height_{low,middle,high}/swing_clearance_m` | 计划摆动期的实际轮缘法向净空均值；拖地也计入 |
| 同一高度组的 `swing_target_m`、`swing_shortfall_m` | 同批摆动样本的相位目标均值、`max(target-clearance,0)` 均值 |
| 同一高度组的 `leg_soft_limit_excess_rad` | 每步六个腿关节软限位越界量之和，再按该高度组求均值 |
| 同一高度组的 `peak_target_m` | 当前滤波后的地形目标峰值均值 |
| `CycleMean/{vx,vy,yaw}_{abs,rms}_{m_s,rad_s}` | 满周期速度/yaw 跟踪误差均值的绝对值或 RMS；所有命令均生效 |
| `CycleMean/{height,roll,pitch}_{abs,rms}_{m,rad}` | 高度及坡面相对姿态的周期平均误差；不是瞬时误差 RMS |
| 同一高度组的 `swing_samples`、`step_samples` | 有效摆动脚样本数、控制步样本数；零样本时数值零不表示表现良好 |
| `WheelRaw/over_limit_rate`、`WheelRaw/same_limit_rate` | 原始策略轮输出越过诊断阈值的比例；不代表执行目标，Foot PI 参考始终为 0 |

高度组按每个样本当前的高度命令分配：low `<0.70 m`、middle `[0.70,0.80) m`、high `≥0.80 m`；同一 episode 可贡献多个高度组。净空统计只排除无效扫描与非计划摆动，不使用实际离地或对侧支撑门控，以显示未完成的抬脚。此轮新增统计使用 CommandManager 的执行后更新，排除 reset 状态；由于 Isaac 先重置已结束环境，终止那一步不纳入这些新增统计。reset 时合并当批结束环境的对应样本并清零；存活环境的累计值保留。

优先比较低高度组的 `swing_shortfall_m` 与 `leg_soft_limit_excess_rad`：两者同时偏高说明净空目标和关节余量可能冲突，需要结合视频与目标峰值判断。当前去除整体坡面后的起伏度决定每周期 `5/10 cm` 档位与 `0.65～0.85 m` 高度范围。

已完成的 `2026-09-15_00-48-24_foot_terrain_phase_modes_v3/model_20000.pt` 不能作为合格 Foot checkpoint：末 500 iterations 的直行 XY RMSE 约 `0.380 m/s`，三档高度的计划摆动期平均净空只有约 `2.8～3.2 mm`，地形等级约 `0.002`，轮目标裁剪率约 `82.9%`。MuJoCo 确认其长期双轮着地并主要依靠轮速滚动/差速转向。完整诊断位于 `agent_docs/temp/foot_failure_2026-09-15/report.md`；其中的卸载和净空改造已于 2026-09-16 写入正式配置，并加强实际轮速约束、关闭 XY 加速度成本。后续 v4 也已复测失败：前进双轮接地率 100%、无离地事件。当前 v5 尚无新训练结果；建议从 h08 模型做短微调并验收交替卸载、摆动净空、实际轮速和条件速度误差，再决定是否长训。另检查短暂双轮同时腾空：当前成本随腾空时间平方增长，`0.05/0.10 s` 仅扣 `0.01/0.04`（未乘 dt），不能保证至少一脚始终支撑。

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
Reward/pen_wheel_contact
Reward/pen_base_height
Reward/stand_still
Reward/pen_rolling_error
Reward/pen_wheel_stance_slip
Reward/pen_wheel_target_symmetry
Reward/pen_zero_command_wheel_target
Reward/pen_base_lin_acc_xy
Reward/pen_base_yaw_acc
Reward/pen_base_contact_termination
Reward/pen_terrain_orientation
```

Foot 还应重点查看下列指标。`pen_base_lin_acc_xy` 当前权重为零，不应再把它当作训练约束；奖励项绝对值受新系数影响，不能直接与 v3 的总奖励比较。实际轮速、卸载曲线和短暂双轮腾空还需在 rollout 中记录或结合视频检查：

```text
Reward/pen_lin_vel_xy_tracking_error
Reward/pen_yaw_tracking_error
Reward/gait_contact_schedule
Reward/pen_feet_regulation
Reward/rew_swing_clearance
Reward/pen_planned_support_contact
Reward/foot_landing_vel
Reward/pen_wheel_actual_speed
CycleMean/height_abs_m
Reward/pen_knee_contact_force
Episode/Episode_Termination/sustained_knee_contact
Episode/Metrics/base_velocity/{in_place,straight,lateral,yaw_only,mixed}/{fraction,samples,xy_rmse,wz_rmse,wz_mean,wz_std}
Episode/Metrics/base_velocity/height_{low,middle,high}/{swing_samples,step_samples,swing_clearance_m,swing_target_m,swing_shortfall_m,leg_soft_limit_excess_rad,peak_target_m}
WheelRaw/{over_limit_rate,same_limit_rate}
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

MuJoCo 单专家部署使用 `mujoco/deploy_wheel_policy.py --mode wheel|foot`。Foot 自动读取 checkpoint 旁的步态配置，但轮速参考由当前控制器固定为零；Wheel 仍使用策略轮速目标。旧模型可以形状兼容，却不保证适应 v7 控制语义。

新 Foot checkpoint 部署时必须连同同目录的 `params/env.yaml` 一起保留。MuJoCo 会从其中读取 gait 范围中点以及 `continuous_phase` 标志；命令行 `--gait-frequency`、`--gait-offset`、`--gait-duration`、`--swing-height` 可以逐项覆盖。启动后先核对终端中的 `gait_clock=continuous|legacy`、`gait=(...)`、`wheel_kp`、`wheel_ki` 和 `wheel_target`，再开始比较策略表现。

当前 MuJoCo 状态链路还包含三项与 Isaac Lab 对齐的修复：世界系质心速度旋转到 `base_Link` 坐标系；策略、扫描、遥测和渲染读状态前刷新运动学缓存；高度扫描未命中编码为 `0` 且不参与局部平面拟合。旧部署脚本生成的对照结果若未包含这些修复，不应与当前结果直接混合统计。

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

TensorBoard 的 `Reward/<term>` 已包含奖励权重和 `dt`，并用最大 episode 时长归一化。滚动项用于观察密集轮速一致性；滑移项均值较小并不自动表示无效，还应检查它是否只在少量严重滑移时出现尖峰。

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
- 腿关节 `pen_vel_non_wheel_l2: -0.03 -> -0.001`；当时轮关节保留 `pen_joint_vel_wheel_l2=-0.10`，该历史设置已在 2026-09-09 被软锁轮奖励替换；
- `pen_terrain_orientation: -8.0 -> -10.0`，继续使用局部地形法向，不恢复世界系水平姿态项；
- Foot 的 `pen_base_height=-30.0` 改为与 Wheel 相同的 121 点局部平面拟合和法向高度，门控改为 `max(C_L,C_R)`；
- 新增 `rew_base_height_exp=+1.0`，`std=0.05 m`，与高度 L2 共用局部平面有效性和 `max(C_L,C_R)` 门控；
- `gait_contact_schedule=+1.0` 保持 PF 的 `GaitReward` 公式和权重；
- 删除固定高度跟踪 `pen_swing_height=-4.0`，改为 `pen_feet_regulation=-0.1`：沿每个轮端自己的局部地形法向计算间隙，沿局部切平面计算速度，沿用 PF 的指数衰减结构但不指定摆动高度；
- 删除接触后冲击 `pen_wheel_landing_impact=-0.02`，改为 `foot_landing_vel=-0.5`：轮端距离局部地形小于 `0.08 m`、尚未接触且沿局部法向下降时惩罚速度平方；
- gait command 第四维 `swing_height` 固定为 `0.0`，只用于保持四维网络 schema，不再作为奖励目标。策略依靠 121 维地形扫描和碰撞/能耗结果自行学习所需摆高。

该历史版本仍保留固定支撑端间距、双轮同时腾空、轮端滑移、实际轮速等轮足专用约束，当时通过了真实 Isaac Lab CPU 1-env 冒烟。后续横向宽度、自适应摆动净空、五类命令和连续相位改动见上方 Foot 训练说明；2026-09-15 的 v3 已完成训练但退化为持续双轮接地滚动，不能把高 timeout 比例当作步态成功。

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

## 18. 2026-09-08 Wheel PI 轮速控制与坡上驱动修订

S02 的部分 checkpoint 可以初次爬坡，但反复爬坡后无法恢复轮速。新版做如下修订：

- Wheel 网络动作定义不变，后两维仍是左右轮目标速度；
- 物理层从 PhysX 隐式纯 P 速度 drive 改为 `tau=Kp×e+Ki×integral(e)` 显式力矩控制，以 `0.005 s` 物理步长更新；
- 默认 `Kp=2.0 N·m/(rad/s)`。每次 episode reset 采样 `Kp=2.0×Uniform(0.25,2.0)`，即 `0.5～4.0`；
- 每个环境再采样 `Ki/Kp=Uniform(0,0.25) 1/s`，左右轮共用同一组 PI 增益；
- 轮力矩限制为 `±80 N·m`。当未限幅力矩超限且当前误差仍推向同一饱和方向时，暂停对该轮的积分；reset 时积分清零；
- Wheel 隐式 stiffness/damping 都置零，原有启动和 reset 级执行器增益随机化仅保留给 6 个腿关节，避免在显式 PI 外叠加隐式 PD；
- `pen_base_lin_acc_xy: -0.02→0.0`，保留配置但 RewardManager 在权重为零时跳过函数；
- `pen_base_yaw_acc: -0.02→-0.025`，继续使用 Charbonnier 核抑制 `wz` 抖动。

Isaac Sim Play 使用确定性 `Kp=2.0`、`Ki=0.5`。MuJoCo 默认也是该值，可用 `--wheel-kp` 和 `--wheel-ki` 独立调整；`--wheel-kv` 作为 `--wheel-kp` 的兼容别名保留。这一版必须从头训练，推荐 run name 为 `wheel_pi_drive_yaw_v4`。

## 19. 2026-09-09 至 2026-09-15 Foot、部署与诊断修订

这一轮改动跨越训练动作、命令、奖励、部署和测试，不能只替换 checkpoint 后继续沿用旧结论：

- Foot 从轮目标硬置零改为 `±1 rad/s` 策略目标，并与 Wheel 共用 `200 Hz` 显式 PI、`±80 N·m` 力矩限制和 anti-windup；Dual FSM 对全部 8 维动作插值。
- 高度命令修复了高级索引采样未写回问题；修复后新训练才真正得到 `10%` 最低端点、`10%` 最高端点和 `80%` 区间均匀样本。
- Foot 横向构型改为航向系有符号宽度 `0.30～0.38 m`，不再用 XY 总距离限制前后步幅；持续小腿承重增加力惩罚和 `0.08 s` 终止保护。
- Foot 摆动净空改为地形扫描极差驱动的 `2～10 cm` 峰值、sin² 相位包络和非对称 Huber 成本；零速度命令仍执行步态。
- Foot 命令改为 `15/25/10/20/30%` 五类互斥采样，相位改为连续累积；日志增加分模式、分高度、腿软限位和轮目标裁剪诊断。
- MuJoCo 同步 gait 快照读取、连续相位、PI、Foot 裁剪、机身速度坐标系、缓存刷新和扫描缺失值语义。
- `--resume false` 现在会正确覆盖配置中的 `resume=True`，不会再因 `bool("False")` 造成意外加载 checkpoint。

`foot_terrain_phase_modes_v3/model_20000.pt` 已覆盖上述大部分 Foot 奖励与诊断，但实测退化为长期双轮接地滚动，不能作为成功基线。后续任何新 Foot run 都应同时验收接触时序、计划摆动净空、五类命令条件误差、轮目标裁剪率和分地形成功率，而不是只看 episode length 或最终 checkpoint。

## 20. 2026-09-16 v5：宽跟踪核、4 cm 下限与指数净空奖励

恢复的是 h08 的 XY/yaw 核宽与指数净空评分形式，保留当前 `4/2` 跟踪权重、连续相位、五类命令与所有速度条件下迈步的要求。日志名称从 `pen_swing_clearance` 改为 `rew_swing_clearance`，分高度的净空、相位目标、shortfall 和峰值指标名称保持；指标仍记录计划摆动脚，即使实际脚未离地也不丢弃。104 项 CPU 回归和训练环境中 21 项相关测试通过；尚未完成本版 PhysX 训练或新策略 MuJoCo 验收。


## 21. 2026-09-16 v6：正奖励翻倍与独立周期漂移成本

当前为 **v12**：Foot/Wheel 的 policy 和 critic 均追加六维零命令保持误差/启用标记，与奖励共用同一 tracker；MuJoCo 同步支持。policy 161D、Actor 168D，history/Encoder 仍 340D。Foot 保留 v11 的 2 cm 中段最低净空成本 −1.0、漏迈 −0.2/次及此前跟踪权重。**从头训练，不微调**；详见 [v12 观测与训练说明](wf_hold_observation_v12.md)。

旧 v5 微调约 600 轮已复现抬腿和平移退化：固定平地前进命令 0.4 m/s 时，20100/20600 模型实际 vx 约 0.174/0.090 m/s，后者双轮接地 100%。这是 v5 对照，不是 v6 结果；报告位于 `agent_docs/temp/foot_finetune_v5_2026-09-16/report.md`。
