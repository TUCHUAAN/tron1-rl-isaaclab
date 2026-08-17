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
  --run_name wheel_single_stage_h065_085_stability_v2 \
  --headless \
  --device cuda:1 \
  --kit_args="--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error"
```

该命令不加载旧 checkpoint，从头同时学习 `0.65～0.85 m` 高度控制、零命令稳定和全地形运动。高度目标中 `20%` 为端点采样：`10%` 精确取 `0.65 m`，`10%` 精确取 `0.85 m`，其余 `80%` 均匀采样。地形从 level 0 开始，通过 curriculum 逐级增加难度，其中上台阶高度为 `0.005～0.04 m`。

`Isaac-Limx-WF-Wheel-Height-Pretrain-v0` 仅保留为可选诊断任务，正式训练不使用它，也不再分阶段。

### Foot

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
Episode/Episode_Reward/pen_wheel_target_symmetry
Episode/Episode_Reward/pen_zero_command_wheel_target
Episode/Episode_Reward/pen_zero_command_yaw_rate
Episode/Episode_Reward/pen_base_contact_termination
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
