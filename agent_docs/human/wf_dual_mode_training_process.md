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
| Wheel | `10000` | `500` | `logs/rsl_rl/wf_tron_1a_wheel_mode/` |
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

### Wheel

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Limx-WF-Wheel-Mode-v0 \
  --num_envs 4096 \
  --max_iterations 20000 \
  --save_interval 500 \
  --headless \
  --device cuda:0
```

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
  --logdir logs/rsl_rl/wf_tron_1a_wheel_mode \
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
Episode/Episode_Reward/pen_wheel_contact
Episode/Episode_Reward/pen_wheel_configuration
```

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

## 9. 当前 Wheel 结果

截至 2026-08-05：

```text
run: logs/rsl_rl/wf_tron_1a_wheel_mode/2026-08-04_20-21-08
checkpoint: model_20000.pt
```

| 指标 | 最终值 |
|---|---:|
| Mean reward | `27.96` |
| Mean episode length | `958/1000` |
| Time out | `94.7%` |
| Base contact | `5.3%` |
| Mean terrain level | `4.45` |
| XY 速度误差 | `0.219 m/s` |
| Yaw 速度误差 | `0.275 rad/s` |

结论：该 checkpoint 使用旧 reward 训练，可作为 baseline。新版双轮接触与水平中性点 reward 已实现，需要另开 GPU 实验重新训练。
