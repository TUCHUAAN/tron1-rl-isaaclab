# 原版 PF MuJoCo 测试

该目录用于部署 `Isaac-Limx-PF-Blind-Flat-v0` 训练得到的原版
`PF_TRON1A` checkpoint，不使用 WF Foot Expert 的部署接口。

## 直接测试 model_2000

```bash
cd /home/tuchuaan/tron1-rl-isaaclab

/home/tuchuaan/miniconda3/envs/UDMMR/bin/python \
  pf_mujoco_test/deploy_pf_policy.py \
  --checkpoint logs/rsl_rl/pf_tron_1a_flat/2026-09-03_21-54-16_original_pf_baseline/model_2000.pt \
  --device cpu
```

不传 `--checkpoint` 时，脚本会在
`logs/rsl_rl/pf_tron_1a_flat/` 中寻找修改时间最新的
`model_2000.pt`：

```bash
/home/tuchuaan/miniconda3/envs/UDMMR/bin/python \
  pf_mujoco_test/deploy_pf_policy.py
```

原版 PF 只在平地上训练，因此脚本默认直接加载
`/home/tuchuaan/tron1-mujoco-sim/robot-description/pointfoot/PF_TRON1A/xml/robot.xml`
及其自带平面。

脚本按实际 `model_2000.pt` 校验网络接口：每帧 policy/history observation
均为 `30` 维，10 帧历史输入为 `300` 维，Encoder 为 `300 -> 3`，
Actor 为 `36 -> 6`；额外的 `3` 维是 `[vx, vy, wz]` 命令。运行窗口和终端
都会同时显示速度命令与实际机身系速度。

## 键盘

- `W/S`：前进/后退
- `Q/E`：左移/右移
- `A/D`：左转/右转
- `Space`：速度命令归零
- `R`：重置机器人
- `Esc`：退出

默认 gait command 为 `[2.0, 0.5, 0.5, 0.15]`，都在训练范围内。
可通过 `--gait-frequency`、`--gait-offset`、`--gait-duration` 和
`--swing-height` 调整。

## 网络和控制接口

- 当前观测：`30` 维
- 10 帧历史：`300` 维
- Encoder：`300 -> 3`
- 速度命令：`[vx, vy, wz]`，`3` 维
- Actor：`36 -> 6`
- 动作：六个腿关节的位置偏移，`q_target = q_default + 0.25 * action`
- PD：`Kp=40`、`Kd=2.5`，控制频率 `200 Hz`
- Policy：`50 Hz`

MuJoCo XML 的默认力矩上限是 `80 N·m`。可用 `--torque-limit` 调整。
动作裁剪和力矩低通默认都关闭；如需诊断，可分别使用
`--action-clip 5` 和 `--torque-low-pass --torque-cutoff-hz 20`。

## 无窗口冒烟测试

```bash
/home/tuchuaan/miniconda3/envs/UDMMR/bin/python \
  pf_mujoco_test/deploy_pf_policy.py \
  --headless --no-realtime --max-steps 200 \
  --command-vx 0.5 --device cpu
```
