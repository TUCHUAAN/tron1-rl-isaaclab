# TRON1A 最新轮式策略的 MuJoCo 部署

该目录直接读取以下三个外部输入，不复制模型资源：

- 地形：`/home/tuchuaan/UDMMR/models/terrains/training_terrain.xml`
- 机器人：`/home/tuchuaan/tron1-mujoco-sim/robot-description/pointfoot/WF_TRON1A/xml/robot.xml`
- 策略：自动选择 `logs/rsl_rl/wf_tron_1a_wheel_mode` 下修改时间最新的 `model_*.pt`

运行：

```bash
/home/tuchuaan/miniconda3/envs/UDMMR/bin/python mujoco/deploy_wheel_policy.py
```

启动时会在终端选择地形类型、等级或综合地形号码。窗口按键与 UDMMR 的键盘仿真保持同一套运动语义：

- `W/S`：前进/后退
- `Q/E`：左移/右移
- `Z/C`：升高/降低机身，范围 0.75–0.85 m
- `A/D`：左转/右转
- `Space`：清除当前运动指令
- `R`：重置机器人；终端中可复用当前地形或重新选择
- `Esc`：退出

摄像头会根据机身航向持续更新，固定跟在机器人右后方。需要指定 checkpoint 或跳过交互选择时，例如：

```bash
/home/tuchuaan/miniconda3/envs/UDMMR/bin/python mujoco/deploy_wheel_policy.py \
  --checkpoint logs/rsl_rl/wf_tron_1a_wheel_mode/2026-08-05_23-22-04/model_20000.pt \
  --terrain-type 2 --terrain-level 3
```

无窗口冒烟测试：

```bash
/home/tuchuaan/miniconda3/envs/UDMMR/bin/python mujoco/deploy_wheel_policy.py \
  --headless --no-realtime --max-steps 200 --terrain-type 1 --terrain-level 1
```

注意：当前最新 checkpoint 是轮式专家，训练时横向速度范围固定为 0；因此 `Q/E` 虽然按参考脚本接入了横向指令，但属于训练分布之外的输入，表现可能不稳定。
