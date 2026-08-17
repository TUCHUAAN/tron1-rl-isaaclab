# TRON1A 最新轮式策略的 MuJoCo 部署

该目录直接读取以下三个外部输入，不复制模型资源：

- 地形：`/home/tuchuaan/UDMMR/models/terrains/training_terrain.xml`
- 机器人：`/home/tuchuaan/tron1-mujoco-sim/robot-description/pointfoot/WF_TRON1A/xml/robot.xml`
- 策略：优先读取最新的 `selected_stability_checkpoint.txt`；没有选择文件时才回退到修改时间最新的 `model_*.pt`

运行：

```bash
/home/tuchuaan/miniconda3/envs/UDMMR/bin/python mujoco/deploy_wheel_policy.py
```

启动时会在终端选择地形类型、等级或综合地形号码。窗口按键与 UDMMR 的键盘仿真保持同一套运动语义：

- `W/S`：前进/后退
- `Q/E`：左移/右移
- `Z/C`：升高/降低机身，范围 0.65–0.85 m
- `A/D`：左转/右转
- `Space`：清除当前运动指令
- `R`：重置机器人；终端中可复用当前地形或重新选择
- `Esc`：退出

摄像头会根据机身航向持续更新，固定跟在机器人右后方。需要指定 checkpoint 或跳过交互选择时，例如：

```bash
/home/tuchuaan/miniconda3/envs/UDMMR/bin/python mujoco/deploy_wheel_policy.py \
  --checkpoint /path/to/new_h065_085_model.pt \
  --terrain-type 2 --terrain-level 3
```

策略输入中的机身高度按训练范围归一化：新 checkpoint 使用 `(height - 0.65) / (0.85 - 0.65)`；键盘状态、日志和高度奖励仍使用米。部署脚本会优先读取 checkpoint 同目录下 `params/env.yaml` 的实际范围，因此旧 `0.62～0.85 m` checkpoint 仍会使用旧归一化；缺少配置文件时默认使用新的 `0.65～0.85 m`。

训练完成后先选择稳定 checkpoint，并写入部署选择文件：

```bash
conda run -n isaaclab4.5 python scripts/rsl_rl/select_wf_checkpoint.py \
  logs/rsl_rl/wf_tron_1a_wheel_mode/<run_dir> \
  --window 100 --top 10 --write_selection
```

无窗口冒烟测试：

```bash
/home/tuchuaan/miniconda3/envs/UDMMR/bin/python mujoco/deploy_wheel_policy.py \
  --headless --no-realtime --max-steps 200 --terrain-type 1 --terrain-level 1
```

注意：当前最新 checkpoint 是轮式专家，训练时横向速度范围固定为 0；因此 `Q/E` 虽然按参考脚本接入了横向指令，但属于训练分布之外的输入，表现可能不稳定。

查看最新 Wheel 训练曲线：

```bash
conda activate isaaclab4.5
tensorboard \
  --logdir /home/tuchuaan/tron1-rl-isaaclab/logs/rsl_rl/wf_tron_1a_wheel_mode \
  --port 6006
```

然后在浏览器打开 `http://localhost:6006`。若从另一台电脑访问训练机，增加 `--bind_all`，并访问 `http://训练机IP:6006`。
