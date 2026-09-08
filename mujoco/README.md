# TRON1A Wheel / Foot 专家的 MuJoCo 部署

该目录直接读取以下三个外部输入，不复制模型资源：

- 地形：`/home/tuchuaan/UDMMR/models/terrains/training_terrain.xml`
- 机器人：`/home/tuchuaan/tron1-mujoco-sim/robot-description/pointfoot/WF_TRON1A/xml/robot.xml`
- 策略：`--mode wheel` 搜索 Wheel 日志，`--mode foot` 搜索 Foot 日志；优先读取最新的 `selected_stability_checkpoint.txt`，没有选择文件时回退到修改时间最新的 `model_*.pt`

运行：

```bash
/home/tuchuaan/miniconda3/envs/UDMMR/bin/python mujoco/deploy_wheel_policy.py
```

部署最新 Foot 专家：

```bash
/home/tuchuaan/miniconda3/envs/UDMMR/bin/python mujoco/deploy_wheel_policy.py \
  --mode foot
```

Foot 模式保留网络原始 8 维输出供 `last_action` 观测使用，但在执行器层将左右轮目标速度硬锁为 `0 rad/s`，与训练环境的 `JointVelocityActionCfg(scale=0)` 一致。默认 `wheel` 模式保持原有行为。

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
  --mode wheel \
  --checkpoint /path/to/new_h065_085_model.pt \
  --terrain-type 2 --terrain-level 3
```

默认不对最终关节力矩做滤波。如需在 200 Hz MuJoCo 控制循环中对 8 个关节的最终施加力矩启用一阶低通滤波，可使用：

```bash
/home/tuchuaan/miniconda3/envs/UDMMR/bin/python mujoco/deploy_wheel_policy.py \
  --mode wheel \
  --torque-low-pass \
  --torque-cutoff-hz 20
```

`--torque-cutoff-hz` 默认是 `20 Hz`，但只有同时指定 `--torque-low-pass` 才会生效。机器人重置时滤波器状态也会清零，避免沿用重置前的关节力矩。

可视化运行时会同时打开第二个实时曲线窗口，默认显示最近 `20 s`：

- `vx / vy / wz`：虚线为命令，实线为机身坐标系实际速度；
- `height`：虚线为高度命令，实线为 base 到 121 点扫描所拟合局部地形平面的法向距离；
- `roll / pitch`：世界系 ZYX 欧拉角，单位为度；虚线为保持当前航向并使 base-up 对齐局部地形法向时的参考姿态。

曲线使用轻量 Tk Canvas 绘制，默认以 `20 Hz` 采样、`5 Hz` 刷新，避免影响控制循环。可用 `--plot-sample-hz 10`、`--plot-hz 2` 和 `--plot-history 30` 调整采样率、刷新率与显示时长；`--no-plot` 可只保留 MuJoCo 主窗口，`--headless` 不创建任何窗口。关闭曲线窗口不会结束仿真，退出仍使用 MuJoCo 主窗口的 `Esc`。

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
