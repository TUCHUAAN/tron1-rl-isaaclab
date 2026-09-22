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

注意：截至 2026-09-15，自动搜索到的最新 Foot run `foot_terrain_phase_modes_v3/model_20000.pt` 已确认退化为双轮长期接地并依靠轮速滚动/差速转向，不是合格的 Foot 步态模型。复现实验时建议用 `--checkpoint` 明确指定待测版本，并结合轮端净空与接触时序判断，不要只看是否存活；详见 `agent_docs/temp/foot_failure_2026-09-15/report.md`。

Foot 未显式指定的步态参数会读取 checkpoint 同目录下 `params/env.yaml` 的 `commands.gait_command.ranges`，固定范围取固定值，其余范围取中点。当前 Foot 模型因此使用 `(frequency, offset, duration, swing_height) = (1.7, 0.5, 0.5, 0.0)`；缺少配置文件时也回退到这组 Foot 默认值，并在终端提示。启动日志会打印最终的 `gait=(...)`，当前观测、历史观测和重置均使用这组参数。命令行 `--gait-frequency`、`--gait-offset`、`--gait-duration`、`--swing-height` 可逐项覆盖。Wheel 默认仍为 `(1.7, 0.5, 0.525, 0.14)`。

2026-09-15 起新 Foot 训练启用连续相位：`params/env.yaml` 中 `commands.gait_command.continuous_phase=true`。MuJoCo 自动按保存的标志选择时钟，启动打印 `gait_clock=continuous|legacy`。连续时钟只在策略实际出动作后推进一次，改频率保持当前相位，reset 清零；历史观测和当前观测共用相位。旧快照没有该字段时使用原有 `time×frequency` 公式。部署新模型时应同时保留 `params/env.yaml`，否则无法识别新时钟。

2026-09-09 修正了策略角速度输入与实时速度曲线的坐标系：现在从 MuJoCo 读取世界系质心速度，再旋转到 `base_Link` 坐标系，与 Isaac Lab 一致。此前 `mjOBJ_BODY` 的局部速度使用惯性主轴坐标系；本模型的惯性主轴相对机身约旋转 27°，会混合 roll/yaw 角速度，导致 Foot 策略快速失稳。该修正不改变网络权重。平地短时对照已验证倒地现象改善，行走跟踪和全地形能力仍需单独评估。

2026-09-10 补充修正了状态读取时机和扫描未命中编码：策略、地形扫描、实时曲线及画面读取机身状态前会刷新运动学缓存，使其对应当前 `qpos/qvel`；刷新不推进仿真时间，也不重新求解动力学。扫描未命中时的网络输入改为 `0`，与训练的裁剪结果一致，无效点仍不参与地形平面拟合。训练侧同时修正了高度目标采样的张量写回，后续启动的训练会恢复连续高度目标采样；已有模型的训练数据分布不会因此改变。

Foot 模式始终将轮速参考固定为 0，由 PI 闭环制动；策略最后两维保留以兼容网络形状，但不影响轮速目标。训练保留实际轮速惩罚，删除轮速参考惩罚。Wheel 模式继续使用策略轮速目标。旧 checkpoint 在这个新控制语义下的表现不等同于原控制器表现。

轮速使用带 anti-windup 的 PI 控制，默认为 `Kp=2.0 N·m/(rad/s)`、`Ki=0.5 N·m/rad`。可用 `--wheel-kp` 和 `--wheel-ki` 分别调整，Wheel 和 Foot 模式都生效。`--wheel-kv` 作为 `--wheel-kp` 的兼容别名保留。例如：

```bash
/home/tuchuaan/miniconda3/envs/UDMMR/bin/python mujoco/deploy_wheel_policy.py \
  --mode wheel \
  --wheel-kp 2.0 \
  --wheel-ki 0.5
```

对应轮力矩为 `tau = Kp * error + Ki * integral(error)`，积分在 `200 Hz` 物理控制循环中更新。轮力矩限制为 `±80 N·m`；当力矩已饱和且误差仍继续推向饱和方向时停止积分，按 `R` 重置机器人时积分清零。`--wheel-ki 0` 可退化为纯 P 控制。Foot 模式使用策略目标，但额外执行 `±1 rad/s` 安全裁剪。

部署侧纯 Python 回归测试使用同时包含 `mujoco` 和 `glfw` 的 UDMMR 环境：

```bash
cd /home/tuchuaan/tron1-rl-isaaclab/tests
/home/tuchuaan/miniconda3/envs/UDMMR/bin/python -m unittest \
  test_mujoco_gait_command test_mujoco_height_scan test_mujoco_velocity_frame
```

截至 2026-09-15 共 `19` 个测试，覆盖 checkpoint gait 参数读取、连续/legacy 相位推进、扫描未命中编码、运动学刷新和 `base_Link` 速度坐标变换；这些测试不打开窗口，也不代替实际地形行为验收。

启动时会在终端选择地形类型、等级或综合地形号码。窗口按键与 UDMMR 的键盘仿真保持同一套运动语义：

- `W/S`：前进/后退
- `Q/E`：左移/右移
- `Z/C`：升高/降低机身，范围 0.65–0.85 m
- `A/D`：左转/右转
- `Space`：清除当前运动指令
- `R`：重置机器人；终端中可复用当前地形或重新选择
- 双模态联调时：`1` 切换 Wheel，`2` 切换 Foot；Foot→Wheel 会等待当前步态周期结束
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


### Foot 落脚区域预览

测试命令添加 `--show-footholds`：青色/橙色椭圆为左右脚目标区域，绿/红点为最近有效落点在区域内/外。读取 checkpoint 保存的参数；旧模型没有参数时明确提示使用当前默认值，仅作目标预览，不改变旧策略行为。详细算法和范围见 [v9 说明](../agent_docs/human/wf_foot_foothold_v9.md)。

### v12 零命令保持观测

Foot/Wheel 新模型 policy 为 161D（Actor 总输入 168D），末尾追加 x/y/yaw 保持误差及三个启用标记；保持 tracker 与训练共用，R 重置清空状态。必须保留同一 run 的 `params/env.yaml`。旧 155D policy 继续按旧结构加载，不会自动获得保持观测。历史仍为 34×10；详细时序与从头训练说明见 [v12 文档](../agent_docs/human/wf_hold_observation_v12.md)。
