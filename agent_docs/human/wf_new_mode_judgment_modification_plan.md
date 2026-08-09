# Wheel 合规奖励实现说明

## 目标

Wheel Expert 只围绕两个条件训练：

```text
1. 左右轮都形成有效地面接触
2. 轮心在 base 坐标系下的 x/y 接近中性点
```

不实现 Foot gait 时间窗，不修改 FSM，不修改 Reset。

## 1. 双轮有效接触

单轮有效接触：

```text
ground_contact = filtered_ground_force × wheel_ground_geometry
```

新增传感器：

```text
wheel_L_ground_contact
wheel_R_ground_contact
wheel_L_ground_scan
wheel_R_ground_scan
```

- 接触传感器只过滤 `/World/ground/terrain/mesh`，排除自碰撞和非地面接触。
- 轮下射线检查轮心到局部地面的距离是否接近轮半径 `0.128 m`。
- 不使用 actor 的 base height scan 判断轮地接触。

参数：

```text
force_off / force_on: 5 N / 10 N
geometry tolerance on / off: 0.020 m / 0.035 m
```

速度奖励门控：

```text
all_contact_confidence = min(contact_left, contact_right)
velocity_reward *= all_contact_confidence
yaw_reward *= all_contact_confidence
```

失联惩罚：

```text
missing_contact = (1-contact_left) + (1-contact_right)
weight = -4.0
```

## 2. 水平中性点

使用 `model_20000.pt` 稳定轨迹标定：

```text
左轮中位数约 (-0.004,  0.168) m
右轮中位数约 ( 0.001, -0.176) m
```

训练目标采用对称值：

```text
left neutral  = (0.0,  0.17) m
right neutral = (0.0, -0.17) m
```

只检查 base 系 `x/y`：

```text
tolerance x/y = 0.04 / 0.03 m
scale x/y     = 0.02 / 0.02 m
weight        = -4.0
```

超出容差后使用 Huber 惩罚；`z` 不参与，因此允许腿竖直调高和适应坡面。

## 3. 已删除的 Wheel 项

- 任意一轮接触即可开放正奖励的 `amax`；
- `rew_same_foot_x_position`；
- `pen_feet_distance`；
- 轮端竖直速度惩罚；
- 轮轴—地面法向/轮心高度独立惩罚；
- 左右轮 z 差和轮距包络惩罚；
- 轮心 xyz 相对速度惩罚。

## 4. 保留项

- 速度和 yaw 跟踪；
- wheel air time；
- 滚动误差和支撑滑移；
- 落地冲击；
- 高度和地形姿态；
- 动作平滑、力矩、功率和关节限位；
- base contact termination。

## 5. 主要代码

| 文件 | 修改 |
|---|---|
| `cfg/WF/limx_base_env_cfg.py` | 增加可选轮地传感器字段 |
| `robots/limx_wheelfoot_mode_env_cfg.py` | 配置轮地传感器和新 Wheel reward |
| `mdp/reward_math.py` | 双轮门控、轮地几何和水平中性点数学函数 |
| `mdp/rewards.py` | 新 reward adapter，删除旧冲突项 |
| `tests/test_reward_math.py` | 7 个纯张量测试 |
| `tests/smoke_wf_reward_env.py` | 检查新 reward、传感器和旧项删除 |
| `scripts/rsl_rl/cli_args.py` | CLI `--device` 同步到 PPO runner |

## 6. 验证结果

2026-08-05 已完成：

- 纯张量单元测试：`7/7 PASS`；
- Wheel CPU 环境冒烟：`23` 个 reward 全部有限；
- Foot CPU 回归冒烟：`24` 个 reward，`PASS`；
- Wheel CPU 1-iteration 训练：生成 `model_1.pt`；
- 旧 `model_20000.pt` 在新环境运行 400 步：
  - 左/右接触置信度均值约 `0.990 / 0.988`；
  - 双轮置信度大于 0.5 的比例约 `98.75%`；
  - filtered ground force 有有效输出。

## 7. 下一步

用新 reward 开独立 GPU 实验，不覆盖旧日志：

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Limx-WF-Wheel-Mode-v0 \
  --num_envs 4096 \
  --max_iterations 20000 \
  --run_name wheel_contact_neutral_v1 \
  --headless \
  --device cuda:0
```

旧 `model_20000.pt` 只作为 baseline 或 warm start；正式结论以新 reward 重训结果为准。
