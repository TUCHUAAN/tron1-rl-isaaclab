# Foot v11：独立的摆动中段最低净空要求

> 历史阶段记录：零命令保持误差观测现已在 [v12](wf_hold_observation_v12.md) 加入；当前请使用 v12 从头训练命令。

更新于 2026-09-21。当前只实施两步方案的第一步：改善左右腿独立迈步。用户明确要求从头训练，不微调、不加载旧模型。未启动训练。

## 相对 v10 的改动

| 项目 | v10 | v11 |
|---|---:|---:|
| `pen_swing_min_clearance` | 无 | −1.0，新增密集成本 |
| `pen_missed_swing` | −0.1/次 | −0.2/次 |

XY/yaw/高度指数权重保持 5.5/3.0/1.5，yaw Huber −1.0，高度 L2 −30；核宽、周期平均项、两档摆线净空正奖励、零轮速参考、落脚区域、计划支撑项均保持。Wheel 奖励不变，仍有 v10 零命令保持。

## 最低净空成本

实现为 `swing_min_clearance_shortfall` 与适配器 `swing_min_clearance_penalty`，只接入 Foot。

每只脚的摆动进度 u 从 0 到 1。支撑期、摆动前 20% 和后 20% 不处罚；20%～35% 用 smoothstep 从 0 增至 1；35%～65% 全强度；65%～80% 平滑降至 0。

```text
shortfall_i = clamp((0.02 - clearance_i) / 0.02, 0, 1)
raw_cost = sum_i(envelope_i * shortfall_i)
step_reward = -1.0 * raw_cost * step_dt
```

clearance 是轮心到该轮局部地形平面的法向距离减轮半径 0.128 m，不使用固定世界 Z。全强度时，贴地成本 1、净空 1 cm 成本 0.5、净空≥2 cm 成本 0。低于地面时成本封顶；过高不额外给奖励。

左右分别计算后相加，一条腿达到高度不能抵消另一条腿的不足。没有 moving 门控、不读取另一条腿支撑置信度，原地/前后/侧移/纯转向同样生效。无效局部扫描或非有限高度只跳过对应脚，不屏蔽另一脚。原有 `rew_swing_clearance` 继续塑造 5/10 cm 摆线；本项仅负责贴地到有效离地的连续激励。

漏迈事件仍在完整摆动结束后结算一次，要求之前支撑后净空≥1 cm、无接触≥60 ms；系数改为 −0.2。事件项抵消 RewardManager 的 dt，新密集项正常乘 dt，两者不能直接按数字大小比较。

## 零命令保持的第二步

本轮没有加入位置/航向保持误差观测，没有修改网络输入或部署控制器。v10 奖励锚点仍然不在策略观测中，不能声称已解决主动回正。先验证迈步改善，再实施 Foot/Wheel 共用误差观测及训练/部署同步；后续按用户要求从头训练，不以旧 checkpoint 迁移为默认方案。

## 从头训练

在仓库根目录、激活 isaaclab4.5 后执行；可先创建 `tmux new -s foot_v11`。

```bash
export PYTHONPATH="$PWD/exts/bipedal_locomotion:$PWD/rsl_rl${PYTHONPATH:+:$PYTHONPATH}"
PYTHONWARNINGS=ignore python scripts/rsl_rl/train.py \
  --task Isaac-Limx-WF-Foot-AllTerrain-v0 \
  --num_envs 4096 \
  --resume false \
  --max_iterations 20000 \
  --save_interval 500 \
  --run_name foot_swing_floor_v11 \
  --headless \
  --device cuda:1 \
  --kit_args="--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error"
```

日志：`logs/rsl_rl/wf_tron_1a_foot_all_terrain/<时间戳>_foot_swing_floor_v11/`，不覆盖历史模型。

## 验证与验收

158 项 CPU 回归通过；新增测试覆盖贴地到 2 cm 的单调成本、过高/负净空封顶、左右独立、边界平滑、局部扫描无效隔离、实际 Foot 配置与奖励适配器、另一腿接触消失不屏蔽成本。尚未运行完整 Isaac Lab/PhysX smoke 或训练新策略。

训练后用与 v10 19500 相同的评测协议，重点比较原地、右侧移的左右有效离地率及净空，避免只看总 reward。还要检查双脚腾空、速度跟踪和高度抖动是否变差。
