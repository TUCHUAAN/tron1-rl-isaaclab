# v12：Foot / Wheel 零命令保持误差观测

更新于 2026-09-21。v11 的迈步改动保留，本轮完成零命令保持的观测与 MuJoCo 同步。用户要求从头训练，不迁移旧模型参数。没有启动训练。

## 六维状态与网络布局

Foot、Wheel 的 policy 与 critic 末尾各追加同样的六维，不额外加噪声；obsHistory/Encoder 保持原尺寸。

| 新增分量 | 定义 | 归一化/范围 |
|---|---|---|
| error_x | 实际位置相对参考的前后误差 | 除以 position_scale=0.05 m，限幅 ±5 |
| error_y | 实际位置相对参考的侧向误差 | 同上 |
| error_yaw | 实际 yaw 减参考 yaw，取最短角差 | 除以 yaw_scale=0.10 rad，限幅 ±5 |
| active_x | vx 零命令保持已启用 | 0/1 |
| active_y | vy 零命令保持已启用 | 0/1 |
| active_yaw | wz 零命令保持已启用 | 0/1 |

正值表示实际相对参考的正向偏移，策略应学习反向纠正。未启用的误差分量严格置零；误差在奖励死区内仍可见，不先减死区。全零保持时，把固定世界位置锚点到当前位置的向量按当前航向旋转到机身水平坐标；部分零命令时，暴露与奖励相同的 1 s 有符号位移窗口分量。全零时三个 active 均为 1，因此策略可识别固定位置保持状态。

| 项目 | v11 及之前 | v12 |
|---|---:|---:|
| policy | 155 | 161 |
| critic observation | 原尺寸 | 原尺寸 +6 |
| history 单帧 / Encoder 输入 | 34 / 340 | 34 / 340 |
| Encoder latent | 3 | 3 |
| commands | 4 | 4 |
| Actor 总输入 | 162 | 168 |
| Actor 输出 | 8 | 8 |

policy `[0:155]` 保留，`[155:158]` 为误差，`[158:161]` 为标记。Actor 实际拼接顺序为 latent `[0:3]`、policy `[3:164]`、commands `[164:168]`；新保持六维位于 Actor `[158:164]`。Critic 的原有特权信息顺序不变，在末尾追加六维，然后 runner 再拼接 commands。Encoder 仍学习原有历史，不重复加入保持状态。

## 与奖励共用同一个参考

训练 `zero_command_hold_observation` 读取 `ZeroCommandHoldPenalty` 的 tracker，没有第二套锚点或计时。`ZeroCommandHoldTracker.observation()` 输出上述六维。奖励与观测的 position_scale/yaw_scale 因而一致。

- 每个实际控制步只推进一次位置历史和判零计时。
- Isaac 的奖励计算先于命令重采样；观测读取时额外 synchronize_command：立即释放新非零分量、清空其历史，新零分量从零计时，不多推进 dt。
- actor、critic、多次观测读取均不额外推进状态。
- reset 单独清空相应环境的锚点、窗口及标记；第一次读取只记录初始位置，不把 reset 当成已流逝的控制时间。
- ObservationManager 构建时早于 RewardManager；此时仅返回六维零张量供维数探测，运行时再使用真实 reward tracker。
- 仍需启用 `pen_zero_command_hold`，当前两种模态均为 −0.5；本轮没有改它的容差或权重。

## MuJoCo 与兼容性

新 161D policy 根据 actor/encoder 输入尺寸识别，必须保留 checkpoint 同目录的 `params/env.yaml`。部署检查 zero_command_hold 观测声明和启用的奖励项，读取其 tracker_options，直接加载训练使用的纯 PyTorch tracker。

每次策略调用先处理刚执行完的上一命令区间，再同步本次键盘命令，与 Isaac 的 reward→command→observation 时序一致。R reset 清空保持状态。旧 155D policy 仍走旧输入，不追加六维，不要求存在新配置；两种版本的历史编码器都为 340D。

Dual Play 使用当前 161D 环境；两个专家都需由新输入结构训练。FSM 覆盖速度命令后会重新同步保持误差/标记。旧 162D Actor 不能直接加载到新 168D Actor 中；本轮不提供迁移/微调，而按用户要求从头训练。旧模型可继续用 MuJoCo 旧维数分支测试。

这是提供回正所需的信息，不是直接增加位置 PID。实际回正能力、长期稳定性与迈步仍需新训练后的实测。

## 从头训练 Foot

```bash
cd /home/tuchuaan/tron1-rl-isaaclab
conda activate isaaclab4.5
export PYTHONPATH="$PWD/exts/bipedal_locomotion:$PWD/rsl_rl${PYTHONPATH:+:$PYTHONPATH}"
PYTHONWARNINGS=ignore python scripts/rsl_rl/train.py \
  --task Isaac-Limx-WF-Foot-AllTerrain-v0 \
  --num_envs 4096 \
  --resume false \
  --max_iterations 20000 \
  --save_interval 500 \
  --run_name foot_hold_observation_v12 \
  --headless \
  --device cuda:1 \
  --kit_args="--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error"
```

Wheel 同样从头训练，task 改为 `Isaac-Limx-WF-Wheel-Mode-v0`，run_name 改为 `wheel_hold_observation_v12`。分别运行，不同时占用同一显卡。

## 验证

163 项 CPU 回归通过，新增测试覆盖：初始维数探测、旋转坐标/归一化/限幅、非有限输入与停用清零、训练与 MuJoCo 100 步逐帧对齐（包含零/非零命令切换、跨 ±π、reset、重复读取）、真实 MuJoCo 小模型上合成新旧 checkpoint 的 actor 输入/输出与历史尺寸、新模型缺失快照时报错、policy/critic 配置绑定。

合成 checkpoint 仅用于接口测试，不代表训练效果。旧 v10 19500 完成 2 秒真实机器人 MuJoCo headless 回放，验证旧维数兼容。完整 Isaac Lab smoke 已尝试，但当前 Python 环境在 Omniverse EULA 提示处因无交互输入退出，尚未进入仿真；没有替用户接受许可，也不将其记为通过。新模型训练和实机效果未验证。
