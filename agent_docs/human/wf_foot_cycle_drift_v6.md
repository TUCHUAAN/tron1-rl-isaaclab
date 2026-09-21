# Foot v6：奖励翻倍与独立周期积分（2026-09-16）

> 本文是 v6 历史归档。两个零命令周期项及其 manager 已在 v7 删除，当前配置见 [v7 说明](wf_foot_cycle_mean_v7.md)。


## 1. 生效范围和参数

仅修改 `Isaac-Limx-WF-Foot-AllTerrain-v0` 的 Foot 奖励；Wheel/PF 不变。当前运行中的旧训练进程不会热更新这些配置，必须启动新进程才生效。

| 项目 | v5 | v6（历史） |
|---|---:|---:|
| `rew_lin_vel_xy` | +4.0 | **+8.0** |
| `rew_ang_vel_z` | +2.0 | **+4.0** |
| `rew_swing_clearance` | +0.5 | **+1.0** |
| `pen_base_yaw_acc` | -0.025 | **0.0**，保留配置但跳过计算 |
| `pen_zero_vy_cycle_drift` | 无 | **-0.2**，`error_scale=0.02 m` |
| `pen_zero_yaw_cycle_drift` | 无 | **-0.2**，`error_scale=0.03 rad` |

三个正奖励翻倍是用户确认值；两个新增负权重、尺度与 `zero_command_threshold=1e-6` 是可配置的试验初值，并非已验证的最优参数。保持 XY/yaw 指数核 `std²=0.20/0.25`，净空目标峰值 `4～10 cm` 和 `std=0.025 m` 不变。瞬时 XY/yaw Huber、其他动作平滑/轮速/膝保护项、轮 PI、命令采样及地形均不变。

## 2. 独立门控

| 整个窗口内的对应命令 | vy 积分 | 净偏航 |
|---|---|---|
| `vy_cmd≈0`，`wz_cmd≈0` | 启用 | 启用 |
| `vy_cmd≈0`，`wz_cmd≠0` | 启用 | 关闭 |
| `vy_cmd≠0`，`wz_cmd≈0` | 关闭 | 启用 |
| 二者非零 | 关闭 | 关闭 |

“≈0”指绝对值 `<=1e-6`（vy 单位 m/s，yaw 单位 rad/s）；不是根据实际速度判定，也不是要求 vx=0。任一对应命令非零会清空其自己的窗口；重新归零后需连续收集满一周期。另一分量变化不清空本项；零命令之间的重采样也不清空。episode reset 只清空对应环境的两项历史，其他并行环境不受影响。第一步仅建立相位/航向基准，不计入窗口，避免把 reset 跳变当作运动。

## 3. 一个完整步态周期的滑动窗口

每个控制步回看累计 gait phase 恰好推进 `1.0` 圈的区间。恒定步频 f 时，时长 T=1/f；当前 `1.2～2.2 Hz` 对应约 `0.833～0.455 s`。变步频时按历史实际连续相位推进量确定区间，而不是用最新频率重解释旧时间。最旧样本跨越周期边界时按相位占比线性分摊其积分和时长。

窗口满后每步更新，不是在相位归零时才扣一次。未满一个周期不计算半周期成本；允许完整周期内的左右摆动相互抵消。对应命令非零、无效输入或缺失控制步会丢弃历史；同一步重复读取不重复累计。两项均无接触/跟踪软门控，无额外免罚死区。

## 4. 两种有符号积分

### vy：机体系侧向速度累计

`I_y = Σ(v_body_y × dt × window_fraction)`。

允许转向时启用：累计的是每步机体系 vy 的有符号偏差，不是沿窗口起点方向投影的世界位移。转弯时参考方向变化，因此不要把该量称为固定世界方向的净侧移。正常左右重心转移可以抵消；不使用 `Σ|vy|dt` 或 `Σvy²dt`。

### yaw：实际航向小增量累计

从 `root_link_quat_w` 的机身前向轴水平投影提取航向 psi，计算 `delta_psi=atan2(sin(psi_t-psi_prev),cos(psi_t-psi_prev))`，累计最近一圈的这些小增量（含边界比例）。不是直接积分机体系 wz，避免把有 roll/pitch 时的 body-z 角速度等同于航向导数。前向轴接近竖直时航向不可靠，清空该环境的窗口。

实际采样假定相邻 20 ms 控制步间航向变化小于 pi；按最短角差处理跨越 ±pi，不对整个周期的总和再做角度取模。

### Huber 与时间尺度

`cost = Huber(I / error_scale)`，其中 Huber(x)=0.5*x²（|x|<=1），否则为 |x|-0.5。函数返回非负原始成本，由 RewardManager 乘负权重和 policy dt。vy 积分中的 dt 是物理积分所需，不能因此在 reward 返回值再手动乘奖励 dt。

例如净侧向积分 0.02 m 或净偏航 0.03 rad 都对应原始 cost=0.5；权重 -0.2、dt=0.02 下，该项当步贡献为 -0.002。小于尺度也有小惩罚，尺度不是免罚阈值。

该方案不是长期位置/航向锁定，允许周期内回摆，也无法保证回到初始绝对位置或航向；原有瞬时速度跟踪继续约束周期内误差。

## 5. 网络、日志与部署

Actor/Critic/History 输入维度不变；两个奖励实例维护独立的历史缓冲，不向观测追加积分状态。h08 权重形状仍兼容，MuJoCo 无需实现训练奖励。部署仍须保留新 run 的 `params/env.yaml`。保持网络兼容的代价是积分状态对网络不完全显式可见；这些测试不证明历史奖励的可观测性或训练收敛。

新增奖励曲线：`Episode/Episode_Reward/pen_zero_vy_cycle_drift`、`pen_zero_yaw_cycle_drift`。

新增条件诊断位于 `Episode/Metrics/base_velocity/zero_vy_cycle/` 与 `zero_yaw_cycle/`：
- `samples`：已满周期且当前 episode 正常运行的控制步样本数，不是独立周期数；
- `valid_fraction`：上述样本占全部有效控制步的比例（包含命令门控和启动等待影响）；
- `abs_integral_m` / `abs_integral_rad`：只对有效窗口统计的绝对积分均值；
- `rms_integral_m` / `rms_integral_rad`：有效窗口积分 RMS；
- `duration_s`：有效窗口实际时长均值。

零样本时指标返回零，必须结合 samples / valid_fraction 解读；不能把“未启用”当作“没有漂移”。日志不修改奖励缓冲，也不增加网络输入。

## 6. 训练入口和验证范围

从原 h08 `2026-09-11_14-32-20_foot_width_clearance_h08_v1/model_20000.pt` 新起微调，不覆盖 v5 run。新微调父目录使用 `logs/foot_finetune_v6/`，后缀 `h08_to_cycle_drift_v6`；从头训练后缀 `foot_cycle_drift_v6`。完整命令见[训练说明](wf_dual_mode_training_process.md#foot-从-h08-模型初始化)。

CPU 测试覆盖满周期前零成本、符号抵消、恒定漂移、非整数采样周期、变步频、独立命令门控、命令切换、局部 reset、角度跨界、无效输入、重复读取与配置绑定；另有真实 Isaac 环境 smoke 检查脚本。数学与接口测试不代表已经学会抬腿或改善跟踪；本次未启动 v6 训练。

本次验证记录：UDMMR 环境 `123` 项 CPU 回归通过，isaaclab4.5 Python 环境 `14` 项周期积分测试通过。实际 PhysX smoke 尝试在 Isaac Sim 启动阶段被交互式 EULA 确认阻断（非交互输入 EOF），尚未运行环境仿真；未自动接受许可。
