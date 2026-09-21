# Foot v7：周期平均跟踪与固定零轮速参考

> v7 历史记录；当前配置已更新为 [v8](wf_foot_gait_v8.md)。

更新时间：2026-09-18。v7 已有训练 run：`2026-09-17_21-39-55_foot_cycle_mean_zero_wheel_v7`。用户反馈实际步态时序和落脚位置仍有问题，尚未验收达标。

## 周期平均跟踪

新增 `pen_cycle_mean_{vx,vy,yaw,height,roll,pitch}`，与原有瞬时跟踪共存；删除旧的 `pen_zero_vy_cycle_drift`、`pen_zero_yaw_cycle_drift` 及旧 manager 实现。所有速度命令（包括非零命令）都启用，不使用 moving 或接触门控。

`mean_error = integral(actual - historical_reference) / actual_window_duration`

`cost = Huber(abs(mean_error) / error_scale)`，乘负权重，由 RewardManager 统一乘控制 dt；函数内部不重复缩放 dt。每个物理量独立保存最近完整一个实际步态相位周期，最旧区间按相位占比截取。窗口不满、无效观测、reset 或采样中断时不连接旧历史；同一步重复读取不累计。

| 分量 | 权重 | 归一化尺度 | 实际量与参考 |
|---|---:|---:|---|
| vx | −0.2 | 0.2 m/s | 机体系前向速度减当时的 vx 命令 |
| vy | −0.2 | 0.2 m/s | 机体系侧向速度减当时的 vy 命令 |
| yaw | −0.2 | 0.3 rad/s | 连续展开的航向增量减当时 yaw 速度命令的积分，再除窗口时长 |
| height | −0.1 | 0.02 m | 沿局部地面法向的机身高度减当时高度命令 |
| roll | −0.1 | 3° | 地面法向在机体系中的有符号 roll 倾角 |
| pitch | −0.1 | 3° | 地面法向在机体系中的有符号 pitch 倾角 |

当前任务没有独立 roll/pitch 命令输入，因此这两项的目标为局部地面法向对齐，并不增加网络输入。地面法向转到机体系后，roll=`atan2(n_y,n_z)`，pitch=`atan2(-n_x,sqrt(n_y²+n_z²))`；二者与瞬时地形姿态项共享对齐目标。无效地面扫描清空相应窗口，不使用世界水平面替代。

上述是**先平均有符号误差，再惩罚**。周期内正负误差可以抵消，所以保留瞬时项。尺度是归一化参数，不是死区或硬阈值；权重是初始试验值。

日志新增 `cycle_mean_<component>/abs_error_<unit>`、`rms_error_<unit>`、`valid_fraction`、`samples`、`duration_s`。记录完整窗口的平均误差，不能与 v6 的积分位移/转角数值直接比较。

## 固定零轮速参考

Foot 的 `WheelVelocityPIActionCfg.fixed_zero_target=True`，动作处理后轮目标严格清零，执行时再次使用零参考。MuJoCo Foot 控制器同样忽略策略轮输出。双专家 FSM 的 Foot 分支轮输出清零，过渡阶段从 Wheel 目标平滑过渡到零。

PI 增益、力矩限制、抗积分饱和、实际轮速惩罚 `−0.5` 均保留。删除 `pen_wheel_target_zero`，因为实际执行目标已经恒为零。参考为零不意味着瞬时实际轮速严格为零，仍受控制器动态和力矩限幅影响。

原始轮输出诊断改名为 `wheel_raw_output_over_limit_rate` / `wheel_raw_output_same_sign_over_limit_rate`，只表示网络原始输出超过旧诊断阈值，不能解读为执行器目标或实际轮速。

保留 8 维策略输出和原始动作历史以兼容 checkpoint 的张量形状，Foot 最后两维不再影响执行器；原动作变化/平滑成本仍按原始动作计算。Wheel 模式保持策略轮速控制。旧模型的张量可以加载，但控制语义已改变，不能把换控制器后的行为视为新奖励训练结果。

## 本次没有修改

瞬时 XY/yaw 权重仍为 8/4，核宽不变；抬脚净空、支撑和双脚腾空奖励、地形与命令课程、网络结构均保持。支撑缺失与及时双脚腾空惩罚属于前面讨论的另一个方案，本次未实施。

## 验证

CPU 回归覆盖：完整周期门槛、可变步频、窗口边界插值、符号抵消、历史非零命令、yaw 跨 ±π、reset、无效观测、平均高度与姿态、重复读取，以及 Isaac/MuJoCo PI 力矩一致性、任意 Foot 轮输出均被忽略、Wheel 控制与 FSM 过渡。

执行命令：

```bash
OMP_NUM_THREADS=1 /home/tuchuaan/miniconda3/envs/UDMMR/bin/python -m unittest discover -s tests -p 'test_*.py'
```

实现时完整 Isaac Lab/PhysX smoke 脚本已更新新断言，127 项 CPU 回归通过；之后已有用户启动的 v7 训练。CPU 回归不能证明策略已学会稳定步态。

## TensorBoard 短标题（2026-09-18）

Runner 写入日志时缩短显示标签，不修改环境内部指标名、奖励或统计值：

- `Metrics/base_velocity/cycle_mean_height/abs_error_m` → `CycleMean/height_abs_m`
- `Metrics/base_velocity/cycle_mean_roll/rms_error_rad` → `CycleMean/roll_rms_rad`
- 周期项 `samples/valid_fraction/duration_s` → `<分量>_n/valid/period_s`
- 分模式速度指标 → `Tracking/straight_xy_rmse` 等
- 分高度抬脚指标 → `Swing/low_clearance_m` 等
- 原始轮输出指标 → `WheelRaw/...`
- `Episode/Episode_Reward/...` → `Reward/...`
- `Episode/Episode_Termination/...` → `Termination/...`

只对重新启动训练或恢复训练后写出的日志生效；已运行进程不会热更新。历史事件文件保持原样，恢复到同一日志目录可能同时看到新旧标签。TensorBoard 过滤词应使用上述新分组名。
