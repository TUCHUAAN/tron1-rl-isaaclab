# WF 双专家方案概览

> **当前配置：v12，文档更新于 2026-09-21。** 当前表格与执行器说明已同步；明确标注历史的 v5/v6/v7 数据仅用于版本对照。

## 目标

为 `WF_TRON1A` 分别训练两个策略，由外部 FSM 负责切换：

| 专家 | 任务 | 作用 |
|---|---|---|
| Wheel Height Pretrain（可选） | `Isaac-Limx-WF-Wheel-Height-Pretrain-v0` | 仅用于平地诊断，不属于正式训练流程 |
| Wheel | `Isaac-Limx-WF-Wheel-Mode-v0` | 双轮持续接触，完成滚动、转向和高度调节 |
| Foot | `Isaac-Limx-WF-Foot-AllTerrain-v0` | 小范围主动轮制动配合步态处理复杂地形 |
| Dual Play | `Isaac-Limx-WF-Dual-Mode-Play-v0` | 同时加载两个 checkpoint 并平滑切换 |

## 统一接口

- 动作：8 维 = 6 个腿关节位置 + 2 个轮关节速度。
- 命令：`[vx, vy, wz, body_height_norm]`；高度在命令生成器和奖励中保持米制，仅网络输入归一化到 `[0, 1]`。
- 两个专家保持相同 observation/action schema。
- Foot 的最后两维保留网络形状，训练与 MuJoCo 均固定零轮速参考；FSM Foot 分支清零，过渡到 Wheel 时平滑混合目标。

## 专家差异

| 项目 | Wheel | Foot |
|---|---|---|
| 速度范围 | `vx ±1.5`，`vy=0`，`wz ±1.0` | `vx ±0.8`，`vy ±0.4`，`wz ±0.8` |
| 高度范围 | `0.65～0.85 m` | `0.65～0.85 m` |
| 地形 | 平地、坡面、粗糙、低上/下楼梯 | 平地、坡面、粗糙、上下楼梯 |
| 轮动作 | 正常生效 | 固定零参考，PI 制动 |
| 重点约束 | 轮地接触、滚动、固定水平构型 | 步态时序、地形自适应摆腿、实际轮速和轮目标趋零 |

Wheel 在近零转向命令下软约束左右轮目标一致，在全部速度命令近零时软约束双轮目标趋零；轮目标保留 `0.15 rad/s` 死区以允许平衡修正。独立的零命令实际 `wz²` 惩罚已经删除，由 `stand_still=-7.0` 覆盖静止约束。高度和静止项使用轮地支撑软门控，腾空阶段不施加这些状态约束。正式训练采用单阶段全地形 curriculum，从头同时学习高度、静止稳定和地形运动。

Wheel 速度命令当前按站立/直行/纯转向/混合 `25%/30%/10%/35%` 互斥采样，平移覆盖为 `65%` 并保留显式纯 `wz`。实际 base 平面加速度惩罚权重归零；偏航加速度的 Charbonnier 鲁棒核小幅加强到 `-0.025`，保留近目标跟踪门控和联合 `C_any`。Wheel 轮速执行器已由隐式纯 P 改为带 anti-windup 的显式 PI：训练每次 reset 采样 `Kp∈[0.5,4.0]`、`Ki/Kp∈[0,0.25] 1/s`。

## 当前 Foot 命令与步态

2026-09-15 Foot 补充：速度命令按原地踏步/纯前后/纯侧移/纯转向/混合 `15/25/10/20/30%` 互斥采样，保留原速度范围；零命令仍迈步。相位改为累积频率，训练奖励与各组观测共享，MuJoCo 按训练快照的 `continuous_phase` 标志同步。新增分模式跟踪、分机身高度净空与腿关节软限位、轮目标裁剪比例日志；抬脚目标公式、已确认的奖励权重和高度范围保持，不引入 yaw 积分。

当前为 **v12**：Foot/Wheel 的 policy 和 critic 均追加六维零命令保持误差/启用标记，与奖励共用同一 tracker；MuJoCo 同步支持。policy 161D、Actor 168D，history/Encoder 仍 340D。Foot 保留 v11 的 2 cm 中段最低净空成本 −1.0、漏迈 −0.2/次及此前跟踪权重。**从头训练，不微调**；详见 [v12 观测与训练说明](wf_hold_observation_v12.md)。

## 当前 Foot 训练入口与日志目录

- **从头新训练**：`--resume false`，15000 iterations、每 500 次保存，默认 PPO `1e-3 adaptive`、Encoder `1e-3`；日志为 `logs/rsl_rl/wf_tron_1a_foot_all_terrain/<时间戳>_foot_hold_observation_v12/`。
- **历史参考，当前不采用：h08 → v10 微调**：显式加载 `2026-09-11_14-32-20_foot_width_clearance_h08_v1/model_20000.pt`，PPO/Encoder 均固定 `1e-4`，追加 1000 iterations、每 100 次保存；独立日志为 `logs/foot_finetune_v10/<时间戳>_h08_to_zero_hold_swing_v10/`。
- 当前 v12 使用新观测结构从头训练；上述 h08 微调仅为历史记录，旧 Actor 不可直接用于新环境。模型、TensorBoard 和 `params/` 随新 run 保存。
- 完整命令见[训练说明](wf_dual_mode_training_process.md#foot-从-h08-模型初始化)，差异见[训练对比](与原有WF及PF训练对比.md#102-当前两个专家分别训练)。v7 已有训练 run；用户反馈步态时序仍有问题，尚未验收达标。

## Checkpoint 与部署兼容性

| 模型/文件 | 当前代码下的处理 |
|---|---|
| 原有 WF/PF Blind-Flat checkpoint | Actor 输入/输出维度不同，不能直接加载到当前 Wheel/Foot 任务 |
| 显式 PI 之前的 Wheel checkpoint | 网络形状可能相同，但执行器动力学已改变，只适合作历史对照，不应 resume optimizer 或作为当前 PI 方案结论 |
| 轮动作硬锁零时期的 Foot checkpoint | 最后两维从未按有效轮速目标训练，不能仅凭张量形状兼容推断符合当前 v10 的奖励和执行器行为 |
| 缺少 `params/env.yaml` 的新 Foot checkpoint | MuJoCo 会回退到默认 gait，但无法确认训练是否使用连续相位；正式归档必须保留该文件 |

MuJoCo 侧已同步显式 PI、Foot 轮目标裁剪、连续 gait 时钟、机身速度坐标系、状态缓存刷新和扫描未命中编码。启动日志会打印最终 `gait_clock`、`gait=(...)`、`wheel_kp` 与 `wheel_ki`，部署验收应保存这段日志。

## 模式切换

控制状态：

```text
wheel → wheel_to_foot → foot → foot_to_wheel → wheel
```

切换期间：

- 速度命令归零；
- 两个专家动作渐变；
- Wheel 与 Foot 的全部 8 维动作均连续渐变；
- Foot→Wheel 需要低机身速度、低轮速、双轮接触和直立状态。

当前 `auto` 模式按 height-scan relief 请求模式，后续计划改为基于双专家实测结果的地形可行性表。

## 变更与实验记录（更新至 2026-09-16）

- 2026-08-12 单阶段 Wheel 训练完成，但实际高度 RMS 误差约 `7.8 cm`，MuJoCo 端点命令响应不足。
- 2026-08-13 的加强版单阶段训练高度 `MAE 7.26 cm / RMSE 11.78 cm`，base contact 终止率 `57.5%`；根因是未触地阶段强高度负奖励和过紧静止约束。
- 已将高度范围改为 `0.65～0.85 m`、端点比例降到 `20%`，为高度/静止/实际 yaw 加入支撑门控，并加入 `base_contact` 终止惩罚。
- `2026-08-15_00-26-20_wheel_single_stage_h065_085_supportgate` 因误用持久化的 `is_terminated_term`，复位后仍逐步扣除终止惩罚，约 4020 iterations 的结果无效；现已改为当前步 `mdp.is_terminated`，必须重新从头训练。
- 修复后的 20k 训练在 `model_6500` 达到 `92.2%` timeout，最终 `model_20000` 降至 `76.6%`，证明最后一个 checkpoint 不是最佳模型；已增加基于高度/速度/地形门槛和 base contact 的选择工具。
- 当前 Wheel 高度权重为 `-60`、倒地权重 `-500`、零命令轮目标 `-0.10`；竖直速度惩罚在高度命令变化时平滑降至 `20%`，接近目标后恢复。等待从头进行新的单阶段全地形训练。`Wheel-Height-Pretrain` 仅作为可选诊断任务保留。
- 2026-08-29 Wheel 保持高度 L2 权重 `-60`，新增局部地形与支撑门控的高斯核/RBF 高度奖励 `+1.0`（`std=0.05 m`）；两个 Wheel 高度项均改用 `max(C_L,C_R)` 门控，任意一轮有效支撑时即完整生效；线速度/Yaw 跟踪调为 `+3.5/+1.25`，地形姿态调为 `-10`，动作变化/二阶平滑调为 `-0.15/-0.08`。
- 2026-08-30 针对 Wheel 的速度稳态误差和 yaw 超调，将 XY/Yaw 指数跟踪的 `std²` 均收紧为 `0.12`，Yaw 权重调为 `+1.5`，`stand_still` 调为 `-7.0`；Foot 配置不变。
- 2026-08-31 Wheel 的 XY/Yaw 跟踪由 `C_all=min(C_L,C_R)` 改为联合支撑 `C_any=max(C_L,C_R)`，`stand_still` 也改为任意轮地形过滤力支撑即完整生效；权重与网络输入不变。TensorBoard 新增左右轮力/几何/联合置信度、`C_all/C_any`、支撑比例与门控前后跟踪值。
- 2026-08-31 Wheel 地形课程从 `10` 级加密为 `12` 级，实际上坡比例提高到 `25%`；修正了从中心平台向外运动时 Pyramid/InvertedPyramid 的上下坡和上下楼梯方向，真正的上楼梯范围为 `0.005～0.04 m`。升级除了 `4 m` 位移还要求 XY 跟踪 `>=0.55`、`C_any>=0.75` 且不倒地；倒地或移动期跟踪 `<0.25` 会降级。
- 2026-09-01 Wheel 命令改为站立/直行/纯转向/混合 `25%/20%/20%/35%`；新增 `pen_base_lin_acc_xy=-0.05`（饱和尺度 `3.0 m/s²`）和 `pen_base_yaw_acc=-0.05`（饱和尺度 `4.0 rad/s²`），均用 `std=0.30` 跟踪门控及联合 `C_any`，reset 后前两步为零。四类命令比例和两个新增奖励均进入 TensorBoard；Foot 与网络 schema 不变。
- 2026-09-02 根据上一轮 14k 后坡面能力退化的日志，命令比例调整为 `25/30/10/35%`；`pen_base_lin_acc_xy` 降为 `-0.02`，`pen_base_yaw_acc` 降为 `-0.02` 并改用不会完全饱和的 Charbonnier 核。TensorBoard 新增直行、纯转向和混合模式的条件误差及纯转向 `wz` 加速度 RMS；Foot 与网络 schema 不变。
- 2026-09-06 Foot 的 `pen_feet_distance` 按原有 PF 改为 `0.115～1.0 m`，解除原有 WF `0.32～0.35 m` 上限对正常前后步幅的限制；gait contact duration 同样按 PF 固定为 `0.5`，避免计划双摆动区间与 `pen_all_wheels_air_time` 冲突。速度 command 仍从连续实数范围均匀采样，但每 `3～15 s` 直接重采样一次，没有时间平滑。
- 2026-09-07 新 Foot 日志表明 `pen_feet_distance` 已接近零，但 `undesired_contacts` 从约 500 iterations 起持续恶化并在 4k 后稳定接近 `-1`，策略已把一条小腿作为长期支撑。Foot 因此新增 `pen_knee_contact_force=-2.0`：超过 `10 N` 的小腿力峰值按 `min((F-10)/100,3)^2` 逐腿惩罚，封顶防止冲击尖峰造成数值不稳定；任一小腿接触力 `>20 N` 连续 `0.08 s` 时终止 episode。原有二值非法接触惩罚继续保留，用于覆盖其他非轮部件及轻微接触。
- 2026-09-08 针对 S02 部分 checkpoint 反复爬坡后轮速误差无法恢复，Wheel 训练改用 `200 Hz` 显式 PI 轮速力矩控制。默认 `Kp=2.0`，每次 reset 采样其 `0.25～2.0` 倍，`Ki/Kp` 采样 `0～0.25 1/s`；力矩限幅 `±80 N·m`并带条件积分 anti-windup。同时 `pen_base_lin_acc_xy: -0.02→0.0`，`pen_base_yaw_acc: -0.02→-0.025`。MuJoCo 默认 `Kp/Ki=2.0/0.5`，可用 `--wheel-kp/--wheel-ki` 调整。该版执行器动力学已变，必须从头训练，不能 resume S02 optimizer。
- 2026-09-09 Foot 的硬零轮目标改为可学习软锁轮：最后两维轮目标裁剪为 `±1 rad/s` 后进入同一显式 PI，训练随机化 `Kp∈[0.5,4.0]`、`Ki/Kp∈[0,0.25] 1/s`。删除 `pen_joint_vel_wheel_l2=-0.10`，改为接地与腾空时均生效的 Huber 实际轮速惩罚 `pen_wheel_actual_speed=-0.02`，以及带 `0.1 rad/s` 死区的目标惩罚 `pen_wheel_target_zero=-0.002`。FSM 和 MuJoCo 同步执行 Foot 轮目标；旧 Foot checkpoint 的最后两维从未受动力学训练，不能评价这一版。
- 2026-08-28 Wheel 奖励去重后已通过真实 Isaac Lab CPU 1-env / 3-step 冒烟，25 个奖励项均为有限值：删除 `pen_zero_command_yaw_rate` 和 `pen_wheel_air_time`，`stand_still` 调为 `-6.0`，滚动误差改为逐轮考虑实际 `wz`；支撑滑移保留。MuJoCo 无窗口冒烟确认会优先加载稳定性选择文件指定的 checkpoint。
- 2026-08-29 Foot 删除 `stand_still`、保留 `rew_leg_symmetry=+0.5`，并将力矩、加速度、动作变化/平滑、功率和腿关节速度正则的数值权重对齐原有 PF；局部地形姿态调为 `-10.0`。需要注意，当时 Foot 的力矩、加速度和功率项会统计 8 个关节，动作项统计 8 维输出，而 PF 只有 6 个可动腿关节/6 维动作，所以只是权重对齐。高度改用与 Wheel 相同的局部平面法向定义和 `max(C_L,C_R)` 门控，保留 L2 `-30.0` 并新增指数正奖励 `+1.0`。步态三项采用 PF 的目的和权重：保留 `GaitReward=+1.0`，以逐轮局部地形版 `pen_feet_regulation=-0.1` 和 `foot_landing_vel=-0.5` 替换固定摆高和接触后冲击；第四维摆高输入固定为零。Foot 真实 Isaac Lab CPU 1-env 冒烟通过。
- 历史 Foot 训练曾生成 `2026-08-17_21-27-09_foot_local_terrain_orientation_v1/model_20000.pt`，当时 MuJoCo 在执行器层硬锁左右轮目标速度；该结果只代表旧方案。
- Dual Play、统一网络接口和新的 Foot 软锁轮动作路由已通过既有冒烟验证。

- 2026-09-10 修复高度命令均匀采样的张量写回：旧高级索引写法使 `80%` 非端点样本保留上一次目标，当前实现会真正从 `0.65～0.85 m` 全区间采样；已有 checkpoint 的历史训练分布不变。同日 MuJoCo 刷新运动学缓存，并将扫描未命中编码统一为训练侧裁剪后的 `0`。

- 2026-09-11 Foot 几何奖励更新：`pen_feet_distance` 从 XY 总距离改为航向系有符号横向宽度，目标区间 `0.30～0.38 m`、Huber 尺度 `0.05 m`、权重 `-1.0`，保持正常前后迈步自由度；`pen_feet_regulation` 高度衰减尺度改为 `0.05 m`；新增移动命令与对侧支撑门控的 `rew_swing_clearance=+0.5`，局部地面相对目标峰值 `0.05 m`、std `0.025 m`。gait 第四维仍为零，速度跟踪、膝保护与轮控制不变。CPU 数值/包装函数检查通过；运动学验证目标在三档机身高度可达，尚未验证重新训练后的策略表现。

- 2026-09-11 按进一步提高抬脚高度的要求，将 Foot `rew_swing_clearance.peak_height` 上限从 `0.05 m` 提到 `0.08 m`。最低姿态的膝限位检查表明不宜统一抬到 `8 cm`，因此高度命令 `0.65～0.70 m` 对应峰值 `5～8 cm`，更高时保持 `8 cm`；奖励权重 `+0.5`、std `0.025 m` 和 `pen_feet_regulation.height_scale=0.05 m` 保持。这是局部地面相对的摆动净空目标，需要新训练验证，尚未加入楼梯路径前瞻。

- 2026-09-14 Foot 按确认方案启用零命令原地踏步的净空成本，替换固定高度指数项为实时全图极差 `clip(z_max-z_min,0.02,0.10)` 与非对称 Huber 容差带；快升慢降滤波时间常数 `0.04/0.15 s`，净空成本权重 `-0.5`。XY/yaw 跟踪权重 `4/2`，核宽度统一为 `sqrt(0.12)`，新增 yaw 误差 Huber `-0.2`。新增 XY/yaw Charbonnier 加速度项 `-0.005/-0.025`，Foot 跟踪门控下限 `0.1`，XY 使用世界系差分。Wheel 的默认计算与权重保持；观测维度、控制器、横向宽度、膝保护和旧 `pen_feet_regulation` 保持。yaw 积分仍处于方案讨论阶段，未激活。

- 2026-09-15 Foot 改为五类互斥速度命令与连续 gait phase，新增分模式、分高度和轮目标裁剪诊断。对应 `foot_terrain_phase_modes_v3/model_20000.pt` 已完成训练，但确定性 MuJoCo 复测显示策略退化为双轮长期接地、依靠小轮速滚动/差速转向：末 500 iterations 的直行 XY RMSE 约 `0.380 m/s`，平均摆动净空仅约 `2.8～3.2 mm`，地形等级约 `0.002`，轮目标裁剪率约 `82.9%`。因此该 checkpoint 不应视为合格 Foot 策略；完整证据见 `agent_docs/temp/foot_failure_2026-09-15/report.md`。当时下一版奖励建议尚未写入正式训练代码；2026-09-16 的实现见下条。

- 2026-09-16 v4 历史奖励修正：按用户纠正将 `pen_base_lin_acc_xy` 置 `0.0`，保留 yaw 加速度 `-0.025`；实际轮速 Huber 权重 `-0.02→-0.5`，弱轮目标成本 `-0.002` 与 PI 保持。摆动接触力改为 `100 N` 尺度 Huber 核、内部系数 `-4.0`，PF 默认指数核保持；净空不足系数 `1→4`、过高系数仍为 `0.1`。XY 瞬时 Huber 跟踪误差 `-0.5`、尺度 `0.4 m/s` 已加入，无命令积分。当时全套 CPU 单元测试 103 项通过；随后 v4 训练与 MuJoCo 复测仍为双轮长期贴地，详见诊断报告。

- 2026-09-16 v5：对照能迈腿的 h08 模型后，恢复较宽 XY/yaw 跟踪核 `std²=0.20/0.25`、保留权重 `4/2`，轮目标成本改 `-0.01`。抬脚恢复指数正奖励 `+0.5`、std `0.025 m`，地形峰值 `4～10 cm`，关闭 `pen_swing_clearance`，诊断改读 `rew_swing_clearance`。104 项 CPU 回归与训练环境 21 项相关测试通过；尚未新训练。h08 平地前进 18 s 有 61 次左右交替离地、平均峰值 14.7 mm、实际 vx 0.219 m/s；原地和纯左转仍不迈步。h08 网络兼容，可作为微调起点。

- 2026-09-16 微调入口：新增显式日志根目录与 PPO/Encoder 学习率、fixed/adaptive 调度覆盖，fixed 禁止线性退火；修复原 `--experiment_name` 未应用的问题。h08→v5 微调命令使用 `logs/foot_finetune/` 独立目录、两个优化器固定 `1e-4`、追加 1000 iterations、每 100 保存。从头训练默认行为保持；本次未启动训练。

- 2026-09-16 v6：用户确认将 Foot XY/yaw/净空奖励翻倍为 8/4/1，关闭 yaw 加速度；新增独立门控的完整周期 vy/航向积分 Huber 成本，配置及测试见 [v6 说明](wf_foot_cycle_drift_v6.md)。v5 微调退化已保存对照报告；本版未启动训练。

## 文档导航

- 训练与运行：[wf_dual_mode_training_process.md](wf_dual_mode_training_process.md)
- 当前奖励表：[wf_dual_mode_reward_penalty_table.md](wf_dual_mode_reward_penalty_table.md)
- 原方案对比：[与原有WF及PF奖励对比.md](与原有WF及PF奖励对比.md)、[与原有WF及PF训练对比.md](与原有WF及PF训练对比.md)、[与原有WF及PF网络输入对比.md](与原有WF及PF网络输入对比.md)
- 2026-08-05 Wheel 合规奖励历史方案：`wf_new_mode_judgment_modification_plan.md`
