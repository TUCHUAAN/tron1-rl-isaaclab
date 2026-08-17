# 进度日志

## 会话：2026-08-03——模态判定与训练条件重构

- **状态：** in_progress
- 已恢复并读取既有 `task_plan.md`、`findings.md`、`progress.md`。
- 已把用户提出的腾空态、固定姿态轮行态、步态足行态定义加入阶段10。
- 已核对现有 reward、termination、reset、动作和接触传感器配置。
- 已形成 `agent_docs/human/wf_mode_reward_reset_redesign.md`，包含四状态判定、轮系几何替代表达、两专家奖惩、严格终止、两套 reset 分布、课程和验收指标。
- 下一步待用户确认姿态目标后实施：固定构型随坡面倾斜，还是车体保持世界水平并允许腿部不对称调平。
- 根据用户修订，已删除 Wheel 环境的 `both_wheels_airborne` termination，并同步更新全部双模态训练/奖励文档；AIRBORNE 改为可恢复状态。
- 用户要求按最新奖励惩罚表修改代码；本轮范围限定为 reward 函数、环境权重/参数、单元测试和冒烟验证，IK/reset 标定另阶段处理。
- 已发现现有 air-time 惩罚使用 `last_air_time`，本轮改为当前 air time；现有接触项只看瞬时力，本轮改用 4 帧历史置信度。
- 已核对 IsaacLab 2.2.1：可使用 `current_air_time`、`compute_first_contact(step_dt)` 和 class reward reset；速度跟踪需本地实现接触门控 wrapper。
- 已新增纯 PyTorch `mdp/reward_math.py`：4 帧接触置信度、地面平面拟合、轮系几何/构型包络、地形姿态、接触点滑移和落地冲击数学函数。
- 已在 `mdp/rewards.py` 接入环境 reward wrapper，并把 air-time 改为 `current_air_time`；增加 Wheel 接触门控速度/yaw、地形相对高度、几何/构型、相对足端运动、落地冲击、支撑滑移，Foot 增加同时腾空、冲击和滑移项。
- 已更新 Wheel/Foot 环境权重；静态语法和 `git diff --check` 通过，IsaacLab 2.2.1 字段名核对通过。
- 已新增 `tests/test_reward_math.py`，7 个纯张量测试全部通过：接触历史、坡面拟合、轮系几何、构型包络、无滑滚动、落地冲击和地形姿态。
- 宿主环境级冒烟通过：Wheel 28 个 reward、Foot 24 个 reward，各执行 3 个零动作步且全部为有限值；确认两个任务均无腾空 termination。
- Wheel 冒烟接触门控奖励为 0、接触损失为 `-8`，说明 reward 行为符合设计但 reset 仍无有效轮地接触；未将 reset 问题误判为 reward 失败。
- 已同步奖励惩罚表的实际权重、公式和运行时数量：Wheel 28 项、Foot 24 项；奖励代码阶段完成，reset/模态估计仍按阶段10继续。

## 会话：2026-08-02

### 阶段 1：现状分析与接口设计
- **状态：** complete
- **开始时间：** 2026-08-02
- 执行的操作：
  - 检查已有 PF/SF/WF 楼梯环境类和地形生成配置。
  - 确认楼梯和斜坡尚未注册为 Gym 任务。
  - 检查固定机身高度奖励、gait swing_height 命令和 Runner 命令拼接。
  - 起草任务矩阵、高度命令语义、奖励和验收方案。
  - 明确高度命令进入 commands 观测组，由现有 Runner 拼接给 actor/critic。
- 创建/修改的文件：
  - task_plan.md
  - findings.md
  - progress.md

### 阶段 2：兼容基线与任务注册
- **状态：** pending

## 测试结果
| 测试 | 输入 | 预期结果 | 实际结果 | 状态 |
|------|------|---------|---------|------|
| 方案阶段源码盘点 | 楼梯、斜坡、高度相关源码 | 明确已有能力和缺口 | 已确认环境类存在、任务未注册、高度为固定目标 | 通过 |
| 当前训练基线（前序测试） | PF Flat, 2 env, 1 iteration | 完成训练 | 仿真创建成功，受 Isaac Lab API 不兼容阻塞 | 阻塞已记录 |
| reward 数学单测 | `conda run -n isaaclab4.5 python -m unittest tests.test_reward_math -v` | 全部通过 | 7/7 通过 | 通过 |
| Wheel reward 环境冒烟 | 1 env、CPU PhysX、3 个零动作步 | 28 项均有限、无腾空终止 | PASS | 通过 |
| Foot reward 环境冒烟 | 1 env、CPU PhysX、3 个零动作步 | 24 项均有限、无腾空终止 | PASS | 通过 |

## 错误日志
| 时间戳 | 错误 | 尝试次数 | 解决方案 |
|--------|------|---------|---------|
| 2026-07-31 | dump_pickle 在当前 Isaac Lab 中不存在 | 1 | 实施前固定兼容版本或迁移 |
| 2026-07-31 | Runner 期望旧观测二元组，当前返回 TensorDict | 1 | 实施前固定兼容版本或迁移 |

## 五问重启检查
| 问题 | 答案 |
|------|------|
| 我在哪里？ | 方案已完成，等待用户确认 |
| 我要去哪里？ | 用户确认方案后进入兼容基线、任务注册和高度命令实现 |
| 目标是什么？ | 增加楼梯/纯斜坡任务并支持运行时机身高度命令 |
| 我学到了什么？ | 见 findings.md |
| 我做了什么？ | 已形成初版持久化实施方案，尚未修改业务代码 |

### 需求修订：WF 双模态
- **状态：** complete
- **日期：** 2026-08-02
- 执行的操作：
  - 将范围从三类机器人收缩为仅 WF_TRON1A。
  - 核对 WF 当前 6 维腿位置动作和 2 维轮速度动作。
  - 识别当前 wheel velocity 惩罚与轮态目标冲突。
  - 设计 Wheel Expert、Foot Expert 和外部模式切换 FSM。
  - 将足态轮速 0 定义为动作层硬约束，而非仅奖励约束。
  - 记录轮地接触目标与下楼梯之间的物理约束；后续已修订为腾空不终止。
- 修改的文件：
  - task_plan.md（替换为 WF 双模态实施计划）
  - findings.md（替换为 WF 双专家设计发现）
  - progress.md


### 实施阶段启动
- **状态：** in_progress
- **日期：** 2026-08-02
- 执行计划：
  - 实现共享高度命令和模式专用动作接口。
  - 实现 Wheel/Foot 环境、奖励、地形与 Gym 注册。
  - 增加双策略 FSM/Play 基础设施。
  - 在 agent_docs/human 下记录简要修改说明。
- 已实现 BodyHeightCommand：随机目标、变化限速、Play 外部 target 覆盖接口。
- 已新增模式奖励/终止函数：相对地面高度跟踪、轮接触损失、双轮腾空、轮空中时间、轮竖直速度、滚动速度一致性、足态摆动高度。
- 已新增 Wheel Mode 连续滚动地形课程和 Foot All-Terrain 全地形课程。
- 已新增 Wheel/Foot/Dual Play 环境配置；两专家使用相同观测、命令和 8 维动作 schema，足态轮速 action 在环境层硬 mask 为 0。
- 已新增两个独立 PPO Runner 配置及 5 个 Gym 任务注册：Wheel train/play、Foot train/play、Dual play。
- 已让自带 OnPolicyRunner 同时兼容 Isaac Lab 旧版 `(policy_obs, extras)` 与新版 TensorDict 观测接口。
- 已为 train.py 增加 dump_pickle 兼容 fallback。
- 已实现确定性 WheelfootModeFSM：显式模式请求、安全条件、动作渐变、Foot 状态轮速双重硬 mask、地形 relief 自动请求接口。
- 已让单策略 play.py 兼容旧/新 Isaac Lab 观测 API。
- 已新增 scripts/rsl_rl/play_wf_dual.py：加载两个 checkpoint、共享观测推理、手动/自动模式请求、FSM 安全渐变和 Foot 轮速强制归零。

## 错误日志补充
| 2026-08-03 | 在普通 conda Python 中导入 `isaaclab.sensors` 以 inspect 源码，因未启动 SimulationApp 缺少 `carb` | 1 | 不重复导入；直接读取 `/home/tuchuaan/IsaacLab-2.2.1/source/isaaclab` 源文件 |
| 2026-08-03 | Wheel CPU 冒烟在 CommandManager 创建调试箭头时访问远程 USD，沙箱无网络导致 FileNotFoundError | 1 | 将 `base_velocity.debug_vis` 设为仅 Play 模式启用；headless train 离线创建不再依赖 marker USD |
| 2026-08-03 | 第二次 Wheel CPU 冒烟在 Warp 编译 raycast kernel 时无法写 `~/.cache/warp` | 1 | 测试命令设置 `WARP_CACHE_PATH=/tmp/codex-warp-cache` 与 `MPLCONFIGDIR=/tmp/codex-mpl-cache`，不改产品代码 |
| 2026-08-03 | 宿主环境冒烟解析全部新增 RewardTerm 后关闭，但 `SimulationApp.close()` 使原异常未显示 | 1 | 测试脚本在关闭 Kit 前显式打印 traceback 和阶段标记，再做定向诊断 |
| 2026-08-03 | 一次奖励表收尾补丁因上下文块未匹配而未应用 | 1 | 读取精确行号后拆成小上下文补丁，文档已正确更新 |
| 2026-08-02 | 直接导入 bipedal_locomotion 测试纯 FSM 时触发 Isaac Lab/pxr 初始化，未启动 AppLauncher | 1 | 改用 importlib 直接加载纯 Python 模块；完整包导入留给 Isaac Sim 冒烟测试 |

| 2026-08-02 | 当前 PyTorch 无 torch.nanmax/nanmin | 1 | 改为 finite mask 配合 max/min，兼容当前及旧版 PyTorch |
| 2026-08-02 | Wheel 冒烟测试中 mdp 未导出 UniformBodyHeightCommandCfg | 1 | 检查 commands/__init__.py 导出并修正 |
| 2026-08-02 | Wheel 冒烟测试完成配置解析和地形生成，但 PhysX 在 GPU0 分配 640 MiB contact-pair buffer 时 OOM | 1 | nvidia-smi 显示 GPU0 仍有 11.6 GiB；下一次隔离 CUDA_VISIBLE_DEVICES=0，避免多 GPU/P2P 上下文干扰 |
| 2026-08-02 | CUDA_VISIBLE_DEVICES 隔离后 Omniverse 将 3080 判为 CUDA bad state，仍由图形 GPU/映射异常导致 PhysX OOM | 2 | 不再重复 GPU 方案；改用 1 env CPU PhysX 完成配置与训练链路验证 |
| 2026-08-02 | CPU PhysX 环境成功完成场景、命令、动作、观测、奖励初始化，但自定义 observation 用全局 CUDA 可用性选设备，造成 CPU/cuda:1 混合 | 1 | 将自定义 observation 改为使用 asset/env 实际 device，修复 CPU 与多 GPU 正确性 |
| 2026-08-02 | Wheel 环境已创建、Runner 已初始化，但 encoder 输入仍假设 history_dim = history_len × policy_obs_dim；新增 height scan 只在 policy 中导致 340 vs 1550 | 1 | Runner 改为直接读取 obsHistory.flatten 后的实际维度，解除历史组必须等于 policy 组的旧假设 |
| 2026-08-02 | Wheel rollout 首步触发旧版自定义 push event，访问已移除的 Articulation._external_force_b | 1 | 改用公开 set_external_force_and_torque API，保持旧/新版本兼容 |
- Wheel Mode CPU 冒烟训练通过：1 env、1 iteration，生成 model_0.pt 和 model_1.pt；新地形、命令、奖励、Runner TensorDict 兼容链路可运行。
- Foot AllTerrain CPU 冒烟训练通过：1 env、1 iteration，生成 model_0.pt 和 model_1.pt；保存配置确认 body_height 范围 0.65～0.82 m。
| 2026-08-02 | Dual Play 测试 shell 在同一前缀赋值时先展开 $WHEEL_CKPT/$FOOT_CKPT，参数变成空字符串 | 1 | 改为在参数中直接使用命令替换后的实际路径；非项目代码问题 |
- 已在 agent_docs/human/wf_dual_mode_changes.md 写入简要修改说明、任务/命令、验证结果和限制。
- 新双模态任务暂时关闭 stochastic push，避免未收敛阶段干扰并消除旧 API 高频警告；后续课程阶段再恢复。
- Dual Mode Play 使用两个冒烟 checkpoint 成功运行 10 步。
- 修正 FSM 过渡潜在死锁：Wheel→Foot 在稳定前也会逐步衰减轮驱动；Dual Play 在过渡状态强制速度命令为零，稳定后完成切换。
| 2026-08-02 | FSM 新测试按 5 次 update 预期完成 5 步过渡，但第一次 update 只进入过渡态、未累计 blend step | 1 | 测试改为进入状态后再累计 transition_steps 次；实现逻辑保持 |
- 修复扩展打包：setup.py 改用 find_packages()，确保 commands、robots、utils 等子包在非 editable 安装时也会包含。

- Wheel Mode 地形加入 10% 低难度下楼梯（0.03～0.12 m），仍不包含上楼梯；后续已取消双轮腾空硬终止，改为连续惩罚并允许恢复。


### 当前实施结果
- **状态：** 核心实现完成，训练调参待继续
- Wheel/Foot 均完成 CPU 1 env / 1 iteration checkpoint 验证。
- Dual Play 加载两个 checkpoint 运行 10 步通过。
- GPU PhysX 受主机 CUDA bad state / contact buffer OOM 影响，未完成 GPU 并行训练验证。
- 最终 Wheel 配置（含 10% 低难度下楼梯、关闭 push）完成 CPU 1 env / 0 iteration 环境启动验证；3 个命令项和 8 维动作空间解析正确。
- 已新增 agent_docs/human/wf_dual_mode_training_process.md：训练任务、PPO、网络、周期、地形课程、随机化、命令、训练顺序与验收指标。
- 已新增 agent_docs/human/wf_dual_mode_reward_penalty_table.md：Wheel/Foot 实际奖励惩罚项、权重、公式、终止条件、调参建议与已知注意点。

## 会话：2026-08-05——新版模态判断方法修改方案

- **状态：** plan_complete_implementation_pending
- 已读取外部最新版 `/home/tuchuaan/UDMMR-Extra-Docs/课题md/模态判断方法.md`，并与项目内旧版文档、现有 reward math、Wheel 环境、Dual Play FSM 和 Isaac Lab 2.2.1 contact sensor API 对照。
- 确认关键不一致：当前任一轮接触即可开放速度正奖励；接触力未过滤地面碰撞；axis-normal 仍参与 reward；构型/相对速度仍限制 z 方向；Dual Play 仍使用 `force > 1 N`；auto mode 仍按 height relief 预判。
- 已输出 `agent_docs/human/wf_new_mode_judgment_modification_plan.md`，涵盖三模态 + valid/confidence 接口、左右轮独立过滤接触、wheel-local rays、水平中性构型、时间窗/滞回、奖励调整、FSM 接入、地形四分类、测试矩阵和 M0～M5 实施顺序。
- 已在 `task_plan.md` 新增阶段 11；本轮只形成方案，没有修改训练业务代码或已有 checkpoint。

### 2026-08-05 方案修订：移除 Foot gait 时间窗

- 根据用户反馈，将方案从“三模态运行时自动识别”简化为“Wheel 合规状态 + 外部 ControlMode”。
- 删除 Wheel reward 对 gait evidence window、腿运动 RMS、接触切换周期和 gait phase 的依赖。
- 保留两项直接训练信号：全轮有效地面接触、base 系水平中性点偏差；允许 z 方向高度调节。
- Foot/Wheel 由外部 FSM 明确选择，Foot→Wheel 只需要 wheel-ready 安全条件稳定，不需要确认 FOOT_GAIT。
- 已重写 `agent_docs/human/wf_new_mode_judgment_modification_plan.md`，业务代码仍未修改。

## 会话：2026-08-05——human 文档精简

- **状态：** complete
- 已重写 `agent_docs/human/` 下全部 4 份 Markdown，按“方案概览、当前奖励、训练运行、下一版修改”分工。
- 总行数从 1313 行压缩到 521 行，删除重复背景、过期冒烟结论和已取消的 Foot gait 时间窗方案。
- 已更新 Wheel 最新训练状态：`model_20000.pt`、20,000 iterations、episode length 958、timeout 94.7%、base contact 5.3%。
- 已校验任务注册、checkpoint 路径、Markdown 换行和尾随空格，全部通过。

### 2026-08-05 方案修订：取消 Reset 有效性筛选

- 按用户决定，删除 Reset 有效性公式、无效状态拒绝和重新采样方案。
- Reset 保持当前实现，不纳入本轮 Wheel 合规判断和 reward 修改。
- 已同步更新 human 修改方案、`task_plan.md` 和 `findings.md`。

## 会话：2026-08-05——按新方案实施 Wheel 奖励

- **状态：** wheel_reward_complete_gpu_retrain_pending
- 新增左右轮独立 ground-filtered ContactSensor 和 wheel-local RayCaster。
- 新增纯张量函数：all-support confidence、轮心 clearance confidence、水平中性点 dead-zone Huber。
- Wheel 速度/yaw 奖励改为双轮有效接触门控；接触惩罚改为过滤地面力与几何联合判断。
- 新增 `pen_wheel_horizontal_neutral=-4.0`，目标 `(0, ±0.17) m`，只约束 x/y。
- 删除/禁用旧 x/z 构型、轮距、轮轴法向、轮端竖直速度、xyz 相对速度和重复水平位置项。
- 未修改 FSM、Dual Play 和 Reset，符合本轮仅训练 Wheel Expert 的范围。
- 测试：纯张量 `7/7 PASS`；Wheel CPU 环境 23 项 reward 冒烟通过；Foot 24 项回归通过；Wheel CPU 1 iteration 生成 `model_1.pt`。
- 旧 `model_20000.pt` 在新环境 CPU 运行 400 步：双轮接触置信度 >0.5 比例约 98.75%，旧 checkpoint schema 兼容。
- 修复两次验证问题：补充遗漏的 `ContactSensorCfg` import；CLI device 同步到 PPO runner 后 CPU 训练通过。

## 会话：2026-08-13——高度控制与零命令 yaw 二次修订

- **状态：** implementation_complete_single_stage_retrain_pending
- 诊断 `2026-08-12_20-23-06_wheel_hnorm_h062_085_stairs004/model_20000.pt`：末段高度 RMS 误差约 `7.8 cm`；MuJoCo 中 `0.62/0.85 m` 端点命令只产生约 `3 cm` 的稳态高度差。
- Wheel 高度奖励从 `-25` 提升到 `-100`，零命令环境比例从 `10%` 提升到 `25%`。
- 高度命令增加 `40%` 端点采样：最小/最大高度各 `20%`，其余区间均匀采样。
- 新增 episode 平均实际高度 `MAE/RMSE` 与命令 slew MAE 指标，替换易误解的旧 `height_command_error`。
- Wheel 左右轮目标对称权重提升到 `-0.5`；零命令轮目标权重提升到 `-1.0`，死区降到 `0.05 rad/s`；新增零命令实际 `wz²` 惩罚 `-2.0`。
- 新增 `Isaac-Limx-WF-Wheel-Height-Pretrain-v0` 平地诊断任务和独立日志目录；按用户决定，正式训练不分阶段，改为从零运行 `Wheel-Mode` 全地形 curriculum 20000 iterations。
- 纯张量单元测试扩展到 10 项并全部通过；完整 Isaac Sim 环境冒烟待执行。

## 会话：2026-08-15——支撑门控与倒地终止惩罚

- **状态：** implementation_complete_single_stage_retrain_pending
- 诊断 `2026-08-13_18-19-28_wheel_single_stage_h062_085_stairs004/model_20000.pt`：末100次迭代 episode length `608.5/1000`，base contact 终止率 `57.5%`，高度 `MAE 7.26 cm / RMSE 11.78 cm`。
- 高度范围统一改为 `0.65～0.85 m`，网络输入及 MuJoCo 同步归一化；端点采样从 `40%` 降为 `20%`，最小/最大各 `10%`。
- 新增地面过滤力的平均支撑置信度：双轮/单轮/腾空门控约为 `1.0/0.5/0.0`；Wheel/Foot 的高度和静止惩罚、Wheel 零命令实际 yaw 惩罚均使用该门控。
- 原配置新增 `pen_base_contact_termination=-200`，意图只惩罚 `base_contact`；其持久状态误用已按下方记录修复。
- 零命令轮目标权重从 `-1.0` 回调到 `-0.25`，死区从 `0.05` 恢复到 `0.15 rad/s`；实际 yaw 惩罚 `-2.0` 保留。
- MuJoCo 默认范围同步为 `0.65～0.85 m`，并会从 checkpoint 邻接的 `params/env.yaml` 自动恢复旧模型的训练范围；旧 `0.62～0.85 m` 模型4步无窗口冒烟通过且正确识别旧范围。
- 纯张量单元测试扩展到 11 项并通过；完整 Isaac Sim 环境冒烟在应用导入前被 NVIDIA EULA 交互提示阻止，未代替用户接受协议。

### 2026-08-15 终止惩罚回归修复

- 诊断运行 `2026-08-15_00-26-20_wheel_single_stage_h065_085_supportgate`：约 iteration 4020 时末 100 次 mean reward `-80.99`、episode length `15.3/1000`、base contact 终止率 `100%`。
- `pen_base_contact_termination/keep_balance` 约为 `-200`，确认 `mdp.is_terminated_term` 将持久化的 `_term_dones` 当作当前步信号，导致 reset 后每一步重复扣分；该运行标记为无效且不得 resume。
- 终止惩罚改用 `mdp.is_terminated` 当前步非 timeout 信号；当前唯一非 timeout 终止项为 `base_contact`。新增环境冒烟断言，禁止再次配置为持久化的 per-term 历史。

## 会话：2026-08-16——稳定性再平衡与高度切换门控

- **状态：** implementation_complete_single_stage_retrain_pending
- 诊断修复后 20k 运行：`model_6500` 的 base contact/timeout 为 `7.8%/92.2%`，最终 `model_20000` 退化为 `23.4%/76.6%`，二者高度 RMSE 均约 `7.3 cm`。
- Wheel 高度权重从 `-100` 回调到 `-60`，当前步非 timeout 终止权重从 `-200` 提升到 `-500`，零命令轮目标从 `-0.25` 降到 `-0.10`，死区保持 `0.15 rad/s`。
- `pen_lin_vel_z` 保持 `-0.3`，但目标差 `≤0.01 m` 时为全量、`≥0.04 m` 时缩放到 `20%`、中间线性过渡。
- 新增低/中/高高度段的 base contact rate 和 reset fraction 指标，边界为 `0.70/0.80 m`。
- 新增 `select_wf_checkpoint.py`：在高度/速度/地形门槛内按 100-iteration base contact 均值选择 checkpoint；对上一轮日志正确推荐 `model_6500.pt`，可写部署选择文件供 MuJoCo 优先加载。
- 纯张量单元测试扩展到 `13/13 PASS`；真实 Isaac Lab Wheel CPU 1-env 冒烟通过，27 个奖励项连续 3 步均为有限值，并确认新权重和竖直速度门控函数已注册。
