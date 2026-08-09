# 发现与决策：WF 双模态方案

## 最新需求
- 只训练轮足 WF_TRON1A。
- 使用两个独立神经网络模态。
- 轮态：依靠轮子移动，不主动抬轮/迈步，处理除上楼梯之外的场景。
- 足态：左右轮速度目标为 0，依靠腿式步态移动，处理全地形。
- 两个模态继续支持运行时机身高度命令。
- 新的在线模态定义：
  - 腾空态：所有受监测接触消失；
  - 固定姿态轮行态：轮地接触成立，轮心拟合平面法向与局部地面法向近似平行，且足端在基座系的固定构型包络内；
  - 步态足行态：非腾空且不满足固定轮行条件，并存在腿构型或接触状态的步态性时序变化。
- 用户要求据此重新设计训练奖惩与 reset 条件。

## 模态判定重构原则（2026-08-03）
- 单帧接触力不足以稳定决定模态；接触、几何、构型和步态性都需要时间窗、进入/退出双阈值及最短驻留时间。
- “状态分类”与“失败终止”分开：`AIRBORNE` 是可恢复状态，不触发终止；用连续惩罚、运动奖励门控和落地冲击约束抑制跳跃。
- 只有两个轮心时，“拟合平面法向”数学上不唯一；实现时应采用轮轴方向与局部地面法向的正交关系、基座姿态，或加入更多几何点构造支撑平面。

## 奖惩代码实施核对（2026-08-03）
- 当前 ContactSensor 已配置 `history_length=4` 和 `track_air_time=True`，可以直接用短历史生成连续接触置信度。
- 现有 `wheel_air_time_l2` 使用 `last_air_time`；应优先改为 `current_air_time`，使惩罚随当前腾空持续增长，并保留旧版字段兼容。
- 当前 `wheel_contact_loss` 和摆动高度均只看瞬时 `net_forces_w`，需要改成短历史力值与连续置信度，避免单帧接触噪声。
- 当前滚动误差只比较两轮绝对速度均值与 `|base_vx|`，暂不含轮速符号/差速 yaw；本轮先保留兼容公式，后续可按实际轮关节轴符号标定。
- 仓库没有现成 reward 单元测试目录；本轮需要新增不依赖完整 SimulationApp 的纯张量辅助函数测试。
- IsaacLab 2.2.1 的 `ContactSensorData` 明确提供 `current_air_time/current_contact_time`，且 `compute_first_contact(dt)` 可直接识别本控制周期内的新接触。
- class-based reward term 会在 `RewardManager.reset(env_ids)` 中收到逐环境 reset，适合实现需要前一帧状态的落地冲击或接触滞回；但第一版优先使用传感器原生时长，减少重复状态。
- IsaacLab 原生速度跟踪奖励是无条件指数奖励，需要新增本地 wrapper 乘以支撑接触置信度，才能让完全腾空时不靠惯性获得正向运动奖励。
- 基类 `UniformVelocityCommandCfg.debug_vis=True` 会在 headless train 创建远程箭头 USD；双专家训练配置应显式关闭，仅 Play 开启。
- 宿主 IsaacLab 4.5/2.2.1 冒烟通过：Wheel 28 个、Foot 24 个 reward term 均完成实体解析并在 3 个零动作步内保持有限值；两者 termination 仅有 `time_out/base_contact`。
- Wheel 冒烟第 3 步 `pen_wheel_contact=-8.0`、接触门控的速度/yaw 奖励均为 0，证明新门控有效，也再次证明当前 reset 没有把轮子放入有效接触状态。
- Wheel 初始 `pen_wheel_terrain_geometry≈-17.66`，在 reset 修复前该强惩罚主要反映无效出生几何；正式训练前应先修 reset，再根据有效站立样本重新校准 clearance scale/weight。

## WF 源码现状
- 机器人有 8 个受控关节：6 个腿关节和 2 个轮关节。
- 当前动作空间已经天然分成两项：
  - JointPositionAction：6 个腿关节，scale=0.25。
  - JointVelocityAction：2 个轮关节，scale=1.0。
- 轮执行器配置：stiffness=0、damping=0.8、effort_limit=80、velocity_limit=15。
- policy 观测排除了 wheel position，但包含全部 joint velocity，因此可以感知轮速。
- 当前 WF 只有 velocity command，没有 gait command。
- 当前 WF 固定机身高度目标为 0.80 m。
- 当前奖励包含 wheel joint velocity L2 惩罚，这与轮态高速滚动目标冲突。
- 当前已有 WF Blind Stair、Stair、Rough 等环境类，但没有完整 Gym 注册。

## 双专家架构

### Wheel Expert
- 模型输入：本体状态、轮速、上一步动作、速度命令、机身高度命令；可选低维地形统计。
- 模型输出：6 个腿位置动作 + 2 个轮速度动作。
- 地形范围：平地、上/下坡、连续粗糙地面；下楼梯作为晚期可选课程。
- 腾空处理：作为可恢复状态惩罚并门控运动正奖励，不使用硬终止。
- 软约束：提高双轮接触率、抑制主动摆腿和单轮长时间离地。

### Foot Expert
- 模型输入：本体状态、地形 height scan、步态相位/参数、速度命令、机身高度命令。
- 模型输出：仅 6 个腿位置动作。
- 控制适配器：每个控制周期追加 [0, 0] 作为左右轮速度目标。
- 地形范围：平地、坡面、随机粗糙、上/下楼梯。
- 硬约束：轮速度命令恒为 0。
- 训练监控：实际轮速、轮电机保持力矩和打滑仍需作为指标/惩罚。

## 轮态接触约束的可行定义
- 在平地、坡面和连续粗糙地面上，可以要求双轮接触率接近 100%。
- 在下台阶边缘，几何上可能发生极短暂单轮失联；若两个轮子每一帧必须都接触，则下楼梯必须从轮态地形中移除。
- 推荐验收定义：
  - 不允许主动步态抬轮；
  - 双轮同时腾空计入 AIRBORNE 指标并持续惩罚，但允许重新接触恢复；
  - 单轮离地只允许短暂发生，并施加强惩罚；
  - 对连续地形统计每轮接触率。

## 模式管理
- 首版使用外部 mode command：WHEEL / FOOT。
- 后续可以由高度扫描、深度相机或地图模块自动选择，但不建议首版再训练第三个 gating 网络。
- 必须有确定性切换 FSM：
  - Wheel→Foot：速度降到阈值、轮速归零、双轮接触、姿态稳定后切换。
  - Foot→Wheel：进入双支撑、两轮接触、目标高度稳定后缓慢增加轮速。
- 模式切换需加入滞回和最短驻留时间，防止地形边缘频繁抖动。

## 奖励拆分

### Wheel 模式
- 速度/航向跟踪。
- 高度命令跟踪。
- 双轮接触奖励、air-time 惩罚和空中运动奖励门控；不使用双轮腾空终止。
- 单轮离地时长惩罚。
- 轮侧滑惩罚。
- 轮角速度与机身前向速度/radius 一致性。
- 腿姿态、左右对称和动作平滑。
- 抑制腿部周期性大动作和轮子竖直速度。
- 移除或显著降低现有 wheel velocity L2 惩罚。

### Foot 模式
- 速度/航向跟踪。
- 高度命令跟踪。
- gait contact schedule。
- 摆动轮（作为足端）的离地净空与落脚位置。
- 落地冲击和非轮部件碰撞惩罚。
- 实际 wheel joint velocity 强惩罚。
- 轮速度目标硬编码 0。
- 地形课程和台阶通过奖励。

## 高度命令
- 两个专家都接收 body_height_command。
- 初始建议范围：
  - Wheel：0.75～0.85 m，默认 0.80 m。
  - Foot：0.65～0.82 m，默认 0.78～0.80 m。
- 高度按相对局部地面计算。
- 训练先采用 episode 内固定随机高度，再增加 5～8 s 重采样和 0.05～0.10 m/s 变化限速。

## 建议任务名
- Isaac-Limx-WF-Wheel-Mode-v0
- Isaac-Limx-WF-Wheel-Mode-Play-v0
- Isaac-Limx-WF-Foot-AllTerrain-v0
- Isaac-Limx-WF-Foot-AllTerrain-Play-v0

可选调试任务：
- Isaac-Limx-WF-Foot-Flat-v0
- Isaac-Limx-WF-Wheel-Flat-v0
- Isaac-Limx-WF-Wheel-Slope-v0
- Isaac-Limx-WF-Foot-Stair-v0

## 主要代码位置
- assets/config/wheelfoot_cfg.py：轮/腿执行器配置。
- cfg/WF/limx_base_env_cfg.py：WF 动作、观测、命令和奖励。
- cfg/WF/terrains_cfg.py：双模态地形课程。
- robots/limx_wheelfoot_env_cfg.py：Wheel/Foot 环境类。
- robots/__init__.py：Gym 注册。
- mdp/commands/：高度与 gait command。
- mdp/rewards.py：模式专用接触、轮速、滑移、脚高和高度奖励。
- scripts/rsl_rl/play.py：双模型加载和 FSM 切换。

## 外部资料
- 本轮结论来自本地源码，不包含网页指令或外部不可信内容。

## 实施前接口核对（2026-08-02）
- WF 当前 action manager 已是两个 action term：6 维 JointPositionAction + 2 维 JointVelocityAction。
- Policy/History 的 joint velocity 包含轮速，joint position 排除轮角度；适合两个专家共享大部分本体观测。
- Critic 已包含 height scan、轮端线速度和轮端接触力等特权信息。
- 当前 rough terrain 是平地、波浪、低网格凸起、随机高度场各 25%，适合作为轮态连续地形起点。
- 当前 stairs terrain 混合上/下楼梯 80% 和上/下坡 20%；实施时足态保留全地形，轮态单独使用不含上楼梯的地形配置。
- 当前代码中 wheel_slide_penalty 仅被注释引用，未发现本地实现；需要新增模式专用滚动/接触奖励函数。
- ActionManager 按各 term.action_dim 切片，理论上支持 0 维固定动作 term；但为兼容 Isaac Lab v2.2.1，实施优先采用同为 8 维的足态动作并将 wheel velocity action 的 scale/offset 固定为 0，确保控制目标为零且避免自定义 0 维 API 风险。
- 两个专家保持相同 8 维输出可简化 Runner、checkpoint、导出和 FSM 切换；足态最后两维由环境动作层硬 mask，神经网络输出不会影响轮电机。
- 为降低对旧 Isaac Lab API 的改动风险，模式环境可新建独立配置文件并继承现有 WFBaseEnvCfg，在 __post_init__ 中增删命令、观测、奖励和地形；基础 Flat 任务保持不变。
- 足态采用 8 维统一动作接口：腿动作正常处理，轮速度 action term 的 scale=0 且 offset/default velocity=0，从 action manager 层把两个轮目标硬 mask 为零。这样不改自带 Runner 的 action shape 逻辑，也便于两个模型统一部署。
- 为支持单一仿真环境内无缝切换两个 checkpoint，两个专家应保持相同 policy observation、history、commands 和 8 维 action schema。
- 两个任务都提供 height scan、gait phase/gait command；Wheel Expert 中 gait command 固定且不参与 gait reward，Foot Expert 使用相同字段训练步态。这样 dual-play 可对同一 obs 同时运行两个 actor。
- Foot Expert 的最后两维动作由训练环境 scale=0 硬屏蔽；双策略 Play/FSM 在 FOOT 状态再次将最后两维强制设 0，形成双层保护。

## 新版模态判断方法差异审查（2026-08-05）

- 权威输入为 `/home/tuchuaan/UDMMR-Extra-Docs/课题md/模态判断方法.md`；项目内 `docs/模态判断方法.md` 仍是旧版，包含“轮轴/轮心平面法向参与轮态判定”的过期语义，需要实施阶段同步。
- 新版公开定义只有 `AIRBORNE/WHEEL_FIXED/FOOT_GAIT` 三种物理模态；证据不足不应硬造第四种运动模态，建议通过独立的 `valid/confidence/transitioning` 表示，内部可使用 `-1` sentinel。
- 新版轮态要求所有轮子有效触地；当前 `support_confidence()` 使用 `amax`，会在只有一个轮子接触时开放 Wheel 速度正奖励，与新定义冲突。
- 新版轮地接触要求“地面过滤接触力 AND 轮心—局部地面几何距离”；当前通用 contact sensor 的总力无法排除墙、台阶立面和自碰撞。
- Isaac Lab 2.2.1 filtered contact 只可靠支持 one-to-many；左右轮需要两个独立 ground-contact sensor，不能用一个多轮 filtered sensor。
- 新版明确轮轴—地面法向不参与判断、最多作为 debug；当前 `wheel_geometry_penalty()` 将 axis error 纳入 reward，必须移出 reward/mode/pass-fail。
- 新版只约束轮心在 base 系下的水平中性位置并允许竖直调节；当前构型项约束左右轮 z 差，当前相对速度项惩罚 xyz，当前 wheel vertical velocity 项直接惩罚 z 运动，三者都需调整。
- 新版不建议用送入网络的采样点直接判断轮下地面；建议增加两个不进入 policy observation 的 wheel-local downward ray sensors。
- 新版地形是否适合轮态由专家实测划分；当前 `terrain_mode_request()` 的 height-relief 逻辑只能保留为未知地形 fallback，正式 auto mode 应消费双专家 feasibility map。
- 详细文件级方案已写入 `agent_docs/human/wf_new_mode_judgment_modification_plan.md`。

## 方案简化决策（2026-08-05）

- 用户确认 Wheel Expert 不需要 Foot gait 时间窗证据；训练目标直接改为“所有轮子有效触地”和“轮心/腿端水平位置不远离中性点”。
- 不再用腿运动 RMS、接触切换周期或 gait phase 决定 Wheel reward，也不在 runtime 自动推断 `FOOT_GAIT`。
- 非腾空但不满足 Wheel 合规条件的状态记录为 `wheel_invalid/non_wheel_compliant`，避免把单轮失联、卡住等失败状态错误命名为足态。
- 时间维度只保留在接触力/几何的短时防抖，以及 Foot→Wheel 切换条件的稳定确认；这两者不是 gait 时间窗。
- Wheel 正奖励使用 all-wheel contact gate，另加单轮缺失接触连续惩罚和 base 系 x/y 中性点 dead-zone 惩罚；z 调节不受该构型项惩罚。
- 详细方案已覆盖更新 `agent_docs/human/wf_new_mode_judgment_modification_plan.md`。

## Wheel 新合规奖励实施结果（2026-08-05）

- Wheel 速度/yaw 正奖励已从“任意轮接触”改为左右轮有效接触置信度的最小值。
- 单轮有效接触由独立 ground-filtered contact force 与 wheel-local ground clearance 相乘得到；地形过滤 prim 为 `/World/ground/terrain/mesh`。
- 新增左右轮独立 ContactSensor 和 RayCaster；实测 filtered force 最大值约 349 N / 267 N，确认过滤矩阵有有效输出。
- 使用旧 `model_20000.pt` 400 步标定，轮心 base 系中位数约为左 `(-0.004, 0.168)`、右 `(0.001, -0.176)` m；训练目标取对称值 `(0, ±0.17)` m。
- 新水平中性点项只约束 x/y，容差为 0.04/0.03 m，不惩罚 z 调节。
- 已关闭旧的 same-foot-x、feet-distance、wheel vertical velocity、axis/clearance 独立项、x/z/轮距包络和 xyz relative velocity 项。
- `--device cpu` 之前只影响环境、未覆盖 PPO runner device；已在 `scripts/rsl_rl/cli_args.py` 同步 agent config device。
