# Assignment 3 项目方案：Snake 上的值函数深度强化学习

> 团队最终决定：方向 B，载体 Snake。本文件 = 可执行设计：环境与状态/奖励设计 → 算法阶梯 → 实验矩阵（项目的分数所在）→ 评估协议 → 5090 方案 → 分工/时间线 → Part A 草稿 → 答辩清单。
> 开宗明义把最大风险钉在墙上：**Snake+DQN 教程全网泛滥，这个项目的及格线是"能跑"，得分线是"受控实验 + 统计严谨 + 行为分析"。** 所有设计都围绕后者。

---

## 1. 环境设计（自建，约 150 行，一周内完成）

| 决定 | 选择 | 理由 |
|---|---|---|
| 网格 | **10×10 主体**（8×8 调试、16×16 扩展实验） | 难度旋钮；10×10 上 DQN 数小时可见成效 [INFERRED, MED] |
| 动作空间 | **相对 3 动作：直行/左转/右转**（绝对 4 方向作消融） | 相对动作天然排除 180° 回头的非法动作；两种动作空间的对比本身是一个实验 |
| 终止 | 撞墙 / 撞自己 / **饥饿截断**（连续 H=100 步没吃到食物强制结束，记为 starve） | 饥饿截断是反"绕圈刷分"的制度性防线，必须有 |
| 食物 | 均匀随机刷在空格 | 保持环境随机性 → 策略必须泛化，不能背板 |

**接口按 Gymnasium API 写**（reset/step 返回新 5 元组）：与课程工具链一致，向量化并行（第 5 节）零成本接入。

## 2. 状态表示——第一大实验轴（同一环境，喂两种眼睛）

**表示 A：11 维特征向量 + MLP（工程特征基线）**
三方向危险标志（直/左/右一步会死？）3 + 当前方向 one-hot 4 + 食物相对方位 4 = 11 个布尔量。诚实声明：这是社区教程的经典表示 [COMMON, HIGH]——**报告里大方引用它作 related work，我们的贡献是把它与其他表示做受控对比**,而不是假装原创。这个坦白本身就是对"教程复制"质疑的最好防御。

**表示 B：全网格多平面图像 + CNN（表示学习路线）**
3–4 个 10×10 平面：蛇头 / 蛇身 / 食物 / **方向信息**。

**[关键理论点，答辩必讲] 单帧网格不含方向 → 非 Markov 状态**：只看棋盘快照,你不知道蛇头正朝哪走,下一状态不可预测——这直接违反课程 Week 3 的 Markov 性质。修复方案（三选一，选哪个都要能说出理由）：加方向平面 / 叠两帧（frame stacking，DQN 论文的做法）/ 蛇身按"年龄"编码（身体各段的时序信息隐含方向）。**把课程的 Markov 理论用在自己的状态设计决策上，是全项目最能体现"学懂了"的地方。**

预期结论（先当假设写进 Part B，用实验验证或推翻）：A 学得快（先验知识浓缩）但天花板受限于特征遗漏（看不到蛇身整体布局 → 不会避免把自己围死）；B 样本效率低但上限高。**"特征工程 vs 表示学习"就是本项目的第一研究问题。**

## 3. 奖励设计——第二大实验轴

| 变体 | 定义 | 预期/看点 |
|---|---|---|
| R0 稀疏 | 吃食 +1，死亡 −1，其余 0 | 最干净但学得慢——信用分配困难的活教材 |
| R1 步惩罚 | R0 + 每步 −0.01 | 抑制磨蹭；惩罚过大蛇会"自杀求解脱"——若复现，录像留证 |
| R2 势函数塑形 | R0 + F = γΦ(s′) − Φ(s)，Φ = −到食物的归一化距离 | **唯一保证不改变最优策略的塑形**（Ng et al. 1999 [KNOWN, HIGH]）|
| R3 朴素距离塑形 | 靠近食物 +0.01、远离 −0.01（非势函数式） | 理论上可被钻空子（来回横跳刷分）；配合无饥饿截断跑一组，抓 **reward hacking** 现场 |

R2 vs R3 的对比是报告理论深度的支点：同样是"引导蛇靠近食物"，一个有最优策略不变性的证明，一个没有——**跑出行为差异就是论文级的图，跑不出差异也要解释为什么（塑形量级小/截断兜底）**。诚实汇报，两种结局都能写。

## 4. 算法阶梯与实验矩阵

**Phase 0 非学习基线（第 1 周）**：随机；贪心 A*（最短路吃食，不管把自己困死）；**汉密尔顿回路 bot**（沿固定遍历回路走，偶数网格上近乎不死 [COMMON, MED]）——它是"非学习规划上界"：慢但几乎满分。RL agent 夹在 A* 和 Hamiltonian 之间的位置，给整个项目一个诚实的坐标系,这个对照别的组几乎不会做。

**Phase 1 表格 Q-learning（课程直连）**：表示 A 的 11 布尔特征 → 状态空间约 2¹¹ 量级,表格完全放得下。作用：无神经网络干扰地调通 ε 衰减/α/γ；报告里"函数逼近之前的世界"参照点；与课程 model-free 周无缝衔接。

**Phase 2 DQN 核心**：replay buffer + target network，MLP（表示 A）与 CNN（表示 B）各训一版。**组件剔除消融**：完整 / −replay / −target / 双删,四组曲线正面验证 deadly triad 叙事（函数逼近+自举+off-policy 无收敛保证,S&B §11.3 [KNOWN, HIGH]）。

**Phase 3 改进族（分工切分线）**：
- **Double DQN**：动作选择与评估解耦。**必做配套测量**：训练全程记录 max_a Q 的均值 vs 实际实现回报——vanilla DQN 的系统性过估计肉眼可见 [KNOWN, HIGH, van Hasselt 2015]，这是全项目最有说服力的一张图。
- **Dueling**：V+A 分流。Snake 大量状态下"三个动作差别不大" → Dueling 的理论适用场景，验证之。
- **PER**：按 TD 误差优先采样。
- 刹车原则：三个改进 + 完整消融 > 六个改进无消融。宁深勿广。

**实验矩阵总表（Part E 实验章的骨架）**：

| 轴 | 对比项 | 主指标 |
|---|---|---|
| 状态表示 | A vs B（vs A+B 拼接，可选） | 样本效率曲线 + 最终分数 |
| 奖励设计 | R0–R3 | 分数 + 行为统计（死因/轨迹熵） |
| 组件消融 | ±replay ±target | 稳定性（跨种子方差） |
| 算法改进 | DQN/DDQN/Dueling/PER | 分数 + Q 过估计曲线 |
| 超参敏感 | ε 衰减、buffer 大小、target 同步周期、γ | 分数热力图 |
| 动作空间 | 相对 3 vs 绝对 4 | 收敛速度 |

全矩阵跑不完就砍行,但**每行 5 种子 + 置信区间**不许砍——统计严谨是这个项目的立身之本。

## 5. 评估协议与行为分析

- **训练/评估分离**：每训练 N 步，冻结网络跑 100 局**纯贪心**评估（ε=0）。报告一律画评估曲线；训练期回报噪声大且含探索，不可作为主要证据。
- **指标**：平均得分（吃食数）、得分分布（箱线图,别只报均值）、存活步数、**死因分解**（撞墙/撞己/饿死比例随训练的演变——从"到处撞墙"到"主要死于把自己围死"的迁移本身就是学习进程的叙事）。
- **行为分析（差异化亮点，成本极低）**：访问热力图（agent 偏好贴边还是穿中）、临死前轨迹回放、reward hacking 案例集。答辩放行为图比放 loss 曲线高一个档次。
- 上下界锚点：随机 / A* / Hamiltonian 与 RL 各版本同图。

## 6. RTX 5090 方案

[COMMON, HIGH] 瓶颈在 Python 环境步进，不在 GPU。对策：
- **向量化环境**：NumPy 批量实现 128–256 个并行 Snake（棋盘就是一个 (N,10,10) 张量,碰撞/移动全部向量化）,一次前向推理全部动作。这一条做对,训练吞吐差一个数量级。
- **实验吞吐才是 5090 的主战场**：夜间批量 5 种子 × 矩阵各行。多数组在 Colab 上给不出置信区间,你们给得出——硬件优势兑现成分数的唯一通道。
- 环境:PyTorch 2.7+/CUDA 12.8+（Blackwell）[KNOWN, MED,以官网为准]；`requirements.txt` + 种子固定 + 每次运行记录 config/git hash,第一天就立规矩。

## 7. GUI 与演示（Part D / F）

- **GUI**：pygame。三种模式：人玩 / agent 玩（实时侧栏：三个动作的 Q 值条形图 + 当前选择——**让评委看见 agent 在想什么**）/ 录像回放。
- **演示脚本**：① 训练早/中/晚 checkpoint 三段录像（乱撞 → 会吃食 → 会绕开自己身体）；② reward hacking 花絮（若抓到）；③ 现场让评委玩一分钟,再放 agent 同种子棋盘的表现;④ Q 值侧栏实时解说一次"它为什么选择转弯"。

## 8. 分工（4 人示例，按"人人可答辩"设计）

| 成员 | 主责 | 答辩必会 |
|---|---|---|
| A | 环境 + 向量化 + 非学习基线（A*/Hamiltonian） | 环境设计决策、Markov 性讨论、基线定位 |
| B | 表格 Q-learning + DQN 核心 | Q-learning 更新式、replay/target 的存在理由、deadly triad |
| C | DDQN/Dueling/PER + 过估计测量 | max 算子过估计机制、各改进的动机 |
| D | 评估协议 + 行为分析 + GUI/演示 | 为什么评估要贪心分离、种子与置信区间、死因分解解读 |

git 提交记录从第一周留痕（最终分按个人贡献分配）。每周 30 分钟互讲。

## 9. 时间线（对齐 Part A–G）[INFERRED]

| 周期 | 里程碑 | 交付 |
|---|---|---|
| 现在–W6 | **A1 优先（9/4 截止）**；项目仅做:定题 consultation、Part A 提交、环境+基线骨架 | Part A、B |
| W7 | 表格 Q-learning 出结果;DQN(表示A)首条能看的评估曲线 | — |
| W8 | DQN(表示B/CNN)跑通;初步对比表 | **Part C** |
| W8–9 | GUI 三模式成型 | **Part D** |
| W9–11 | 实验矩阵夜间批量;行为分析图;改进族+过估计测量 | Part E 素材 |
| W12– | 报告、演示排练、答辩互考 | Part E、F、G |

兜底线：W8 若 CNN 线不收敛,主线降级为"表示 A 全家桶 + 深挖奖励塑形与消融",CNN 转为负面结果章节（诚实的负面结果 + 原因分析,答辩反而好讲）。

## 10. Part A 草稿（题目 + Abstract，可直接填注册模板）

**Project Title**: Value-Based Deep Reinforcement Learning on Snake: A Controlled Study of State Representation, Reward Design, and DQN Stabilisation

**Abstract (~200 words)**:
This project uses the classic game Snake as a controlled testbed for value-based reinforcement learning, progressing from tabular Q-learning to Deep Q-Networks (DQN) and their modern refinements. We build a custom Gymnasium-compatible environment and study three questions central to deep RL practice. First, state representation: we compare a compact engineered feature vector (danger flags, heading, food direction) against a raw multi-plane grid processed by a convolutional network, examining the trade-off between sample efficiency and performance ceiling, and addressing the Markov property explicitly in each design. Second, reward design: we contrast sparse rewards, step penalties, and potential-based shaping — the only shaping form guaranteed to preserve optimal policies — and document failure modes such as reward hacking under naive distance-based shaping. Third, stabilisation: we ablate experience replay and target networks to connect training instability with the deadly-triad analysis, and quantify the overestimation bias of vanilla DQN against Double DQN, additionally evaluating Dueling architectures and prioritised replay. All experiments run with five random seeds on vectorised environments (local RTX 5090), reported with confidence intervals against non-learning baselines including an A* greedy player and a Hamiltonian-cycle upper bound. Deliverables include an interactive GUI visualising live Q-values during play.

（[COMPUTED] 约 200 词。填模板时补团队名、成员、学号。）

## 11. 答辩互考清单（全员必过）

1. 写出表格 Q-learning 更新式与 DQN 的损失函数;指出"哪部分被冻结在 target 网络里"、为什么。
2. experience replay 解决什么问题?它为什么**要求算法是 off-policy**?（buffer 里是旧策略的数据 → on-policy 的 SARSA 不能直接用,Q-learning 可以——这题最能筛掉"只跑过代码"的人）
3. deadly triad 三要素;本项目哪条配置踩满三样。
4. 势函数塑形 F=γΦ(s′)−Φ(s) 为什么不改变最优策略（回报望远镜求和后只差常数 Φ(s₀)）;R3 为什么没有这个保证。
5. 你的网格状态表示如何保证 Markov 性?若去掉方向信息会发生什么?
6. max 算子为什么导致过估计（对带噪估计取 max 的期望 ≥ 真值的 max）;Double DQN 怎么解耦。
7. 为什么评估曲线要用贪心策略单独跑,而不是直接画训练回报?
8. 你们的 agent 和 A*、Hamiltonian 比,赢在哪输在哪?RL 在这个问题上到底买到了什么?
9. 哪个实验失败了/结果与预期不符,你们学到什么?（提前准备,这是送分题也是必问题）
