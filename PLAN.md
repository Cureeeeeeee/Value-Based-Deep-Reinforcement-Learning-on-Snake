# Ass3 实施计划 v1（代码阶段）

> 2026-09-15。前提（你今天定的）：Part A/B 已交（项目号 48）；你主导全部代码线；新建 GitHub 仓库；不按截止日期排，按里程碑推进；**第一步 = 环境**。
> 本文件是"怎么造"的契约，不重复 Part B 的"造什么"。Part B 里已写死的设计（10×10、相对 3 动作、R0–R3、A/B 两种观测、饥饿截断 100）一律沿用；下面只写 Part B 没定或定错的部分。

---

## 0. 先说反面：Part B 哪里要砍、哪里要改

1. **人力与矩阵不匹配。** Part B 的实验矩阵（6 轴 × 5 种子）是按 4 人写的，现在一人主导。[INFERRED, HIGH] 全矩阵做不完。裁剪顺序改为 **PER → Dueling → 超参敏感性 → 动作空间**。Part B 写的顺序（动作空间 → 超参 → PER）把最便宜的实验（动作空间只是一个环境开关，成本≈0）排在最先砍，把实现成本最高、教学价值最低的 PER 排最后砍，顺序反了。
2. **Markov 论述要精确化，且 Part B 给的修复不够。** Part B："单帧网格无方向信息 → 非 Markov，加方向平面即可修复"。严格说，二值身体平面丢掉的是**身体的顺序**：蛇身盘绕时，颈（决定朝向）和尾（决定下一步哪格腾出）都不能从快照唯一恢复，两者都影响下一状态 → 非 Markov。[INFERRED, HIGH] 只加方向平面只修复了颈，尾仍然不可恢复；**身体"年龄"平面**（尾=1 … 头=len）一次修复两者。所以观测 B 以年龄平面为主，方向平面只是冗余。这给出一个近零成本实验：**CNN 有/无年龄平面**，直接检验课程 Week 3 的 Markov 概念。Part E 里把 Part B 这句话改掉。
3. **truncated ≠ terminated。** 饥饿截断时下一状态仍要 bootstrap（不能把 target 置为 r），撞墙/撞己才置零。教程代码普遍做错。[COMMON, HIGH] 这也是 R3 能否出现 reward hacking 的机制点：若把截断当终止，绕圈刷分的价值会被人为压低。
4. **5090 是 Blackwell（sm_120）**：需要 PyTorch ≥ 2.7 的 CUDA 12.8 wheel [KNOWN, MED，以官网为准]；你本地 Python 3.13.7 与之兼容 [KNOWN, MED]。Part B 写的 Python 3.11 不必迁就。**第一天必须验证 `torch.cuda.is_available()`**，否则 Part B 的整个吞吐假设作废。
5. **向量化会悄悄改掉 update-to-data 比。** 128 个并行环境每个 vec-step 产生 128 条 transition；若每个 vec-step 只做 1 次梯度更新，UTD = 1/128，远低于 DQN 常规的 1/4，网络学不动。[COMMON, MED] 梯度步数/vec-step 必须是显式配置项（默认 N_envs/4 = 32）。
6. **11 维特征 A 本身就不是 Markov 的**（丢掉了身体布局）。这不是缺陷，是假设"A 天花板受限"的机制来源；报告里要明说，别只说 B 的 Markov 问题。[INFERRED, HIGH]

---

## 1. 工作方式

本仓库在本地机器上用 Claude Code 开发（Windows 11，Python 3.13.7，RTX 5090）。分层：**短跑在前台、长跑在后台**。

| 类型 | 做法 | 时长上限 |
|---|---|---|
| 单元测试 / 冒烟训练（8×8，几万步，CPU 或 GPU） | Claude Code 直接跑，结果进对话 | 分钟级 |
| 正式训练 / 矩阵批量（5 种子 × 配置） | `scripts/run_matrix.py` 生成命令，用户在独立终端后台跑；Claude Code 只读 `runs/` 里的日志做分析 | 小时级，夜间 |
| GUI 演示 | 用户手动运行 `python -m snake_rl.gui.app` | — |

规矩（Part B 承诺的可复现性，第一天就立）：每次运行写 `config.yaml + seed + git hash` 到 run 目录；所有表/图由 `analysis/` 从日志重生成，不手改；`runs/` 不进 git（`.gitignore` 已排除），但永远不删。

---

## 2. 仓库结构

仓库名建议 `snake-rl`（或 `43008-snake-dqn`）。

```
snake-rl/
  README.md
  requirements.txt          # numpy torch gymnasium pygame pytest matplotlib pandas pyyaml
  snake_rl/
    env/
      core.py               # 标量参考环境：gymnasium.Env 子类，纯 Python，可读性优先
      vec_env.py            # 向量化环境：(N,W,W) NumPy 批量步进，autoreset
      obs.py                # 观测编码 A（11 维）/ B（多平面）；两种环境共用
      rewards.py            # R0–R3；两种环境共用
    agents/
      baselines.py          # random / BFS 贪心 / Hamiltonian
      tabular_q.py          # 表格 Q-learning（观测 A；同步批量形式，N=1 时即教科书更新）
      schedules.py          # ε 线性调度（表格 Q 与 DQN 共用）
      dqn/
        config.py           # DQNConfig（不导入 torch；M5 新增）
        networks.py         # MLP / CNN / Dueling 头
        replay.py           # 均匀 buffer；PER 放最后
        agent.py            # DQN，flags: double / dueling
    config.py               # YAML 实验配置：加载、--set 覆盖、校验、种子布局（M4 新增）
    train.py                # 配置驱动；写 run 目录（config、seed、git hash、csv 日志、checkpoint）
    evaluate.py             # 贪心评估协议：100 局、固定评估种子、死因分解；标量版与向量化版（evaluate_batched）逐局相同
    analysis/               # aggregate.py（从 run 目录生成结果表与学习曲线，跨种子 t 区间）、plots.py
    gui/app.py              # pygame：人玩 / agent 玩 + Q 值侧栏 / 回放
  configs/                  # 每个实验一份 yaml
  tests/                    # test_env.py test_vec_env.py test_obs.py test_rewards.py test_baselines.py；M4 起另有 test_tabular_q / test_train / test_evaluate_batched / test_aggregate / test_run_matrix 等
  scripts/run_matrix.py     # 批量跑种子 × 配置（另有 check_env.py、bench_vec_env.py、gui_screenshots.py）
  notebooks/results.ipynb   # Colab：从日志/checkpoint 重生成主图（提交要求）
```

关键结构决定：**两个环境实现共享 `obs.py`/`rewards.py`**，观测和奖励只写一遍；`core.py` 是"显然正确"的参考实现（给测试、GUI、答辩讲解用），`vec_env.py` 是训练引擎；`tests/test_vec_env.py` 强制两者逐步等价。

---

## 3. 环境规格（第一步；冻结的契约）

Part B 已定的不重复。以下是实现层面的决定，标 **[决定]** 的可改，改了要同步改 Part B 描述。

### 3.1 内部状态（标量与向量化同一套语义）

- `W` 网格边长（默认 10；8 调试；16 扩展）。
- `body_age: int[W,W]`：0 = 空；k > 0 = 该格身体还能存活 k 步。尾 = 1，颈 = len−1，头 = len。**一个数组同时给出身体、头、尾、朝向（颈=头邻域中值最大者）和观测 B 的年龄平面。**
- `heading ∈ {0 上, 1 右, 2 下, 3 左}`、`food (r,c)`、`length`、`steps_since_food`、`score`。
- 初始 **[决定]**：长度 3，水平居中，朝右；食物随机。

### 3.2 动作

- 相对模式（主）：`a ∈ {0 直行, 1 左转, 2 右转}` → `heading' = (heading + {0, −1, +1}) mod 4`。
- 绝对模式（消融）：`a ∈ {0..3}`；**[决定] 与当前朝向相反的动作视为直行，不判死**（常见处理 [COMMON, MED]；判死的话 4 动作会被"25% 的动作必死"这个人为因素压低，对比不公平）。

### 3.3 `step` 的执行顺序（顺序即语义，测试按这个写）

1. 由动作得 `heading'`，算 `new_head`。
2. 越界 → `terminated`, cause = `wall`。
3. `new_head == food` → 吃：**不递减** `body_age`（尾不动），`length += 1`，`body_age[new_head] = length`，`score += 1`，`steps_since_food = 0`，在空格上均匀重生食物；**若已无空格（`length == W²`）→ `terminated`, cause = `win`**（满盘；10×10 最高分 W²−3 = 97）。
4. 否则 → 先递减所有正格（尾格归 0，腾出），再查 `body_age[new_head] > 0` → `terminated`, cause = `self`；否则 `body_age[new_head] = length`，`steps_since_food += 1`。
   **[决定] 追尾进入"本步正在腾出的尾格"合法**（先递减后判定自然给出；这是标准 Snake 规则）。
5. 存活且 `steps_since_food ≥ H`（默认 100）→ `truncated`, cause = `starve`。
6. 返回 Gymnasium 五元组 `(obs, reward, terminated, truncated, info)`；`info = {cause, score, length, steps}`。

`steps_since_food` **不进观测**（A、B 都不进）。这意味着饥饿截断对值函数是不可观测的时限，正因如此截断步必须 bootstrap（第 0 节第 3 条）——把截断当终止，等于让网络去拟合一个它看不见的变量。[INFERRED, HIGH；Pardo et al. 2018 讨论的正是这一类时限问题, KNOWN, MED]

`H = 100` 在 10×10 下不会误杀 Hamiltonian 玩家：沿 100 格回路到食物最远 99 步。[COMPUTED] 16×16 下要放大到 ≥ 256（回路 256 格），所以 `H` 是配置项，默认 `max(100, W²)`。**[决定]**

### 3.4 观测

- **A（11 bool）**：危险 直/左/右（下一格是墙，或 `body_age > 1`——尾格会腾出所以不算危险，与 3.3 一致）3；朝向 one-hot 4；食物在头的上/下/左/右（绝对坐标系，最多两个为真）4。这是社区经典表示 [COMMON, HIGH]，报告作 related work 引用。
- **B（多平面 float32, C×W×W）**：`[head, body(二值), food, body_age/length, heading×4 广播]` = 8 平面。配置开关可去掉后 5 个平面得到 3 平面版 → 这就是第 0 节第 2 条的"非 Markov 消融"（年龄平面是使状态 Markov 的那一个；方向平面冗余，保留只为让 CNN 少学一步）。

### 3.5 奖励（`rewards.py`，共用）

| 变体 | 定义 | 实现注记 |
|---|---|---|
| R0 | 吃 +1，死 −1，其余 0 | 饥饿截断 **不给 −1**（它不是死亡，是截断） **[决定]** |
| R1 | R0 − 0.01/步 | |
| R2 | R0 + γΦ(s′) − Φ(s)，Φ(s) = −曼哈顿距离/(2(W−1))，Φ(终止) = 0 | 吃到后 s′ 的 Φ 用新食物算，仍是合法势函数；Ng et al. 1999 的最优策略不变性成立 [KNOWN, HIGH] |
| R3 | R0 ± 0.01（靠近/远离） | 每步曼哈顿距离必变 ±1，故非吃步永远 ±0.01。**注意**：对称 ±c 恰好等于 Φ(s)=−c·d 的势函数塑形取 γ=1，与 R2 形式只差 (1−γ)Φ(s′)≈1e-4/步 [COMPUTED]，几乎不可能产生 reward hacking。要做"可被钻空子"的对照，应改为**非对称 R3′：靠近 +0.01，远离 0**（来回横跳每两步净得 +0.01，bootstrap 下无限循环价值 ≈ 0.5）。待拍板，见 `docs/PROGRESS.md` |

测试用的不变量：一局内 `Σ γ^t F_t = γ^T Φ(s_T) − Φ(s_0) = −Φ(s_0)`（终止 Φ=0）。[COMPUTED] 注意这是**折扣**和，Part B 写的"telescopes to R0 plus a constant"在 γ<1 时要按这个式子验。

### 3.6 向量化环境

- 同一语义作用在 `(N,W,W)` 数组上；`body_age` 的递减/置头/碰撞全是掩码运算。
- 食物重生：给每格采一个随机键，占用格置 +∞，`argmin` → 均匀分布且完全向量化。**标量环境也用同一套采样协议**，否则等价性测试过不了。
- 自动重置：done 的环境立即 reset；`step` 返回的 `obs` 是新局首观测（未结束的环境即后继观测）。`info = {final_obs, done, cause, score, length, steps}`，均带前导 N 轴：`final_obs` 是本步转移的**真实后继观测**（重置前快照；replay buffer 必须存它作 `next_obs`，否则饥饿截断步无法 bootstrap，违反第 0 节第 3 条）；`done = terminated | truncated`；`cause` 是 `common.CAUSE_*` 整数 id（标量环境给字符串，用 `CAUSE_NAMES` 映射）；`score / length / steps` 是重置前的值（仅在 `done` 处是整局总量）。`reset()` 只返回观测，不带 info。（2026-10-04 M3 更新：原文写的是 `done_mask / final_score / final_cause`，且漏了 `final_obs`。）
- RNG：每个环境一个 `np.random.Generator`（种子 = base_seed + i）。推论：两个 base_seed 相差小于 N 的向量化环境共用大部分生成器，所以不同 run 的 base_seed 必须至少相隔 N（M5 的 `train.py` 负责）。

### 3.7 测试清单（`tests/`，M1 验收）

Part B 承诺的全部 + 追尾合法性 + `H` 恰好触发 + R2 折扣望远镜不变量 + 观测 A 的危险位与 3.3 规则一致 + 同网格不同朝向 → 观测不同（Markov 检查）+ 标量/向量化在同种子同动作序列下逐步等价。

### 3.8 非学习基线（`baselines.py`）

- random；
- **BFS 贪心**（无权网格上 BFS = A*，报告里别叫 A* 以免被问启发函数；找不到路时任选安全动作，找不到安全动作则直行）；
- **Hamiltonian**：偶数 W 的固定回路（蛇形扫描 + 回程列）；不做捷径优化。

---

## 4. 里程碑（顺序即依赖；每条带验收标准）

| # | 内容 | 验收 |
|---|---|---|
| M0 | 仓库骨架、requirements、README、pytest 空跑 | 你本地 `pip install -r requirements.txt` 通过，`torch.cuda.is_available()==True` |
| M1 | `core.py` + `obs.py` + `rewards.py` + 测试 | 3.7 全绿 |
| M2 | 基线三件套 + `evaluate.py` | 10×10 上三条基线的分数表（100 局、死因分解）；Hamiltonian 近满分，BFS 主要死于自困 |
| M3 | `vec_env.py` + 等价性测试 + 吞吐基准 | 等价测试绿；报出 env-only steps/s（采集/评估速度的上界；每个 run 的时间预算由梯度更新次数决定，M3 实测见 `docs/PROGRESS.md`） |
| M4 | 表格 Q-learning（观测 A，8×8 → 10×10） | 数分钟内明显超过 random；给出 ε/α/γ 起点 |
| M5 | DQN 核心（MLP on A）+ 评估协议 + 日志/checkpoint | 首条评估曲线；DQN-A ≈ 表格 Q 同特征水平 |
| M6 | CNN on B（8 平面 / 3 平面）；±replay ±target 消融 | 四条消融曲线；B 有/无方向信息对比 |
| M7 | Double DQN + 过估计测量（max-Q 均值 vs 实际回报）；Dueling；PER（最后） | 过估计曲线出图 |
| M8 | 奖励轴 R0–R3 + 行为分析（死因随训练演变、访问热力图、hacking 案例） | |
| M9 | pygame GUI 三模式 | 人玩 + agent 玩带 Q 值侧栏可演示 |
| M10 | 矩阵批量（5 种子）、`analysis/` 出全部图表、Colab notebook | |

砍单顺序（时间不够时从后往前）：PER → Dueling → 超参敏感性 → 动作空间 3 vs 4。种子数不砍。

Part C 最小产出 = M2 + M4 + M5：一张表（random / BFS / Hamiltonian / 表格 Q / DQN-A，均值 ± CI，死因）+ 一条评估曲线 + 仓库链接。Part D 最小产出 = M9 的前两个模式。

---

## 5. 需要你拍板的项（不回复即按默认）

1. 绝对模式反向动作：默认"视为直行"。
2. 追尾进入腾出的尾格：默认"合法"。
3. 初始长度 3、居中朝右；饥饿上限 `max(100, W²)`。
4. 仓库名 `snake-rl`；建好后把 URL 给我。
5. 本地环境：Python 3.13 + `torch` 2.7+ cu128 wheel，先装先验 CUDA。

下一步：M0 + M1（骨架 + 标量环境 + 观测/奖励 + 全部测试），一次交付。
