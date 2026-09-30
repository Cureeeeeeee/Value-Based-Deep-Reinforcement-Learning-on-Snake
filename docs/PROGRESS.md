# PROGRESS — 里程碑进度与交接

每次会话：开始先读本文件；结束前更新"当前里程碑"、"会话日志"。里程碑定义与验收标准见 `../PLAN.md` §4。

## 当前里程碑

**M2**（未开始）。M0、M1 已完成（2026-09-30）。下一步：`snake_rl/agents/baselines.py`（random / BFS 贪心 / Hamiltonian）+ `snake_rl/evaluate.py`（100 局、固定评估种子、死因分解）+ `tests/test_baselines.py`。

## 里程碑清单

| # | 内容 | 状态 | 验收结果 / 备注 |
|---|---|---|---|
| M0 | 仓库骨架、requirements、README、pytest 空跑、CUDA 验证 | ✅ 2026-09-30 | `.venv` Python 3.13.7；torch 2.11.0+cu128，`cuda.is_available()==True`，RTX 5090 识别为 sm_120（arch list 含 sm_120），GPU matmul 正常。复跑：`python scripts/check_setup.py`。GitHub remote：`git@github.com:Cureeeeeeee/Value-Based-Deep-Reinforcement-Learning-on-Snake.git` |
| M1 | `core.py` + `obs.py` + `rewards.py` + `tests/`（PLAN §3.7 全部） | ✅ 2026-09-30 | `pytest`：75 passed，0.8 s。覆盖：撞墙/撞己/吃食生长/食物不落蛇身/饥饿恰在第 H 步截断且不给 −1/追尾腾出格合法/age-2 格撞死/满盘 win/绝对模式反向=直行/种子可复现/食物均匀采样/A 的危险位与 `step` 逐动作一致（随机对局 + 手工局面）/A 食物方位与朝向 one-hot/B 8·7·4·3 平面形状与内容/同网格不同朝向观测不同（Part B 测试）/**同网格同朝向不同尾：仅年龄平面可分辨，且后继状态确实不同**（PLAN §0.2 的 Markov 论证）/R0–R3 六类事件奖励/R2 折扣望远镜不变量（终止与截断两类对局都验）/R3 非对称默认 + 对称开关 + 来回横跳净 +0.02/批量 `compute_reward` 与标量一致/gymnasium `check_env` A、B 均过 |
| M2 | 基线 random / BFS / Hamiltonian + `evaluate.py` | ☐ | |
| M3 | `vec_env.py` + 等价测试 + 吞吐基准（steps/s） | ☐ | |
| M4 | 表格 Q-learning（观测 A） | ☐ | |
| M5 | DQN 核心（MLP on A）+ 评估协议 + 日志/checkpoint | ☐ | → Part C 可交 |
| M6 | CNN on B（8 平面 / 3 平面）；±replay ±target 消融 | ☐ | |
| M7 | Double DQN + 过估计测量；Dueling；PER | ☐ | PER 最先砍 |
| M8 | 奖励轴 R0–R3 + 行为分析 | ☐ | |
| M9 | pygame GUI 三模式 | ☐ | → Part D |
| M10 | 矩阵批量（5 种子）、`analysis/` 全部图表、Colab notebook | ☐ | → Part E |

## M1 实现层面的决定（PLAN §3 之外，代码即文档）

| 项 | 决定 | 影响 |
|---|---|---|
| 共享规则模块 | 新增 `snake_rl/env/common.py`（朝向/动作常量、`next_heading`、`sample_food`、`body_age_from_cells` 等），`core.py` 与将来的 `vec_env.py` 都从它取规则 | PLAN §2 目录多一个文件；观测/奖励/规则都只写一遍 |
| 观测编码器一律批量 | `obs.encode(cfg, body_age[N,W,W], heading[N], food[N,2], length[N])`；标量环境以 N=1 调用后取 `[0]` | M3 向量化环境零改动复用；`tests/test_obs.py` 已验证批量 == 逐环境 |
| 观测 B 的开关 | `ObsConfig(kind="B", age_plane=True, heading_planes=True)` → 8/7/4/3 平面 | M6 消融：8 平面（完整）vs 7 平面（无年龄，非 Markov）vs 3 平面 |
| R3 默认非对称 | `RewardConfig(closer=0.01, farther=0.0)`；`farther=-0.01` 还原 Part B 的对称版 | 见"偏离 Part B" |
| R3 塑形的判定基准 | 以本步开始时的食物位置算"靠近/远离"；吃食步与死亡步也加塑形项（吃食步 = 1.01） | 每个非吃步恒为 ±c，测试可精确断言 |
| 死亡步的终止观测 | 撞墙：网格不变；撞己：尾已腾出、头未放置 | 终止观测不参与 bootstrap，无学习影响；GUI 画"撞击"时需自己用 `heading` 推撞击格 |
| `info` | 严格 `{cause, score, length, steps}`；`steps_since_food` 作为环境属性暴露给 GUI，不进 info | 与契约一致 |
| `SnakeEnv.set_state(cells, food, heading=None, steps_since_food=0)` | 测试与 GUI 回放用的直接置盘接口；朝向默认由颈→头推断 | 全部手工局面测试靠它 |
| Gymnasium 注册 | `import snake_rl.env` 注册 `Snake-v0`（`gym.make("Snake-v0", cfg=EnvConfig(...))`） | 与课程工具链一致 |
| RNG | 环境自持 `np.random.default_rng(seed)`；`reset(seed=None)` 延续同一流 | M3 每个向量化子环境 `default_rng(base_seed+i)` 即可逐步等价 |

## 待拍板（按默认值继续，用户可推翻）

| 项 | 默认 | 理由 |
|---|---|---|
| 绝对动作模式下的反向动作 | 视为直行，不判死（已实现） | 判死会让 4 动作对比多出"25% 动作必死"的人为劣势 |
| 追尾进入本步腾出的尾格 | 合法（已实现） | 标准 Snake 规则；先递减后判定自然给出 |
| 初始状态 | 长度 3，水平居中，朝右（已实现：头在 `(W//2, W//2)`，身体向左） | 无争议 |
| 饥饿上限 H | `max(100, W²)`（已实现） | 10×10 下 Hamiltonian 最远 99 步不会被误杀；16×16 需 ≥256 |
| 饥饿截断的奖励 | 不给 −1（已实现，四个变体都测过） | 截断不是死亡；与 bootstrap 规则一致 |
| 观测 B 的 Markov 修复 | 身体年龄平面（方向平面仅冗余）（已实现） | 二值身体平面丢的是身体顺序，方向平面只修颈不修尾 |
| 满盘（`length == W²`） | `terminated`, cause = `win`（已实现） | 无空格可放食物；Part B 未定义此终止，2026-09-16 补入契约 |
| R3 的定义 | **非对称**：靠近 +0.01，远离 0（已按此实现为默认；`farther=-0.01` 还原 Part B） | 对称 ±0.01 ≡ Φ=−0.01·d 的势函数塑形取 γ=1，与 R2 只差 ~1e-4/步，不会出现 hacking；非对称版才有可钻的循环（两步净 +0.01） |

## 偏离 Part B 的记录

| 日期 | 偏离 | 原因 |
|---|---|---|
| 2026-09-15 | 砍单顺序改为 PER → Dueling → 超参 → 动作空间（Part B：动作空间 → 超参 → PER） | 动作空间只是环境开关，成本≈0；PER 实现成本最高、教学价值最低 |
| 2026-09-15 | Markov 修复由"方向平面"改为"身体年龄平面为主" | Part B 的说法不精确：二值身体平面下颈和尾都不可恢复，方向平面只修颈 |
| 2026-09-15 | Python 3.13.7（Part B 写 3.11） | 本地已装 3.13.7，torch ≥2.7 兼容 |
| 2026-09-30 | R3 默认改为非对称（靠近 +0.01，远离 0）；Part B 的 ±0.01 保留为 `RewardConfig(farther=-0.01)` | 对称版等价于 γ=1 的势函数塑形，理论上不可 hack，达不到 Part B 写它的目的（"deliberately to expose reward hacking"）。Part E 里两版都跑并解释 |
| 2026-09-30 | 新增 `win` 终止（满盘）| Part B 只列了 wall / self / starvation；满盘时无法刷食物，必须终止 |

## 会话日志

- 2026-09-15：项目文件夹建立；`PLAN.md`、`CLAUDE.md`、`docs/` 就位。代码未开始。
- 2026-09-30：M0 完成。`git init`（main）+ remote；`.venv`（`py -3.13`）；torch 2.11.0+cu128 装自 `download.pytorch.org/whl/cu128`，其余依赖走 `requirements.txt`（numpy 2.5.3 / gymnasium 1.3.0 / pygame 2.6.1 / pytest 9.1.1）。骨架：`snake_rl/{env,agents/dqn,analysis,gui}`、`tests/`、`configs/`、`scripts/check_setup.py`、`notebooks/`、`pyproject.toml`（pytest `pythonpath=.`）。注意：本机 PATH 上默认 `python` 是 3.11.9，建 venv 必须用 `py -3.13`。
- 2026-09-30：M1 完成。`snake_rl/env/{common,core,obs,rewards}.py` + `tests/{helpers,test_env,test_obs,test_rewards}.py`，75 测试全绿。实现决定见上表。未 push（SSH 从 Claude Code 的 shell 到 GitHub 报 publickey 拒绝，push 由用户在自己终端做）。
