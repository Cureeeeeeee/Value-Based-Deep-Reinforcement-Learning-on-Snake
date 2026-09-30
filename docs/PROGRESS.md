# PROGRESS — 里程碑进度与交接

每次会话：开始先读本文件；结束前更新"当前里程碑"、"会话日志"。里程碑定义与验收标准见 `../PLAN.md` §4。

## 当前里程碑

**M1**（进行中）。M0 已完成（2026-09-30）。

## 里程碑清单

| # | 内容 | 状态 | 验收结果 / 备注 |
|---|---|---|---|
| M0 | 仓库骨架、requirements、README、pytest 空跑、CUDA 验证 | ✅ 2026-09-30 | `.venv` Python 3.13.7；torch 2.11.0+cu128，`cuda.is_available()==True`，RTX 5090 识别为 sm_120（arch list 含 sm_120），GPU matmul 正常。复跑：`python scripts/check_setup.py`。GitHub remote：`git@github.com:Cureeeeeeee/Value-Based-Deep-Reinforcement-Learning-on-Snake.git` |
| M1 | `core.py` + `obs.py` + `rewards.py` + `tests/`（PLAN §3.7 全部） | ☐ | |
| M2 | 基线 random / BFS / Hamiltonian + `evaluate.py` | ☐ | |
| M3 | `vec_env.py` + 等价测试 + 吞吐基准（steps/s） | ☐ | |
| M4 | 表格 Q-learning（观测 A） | ☐ | |
| M5 | DQN 核心（MLP on A）+ 评估协议 + 日志/checkpoint | ☐ | → Part C 可交 |
| M6 | CNN on B（8 平面 / 3 平面）；±replay ±target 消融 | ☐ | |
| M7 | Double DQN + 过估计测量；Dueling；PER | ☐ | PER 最先砍 |
| M8 | 奖励轴 R0–R3 + 行为分析 | ☐ | |
| M9 | pygame GUI 三模式 | ☐ | → Part D |
| M10 | 矩阵批量（5 种子）、`analysis/` 全部图表、Colab notebook | ☐ | → Part E |

## 待拍板（按默认值继续，用户可推翻）

| 项 | 默认 | 理由 |
|---|---|---|
| 绝对动作模式下的反向动作 | 视为直行，不判死 | 判死会让 4 动作对比多出"25% 动作必死"的人为劣势 |
| 追尾进入本步腾出的尾格 | 合法 | 标准 Snake 规则；先递减后判定自然给出 |
| 初始状态 | 长度 3，水平居中，朝右 | 无争议 |
| 饥饿上限 H | `max(100, W²)` | 10×10 下 Hamiltonian 最远 99 步不会被误杀；16×16 需 ≥256 |
| 饥饿截断的奖励 | 不给 −1 | 截断不是死亡；与 bootstrap 规则一致 |
| 观测 B 的 Markov 修复 | 身体年龄平面（方向平面仅冗余） | 二值身体平面丢的是身体顺序，方向平面只修颈不修尾 |
| 满盘（`length == W²`） | `terminated`, cause = `win` | 无空格可放食物；Part B 未定义此终止，2026-09-16 补入契约 |
| R3 的定义 | **建议改为非对称**：靠近 +0.01，远离 0（Part B 写的是 ±0.01） | 对称 ±0.01 ≡ Φ=−0.01·d 的势函数塑形取 γ=1，与 R2 只差 ~1e-4/步，不会出现 hacking；非对称版才有可钻的循环（两步净 +0.01） |

## 偏离 Part B 的记录

| 日期 | 偏离 | 原因 |
|---|---|---|
| 2026-09-15 | 砍单顺序改为 PER → Dueling → 超参 → 动作空间（Part B：动作空间 → 超参 → PER） | 动作空间只是环境开关，成本≈0；PER 实现成本最高、教学价值最低 |
| 2026-09-15 | Markov 修复由"方向平面"改为"身体年龄平面为主" | Part B 的说法不精确：二值身体平面下颈和尾都不可恢复，方向平面只修颈 |
| 2026-09-15 | Python 3.13.7（Part B 写 3.11） | 本地已装 3.13.7，torch ≥2.7 兼容 |

## 会话日志

- 2026-09-15：项目文件夹建立；`PLAN.md`、`CLAUDE.md`、`docs/` 就位。代码未开始。
- 2026-09-30：M0 完成。`git init`（main）+ remote；`.venv`（`py -3.13`）；torch 2.11.0+cu128 装自 `download.pytorch.org/whl/cu128`，其余依赖走 `requirements.txt`（numpy 2.5.3 / gymnasium 1.3.0 / pygame 2.6.1 / pytest 9.1.1）。骨架：`snake_rl/{env,agents/dqn,analysis,gui}`、`tests/`、`configs/`、`scripts/check_setup.py`、`notebooks/`、`pyproject.toml`（pytest `pythonpath=.`）。注意：本机 PATH 上默认 `python` 是 3.11.9，建 venv 必须用 `py -3.13`。
