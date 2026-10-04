# CLAUDE.md — snake-rl 项目指令

本文件由 Claude Code 自动读取。它是项目级规则；`PLAN.md` 是技术契约；`docs/` 是背景。冲突时以本文件 > `PLAN.md` > `docs/` 为准，但 `PLAN.md` §3 的环境契约不得在未更新 `PLAN.md` 的情况下被代码偷偷改掉。

## 这是什么项目

UTS 43008 Reinforcement Learning（Spring 2026）Assignment 3 团队项目，项目号 48，队名 The Deadly Triad（3 人；本仓库由 Jiahao Liu 14191136 主导全部代码）。

题目：**Value-Based Deep Reinforcement Learning on Snake: A Controlled Study of State Representation, Reward Design, and DQN Stabilisation**。

一句话定位：自建 Gymnasium 风格的 Snake 环境，作为**受控实验**的载体——从表格 Q-learning 到 DQN / Double DQN / Dueling / PER，比较状态表示（11 维特征 vs 多平面网格）、奖励设计（R0–R3，含势函数塑形 vs 朴素距离塑形）、稳定化组件消融（±replay ±target），对照非学习基线（random / BFS 贪心 / Hamiltonian 回路）。**项目的分数来自受控实验 + 统计严谨（5 种子 + 置信区间）+ 行为分析，不是"能跑"。** Snake+DQN 教程满网都是；我们的贡献是对比研究本身，经典 11 维特征表示要作为 related work 诚实引用。

## 先读什么

1. `PLAN.md` — 冻结的环境契约（§3）、仓库结构（§2）、里程碑与验收标准（§4）、砍单顺序。**写任何代码前先读 §3。**
2. `docs/PartB-submitted-plan.md` — 已提交给学校的实施计划，是我们对外的承诺；偏离它的地方必须在 `docs/PROGRESS.md` 里记"偏离 Part B：原因"。
3. `docs/ASSESSMENT3-spec.md` — 官方评分要求（Part A–G；代码要以 Colab/IPython notebook + 仓库形式提交；GUI 要求）。
4. `docs/Snake项目方案-设计依据.md` — 每个设计决定的理由和答辩清单。写报告或被问"为什么这样设计"时查它。
5. `docs/PROGRESS.md` — 里程碑进度与交接记录。每次会话开始先看它，结束前更新它。

## 不可协商的规则

- **环境契约冻结**：`PLAN.md` §3 的 `step` 执行顺序、`body_age` 语义、观测 A/B 定义、R0–R3 定义、饥饿截断规则是契约。要改，先改 `PLAN.md` 并在 `PROGRESS.md` 记录，再改代码。
- **truncated ≠ terminated**：饥饿截断（`truncated=True`）的 DQN target 必须 bootstrap；只有撞墙/撞己（`terminated=True`）才把下一状态价值置零。饥饿截断不给 −1。
- **标量环境与向量化环境逐步等价**：`tests/test_vec_env.py` 是回归门禁，任何环境改动后必须过。
- **测试先于功能**：每个里程碑的验收标准（`PLAN.md` §4）就是该里程碑的测试；测试不绿不算完成，不进入下一个里程碑。
- **可复现**：每次训练运行写 `config.yaml`、seed、git hash 到 `runs/<exp>/<seed>/`；所有表和图由 `snake_rl/analysis/` 从日志重生成，不手改数字。`runs/` 不进 git，也永远不删。
- **5 种子不砍**。时间不够砍实验行（顺序：PER → Dueling → 超参敏感性 → 动作空间），不砍种子。
- **reward hacking 不是 bug**：R3 下出现绕圈刷分、R1 下出现"自杀求解脱"之类行为，录像/记录下来作为发现，不要"修掉"。
- **不碰 A1 和已提交的 Part A/B 文档**。

## 硬件与环境

- Windows 11；Python 3.13.7；虚拟环境 `.venv`（仓库根目录）。
- GPU：RTX 5090（Blackwell, sm_120）→ 需要 PyTorch ≥ 2.7 的 CUDA 12.8 wheel（`pip install torch --index-url https://download.pytorch.org/whl/cu128`；以官网当前指引为准）。M0 第一步：`python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` 必须打印 `True`。
- 训练吞吐瓶颈在**梯度更新次数**（每次约 2 ms 的 CPU 侧调用开销，与 batch 大小基本无关），不在环境步进也不在 GPU 算力（M3 实测：默认配置下 `env.step` 只占整环路 <1%，见 `docs/PROGRESS.md` 的"M3 验收"）。墙钟预算 ≈ 更新次数 × 约 2 ms（观测 A）/ 3 ms（B 8 平面）。128–256 个并行环境（B 8 平面用 128）+ 显式的"梯度步数/vec-step"参数（默认 N_envs/4，`PLAN.md` §0.5）。
- 跑测试：`pytest -q`。跑冒烟：`python -m snake_rl.train --config configs/smoke_8x8.yaml`（M5 后存在）。
- 长时间训练（>10 分钟）不要在 Claude Code 的前台跑：生成命令，让用户在独立终端后台跑，之后只读 `runs/` 分析。

## 工作流

1. 会话开始：读 `docs/PROGRESS.md`，确认当前里程碑；`git status` 干净再动手。
2. 一次只做一个里程碑；里程碑内先写测试再写实现。
3. 里程碑完成 = 测试全绿 + `PROGRESS.md` 更新（做了什么、验收结果、偏离 Part B 的地方、下一步）+ 一次 git commit。
4. Commit message 英文，格式 `M<n>: <what>`（如 `M1: scalar env, obs A/B, rewards R0-R3, 23 tests`）。用户负责 push。
5. 不确定的设计决定：先在 `PROGRESS.md` 的"待拍板"里列出默认值和理由，按默认值继续，不要停下来等。

## 代码规范

- 代码、注释、docstring、commit message 用**英文**（仓库要提交给课程协调人）；与用户的对话用**中文**。
- Python 3.11+ 语法，类型标注，`dataclass` 配置，`numpy` 向量化优先于 Python 循环；无全局可变状态。
- 标量环境 `snake_rl/env/core.py` 以**可读性**为先（它是答辩讲解和 GUI 用的参考实现）；向量化环境 `vec_env.py` 以**吞吐**为先。两者共用 `obs.py` / `rewards.py`，观测和奖励只写一遍。
- 网络、replay、agent 分文件；Double / Dueling 是 `agent`/`networks` 的开关，不是复制一份 agent。
- 不引入超出 `requirements.txt` 的依赖；不用 stable-baselines3 之类的现成 RL 库（项目要求自己实现算法）。

## 沟通规范（用户偏好，严格执行）

- 直接、专家口吻、不恭维、不免责声明；先说反面（风险、反证、做不到的地方），再说方案。
- 每个非显然的论断打标签：`[KNOWN]` 训练事实 · `[COMPUTED]` 算出来的 · `[INFERRED]` 推断 · `[COMMON]` 领域常识 · `[GUESS]` 无依据；并给置信度 HIGH ≥80% / MED 50–80% / LOW 20–50%。实验结果引用时给 run 目录路径。
- 不知道就第一行写"我不知道"，不编造引用、数字或 API。
- 发现 `docs/` 或 Part B 里的事实/机制错误，主动指出并记入 `PROGRESS.md`，不要沉默地照做。
- 里程碑汇报只说结果与验收，不复述过程。

## 交付物映射（`docs/ASSESSMENT3-spec.md`）

- Part C（初步实验结果）= M2 + M4 + M5：一张表（random / BFS / Hamiltonian / 表格 Q / DQN-A，均值 ± CI，死因分解）+ 一条评估曲线 + 仓库链接。
- Part D（环境/GUI）= M9 前两个模式（人玩；agent 玩 + 实时 Q 值侧栏）。
- Part E/F/G = M10 全矩阵 + `notebooks/results.ipynb`（Colab 可从日志/checkpoint 重生成主图）+ 行为分析图 + 演示录像。

## 当前状态

见 `docs/PROGRESS.md`。起点：M0（仓库骨架 + requirements + CUDA 验证）→ M1（标量环境 + 观测 + 奖励 + 全部测试）。
