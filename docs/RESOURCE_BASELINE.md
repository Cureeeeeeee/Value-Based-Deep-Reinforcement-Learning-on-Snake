# RESOURCE_BASELINE — A3 snake-rl 与 resources 代码的关系（2026-10-08 起生效）

> 用户指令（2026-10-08，原话）：**"任务所生成的代码都是以我上传的在 resource 中的代码为基础，如 resource 中的代码可以实现同样的效果就原样采用，尽量少做更改。"**
>
> resources 路径：`../../../resources/weekNN/`（相对 snake-rl 根目录）。只用 Solution 版 notebook。

---

## 0. 先说反面

1. **A3 能直接复用的 resource 代码很少。** resource 的深度 RL 部分是 SB3（W8 Part C、W9 Part C/D、W10）或 Keras + 旧 gym API（W9 Part B）。A3 要做的 ±replay、±target、Dueling、PER、过估计测量，SB3 的 `DQN` 类都表达不了；Keras 版与本仓库的 PyTorch 栈不兼容。按指令的条件（"可以实现同样的效果"）判断，这些部件只能 L3 新写。[INFERRED, HIGH]
2. **resource 的两种 CNN 都不能原样跑在 8×8/10×10 棋盘上。** [COMPUTED]
   - W9 Part D / W10 Part C 的 `CustomCNN`（Nature 结构）：10×10 输入经 8×8/4 卷积后只剩 1×1，下一层 4×4 卷积直接报错。
   - W9 Part A 的 `SimpleCNN`：10 → conv3(pad1) 10 → pool 5 → conv3 3 → pool 1 → 第三层 conv3 报错；它的 `fc1 = Linear(512, 10)` 也是按 28×28 输入硬编码的。
3. **已完成的 M0–M5 代码不按 resource 重写（默认，待用户确认）。** Part C 的数字由这些代码产生，重写会让已提交的结果失去对应的代码；而且 resource 里没有向量化环境、等价门禁、观测 A/B 编码这类东西可以替换。规则只约束 **M6 起新写的代码**。

## 1. 改动分级

同 A2：L0 原样 / L1 只改参数值 / L2 最小补丁（行尾 `# CHANGED: <原因>`）/ L3 新增（`# NEW: no resource equivalent — <why>`）。复用 resource 的代码段时，在函数或类的 docstring 里写明出处，例如 `Adapted from resources/week09/...Part-D...ipynb cell 13 (conv hyperparameters changed for 10x10 boards)`。

## 2. 逐里程碑映射（M6 起）

| 里程碑 / 部件 | resource 出处 | 级别 | 做法 |
|---|---|---|---|
| M6 CNN（观测 B） | W9 Part D cell 13 `CustomCNN` | L2 | 保留类骨架：`nn.Sequential` 卷积栈 → `Flatten` → 用一次 dummy forward 求 `n_flatten`（这段原样）→ `self.linear = Sequential(Linear(n_flatten, features_dim), ReLU)` → Q 头。**只改卷积超参**使之适配小棋盘（默认：3 层 3×3、stride 1、padding 1，通道 32/64/64，无池化），在 docstring 里记 §0 第 2 条的原因。基类不用 SB3 的 `BaseFeaturesExtractor`，改为 `nn.Module`（本仓库不依赖 SB3）。 |
| M6 ±replay ±target 消融 | W9 Part B `SimpleDQN`（`deque` 回放、`target_model` 每局同步） | 概念参照，无代码复用 | 本仓库的 PyTorch agent 已有 replay 和 target；消融开关是 L3。报告里可以把 W9 Part B 的写法当作课程里的参照实现 |
| M7 Double DQN | W8 `double_q_learning` | 已实现，不改 | `agent.py` 第 282 行起已有 `cfg.double`（在线网络选动作、目标网络评估）。它对应 W8 的更新行 `best_next_action = argmax(Q1[s']); TD_target = r + γ·Q2[s', best_next_action]`，报告引用这一对应关系 |
| M7 过估计测量 | 无 | L3 | — |
| M7 Dueling、PER | 无 | L3 | 砍单顺序不变：PER → Dueling |
| M9 GUI | W8 Part C / W9 Part B 的 `run_game`（人类输入 → `env.step` → `render` 的回合循环，人机对战） | 模式参照 | "人玩"和"agent 玩"两种模式的主循环照 `run_game` 的结构组织；pygame 渲染与 Q 值侧栏是 L3 |
| M10 Colab notebook | 各 resource notebook 的安装格；W9 Part C 的 `plot_results` | L1 / 参照 | 安装格照 resource 写法（`!pip install ...`）；主图仍由 `snake_rl/analysis/` 生成（M5 已冻结的 CI 带画法不改） |
| CUDA 检查 | W8/W9 `check_cuda()` | — | M0 已有 `scripts/check_env.py`，不替换 |

## 3. 与本仓库既有规则的关系

- `CLAUDE.md` 的"不用 stable-baselines3 之类的现成 RL 库"保持不变。注意：`docs/ASSESSMENT3-spec.md` 里没有禁止使用库的条款（2026-10-08 grep 核对），这条是项目自己的决定，理由是要做 SB3 表达不了的消融。[COMPUTED]
- `PLAN.md` §3 的环境契约不受影响（resource 没有 Snake 环境）。
