# PROGRESS — 里程碑进度与交接

每次会话：开始先读本文件；结束前更新"当前里程碑"、"会话日志"。里程碑定义与验收标准见 `../PLAN.md` §4。

## 当前里程碑

**M4**（未开始）。M0–M2 完成于 2026-09-30，M3 完成于 2026-10-04。下一步：`snake_rl/agents/tabular_q.py`（观测 A 的 11 位特征 → 2¹¹ 状态 × 3 动作的 Q 表，8×8 → 10×10），验收 = 数分钟内明显超过 random，给出 ε/α/γ 起点。之后 M5（DQN-A）→ **Part C 可交**。Part C 截止：2026-10-07 前后（用户 2026-10-04 说"还有 3 天"）。

## 里程碑清单

| # | 内容 | 状态 | 验收结果 / 备注 |
|---|---|---|---|
| M0 | 仓库骨架、requirements、README、pytest 空跑、CUDA 验证 | ✅ 2026-09-30 | **可复现性数字（进报告）**：Windows 11 10.0.26200；Python 3.13.7（`.venv`）；NVIDIA driver 616.92；torch 2.11.0+cu128（`torch.version.cuda` 12.8，装自 `download.pytorch.org/whl/cu128`，PyPI 的 Windows torch 是 CPU 版）；`cuda.is_available()==True`；设备 NVIDIA GeForce RTX 5090，compute capability (12, 0) = sm_120，arch list `[sm_75, sm_80, sm_86, sm_90, sm_100, sm_120]`；1024×1024 fp32 matmul 0.060 ms/op ≈ 35.6 TFLOP/s；numpy 2.5.3 / gymnasium 1.3.0 / pygame 2.6.1（cp313 wheel 存在，无需 pygame-ce）/ pytest 9.1.1。复跑：`python scripts/check_env.py`（任一项失败 exit 1）。`pip install -e .` 已装（snake-rl 0.1.0）。`tests/test_smoke.py` 导入全部 PLAN §2 模块（未实现的模块是只含 docstring 的桩）。GitHub remote：`https://github.com/Cureeeeeeee/Value-Based-Deep-Reinforcement-Learning-on-Snake.git`（HTTPS；本机 SSH key 未注册到 GitHub，见会话日志） |
| M1 | `core.py` + `obs.py` + `rewards.py` + `tests/`（PLAN §3.7 全部） | ✅ 2026-09-30 | `pytest`：75 passed，0.8 s。覆盖：撞墙/撞己/吃食生长/食物不落蛇身/饥饿恰在第 H 步截断且不给 −1/追尾腾出格合法/age-2 格撞死/满盘 win/绝对模式反向=直行/种子可复现/食物均匀采样/A 的危险位与 `step` 逐动作一致（随机对局 + 手工局面）/A 食物方位与朝向 one-hot/B 8·7·4·3 平面形状与内容/同网格不同朝向观测不同（Part B 测试）/**同网格同朝向不同尾：仅年龄平面可分辨，且后继状态确实不同**（PLAN §0.2 的 Markov 论证）/R0–R3 六类事件奖励/R2 折扣望远镜不变量（终止与截断两类对局都验）/R3 非对称默认 + 对称开关 + 来回横跳净 +0.02/批量 `compute_reward` 与标量一致/gymnasium `check_env` A、B 均过 |
| M2 | 基线 random / BFS / Hamiltonian + `evaluate.py` | ✅ 2026-09-30 | `pytest`：127 passed（+30）。验收表见下（10×10、100 局、评估种子 10000+i，`runs/baselines/*.json`）：Hamiltonian 100/100 局 `win`、97 分满分；BFS 30.6 ± 1.6，84% 撞己、16% 撞墙（被困后直行撞墙）、0 饿死；random 0.18。复跑：`python -m snake_rl.evaluate --W 10 --episodes 100`（约 30 s，Hamiltonian 占 24 s） |
| M3 | `vec_env.py` + 等价测试 + 吞吐基准（steps/s） | ✅ 2026-10-04 | `pytest`：224 passed，9.1 s（`tests/test_vec_env.py` 97 项）。等价门禁：19 组配置逐步逐位比较，58,580 次转移（wall 1597 / self 86 / starve 1362 / win 87，吃食 5275，自动重置 3132）。吞吐（env-only，10×10，随机动作，`runs/bench/vec_env_1.json`，git `b60a2e6`）：观测 A 在 N=128 为 622 k steps/s（标量 14 k 的 44.6×），N=256 为 1.07 M。详见下方"M3 验收"。复跑：`python scripts/bench_vec_env.py`（约 80 s） |
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

## M2 验收表（10×10，100 局，评估种子 10000+i；由 `python -m snake_rl.evaluate` 生成，勿手改）

| policy | episodes | score mean ± 95% CI | median | max | mean steps | wall | self | starve | win |
|---|---|---|---|---|---|---|---|---|---|
| random | 100 | 0.18 ± 0.08 | 0.0 | 2 | 23.0 | 0.98 | 0.01 | 0.01 | 0.00 |
| bfs | 100 | 30.61 ± 1.60 | 31.0 | 58 | 255.8 | 0.16 | 0.84 | 0.00 | 0.00 |
| hamiltonian | 100 | 97.00 ± 0.00 | 97.0 | 97 | 2409.7 | 0.00 | 0.00 | 0.00 | 1.00 |

解读：Hamiltonian 是分数上界（满盘即 `win`，永不饿死因为 H ≥ W²）但每局 2410 步，是速度下界；BFS 贪心中位数 31、最高 58，死因以自困为主——"撞墙"的 16% 也是被困后无安全动作直行撞墙，语义上同属自困。RL agent 的坐标系：random ≪ 表格 Q ≈ DQN-A ≤ ? ≤ BFS < Hamiltonian。

## M2 实现层面的决定

| 项 | 决定 | 影响 |
|---|---|---|
| 策略接口 | `snake_rl.agents.Policy`：`act(obs, env) -> action`，`reset()`，`name` | 基线读 `env` 全状态，学习型 agent 只读 `obs`；同一个 `evaluate()` 通吃 |
| BFS 时间感知 | 年龄 k 的身体格在深度 d 到达时可通过当 k ≤ d（尾巴逐步让开；到食物前不会吃到东西，所以精确） | 头被围时会沿自己尾巴逃出（测试 `test_bfs_uses_vacating_tail_to_escape_pocket`）；仍不做防自困，符合 PLAN §3.8 "贪心" |
| BFS 无路时 | 直/左/右第一个安全动作；都不安全 → 直行 | 确定性；被困后撞墙记为 `wall` |
| Hamiltonian 方向 | 每步走"不是脖子的那个回路邻居"，回路两个方向都可走 | W ≡ 2 (mod 4) 时初始朝向与回路方向相反（10×10 就是），自动反向走，不需要改初始状态 |
| 评估种子 | 第 i 局 `reset(seed=eval_seed+i)`，默认 `eval_seed=10000`（远离训练种子） | 所有 agent 与基线在同一组棋盘上比较 |
| 评估输出 | `EvalResult` → JSON：env config、seed、git hash（`+dirty` 标记）、逐局记录、summary（均值、95% CI 正态近似、中位数、最大、步数、死因比例） | 表格由文件重生成；跨种子 CI 留给 M10 `analysis/` |
| `snake_rl/utils.py` | `git_hash()` | 训练日志（M5）复用 |

## M3 验收

**等价性**（`tests/test_vec_env.py`，回归门禁）。子环境 i 与"`reset(seed=seed+i)`、喂同样动作、每局结束后无种子 `reset()`"的标量 `SnakeEnv` 逐步逐位相等：`obs`、`info["final_obs"]`、reward、terminated/truncated、cause/score/length/steps、7 个状态数组。19 组配置覆盖观测 A/B8/B7/B4/B3、R0/R1/R2/R3/R3 对称、相对/绝对动作、W∈{4,6,8,10}、默认与小饥饿上限、random/BFS/Hamiltonian/survive/mixed 策略；另有覆盖度断言（四种死因都必须真的出现过，否则测试失败）。

测试有效性的证据（均为 [COMPUTED]，脚本在会话临时目录，不在仓库）：

- 测试与实现由两个 agent 按同一份规格独立编写，首次合并 219/219 通过，无规格分歧。
- 对真实 `vec_env.py` 共 105 个单行变异（两轮 59 + 46）：87 个被抓住；18 个存活，其中 5 个是等价变异，其余是 API 表面的缺口——值得补的已补测试（`reset(seed=0)`、`seed` 属性、`head` 属性、`set_state` 的下标与异常安全、R2 下的饥饿截断、`obs` 与 `final_obs` 不共享内存），补完后 9 个对应变异全部被抓住。
- 随机化差分 fuzz：1149 组随机配置（W 4–16、全部观测/奖励变体、随机奖励参数、`set_state` 注入 57,104 次、中途 reset）共 15,780,028 次标量转移，0 分歧（比较内容含每个子环境的 `bit_generator.state`）。
- M5 数据通路探针：678,400 条存入 replay 的转移逐条对照标量环境——12,300 条饥饿截断全部 `terminated=False` 且 `next_obs` 是真实后继；68,951 条 wall/self/win 全部 `terminated=True`。若错存 `obs` 而不是 `final_obs`，81,251 条 done 转移里会有 79,713 条是错的。

**吞吐**（env-only；10×10，R0，相对动作；5 遍 × 每格 1 s，空闲机器；`runs/bench/vec_env_1.json`，git `b60a2e6`；千 env-steps/s）：

| obs | policy | scalar | N=1 | N=16 | N=64 | N=128 | N=256 | N=512 | N=1024 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | random | 14.0 | 9.9 | 114.9 | 357.0 | 622.5 | 1,067.0 | 1,625.7 | 2,324.9 |
| A | safe | 14.2 | 10.0 | 141.6 | 447.9 | 711.3 | 1,127.5 | 1,726.2 | 2,442.7 |
| B8 | random | 30.4 | 15.9 | 179.8 | 500.3 | 712.6 | 625.5 | 825.0 | 765.5 |
| B8 | safe | 28.5 | 15.2 | 207.6 | 589.6 | 854.4 | 684.2 | 883.3 | 820.8 |
| B3 | random | 35.9 | 17.2 | 212.5 | 621.9 | 976.2 | 1,476.2 | 1,431.6 | 1,490.8 |
| B3 | safe | 34.9 | 17.2 | 233.1 | 736.1 | 1,113.7 | 1,648.1 | 1,606.4 | 1,553.9 |

`random` = 均匀随机动作（局长约 20 步，重置密集，训练早期）；`safe` = 在不立即致死的动作里均匀随机（局长约 140 步，接近训练中期）。两种策略每 1000 步都只吃约 8 个食物，**没有测"会吃"的策略**；探针显示吃食密集时 N=256 再慢 7–23%（每个吃食的子环境要单独从自己的 Generator 抽一次）。

解读：

- 观测 A 一直涨到 N=1024；**B8 在 N=128 处到顶**（每步分配两个整批观测数组，0.5–1 MB 以上的块 Windows 堆每次重新换页；机制为 [INFERRED, MED]）。实际设置：B8 用 N=128，B3 ≤ 256，A 任意。
- `VecSnakeEnv(n_envs=1)` 只有标量环境的 0.5–0.7×：单环境评估和 GUI 继续用 `SnakeEnv`。
- **这张表不是训练时间预算。** 整个 DQN 环路里环境只占 <1%，见下节。

**整环路时间分配**（审查阶段的一次性探针；脚本在会话临时目录，**不可从仓库复现，M5 冒烟训练时必须用 `train.py` 重测**）。条件：观测 A，MLP 2×256 在 RTX 5090，batch 256，numpy replay，空闲机器：

| n_envs | 每 vec-step 梯度步数 G | env.step | 单次更新 | 整环路 env-steps/s | 环境占比 | 更新占比 | 每 2M env-steps |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 16（=N/4） | 284 µs | 2.09 ms | 1,879 | 0.8% | 98.1% | 17.7 min |
| 128 | 32（=N/4） | 332 µs | 2.08 ms | 1,902 | 0.5% | 99.0% | 17.5 min |
| 256 | 64（=N/4） | 390 µs | 2.00 ms | 1,991 | 0.3% | 99.4% | 16.7 min |
| 128 | 1 | 329 µs | 2.18 ms | 44,265 | 11.4% | 75.5% | 0.75 min |
| 128（B8，小 CNN） | 32 | 324 µs | 3.00 ms | 1,322 | 0.3% | 99.1% | 25.2 min |

- 预算公式：**墙钟 ≈ 梯度更新次数 × 约 2 ms（A）/ 3 ms（B8）**。单次更新耗时在 batch 64–1024 之间基本不变（是 CPU 侧的每次调用开销，不是 GPU 算力）。
- 固定 UTD 时，增大 `n_envs` 不省墙钟（1.88k / 1.90k / 1.99k env-steps/s）；`n_envs` 按去相关来选。
- 5 个种子作为 5 个并行进程共用一块 GPU：总吞吐只有 2.0×，不是 5×。默认配置下 5 种子 × 2M env-steps ≈ 44 分钟（并行）/ 88 分钟（串行）。
- 所有正式训练都超过 `CLAUDE.md` 的 10 分钟前台上限，必须由用户在独立终端跑。

## M3 实现层面的决定

| 项 | 决定 | 影响 |
|---|---|---|
| 自动重置接口（**PLAN §3.6 已同步更新**） | `step()` 返回 `(obs, reward, terminated, truncated, info)`；`obs` = 新局首观测（done 处）；`info = {final_obs, done, cause, score, length, steps}`，其中 `final_obs` 是重置前的真实后继观测 | replay 的 `next_obs` **必须存 `info["final_obs"]`**，bootstrap 掩码**只用 `terminated`**（不用 `done`）。PLAN 原文的 `done_mask / final_score / final_cause` 对应 `info["done"]`、`info["score"][done]`、`info["cause"][done]` |
| `obs` 与 `final_obs` | 永远是两个独立数组（每步多一次整批拷贝） | 调用方原地改 `obs`（归一化、`torch.from_numpy` 后原地运算）不会污染要进 buffer 的 `next_obs`。代价没有做对照测量；整环路里环境只占 <1%，可忽略 |
| `reset()` | 只返回观测，不返回 `(obs, info)` | 与标量环境 / Gymnasium 不同；从标量代码移植时注意 |
| `info["cause"]` | 整数 id（`common.CAUSE_*`） | 标量环境给的是字符串；共用的 episode logger 要经 `CAUSE_NAMES` 归一 |
| 食物采样 | 需要抽食物的子环境（吃食者、重置者）各自 `rng_i.random(out=row)`，掩码与 argmin 批量做 | 与 `common.sample_food` 抽数完全一致（fuzz 比较了生成器状态）；这是 `step()` 里唯一的 Python 循环 |
| 头位置缓存 | 私有 `_head`，`check_invariants()` 校验它与 `body_age` 一致；公开只读属性 `head` 返回拷贝 | **不要直接写 `body_age`**（会与缓存脱节，`debug=False` 时静默出错）；GUI 回放一律走 `set_state` |
| `check_invariants()` / `debug=True` | 年龄恰为 1..length 各一次、相邻年龄四连通、食物在界内空格、朝向 0..3、`0 ≤ steps_since_food < H`、计数非负、头缓存一致；`debug=True` 时每次 `reset`/`step` 后自动检查 | 兑现 Part B 的 "invariant assertions in debug mode on every step"（M1 的标量环境没有做这一项） |
| `set_state(i, ...)` | 比标量版严格：越界格子/食物、空蛇、非法朝向 → `ValueError`；下标越界 → `IndexError`；全部校验先于任何写入 | 仅测试与 GUI 用，不在训练路径上 |
| 动作输入 | 形状必须是 `(N,)`、整数 dtype、值在范围内，否则 `ValueError`，且校验先于任何状态/RNG 变化 | CUDA tensor 要先 `.cpu().numpy()`；float/bool dtype 被拒绝 |
| `scripts/bench_vec_env.py` | 新增（PLAN §2 目录多一个脚本）；绝不覆盖 `runs/` 下已有文件，重名自动加数字后缀 | `runs/bench/vec_env.json` 是提交前测的（git 标记 `+dirty`），**只引用 `vec_env_1.json`** |

M5 调用方要自己负责的事（审查探针的结论，写 M5 时照做）：

- **种子间隔**：子环境 i 用 `seed + i`，所以 base seed 相差小于 `n_envs` 的两个 run 共用绝大部分食物流。`train.py` 用 `env_seed = 100_000 × (run_seed + 1)` 并断言 `n_envs ≤ 100_000`（同时避开评估种子块 10000–10099）；写一个测试断言 5 个 run 的种子区间两两不相交。
- **向量化评估**（单个审查 agent 的 [COMPUTED] 结论，未经第二个 agent 复核，M5 写评估时自己验证）：`VecSnakeEnv(cfg, n_envs=K, seed=eval_seed)`，只取每个子环境的**第一局**，与标量 `evaluate()` 逐局相同（100/100）且快约 20 倍。"取最先结束的 K 局"会偏向短局（实测低估 3.45 分），不能用。
- **episode return**（同上，未复核）：`info` 不带回报；R1/R2/R3 下回报不能由 score 推出，训练环路自己维护一个 `(N,)` 累加器，在 done 处清零。
- **配置校验放在 M5 的 config loader**：`starve_limit ≥ 1`（`starve_limit ≤ 0` 时 `debug=True` 会在 `reset()` 报错，而标量环境和 `debug=False` 会静默接受）；`seed`、`n_envs` 必须是整数（现在 `int()` 会把 3.9 截成 3）；奖励变体名拼错要在构造时就报错（现在要到第一次 `step` 才报）。

## 待拍板（默认值已按此实现或将按此实现；未收到反对即视为定案）

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
| （M5）每 vec-step 的梯度步数 G 与 batch | **已拍板（用户，2026-10-04）**：M5 冒烟时 `G = N_envs/4`、batch 256 与 `G = 8`、batch 1024 各跑一条学习曲线，再定正式训练用哪个；定下来之前默认仍是 PLAN §0.5 的前者 | 备选 `G = 8`、batch 1024：每 vec-step 采样的转移数相同，探针实测快 3.5–3.9×（单次更新耗时几乎不随 batch 变），但改变了优化动态（更新次数少 4 倍）。选了哪个、依据哪两条曲线，都要写进报告 |
| （M5）优化器 | `Adam(fused=True)` | 探针实测单次更新 −24%（1576 → 1204 µs），数值上等价于普通 Adam |
| （M5）`n_envs` | 观测 A 用 128，B8 用 128 | 固定 UTD 时 `n_envs` 不影响墙钟；B8 在 N>128 时 env-only 吞吐反而下降 |
| CI | **未做**。Part B 写了 "the vectorised/single-environment equivalence test runs in CI" | 需要一个 GitHub Actions workflow（CPU，装 numpy/gymnasium/pytest 即可跑 `tests/`）。推送 workflow 文件需要 token 带 `workflow` 权限。默认：Part C 交完后再加；不加就要在"偏离 Part B"里记一条 |
| `CLAUDE.md` 第 36 行的"瓶颈在环境步进" | **已改（用户 2026-10-04 批准）**：改为"瓶颈在梯度更新次数，不在环境步进也不在 GPU 算力"，并写入预算公式 | 见"偏离 Part B 的记录"2026-10-04 行 |
| `设计依据.md` 第 78 行的同一说法 | **未改**（用户的批准只涉及 `CLAUDE.md`）。写报告引用该文档 §6 时以本文件"M3 验收"的实测为准 | 该文档是背景材料，更正已记在本文件 |

## 偏离 Part B 的记录

| 日期 | 偏离 | 原因 |
|---|---|---|
| 2026-09-15 | 砍单顺序改为 PER → Dueling → 超参 → 动作空间（Part B：动作空间 → 超参 → PER） | 动作空间只是环境开关，成本≈0；PER 实现成本最高、教学价值最低 |
| 2026-09-15 | Markov 修复由"方向平面"改为"身体年龄平面为主" | Part B 的说法不精确：二值身体平面下颈和尾都不可恢复，方向平面只修颈 |
| 2026-09-15 | Python 3.13.7（Part B 写 3.11） | 本地已装 3.13.7，torch ≥2.7 兼容 |
| 2026-09-30 | R3 默认改为非对称（靠近 +0.01，远离 0）；Part B 的 ±0.01 保留为 `RewardConfig(farther=-0.01)` | 对称版等价于 γ=1 的势函数塑形，理论上不可 hack，达不到 Part B 写它的目的（"deliberately to expose reward hacking"）。Part E 里两版都跑并解释 |
| 2026-09-30 | 新增 `win` 终止（满盘）| Part B 只列了 wall / self / starvation；满盘时无法刷食物，必须终止 |
| 2026-10-04 | **事实更正**：Part B 第 38 行 "throughput of the Python step loop is the real bottleneck, not the GPU"（`CLAUDE.md` 第 36 行、`设计依据.md` 第 78 行、`PLAN.md` §4 M3 行同此说法）**不成立** | M3 实测：向量化后在默认 `G = N/4` 下 `env.step` 只占整环路 0.3–0.8%，梯度更新占 98–99%；即使 `G = 1` 环境也只占 11%。后半句"不在 GPU"成立——单次更新约 2 ms 且与 batch 大小基本无关，是 CPU 侧的调用开销。准确说法：**瓶颈是梯度更新次数**。`PLAN.md` §4 与 `CLAUDE.md` 第 36 行已改；Part B 是已提交文档不动，Part E 报告里按实测写 |
| 2026-10-04 | 向量化环境的自动重置接口多了 `info["final_obs"]`，字段名与 PLAN §3.6 原文不同 | 原文只说"返回新局首观测"，照做会丢掉饥饿截断步的真实后继，DQN 无法对截断步 bootstrap。PLAN §3.6 已同步更新；Part B 没有写到这一层，不算偏离其承诺 |

## 会话日志

- 2026-09-15：项目文件夹建立；`PLAN.md`、`CLAUDE.md`、`docs/` 就位。代码未开始。
- 2026-09-30：M0 完成。`git init`（main）+ remote；`.venv`（`py -3.13`）；torch 2.11.0+cu128 装自 `download.pytorch.org/whl/cu128`，其余依赖走 `requirements.txt`（numpy 2.5.3 / gymnasium 1.3.0 / pygame 2.6.1 / pytest 9.1.1）。骨架：`snake_rl/{env,agents/dqn,analysis,gui}`、`tests/`、`configs/`、`scripts/`、`notebooks/`、`pyproject.toml`（pytest `pythonpath=.`）。注意：本机 PATH 上默认 `python` 是 3.11.9，建 venv 必须用 `py -3.13`。
- 2026-09-30（M0 补齐，用户六步规格）：`requirements.txt` 去掉 torch（README 写明安装顺序 venv → torch cu128 → requirements → `pip install -e .` → `scripts/check_env.py`）；`check_setup.py` 改为 `check_env.py`（加 matmul 计时、pygame 导入、失败 exit 1）；按 PLAN §2 建 12 个桩模块 + `tests/test_smoke.py`；可复现性数字记入 M0 行。`notebooks/results.ipynb` 未建桩（M10 再建）。
- 2026-09-30：M1 完成。`snake_rl/env/{common,core,obs,rewards}.py` + `tests/{helpers,test_env,test_obs,test_rewards}.py`，75 测试全绿。实现决定见上表。push 由用户在自己终端做。**remote 已改为 HTTPS**（`https://github.com/Cureeeeeeee/Value-Based-Deep-Reinforcement-Learning-on-Snake.git`），凭据由 `gh auth setup-git` 提供（`gh` 已登录 Cureeeeeeee，token 含 `repo`）；原因是本机 `~/.ssh/id_ed25519` 没有注册到 GitHub 账号（用户终端同样 publickey 拒绝），不是沙箱问题。
- 2026-09-30：M2 完成。`agents/__init__.py`（Policy 协议）、`agents/baselines.py`、`evaluate.py`（含 CLI）、`utils.py`、`common.heading_to_action`、`tests/test_baselines.py`（30 测试）。验收表已录入。
- 2026-10-04：上次会话非正常关闭；检查后确认磁盘无丢失（工作区干净，M3 未开始）。M2 的 commit 由用户 push。确认 Part C 需要做到 M5（经 M3、M4）；Part B 时间表里 Week 8 还写了 "DQN with CNN (representation B) running"，那是 M6 前半截——截止前有余量就补一个 CNN-on-B 单次训练，来不及就按 M5 交并在上表记偏离。
- 2026-10-04：M3 完成（commit `b60a2e6`）。`snake_rl/env/vec_env.py`（`VecSnakeEnv`，405 行）、`tests/test_vec_env.py`（97 项）、`scripts/bench_vec_env.py`、`snake_rl/env/__init__.py` 导出 `VecSnakeEnv`；`PLAN.md` §3.6 与 §4 M3 行同步更新。顺带补了 `tests/test_env.py::test_set_state_validation` 的缺口（原来的非相邻用例只靠 `infer_heading` 的副作用报错，没有钉住相邻性检查本身）。本文件修了三处旧毛病：M0 行的 remote 地址、"待拍板"表丢失的标题、会话日志 M1/M2 顺序。做法：测试与实现各由一个 agent 按同一规格独立编写，再做四路对抗式审查（差分 fuzz、逐行审查、M5 数据通路探针、测试门禁与基准审计），每条发现由另一个 agent 尝试证伪；未发现引擎 bug，采纳的修改都是加固（`obs`/`final_obs` 不再共享内存、`set_state` 与 `reset(seed=)` 的异常安全、补测试）。
- 2026-10-04：用户拍板两项（见"待拍板"表）：批准把 `CLAUDE.md` 第 36 行的瓶颈说法改为实测结论（已改）；M5 的 G 与 batch 按"冒烟时两种配置各跑一条曲线再定"执行。
