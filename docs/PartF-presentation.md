# Part F / G — 视频脚本、现场演示 runbook、答辩速查

Part F（15 分）= 预录视频（3–5 分钟）+ 现场 presentation / demo；Part G（25 分）= 口头答辩。
本文件给出：§1 视频逐镜头脚本（旁白英文，可直接念）；§2 现场演示的命令、顺序、预案；
§3 以结果为依据的答辩速查（理论题见 `Snake项目方案-设计依据.md` §11）；§4 演示前待办。

所有命令于 2026-11-01 在本机（`.venv`，git `570706e`，`runs/` 全量在位）用 `--headless` 实跑验证过；
括号里的启动耗时和训练耗时是当时实测。

## 0. 先说风险

1. **视频模板是为"商业问题"写的**（Introduction 1 min / Business problem 1 min / Demo 1–2 min）。我们是受控研究，没有客户。
   不要硬编一个业务场景；把"问题"段明确讲成"所有 Snake+DQN 项目都在默默做三个设计决定，没人量过哪个重要；
   任何把值函数 RL 用到实际问题上的人都面对同样的决定"。这段若含糊，评分表上 Business problem 一栏会被扣。
2. **3–5 分钟是硬上限**。§1 的旁白约 620 词，正常语速（140 词/分）约 4:25；录完务必看时长，超过 4:50 就删 §1.3 的"参考玩家"半句。
3. **`runs/` 不进 git**。换一台机器演示时，`git clone` 下来 GUI 的 `--player runs/...` 全部失败。要么带本机，要么按 §2.1 打"演示包"。
4. **CNN 的 GUI 启动约 13 秒**（torch 导入 + 53 MB `model.pt`，默认 `--device cpu`），MLP 约 3 秒。现场所有窗口**进场前预启动**，开着暂停等。
5. **人玩模式起始是暂停态**：每按一次方向键走一步，Space 才连续跑。不先说清楚，教授会以为"卡了"。
6. **Hamiltonian 在 10×10 走 2,400 步才填满**，6 步/秒要 7 分钟。演示用 `--W 8 --fps 60`（约 1,000 步，17 秒）或讲完就 Q 退出。
7. **notebook 还没在 Colab 上真跑过**（本机按 Colab 路径跑过：clone + 下载日志 zip + 全部表图一致，26 s）。规范写 "Google Colab (preferred)"。
   教授若现场要看 Colab，没跑过就是风险；见 §4 第 1 条。
8. **Part G 按个人贡献给分**（Appendix D：40 / 30 / 30）。三人都得能独立讲清 §3 的问题；只有一个人能答会直接体现在分上。[INFERRED，HIGH]

## 1. 视频（3–5 分钟）

### 1.1 结构与时间线

| 时间 | 模板段落 | 屏幕上 | 旁白词数 |
|---|---|---|---|
| 0:00–0:15 | Introduction | 标题页（题目、项目号 48、队名、三人姓名） | 40 |
| 0:15–1:00 | Introduction / background | GUI：CNN agent 在 10×10 上玩（§1.3 clip A）→ 切仓库 README 的结构树 3 秒 | 110 |
| 1:00–2:00 | Problem | 幻灯：三个问题各一行；"64 configurations × 5 seeds × 100 boards" | 150 |
| 2:00–2:40 | Demo：界面 | clip A 继续，在一次转弯处暂停并读 Q 值；clip B 人玩 + 同种子 agent | 110 |
| 2:40–3:05 | Demo：回放与 hacking | clip C：R3 agent 绕圈至饿死的回放，面板 Q 值 | 60 |
| 3:05–4:00 | Demo：结果 | `runs/report/figures/gamma_W10_curve.png` → `components_B8_W8_curve.png` → `score_distributions.png`；Table 2 截图 | 130 |
| 4:00–4:15 | Demo：复现 | 终端：`python -m snake_rl.analysis.report`、notebook 滚动 | 30 |
| 4:15–4:30 | Close | 回到 GUI 定格 + 仓库 URL | 50 |

### 1.2 旁白（英文，直接念；【】是镜头与动作）

**0:00 Introduction**

【标题页】
"Hello, we are The Deadly Triad: Yue Liu, Jiahao Liu and Siqin Jin. Our project is value-based deep reinforcement learning on Snake, used as a controlled study."

【clip A：CNN agent 在 10×10 上玩，面板可见】
"Snake is a small, fully observable game with an endless supply of random boards. That makes it cheap enough to run hundreds of training runs, which is exactly what a controlled study needs. We built the environment ourselves in the Gymnasium style, with a vectorised version that steps up to a million boards per second, and we check the two versions against each other bit for bit. On top of it we implemented, from scratch, tabular Q-learning, DQN, Double DQN, Dueling and prioritised replay, and three non-learning reference players: random, a breadth-first greedy planner, and a Hamiltonian cycle, which fixes the top of the scale because it fills the board every time."

**1:00 The problem**

【幻灯：三行问题】
"The problem we address is not whether a network can learn Snake. It can, and there are hundreds of tutorials that show it. The problem is that every one of those projects silently makes three design decisions, and nobody measures which of them matter. How is the state represented: eleven hand-made features, or the whole grid as image planes? How is the reward designed: sparse, a step penalty, potential-based shaping, or a naive distance bonus that could be exploited? And which stabilisers does DQN actually need: experience replay, the target network, Double DQN?"

【幻灯：64 × 5 × 100】
"Anyone who builds a value-based agent for a real problem faces these choices, and an answer from one run with one seed is noise. So our contribution is the protocol: sixty-four configurations, five seeds each, the same hundred test boards for everyone, confidence intervals and paired differences, and every table regenerated from the logs by one command."

**2:00 Demo: the interface**

【clip A 继续；在一次真实转弯处按 Space 暂停，读面板上的三个数——不要念脚本里的数字】
"This is our interface. The convolutional agent is playing the ten-by-ten board, and the panel shows what it is thinking: the value of going straight, left or right in the current state, and the action it picked. Here it turns, because turning is worth more than going straight: straight would lead into its own body."

【clip B：人玩同一种子（advisor 面板），接 agent 同种子】
"The examiner can play the same board with the agent advising in the panel, and then watch the agent on the very same board, because episodes follow fixed evaluation seeds."

**2:40 Demo: recordings and reward hacking**

【clip C：`--replay` R3 绕圈录像，面板 Q 值随步变化】
"Recordings replay deterministically. This is an agent trained with the exploitable distance bonus, circling until it starves. The panel shows the value the agent itself assigns to the loop, and it is far below what the loop actually earns. So the loops are estimation errors, not exploitation: at the bonus sizes we tested, the naive reward was never hacked."

**3:05 Demo: results**

【`gamma_W10_curve.png`】
"And the results. Of the three planned axes, only the representation changed the score at the planned discount, and by under two points. The largest effect was a hyper-parameter the plan had held fixed: raising the discount factor from 0.9 to 0.99 adds about five points to the convolutional agent on both boards, all of it survival time, and brings it level with the greedy planner on the small board."

【`components_B8_W8_curve.png`】
"Removing replay costs the convolutional agent nine points on the small board; removing the target network costs nothing we can measure."

【`score_distributions.png` 或 Table 2 截图】
"Double DQN removes about a third of the overestimation where it is large, but leaves the score unchanged. Dueling and prioritised replay change nothing. Reward shaping moves the score by about a point at most."

**4:00 Demo: reproducibility**

【终端：报告命令跑起来；notebook 滚动】
"Every number in the report comes from this one command, and the notebook rebuilds every table and figure from the exported logs."

**4:15 Close**

【GUI 定格在一次撞己死亡的局终统计；叠仓库 URL】
"What reinforcement learning bought here is the planner's eating speed without planning, or the planner's score by surviving longer. What no agent learned is what the Hamiltonian cycle knows: never enclose yourself. That gap, and the protocol that makes it visible, is our project. Thank you."

数字对照（念之前核对，全部来自 `docs/PartE-final-report.md` Table 2 / Table 3）：
representation +1.71 [+1.26, +2.15]（8×8，γ 0.9）；γ 0.9→0.99 B8 +4.91（8×8）/ +5.03（10×10）；
B8 γ 0.99 对 BFS −0.35 [−1.39, +0.70]（8×8）；−replay −9.26（B8 8×8，17.8→8.6）；−target +0.20 / +0.34；
Double 高估 −0.87 of +2.53（B8 γ 0.99）；奖励轴最大配对差约 1 点；R3 在 0.01 / 0.1 两档未被利用。

### 1.3 录制清单

录屏工具：OBS 或 Win+G（Xbox Game Bar），1920×1080，30 fps，系统声音关、麦克风开；旁白可后配。
GUI 用 `--cell 72`（窗口 1200×832，投影和 1080p 录屏都清楚）。每条 clip 录到独立 mp4，再剪。

```powershell
.venv\Scripts\activate
# clip A: CNN agent, 10x10, gamma 0.99 (the best 10x10 agent); start paused, press Space when recording
python -m snake_rl.gui.app --player runs/dqn_B8_W10_gamma099/0 --cell 72 --fps 6
# clip B1: human plays board 10000 with the CNN advising (arrows step, Space runs)
python -m snake_rl.gui.app --mode human --player runs/dqn_B8_W10_gamma099/0 --seed 10000 --cell 72
# clip B2: the agent on the same board
python -m snake_rl.gui.app --player runs/dqn_B8_W10_gamma099/0 --seed 10000 --cell 72 --fps 10
# clip C: an R3 agent looping until starvation (score 19, 265 steps); speed up with + after the first loop
python -m snake_rl.gui.app --replay runs/behaviour/dqn_B8_W10_gamma099_R3_final/1/hack_1_10029.json --cell 72 --fps 12
# optional clip D: the reference players on the same board
python -m snake_rl.gui.app --player bfs --seed 10000 --cell 72 --fps 20
python -m snake_rl.gui.app --player hamiltonian --W 8 --fps 60 --cell 72
# optional clip E: early / mid / final checkpoints of the same run (fumbling -> eating -> avoiding itself)
python -m snake_rl.gui.app --player runs/dqn_B8_W10_gamma099/0/checkpoints/step_00012800.pt --cell 72
python -m snake_rl.gui.app --player runs/dqn_B8_W10_gamma099/0/checkpoints/step_00140800.pt --cell 72
# reproducibility shot (about 1 min; record only the first seconds and the finished index)
python -m snake_rl.analysis.report --out-dir runs/report
```

结果图直接用 `runs/report/figures/`（gamma_W10_curve、components_B8_W8_curve、score_distributions、obs_W8_curve、
causes_B8_W8_gamma099_R0R2R3）；Table 2 从 Part E PDF 截图。标题页与"三个问题"页用 PowerPoint 两页即可。
导出 mp4（H.264），文件名按提交页要求（本仓库文档里没有记录，提交前查 Canvas）。

### 1.4 三人分工建议（按 Appendix D 的贡献陈述对应，未经团队确认）[GUESS]

- 0:00–1:00 Introduction：一人；1:00–2:00 Problem：一人；2:00–4:30 Demo + Close：代码负责人（Jiahao Liu）。
- 三人都出镜或出声；规范写 "all team members must contribute equally in the project development and delivery"。

## 2. 现场演示 runbook

### 2.1 演示前一天

- [ ] `git status` 干净；`git pull`；`.venv\Scripts\python.exe -c "import torch, pygame; print(torch.__version__)"`。
- [ ] 若不用本机：演示包 `runs/export/demo_pack.zip`（`python scripts/make_demo_pack.py` 生成，115 MB）解到目标机器的仓库克隆根目录，
  `python scripts/demo_check.py` 全 PASS。macOS 的逐步安装与演示流程见 `docs/DEMO-macOS.md`（Mac 上收尾不用 `matrix --status`，演示包只有两个 run）。
- [ ] 显示：Windows 缩放 100 % 或 125 %，关通知（专注助手），关 cs2 等占 GPU / CPU 的东西。
- [ ] 预案资产在 U 盘 + 云端：视频 mp4、Part E PDF、`runs/figures/partD_gui_*.png`、§1.3 的各 clip。
- [ ] 计时排练两遍（10 分钟版和 5 分钟版）。

### 2.2 进场前 15 分钟：预启动四个窗口

```powershell
.venv\Scripts\activate
# W1: the main demo, starts paused
python -m snake_rl.gui.app --player runs/dqn_B8_W10_gamma099/0 --cell 72
# W2: a terminal in the repository root with the venv active (for human play, training, replay)
# W3: explorer on runs\report\figures  (gamma_W10_curve.png, components_B8_W8_curve.png, score_distributions.png open)
# W4: the Part E PDF at Table 2 (page with "The questions and the answers") and Table 3
```

### 2.3 十分钟流程

| 分钟 | 做什么 | 窗口 / 命令 | 说什么、指什么 |
|---|---|---|---|
| 0–1 | 一句话定位，直接给看 | W1，按 Space 开跑 | "Self-built Gymnasium-style Snake; the CNN agent trained for a million steps at γ 0.99 is playing the 10×10 evaluation boards." 指面板：三条 Q 值、chosen、Score、Steps since last food / 100、Return、死因计数。 |
| 1–2.5 | 解读一次决策 | W1：Space 暂停在一个转弯，N 单步 | 读三个 Q 值的差；"the three values are usually within a few tenths because all three moves are safe; the ordering is what the greedy policy uses"。按 + 把速度加到 20 步/秒，跑 2–3 局，指死因计数：几乎全是 self。 |
| 2.5–4 | 教授来玩 | W2：`python -m snake_rl.gui.app --mode human --player runs/dqn_B8_W10_gamma099/0 --seed 10000 --cell 72` | 先说："It starts paused: each arrow key is one step, Space runs. The panel is the agent's advice for your state." 让教授玩到死或 1 分钟。然后 `--player runs/dqn_B8_W10_gamma099/0 --seed 10000`：同一张棋盘换 agent。 |
| 4–5 | 参考玩家定标 | W2：`--player bfs --seed 10000 --fps 20`，再 `--player hamiltonian --W 8 --fps 60` | "BFS eats as fast as any learned policy and traps itself at about 31 on 10×10; the Hamiltonian cycle never dies and fills the board (97), at a third of the speed. Every learned agent sits between them." |
| 5–6.5 | 现场训练一个 DQN | W2：`python -m snake_rl.train --config configs/smoke_8x8.yaml --seed 1101 --exp demo_live`（约 6 s）→ `python -m snake_rl.gui.app --player runs/demo_live/1101 --W 8 --cell 72` | 边跑边念进度行：eps、eval 均值 ± CI、死因、updates。"16k steps, 3.7k updates, about three seconds of compute: the feature-vector agent learns all it will learn in a few hundred updates, which is also why replay barely helps it." 每次演示换一个 `--seed`（run 目录不覆盖）。 |
| 6.5–7.5 | 回放与 reward hacking | W2：`python -m snake_rl.gui.app --replay runs/behaviour/dqn_B8_W10_gamma099_R3_final/1/hack_1_10029.json --cell 72 --fps 12` | "Recorded under the exploitable reward R3; it circles until starvation at score 19. The Q values in the panel are the agent's own estimate inside the loop: far below the bonus the loop collects, so these are estimation errors, not hacking; a forager earns the bonus more often than a loop." |
| 7.5–9.5 | 结果 | W3 三张图；W4 Table 2 | γ 轴：+5 点，全是存活时间（food/100 steps 11.0→7.7，steps 188→334）；组件轴：−replay −9.3，−target ≈ 0，t31 对照把 5.5 点要回来；箱线图：对 BFS 的位置。最后一句：没有任何 agent 学会"不把自己围起来"（10×10 撞己 84–99 %）。 |
| 9.5–10 | 复现 | W2：`python -m snake_rl.analysis.matrix --status`（4 s） | "64 rows, 5/5 seeds each, config checked against the matrix. One command rebuilds every table and figure; the notebook does the same from the exported logs." |

### 2.4 五分钟压缩版

0–1.5 W1 agent 玩 + 读一次 Q 值；1.5–3 教授玩同种子 + agent 同种子；3–4 现场训练 6 秒并让它玩；4–5 W3 的 γ 曲线与组件曲线 + 一句结论。

### 2.5 预案

| 故障 | 处理 |
|---|---|
| W1 起不来 / pygame 黑窗 | 用 §1.3 录好的 clip A；或 `python scripts/gui_screenshots.py` 的 `runs/figures/partD_gui_*.png`。 |
| CNN 加载慢（>15 s） | 切 `--player runs/dqn_A_W10/0`（3 s 启动，面板多出 11 维特征显示，也是个看点）。 |
| 教授按键没反应 | 窗口焦点；提醒"starts paused"；Space。180° 反向键被忽略是规则不是 bug。 |
| 现场训练报错 "run directory exists" | 换 `--seed`。 |
| 投影分辨率低 | `--cell 56`。 |
| 整机故障 | 视频 mp4 + Part E PDF（Table 2、3，Figure 1）。 |

### 2.6 不要做的事

- 不现场跑 CNN 训练（8×8 一条 17–37 分钟）、不跑 `pytest -q`（1,916 项，>1 分钟，中途有人问话就尴尬）、不跑 `analysis.report`（约 1 分钟）——这些说"已经跑过"，指 `runs/report/index.md` 的时间戳和 git hash。
- 不在人玩模式里替教授按键。
- 不说"我们的 agent 超过了 BFS"。10×10 上最好的学习策略 25.7 对 BFS 30.6；8×8 上"分不开"（−0.35 [−1.39, +0.70]），不是"超过"。

## 3. 答辩速查（Part G）：以结果为依据的问题

理论题（Q-learning 更新式、replay 为何要 off-policy、deadly triad、势函数定理、max 过估计、评估为何用贪心）见
`Snake项目方案-设计依据.md` §11。下面是评审看了报告和演示最可能追问的、必须带数字答的问题；章节号指 Part E。

1. **Part B 说 γ = 0.99，为什么主线是 0.9？** 表格 Q 的 α–γ 扫描（附录 C）里 γ 0.99 配 α ≥ 0.1 绕圈到饿死、低 2 点；DQN 参照要和表格共用折扣。γ 0.99 作为矩阵的一行补回来，结果成了最大效应（§6.3、§7.4）。
2. **为什么 γ 0.99 帮 CNN 不帮 MLP？** 11 维特征只看一格危险，表示不了更长视野要的信息（A：−0.20 [−1.35, +0.96]）；B8 用吃得慢换活得久：food/100 steps 11.0→7.7、步数 188→334（Table 3、§6.3）。
3. **目标网络没用，是 DQN 设计错了吗？** 不是，是问题尺度：3 动作、100 格、值函数小而光滑、lr 2.5e-4。我们只在用过的周期上说"测不出"，不做一般性结论（§7.2）。
4. **"去掉 replay"为什么拆成三件事？** 去掉 replay 同时改变了数据新鲜度、每条转移的更新次数 G、和以梯度步计的目标网络周期；`t31` 对照（周期按环境步匹配）要回 5.5 / 5.1 点，所以目标过期是最大一项（§6.4，Table 6）。
5. **有没有发散？** 320 条 run 没有一条发散，包括无 target、无 replay 的；看到的病态是相反的——目标过期时值被低估几倍（§7.2）。
6. **R3 被 hack 了吗？** 0.01 和 0.1 两档都没有。先验：觅食者拿 bonus 的频率高于绕圈者；证据：绕圈里 agent 自己的 Q 远低于绕圈实际能拿的值，所以是估计误差（§6.7）。年金机制（更大 bonus）没跑，不外推。
7. **统计怎么保证诚实？** 5 种子、同 100 张评估棋盘、配对差 + 95 % t 区间、符号计数、留一法；分辨率约 0.7 点（last-5）/ 1 点（final）。"无差别"= |效应| < 约 1，不是 0（§5.1、§7.3）。
8. **为什么 DQN 没赢 BFS？** γ 0.9 的 agent 吃得一样快但死得早（低 7–10 点）；B8 γ 0.99 活得更久但吃得慢，10×10 低 4.9 点，8×8 分不开。没有 agent 学会 Hamiltonian 会的事：不把自己围死（10×10 撞己 84–99 %）（§7.1）。
9. **什么没做成？** 饥饿上限探针什么都没测到（训练分布到不了上限）；第一版 no-replay 结果不可解释直到加了对照；PER 在 TD 误差趋零处表现差，是读 checkpoint 里优化器状态才明白的（§7.5）。
10. **为什么不做真实问题 / Atari？** 受控研究要几百条 run；10×10 一条不到 1 小时，8×8 的 MLP 5 分钟。方法（固定协议、种子、配对、参考玩家）才是可迁移的部分。
11. **时间花在哪？** 梯度更新占墙钟 94–99 %，环境 <1 %；MLP 一步 1.7 ms（CPU），CNN 2.6 ms（GPU）/ 40–69 ms（CPU）（§5.4、附录 A）。Part B 以为瓶颈在 Python 步进，错了，向量化环境之后就不是。
12. **复现性？** 同机从 `config.yaml` 重训逐位相同；每条 run 记 seed、git hash、软件版本；表和图一条命令重生成；notebook 从日志 zip 重建并逐格比对报告（§5.3、附录 A）。跨机器只保证分布相同。
13. **11 维特征是教程的，你们的贡献是什么？** 如实引用为 related work [11]；贡献是对比研究本身：环境、向量化等价门禁、协议、消融对照、过估计测量、行为分析，全部自写。
14. **个人贡献？** 每人按 Appendix D 自己那段讲，并能指出对应代码 / 实验 / 章节。

## 4. 演示前待办（用户）

1. **Colab 实跑一次** `notebooks/results.ipynb`（Run all；上传 `runs/export/runs_logs.zip` 或设 `LOGS_URL`）。
   若 Part E 还能改，去掉 §5.3 / §7.3 / 附录 A 三处 "not yet on Colab"；若已提交，至少现场能打开跑过的 Colab 页。
2. 查 Canvas 提交页：视频命名、时长、上传位置、现场演示的时长与日期（本仓库没有记录）。
3. 录 §1.3 的 clip，配音，导出 mp4，核时长 ≤ 5:00。
4. 定三人分工（§1.4 是建议），各自过一遍 §3。
5. 若换机器演示，按 §2.1 打包并在目标机器上实跑 §2.2 全部命令。
