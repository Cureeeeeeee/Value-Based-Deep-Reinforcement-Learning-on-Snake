# 在 macOS 上现场演示 snake-rl：完整指导书

适用：演示不在训练机（Windows 工作站）上做，而在一台 Mac 笔记本上做。
本文从"把东西搬过去"到"演示结束"逐步写；视频脚本与答辩题在 `PartF-presentation.md`。

## 0. 先说清楚

- **Mac 上的步骤没有在 Mac 上实测过**（我们只有 Windows 机器）。但同一套流程——干净 `git clone` + 解开演示包 +
  `python scripts/demo_check.py`——已于 2026-11-02 在 Windows 的干净目录里跑过：11 项全 PASS，24 秒。[COMPUTED]
  代码本身没有平台相关的部分（pygame、torch CPU、numpy），所以 Mac 上的风险集中在**安装**这一步，
  因此整份指南围绕一条自检命令组织：`python scripts/demo_check.py` 全 PASS 才算准备好，**至少提前三天做**。
- **先看芯片**：终端里 `uname -m`。`arm64` = Apple Silicon（M1–M4），按 §3 走。`x86_64` = Intel Mac，
  PyTorch 自 2.3 起不再发布 Intel Mac 安装包，而本项目要求 torch ≥ 2.7，正常安装必定失败；见 §8 的未验证备选方案，
  并认真考虑借一台 Apple Silicon 的 Mac。[KNOWN, HIGH]
- **macOS 自带的 `python3` 是 3.9**，项目要求 ≥ 3.11，必须另装 Python 3.12（§3）。[KNOWN, HIGH]
- **`runs/`（权重、录像、图）不在 git 里**。只 `git clone` 的话，GUI 的 `--player runs/...` 全部报 "no such checkpoint"。
  必须带演示包（§2）。
- 权重是在 GPU 上训练的，但加载代码写死了 `map_location="cpu"`，GUI 默认 `--device cpu`，所以在没有 CUDA 的 Mac 上能开。
  [COMPUTED：`snake_rl/agents/dqn/agent.py` `DQNAgent.load`；干净 clone 验证通过]
- 收尾那条 `python -m snake_rl.analysis.matrix --status` **在 Mac 上不要用**：演示包只带了两个 run 目录，
  它会显示 62 行 0/5。改用 `runs/report/index.md`（§5 最后一行）。

## 1. 时间线

| 何时 | 做什么 | 验收 |
|---|---|---|
| 提前 ≥ 3 天 | 演示包拷到 Mac（§2）；安装（§3）；自检（§3 第 9 步） | `demo_check.py` 打印 `11 / 11 passed` |
| 提前 1 天 | 接投影或外接屏按 §4 设置；按 §5 完整排练两遍、计时；拷贝预案文件（§7） | 能在 10 分钟内讲完 |
| 当天提前 15 分钟 | §4 的当天设置 + §4.3 预启动四个窗口 | 主窗口已开、处于暂停 |

## 2. 演示包（在 Windows 训练机上做；已做好）

已生成：`C:\STUDY\Reinforcement_Learning\assignment\Ass3\snake-rl\runs\export\demo_pack.zip`
（174 个文件，130.6 MB 原始 / 115.3 MB 压缩，2026-11-02，git `ce6f698` 时的 `runs/`）。
若之后 `runs/` 有变化，重新生成：

```powershell
.venv\Scripts\activate
python scripts/make_demo_pack.py          # -> runs/export/demo_pack.zip
```

包里有什么（清单在 `scripts/make_demo_pack.py` 的 `MANIFEST`）：

| 内容 | 用途 | 大小 |
|---|---|---|
| `runs/dqn_B8_W10_gamma099/0/`：`config.yaml`、`model.pt`、`final_eval.json`、`metrics.csv` | 主演示 agent（CNN，10×10，γ 0.99，最好的 10×10 学习策略） | 52 MB |
| 同一 run 的 `checkpoints/step_00012800.pt`、`step_00140800.pt` | 早 / 中期 checkpoint，"乱撞 → 会吃 → 会躲自己" | 52 MB |
| `runs/dqn_A_W10/0/`：同上四个文件 + `checkpoints/step_00012800.pt` | MLP agent（3 秒启动，面板多显示 11 维特征） | 1.6 MB |
| `runs/tabular_q_W10/0/` | 表格 Q（人玩时的顾问） | 0.1 MB |
| `runs/behaviour/dqn_B8_W10_gamma099_R3_final/`、`.../gamma099_early/` | `hack_*.json` 绕圈录像、热力图、临死局面图 | 1.3 MB |
| `runs/report/`（tables、figures、index.md、manifest.json） | 结果图表 | 5 MB |
| `runs/export/runs_logs.zip` | notebook 用的日志包（可选） | 15 MB |
| `runs/figures/partD_*` | Part D 截图（GUI 开不出来时的替身） | 0.2 MB |
| `runs/baselines/*.json` | 基线评估结果 | 0.1 MB |

传到 Mac：U 盘（exFAT 格式两边都能读）、OneDrive / Google Drive / iCloud 网页版、或局域网 AirDrop 不行（Windows 没有）。
一起带上的预案文件见 §7。

## 3. 在 Mac 上安装（Apple Silicon）

全部在"终端"（Terminal.app，Spotlight 搜 Terminal）里做，shell 是 zsh。每行一条命令。

1. **确认芯片与系统**
   ```zsh
   uname -m            # 期望 arm64
   sw_vers             # macOS 版本，13 以上都行
   ```
2. **命令行开发工具**（提供 git；已装过会提示 already installed）
   ```zsh
   xcode-select --install
   ```
3. **Python 3.12**（二选一；不要用系统自带的 3.9，也不要赌 3.14——PyTorch 的 3.14 安装包未必有）[INFERRED, MED]
   - 方案 A，Homebrew：
     ```zsh
     /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
     brew install python@3.12
     ```
     装完 Homebrew 会提示把 `eval "$(/opt/homebrew/bin/brew shellenv)"` 加进 `~/.zprofile`，照做并重开终端。
   - 方案 B，不装 Homebrew：到 https://www.python.org/downloads/macos/ 下载 **Python 3.12.x macOS 64-bit universal2 installer**，双击安装。
   - 验证：
     ```zsh
     python3.12 --version      # Python 3.12.x
     ```
4. **克隆仓库**（公开仓库，不需要登录）
   ```zsh
   cd ~
   git clone https://github.com/Cureeeeeeee/Value-Based-Deep-Reinforcement-Learning-on-Snake.git snake-rl
   cd ~/snake-rl
   git log --oneline -1      # 应不早于 ce6f698（含 docs/PartF-presentation.md）
   ```
5. **虚拟环境**
   ```zsh
   python3.12 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   ```
   以后每开一个新终端标签页都要先 `cd ~/snake-rl && source .venv/bin/activate`（提示符前出现 `(.venv)`）。
6. **安装项目与依赖**。Mac 上直接从 PyPI 装 torch 即可（PyPI 的 macOS arm64 包就是 CPU/MPS 版），
   **不要**用 README 里 Windows 的 `--index-url .../cu128`。
   ```zsh
   pip install -e .
   pip install -r requirements.txt     # 多装 pytest
   ```
   torch 约 70–200 MB 下载；校园网慢就提前做。
7. **验证导入**（不要运行 `scripts/check_env.py`：它检查 CUDA，在 Mac 上按设计返回失败）
   ```zsh
   python -c "import torch, pygame, numpy, gymnasium, snake_rl; print(torch.__version__, pygame.__version__, numpy.__version__)"
   ```
8. **解开演示包到仓库根目录**。不要在 Finder 里双击解压（会解到 Downloads 里一个 `runs` 文件夹）；用命令，`-d` 指向仓库：
   ```zsh
   unzip ~/Downloads/demo_pack.zip -d ~/snake-rl
   ls ~/snake-rl/runs      # baselines behaviour dqn_A_W10 dqn_B8_W10_gamma099 export figures report tabular_q_W10
   ```
   如果已经在 Finder 里解开了，把得到的 `runs` 文件夹整个拖进 `~/snake-rl/`（与 `snake_rl`、`scripts` 同级）。
9. **自检**（关键一步；每项是演示命令的无窗口版，另加一次 6–10 秒的现场训练）
   ```zsh
   cd ~/snake-rl && source .venv/bin/activate
   python scripts/demo_check.py
   open demo_check            # Finder 打开截图目录，逐张看一眼
   ```
   期望末尾：
   ```
   PASS  files_report
   PASS  gui_cnn_W10
   PASS  gui_cnn_W10_early_ckpt
   PASS  gui_mlp_W10
   PASS  gui_human_cnn_advisor
   PASS  gui_human_tabular_advisor
   PASS  gui_bfs
   PASS  gui_hamiltonian_W8
   PASS  gui_replay_R3_loop
   PASS  train_smoke_8x8
   PASS  gui_play_smoke_run

   11 / 11 passed
   ```
   每行后面的秒数就是该命令在这台 Mac 上的启动时间，`gui_cnn_W10` 的秒数决定你提前多久预启动。
   再跑一次自检要换种子（run 目录不覆盖）：`python scripts/demo_check.py --train-seed 1`。
   任何 FAIL 会打印该命令输出的最后几行；对照 §7 处理。
10. **可选：跑 GUI 的测试**（57 项，几秒）
    ```zsh
    pytest -q tests/test_gui.py
    ```
11. **第一次开真窗口**
    ```zsh
    python -m snake_rl.gui.app --player bfs --cell 72
    ```
    会弹出标题为 "snake-rl - Agent play" 的窗口，起始是暂停态。**先用鼠标点一下窗口**（macOS 下 pygame 窗口第一次常常没有键盘焦点），
    再按 Space 开跑，`=` 加速，`-` 减速，Q 退出。窗口被终端挡住就 Cmd+Tab 切到 "python"。

## 4. 演示前设置

### 4.1 显示

- 接投影 / 外接屏后：系统设置 → 显示器 → 选"镜像"（Mirror），让你看到的和观众一样。
- 按镜像后的分辨率选格子大小（窗口 = 10 格 + 侧栏）：

  | 投影分辨率 | `--cell` | 窗口尺寸 |
  |---|---|---|
  | 1920×1080 | 72 | 1200×832 |
  | 1366×768 / 1280×800 | 56 | 1040×672 |
  | 1024×768 | 48 | 960×592 |

- 终端字号调大（Cmd 和 `+`）到后排能看清训练进度行。

### 4.2 系统

- 插电源。
- 防休眠：开一个终端标签页跑 `caffeinate -dimsu`（演示结束 Ctrl+C），或 系统设置 → 锁定屏幕 → 接通电源时永不关闭显示器。
- 打开"勿扰"（控制中心 → 专注模式），关掉微信 / 邮件 / 日历弹窗。
- 关掉占 CPU 的程序（浏览器几十个标签、Docker、同步盘）。

### 4.3 进场前 15 分钟：预启动四个窗口

终端 Cmd+T 开标签页；每个标签页先：`cd ~/snake-rl && source .venv/bin/activate`。

```zsh
# 标签页 1（主演示；开好后保持暂停，不要按 Space）
python -m snake_rl.gui.app --player runs/dqn_B8_W10_gamma099/0 --cell 72

# 标签页 2：空着，激活了 .venv，演示中所有命令在这里敲

# 窗口 3：三张结果图在"预览"里（一个窗口，方向键翻页）
open runs/report/figures/gamma_W10_curve.png runs/report/figures/components_B8_W8_curve.png runs/report/figures/score_distributions.png

# 窗口 4：Part E 的 PDF，翻到 Table 2（"The questions and the answers"）与 Table 3
open ~/Downloads/Assignment-3-PartE-JiahaoLiu-14191136.pdf
```

把标签页 2 里要用的命令按 §5 顺序事先敲好放在剪贴板或一个文本文件里，演示时粘贴，不要现场打字。

## 5. 演示流程（10 分钟）

按键（在 GUI 窗口里，先点一下窗口）：Space 暂停 / 继续；N 或 `.` 单步；`=` 加速、`-` 减速；R 新一局；S 把本局存成录像到 `runs/recordings/`；
Q 或 Esc 退出。人玩模式：方向键或 WASD 给出想去的方向，180° 掉头被忽略。

| 分钟 | 做什么 | 窗口 / 命令 | 说什么、指什么 |
|---|---|---|---|
| 0–1 | 一句话定位，直接给看 | 标签页 1 的窗口：点窗口，按 Space | "Self-built Gymnasium-style Snake; the CNN agent trained for a million steps at γ 0.99 is playing the 10×10 evaluation boards." 指面板：三条 Q 值与 chosen、Score、Steps since last food / 100、Return、局终死因计数。 |
| 1–2.5 | 解读一次决策 | Space 暂停在一个转弯处，N 单步 | 读三个 Q 值的差："the three values are usually within a few tenths because all three moves are safe; the ordering is what the greedy policy uses." 按 `=` 加到 20 步/秒，跑 2–3 局，指死因计数：几乎全是 self。 |
| 2.5–4 | 教授来玩 | 标签页 2：`python -m snake_rl.gui.app --mode human --player runs/dqn_B8_W10_gamma099/0 --seed 10000 --cell 72` | 先说："It starts paused: each arrow key is one step, Space runs. The panel is the agent's advice for your state." 让教授玩到死或 1 分钟，Q 退出。再 `python -m snake_rl.gui.app --player runs/dqn_B8_W10_gamma099/0 --seed 10000 --cell 72`：同一张棋盘换 agent。 |
| 4–5 | 参考玩家定标 | `python -m snake_rl.gui.app --player bfs --seed 10000 --fps 20 --cell 72`；然后 `python -m snake_rl.gui.app --player hamiltonian --W 8 --fps 60 --cell 72` | "BFS eats as fast as any learned policy and traps itself at about 31 on 10×10; the Hamiltonian cycle never dies and fills the board (97), at a third of the speed. Every learned agent sits between them." |
| 5–6.5 | 现场训练一个 DQN | `python -m snake_rl.train --config configs/smoke_8x8.yaml --seed 1102 --exp demo_live`（Mac 上约 6–10 秒）→ `python -m snake_rl.gui.app --player runs/demo_live/1102 --cell 72` | 边跑边念进度行：eps、eval 均值 ± CI、死因、updates。"16k steps, 3.7k updates, a few seconds of compute: the feature-vector agent learns all it will learn in a few hundred updates, which is also why replay barely helps it." 每次演示换一个 `--seed`（run 目录不覆盖，重复会报错）。 |
| 6.5–7.5 | 回放与 reward hacking | `python -m snake_rl.gui.app --replay runs/behaviour/dqn_B8_W10_gamma099_R3_final/1/hack_1_10029.json --fps 12 --cell 72` | "Recorded under the exploitable reward R3; it circles until starvation at score 19. The Q values in the panel are the agent's own estimate inside the loop: far below the bonus the loop collects, so these are estimation errors, not hacking; a forager earns the bonus more often than a loop." |
| 7.5–9.5 | 结果 | 窗口 3 三张图；窗口 4 Table 2 | γ 轴：+5 点，全是存活时间（food/100 steps 11.0→7.7，steps 188→334）；组件轴：−replay −9.3，−target ≈ 0，`t31` 对照把 5.5 点要回来；箱线图：对 BFS 的位置。最后一句：没有任何 agent 学会"不把自己围起来"（10×10 撞己 84–99 %）。 |
| 9.5–10 | 复现 | `head -30 runs/report/index.md` | "Every table and figure of the report is regenerated from the run logs by one command; the index records the code commit; the notebook does the same from the exported logs." |

可选加段（有时间、被问"训练过程长什么样"时）：早期 checkpoint 直接玩
`python -m snake_rl.gui.app --player runs/dqn_B8_W10_gamma099/0/checkpoints/step_00012800.pt --cell 72`（12.8k 步：乱撞、绕圈）
→ `.../step_00140800.pt`（140.8k 步：会吃、常撞己）→ `runs/dqn_B8_W10_gamma099/0`（最终）。

### 5.1 五分钟压缩版

0–1.5 agent 玩 + 读一次 Q 值；1.5–3 教授玩同种子 + agent 同种子；3–4 现场训练并让它玩；4–5 窗口 3 的 γ 曲线与组件曲线 + 一句结论。

### 5.2 不要做的事

- 不跑 `matrix --status`（演示包只有两个 run，显示一片 0/5）、不跑 `pytest -q` 全量（1,916 项）、不跑 `analysis.report`（包里没有全部 run，会报缺）。
- 不在人玩模式里替教授按键。
- 不说"超过了 BFS"：10×10 最好的学习策略 25.7 对 BFS 30.6；8×8 上"分不开"（−0.35 [−1.39, +0.70]）。

## 6. 排练清单（提前一天，接着投影做）

- [ ] §4.3 四个窗口开齐，主窗口启动秒数记下来。
- [ ] §5 从头到尾按表走一遍，计时；第二遍只走 §5.1。
- [ ] 人玩模式自己玩一局，确认方向键 / 焦点正常。
- [ ] 现场训练那条命令跑一次（用一个排练种子，如 `--seed 1101`），确认能载入 GUI。
- [ ] 演示结束 Q 退出都正常，终端没有报错残留。

## 7. 预案

| 故障 | 处理 |
|---|---|
| `pip install -e .` 在 torch 上失败 | 看报错第一行。"No matching distribution for torch" → Python 版本不对（必须 3.12/3.13，`python --version` 看 venv 里的）或 Intel Mac（§8）。网络超时 → 换网络或手机热点再 `pip install -e .`。 |
| `demo_check.py` 某项 FAIL："no such checkpoint" / "missing:" | 演示包没解到仓库根：`ls ~/snake-rl/runs` 应有 8 个目录；重做 §3 第 8 步。 |
| `demo_check.py` 的 GUI 项 FAIL，报 pygame / SDL 错 | `pip install --upgrade pygame`；仍失败则 `pip install pygame-ce` 并 `pip uninstall pygame`（同 API）[GUESS, LOW]；再不行用预案截图 `runs/figures/partD_gui_*.png` 和视频。 |
| 窗口开了但按键没反应 | 用鼠标点一下 GUI 窗口；确认不是暂停态（面板底部 "paused" / "running"）；人玩模式本来就是一键一步，按 Space 才连续跑。 |
| 窗口不见了 | Cmd+Tab 切到 "python"；或在终端 Q 退出重开。 |
| CNN 加载太慢 | 换 `--player runs/dqn_A_W10/0`（MLP，约 3 秒，面板多出 11 维特征显示，也是看点）。 |
| 现场训练报 "run directory exists" | 换 `--seed`。 |
| 投影分辨率低、窗口出屏 | Q 退出，用 `--cell 56` 或 `--cell 48` 重开。 |
| 整机故障 | 预录视频 mp4 + Part E PDF（Table 2、3，Figure 1）+ `runs/figures/partD_gui_*.png`。这三样提前放到 Mac 的 `~/Downloads/` **和** U 盘。 |

## 8. Intel Mac（x86_64）：未验证的备选方案

PyTorch 最后一个带 Intel Mac 安装包的版本是 2.2.2，只支持 Python ≤ 3.11，且它是对 NumPy 1.x 编译的。[KNOWN, HIGH]
本项目声明 torch ≥ 2.7、numpy ≥ 2.0，所以要绕过依赖声明装旧版本；代码是否在 torch 2.2 / numpy 1.26 上全部正常**没有验证过**。[GUESS]
优先级：借 Apple Silicon 的 Mac > 下面的办法 > 纯视频 + PDF 演示。

```zsh
brew install python@3.11
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install "torch==2.2.2" "numpy<2" "gymnasium>=1.0" "pygame>=2.6" "matplotlib>=3.9" "pandas>=2.2" "pyyaml>=6.0" "pytest>=8.0"
pip install -e . --no-deps
python scripts/demo_check.py        # 全 PASS 才能用；任何 FAIL 就放弃这条路
```

## 9. 一页速查（打印或放在手机上）

```zsh
cd ~/snake-rl && source .venv/bin/activate
python scripts/demo_check.py --train-seed 2                                                   # 自检
python -m snake_rl.gui.app --player runs/dqn_B8_W10_gamma099/0 --cell 72                      # 主演示（预启动）
python -m snake_rl.gui.app --mode human --player runs/dqn_B8_W10_gamma099/0 --seed 10000 --cell 72   # 教授玩，CNN 顾问
python -m snake_rl.gui.app --player runs/dqn_B8_W10_gamma099/0 --seed 10000 --cell 72         # agent 玩同一棋盘
python -m snake_rl.gui.app --player bfs --seed 10000 --fps 20 --cell 72                       # BFS
python -m snake_rl.gui.app --player hamiltonian --W 8 --fps 60 --cell 72                      # Hamiltonian 8x8
python -m snake_rl.train --config configs/smoke_8x8.yaml --seed 1102 --exp demo_live          # 现场训练，约 6–10 s
python -m snake_rl.gui.app --player runs/demo_live/1102 --cell 72                             # 让它玩
python -m snake_rl.gui.app --replay runs/behaviour/dqn_B8_W10_gamma099_R3_final/1/hack_1_10029.json --fps 12 --cell 72   # R3 绕圈回放
open runs/report/figures/gamma_W10_curve.png runs/report/figures/components_B8_W8_curve.png runs/report/figures/score_distributions.png
head -30 runs/report/index.md                                                                 # 收尾
```

按键：Space 暂停/继续 · N 单步 · `=` / `-` 变速 · R 新局 · S 存录像 · Q 退出 · 人玩：方向键 / WASD。

## 10. 让 Mac 上的 Claude Code 代劳

§3 的安装可以交给 Mac 上的 Claude Code 做，但有三件事它做不了或不该做：需要点对话框 / 输密码的安装（Xcode 命令行工具、Homebrew），
把演示包放到 Mac 上，以及最后用眼睛和键盘确认真窗口能用。

### 10.1 人先做（约 15 分钟 + 下载时间）

1. 终端里 `xcode-select --install`，弹窗点"安装"（给 git）。
2. 装 Homebrew（会要开机密码）：`/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`，
   按它结尾的提示把 `brew shellenv` 那行加进 `~/.zprofile`，重开终端；然后 `brew install python@3.12`。
   不想装 Homebrew 就用 python.org 的 3.12 安装器（§3 第 3 步方案 B）。
3. 把 `demo_pack.zip` 放到 `~/Downloads/`（U 盘 / 网盘；若已挂成 GitHub Release，Claude Code 可以自己 `curl` 下载，见提示词）。
   校验：`shasum -a 256 ~/Downloads/demo_pack.zip` 应为 `f0dc432887ef43b26f18f99e04412d759cfc853fee714ec6c50c29f99154f8fb`（115,325,347 字节，2026-11-02 版）。
4. 把预案文件也放进 `~/Downloads/`：预录视频 mp4、Part E 的 PDF。
5. 克隆仓库并在仓库目录里启动 Claude Code：
   ```zsh
   cd ~ && git clone https://github.com/Cureeeeeeee/Value-Based-Deep-Reinforcement-Learning-on-Snake.git snake-rl
   cd ~/snake-rl && claude
   ```
6. 粘贴 §10.2 的提示词。之后它每次要跑 `pip` / `unzip` / `python` 会请求权限，逐个允许即可。

### 10.2 粘贴给 Claude Code 的提示词

```text
这台 MacBook 是演示机，不是训练机。目标：按 docs/DEMO-macOS.md §3 把演示环境装好，
让 `python scripts/demo_check.py` 打印 `11 / 11 passed`，然后开一个真窗口给我看。先读 docs/DEMO-macOS.md 全文。

本会话的规则（覆盖 CLAUDE.md 里与 Windows / CUDA 有关的内容）：
- 不装 CUDA 版 torch，不用 --index-url；`pip install -e .` 从 PyPI 装即可。不要运行 scripts/check_env.py（它检查 CUDA，在 Mac 上按设计失败）。
- 先 `uname -m`。若是 x86_64（Intel），停下来告诉我，按文档 §8 之前先问我要不要继续。
- 用 python3.12 建 .venv（`python3.12 -m venv .venv`）；系统自带的 python3 是 3.9，不能用；也不要用 3.14。
- 演示包在 ~/Downloads/demo_pack.zip，先 `shasum -a 256` 核对
  f0dc432887ef43b26f18f99e04412d759cfc853fee714ec6c50c29f99154f8fb，再 `unzip ~/Downloads/demo_pack.zip -d ~/snake-rl`，
  解完 `ls runs` 应有 8 个目录（baselines behaviour dqn_A_W10 dqn_B8_W10_gamma099 export figures report tabular_q_W10）。
- 不改 snake_rl/ 和 scripts/ 下的代码，不训练（demo_check 自带的 smoke 训练除外），不 commit，不 push；
  不跑全量 pytest，只跑 `pytest -q tests/test_gui.py`。
- 任一步失败：把命令和输出的最后 20 行原样给我，对照文档 §7 的预案表处理；不要为了让检查通过而改代码或改测试。

做完后汇报：uname -m、macOS 版本、python / torch / pygame / numpy 版本、demo_check 每一项的秒数、FAIL 项（若有）。
最后在后台启动 `python -m snake_rl.gui.app --player runs/dqn_B8_W10_gamma099/0 --cell 72`，
告诉我窗口叫什么、怎么操作（先点窗口取得焦点，Space 开跑，`=` 加速，Q 退出），然后停下等我确认。
```

若演示包已挂成 GitHub Release，把提示词里"演示包在 ~/Downloads/demo_pack.zip"一句换成：
"先 `curl -L -o ~/Downloads/demo_pack.zip <Release 资产的 URL>` 下载演示包（115 MB）"。

### 10.3 Claude Code 汇报之后，人再做

- 点一下它开的窗口，按 Space / `=` / Q，确认键盘能用（§3 第 11 步）。
- 按 §4 接投影设置，按 §6 排练两遍。
- 若它报告某项 FAIL 且 §7 没覆盖，把它的汇报原文发回训练机这边的会话。
