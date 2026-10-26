# Assignment 3, Part E — Final Project Report (text of the submission)

This file is the source text of the Part E document
(`43008-Project-Part-E-TheDeadlyTriad.docx`). Every table is taken from the
tables that `python -m snake_rl.analysis.report` generates in `runs/report/`
from the run directories, every learning curve and value curve is one of its
figures, the behaviour figures come from `python -m snake_rl.analysis.behaviour`
and `scripts/report_extra_figures.py`, and the paired differences quoted in the
text are the ones recorded, independently re-computed and reviewed in
`docs/PROGRESS.md` (sections "M4 验收" to "M10 验收"). Numbers are never typed
in by hand; `runs/review/parte/verify_parte.py` checks the tables of this file
against the run directories. The cover page, the table of contents and the
page layout follow the copy of the document that the team edited on
23 October 2026 (`../43008-Project-Part-E-TheDeadlyTriad_FILLED.docx`); the
contribution statements of Appendix D are the ones the three members wrote
into that copy.

**Project title and number:** Value-Based Deep Reinforcement Learning on Snake: A Controlled Study of State Representation, Reward Design, and DQN Stabilisation — Project number: 48

**Team name:** The Deadly Triad

**Team members:** Yue Liu (25211451, Yue.Liu-14@student.uts.edu.au); Jiahao Liu (14191136, Jiahao.Liu-4@student.uts.edu.au); Siqin Jin (26145324, siqin.jin@student.uts.edu.au)

**Date:** 23 October 2026

**Code:** https://github.com/Cureeeeeeee/Value-Based-Deep-Reinforcement-Learning-on-Snake (environment, agents, analysis, GUI, 1,916 automated tests; `notebooks/results.ipynb` carries the code of the study and rebuilds every table and figure of this report from the exported logs)

## Abstract

We employ the game Snake, on an environment we developed ourselves, as a controlled test-bed for value-based reinforcement learning, ranging from tabular Q-learning to Deep Q-Networks (DQN), and three refinements (Double DQN, Dueling, prioritised replay). Three questions are explored across one protocol: in what way is the state represented? (an 11-feature vector with a multilayer perceptron vs a multi-plane grid with a convolutional network); how is the reward designed? (sparse, step penalty, potential-based shaping, naive distance shaping developed to promote reward hacking); and which stabilisation components are important (experience replay; target network; Double DQN). Each configuration is trained with five seeds, tested greedily on the same 100 boards and reported as a mean, with a 95 % Student-t interval and paired differences, with the non-learning players (random, breadth-first greedy, Hamiltonian cycle) fixing the scale. The matrix contains 64 configurations (320 runs and a tabular sweep of 105 runs). The biggest effect is not one of the three designed axes: Increasing the discount factor from 0.9 to 0.99 raises the convolutional agent by about 5 points on both boards, solely due to longer survival and the result on 8 × 8 is indistinguishable from the greedy planner. The grid representation outperforms the feature vector by 1.7 points on the small board and by nothing measurable on the large one at γ = 0.9, and by 5–7 points at γ = 0.99, while the reward design shifts the score by at most about one point; as for the target network, it has no measurable gain and the loss from removing replay decomposes into target staleness, time-mixing of the batch and number of gradient steps per transition, where the shares in the breakdown depend on the order of decomposition. Reward hacking never appeared, even when the shaping bonus was ten times bigger: according to the network's own value estimates, the looping policies seen early on in training were errors, not exploitation. Double DQN eliminates nearly a third of the overestimation of the convolutional agent at γ = 0.99 and practically none of the feature-vector agent's, without any consistent effect on the score.

## 1 Introduction

Snake is a small, fully observable game with a basic rule set and an infinite supply of random boards, which makes it a convenient place for reinforcement-learning experiments, which would be too costly on Atari. Furthermore, it is a game with a lot of Deep Q-Network tutorials, but the tutorials do not compare anything under a fixed protocol. This project views Snake as a tool instead of a goal and the question is not "can a DQN play Snake" (yes it can) but which of the design decisions that each such project makes silently affect the outcome, and by how much, once seed-to-seed variation is taken into account. We ask three questions, which are fixed in the Part B plan:

- **State representation.** Many Snake tutorials [11] represent the state as a compact, hand-designed vector of 11 binary features (danger ahead / left / right, heading, food direction), which is easy to learn from, but loses the layout of the body. A multi-plane grid is fed to a convolutional network, which keeps everything. Does the additional information give a higher ceiling and what does it cost in terms of sample efficiency?
- **Reward design.** The honest goal is a sparse reward (food +1, death −1). The typical approaches to helping are a small step penalty, a potential-based shaping [5] and a naive distance bonus. Are they helpful and does the naive bonus get exploited?
- **Stabilisation.** The "deadly triad" of function approximation, bootstrapping and off-policy learning [6] is the source of motivation for DQN's two stabilisers, experience replay and the target network. Which of them seems to have a measurable effect here, and does Double DQN minimise the overestimation that the max operator causes [2, 12]?

Everything else is held fixed: the environment, evaluation protocol, 5 training seeds per configuration and the statistics (Section 5); the scale is provided by the three non-learning players (random, breadth-first greedy, Hamiltonian cycle), and the matrix of 64 configurations on two boards (8 × 8 and 10 × 10) is regenerated into this report in one command. Table 2 provides the answers to the three questions in one place, and the biggest effects were hyper-parameters that the plan had kept fixed. The background, environment, learners and protocol are provided in Sections 2–5; the results axis by axis in Section 6; the discussion, limitations and deviations from the plan in Section 7; and the reproduction recipe, full tables and tabular sweep in Appendices A–C.

## 2 Background and related work

**Q-learning and DQN.** The optimal action values for a finite Markov decision process satisfy the following equation: Q⋆(s, a) = E[r + γ max_a′ Q⋆(s′, a′)]. Tabular Q-learning [7] updates one entry per transition, Q(s, a) ← Q(s, a) + α [r + γ max_a′ Q(s′, a′) − Q(s, a)]. A DQN [1] replaces the table by a network trained to minimise the (here Huber) error between Q(s, a; θ) and the target r + γ max_a′ Q(s′, a′; θ⁻), with two additions: transitions are stored in a replay buffer [8] and sampled uniformly, thus decorrelating consecutive samples and reusing each transition multiple times; the target uses a frozen copy θ⁻ of the network, refreshed periodically, so that the regression target does not shift with every gradient step. Replay requires an off-policy algorithm, and Q-learning is off-policy; in SARSA the stored next action is based on an old policy, and the replayed update is biased.

**The deadly triad.** The combination of function approximation, bootstrapping and off-policy learning does not have a guarantee of convergence, and can diverge [6, §11.3]. DQN contains all three; replay and the target network are engineering solutions to the problem of instability, not theoretical solutions, and whether they are necessary is an empirical question, and that is why we ablate them. On Atari, Fedus et al. [13] demonstrated that the replay ratio (gradient steps per transition) and the age of the data in the buffer affect the performance of DQN; our budget axis (Section 6.4) poses a similar question on a small problem.

**Overestimation and Double DQN.** The max in the target is computed on noisy estimates, with an expected maximum that is greater than the maximum of their expectations; Thrun and Schwartz [12] bound the bias and demonstrate that it increases with the discount factor as it accumulates through bootstrapping. Double DQN [2] separates selection and evaluation: online network selects a′ = argmax Q(s′, ·; θ) and target network evaluates Q(s′, a′; θ⁻). Van Hasselt et al. define the bias as the difference between max_a Q and the discounted return that the greedy policy subsequently realises; Section 4.5 records the same quantity.

**Dueling and prioritised replay.** The Dueling architecture [3] has two output streams and writes Q(s, a) = V(s) + A(s, a) − mean_a A(s, a), for problems where the choice of action does not always matter much; Snake, with three actions that are usually all safe, is such a situation. Prioritised experience replay (PER) [4] samples transitions proportional to (|δ| + ε)^α, where δ is the last temporal-difference error, and corrects the bias with importance weights (N P(i))^−β normalised by the maximum weight in the buffer, β annealed to 1.

**Reward shaping.** For any possible Φ, adding the reward F(s, s′) = γ Φ(s′) − Φ(s) to the reward function does not change the optimal policies [5] because the additional terms telescope along any trajectory. A bonus for moving towards the food, with no matching penalty for moving away is not of this form and in principle allows a policy to accumulate reward by circling. We include both, intentionally.

**Snake, and statistics.** The 11-feature representation we refer to as observation A is the one that was popularised by a widely copied PyTorch tutorial [11]; we apply it unchanged without claiming anything for it. The two non-learning players (Section 3.4) are not literature, but rather folklore. Henderson et al. [9] demonstrated that two groups of five seeds of the same algorithm can show a "significant" difference and suggested more seeds, intervals and paired designs; we follow them.

## 3 The environment and the interface

### 3.1 Rules

The environment is a W × W grid (where the most commonly used board is 10 × 10, and the quick one is 8 × 8), implemented using Gymnasium API [10] and registered as `Snake-v0`. The snake begins at length 3 in the middle, heading to the right. At any given moment there is a single food item on the board; when it is eaten it reappears uniformly at random on an empty cell, thus a policy needs to generalise, not memorise boards. An episode score is the number of food items consumed. The internal state is a single integer array `body_age`: 0 if the cell is empty, k > 0 if the cell is occupied for k more steps (tail = 1, head = length) which gives body, head, tail and heading in a single array, making the rule "the tail cell may be entered because it vacates during the move" follow from the update order.

**Actions.** There are 3 relative actions (straight, turn left, turn right), and a reversal is impossible by construction. In addition, a four-direction absolute mode exists as an ablation and was not applied in the matrix.

**Episode end.** Terminated: the head comes out of the grid (wall), collides with the body (self), or the board is full (win). Truncated: 100 steps with no food, the starvation limit H = max(100, W²). Truncation is not a death but a time-out and carries no penalty; all learners bootstrap from the successor state when an episode is truncated, as the agent cannot see the hidden step counter; considering the time-out as terminal would teach it that the state it was in has no future. Deaths are recorded by cause.

### 3.2 Observations: the first experimental axis

**Observation A** is the 11-feature vector: danger straight / left / right (would the next cell kill us; a cell with `body_age` 1 does not count, because it moves away), heading one-hot (4), food above / below / left / right of the head (4). Only 256 of the 2¹¹ codes can occur. It is compact and standard but not Markov: the structure of the body is not stored, so two boards with different body folding will have the same observation and different futures.

**Observation B** is a stack of W × W planes: head, body (binary), food, `body_age` / length (a continuous plane), and four broadcast heading planes (8 planes, "B8"). A single binary snapshot of the body is not Markov either and the reason is more precise than the Part B plan suggested: a binary body plane loses the order of the body, which means that neither the neck (which fixes the heading) nor the tail (which fixes the cell that vacates next) can be recovered from it. A heading plane only repairs the neck, the age plane repairs both, and that is the reason why the age plane is the Markov-restoring ingredient in our design and why the heading planes are redundant. This yields two cheap ablations: B7 (no age plane, so the tail is unknown) and B3 (head, body, food only; no explicit heading). A test examines whether the same grid with different headings, and the same grid and heading with different tails, yield different observations under B8 and whether their successor states actually differ.

### 3.3 Rewards: the second experimental axis

- **R0**, sparse: +1 for food, −1 for death (wall or self), 0 otherwise. The objective.
- **R1**, step penalty: R0 − 0.01 per step.
- **R2**, potential-based shaping: R0 + γ Φ(s′) − Φ(s) with Φ(s) = −(Manhattan distance from head to food) / (2 (W − 1)), so that Φ ∈ [−1, 0]; γ is the agent's discount factor.
- **R3**, naive distance shaping: R0 + 0.01 if distance to the food is decreased by the move, 0 otherwise. The plan had specified ±0.01 (closer / farther), but we changed the default to the asymmetric form since the symmetric form is a potential-based shaping with Φ = −0.01 d and γ = 1, which is about 10⁻⁴ per step away from R2 and cannot be exploited, so it would not have fulfilled its purpose. The symmetric version is stored as a configuration and was also executed. In the asymmetric R3, a closed loop collects reward: each step on the grid shifts the distance by exactly ±1, so the number of closer steps is equal to the number of farther steps in any closed loop, thus the average reward collected is 0.01 / 2 = 0.005 per step, worth 0.01 / (2 (1 − γ)) = 0.5 at γ = 0.99 if continued infinitely.

Under all rewards, scores are always counted in food eaten.

### 3.4 Reference players

Random; **BFS**, which traces a breadth-first shortest path from head to food on the current board (the body is an obstacle; cells that vacate in time are passable), and when no path exists, takes any safe move, and when no safe move exists, goes straight; and the **Hamiltonian cycle**, which traces a fixed cycle through all W² cells and therefore never dies, and always fills the board. On 10 × 10 they score 0.18, 30.6 and 97 (the board less the initial length). BFS dies by trapping itself in 84 % of its games and the average number of steps taken by the Hamiltonian player is 2,410 per game, which brackets what a learned policy can hope for: eat as quickly as the planner, and live as long as the cycle.

### 3.5 Implementation and tests

There are two implementations that share one rule set, one observation module and one reward module: the scalar reference environment (`snake_rl/env/core.py`, written for readability; used by the tests, the evaluation protocol, the behaviour analysis and the interface) and the vectorised NumPy environment that steps N boards at once as one (N, W, W) array with automatic reset (used for training, with 128 boards). A regression test compares the two step by step, bit by bit, while a differential fuzz of 15.8 million transitions did not spot any discrepancy; the vectorised environment steps 0.6–1.1 million boards per second on observation A, where the environment stepping is taken out of the budget (Section 5.4). Other implementation checks and the size of the package and test suite are in Appendix A.

### 3.6 The interface

The demonstration tool is an application that uses pygame in the same scalar environment (`python -m snake_rl.gui.app`) and contains the 3 modes of the plan. In human play the arrow keys move the snake, optionally with a learned agent as "advisor" whose action values are displayed for the human's position. In agent play, any reference player, Q table or DQN checkpoint plays live, while a panel displays the three action values, the action selected, the 11 features of observation A (lit squares), the score, steps since last food vs the starvation limit, and the running death-cause counts (Figure 1). Replay re-executes a recorded episode exactly. The evaluation seeds are used in episodes, so a human and an agent are comparable on the same board. The logic of this application is executed without a window and is covered by 56 tests. It was presented in Part D and is the instrument of the demonstration (Part F); it will not be discussed further here.

@figure runs/figures/partD_gui_agent.png
Figure 1. Agent-play mode of the interface: the DQN of the Part C submission (observation A, 10 × 10) at step 90 of evaluation board 10000, score 13. The bars are the action values of the current state: turn left is −0.97 because the cell to the left is the body (danger-left lit); the food is three cells above the head (food-up lit), behind that body, so neither safe move approaches it, and straight (+0.82) is preferred to right (+0.76) by a small margin. (The caption of the same frame in our Part D document misread the food as straight ahead.)

## 4 Learning agents

### 4.1 Tabular Q-learning

One table entry per (state, action) on the 256 reachable codes of observation A; ε-greedy exploration with linear decay of ε from 1.0 to 0.05 across the first 20 % of the steps; α = 0.1, γ = 0.9, selected on an 8 × 8 sweep of 105 runs over α × γ × ε schedules (Appendix C) where γ = 0.99 is unreliable within the budget since greedy snapshots start circling until the starvation limit. There are 50 environments stepped in parallel, and any transitions that end up in the same table cell within one step have their TD errors averaged, which in one environment reduces exactly to the textbook update (one test verifies this). A run takes about 9 s on 10 × 10. The results include a control table trained with γ = 0 (no look-ahead at all).

### 4.2 DQN

The same 11 features feed an MLP 11–256–256–3 (observation A; 69,635 parameters); observation B feeds a CNN of three 3 × 3 convolutions (stride 1, padding 1, channels 32 / 64 / 64, no pooling), flattened into a 512-unit layer and the 3-way output (2,156,963 parameters on 8 × 8, 3,336,611 on 10 × 10; the Atari architecture of the course material leaves a 1 × 1 map on a 10 × 10 input, hence only the kernel hyper-parameters were changed). The training settings are outlined in Table 1; the ones that play an important role below are that each vectorised step of the 128 boards is followed by G = 32 gradient steps on batches of 256, from a buffer of 100,000 transitions, such that each transition is applied about 63 times (we refer to this as the replay ratio), that the target network is replicated every 1,000 gradient steps (or about 4,000 environment steps), that γ = 0.9 in the main line with 0.99 on the discount axis. A gradient clipping at 10 is never active (logged norm is never more than 1.5). The MLP is trained on the CPU, and the cost of a gradient step is approximately the same as on the GPU, and runs are bit-reproducible; the CNN trains on the GPU (timings in Appendix A).

### 4.3 Ablations of the stabilisers

**No target network** implies the target copy is refreshed with each gradient step (`target_update_every = 1`), meaning that the target is the online network. **No replay** is not defined in the DQN literature and the definition is important, ours makes exactly one gradient step per vectorised step on the 128 transitions that step just generated, each transition used exactly once. Compared to the full agent this changes three things at once, which two controls split: (a) the replay ratio (63 → 1), (b) the time-mixing of the batch (transitions from about 780 recent vectorised steps against 1), and (c), a subtle one: the staleness of the target network, whose period of 1,000 gradient steps becomes 128,000 environment steps when there is only one gradient step per vectorised step. The control `t31` uses no replay, but sets the target period to 31 gradient steps, which is the same as the main line (4,000 environment steps); `G1` keeps replay but makes one gradient step per vectorised step, while `G1_t31` does both. These short names are used in Table 6.

### 4.4 Refinements

**Double DQN**: the target becomes r + γ Q(s′, argmax_a Q(s′, a; θ); θ⁻). **Dueling**: the final hidden layer is divided into a value stream and an advantage stream (each 256 units on A and each 512 units on B), which are combined as Q = V + A − mean A thus doubling the number of parameters of the head (A: 135,684; B on 8 × 8: 4,255,140). **PER**: the proportional version of Section 2 using α = 0.6, β from 0.4 to 1 as the training progresses, ε = 10⁻⁶, new transitions at the maximum priority and stratified sampling, implemented using a vectorised cumulative sum rather than a sum tree. All are a switch of the same agent and not a different implementation.

### 4.5 Measuring overestimation

For every evaluation, for each step in every greedy episode, we record v_t = max_a Q(s_t, a) and the reward, and after the episode the realised discounted return G_t = r_t + γ G_{t+1} (G = 0 following the last step including a starvation time-out: this is what the policy actually achieved). Two summaries are recorded: the pair (v_0, G_0) at the initial state, averaged across the episodes, and the pair averaged over all visited states (where long episodes have more weight as in [2]). The difference v − G is the overestimation, and the ratio v / G is a scale-free version. Runs conducted prior to this measurement were back-filled using the run's checkpoints, and the check was made that the back-filled score of every checkpoint was equal to the logged score of the same evaluation.

Table 1. Training settings of the learners. All DQN variants share these values unless a row of the matrix changes exactly the named key.

| Setting | Tabular Q-learning | DQN (MLP on A) | DQN (CNN on B) |
|---|---|---|---|
| Discount γ | 0.9 (0 for the control) | 0.9 main line; 0.99 on the discount axis | same |
| Step size / optimiser | α = 0.1 | Adam, 2.5 × 10⁻⁴, Huber loss, clip 10 | same |
| Exploration | ε 1.0 → 0.05, linear over the first 20 % of the steps; ε = 0 at evaluation | same | same |
| Parallel environments | 50 | 128 | 128 |
| Environment steps (8 × 8 / 10 × 10) | 500,000 / 1,000,000 | 512,000 / 1,024,000 | same |
| Network | table, 256 codes × 3 | 11–256–256–3 | conv 3 × 3 (32, 64, 64) – 512 – 3 |
| Replay / batch / warm-up | – | 100,000 / 256 / 10,000 steps | same |
| Gradient steps per vectorised step (G) | one update per step of the 50 boards | 32 (125,504 / 253,504 per run) | same |
| Target network | – | copied every 1,000 gradient steps (≈ 4,000 environment steps) | same |
| Device | CPU | CPU, 2 threads | GPU (RTX 5090) |
| Evaluation | every 25,000 / 50,000 steps | every 12,800 steps (41 / 81 evaluations) | same |
| Wall-clock per run (8 × 8 / 10 × 10) | 6 s / 9 s | 5 min / 10 min | 17–37 min / 37–53 min with five runs sharing the GPU |

## 5 Experimental protocol

### 5.1 Evaluation and statistics

All players, learned or not, are tested greedily (ε = 0) on the same 100 episodes: episode i is seeded with 10000 + i, which fixes the position of the food; these seeds are not used in training. All learners are trained with seeds 0–4 (a seed fixes the initialisation, the training boards and the exploration stream) and a configuration that is different from the main line in one key is trained with the same 5 seeds, thus comparisons between configurations are paired. We report the mean over the five agents of their mean score on the 100 boards with a 95 % Student-t interval over the five (t₀.₉₇₅,₄ = 2.776), for the policy after the last update ("final") and, because the greedy snapshots jitter by approximately one point between evaluations, for the mean of each agent's last five evaluations ("last-5", the last 51,200 steps). The reference players have one policy, thus their interval is a normal interval over the 100 episodes; the two types of interval cannot be compared by overlap. The differences between configurations are paired by seed; a learner is compared with BFS by pairing on boards (agent minus BFS on the same 100 boards, averaged, then a t interval over seeds). In addition to this, for small differences, we also report the sign count over seeds, leave-one-out stability and the last-10 and last-20 windows, as well as report when they differ. All absolute scores depend on the same 100 boards, and if these boards are resampled, then all the rows would move about ±1 together, which is why the comparisons are paired. Death causes are the proportions of the 500 final-evaluation episodes which end at a wall, within the body, at the starvation limit, or when the board is filled.

### 5.2 Budgets and the matrix

Budgets are set per board (Table 1) and are not tuned per configuration; some curves are still rising when the budget ends, and we state which. The matrix (`snake_rl/analysis/matrix.py`) defines 64 configurations, each a configuration file and overridden keys: 48 on 8 × 8, and 16 on 10 × 10, as outlined in Appendix B; the 10 × 10 board has the Part C pair, the representation axis, B8 at γ = 0.99, the component axis with two controls, and the reward axis on B8 at γ = 0.99; the 8 × 8 board has everything else. Each row contains five completed seeds (320 runs, plus the 105-run tabular sweep: approximately 257 million environment steps) and the matrix verifies the stored configuration for each run against its definition.

### 5.3 Reproducibility

Each run writes its full configuration, seed, git commit and software versions before training begins, its evaluation curve during training, and the per-episode records of its final evaluation at the end; a run directory never gets overwritten. Re-running a configuration file reproduces the run bit for bit on the same machine and PyTorch build (tested for the tabular agent and for the MLP); regenerating tables and figures from stored logs is machine-independent, whereas across machines the result is another draw from the same distribution. The two commands provided in Appendix A are used to generate all tables and figures of this report from the run directories, one of which additionally compares every table with this text cell by cell; the notebook `notebooks/results.ipynb` contains the code of the study, trains in the session and repeats that reproduction from the exported log files (executed on the training machine, not yet on Colab). The hardware and timings can be found in Appendix A.

### 5.4 Where the time goes

The plan was based on the assumption that stepping the Python environment would be the limiting factor. It is not: with the vectorised environment, gradient updates consume 94–99 % of a main-line DQN run's wall-clock (the remainder is spent on evaluation), and environment stepping under 1 %, at any batch size from 64 to 1,024, as a single update costs about 2 ms (MLP, CPU) or 3 ms (CNN, GPU) of per-call overhead regardless of the size. Therefore, the number of gradient steps is the budget of a run, which is why G is an axis of its own; at G = 32, a 10 × 10 CNN run is 253,504 updates and 38 minutes, while at G = 1 it is 7,922 updates and 2.5 minutes.

## 6 Results

The answers are summarised in Table 2, and the evidence is provided in the sections that follow. Throughout, "feature-vector agent" refers to the DQN on observation A (MLP) and "convolutional agent" the DQN for observation B8 (CNN); tables are labelled by their respective short names A (MLP) and B8 (CNN).

Table 2. The questions and the answers. Effects are paired differences with 95 % t intervals over five seeds (final policy unless stated); "rising" means the curve had not flattened when the budget ended.

| Question (axis) | Expected (Part B) | Found | Board; still rising? |
|---|---|---|---|
| Representation: grid (B8) against features (A) | A fast with a low ceiling; B slow with a high ceiling | +1.71 [+1.26, +2.15] at γ = 0.9; +0.40 [−0.83, +1.64]; B learns 4–14 times slower; at γ = 0.99 the gap is 5–7 points | 8 × 8; 10 × 10 (B8 near plateau) |
| Which planes: age plane (B8 vs B7), heading (B8 vs B3) | age plane necessary (Markov) | age plane: 8 × 8 none, 10 × 10 −1.15 [−2.00, −0.30] (B7 better); heading: −4.27 / −4.57 | both boards |
| Discount factor (0.9 → 0.99) | not an axis | A: −0.20 [−1.35, +0.96]; B8: +4.91 [+3.48, +6.34] and +5.03 [+4.26, +5.80], all survival time; on 8 × 8 not distinguishable from BFS | both; γ = 0.99 rising |
| Target network | stabilises training | none: −target +0.20 [−0.21, +0.61] (8 × 8), +0.34 [−0.51, +1.19] (10 × 10); no effect on overestimation | both |
| Replay | stabilises training | −9.26 [−11.64, −6.88] (8 × 8), −4.09 [−6.04, −2.15] (10 × 10); three factors, target staleness the largest on one path | both; see Table 6 |
| Replay ratio (63 → 2 uses per transition, target period matched) | not an axis | −1.21 [−1.97, −0.45] (8 × 8); +2.34 [+1.07, +3.60] (10 × 10), the best 10 × 10 agent at γ = 0.9 | 10 × 10 rising |
| Double DQN | less overestimation, better score | overestimation −0.87 [−0.95, −0.79] of +2.53 (B8, γ = 0.99); −0.39 of +6.14 (A, γ = 0.99); none at γ = 0.9; score: no consistent direction | 8 × 8 |
| Dueling; PER | better score | Dueling: score unchanged, B8 overestimation −0.08 [−0.11, −0.06]; PER: score −0.56 [−1.07, −0.06] (B8, last-5), slower | 8 × 8 |
| Reward: R1, R2, R3 against R0 | R2 safe, R3 exploitable | all within about one point; R2 fastest start, worst finish (−1.10 [−1.56, −0.63], B8 γ = 0.99, last-10); R1 −0.95 [−1.54, −0.37] on 10 × 10; R3 ≥ R0, weakly | both |
| Reward hacking under R3 | expected | none at bonus 0.01 or 0.1; loops are value-estimation errors (Q in loops ≪ loop value) | 8 × 8 |

### 6.1 The scale, and the tabular reference

Table 3 places each headline configuration between the reference players on both boards, and Figure 2 displays the per-episode distributions behind the means. To understand all that follows, three observations about the scale are required.

Table 3. Headline results. Final = the policy after the last update; last-5 = mean of each agent's last five evaluations; both are means over five agents with a 95 % t interval over agents. Reference players: mean over the 100 episodes with a normal interval. Steps are means over the 500 (or 100) final-evaluation episodes; food per 100 steps is 100 × food / steps over a seed's 100 episodes, averaged over seeds; the last column pairs the agent with BFS on the same 100 boards (mean over boards per seed, t interval over seeds).

| Player | Score, final | Score, last-5 | Steps per episode | Food per 100 steps | Wall / self / starve (%) | Agent − BFS, same boards |
|---|---|---|---|---|---|---|
| **10 × 10 board** | | | | | | |
| Random | 0.18 ± 0.08 | – | 23.0 | – | 98 / 1 / 1 | – |
| BFS greedy | 30.61 ± 1.60 | – | 255.8 | 11.97 | 16 / 84 / 0 | – |
| Hamiltonian cycle | 97 (fills the board) | – | 2,409.7 | 4.03 | 0 / 0 / 0 | – |
| Tabular Q-learning (γ = 0.9) | 18.51 ± 1.10 | 18.14 ± 0.29 | 150.6 | 12.29 | 10 / 90 / 0 | −12.10 [−13.20, −11.00] |
| Tabular Q-learning, γ = 0 (control) | 19.69 ± 0.00 † | 19.69 ± 0.00 † | 239.2 | 8.23 | 4 / 96 / 0 | −10.92 † |
| DQN, observation A (MLP) | 20.29 ± 1.55 | 20.44 ± 1.41 | 167.8 | 12.10 | 16 / 84 / 0 | −10.32 [−11.87, −8.77] |
| DQN, observation B8 (CNN) | 20.69 ± 0.52 | 20.85 ± 0.59 | 187.6 | 11.03 | 7 / 93 / 0 | −9.92 [−10.44, −9.40] |
| DQN, B8, G = 1, target period 31 (`G1_t31`) | 23.03 ± 1.11 | 22.92 ± 0.78 | 196.3 | 11.73 | 1 / 99 / 0 | −7.58 [−8.69, −6.47] |
| DQN, B8, γ = 0.99 | 25.72 ± 0.83 | 25.46 ± 0.40 | 333.7 | 7.71 | 3 / 94 / 3 | −4.89 [−5.72, −4.07] |
| **8 × 8 board** | | | | | | |
| Random | 0.23 ± 0.10 | – | 13.2 | – | 98 / 2 / 0 | – |
| BFS greedy | 23.07 ± 1.13 | – | 151.4 | 15.24 | 35 / 65 / 0 | – |
| Hamiltonian cycle | 61 (fills the board) | – | 985.9 | 6.19 | 0 / 0 / 0 | – |
| Tabular Q-learning (γ = 0.9) | 15.67 ± 0.87 | 15.67 ± 0.40 | 102.0 | 15.37 | 19 / 81 / 0 | −7.40 [−8.28, −6.53] |
| Tabular Q-learning, γ = 0 (control) | 16.07 ± 0.10 † | 16.07 ± 0.10 † | 151.5 | 10.61 | 7 / 93 / 0 | −7.00 [−7.10, −6.90] † |
| DQN, observation A (MLP) | 16.11 ± 0.88 | 16.07 ± 0.64 | 104.9 | 15.36 | 21 / 79 / 0 | −6.96 [−7.85, −6.08] |
| DQN, observation B8 (CNN) | 17.81 ± 0.90 | 17.63 ± 0.28 | 131.4 | 13.55 | 5 / 95 / 0 | −5.26 [−6.16, −4.36] |
| DQN, B8, γ = 0.99 | 22.72 ± 1.04 | 22.39 ± 0.55 | 220.3 | 10.33 | 3 / 96 / 1 | −0.35 [−1.39, +0.70] |
| DQN, B8, γ = 0.99, R3 with bonus 0.1 | 24.44 ± 1.07 | 24.10 ± 0.57 | 237.0 | 10.32 | 4 / 96 / 1 | +1.37 [+0.29, +2.44] |

† With γ = 0, the 5 agents converge to (almost) the same deterministic policy, thus the spread across seeds is not a useful indicator, and the paired interval against BFS is degenerate; the true uncertainty of that row is the choice of the 100 boards, which is around ± 1.

@figure runs/report/figures/score_distributions.png
Figure 2. Score per episode of the final policies on 8 × 8 (left) and 10 × 10 (right): box = quartiles and median, whiskers = 1.5 interquartile ranges, dots = outliers; reference players over their 100 episodes, learners over the 500 episodes of their five seeds. The learned policies' distributions have the planner's shape shifted down; only the γ = 0.99 agent on 8 × 8 and the probe overlap BFS, and a few of their episodes score 0–2 (loops; Section 6.8). The Hamiltonian cycle scores the same in every game.

**The tabular agent and the DQN on the same features are at the same level.** On 8 × 8 they cannot be distinguished (DQN − table: +0.44 [−0.59, +1.47], a two-sample t interval, since the two sets of agents are independent); on 10 × 10 the DQN is higher (+1.78 [+0.17, +3.39] final, +2.30 [+0.91, +3.69] last-5). Both are 60–70 % of BFS, and both die in a similar manner: in all 500 final episodes of either learner on either board the snake dies in some spot where all 3 moves are deadly, and in every reachable observation code with a safe move, the greedy action is safe. They do not walk into an obstacle they can see; they enclose themselves, and the 11 features can only show them at the last step.

**That level needs no look-ahead.** A table trained with γ = 0, that only seeks the immediate reward, performs the same as the γ = 0.9 table on 8 × 8 (16.07 versus 15.67) and better on 10 × 10 (19.69 versus 18.51) within the DQN's interval on both boards. The score on these features represents a reactive policy ("do not take a blocked move; turn towards the food when it is in line"), and no evidence of multi-step value learning can be read on observation A. The control is inexpensive and we recommend it to anyone who reports a DQN on the 11 features.

**The level is probably not the ceiling of the features.** A hill-climbing search over deterministic policies over the same 256 codes, run from a learned table and tested on boards not used in the search, was able to obtain policies with scores of 28 on 8 × 8 and 33 on 10 × 10 within minutes, outperforming BFS. This is preliminary (not a logged run of the matrix), but the limit appears to be the learning method (one-step temporal-difference learning on aliased states cannot assign being enclosed to the moves that caused it), rather than what the 11 features can represent; the learned scores on A are a lower bound of the representation.

### 6.2 State representation

The representation axis is provided in Table 4 and in Figure 3 at the main-line discount. The convolutional agent outperforms the feature-vector agent by 1.7 points on 8 × 8 (+1.71 [+1.26, +2.15] final, every seed higher; +1.56 [+1.08, +2.04] last-5) and by nothing measurable on 10 × 10 (+0.40 [−0.83, +1.64]), where the feature-vector agent's seed spread (± 1.55) conceals anything under about a point. We choose to interpret the difference between the boards as a property of the training configuration, rather than of the budget: the convolutional agent's mean curve is flat over the final ten evaluations on 10 × 10 (−0.05 points per 100k steps; +0.34 over the last twenty; three seeds still climbing, thus near a plateau but not converged), and the same network reaches 23 with 32 times fewer gradient steps (Section 6.4) and 25.7 at γ = 0.99 (Section 6.3). The 20–21 point level is only one point above the no-look-ahead level of the γ = 0 table.

**Speed.** The feature-vector agent has reached its plateau in its first two or three evaluations (at most 40k steps, the first gradient step is at 10,100). B8 and B7 first reach 12 points after 90k–115k steps on 8 × 8 and 166k–179k on 10 × 10, B3 later still (128k–154k and 218k–256k): 3.5–9 times (8 × 8) and 6.5–14 times (10 × 10) slower than observation A to the same level. The ratio is appropriate because the two share the same ε schedule and the replay ratio. It is confirmed that the expected trade-off, "A learns fast with a low ceiling, B learns slowly with a high ceiling", applies to speed, and, at γ = 0.9, weakly applies to the ceiling.

**Which planes matter.** Removing the heading information (B3) costs 4.3 points on 8 × 8, 4.6 points on 10 × 10 and slows learning by 40k–370k steps; since the body plane plus the head allows the network to infer the neck in most states, this is the price of inferring the heading from two planes plus the price of the genuinely ambiguous states, which we are unable to separate. Removing the age plane (B7), which by the Markov argument of Section 3.2 should hurt (because the tail becomes unobservable), does not hurt: on 8 × 8 the two cannot be distinguished (|effect| < about 1 point), and on 10 × 10 the point estimate favours B7 in each window (+1.15 [+0.30, +2.00] final, all five seeds; +0.64 [−0.24, +1.53] last-5; the final-window significance relies on one seed). The theoretical non-Markov property of B7 does not have a measurable cost at this level of play; we read that the tail only becomes important when the snake is about to enter the vacating cell, which is rare at 18–22 points, whereas a continuous-valued plane makes the fitting function less regular for the CNN.

**The dip of the feature-vector agent on 10 × 10.** The feature-vector agent's mean curve on 10 × 10 declines from 19–21 (25k–130k steps) to the table's level, 18.1–18.9, between approximately 290k and 450k steps, and recovers after 700k (Figure 3). The dip is seen in all three feature-vector variants and in none of the seven convolutional variants, so the buffer (by then containing only ε = 0.05 data) cannot be the cause. It corresponds to the correction of an early peak in the value estimates (visited-state estimate 1.48 → 1.31 between 256k and 384k steps while the realised return remains stable at 1.08–1.10), which we read but have not shown, as optimistic values learned during exploration being unlearned; the convolutional runs get to their plateau only after ε has bottomed out, so "a property of observation A" and "a property of runs that plateau before ε does" cannot be separated.

Table 4. The representation axis (DQN, γ = 0.9, 5 seeds). Paired differences are against B8 on the same seeds, final / last-5.

| Observation | 8 × 8, final | 8 × 8, last-5 | 8 × 8, minus B8 | 10 × 10, final | 10 × 10, last-5 | 10 × 10, minus B8 |
|---|---|---|---|---|---|---|
| A: 11 features, MLP | 16.11 ± 0.88 | 16.07 ± 0.64 | −1.71 [−2.15, −1.26] / −1.56 [−2.04, −1.08] | 20.29 ± 1.55 | 20.44 ± 1.41 | −0.40 [−1.64, +0.83] / −0.40 [−1.47, +0.66] |
| B3: head, body, food | 13.54 ± 0.61 | 13.48 ± 0.36 | −4.27 [−5.39, −3.15] / −4.15 [−4.58, −3.71] | 16.12 ± 0.25 | 15.81 ± 0.33 | −4.57 [−5.09, −4.05] / −5.04 [−5.64, −4.45] |
| B7: B8 without the age plane | 18.18 ± 0.85 | 17.93 ± 0.81 | +0.37 [−1.00, +1.74] / +0.30 [−0.40, +1.00] | 21.84 ± 0.74 | 21.49 ± 0.43 | +1.15 [+0.30, +2.00] / +0.64 [−0.24, +1.53] |
| B8: head, body, food, age, heading × 4 | 17.81 ± 0.90 | 17.63 ± 0.28 | – | 20.69 ± 0.52 | 20.85 ± 0.59 | – |

@figure runs/report/figures/obs_W10_curve.png
Figure 3. The representation axis on 10 × 10: greedy score during training, mean over five seeds with a 95 % t band; BFS as the dashed line. Observation A is at its level after the first evaluations and dips between 290k and 450k steps; B8 and B7 reach 15 points after about 230k steps and A's level after about 500k; B3 (no heading) stays 4–5 points lower throughout.

### 6.3 Discount factor: the largest effect, and it depends on the representation

The plan began from γ = 0.99 and the project proceeded to 0.9 for the tabular reference (Section 4.1); γ = 0.99 was then run as a hyper-parameter row and became the most significant effect in the matrix (Table 5, Figure 4). On observation A the discount does nothing (8 × 8: −0.20 [−1.35, +0.96]), as predicted by Section 6.1. On observation B8 it adds +4.91 [+3.48, +6.34] on 8 × 8 and +5.03 [+4.26, +5.80] on 10 × 10, every seed in the same direction, and on 8 × 8 the resulting agent cannot be distinguished from BFS on the same boards (−0.35 [−1.39, +0.70]; one seed of five above it). The γ = 0.99 curves are still climbing at the end of the budget (last-20 slopes +0.95 ± 0.41 and +0.66 ± 0.38 points per 100k steps, against +0.48 and +0.34 at γ = 0.9), thus they are fixed-budget numbers.

**Where the five points come from.** Not from eating faster: the γ = 0.99 agent consumes 24–30 % less food per 100 steps (10.33 vs 13.55 on 8 × 8; 7.71 ± 0.26 vs 11.03 ± 0.11 on 10 × 10, paired −3.32 [−3.64, −3.00]) and survives 68–78 % longer (220 vs 131 steps; 334 vs 188, paired +146 [+130, +162]). On 10 × 10 it outlives BFS (334 vs 256 steps), but eats significantly slower (7.7 vs 12.0 per 100 steps), so it still lags behind the planner there by 4.9 points; on 8 × 8 the two trade-offs balance. At γ = 0.99 the convolutional agent ends 1 % (8 × 8) and 2–3 % (10 × 10, under each reward) of its episodes at the starvation limit, against 0 % at γ = 0.9 (paired +0.03 [+0.01, +0.05] on 10 × 10); Section 6.8 illustrates what those episodes are. The highest single episode on 10 × 10 has a score of 42 (BFS: 58; medians 26 vs 31); the highest of any learned policy is 46.

**Overestimation grows with γ.** Along the states its greedy policy visits, the convolutional network's max_a Q surpasses the realised discounted return by 0.24–0.30 (25–26 %) at γ = 0.9 and by 50 % (8 × 8) to 78 % (10 × 10) at γ = 0.99 (Figure 6b); on observation A at γ = 0.99 the difference is 114 %, with no score to show for it. At the end of training, the estimate is still increasing but slows down (on 10 × 10, +1.17, +0.65, +0.56, +0.32 and +0.15 per successive 128k steps while the realised return remains at 4.2–4.7) which suggests convergence towards a finite bias rather than drift.

**Qualifications.** The 10 × 10 matrix lacks an A × γ = 0.99 cell, thus the interaction was established on 8 × 8 and the B8 main effect on both boards; B7 and the low-replay-ratio configuration, which were already 1–2 points above the B8 main line at γ = 0.9, were not run at γ = 0.99. This effect is specific to this task (a death costs −1, and only a lengthy horizon makes self-trapping ten steps ahead noticeable in the target), not a general recommendation for γ.

Table 5. The discount axis (DQN, 5 seeds). Paired differences are γ = 0.99 minus γ = 0.9 on the same seeds.

| | 8 × 8, γ = 0.9 | 8 × 8, γ = 0.99 | 8 × 8, paired | 10 × 10, γ = 0.9 | 10 × 10, γ = 0.99 | 10 × 10, paired |
|---|---|---|---|---|---|---|
| A (MLP), final | 16.11 ± 0.88 | 15.91 ± 0.60 | −0.20 [−1.35, +0.96] | 20.29 ± 1.55 | not run | – |
| B8 (CNN), final | 17.81 ± 0.90 | 22.72 ± 1.04 | +4.91 [+3.48, +6.34] (5/5) | 20.69 ± 0.52 | 25.72 ± 0.83 | +5.03 [+4.26, +5.80] (5/5) |
| B8, last-5 | 17.63 ± 0.28 | 22.39 ± 0.55 | +4.76 [+4.45, +5.08] | 20.85 ± 0.59 | 25.46 ± 0.40 | +4.61 [+4.11, +5.12] |
| B8, steps per episode | 131.4 | 220.3 | | 187.6 | 333.7 | +146 [+130, +162] |
| B8, food per 100 steps | 13.55 ± 0.13 | 10.33 ± 0.45 | | 11.03 ± 0.11 | 7.71 ± 0.26 | −3.32 [−3.64, −3.00] |
| B8, starvation rate | 0.00 | 0.01 | | 0.00 | 0.03 | +0.03 [+0.01, +0.05] |
| B8, overestimation on visited states (ratio) | +0.30 (1.26) | +2.53 (1.50) | | +0.24 (1.25) | +3.68 (1.78) | |
| B8, minus BFS on the same boards | −5.26 [−6.16, −4.36] | −0.35 [−1.39, +0.70] | | −9.92 [−10.44, −9.40] | −4.89 [−5.72, −4.07] | |

@figure2 runs/report/figures/gamma_W8_curve.png | runs/report/figures/gamma_W10_curve.png
Figure 4. The discount axis on 8 × 8 (left) and 10 × 10 (right). At γ = 0.9 the other learners (A at both discounts, B8 at γ = 0.9) sit within two points of each other; the convolutional agent at γ = 0.99 separates from them after 150k (8 × 8) or 350k (10 × 10) steps and ends within the planner's interval on 8 × 8.

### 6.4 The stabilisers: replay, target network, and how many gradient steps

The component axis on the convolutional line is shown in Table 6 and Figure 5 while the feature-vector line is summarised at the end.

**The target network does not help.** Training with no target network (the target refreshed after every gradient step) is never worse: on 8 × 8 it is somewhat better in each window (final +0.20 [−0.21, +0.61]; last-5 +0.53 [+0.01, +1.06]; last-10 +0.47 [+0.17, +0.78]; last-20 +0.31 [+0.12, +0.51], five seeds of five), on 10 × 10 the direction is the same with intervals that cross zero (+0.34 / +0.61), and on observation A likewise (8 × 8: +0.52 [−0.07, +1.10] last-5; 10 × 10: +0.44 [−2.67, +3.55]). Nothing diverges (the mean Q on training batches remains at 1.5 in either case) and there is no change in overestimation (Section 6.5). The target copy only delays value propagation at the period of 4,000 environment steps.

**Replay matters, but one curve cannot say why.** Removing replay as described in Section 4.3 reduces the 8 × 8 agent from 17.8 to 8.6 (−9.26 [−11.64, −6.88]), with a seed spread of 4.8–10.8 points and 42 % of the final episodes ending in starvation; on 10 × 10 the drop is 4.1 points. The controls were used to separate the three things that changed. Matching the target period in environment steps (`t31`) recovers 5.5 points on 8 × 8 and 5.1 on 10 × 10, where the no-replay agent then matches the full agent in the final and last-5 windows (+1.01 [−0.09, +2.10], −0.18 [−1.11, +0.76]; still −1.05 in the last-20 window) and is climbing at 1.5 points per 100k steps. The largest loss on this path is the target period, not the lack of replay, since for 1 gradient step per vectorised step the target is refreshed three times in an entire 8 × 8 run (Section 4.3), and bootstrapping from a target that is random for 138k steps and then 118k steps old keeps the values five to nine times too low (mean Q 0.17–0.31 against 1.5), not divergence but its opposite; the same happens with replay when only the target period is stale (`G1`: 13.7 ± 3.4). The other path has its mixing step as the larger step (+5.18 [+0.84, +9.53] from `no replay` to `G1`), thus an interaction of the two factors occurs (−2.62 [−8.31, +3.07]): a stale target is more damaging with no replay. All three factors are measurable, and their shares depend on the path, so we provide both paths, not a split.

**Fewer gradient steps.** The other control, replay with one gradient step per vectorised step and target period kept at 4,000 environment steps (`G1_t31`, replay ratio 2 instead of 63), is 1.2 points below the full agent on 8 × 8 but the best agent at γ = 0.9 on 10 × 10: 23.03 ± 1.11 against 20.69 ± 0.52 (+2.34 [+1.07, +3.60] final, every seed; +2.07 [+0.88, +3.25] last-5), with one thirty-second of the gradient steps and one fifteenth of the wall-clock, and still climbing at one point per 100k steps when the full agent has flattened (Figure 6a). Overestimation moves with the replay ratio (gap on visited states +0.24 at 63 uses, +0.13 at 2, +0.06 at 1, and an underestimate of −0.38 with the stale target). The configuration changes two keys (G and the target period, matched in environment steps) and there is no 10 × 10 control that changes G only, but the full agent with the target refreshed every step is only +0.34, so the more frequent refresh cannot be the explanation for the +2.3. What we can say is that a higher replay ratio gains the speed in the first 400k steps and, in this budget, a lower final score on the large board; the mechanism we suspect, over-fitting the 100k most recent transitions, is not tested. This direction is consistent with Fedus et al. [13]. For comparability with the 8 × 8 experiments, the main line was not changed.

**The feature-vector line.** On observation A similar ablations are nearly null: eliminating replay costs 0.4–0.7 points on 8 × 8 (−0.71 [−1.14, −0.28]) and 2 points on 10 × 10 (`t31`: −2.07 [−3.83, −0.31]); the target network does nothing; the budget rows (G = 4 and 1: 15.39 and 15.02 vs 16.11) show that 63 uses of each transition obtain around half a point; a 16-unit hidden layer gets to the same level (15.60 ± 1.23) and a linear Q-function does not learn (3.04 ± 5.30, 76 % starvation). With no replay the values on A are three times too small (mean Q 0.5, against 1.5) at the same score, in accordance with Section 6.1: the score on A does not require the values to be correct.

Table 6. The component axis on observation B8 (5 seeds). The target period is in gradient steps, with its equivalent in environment steps; G is the number of gradient steps per vectorised step of the 128 boards; the short names are those used in the text.

| Configuration | Replay | G | Target period | 8 × 8, final | 8 × 8, last-5 | 10 × 10, final | 10 × 10, last-5 |
|---|---|---|---|---|---|---|---|
| B8, full | yes | 32 | 1,000 (≈ 4k env steps) | 17.81 ± 0.90 | 17.63 ± 0.28 | 20.69 ± 0.52 | 20.85 ± 0.59 |
| no target network | yes | 32 | 1 | 18.01 ± 0.67 | 18.16 ± 0.63 | 21.03 ± 0.55 | 21.45 ± 0.40 |
| no replay | no | 1 | 1,000 (= 128k env steps) | 8.55 ± 3.09 | 8.38 ± 3.32 | 16.60 ± 2.33 | 16.03 ± 2.81 |
| no replay, no target network | no | 1 | 1 | 13.90 ± 0.84 | 13.70 ± 0.65 | not run | – |
| no replay, target period matched (`t31`) | no | 1 | 31 (≈ 4k env steps) | 14.03 ± 1.39 | 13.98 ± 0.77 | 21.70 ± 1.07 | 20.67 ± 0.63 |
| replay, G = 1 (`G1`) | yes | 1 | 1,000 (= 128k env steps) | 13.73 ± 3.38 | 11.83 ± 2.15 | not run | – |
| replay, G = 1, target period matched (`G1_t31`) | yes | 1 | 31 (≈ 4k env steps) | 16.60 ± 0.82 | 16.11 ± 0.65 | 23.03 ± 1.11 | 22.92 ± 0.78 |

@figure2 runs/report/figures/components_B8_W8_curve.png | runs/report/figures/components_B8_W10_curve.png
Figure 5. Replay and target network on observation B8, 8 × 8 (left) and 10 × 10 (right). Without a target network the curve is indistinguishable from the full agent's. Without replay the agent learns nothing for 250k–500k steps; matching the target period in environment steps ("target 31") recovers most of the loss, and on 10 × 10 the two curves meet at the end of the budget.

@figure2 runs/report/figures/budget_W10_curve.png | runs/report/figures/gamma_W10_values.png
Figure 6. (a) One gradient step per vectorised step with the target period matched (`G1_t31`) against the full agent on 10 × 10: slower for 400k steps, then higher and still rising. (b) The overestimation measurement on 10 × 10: solid lines are the network's max_a Q on the states its greedy policy visits, dashed lines the discounted return it then realises. At γ = 0.9 the gap is 0.2–0.25; at γ = 0.99 the estimate climbs to 8.4 against a return of 4.7.

### 6.5 Double DQN, Dueling and prioritised replay

Table 7 presents the three refinements on 8 × 8 and the overestimation that was measured for each. In all configurations (Table 7; Section 6.3), the baseline DQN overestimates the return of its own greedy policy, every seed positive. The gap depends on where it is measured: for observation A, the start state is *under*-estimated (−0.21 at γ = 0.9), whereas the visited-state mean is over-estimated, which we assign to the states near death that the 11 features cannot tell apart from safe ones, and whose realised return has the discounted −1 (the CNN, which can view them, overestimates the start state as well, +0.17).

**Double DQN** acts on the overestimation only where there is a lot of it: for the CNN at γ = 0.99 it eliminates a third of the gap on states visited (+2.53 → +1.66, −0.87 [−0.95, −0.79], every seed; Figure 7) and brings the start-state estimate from +0.52 to unbiased (−0.13 ± 0.13); for the MLP at γ = 0.99, the biggest gap of the study, it removes 6 % (6.14 → 5.75); at γ = 0.9 it does nothing (A ± 0.00, B8 −0.03). The direction agrees with [2]; the magnitude is far smaller than in Atari, where Double DQN basically eliminates the bias, which we attribute, without having tested it, to our target period (1,000 gradient steps against 10,000 parameter updates in [2], 30,000 in their tuned variant): the two estimates on which Double DQN relies are dependent, since the online and target networks are highly correlated. On the score Double DQN lacks consistent direction: +1.1 on A at γ = 0.9 (four seeds out of five, not robust to omitting one), +0.5 on A at γ = 0.99, 0 on B8 at γ = 0.9, and a transient of up to −0.9 in the last-20 window on B8 at γ = 0.99, which shrinks to −0.03 and −0.34 in the last two evaluations, while the Double curve climbs more quickly. Here, measured overestimation and greedy score are not monotonically related: A's score increases without changing its gap while B8's gap declines with its score unaltered or slightly lower.

**Dueling** does not change the score on both representations (−0.66 [−1.65, +0.33] and +0.28 [−0.71, +1.26]) and it decreases the overestimation of the CNN by a small clean amount (+0.30 → +0.22, every seed), at the expense of doubling the parameters of the head; whether the reduction is due to the V / A decomposition or to the additional parameters we cannot separate. **PER** leaves the score unchanged on A and slightly reduces it on B8 in the tail windows (last-5 −0.56 [−1.07, −0.06]; last-20 −0.83 [−1.29, −0.36]) with later learning (first 15 points at 154k–218k steps against 128k–154k). The logs explain why: once the majority of transitions have a near-zero TD error, the importance weights normalised by the maximum weight in the buffer are very small for typical samples, and the weighted loss and gradient norm are three orders of magnitude smaller than the DQN's; Adam removes the scale (the real step length, read from the optimiser state, is just 1.4–1.5 times smaller), but the weighting within the batch is dominated by a few low-priority samples and only 7–9 % of the parameters get non-negligible gradient, against 49–82 %. The curve is meant to answer the question "what does PER as published do here" not "does prioritised sampling help"; a batch-normalised weight (as in Dopamine) would be the fair comparison and was not run.

Table 7. The refinements on 8 × 8 (5 seeds). Score differences are variant minus baseline, paired by seed, for the final / last-5 / last-20 windows; the overestimation gap is estimate minus realised return on visited states, mean of the last five evaluations, baseline → variant.

| Baseline → variant | Baseline score | Variant score | Score difference: final / last-5 / last-20 | Overestimation gap, visited states | Start-state gap |
|---|---|---|---|---|---|
| A, γ = 0.9 → Double | 16.11 ± 0.88 | 17.21 ± 0.59 | +1.11 [+0.12, +2.10] / +1.07 [+0.34, +1.79] / +0.68 [+0.02, +1.34] | 0.25 → 0.25 | −0.21 → −0.20 |
| A, γ = 0.99 → Double | 15.91 ± 0.60 | 16.37 ± 0.80 | +0.46 [−0.42, +1.34] / +0.57 [−0.00, +1.15] / +0.54 [+0.08, +1.00] | 6.14 → 5.75 (−0.39 [−0.58, −0.21]) | 2.05 → 1.54 |
| A, γ = 0.9 → Dueling | 16.11 ± 0.88 | 15.45 ± 0.87 | −0.66 [−1.65, +0.33] / −0.41 [−1.25, +0.43] / −0.34 [−1.03, +0.35] | 0.25 → 0.23 (−0.02 [−0.03, −0.01]) | −0.21 → −0.23 |
| A, γ = 0.9 → PER | 16.11 ± 0.88 | 15.84 ± 0.78 | −0.27 [−1.59, +1.06] / −0.14 [−1.38, +1.10] / −0.09 [−1.27, +1.10] | 0.25 → 0.27 | −0.21 → −0.20 |
| B8, γ = 0.9 → Double | 17.81 ± 0.90 | 17.69 ± 0.56 | −0.12 [−1.22, +0.98] / −0.06 [−0.54, +0.42] / −0.07 [−0.40, +0.26] | 0.30 → 0.27 (−0.03 [−0.06, −0.01]) | 0.17 → 0.14 |
| B8, γ = 0.99 → Double | 22.72 ± 1.04 | 22.38 ± 1.10 | −0.34 [−1.70, +1.02] / −0.50 [−1.20, +0.21] / −0.89 [−1.68, −0.11] | 2.53 → 1.66 (−0.87 [−0.95, −0.79]) | 0.52 → −0.13 |
| B8, γ = 0.9 → Dueling | 17.81 ± 0.90 | 18.09 ± 0.93 | +0.28 [−0.71, +1.26] / +0.21 [−0.44, +0.85] / +0.02 [−0.25, +0.28] | 0.30 → 0.22 (−0.08 [−0.11, −0.06]) | 0.17 → 0.16 |
| B8, γ = 0.9 → PER | 17.81 ± 0.90 | 17.14 ± 1.08 | −0.67 [−2.26, +0.91] / −0.56 [−1.07, −0.06] / −0.83 [−1.29, −0.36] | 0.30 → 0.28 | 0.17 → 0.15 |

@figure runs/report/figures/values_double_B8_W8_values.png
Figure 7. Double DQN against DQN on observation B8 at γ = 0.99 (8 × 8): the max-Q estimate on visited states (solid) against the realised return (dashed). Both estimates lag the return until about 250k steps and then overtake it; Double DQN's rises more slowly and ends a third closer to it. The realised returns of the two agents are the same.

### 6.6 Reward design

The reward axis on both representations and both discounts on 8 × 8, and on the convolutional agent at γ = 0.99 on 10 × 10 (Figure 8) are reported in Table 8. Reward design is a second-order effect in this environment: on 8 × 8 none of the 15 paired differences against R0 in the final window has an interval away from zero, on 10 × 10 all three are within one point, and the representation (+1.7 to +6.8 points depending on γ, the last being B8 at γ = 0.99 against A at γ = 0.99) and the discount (+5) dwarf them.

**Potential-based shaping (R2) starts fastest and ends worst**. At the first evaluation, 12,800 steps, the R2 agents already have a score of 16 on A (other variants 11–13) and 2.7–2.9 on B8 (others 0.5), since the shaping term provides a gradient towards the food from the first step. However, the convolutional agents under R2 reach 15 and 20 points later than under R0 (by 18–33k and around 100k steps) and finish lower: −1.10 [−1.56, −0.63] in the last-10 window at γ = 0.99 on B8, no seed positive, robust to leaving any seed out, the only reward effect that is; −0.5 at γ = 0.9 (significant in the last-20 window only); −0.40 [−1.04, +0.24] on 10 × 10, where the potential is flatter (1/18 against 1/14 per cell). The guarantee of [5] concerns the optimum and not learning with function approximation: under R2 the network must learn Q′ = Q − Φ, i.e. to compute the Manhattan distance from the planes to the food, and add it to each action value; and until it does, the γ Φ(s′) term of each food-eating step (depending on where the next food appears) is noise in the target. The value logs back this up: at γ = 0.99 the realised return increases by the telescoped potential (+0.21) and the estimate does not follow (−0.05 [−0.19, +0.08]); at γ = 0.9 the estimate does follow (+0.32) and the cost is half.

**The step penalty (R1) costs nothing on 8 × 8 and about a point on 10 × 10** (−0.95 [−1.54, −0.37] in the last-5 window; the last-10 and last-20 intervals cross zero). Episode length and death causes under R1 are those of R0 on both boards and there is no evidence of "suicide": the wall-collision rate is, if anything, lower (−0.02 [−0.04, −0.00] on B8 at γ = 0.99). At γ = 0.9, the discounted cost of wandering (−0.1) is far smaller in magnitude than the cost of dying (−1), and at γ = 0.99, where the two are equal, the agent eats within a few steps and never faces the choice.

**Naive distance shaping (R3) is not worse than R0 and is slightly better late in training, weakly**. On B8 at γ = 0.99 the last-5 and last-10 windows offer +0.61 [+0.14, +1.08] and +0.47 [+0.31, +0.62] (all seeds), the last-20 window +0.34 [−0.40, +1.08], and the final window +0.80 [−0.84, +2.43]: two seeds were a point lower than R0 at 270k–380k steps and passed it in the last 50k. On 10 × 10 the pattern is reversed (last-5 and last-10 cross zero; last-20 +0.57 [+0.09, +1.05], four seeds out of five). The symmetric R3 of the plan acts as R2 on A (−0.2 to −0.5, always) and as R0 on B8.

Table 8. The reward axis: score (food eaten) of the final policy, 5 seeds, and paired differences against R0. 8 × 8 differences are final / last-10; 10 × 10 differences are last-5.

| Reward | A, γ = 0.9 | A, γ = 0.99 | B8, γ = 0.9 | B8, γ = 0.99 | B8, γ = 0.99, 10 × 10 |
|---|---|---|---|---|---|
| R0, sparse | 16.11 ± 0.88 | 15.91 ± 0.60 | 17.81 ± 0.90 | 22.72 ± 1.04 | 25.72 ± 0.83 |
| R1, step penalty | 16.47 ± 0.30 | 16.10 ± 1.07 | 17.56 ± 0.46 | 22.72 ± 1.49 | 24.72 ± 0.64 |
| R2, potential-based shaping | 16.13 ± 0.41 | 15.62 ± 0.93 | 17.12 ± 0.62 | 21.59 ± 0.62 | 25.27 ± 0.71 |
| R3, naive distance shaping | 16.81 ± 1.36 | 16.10 ± 0.57 | 18.08 ± 0.89 | 23.52 ± 0.92 | 25.76 ± 0.67 |
| R3 symmetric (± 0.01) | – | 15.58 ± 0.64 | – | 22.88 ± 1.10 | – |
| R3, starvation limit 1,000 | – | 16.10 ± 0.57 | – | 23.15 ± 0.77 | – |
| R3, bonus 0.1 (probe) | – | – | – | 24.44 ± 1.07 | – |
| BFS greedy, same boards | 23.07 ± 1.13 | | | | 30.61 ± 1.60 |
| R1 − R0 | +0.37 [−0.80, +1.53] / +0.22 [−0.86, +1.29] | +0.19 [−1.23, +1.61] / +0.10 [−0.46, +0.66] | −0.25 [−1.25, +0.75] / −0.01 [−0.24, +0.22] | +0.00 [−0.94, +0.95] / −0.45 [−1.25, +0.36] | −0.95 [−1.54, −0.37] |
| R2 − R0 | +0.02 [−0.83, +0.87] / +0.12 [−0.27, +0.51] | −0.29 [−1.20, +0.61] / −0.25 [−0.54, +0.04] | −0.69 [−1.85, +0.47] / −0.49 [−1.13, +0.15] | −1.13 [−2.33, +0.07] / −1.10 [−1.56, −0.63] | −0.40 [−1.04, +0.24] |
| R3 − R0 | +0.70 [−0.72, +2.12] / +0.40 [−0.45, +1.25] | +0.19 [−0.80, +1.19] / −0.21 [−0.59, +0.18] | +0.27 [−0.61, +1.15] / +0.41 [−0.14, +0.97] | +0.80 [−0.84, +2.43] / +0.47 [+0.31, +0.62] | +0.34 [−0.33, +1.01] |

@figure2 runs/report/figures/reward_B8_W8_gamma099_curve.png | runs/report/figures/reward_B8_W10_gamma099_curve.png
Figure 8. The reward axis on observation B8 at γ = 0.99, 8 × 8 (left, with the two hacking probes) and 10 × 10 (right). The curves are within a point of each other for most of training; R2 is highest at the first evaluation and lowest at the end on 8 × 8.

### 6.7 Reward hacking: the probe

Under the asymmetric R3 a looping policy gathers 0.005 per step for ever, worth 0.5 at γ = 0.99 (Section 3.3), and the starvation limit of 100 steps was expected to be the built-in defence. We searched for exploitation in three ways and did not find any.

**Signatures in the evaluation metrics**. None is found (Table 8; Figure 9). Under R3 the final policies loop in 0.8 % of their episodes (R0: 0.4 %, R1: 0.6 %, R2: 0.4 %); all four end 0.6–1.2 % of their episodes at the starvation limit; episode length and food rate under R3 are the same as R0; and the probe with the starvation limit raised to 1,000 so that a looping policy could show itself, scores the same as R3 and loops no more often (0.2 %). That probe appeared to be a null operation on training, instructively: ε-greedy training episodes almost never go 100 steps without food (0–5 truncated training episodes per run among about 9,000), so the limit never enters the data, and on observation A the 5 trained networks are byte-identical to R3's. A starvation limit cannot shape a policy that never reaches it.

**The direct test: what the network thinks a loop is worth.** Reward hacking is a policy that has correctly priced the loop (≈ 0.5) and chosen it; a trap is a policy that is in a loop while believing it is on a high-value path. The behaviour analysis (Section 6.8) records max_a Q at every step, so the two can be told apart. In the recorded 1,000-step loop of the H = 1,000 probe (period 6; +5.10 of shaping reward collected while starving; `runs/behaviour/dqn_B8_W8_gamma099_R3_H1000_final/2/hack_1_10056.json`), the network's estimate inside the loop is about 11.3: twenty times the loop's true value and the same as at the start of a normal foraging episode. Every looping episode we found, under every reward, is of this kind. Loops are a stage of learning, not a product of the reward: at 51k steps 32 % (R0) and 38 % (R3) of the evaluation episodes end in a loop that lasts until the starvation limit, by 256k steps 1–2 %, at the end 0.4–0.8 %, with the same decline under R0 and R3.

**Raising the stakes.** If 0.01 per closer step does not tempt the agent, 0.1 might: the loop is then worth 5.0, comparable to the discounted return of a whole foraging episode early in training. The probe (`R3, bonus 0.1`; Table 3; Figure 10) produced no hacking either, and instead the highest 8 × 8 score of the matrix: 24.44 ± 1.07, +1.09 [+0.36, +1.82] above R3 and +1.70 [+0.94, +2.46] above R0 in the last-5 window (every seed), and the only learned configuration whose paired interval against BFS on the same boards excludes zero (+1.37 [+0.29, +2.44]; each seed beats BFS on only 47–58 % of the boards and the interval includes zero when either of two seeds is left out, so "first interval away from zero", not "robustly above"). Its final policies loop in 0 of 500 episodes; at 51k steps 54 % of its episodes loop (more than R0's 32 %), but the network's estimate in those loops is 2.3, below the loop's value of 5.0 and below its estimate in non-looping episodes (2.5), so even early the loops are not priced as loops. The extra 1.7 points over R0 are, by point estimate, survival (+17 steps per episode [−7, +40], four seeds of five) rather than food rate (−0.01 per 100 steps [−0.57, +0.54]).

**Why not.** Any closed loop earns the bonus on exactly half its steps; foraging earns it on 77 % of them (measured from the shaping reward collected per episode: 77.7 % under the probe, 77.1 % under R3 at 0.01, 77.3 % on 10 × 10; we did not measure the share under R0), plus the food. As long as the policy can reach food, foraging collects more shaping reward per step than circling, and scaling the bonus scales both sides; it only steepens the gradient towards the food, which is why the probe learns to approach food earlier (score 2.0 against 0.6 at 12.8k steps) and finishes higher. Whether a bonus comparable to the food itself (0.5–1 per step, a risk-free annuity against a finite, risky foraging life) would change that is a different experiment, about decoupling the reward from the objective, and was not run. What we can show is that at neither size did the agents ever price a loop as a loop.

@figure runs/report/figures/causes_B8_W8_gamma099_R0R2R3.png
Figure 9. How the evaluation episodes end during training, observation B8, γ = 0.99, 8 × 8, for R0, R2 and R3 (wall, self, starvation; mean over seeds with t bands; the panels of R1 and of the two probes, in the generated report, have the same shape). Every variant goes through the same three stages: starvation while the agent cannot find food (0–100k steps), a wall-collision peak as it starts to move purposefully, then almost exclusively self-collision. R2's starvation peak at 12.8k steps is much lower (0.3 against 0.9) and its later one higher. No reward produces a late rise in starvation.

@figure runs/report/figures/probe_W8_curve.png
Figure 10. The hacking probe: R3 with the closer-step bonus raised from 0.01 to 0.1, against R3 and R0 (B8, γ = 0.99, 8 × 8). The larger bonus learns faster from the first evaluation and ends highest, above the BFS line.

### 6.8 Behaviour

The behaviour analysis replays the final (and early and middle) checkpoints of selected experiments on the 100 evaluation boards in the scalar environment, recording every head position, action, reward and action value, and summarises visit heat-maps, the share of time on the outer ring of cells ("border share"), the tail of the episode after the last food, and loops (a repeating sequence of head position and heading in that tail; a loop of 20 states or more counts). Table 9 gives the 10 × 10 numbers; Figures 11–13 the heat-maps and the final positions.

**Where the agents go.** Every learned policy, and BFS, spends less time on the border than a uniform distribution would (0.36 of the cells on 10 × 10, 0.44 on 8 × 8): DQN agents 0.22–0.27 on 10 × 10 and 0.31–0.39 on 8 × 8, BFS 0.27 and 0.34, the Hamiltonian cycle exactly uniform. The heat-maps peak in the centre of the board (central cells 0.016 per state, corners 0.0035 for the γ = 0.99 agent on 10 × 10). The agents play through the middle; nothing in the reward design changes this (border share 0.36–0.38 under R0–R3 on 8 × 8 at γ = 0.99).

**How they die, over training.** Figure 9 shows the three stages of the convolutional agent (starvation, a wall-collision peak, then self-collision at 93–98 % for the rest of training); the feature-vector agent passes through the first two within its first two evaluations (12.8k steps: starvation 35 %, self 56 %, wall 9 %), from 25.6k steps dies in its own body in 97 % of episodes, and its wall fraction then drifts up to 16–21 % by the end, the share of its three-moves-fatal positions in which the argmax happens to point at the wall. Figure 13 shows the end: the snake has walled itself in. The tail after the last food is 5–7 steps for the γ = 0.9 agents (they die in the turn after eating), 11–13 on 8 × 8 and 17–18 on 10 × 10 at γ = 0.99 (BFS: 6–9), another view of the survival effect of Section 6.3.

**Loops.** Early checkpoints loop in a third to a half of their episodes (Section 6.7; 45 % on 10 × 10 at 51k steps), always until starvation and with value estimates no higher than in non-looping episodes: these agents cannot yet find food. Final checkpoints loop in 0–1.4 % of episodes. The 3 % starvation of the γ = 0.99 agent on 10 × 10 (Section 6.3) is about half such loops, 6 of 500 episodes that start circling almost from the first move (score 0.2) with value estimates of 11–12, above the 8.6 of normal episodes: traps in which the network expects food it never reaches. A similar number (4–7 of 500) appears under R1, R2 and R3, so they are a property of the long-horizon policy, not of the reward; the remaining 2 % wander without a closed loop. The γ = 0.9 agent never loops and never starves.

Table 9. Behaviour of the final policies on 10 × 10, 100 evaluation episodes per player (DQN rows: mean over 5 seeds ± t interval where it is not negligible). Border share is the mean over episodes of the share of states on the outer ring (uniform: 0.36; the heat-map titles of Figures 11 and 12 pool all visited states instead, which weights long episodes more and differs by a few points); loops are episodes whose post-food tail repeats for 20 states or more; Q in loops is the mean max_a Q in those episodes against the others.

| Player | Score | Steps | Self / wall / starve | Border share | Looping episodes | Q estimate: loops / others | Tail after last food (steps) |
|---|---|---|---|---|---|---|---|
| BFS greedy | 30.61 | 255.8 | 0.84 / 0.16 / 0 | 0.27 | 0 | – | 9.2 |
| Hamiltonian cycle | 97 (all won) | 2,409.7 | – | 0.36 | 0 | – | 0 |
| B8, γ = 0.9 | 20.71 ± 0.52 | 187.8 ± 5.3 | 0.94 / 0.06 / 0 | 0.22 ± 0.01 | 0 | – | 7.0 ± 0.7 |
| B8, γ = 0.99, 51k steps | 6.04 ± 0.62 | 136.1 ± 23.6 | 0.31 / 0.17 / 0.52 | 0.35 ± 0.04 | 0.45 ± 0.15 (227) | 3.12 / 3.12 | 59.8 ± 14.3 |
| B8, γ = 0.99, 256k steps | 20.95 ± 0.34 | 250.6 ± 2.2 | 0.98 / 0.00 / 0.014 | 0.27 ± 0.03 | 0.006 (3) | 9.65 / 6.96 | 15.0 ± 2.2 |
| B8, γ = 0.99, final | 25.75 ± 0.81 | 334.0 ± 13.7 | 0.94 / 0.03 / 0.032 | 0.25 ± 0.01 | 0.012 (6) | 12.08 / 8.61 | 17.7 ± 3.4 |
| B8, γ = 0.99, R1, final | 24.70 ± 0.74 | 321.6 ± 17.9 | 0.95 / 0.02 / 0.030 | 0.26 ± 0.02 | 0.008 (4) | 11.33 / 7.80 | 18.0 ± 2.1 |
| B8, γ = 0.99, R2, final | 25.19 ± 0.66 | 320.7 ± 15.7 | 0.95 / 0.03 / 0.024 | 0.26 ± 0.02 | 0.012 (6) | 11.18 / 8.87 | 16.9 ± 2.3 |
| B8, γ = 0.99, R3, final | 25.75 ± 0.69 | 324.0 ± 10.0 | 0.93 / 0.04 / 0.024 | 0.25 ± 0.02 | 0.014 (7) | 12.34 / 9.53 | 16.7 ± 2.9 |

@figure runs/report/figures/behaviour_heatmaps_W10.png
Figure 11. Where the head is on 10 × 10, over the 100 evaluation episodes, one colour scale; the panel titles give the share of all visited states on the outer ring (pooled, so long episodes weigh more). The planner and all learned policies concentrate in the middle; the Hamiltonian cycle is uniform by construction; the early γ = 0.99 checkpoint, which loops in half its episodes, is the only learned one whose pooled visits exceed the uniform border share.

@figure runs/report/figures/behaviour_heatmaps_W8.png
Figure 12. The same on 8 × 8 for BFS, the Hamiltonian cycle, the feature-vector agent, the convolutional agent at both discounts and the hacking probe. The probe's distribution is that of the other agents: nothing in it suggests circling.

@figure runs/behaviour/dqn_B8_W10_gamma099_final/0/last_steps.png
Figure 13. The final board of eight evaluation episodes of one γ = 0.99 agent (seed 0, 10 × 10): the body in blue with the head darker, the last ten head positions in red, the food in green. Every death but one is a self-collision inside a pocket the snake has closed around itself, with the food out of reach; the one wall collision (board 10000) is a corridor along the edge with no exit.

## 7 Discussion

### 7.1 What reinforcement learning bought, against the planners

The reference players bracket the learned agents precisely (Table 3). BFS eats as fast as any learned policy and the learned policies at γ = 0.9 die sooner, so they finish 7–10 points below it; the convolutional agent at γ = 0.99 survives longer than BFS but eats much more slowly (Section 6.3), so it finishes 4.9 points below it on 10 × 10 and cannot be told apart from it on 8 × 8. No learned policy approaches the Hamiltonian cycle; none ever fills the board (maximum score 46 against the cycle's 97), and all of them, like BFS, die by trapping themselves (Figure 13). Reinforcement learning bought the planner's eating speed without planning (on A), or the planner's score by a different strategy (survival instead of speed, on B8 with a long horizon). It did not buy, within a million steps and these architectures, the thing the Hamiltonian player has and BFS lacks: a policy that never encloses itself. This is the honest summary of "DQN plays Snake", and it is only visible because the references are on the same boards.

### 7.2 What the matrix says about the three questions, and what it says instead

Of the three axes in the title, only the representation axis gave a clear effect at the planned discount, and only on the small board (Table 2). The effects that dominate the matrix were not axes of the plan: the discount factor, the replay ratio and the staleness of the target network when its period is counted in gradient steps. They were found because the protocol was fixed enough that a hyper-parameter row could be compared with everything else. Two lessons follow: the hyper-parameters that are "held fixed" deserve a row each, because a conclusion about an axis is conditional on them (the representation effect grows from +1.7 to +5–7 points when γ changes); and the interactions matter more than the main effects: γ with the representation, replay with the target period, the replay ratio with the board size (the low-ratio configuration is worse on 8 × 8 and best on 10 × 10, because it has not converged on the small board).

**The deadly triad, here.** Nothing diverged in 320 runs, including those without a target network or without replay; the pathology we found is the opposite one, values several times too low when the target is stale, a learning failure rather than an instability. The target network has no measurable benefit at the period we used, which we read as a statement about the scale of the problem (a 3-action, 100-cell game with a small, smooth value function and a learning rate of 2.5 × 10⁻⁴), not as a general finding. Replay's benefit is real on the convolutional network and nearly absent on the feature vector, which fits its purpose: the CNN needs thousands of updates on decorrelated data to fit a function of the whole board; the 256-code MLP has learned all it will learn after a few hundred updates on anything. Whether either component is *needed* is problem-dependent, and the two controls we added (target period in environment steps; replay ratio) are what made the answer legible.

**Overestimation and shaping.** The bias predicted by [12] and measured by [2] is present in every DQN we trained, grows with the discount factor and with the replay ratio, and depends on where it is measured; Double DQN reduces it where it is large, with no consistent counterpart in the score, because the greedy policy depends only on the ordering of the three action values in each state. Potential-based shaping did what the theorem promises and nothing more: the optimum is unchanged, the learning problem is different, and here it is easier to start and harder to finish. Naive shaping was harmless at both sizes because a forager earns the bonus more often than a loop, a comparison that can be made before running anything; the test that settled it, the agent's own value estimate inside the loop, is the one we would recommend for reward hacking in any value-based agent.

### 7.3 Limitations

- **Budgets are fixed and several curves are still rising**: the convolutional agent at γ = 0.99 on both boards, the low-replay-ratio configuration and the no-replay control on 10 × 10; the convolutional agent at γ = 0.9 on 10 × 10 is near a plateau but not converged. Every "final" number is a 512k-step (8 × 8) or one-million-step (10 × 10) number, and the ordering of configurations that learn at different rates could change with more training.
- **Five seeds resolve differences of about 0.7 points** in the last-5 window and about 1 point in the final window (the median half-widths of the paired intervals); most reward-axis effects and several component effects are at or below that resolution, which is why we report windows, sign counts and leave-one-out checks rather than a single p-value, and why "no measurable difference" means |effect| < about 1, not zero.
- **Confounds we accepted and name.** "No replay" changes three things, separated by controls but not independently (the mixing control also doubles the batch and the reuse). The low-replay-ratio configuration changes G and the target period together. Dueling doubles the head's parameters. PER's result is specific to the published weight normalisation. Missing cells: 10 × 10 has no A × γ = 0.99, no γ = 0.99 versions of B7 or `G1_t31`, no "no replay, no target", no `G1` with the long target period and no feature-vector "no replay" with the long period, so the plan's four-curve ablation is three curves there; the reward axis on 10 × 10 was run at γ = 0.99 only.
- **Scope.** Two board sizes (the planned 16 × 16 extension and the action-space ablation were not run); one network per representation; the hyper-parameter sensitivity of the plan became the γ, G, target-period and capacity rows rather than ε and buffer-size sweeps for the DQN (ε and α were swept for the tabular agent only). The policy-search result of Section 6.1 is preliminary.
- **Bit-exact reproducibility holds on one machine** (Section 5.3); across machines, runs are independent draws from the same distribution, and the tables and figures are machine-independent. The notebook has been executed on the training machine, not yet on Colab.

### 7.4 Deviations from the Part B plan

| Plan | What was done | Why |
|---|---|---|
| γ = 0.99 as the starting value | γ = 0.9 as the main line; γ = 0.99 as a row of the matrix (Section 6.3) | On the tabular sweep γ = 0.99 with α ≥ 0.1 circles until starvation and scores two points lower within the budget (Appendix C); the DQN reference had to share the table's discount. The row turned out to be the largest effect. |
| R3 as ± 0.01 | Asymmetric R3 (+0.01 closer, 0 farther) as the default; the symmetric version run as well | The symmetric form is potential-based with γ = 1 and cannot be exploited; it would not have served its purpose. |
| Heading plane restores the Markov property | The age plane does; heading planes are redundant; B7 / B8 ablation | Section 3.2; the predicted cost of the non-Markov B7 did not appear (Section 6.2). |
| Episode ends: wall, self, starvation | A fourth end, "win" (board full) | A full board has no cell for food; only the Hamiltonian player reaches it. |
| "Throughput of the Python step loop is the real bottleneck" | It is not: gradient updates are 94–99 % of a run, the environment under 1 % | Measured in Section 5.4; the vectorised environment removed the bottleneck the plan expected. |
| Training on the GPU; Python 3.11, PyTorch 2.7+ | MLP on the CPU, CNN on the GPU; Python 3.13, PyTorch 2.11 | One MLP update costs the same on either device and CPU runs are bit-reproducible; the newer versions were what the workstation had. |
| "−replay" as one ablation curve | Defined as one gradient step per vectorised step on the fresh transitions, plus two controls | The plan did not define it; one curve conflates three factors (Section 6.4). |
| Resume interrupted runs from checkpoints | Not implemented; interrupted batches are rerun | Checkpoints hold weights only; resuming also needs the buffer, the RNG streams and the environment state, and the longest run is under an hour. |
| Equivalence test "runs in CI" | Runs locally on every milestone (1,916 tests); no hosted CI workflow | Not set up; pushing a workflow needs a token scope the account did not have. |
| Cut order: action space → sensitivity → PER | Revised early to PER → Dueling → sensitivity → action space; in the end PER and Dueling were run, the ε / buffer-size sweeps, the action-space row and 16 × 16 were not started | PER was the most expensive to implement and the least instructive; the action-space row is a switch of the environment. The γ, G, target-period and capacity rows took the sensitivity slot. Seeds were never cut. |
| Score distributions and last-steps replays in the behaviour analysis | Delivered (Figures 2 and 13; recordings replay in the GUI) | – |
| GUI: checkpoint and observation switching in the side panel; heat-map overlay | Selected on the command line; heat-maps in the analysis module | Lower demonstration value than the live value panel, which was built instead with an observation-feature display the mock-up did not have. |
| Order of milestones: CNN before GUI | GUI before CNN | The Part D deadline required real screenshots; the GUI depends only on the scalar environment. |

### 7.5 What did not work

Three things failed in the ordinary sense: the starvation-limit probe measured nothing, because the training distribution never reaches the limit; the first "no replay" result was uninterpretable until the controls were added, because one configuration key silently changed the meaning of another; and PER as published behaves badly where most TD errors go to zero, which we understood only by reading the optimiser's state out of the checkpoints. In each case the fix was a measurement rather than a tweak, and the lesson of the project is that the deep-RL results that survive are the ones with a control for the obvious alternative explanation, which is usually cheap once the protocol is fixed.

## 8 Conclusion

On a self-built Snake environment with a fixed protocol, five seeds per configuration and paired statistics, we compared two state representations, four reward designs, the two DQN stabilisers and three refinements against non-learning references. Of the planned axes, only the representation changed the score at the planned discount, and the feature vector's level is that of a policy without look-ahead; reward design moved the score by about a point at most, naive distance shaping was never exploited, and the target network had no measurable benefit. The largest effects were hyper-parameters the plan had held fixed: the discount factor, which adds five points of survival to the convolutional agent and brings it level with the greedy planner on the small board, and the number of gradient steps per transition, which together with the target period accounts for the loss from removing replay and, lowered, gives the best 10 × 10 agent at γ = 0.9. Double DQN reduces overestimation where it is large without changing the score; Dueling and PER change nothing. No agent learned what the Hamiltonian cycle knows, never to enclose itself, and that remains the gap between value-based learning and planning on this game. Every number in this report is regenerated from the run logs by one command.

## 9 References

1. Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529–533.
2. van Hasselt, H., Guez, A., and Silver, D. (2016). Deep reinforcement learning with double Q-learning. *Proceedings of the AAAI Conference on Artificial Intelligence*, 30(1), 2094–2100.
3. Wang, Z., Schaul, T., Hessel, M., van Hasselt, H., Lanctot, M., and de Freitas, N. (2016). Dueling network architectures for deep reinforcement learning. *Proceedings of the 33rd International Conference on Machine Learning*, 1995–2003.
4. Schaul, T., Quan, J., Antonoglou, I., and Silver, D. (2016). Prioritized experience replay. *International Conference on Learning Representations*.
5. Ng, A. Y., Harada, D., and Russell, S. (1999). Policy invariance under reward transformations: theory and application to reward shaping. *Proceedings of the 16th International Conference on Machine Learning*, 278–287.
6. Sutton, R. S., and Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
7. Watkins, C. J. C. H., and Dayan, P. (1992). Q-learning. *Machine Learning*, 8, 279–292.
8. Lin, L.-J. (1992). Self-improving reactive agents based on reinforcement learning, planning and teaching. *Machine Learning*, 8, 293–321.
9. Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., and Meger, D. (2018). Deep reinforcement learning that matters. *Proceedings of the AAAI Conference on Artificial Intelligence*, 32(1).
10. Towers, M., Kwiatkowski, A., Terry, J., et al. (2024). Gymnasium: a standard interface for reinforcement learning environments. arXiv:2407.17032.
11. Loeber, P. snake-ai-pytorch: teaching an AI to play Snake with deep Q-learning. GitHub repository, https://github.com/patrickloeber/snake-ai-pytorch (accessed 22 October 2026; the 11-feature state representation used as observation A).
12. Thrun, S., and Schwartz, A. (1993). Issues in using function approximation for reinforcement learning. *Proceedings of the Fourth Connectionist Models Summer School*, 255–263.
13. Fedus, W., Ramachandran, P., Agarwal, R., Bengio, Y., Larochelle, H., Rowland, M., and Dabney, W. (2020). Revisiting fundamentals of experience replay. *Proceedings of the 37th International Conference on Machine Learning*, 3061–3071.

## Appendix A: Reproducing the study

**Software and hardware.** One workstation: Windows 11, Python 3.13, PyTorch 2.11 with CUDA 12.8, one RTX 5090, 31 GB RAM. Installation (the CUDA build of PyTorch must be installed first because the PyPI Windows wheel is CPU-only) and the full command reference are in the repository's `README.md`. Batches of five seeds ran as five processes; five CNN processes saturate the GPU (83 % utilisation, 8–17 ms per gradient step against 2.6 ms alone), so batches were queued one at a time. A single MLP gradient step costs 1.7 ms on the CPU and 1.6 ms on an idle GPU; a CNN step 2.6 ms on the GPU and 40–69 ms on the CPU.

**Implementation checks.** The scalar and vectorised environments are compared bit for bit on 19 configurations (58,580 transitions) in the test suite and were fuzzed on 1,149 random configurations (15.8 million transitions) without a discrepancy; the vectorised environment steps 0.6–1.1 million boards per second on observation A with 128–256 boards (0.7 million on B8 with 128). The tabular agent with one environment is bit-identical to a textbook Q-learning loop; the DQN loop reproduces an independently written plain-PyTorch reference; a spy test confirms that truncated transitions are stored with `terminated = False` and their true successor; the CNN forward pass agrees with a hand-written convolution to 10⁻⁵; a run re-trained from its own configuration file reproduces its weights, checkpoints and logs bit for bit. The package is about 9,200 lines of Python, the test suite about 21,000 lines (1,916 tests).

**Commands.**

- `python -m snake_rl.analysis.matrix --status` lists the 64 configurations with the number of finished seeds and checks every run's stored configuration against the matrix; `--commands` prints the launch lines for any missing row (`scripts/run_matrix.py` runs the five seeds of one configuration as parallel processes).
- `python -m snake_rl.analysis.report --out-dir runs/report` builds every table (`tables/*.md`) and figure (`figures/*.png`) of Sections 6.1–6.7 from the run directories, with `index.md` and a `manifest.json` that lists the experiments, seeds and git commits behind each item.
- `python -m snake_rl.analysis.behaviour --exp runs/<experiment> --checkpoint final|early|mid --out-dir runs/behaviour/<name>` produces the behaviour analyses of Section 6.8 (per-episode records, heat-maps, last-steps figures, loop statistics and recordings of the longest loops, which `python -m snake_rl.gui.app --replay <file>` replays); `python scripts/report_extra_figures.py` assembles Figures 2, 9, 11 and 12 from the analyses and the run directories.
- `python -m snake_rl.analysis.report_tables --report docs/PartE-final-report.md` computes Tables 3–9 and C1 (paired differences on the same seeds, the comparison with BFS on the same boards, the overestimation gap, the behaviour statistics, the tabular sweep) from the run directories and prints every cell that differs from this text.
- `python scripts/export_logs.py --out runs_logs.zip --include-behaviour` packs the logs without weights (2,639 files, 15 MB compressed, 31 MB uncompressed, with the behaviour analyses and the screenshot of Figure 1); `notebooks/results.ipynb` carries the code of the study (the package modules, one per cell; `scripts/build_notebook.py` writes it, a test keeps the two identical), trains the 8 × 8 tabular agent and two short DQN runs, then rebuilds every table and figure of this report from that archive and checks the tables against this text (executed on the training machine, about 3 minutes; not yet on Colab).
- `python -m snake_rl.evaluate --W 10 --episodes 100` re-evaluates the reference players; `pytest -q` runs the tests.

Every run directory `runs/<experiment>/<seed>/` holds `config.yaml` (full configuration, seed, git commit, software versions, device), `metrics.csv` (the evaluation curve with the value and return columns of Section 4.5, death causes, loss, mean Q, gradient norm and timings), the final weights and the per-episode `final_eval.json`. The runs were trained at twelve commits between `c801fff` (the tabular runs) and `bdea984` (the last five rows), recorded per run in `config.yaml` (two 8 × 8 control experiments span two commits that differ only in documentation); features were added between those commits (the CNN, the no-replay mode, the value measurement, Dueling, PER), and the update path of the configurations already run was verified unchanged by re-training a Part C seed with the final code, bit for bit, and by comparing the first 68 evaluations of an interrupted 10 × 10 batch with its re-run under the later code.

## Appendix B: Every configuration

The two tables below are `runs/report/tables/summary_W8.md` and `summary_W10.md` with the columns of this report: final score (mean over 5 seeds ± 95 % t interval; reference players ± normal interval over 100 episodes), last-5 score, mean steps per episode and the fractions of final-evaluation episodes ending at a wall, in the body or at the starvation limit.

Table B1. 8 × 8, every configuration (512,000 environment steps per learner run; tabular Q-learning 500,000).

@table runs/report/tables/summary_W8.md

Table B2. 10 × 10, every configuration (1,024,000 environment steps per learner run; tabular Q-learning 1,000,000).

@table runs/report/tables/summary_W10.md

## Appendix C: The tabular sweep behind the discount factor

Table C1. Tabular Q-learning on 8 × 8, 500,000 steps, final greedy score (mean over 5 sweep seeds ± 95 % t interval) over step size α and discount γ, with the ε schedule of Table 1; in parentheses, the number of greedy evaluation snapshots (of 5 seeds × 20 evaluations) in which at least one episode ended at the starvation limit, i.e. the policy was circling. α = 0.1–0.3 with γ = 0.8–0.95 form a flat region; γ = 0.99 with α ≥ 0.1 is lower and circles. A separate ε sweep (nine schedules at α = 0.1, γ = 0.9) spans 15.2–16.1 with overlapping intervals. From `runs/tables/sweep_tabular_W8_alpha_gamma.md`.

| α \ γ | 0.8 | 0.9 | 0.95 | 0.99 |
|---|---|---|---|---|
| 0.02 | 14.77 ± 0.33 (0) | 15.29 ± 0.66 (0) | 14.58 ± 0.68 (0) | 14.61 ± 1.11 (0) |
| 0.1 | 16.02 ± 0.93 (1) | 15.51 ± 0.59 (1) | 15.45 ± 1.38 (2) | 14.29 ± 2.03 (21) |
| 0.3 | 16.00 ± 1.27 (0) | 15.48 ± 1.25 (6) | 15.32 ± 1.04 (22) | 12.79 ± 3.43 (82) |

## Appendix D: Contribution

#### A. Individual Contribution

| Team Member Name | Contribution |
|---|---|
| Jiahao Liu (14191136) | Repository lead: environment (scalar and vectorised), observations and rewards, reference players, tabular Q-learning, DQN and its variants, evaluation protocol, experiment matrix and batch tooling, analysis and report generation, behaviour analysis, GUI, tests; ran all experiments; drafted Parts C, D and E. |
| Yue Liu (25211451) | This report was reviewed and revised. I also collaborated with the team members to complete the code implementation and subsequent corrections. |
| Siqin Jin (26145324) | Reviewed the final code and cross-checked its consistency with the methods and experiment settings described in the report; searched for and reviewed relevant literature on DQN and its stabilisation techniques; checked references and figure captions; and contributed to proofreading and revising the final report. |

#### B. Individual Contribution Split

| Team Member Name | Percentage |
|---|---|
| Jiahao Liu (14191136) | 40% |
| Yue Liu (25211451) | 30% |
| Siqin Jin (26145324) | 30% |

All the team members have discussed and agreed on the above individual contribution to the project.

Jiahao Liu : [Signature]

Yue Liu : [Signature]

Siqin Jin : [Signature]

<!-- END -->
