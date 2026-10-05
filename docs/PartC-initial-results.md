# Assignment 3, Part C — Initial Experimental Results (text of the submission)

This file is the source text of the Part C document
(`43008-Project-Part-C-TheDeadlyTriad.docx`). Table 1 is taken from tables
generated from the run directories by `python -m snake_rl.analysis.aggregate`
and Figure 1 is drawn by `python -m snake_rl.analysis.plots`; the other numbers
(differences between learners, the classification of deaths, timing) were
computed from the same run directories by separate scripts, as recorded in
`docs/PROGRESS.md` (sections "M4 验收" and "M5 验收").

**Project title and number:** Value-Based Deep Reinforcement Learning on Snake: A Controlled Study of State Representation, Reward Design, and DQN Stabilisation — Project number: 48

**Team name:** The Deadly Triad

**Team members:** Yue Liu (25211451, Yue.Liu-14@student.uts.edu.au); Jiahao Liu (14191136, Jiahao.Liu-4@student.uts.edu.au); Siqin Jin (26145324, siqin.jin@student.uts.edu.au)

## Algorithms/Methods chosen

We study value-based reinforcement learning on Snake as a controlled experiment, on an environment we wrote ourselves (Gymnasium API). Of the Part B plan, Phases 0 and 1 and the first part of Phase 2 (DQN with an MLP) are implemented and evaluated here; the CNN and the full ±replay / ±target-network ablation of Phase 2 are not implemented yet. Code, configurations and 1,237 automated tests: https://github.com/Cureeeeeeee/Value-Based-Deep-Reinforcement-Learning-on-Snake

- **Reference players (no learning).** Random; BFS-greedy, which follows a breadth-first shortest path to the food and ignores self-trapping (the "greedy A*" of Part B; on an unweighted grid both find a shortest path); and a Hamiltonian-cycle player, which fills the board in every game, slowly.
- **Tabular Q-learning (Phase 1).** One table entry per (state, action) on the 11 binary features defined in the next box ("observation A"); ε-greedy exploration; update Q(s,a) ← Q(s,a) + α [r + γ max_a′ Q(s′,a′) − Q(s,a)]. Fifty environments are stepped in parallel and the TD errors of transitions that hit the same table cell are averaged; with one environment this is exactly textbook Q-learning (checked by a test).
- **DQN (Phase 2, MLP on observation A).** The same 11 features feed a multilayer perceptron (11–256–256–3), trained with uniform experience replay, a periodically copied target network, the Huber loss and Adam.

## Experimental settings

**Problem formulation.** W × W grid (10 × 10 is the main board, 8 × 8 the quick one); the snake starts with length 3. Three relative actions: straight, turn left, turn right. Reward R0: +1 for food, −1 for dying (wall or own body), 0 otherwise. An episode terminates on a collision or when the board is full, and is truncated after 100 steps without food (the starvation limit; no penalty). Truncation is a time-out, not a death: the learning target still uses the value of the next state, for both learners. Food reappears uniformly at random on an empty cell. Observation A is 11 binary features: danger straight / left / right (3), heading (4), food above / below / left / right of the head (4); only 256 of the 2¹¹ codes can occur.

**Evaluation protocol.** Every player is evaluated greedily (ε = 0) on the same 100 episodes (seeds 10000–10099, which fix where the food appears and are never used in training). Each learner is trained five times with different seeds; a reported score is the mean over the five trained agents. The intervals are defined under Table 1. Each run stores its configuration, seed and git commit, and Table 1 and Figure 1 are generated from the stored logs.

| Setting | Tabular Q-learning | DQN |
|---|---|---|
| Discount γ | 0.9 | 0.9 |
| Step size / optimiser | α = 0.1 | Adam, learning rate 2.5 × 10⁻⁴ |
| Exploration | ε from 1.0 to 0.05, linear over the first 20 % of the steps | same |
| Parallel environments | 50 | 128 |
| Environment steps (10 × 10 / 8 × 8) | 1,000,000 / 500,000 | 1,024,000 / 512,000 |
| Action-value function | table, 256 reachable states × 3 actions | MLP 11–256–256–3 |
| Update | the rule above, once per step of the 50 environments | replay buffer 100,000; batch 256; one gradient step per 4 environment steps, starting after the first 10,000 steps; target network copied every 1,000 gradient steps; Huber loss; gradient norm clipped at 10 |
| Evaluation every (10 × 10 / 8 × 8) | 50,000 / 25,000 steps | 12,800 steps |
| Training time per run (CPU) | about 9 s / 6 s | about 10 min / 5 min |

**Main differences from the Part B plan.** (i) γ = 0.9 instead of 0.99: in a sweep of 105 tabular runs on 8 × 8, γ = 0.99 with α ≥ 0.1 lowered the mean score by about 2 and produced some greedy policies that circle until the starvation limit; for the DQN on 8 × 8, γ = 0.99 made no measurable difference (results below). (ii) The DQN is trained on the CPU: one gradient step of this small network costs about 2 ms there, and the GPU is not faster. (iii) Environment stepping is not the bottleneck that Part B expected: gradient updates take 97 % of the DQN's wall-clock time, environment stepping under 1 %. (iv) The CNN on the full-grid observation B, planned for this checkpoint, is not implemented yet and is the next item.

## Initial results

| Player | Score, final policy | Score, mean of last 5 evaluations | Steps per episode | Episode ends: wall / self / starved (%) |
|---|---|---|---|---|
| **10 × 10 board** | | | | |
| Random | 0.18 ± 0.08 | – | 23 | 98 / 1 / 1 |
| BFS-greedy | 30.61 ± 1.60 | – | 256 | 16 / 84 / 0 |
| Hamiltonian cycle | 97 (fills the board) | – | 2,410 | 0 / 0 / 0 |
| Tabular Q-learning (γ = 0.9) | 18.51 ± 1.10 | 18.14 ± 0.29 | 151 | 10 / 90 / 0 |
| Tabular Q-learning, γ = 0 (control) | 19.69 † | 19.69 † | 239 | 4 / 96 / 0 |
| DQN (γ = 0.9) | 20.29 ± 1.55 | 20.44 ± 1.41 | 168 | 16 / 84 / 0 |
| **8 × 8 board** | | | | |
| Random | 0.23 ± 0.10 | – | 13 | 98 / 2 / 0 |
| BFS-greedy | 23.07 ± 1.13 | – | 151 | 35 / 65 / 0 |
| Hamiltonian cycle | 61 (fills the board) | – | 986 | 0 / 0 / 0 |
| Tabular Q-learning (γ = 0.9) | 15.67 ± 0.87 | 15.67 ± 0.40 | 102 | 19 / 81 / 0 |
| Tabular Q-learning, γ = 0 (control) | 16.07 † | 16.07 † | 151 | 7 / 93 / 0 |
| DQN (γ = 0.9) | 16.11 ± 0.88 | 16.07 ± 0.64 | 105 | 21 / 79 / 0 |

Table 1. Greedy evaluation on the same 100 episodes; score = food eaten per episode. Reference players: mean ± 95 % normal interval over the 100 episodes. Learners: five trained agents each, mean ± 95 % Student-t interval over the five, for the policy after the last update and for the mean of each agent's last five evaluations (less sensitive to where training stops). The two kinds of interval cannot be compared by overlap. † With γ = 0 the five agents end with (almost) the same deterministic policy, so the spread over seeds is at most ± 0.10; what remains is the choice of the 100 episodes, about ± 1.

Figure 1. Score of the greedy policy during training on 10 × 10: mean over five training seeds, band = 95 % Student-t interval. Dashed lines: BFS-greedy and random; the Hamiltonian player (97) is off the scale.

- **Both learners work, and both stay far below BFS-greedy.** Tabular Q-learning and DQN reach 18.5 and 20.3 on 10 × 10 (random: 0.2) after seconds to minutes of training, which is 60 % and 66 % of the BFS-greedy score (68 % and 70 % on 8 × 8).
- **DQN against the table (the sanity check of Part B).** On 8 × 8 the two cannot be told apart with five seeds (DQN minus table: +0.44, 95 % interval −0.59 to +1.47). On 10 × 10 the final DQN policies score higher (+1.78, interval +0.17 to +3.39), but this is preliminary: it is five agents against five, and in Figure 1 the DQN mean is at the table's level (about 18–19) between roughly 290,000 and 420,000 steps before rising again; the cause is not yet investigated.
- **The control with γ = 0 shows what these scores mean.** A table trained with γ = 0 has no look-ahead: its target is the immediate reward. On both boards it scores no lower than the γ = 0.9 table and within the DQN's interval. On the 11 features this score level therefore needs no look-ahead (do not take a blocked move; turn to the food once it is in line) and is no evidence of multi-step values; the DQN's lead over the γ = 0.9 table on 10 × 10 must not be read as better look-ahead.
- **How the agents die.** In all 500 final-evaluation episodes of the γ = 0.9 table and of the DQN on each board, the snake dies where all three moves are fatal: these agents never step into an obstacle they can see and die only after enclosing themselves, which the 11 features cannot show them. (The wall / self split only records which fatal move is taken.) The γ = 0 control does so in 995 of 1,000 (two boards). This level is not the ceiling of the 11 features: a preliminary search over policies on the same 256 codes (not yet a logged run) found policies that score well above both learners.
- **Speed.** In Figure 1 the DQN is near its final level after about 25,000–40,000 steps and the table after about 200,000, when its exploration rate reaches its minimum. That point moves with the exploration schedule and the DQN reuses each transition about 64 times, so we claim no sample-efficiency factor.
- **What made no measurable difference on the 11 features (8 × 8, five seeds, the same as the main run: 16.11 ± 0.88).** No separate target network, i.e. a copy after every gradient step (16.61 ± 1.04); γ = 0.99 (15.91 ± 0.60); a single hidden layer of 16 units (15.60 ± 1.23). Gradient clipping never became active: the logged mean gradient norm stays below 0.5, against a threshold of 10. A linear Q-function fails (3.04 ± 5.30; 76 % of its final-evaluation episodes end at the starvation limit).

**Consequence for the next phase.** On the 11 features this score level needs no look-ahead, and on 8 × 8 removing the target network made no measurable difference (replay has not been removed yet). We therefore expect the ±replay / ±target-network ablation to show little on observation A; it will be run on both observations, with the full-grid observation B, where the body is visible, as the main line. Next: the CNN on observation B, then the ablations, Double DQN, the reward variants and the GUI (Part D).
