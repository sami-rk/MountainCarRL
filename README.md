# 🚗 MountainCar Q-Learning: Reinforcement Learning

**Course:** Artificial Intelligence — Spring 1405 (Computer Assignment 3)  
**Instructor:** Dr. Fada'i  
**Authors:** Erfan Falahati, Kasra Kashani, Aria Azem

---

## What This Project Does

A classic Reinforcement Learning problem: an **underpowered car is stuck in a valley between two hills** and must learn to reach a flag at the top of the right hill. The engine is too weak to drive straight up, the car must first swing back toward the left hill to build momentum, then use that kinetic energy to crest the right hill.

This project implements **Tabular Q-Learning** from scratch to solve the `MountainCar-v0` environment from OpenAI Gym, and explores discretization, exploration strategies, and reward shaping.

---

## The Physics Intuition

The car starts at the bottom of a valley. The motor alone cannot overcome gravity. The key insight the agent must discover on its own:

1. Drive **left** first (away from the goal) to climb the left hill
2. Convert that height (potential energy) into speed (kinetic energy) on the way back down
3. Use that speed to crest the right hill and reach the flag

This makes it a deceptively hard problem, the car must temporarily move _away_ from the goal to eventually reach it.

---

## Environment Details

|Property|Value|
|---|---|
|State space|Position ∈ [-1.2, 0.6], Velocity ∈ [-0.07, 0.07]|
|Actions|0 = Push Left, 1 = No-op, 2 = Push Right|
|Reward|-1 per step (sparse — no positive signal until the flag)|
|Episode limit|200 steps|
|**Solved when**|**100-episode average reward > -110**|

---

## Installation

```bash
pip install "gym==0.26.2" "numpy<2" pygame matplotlib
```

---

## How to Run

Open `RL_model.ipynb` in Jupyter and run the cells top to bottom. The notebook is structured in self-contained parts, you can run just the sections you are interested in.

---

## Watch the Car Drive 🎮

To see the agent in action in a live window **before training** (random behavior):

```python
env = gym.make('MountainCar-v0')
agent = QLearningAgent(env, buckets=(40, 40), epsilon_strategy='exponential', episodes=500000, seed=400)
watch_agent(agent, episodes=5)  # Opens a render window
```

To watch the **trained agent** solve the environment:

```python
# After training:
scores, avg_scores, epsilons, _ = agent.train()
watch_agent(agent, episodes=5)  # Now it reaches the flag every time
```

`watch_agent()` opens a real-time graphical window using pygame and prints the reward for each episode. A well-trained agent consistently scores between -85 and -140.

> **Note:** The graphical window requires a display. On headless servers, remove the `watch_agent` calls or set `render_mode=None`.

---

## Core Algorithm: Q-Learning

The agent maintains a **Q-table** — a lookup table mapping every (state, action) pair to an expected future reward. After each step, the table is updated using the Bellman equation:


$$Q(s, a) \leftarrow Q(s, a) + \alpha \cdot [r + \gamma \cdot max_a' Q(s', a') − Q(s, a)]$$


| Parameter          | Value       | Role                                                       |
| ------------------ | ----------- | ---------------------------------------------------------- |
| `alpha` ($\alpha$) | 0.05        | Learning rate — how strongly new info overwrites old       |
| `gamma` ($\gamma$) | 0.99        | Discount factor — how much the agent values future rewards |
| `buckets`          | (40, 40)    | Grid size for discretizing the continuous state space      |
| `epsilon_strategy` | exponential | How exploration decays over time                           |
| `episodes`         | 500,000     | Training duration                                          |

### State Discretization

Since Q-learning requires a discrete lookup table, the continuous position and velocity values are mapped to a finite grid:

```
bucket_index = clip(floor((obs - lower) / (upper - lower) * num_buckets), 0, num_buckets - 1)
```

The default 40×40 grid creates 1,600 distinct states × 3 actions = **4,800 Q-values** to learn.

---

## Project Structure

```
.
├── RL_model.ipynb          # Full implementation — all parts
└── README.md
```

### Notebook Sections

|Part|Description|
|---|---|
|**Part 1**|`QLearningAgent` class — discretization, ε-greedy, Q-update, train/evaluate|
|**Part 2**|Training the main agent (500k episodes) + reward and ε plots|
|**Part 3**|Evaluation — 100 episodes with ε=0 (pure exploitation)|
|**Part 4**|Epsilon strategy experiments (constant / linear / logarithmic / exponential)|
|**Part 5**|Bucket size experiments (5×5 vs 40×40 vs 60×60 vs 100×100)|
|**Part 6**|3D value function surface plot|
|**Part 7**|Reward shaping (bonus)|
|**Questions**|10 analytical questions answered in depth|

---

## Exploration Strategies

Four ε-decay schedules were tested over 100,000 episodes:

| Strategy        | Formula                                           | Outcome                                                                                          |
| --------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| **Constant**    | $\epsilon$ = 0.10 always                          | Finds the flag early but plateaus at ~-138; random actions destroy momentum                      |
| **Linear**      | $\epsilon$ goes from 1.0 → 0.01 over all episodes | **Only strategy that solved the environment** — slow start, then rapid improvement once ε < 0.30 |
| **Logarithmic** | $\epsilon = \frac{1}{log(episode + 2)}$           | Drops quickly then flattens ~0.20 — permanently stuck around -150                                |
| **Exponential** | $\epsilon = e^{(-5 × \frac{episode}{total})}$     | Clean, stable learning curve; converges around -120 but misses the -110 threshold                |

**Takeaway:** Linear decay wins because it allows enough early exploration and then cleanly shuts off the noise at the end.

---

## Discretization Grid Size (Curse of Dimensionality)

| Grid    | States | Memory   | Behavior                                                                               |
| ------- | ------ | -------- | -------------------------------------------------------------------------------------- |
| 5×5     | 75     | Tiny     | Cannot solve. **State aliasing** collapses distinct physical states into the same cell |
| 40×40   | 4,800  | Small    | Solves the environment reliably ✓                                                      |
| 60×60   | 10,800 | Moderate | Good quality, slower to converge                                                       |
| 100×100 | 30,000 | Large    | Extremely slow — each of 10,000 cells is rarely revisited, so Q-values barely update   |

**State Aliasing** (5×5 problem): When the grid is too coarse, a car moving fast-left and another moving slow-right end up in the same bucket. Their contradictory training signals cancel each other, making the policy incoherent.

---

## 3D Value Function

After training, the learned value function `V(s) = max_a Q(s, a)` is plotted as a 3D surface over (position, velocity). Only states visited during training are shown, unvisited states have unreliable Q-values and are masked out.

**Key insight from the plot:** States with _negative velocity at the valley bottom_ have **high value** — the agent correctly learned that moving toward the left hill is the necessary first step to build enough momentum to reach the goal. This directly mirrors the physics: potential energy on the left hill converts to kinetic energy for the right-hill climb.

---

## Reward Shaping (Bonus)

The default reward (-1 per step) is **sparse** — the car gets no signal until it reaches the flag, making early learning very slow. Five reward functions were compared over 2,000 episodes:

|Mode|Shaped Reward|Description|
|---|---|---|
|`original`|-1 per step|Sparse baseline|
|`shaping1`|+ γ·position' − position|Rewards moving rightward|
|`shaping2`|+ γ·\|velocity'\| − \|velocity\||Rewards gaining speed|
|`shaping3`|+ γ·(position' + \|velocity'\|) − (position + \|velocity\|)|Combined position + speed|
|`hacked`|\|velocity\| × 10|**Reward hacking** — maximizes speed regardless of direction|

All shaping functions follow the potential-based formula `r + γ·φ(s') − φ(s)`, which preserves the optimal policy while guiding the agent faster. `shaping3` (combined) learns the fastest.

The **hacked** mode achieves a high numerical reward by oscillating at maximum speed in the valley — never reaching the flag. This is a textbook example of **reward hacking**: the agent perfectly optimizes the metric while completely failing the actual task.

---

## Evaluation Results (500k episode agent, ε=0)

|Metric|Score|
|---|---|
|Mean reward|~-110 to -120|
|Best episode|~-85|
|Worst episode|~-140|

The agent reliably solves the environment. Occasional worse scores occur due to discrete state rounding — tiny position/velocity misclassifications can cause a suboptimal action at a critical moment.

---

## Key Concepts Covered

- **Tabular Q-Learning** and the Bellman update equation
- **State discretization** of continuous observation spaces
- **ε-greedy exploration** with multiple decay schedules
- **Exploration vs. Exploitation** trade-off
- **Curse of dimensionality** in tabular methods
- **State aliasing** and its effect on policy quality
- **Reward shaping** with potential-based functions
- **Reward hacking** and misaligned objectives
- **Value function visualization** as a 3D surface

---

## Reproducibility

All random seeds are fixed. To reproduce any experiment exactly, keep `seed=400` (the default). Each experiment section can be re-run independently.