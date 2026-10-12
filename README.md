# Building a Custom Reinforcement Learning Environment

Practice project for building custom [Gymnasium](https://gymnasium.farama.org/) environments from scratch and training tabular Q-learning agents on them.

## Environments

### 1. GridEnv (`GridworldEnv.ipynb`)

A grid-world environment where the agent starts at the top-left corner `(0, 0)` and must learn to navigate to the goal at the bottom-right corner, while avoiding the trap at the bottom-left.

- **Actions**: 4 discrete moves — up, down, left, right
- **Rewards**:
  - Goal (bottom-right corner): `+50`
  - Trap (bottom-left corner): `-50`
  - Every other step: `-1`
- **Episode ends** when the agent reaches the goal or the trap.

### 2. ShowerEnv (`ShowerEnv.ipynb`)

A temperature-control environment where the agent adjusts the hot water flow rate to reach a desired shower temperature of 35°C.

- **State**: current outlet temperature
- **Actions**: 21 discrete valve angle changes (`(action - 10) * 10`)
- **Rewards**: tiered by how close the temperature is to the target (`+10` within the error band, down to `-10` far away)
- **Episode ends** when the temperature falls within the acceptable error band.

## Agents

Two tabular Q-learning agents with epsilon-greedy exploration:

| File | Description |
|------|-------------|
| `Agent.py` | Q-table keyed by discrete states — used for GridEnv. |
| `Agent2.py` | Same algorithm, but converts states to tuples so it can handle the numerical state of ShowerEnv. |

Both implement:

- **Epsilon-greedy action selection** (`choose_actions`)
- **Q-value updates via the Bellman equation** (`learn`)
- **Epsilon decay** to shift from exploration to exploitation over training (`decrement_epsilon`)

## Getting Started

### Requirements

- Python 3.12
- `numpy`, `gymnasium`, `matplotlib`

```bash
pip install numpy gymnasium matplotlib
```

### Running

Open either notebook in Jupyter and run the cells:

```bash
jupyter notebook
```

Each notebook trains the agent for 1000 episodes and plots the learning curve (total reward per episode) as the agent learns.
