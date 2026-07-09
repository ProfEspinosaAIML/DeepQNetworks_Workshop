# DeepQNetworks Workshop

This repository is a hands-on tutorial sequence for introducing Reinforcement
Learning (RL), Q-Learning, and Deep Q-Learning through the CartPole environment.
It is designed for an active learning class: students should run each notebook,
predict what will happen, change small parameters, compare outcomes, and explain
the learning behavior they observe.

## Learning Objectives

By the end of the workshop, students should be able to:

- Describe the RL loop: an agent observes a state, chooses an action, receives a
  reward, and transitions to a new state.
- Explain how Q-Learning estimates the expected long-term utility of taking an
  action in a state.
- Connect tabular Q-Learning to Deep Q-Learning, where a neural network
  approximates Q-values instead of storing every state-action value in a table.
- Interpret the role of linear layers and ReLU activations in a Deep Q-Network.
- Explain why experience replay improves training stability by sampling past
  transitions in random mini-batches.
- Explain why target networks stabilize DQN training by producing less volatile
  target Q-values.
- Read training curves and relate reward trends to exploration, exploitation,
  discounting, replay memory, and target-network synchronization.

## Recommended Execution Sequence

Follow this order to build the concepts from first principles to the full DQN
agent.

| Step | File | Purpose |
| --- | --- | --- |
| 1 | `LinearLayer-One.ipynb` | Start with a single-input, single-output linear layer. This is the smallest example of a learnable mapping. |
| 2 | `LinearLayer-Two.ipynb` | Extend one input to multiple outputs. Relate this to a network that predicts one Q-value per possible action. |
| 3 | `LinearLayer-Three.ipynb` | Use multiple inputs and outputs. Relate this to CartPole's multi-value state vector and action-value predictions. |
| 4 | `ReLUOne.ipynb` | Introduce ReLU non-linearity and discuss why DQN needs more than a purely linear mapping. |
| 5 | `GymCartPoleOne.ipynb` or `GymCartPoleTwo.ipynb` | Explore the CartPole environment, state space, action space, rewards, rendering, and random actions. Use `GymCartPoleOne.ipynb` locally and `GymCartPoleTwo.ipynb` in Google Colab. |
| 6 | `cartpole_q.py` | Run tabular Q-Learning with discretized CartPole states. This is the bridge between the RL loop and the DQN notebooks. |
| 7 | `DQN_Basic-Colab.ipynb` or `DQN-Basic.ipynb` | Train a Deep Q-Network with a policy network, target network, experience replay, epsilon-greedy exploration, and reward visualization. Use `DQN_Basic-Colab.ipynb` for the current Gymnasium API; `DQN-Basic.ipynb` uses the older Gym import style. |

## Concept Flow

### 1. Neural Network Building Blocks

The `LinearLayer-*` notebooks and `ReLUOne.ipynb` prepare students for the DQN
architecture before introducing reinforcement learning complexity. The key idea
is that a DQN receives an environment state and outputs action values. For
CartPole, the input is the state vector and the output is a Q-value for each
action: move left or move right.

Suggested active learning prompts:

- Before running each linear-layer cell, manually compute the expected output.
- Change one weight or bias and predict how the output will change.
- In `ReLUOne.ipynb`, test negative, zero, and positive inputs and explain why
  ReLU helps neural networks model non-linear patterns.

### 2. Reinforcement Learning Environment

The CartPole notebooks introduce the environment before any learning algorithm is
added. Students should identify:

- State: cart position, cart velocity, pole angle, and pole angular velocity.
- Actions: push the cart left or right.
- Reward: feedback received after each step.
- Episode termination: when the pole falls, the cart moves out of bounds, or the
  time limit is reached.

Suggested active learning prompts:

- Run a random policy and record how long it balances the pole.
- Change the environment seed and compare the initial state.
- Discuss why random actions are useful for exploration but insufficient for
  reliable control.

### 3. Tabular Q-Learning Baseline

`cartpole_q.py` demonstrates classic Q-Learning. Because CartPole has continuous
state values, the script discretizes position, velocity, angle, and angular
velocity into bins, then stores Q-values in a table.

Run the training baseline without rendering:

```powershell
python -c "import cartpole_q; cartpole_q.run(is_training=True, render=False)"
```

This creates:

- `cartpole.pkl`: the learned Q-table.
- `cartpole.png`: a plot of recent mean rewards.

Then run the learned policy with rendering:

```powershell
python -c "import cartpole_q; cartpole_q.run(is_training=False, render=True)"
```

Focus questions:

- What do `learning_rate_a`, `discount_factor_g`, and `epsilon` control?
- Why does the script reduce epsilon over time?
- What limitation appears when a continuous state space must be forced into a
  finite Q-table?

### 4. Deep Q-Learning

The DQN notebooks replace the Q-table with a neural network. The network predicts
Q-values for each action from the current state, making it possible to learn from
high-dimensional or continuous observations without manually creating a table for
every state-action pair.

Important components to trace in order:

1. `DQN` class: maps state inputs to action-value outputs.
2. `policy_net`: the network being trained.
3. `target_net`: a slower-moving copy used to compute target Q-values.
4. `replay_buffer`: stores `(state, action, reward, next_state, done)`
   transitions.
5. `train(...)`: samples replay mini-batches, computes Bellman targets, measures
   loss, and updates the policy network.
6. Training loop: balances epsilon-greedy exploration with exploitation and
   periodically synchronizes the target network.
7. Reward plot: shows whether the agent is improving over episodes.

Suggested active learning experiments:

- Lower `num_episodes` for a quick run, then increase it and compare the reward
  curve.
- Change `epsilon_decay` and observe how fast the agent shifts from exploration
  to exploitation.
- Change `sync_target_steps` and discuss how target-network update frequency
  affects stability.
- Change the replay buffer size or batch size and compare training smoothness.
- Temporarily remove target-network synchronization or replay sampling and
  discuss why training becomes less stable.

## Setup

Create and activate a virtual environment, then install the requirements:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Start Jupyter:

```powershell
jupyter notebook
```

For Google Colab, use the Colab-oriented notebooks:

- `GymCartPoleTwo.ipynb`
- `DQN_Basic-Colab.ipynb`

## Repository Map

- `LinearLayer-One.ipynb`, `LinearLayer-Two.ipynb`,
  `LinearLayer-Three.ipynb`: PyTorch linear-layer exercises.
- `ReLUOne.ipynb`: ReLU activation examples.
- `GymCartPoleOne.ipynb`: local CartPole environment exploration.
- `GymCartPoleTwo.ipynb`: Colab-compatible CartPole environment exploration.
- `cartpole_q.py`: tabular Q-Learning implementation for CartPole.
- `DQN-Basic.ipynb`: basic DQN notebook using the older `gym` package style.
- `DQN_Basic-Colab.ipynb`: DQN notebook using `gymnasium`, with explanatory
  markdown for DQN architecture, epsilon, gamma, replay buffers, loss, and
  optimization.
- `images/`: diagrams used by the linear-layer notebooks.
- `documents/3_Workbook_Linear.pdf`: supporting workbook for linear-layer
  exercises.
