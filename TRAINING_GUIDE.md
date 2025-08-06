# Kung Fu Chess RL Training and Evaluation Guide

This guide provides comprehensive instructions for training and evaluating reinforcement learning agents on the Kung Fu Chess environment using PPO (Proximal Policy Optimization).

## Table of Contents

1. [Quick Start](#quick-start)
2. [Training Deep Dive](#training-deep-dive)
3. [Evaluation and Analysis](#evaluation-and-analysis)
4. [Hyperparameter Tuning](#hyperparameter-tuning)
5. [Monitoring and Debugging](#monitoring-and-debugging)
6. [Advanced Techniques](#advanced-techniques)
7. [Troubleshooting](#troubleshooting)

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from kung_fu_chess_env import KungFuChessEnv; print('Environment ready!')"
```

### Basic Training

```bash
# Train agent (100k timesteps, ~30-60 minutes)
python ppo_train.py

# Evaluate trained agent
python ppo_eval.py
```

## Training Deep Dive

### Understanding the Training Script

The `ppo_train.py` script implements professional-grade RL training with the following features:

#### 1. Vectorized Environments
```python
# 4 parallel environments for faster training
env = SubprocVecEnv([make_env(game_mode, i) for i in range(4)])
```

**Benefits:**
- 4x faster data collection
- Better exploration through environment diversity
- More stable gradient updates

#### 2. Optimized Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `learning_rate` | 3e-4 | Balanced learning speed |
| `n_steps` | 2048 | Steps per environment per update |
| `batch_size` | 64 | Minibatch size for optimization |
| `n_epochs` | 10 | Epochs per policy update |
| `gamma` | 0.99 | Discount factor for future rewards |
| `gae_lambda` | 0.95 | GAE parameter for advantage estimation |
| `clip_range` | 0.2 | PPO clipping parameter |
| `ent_coef` | 0.01 | Entropy bonus for exploration |

#### 3. Network Architecture
```python
policy_kwargs=dict(
    net_arch=dict(pi=[256, 256], vf=[256, 256]),
    activation_fn=torch.nn.ReLU
)
```

- **Policy Network**: 256→256→actions (for action selection)
- **Value Network**: 256→256→1 (for value estimation)
- **Activation**: ReLU for stable gradients

### Training Configuration Options

#### Game Modes

```python
# Fast-paced training (recommended for initial experiments)
GAME_MODE = GameMode.LIGHTNING  # 0.2s cooldown, 5 sq/s

# Strategic training (for advanced agents)
GAME_MODE = GameMode.STANDARD   # 10s cooldown, 1 sq/s
```

#### Training Scale

```python
# Quick test (5-10 minutes)
TOTAL_TIMESTEPS = 50_000

# Standard training (30-60 minutes)
TOTAL_TIMESTEPS = 100_000

# Deep training (2-4 hours)
TOTAL_TIMESTEPS = 1_000_000
```

#### Parallel Environments

```python
# Single environment (slower, more deterministic)
N_ENVS = 1

# Balanced (recommended)
N_ENVS = 4

# High-performance (if you have 8+ CPU cores)
N_ENVS = 8
```

### Training Outputs

The training script creates several important outputs:

```
./models/
├── ppo_kung_fu_chess_best.zip      # Best performing model
├── ppo_kung_fu_chess_final.zip     # Final model after training
└── ppo_kung_fu_chess_checkpoints/  # Regular checkpoints
    ├── ppo_kung_fu_chess_50000_steps.zip
    ├── ppo_kung_fu_chess_100000_steps.zip
    └── ...

./logs/
├── PPO_KungFuChess_1/              # TensorBoard logs
├── train_env_0.monitor.csv         # Environment 0 episode logs
├── train_env_1.monitor.csv         # Environment 1 episode logs
└── evaluations.npz                 # Evaluation results
```

## Evaluation and Analysis

### Running Evaluation

```bash
python ppo_eval.py
```

The evaluation script provides:

1. **Single Game Visualization**: Watch the agent play with rendering
2. **Statistical Analysis**: 100-game performance evaluation
3. **Performance Plots**: Visual analysis of agent behavior

### Understanding Evaluation Metrics

#### Win Rate Analysis
```
Total Games: 100
Wins: 45 (45.0%)
Losses: 52 (52.0%)
Draws: 3 (3.0%)
```

**Interpretation:**
- **>60%**: Excellent performance
- **40-60%**: Good performance
- **20-40%**: Learning in progress
- **<20%**: Needs more training

#### Reward Breakdown
```
Reward Statistics:
  Mean: 15.23 ± 45.67
  Range: [-89.45, 156.78]
  Median: 12.34

Capture Statistics:
  Mean capture reward: 8.45
  Mean pieces captured per game: 3.2
  Total pieces captured: 320
```

**Key Insights:**
- **High capture rewards**: Agent learned to prioritize valuable pieces
- **Positive mean reward**: Agent performs better than random
- **Low variance**: Consistent performance

#### Game Length Analysis
```
Game Length Statistics:
  Mean: 234.5 ± 67.8 steps
  Range: [45, 456] steps
  Median: 221.0 steps
```

**Interpretation:**
- **Short games**: Quick decisive play (good or bad)
- **Long games**: Cautious/strategic play
- **High variance**: Inconsistent strategy

### Performance Visualization

The evaluation generates four key plots:

1. **Reward Distribution**: Shows consistency of performance
2. **Game Length Distribution**: Indicates playing style
3. **Capture vs Positional Rewards**: Strategic balance analysis
4. **Win/Loss Pie Chart**: Overall success rate

## Hyperparameter Tuning

### Learning Rate Scheduling

```python
# Constant learning rate (default)
learning_rate=3e-4

# Linear decay
learning_rate=lambda progress: 3e-4 * (1 - progress)

# Exponential decay
learning_rate=lambda progress: 3e-4 * (0.95 ** (progress * 100))
```

### Network Architecture Experiments

```python
# Smaller network (faster training)
net_arch=dict(pi=[128, 128], vf=[128, 128])

# Larger network (more capacity)
net_arch=dict(pi=[512, 512], vf=[512, 512])

# Asymmetric networks
net_arch=dict(pi=[256, 128], vf=[512, 256])
```

### Exploration vs Exploitation

```python
# More exploration (early training)
ent_coef=0.05

# Less exploration (fine-tuning)
ent_coef=0.001

# Adaptive entropy
ent_coef=lambda progress: 0.05 * (1 - progress)
```

## Monitoring and Debugging

### TensorBoard Monitoring

```bash
# Start TensorBoard
tensorboard --logdir ./logs/

# Open browser to http://localhost:6006
```

**Key Metrics to Watch:**

1. **Episode Reward**: Should generally increase over time
2. **Episode Length**: May decrease as agent learns efficient play
3. **Policy Loss**: Should stabilize after initial fluctuations
4. **Value Loss**: Should decrease and stabilize
5. **Entropy**: Should decrease as policy becomes more confident

### Training Progress Indicators

#### Healthy Training Signs
- Episode rewards trending upward
- Decreasing policy and value losses
- Stable entropy (not too high or low)
- Evaluation scores improving

#### Warning Signs
- Rewards plateauing early
- Exploding gradients (very high losses)
- Entropy collapsing to zero
- Evaluation performance degrading

### Common Issues and Solutions

#### Issue: Agent Not Learning
**Symptoms**: Flat reward curves, random-like behavior
**Solutions**:
- Increase learning rate: `learning_rate=1e-3`
- Reduce batch size: `batch_size=32`
- Increase entropy coefficient: `ent_coef=0.05`

#### Issue: Training Instability
**Symptoms**: Oscillating losses, inconsistent performance
**Solutions**:
- Decrease learning rate: `learning_rate=1e-4`
- Increase batch size: `batch_size=128`
- Add gradient clipping: `max_grad_norm=0.3`

#### Issue: Overfitting
**Symptoms**: Training performance >> evaluation performance
**Solutions**:
- Increase environment diversity
- Add regularization: higher `ent_coef`
- Reduce network size

## Advanced Techniques

### Curriculum Learning

Start with easier settings and gradually increase difficulty:

```python
# Phase 1: Lightning mode, 50k timesteps
train_ppo_agent(50_000, GameMode.LIGHTNING)

# Phase 2: Standard mode, 100k timesteps
train_ppo_agent(100_000, GameMode.STANDARD)
```

### Self-Play Training

Train agents against previous versions of themselves:

```python
# Load previous best model as opponent
opponent_model = PPO.load("./models/ppo_kung_fu_chess_best.zip")

# Implement self-play environment wrapper
# (Advanced: requires custom environment modification)
```

### Multi-Agent Training

Train multiple agents simultaneously:

```python
# Use different random seeds for diversity
agents = []
for seed in [42, 123, 456, 789]:
    agent = train_ppo_agent(seed=seed)
    agents.append(agent)

# Tournament evaluation
evaluate_tournament(agents)
```

## Troubleshooting

### Installation Issues

```bash
# CUDA issues
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Pygame issues on Linux
sudo apt-get install python3-pygame

# Stable-Baselines3 issues
pip install stable-baselines3[extra]
```

### Memory Issues

```bash
# Reduce parallel environments
N_ENVS = 2

# Reduce batch size
batch_size = 32

# Reduce network size
net_arch = dict(pi=[128, 128], vf=[128, 128])
```

### Performance Issues

```bash
# Use CPU if GPU is slow
device = 'cpu'

# Reduce logging frequency
eval_freq = 50000

# Disable rendering during training
render_mode = None
```

## Best Practices Summary

1. **Start Small**: Begin with 50k timesteps to verify setup
2. **Monitor Closely**: Use TensorBoard for real-time monitoring
3. **Save Frequently**: Checkpoints prevent lost training progress
4. **Evaluate Regularly**: Use evaluation callback to track progress
5. **Experiment Systematically**: Change one hyperparameter at a time
6. **Document Results**: Keep notes on what works and what doesn't

## Expected Results

With proper training, you should expect:

- **50k timesteps**: Basic move validity, some piece captures
- **100k timesteps**: Strategic piece development, 30-50% win rate
- **500k timesteps**: Advanced tactics, 60-70% win rate vs random
- **1M+ timesteps**: Near-expert play, complex strategic planning

Remember that Kung Fu Chess is a complex real-time strategy game, so achieving strong performance requires patience and experimentation with different approaches!
