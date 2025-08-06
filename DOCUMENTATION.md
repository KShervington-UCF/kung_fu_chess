# Kung Fu Chess Gymnasium Environment Documentation

## Overview

The Kung Fu Chess Gymnasium Environment is a complete implementation of the real-time chess variant "Kung Fu Chess" compatible with OpenAI Gymnasium. This environment allows for training reinforcement learning agents on a unique chess variant where pieces move simultaneously with cooldown mechanics.

## Key Features

- **Real-time Mechanics**: Pieces move simultaneously rather than in turns
- **Cooldown System**: Each piece has a cooldown period after moving
- **Collision Resolution**: First-mover advantage with special knight rules
- **Two Game Modes**: Standard (slow) and Lightning (fast) variants
- **Gymnasium Compatible**: Full compliance with Gymnasium API
- **Rendering Support**: Both human-viewable and RGB array rendering
- **Comprehensive Testing**: Unit tests and example usage included

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from kung_fu_chess_env import KungFuChessEnv, GameMode

# Create environment
env = KungFuChessEnv(mode=GameMode.LIGHTNING, render_mode="human")

# Reset and run
observation, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated:
        break

env.close()
```

## Environment Specifications

### Action Space

The action space is a `MultiDiscrete([16, 8, 8])` representing:
- `piece_index` (0-15): Index of the piece to move from current player's pieces
- `target_row` (0-7): Target row on the chess board
- `target_col` (0-7): Target column on the chess board

### Observation Space

The observation space is a dictionary containing:

```python
{
    'board': Box(low=0, high=12, shape=(8, 8), dtype=int8),
    'cooldowns': Box(low=0, high=1, shape=(32,), dtype=float32),
    'current_player': Discrete(2),
    'game_time': Box(low=0, high=inf, shape=(1,), dtype=float32)
}
```

- **board**: 8x8 grid representing piece positions (0=empty, 1-12=piece types)
- **cooldowns**: Normalized cooldown states for all 32 pieces (0=ready, 1=max cooldown)
- **current_player**: Current player (0=White, 1=Black)
- **game_time**: Time elapsed since game start

### Rewards

The reward system is designed to encourage strategic play and learning:

**Basic Action Rewards:**
- `+0.1`: Valid move attempt
- `-0.1`: Invalid action (cooldown violation, moving already moving piece)
- `-0.2`: Invalid move (illegal chess move)

**Piece Capture Rewards:**
- `+1.0`: Capturing enemy pawn
- `+3.0`: Capturing enemy knight or bishop
- `+5.0`: Capturing enemy rook
- `+9.0`: Capturing enemy queen
- `+100.0`: Capturing enemy king (winning)
- Negative equivalents for losing pieces
- `+0.5x piece_value`: Winning collision (partial reward)

**Game Outcome Rewards:**
- `+1000.0`: Winning the game (capturing enemy king)
- `-1000.0`: Losing the game (losing your king)

**Positional Rewards (small bonuses):**
- Center control: Small bonus for pieces near board center
- Pawn advancement: Bonus for advancing pawns toward promotion
- Knight centralization: Knights perform better in center
- King safety: Early game bonus for keeping king protected

## Game Rules

### Basic Rules
- All normal chess rules apply for piece movement
- No checkmate - must capture the King to win
- Pieces have cooldown periods after moving
- Multiple pieces can move simultaneously

### Cooldown System
- **Standard Mode**: 10 seconds cooldown, 1 square/second movement
- **Lightning Mode**: 0.2 seconds cooldown, 5 squares/second movement

### Collision Resolution
1. If multiple pieces target the same square:
   - First mover wins (based on move start time)
   - Exception: Knights cannot collide during movement
2. Knights kill any piece they land on (including own pieces)

### Movement Mechanics
- Pieces take time to move based on distance and game speed
- During movement, pieces are not visible on the board
- "Illegal" moves are allowed if pieces can move out of the way

## Game Modes

### Standard Mode
- Cooldown: 10 seconds
- Movement speed: 1 square per second
- More strategic, allows for planning

### Lightning Mode  
- Cooldown: 0.2 seconds
- Movement speed: 5 squares per second
- Fast-paced, reaction-based gameplay

## API Reference

### KungFuChessEnv

#### Constructor
```python
KungFuChessEnv(mode=GameMode.STANDARD, render_mode=None)
```

**Parameters:**
- `mode`: Game speed mode (GameMode.STANDARD or GameMode.LIGHTNING)
- `render_mode`: Rendering mode ("human", "rgb_array", or None)

#### Methods

##### reset(seed=None, options=None)
Reset the environment to initial state.

**Returns:** `(observation, info)`

##### step(action)
Execute one step in the environment.

**Parameters:**
- `action`: Array of [piece_index, target_row, target_col]

**Returns:** `(observation, reward, terminated, truncated, info)`

##### render()
Render the current state.

**Returns:** RGB array if render_mode="rgb_array", None otherwise

##### close()
Clean up resources.

## Examples

### Random Agent
```python
import numpy as np
from kung_fu_chess_env import KungFuChessEnv, GameMode

env = KungFuChessEnv(mode=GameMode.LIGHTNING)
obs, info = env.reset()

for step in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated:
        print(f"Game over! Winner: {info['winner']}")
        break

env.close()
```

### Training with PPO (Recommended)

The project includes professional PPO training and evaluation scripts:

```bash
# Install dependencies
pip install -r requirements.txt

# Train agent (100k timesteps, ~30-60 minutes)
python ppo_train.py

# Evaluate trained agent
python ppo_eval.py
```

**Training Features:**
- Vectorized environments for 4x faster training
- Optimized hyperparameters for chess environments
- TensorBoard logging and monitoring
- Automatic model checkpointing and evaluation
- Best model saving based on performance

**Evaluation Features:**
- Comprehensive performance metrics
- Statistical analysis across 100 games
- Visualization plots and charts
- Win/loss rate analysis
- Reward component breakdown

## Training and Evaluation Guide

For comprehensive training and evaluation instructions, see **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** which covers:

- **Training Deep Dive**: Hyperparameters, network architecture, and configuration options
- **Evaluation Analysis**: Understanding metrics, performance indicators, and results interpretation
- **Monitoring**: TensorBoard usage, progress tracking, and debugging techniques
- **Advanced Techniques**: Curriculum learning, self-play, and multi-agent training
- **Troubleshooting**: Common issues and solutions for training problems
- **Best Practices**: Systematic experimentation and optimization strategies

### Quick Training Commands

```bash
# Basic training (100k timesteps, ~30-60 minutes)
python ppo_train.py

# Monitor training progress
tensorboard --logdir ./logs/

# Evaluate trained agent
python ppo_eval.py

# View training outputs
ls ./models/          # Trained models
ls ./logs/            # Training logs
```

## Testing

Run the test suite:
```bash
python test_environment.py
```

Run example usage:
```bash
python example_usage.py
```

## Architecture

The environment follows SOLID principles:

- **Single Responsibility**: Each class has a clear, single purpose
- **Open/Closed**: Easy to extend with new piece types or rules
- **Liskov Substitution**: Proper inheritance hierarchy
- **Interface Segregation**: Clean, minimal interfaces
- **Dependency Inversion**: Depends on abstractions, not concretions

### Key Classes

- `KungFuChessEnv`: Main environment class implementing Gymnasium interface
- `Piece`: Represents individual chess pieces with state
- `PieceType`: Enumeration of chess piece types
- `Color`: Player color enumeration
- `GameMode`: Game speed mode enumeration

## Performance Considerations

- Efficient collision detection using position grouping
- Minimal memory allocation during gameplay
- Optimized rendering for real-time performance
- Numpy arrays for fast numerical operations

## Limitations

- Currently supports 2-player games only
- No en passant or castling (can be added)
- Simple rendering (can be enhanced with better graphics)
- No network play (local only)

## Future Enhancements

- Enhanced graphics and animations
- Network multiplayer support
- Additional chess variants
- Tournament mode
- Advanced AI opponents
- Performance optimizations for large-scale training
