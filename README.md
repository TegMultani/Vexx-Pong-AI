# 🏓 AI Pong - Deep Q-Network Implementation

A classic Pong game featuring an AI opponent trained using Deep Q-Network (DQN) reinforcement learning.

## 🎮 Features

- **Smart AI Opponent**: DQN-based AI that learns optimal Pong strategies
- **Human vs AI Gameplay**: Play against the trained AI
- **Custom Training System**: Train your own VEXX PONG AI with configurable parameters
- **Reward-Based Learning**: Extended reward system for AI training
- **Real-time Visualization**: Watch the AI learn with matplotlib training progress plots
- **Classic Pong Physics**: Consistent ball speed with realistic paddle physics

## 🚀 Quick Start

### Prerequisites

```bash
pip install pygame torch numpy matplotlib
```

### Play Against AI

```bash
python play.py
```

**Controls:**
- `↑` Arrow Key: Move paddle up
- `↓` Arrow Key: Move paddle down
- Close window or `Ctrl+C` to quit

### Train Your Own AI

```bash
python train.py
```

This will train a new AI for 3000 episodes and save the model as `pong_ai.pt`.

## 🧠 How It Works

### Deep Q-Network Architecture

The AI uses a neural network with:
- **Input**: 7-dimensional state vector (ball position, velocity, paddle positions, distances)
- **Hidden Layers**: 2 layers with 128 neurons each + ReLU activation
- **Output**: 3 Q-values for actions (stay, move up, move down)

### Reward System
W
The AI learns through a sophisticated reward system:

| Action | Reward | Value |
|--------|--------|-------|
| Hit ball | Positive | +0.2 |
| Win point | Positive | +1.0 |
| Lose point | Negative | -1.0 |
| Reduce distance to ball (when ball moving toward AI) | Positive | +0.05 * improvement |
| Unnecessary movement | Negative | -0.001 |

### Training Process

1. **Exploration vs Exploitation**: Uses ε-greedy strategy (starts at 100% random, decays to 1%)
2. **Experience Replay**: Stores experiences in memory buffer for batch learning
3. **Target Network**: Stabilizes training with periodic target network updates (every 2000 steps)
4. **Curriculum Learning**: Opponent gradually becomes less random and faster over 300,000 steps
5. **Per-step Updates**: Epsilon decay and target network updates happen every step for smooth learning

### Training Parameters

- **Episodes**: 3000
- **Max Steps per Episode**: 1000
- **Training Frequency**: Every 4 steps
- **Batch Size**: 64
- **Learning Rate**: 0.0005
- **Memory Size**: 50,000 experiences
- **Epsilon Decay**: 0.9995 per step
- **Curriculum Steps**: 300,000 (opponent randomness: 0.30 → 0.05, speed: 8 → 10)

## 📁 Project Structure

```
PONG/
├── game.py           # Core game mechanics and physics
├── dqn_agent.py      # DQN neural network and agent logic
├── train.py          # AI training script
├── play.py           # Human vs AI gameplay
├── pong_ai.pt        # Pre-trained AI model
└── README.md         # This file
```

## 🎯 Training Your Own AI

1. **Modify Parameters**: Adjust training parameters in `train.py`
2. **Run Training**: Execute `python train.py`
3. **Test Performance**: Use `python play.py` to test your trained AI

### Configurable Parameters

You can customize the training environment in `train.py`:

```python
# Environment rewards
hit_ball_reward=0.2
prox_reward_multiplier=0.05
movement_penalty=0.001
losing_penalty=1.0
winning_reward=1.0

# Curriculum settings
TRAINING_BOT_RANDOMNESS_START = 0.30
TRAINING_BOT_RANDOMNESS_END = 0.05
TRAINING_BOT_SPEED_START = 8.0
TRAINING_BOT_SPEED_END = 10.0
CURRICULUM_STEPS = 300_000
```

## 🔧 Troubleshooting

### Common Issues

**Poor AI Performance**
- Train for more episodes (increase `episodes` in train.py)
- Adjust reward parameters (increase `hit_ball_reward` if missing often)
- Reduce `movement_penalty` if AI jitters too much
- Increase `prox_reward_multiplier` if AI camps in center
- Slow curriculum by increasing `CURRICULUM_STEPS`

**Ball Physics Issues**
- Ball speed is now constant at 4 pixels/frame horizontally
- Only Y velocity changes based on paddle hit position
- Classic Pong physics implemented

## 📚 Technical Details

### State Representation

The AI observes the game through a 7-dimensional normalized state vector:
```python
[
    ball_x / SCREEN_WIDTH,           # Normalized ball X position
    ball_y / SCREEN_HEIGHT,          # Normalized ball Y position  
    ball_vel_x / 10,                 # Normalized ball X velocity
    ball_vel_y / 10,                 # Normalized ball Y velocity
    ai_y / SCREEN_HEIGHT,            # Normalized AI paddle Y position
    (ball_y - ai_y) / SCREEN_HEIGHT, # Vertical distance to ball
    (ball_x - ai_x) / SCREEN_WIDTH   # Horizontal distance to ball
]
```

### Action Space

The AI can choose from 3 discrete actions:
- `0`: Stay in place
- `1`: Move paddle up
- `2`: Move paddle down

### Game Physics

- **Ball Speed**: Constant 4 pixels/frame horizontally
- **Paddle Physics**: Y velocity changes based on hit position (-6 to +6)
- **Boundary Bouncing**: Ball bounces off top/bottom walls
- **Scoring**: Point when ball passes opponent's paddle