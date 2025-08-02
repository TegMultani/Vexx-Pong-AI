# 🏓 AI Pong - Deep Q-Network Implementation

A classic Pong game featuring an AI opponent trained using Deep Q-Network (DQN) reinforcement learning.

## 🎮 Features

- **Smart AI Opponent**: DQN-based AI that learns optimal Pong strategies
- **Human vs AI Gameplay**: Play against the trained AI
- **Custom Training System**: Train your own VEXX PONG AI with configurable parameters
- **Reward-Based Learning**: Extended reward system for AI training
- **Real-time Visualization**: Watch the AI learn with matplotlib training progress plots

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

This will train a new AI for 2000 episodes and save the model as `pong_ai.pt`.

## 🧠 How It Works

### Deep Q-Network Architecture

The AI uses a neural network with:
- **Input**: 6-dimensional state vector (ball position, velocity, paddle positions)
- **Hidden Layers**: 2 layers with 128 neurons each + ReLU activation
- **Output**: 3 Q-values for actions (stay, move up, move down)

### Reward System

The AI learns through the default designed reward system:

| Action | Reward | Value |
|--------|--------|-------|
| Hit ball | Positive | +0.1 |
| Win point | Positive | +1.0 |
| Lose point | Negative | -1.0 |
| Stay close to ball | Positive | +0.01 * y-proximity |
| Unnecessary movement | Negative | -0.001 |

### Training Process

1. **Exploration vs Exploitation**: Uses ε-greedy strategy (starts at 100% random, decays to 1%)
2. **Experience Replay**: Stores experiences in memory buffer for batch learning
3. **Target Network**: Stabilizes training with periodic target network updates
4. **Training Opponent**: Plays against a bot with controlled randomness

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



## 🔧 Troubleshooting

### Common Issues

**Poor AI Performance**
- Train for more episodes
- Adjust reward parameters
- Increase network size

## 📚 Technical Details

### State Representation

The AI observes the game through a 6-dimensional normalized state vector:
```python
[ball_x, ball_y, ball_vel_x, ball_vel_y, player_y, ai_y]
```

### Action Space

The AI can choose from 3 discrete actions:
- `0`: Stay in place
- `1`: Move paddle up
- `2`: Move paddle down