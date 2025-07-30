import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

HIDDEN_LAYER_SIZE = 128

class DQN(nn.Module):

    def __init__(self, inputSize, outputSize):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(inputSize, HIDDEN_LAYER_SIZE), nn.ReLU(),
            nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE), nn.ReLU(),
            nn.Linear(HIDDEN_LAYER_SIZE, outputSize)
        )

        def forward(self, x): # x is game state, returns 3 action scores
            return self.model(x)

class DQNAgent:
    def __init__(self):
        self.model = DQN(6, 3)
        self.target = DQN(6, 3)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)