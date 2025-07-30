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
            nn.Linear(inputSize, HIDDEN_LAYER_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER_SIZE, outputSize)
        )

        def forward(self, x): # x is game state, returns 3 action scores
            return self.model(x)

class DQNAgent:
    def __init__(self):
        self.model = DQN(6, 3)
        self.target = DQN(6, 3) 
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = deque(maxlen=10000)
        self.batchSize = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilonMin = 0.1
        self.epsilonDecay = 0.995

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 2)  
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32)
            qValues = self.model(state)
            return torch.argmax(qValues).item()

    def memorize(self, state, action, reward, nextState, done):
        self.memory.append((state, action, reward, nextState, done))

    def train_step(self):
        if len(self.memory) < self.batchSize:
            return

        batch = random.sample(self.memory, self.batchSize)
        states, actions, rewards, nextStates, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        nextStates = torch.tensor(nextStates, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)

        qValues = self.model(states)
        nextQValues = self.target(nextStates).detach()
        qTarget = qValues.clone()

        for i in range(self.batchSize):
            qTarget[i, actions[i]] = rewards[i] + self.gamma * torch.max(nextQValues[i]) * (1 - dones[i])

        loss = nn.functional.mse_loss(qValues, qTarget)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def updateTarget(self):
        self.target.load_state_dict(self.model.state_dict())