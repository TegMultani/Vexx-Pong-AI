import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

HIDDEN_LAYER_SIZE = 128
LEARNING_RATE = 0.0005
MEMORY_SIZE = 50000

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(6, 3).to(self.device)
        self.target = DQN(6, 3).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.batchSize = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilonMin = 0.01
        self.epsilonDecay = 0.9975

        self.updateTarget()

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 2)  
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            qValues = self.model(state)
            return torch.argmax(qValues).item()

    def memorize(self, state, action, reward, nextState, done):
        self.memory.append((state, action, reward, nextState, done))

    def trainStep(self):
        if len(self.memory) < self.batchSize:
            return

        batch = random.sample(self.memory, self.batchSize)
        states, actions, rewards, nextStates, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        nextStates = torch.tensor(np.array(nextStates), dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

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