from game import PongGame
from dqn_agent import DQNAgent
import matplotlib.pyplot as plt
import torch
import random

env = PongGame()
agent = DQNAgent()

TRAINING_BOT_RANDOMNESS = 0.1
MIN_DISTANCE_FROM_BALL_TO_MOVE = 20
TARGET_UPDATE_FREQ = 100
TRAIN_FREQ = 4
MAX_EPISODE_STEPS = 1000

agent.HIT_BALL_REWARD = 0.1
agent.PROX_REWARD_MULTIPLIER = 0.01
agent.MOVEMENT_PENALTY = 0.001
agent.LOSING_PENALTY = 1
agent.WINNING_REWARD = 1

scores = []
episodes = 2000
stepCount = 0

for episode in range(episodes):
    state = env.reset()
    done = False
    score = 0
    episodeSteps = 0

    while not done:
        stepCount += 1
        episodeSteps += 1

        aiAction = agent.act(state)

        # Training Bot with some randomness
        trainingBotAction = 0
        if random.random() < TRAINING_BOT_RANDOMNESS:
            trainingBotAction = random.randint(0, 2)
        else:
            # Imperfect following with delay/error
            ballCenter = env.ball.y + env.ball.height // 2
            trainingBotCenter = env.player.y + env.player.height // 2
            diff = ballCenter - trainingBotCenter

            if abs(diff) > MIN_DISTANCE_FROM_BALL_TO_MOVE:  # Only move if ball is far from paddle center
                if diff > 0:
                    trainingBotAction = 2
                elif diff < 0:
                    trainingBotAction = 1

        nextState, reward, done = env.step(aiAction, trainingBotAction)
        agent.memorize(state, aiAction, reward, nextState, done)

        # Train every TRAIN_FREQ steps
        if stepCount % TRAIN_FREQ == 0:
            agent.trainStep()
        
        state = nextState
        score += reward
        # env.render() # Uncomment to render game during training (slow)

        # Prevent episodes from running too long
        if episodeSteps > MAX_EPISODE_STEPS:
            done = True
    
    # Update target network every UPDATE_TARGET_FREQ episodes
    if episode % TARGET_UPDATE_FREQ == 0:
        agent.updateTarget()
        print(f"Target network updated at episode {episode}")

    agent.epsilon = max(agent.epsilonMin, agent.epsilon * agent.epsilonDecay)

    scores.append(score)
    print(f"Episode {episode+1}/{episodes} | Score: {score} | Epsilon: {agent.epsilon:.3f}")


torch.save(agent.model.state_dict(), "pong_ai.pt")

plt.plot(scores)
plt.xlabel("Episode")
plt.ylabel("Scores")
plt.title("AI Training Progress")
plt.show()


env.close()