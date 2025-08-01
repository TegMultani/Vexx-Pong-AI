from game import PongGame
from dqn_agent import DQNAgent
import matplotlib.pyplot as plt
import torch

env = PongGame()
agent = DQNAgent()

scores = []
episodes = 500


for episode in range(episodes):
    state = env.reset()
    done = False
    score = 0

    while not done:
        aiAction = agent.act(state)
        trainingBotAction = 0
        if env.ball.y > env.player.y:
            trainingBotAction = 2
        elif env.ball.y < env.player.y:
            trainingBotAction = 1
        nextState, reward, done = env.step(aiAction, trainingBotAction)
        agent.memorize(state, aiAction, reward, nextState, done)
        agent.trainStep()
        state = nextState
        score += reward
        # env.render() # Uncomment to render game during training (slow)
    
    agent.updateTarget()
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