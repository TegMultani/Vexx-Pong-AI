from game import PongGame
from dqn_agent import DQNAgent
import matplotlib.pyplot as plt
import torch
import random

# Configure environment and training curriculum
env = PongGame(
    hit_ball_reward=0.2,
    prox_reward_multiplier=0.05,
    movement_penalty=0.001,
    losing_penalty=1.0,
    winning_reward=1.0,
    ai_paddle_speed=10,
    player_paddle_speed=8,
)
agent = DQNAgent()

# Opponent curriculum (smooth, per-step)
TRAINING_BOT_RANDOMNESS_START = 0.30
TRAINING_BOT_RANDOMNESS_END = 0.05
TRAINING_BOT_SPEED_START = 8.0
TRAINING_BOT_SPEED_END = 10.0
CURRICULUM_STEPS = 300_000  # steps to reach end values

MIN_DISTANCE_FROM_BALL_TO_MOVE = 20
TARGET_UPDATE_FREQ_STEPS = 2000
TRAIN_FREQ = 4
MAX_EPISODE_STEPS = 1000

# Keep epsilon decay reasonably slow to ensure exploration (per-step decay)
agent.epsilonDecay = 0.9995

scores = []
episodes = 3000
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

        # Smooth curriculum: interpolate randomness and speed based on total steps
        progress = min(1.0, stepCount / CURRICULUM_STEPS)
        TRAINING_BOT_RANDOMNESS = (
            TRAINING_BOT_RANDOMNESS_START * (1.0 - progress)
            + TRAINING_BOT_RANDOMNESS_END * progress
        )
        TRAINING_BOT_SPEED = (
            TRAINING_BOT_SPEED_START
            + (TRAINING_BOT_SPEED_END - TRAINING_BOT_SPEED_START) * progress
        )
        # Apply opponent speed to environment
        env.player_paddle_speed = TRAINING_BOT_SPEED

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

        # Update target network on a step schedule for smoother updates
        if stepCount % TARGET_UPDATE_FREQ_STEPS == 0:
            agent.updateTarget()
            print(f"Target network updated at step {stepCount}")

        # Per-step epsilon decay
        agent.epsilon = max(agent.epsilonMin, agent.epsilon * agent.epsilonDecay)
        
        state = nextState
        score += reward
        # env.render() # Uncomment to render game during training (slow)

        # Prevent episodes from running too long
        if episodeSteps > MAX_EPISODE_STEPS:
            done = True
    
    # (moved target update and epsilon decay into step loop)

    scores.append(score)
    print(f"Episode {episode+1}/{episodes} | Score: {score:.3f} | Epsilon: {agent.epsilon:.3f} | BotRnd: {TRAINING_BOT_RANDOMNESS:.3f} | BotSpd: {TRAINING_BOT_SPEED:.3f}")


torch.save(agent.model.state_dict(), "pong_ai.pt")

plt.plot(scores)
plt.xlabel("Episode")
plt.ylabel("Scores")
plt.title("AI Training Progress")
plt.show()


env.close()