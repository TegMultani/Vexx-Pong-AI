import pygame
import random
import numpy as np
from torch import normal

HIT_BALL_REWARD = 0.1
PROX_REWARD_MULTIPLIER = 0.01
MOVEMENT_PENALTY = 0.001
LOSING_PENALTY = 1
WINNING_REWARD = 1

SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

PADDLE_SPEED = 10

class PongGame:
    def __init__(self):
        pygame.init()
        self.display = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("VEXX PONG")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.ball = pygame.Rect(SCREEN_WIDTH//2, SCREEN_HEIGHT//2, 12, 12)
        self.player = pygame.Rect(SCREEN_WIDTH - 20, SCREEN_HEIGHT//2, 12, 60)
        self.ai = pygame.Rect(20, SCREEN_HEIGHT//2, 12, 60)
        self.ballVel = [random.choice([-4, 4]), random.choice([-4, 4])]
        self.playerVel = 0
        self.score = {"player": 0, "ai": 0}
        return self.get_state()
    
    def get_state(self):
        return np.array([
            self.ball.x / SCREEN_WIDTH,
            self.ball.y / SCREEN_HEIGHT,
            self.ballVel[0] / 10,
            self.ballVel[1] / 10,
            self.player.y / SCREEN_HEIGHT,
            self.ai.y / SCREEN_HEIGHT
        ], dtype=np.float32)
    

    def step(self, actionAi, action_player): # action = 0 (stay), 1 (up), 2 (down)
        # Store previous positions for reward calculation
        prevBallX = self.ball.x
        prevAiCenter = self.ai.y + self.ai.height // 2
        prevBallCenter = self.ball.y + self.ball.height // 2
        # Movement + Boundaries
        if actionAi == 1:
            self.ai.y -= PADDLE_SPEED
        elif actionAi == 2:
            self.ai.y += PADDLE_SPEED
        self.ai.y = max(0, min(SCREEN_HEIGHT - 60, self.ai.y))

        if action_player == 1:
            self.player.y -= PADDLE_SPEED
        elif action_player == 2:
            self.player.y += PADDLE_SPEED
        self.player.y = max(0, min(SCREEN_HEIGHT - 60, self.player.y))

        # Ball movement
        self.ball.x += self.ballVel[0]
        self.ball.y += self.ballVel[1]
        if self.ball.top <= 0 or self.ball.bottom >= SCREEN_HEIGHT:
            self.ballVel[1] *= -1

        reward = 0
        if self.ball.colliderect(self.ai):
            self.ballVel[0] *= -1
            # Where on the paddle the ball hit (-1 to 1)
            relativeIntersectY = (self.ai.y + self.ai.height / 2) - (self.ball.y + self.ball.height / 2)
            normalizedIntersectY = relativeIntersectY / (self.ai.height / 2)
            # Prevent extreme angles -1 -> 1
            normalizedIntersectY = max(-1, min(1, normalizedIntersectY))
            self.ballVel[1] = normalizedIntersectY * 6 # Max y velocity
            reward += HIT_BALL_REWARD
        elif self.ball.colliderect(self.player):
            self.ballVel[0] *= -1
            relativeIntersectY = (self.player.y + self.player.height / 2) - (self.ball.y + self.ball.height / 2)
            normalizedIntersectY = relativeIntersectY / (self.player.height / 2)
            normalizedIntersectY = max(-1, min(1, normalizedIntersectY))
            self.ballVel[1] = normalizedIntersectY * 6
        
        # Reward for staying close to ball vertically (for better positioning)
        afterAiCenter = self.ai.y + self.ai.height // 2
        distanceToBall = abs(prevBallCenter - afterAiCenter) # TODO: SHOULD IT BE PREVBALLCENTER OR CUR
        maxDistance = SCREEN_HEIGHT
        proximityReward = PROX_REWARD_MULTIPLIER * (1 - distanceToBall / maxDistance)
        reward += proximityReward

        # Small Penalty for unnecessary movement (to encourage efficiency)
        if actionAi != 0:
            reward -= MOVEMENT_PENALTY

        # Scoring
        done = False
        if self.ball.left <= 0:
            self.score["player"] += 1
            reward -= LOSING_PENALTY
            done = True
        elif self.ball.right >= SCREEN_WIDTH:
            self.score["ai"] += 1
            reward += WINNING_REWARD
            done = True

        return self.get_state(), reward, done

    def render(self):
        self.display.fill(BLACK)
        pygame.draw.rect(self.display, WHITE, self.player)
        pygame.draw.rect(self.display, WHITE, self.ai)
        pygame.draw.ellipse(self.display, WHITE, self.ball)
        pygame.display.update()
        self.clock.tick(60)

    def close(self):
        pygame.quit()
