import pygame
import random
import numpy as np

SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

PADDLE_SPEED = 5

class PongGame:
    def __init__(self):
        pygame.init()
        self.display = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("AI PONG BY SDOT BOYZ")
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
    

    def step(self, action_ai): # action_ai = 0 (stay), 1 (up), 2 (down)
        if action_ai == 1:
            self.ai.y -= PADDLE_SPEED
        elif action_ai == 2:
            self.ai.y += PADDLE_SPEED

        # Boundaries
        self.ai.y = max(0, min(SCREEN_HEIGHT - 60, self.ai.y))
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: self.player.y -= 5
        if keys[pygame.K_DOWN]: self.player.y += 5
        self.player.y = max(0, min(SCREEN_HEIGHT - 70, self.player.y))

        # Ball movement
        self.ball.x += self.ballVel[0]
        self.ball.y += self.ballVel[1]

        if self.ball.top <= 0 or self.ball.bottom >= SCREEN_HEIGHT:
            self.ballVel[1] *= -1

        if self.ball.colliderect(self.player) or self.ball.colliderect(self.ai):
            self.ballVel[0] *= -1
        
        # Scoring
        reward = 0
        done = False
        if self.ball.left <= 0:
            self.score["player"] += 1
            reward = -1
            done = True
        elif self.ball.right >= SCREEN_WIDTH:
            self.score["ai"] += 1
            reward = 1
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
