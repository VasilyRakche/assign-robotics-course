import torch
import random
import numpy as np
from collections import deque
# from game import SnakeGameAI, Direction, Point
from lib.model import Linear_QNet, QTrainer
from lib.helper import plot

class Agent:

    def __init__(self, gamma = 0.95, learning_rate = 1e-3, memory_len = 100_000, layers_sizes = [3, 256, 12], batch_size = 1000):
        self.gamma = gamma # discount rate
        self.memory = deque(maxlen=memory_len) # popleft()
        self.batch_size = batch_size
        self.model = Linear_QNet(layers_sizes)
        self.trainer = QTrainer(self.model, lr=learning_rate, gamma = self.gamma)


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        return self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

