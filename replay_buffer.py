import numpy as np
import torch
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity=1000000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        
        return {
            'state': np.array(state),
            'action': np.array(action),
            'reward': np.array(reward),
            'next_state': np.array(next_state),
            'done': np.array(done)
        }
    
    def __len__(self):
        return len(self.buffer) 