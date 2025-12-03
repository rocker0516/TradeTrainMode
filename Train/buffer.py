import numpy as np
import torch
import random
from typing import Dict, Tuple, List

class ReplayBuffer:
    """
    Replay Buffer for Dict Observations (Price Seq + State Vector).
    Supports Multi-Dimensional Cost.
    """
    def __init__(
        self, 
        capacity: int, 
        price_seq_shape: Tuple[int, int], 
        state_dim: int, 
        action_dim: int,
        cost_dim: int = 1,
        device: torch.device = torch.device("cpu")
    ):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        self.cost_dim = cost_dim
        
        # Buffers
        self.price_seqs = np.zeros((capacity, *price_seq_shape), dtype=np.float32)
        self.state_vecs = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.costs = np.zeros((capacity, cost_dim), dtype=np.float32) # For Lagrangian
        
        self.next_price_seqs = np.zeros((capacity, *price_seq_shape), dtype=np.float32)
        self.next_state_vecs = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(
        self, 
        obs: Dict[str, np.ndarray], 
        action: np.ndarray, 
        reward: float, 
        cost: np.ndarray, 
        next_obs: Dict[str, np.ndarray], 
        done: bool
    ):
        self.price_seqs[self.ptr] = obs['price_seq']
        
        # Concatenate states for storage
        self.state_vecs[self.ptr] = np.concatenate([
            obs['account_state'],
            obs['time_state'],
            obs['rhythm_state'],
            obs['cost_state'],
            obs['market_state']
        ])
        
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.costs[self.ptr] = cost 
        
        self.next_price_seqs[self.ptr] = next_obs['price_seq']
        
        # Concatenate next states
        self.next_state_vecs[self.ptr] = np.concatenate([
            next_obs['account_state'],
            next_obs['time_state'],
            next_obs['rhythm_state'],
            next_obs['cost_state'],
            next_obs['market_state']
        ])
        
        self.dones[self.ptr] = float(done)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return {
            'price_seq': torch.FloatTensor(self.price_seqs[indices]).to(self.device),
            'state_vec': torch.FloatTensor(self.state_vecs[indices]).to(self.device),
            'action': torch.FloatTensor(self.actions[indices]).to(self.device),
            'reward': torch.FloatTensor(self.rewards[indices]).to(self.device),
            'cost': torch.FloatTensor(self.costs[indices]).to(self.device),
            'next_price_seq': torch.FloatTensor(self.next_price_seqs[indices]).to(self.device),
            'next_state_vec': torch.FloatTensor(self.next_state_vecs[indices]).to(self.device),
            'done': torch.FloatTensor(self.dones[indices]).to(self.device)
        }
