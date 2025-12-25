import numpy as np
import torch
import random
from typing import Dict, Tuple, List, Optional

class ReplayBuffer:
    """
    Replay Buffer for Dict Observations (Price Seq + Daily Seq + State Vector).
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
        
        # Determine daily seq shape if available (inferred from usage, or need passing)
        # For now, we will lazily init or expect it passed in constructor if we want to be strict.
        # But to avoid breaking signature too much, let's assume standard shape or handle dynamically?
        # No, numpy arrays need fixed shape. We should update __init__ signature in train.py
        # But wait, train.py initializes this. We need to update train.py too.
        # Let's add daily_seq_shape to __init__ with default None for backward compat (though we are changing everything).
        
        # We'll assume the user updates train.py to pass daily_seq_shape.
        # But wait, I can only edit this file. I should make it flexible or hardcode a default if not passed?
        # Actually, looking at previous step, I am modifying train.py later.
        # So I will add daily_seq_shape to __init__.
        
    def __init__(
        self, 
        capacity: int, 
        price_seq_shape: Tuple[int, int], 
        state_dim: int, 
        action_dim: int,
        cost_dim: int = 1,
        device: torch.device = torch.device("cpu"),
        daily_seq_shape: Optional[Tuple[int, int]] = None
    ):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        self.cost_dim = cost_dim
        
        # Buffers
        self.price_seqs = np.zeros((capacity, *price_seq_shape), dtype=np.float32)
        if daily_seq_shape is not None:
            self.daily_seqs = np.zeros((capacity, *daily_seq_shape), dtype=np.float32)
        else:
            self.daily_seqs = None
            
        self.state_vecs = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.costs = np.zeros((capacity, cost_dim), dtype=np.float32) # For Lagrangian
        
        self.next_price_seqs = np.zeros((capacity, *price_seq_shape), dtype=np.float32)
        if daily_seq_shape is not None:
            self.next_daily_seqs = np.zeros((capacity, *daily_seq_shape), dtype=np.float32)
        else:
            self.next_daily_seqs = None
            
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
        if self.daily_seqs is not None and 'daily_seq' in obs:
            self.daily_seqs[self.ptr] = obs['daily_seq']
        
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
        if self.next_daily_seqs is not None and 'daily_seq' in next_obs:
            self.next_daily_seqs[self.ptr] = next_obs['daily_seq']
        
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

    def add_batch(
        self,
        obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        cost: np.ndarray,
        next_obs: Dict[str, np.ndarray],
        done: np.ndarray
    ):
        """
        Batch add transitions.
        Obs values are expected to be (batch_size, ...)
        """
        batch_size = len(action)
        end_idx = self.ptr + batch_size
        
        if end_idx <= self.capacity:
            indices = np.arange(self.ptr, end_idx)
            self._store_batch(indices, obs, action, reward, cost, next_obs, done)
            self.ptr = end_idx % self.capacity
        else:
            # Split into two parts (Wrap around)
            first_part = self.capacity - self.ptr
            indices_1 = np.arange(self.ptr, self.capacity)
            self._store_batch(indices_1, obs, action, reward, cost, next_obs, done, 0, first_part)
            
            second_part = batch_size - first_part
            indices_2 = np.arange(0, second_part)
            self._store_batch(indices_2, obs, action, reward, cost, next_obs, done, first_part, batch_size)
            
            self.ptr = second_part
            
        self.size = min(self.size + batch_size, self.capacity)

    def _store_batch(self, indices, obs, action, reward, cost, next_obs, done, start_slice=0, end_slice=None):
        # Slicing helper
        sl = slice(start_slice, end_slice) if end_slice is not None else slice(start_slice, None)
        
        self.price_seqs[indices] = obs['price_seq'][sl]
        if self.daily_seqs is not None and 'daily_seq' in obs:
            self.daily_seqs[indices] = obs['daily_seq'][sl]
        
        # Batch Concatenate States
        # Axis 1 because 0 is batch dimension
        self.state_vecs[indices] = np.concatenate([
            obs['account_state'][sl],
            obs['time_state'][sl],
            obs['rhythm_state'][sl],
            obs['cost_state'][sl],
            obs['market_state'][sl]
        ], axis=1)
        
        self.actions[indices] = action[sl]
        # Reshape reward/done to (B, 1) if needed
        self.rewards[indices] = reward[sl].reshape(-1, 1)
        self.costs[indices] = cost[sl] # Assuming cost is already (B, cost_dim)
        
        self.next_price_seqs[indices] = next_obs['price_seq'][sl]
        if self.next_daily_seqs is not None and 'daily_seq' in next_obs:
            self.next_daily_seqs[indices] = next_obs['daily_seq'][sl]
        
        self.next_state_vecs[indices] = np.concatenate([
            next_obs['account_state'][sl],
            next_obs['time_state'][sl],
            next_obs['rhythm_state'][sl],
            next_obs['cost_state'][sl],
            next_obs['market_state'][sl]
        ], axis=1)
        
        self.dones[indices] = done[sl].reshape(-1, 1)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        indices = np.random.randint(0, self.size, size=batch_size)
        
        batch = {
            'price_seq': torch.FloatTensor(self.price_seqs[indices]).to(self.device),
            'state_vec': torch.FloatTensor(self.state_vecs[indices]).to(self.device),
            'action': torch.FloatTensor(self.actions[indices]).to(self.device),
            'reward': torch.FloatTensor(self.rewards[indices]).to(self.device),
            'cost': torch.FloatTensor(self.costs[indices]).to(self.device),
            'next_price_seq': torch.FloatTensor(self.next_price_seqs[indices]).to(self.device),
            'next_state_vec': torch.FloatTensor(self.next_state_vecs[indices]).to(self.device),
            'done': torch.FloatTensor(self.dones[indices]).to(self.device)
        }
        
        if self.daily_seqs is not None:
            batch['daily_seq'] = torch.FloatTensor(self.daily_seqs[indices]).to(self.device)
            batch['next_daily_seq'] = torch.FloatTensor(self.next_daily_seqs[indices]).to(self.device)
            
        return batch
