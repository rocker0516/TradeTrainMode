
import torch
import torch.nn as nn
import unittest
from Train.architectures import Actor

class TestActorNaN(unittest.TestCase):
    def test_actor_nan_propagation(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        actor = Actor(
            price_input_channels=6,
            price_window_size=288,
            state_dim=20,
            action_dim=1
        ).to(device)
        
        # 1. Normal Input
        batch_size = 256
        price = torch.randn(batch_size, 288, 6).to(device) # (B, W, C) -> Permuted inside
        state = torch.randn(batch_size, 20).to(device)
        
        mean, log_std = actor(price, state)
        
        if torch.isnan(mean).any():
             print("NaN in Mean with Normal Input")
        if torch.isnan(log_std).any():
             print("NaN in LogStd with Normal Input")
             
        self.assertFalse(torch.isnan(mean).any())
        
        # 2. Input with 0s
        price_z = torch.zeros(batch_size, 288, 6).to(device)
        state_z = torch.zeros(batch_size, 20).to(device)
        mean, log_std = actor(price_z, state_z)
        self.assertFalse(torch.isnan(mean).any())
        
        # 3. Input with Large Values (but not inf)
        price_l = torch.randn(batch_size, 288, 6).to(device) * 100
        state_l = torch.randn(batch_size, 20).to(device) * 100
        mean, log_std = actor(price_l, state_l)
        if torch.isnan(mean).any():
             print("NaN detected with Large Input")
        self.assertFalse(torch.isnan(mean).any())
        
if __name__ == '__main__':
    unittest.main()

