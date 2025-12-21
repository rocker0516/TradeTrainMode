
import numpy as np
import torch

def check_tensor_for_nan(tensor, name="Tensor"):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
        return True
    if torch.isinf(tensor).any():
        print(f"Inf detected in {name}")
        return True
    return False

