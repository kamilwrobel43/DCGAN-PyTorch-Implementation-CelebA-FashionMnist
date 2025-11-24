import numpy as np
import torch
import random

def generate_noise(batch_size : int = 32, device: torch.device = torch.device('cuda')):
    z = torch.randn(batch_size,100, device=device)
    return z

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
