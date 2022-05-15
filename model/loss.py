import imp


import torch
import torch.nn.functional as F

def photometric_loss_l1(input, target, weight=None):
    """
    photometric loss
    """
    if weight is None:
        weight = torch.ones_like(input)
    return torch.mean(weight * torch.abs(input - target))