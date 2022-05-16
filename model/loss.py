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

import numpy as np
import cv2
def loss(output, target):
    # Q=np.array([[1,0,0,-336.83414459228516],[0,1,0,-220.91131019592285],[0,0,0,583.3081203392971],[0,0,1.6687860862434196,0]])
    # Image_target=	cv2.reprojectImageTo3D(target, Q)
    # Image_output=	cv2.reprojectImageTo3D(output, Q)
    # output=np.array()
    output=output.reshape(target.shape)
    valid_idx=target!=0
    valid_num=torch.count_nonzero(valid_idx)
    depth_output=1/output
    depth_target=1/target
    R_k=torch.zeros(target.shape)
    R_k[valid_idx]=depth_target[valid_idx]-depth_output[valid_idx]
    loss_val=(1/valid_num)*torch.sum(R_k**2)-(1/valid_num)**2*(torch.sum(R_k)**2)
    return loss_val