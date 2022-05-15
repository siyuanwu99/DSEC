import numpy as np
# import cv2
def loss(output, target):
    valid_idx=target!=0
    valid_num=np.count_nonzero(valid_idx)
    depth_output=1/output
    depth_target=1/target
    R_k=np.zeros(target.shape)
    R_k[valid_idx]=depth_target[valid_idx]-depth_output[valid_idx]
    loss_val=(1/valid_num)*np.sum(R_k**2)-(1/valid_num)**2*(np.sum(R_k)**2)
    return loss_val
