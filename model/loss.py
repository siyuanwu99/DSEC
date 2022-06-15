import numpy as np
# from import_proj_matl import extract_projmat
import torch
import torch.nn.functional as F
import torch
import model.pytorch_ssim

def get_projectmat():
    Q_np = np.array(
        [
            [1.0000, 0.0000, 0.0000, -336.8341],
            [0.0000, 1.0000, 0.0000, -220.9113],
            [0.0000, 0.0000, 0.0000, 583.3081],
            [0.0000, 0.0000, 1.6688, -0.0000],
        ]
    )
    Q = torch.tensor(Q_np)
    return Q


def photometric_loss_l1(input, target, weight=None):
    """
    photometric loss
    """
    if weight is None:
        weight = torch.ones_like(input)
    return torch.mean(weight * torch.abs(input - target))


def loss(output, target):
    # Q=extract_projmat(path='cam_to_cam.yaml')
    Q = get_projectmat()
    if torch.cuda.is_available() and Q.device.type == "cpu":
        Q = Q.cuda()
    # output = output.reshape(target.shape)
    valid_idx = target != 0
    valid_num = torch.count_nonzero(valid_idx)
    depth_target = Q[2, 3] / ((target - Q[3, 3])*Q[3,2])
    log_depth_target = get_log_depth_gt(depth_target, valid_idx)
    depth_output = output.reshape(log_depth_target.shape)

    target_s1 = target[:, ::2, ::2]
    valid_idx_s1 = target_s1 != 0
    valid_num_s1 = torch.count_nonzero(valid_idx_s1)
    output_s1 = depth_output[:, ::2, ::2]
    depth_target_s1 = Q[2, 3] / ((target_s1 - Q[3, 3])*Q[3,2])

    target_s2 = target_s1[:, ::2, ::2]
    valid_idx_s2 = target_s2 != 0
    valid_num_s2 = torch.count_nonzero(valid_idx_s2)
    output_s2 = output_s1[:, ::2, ::2]
    depth_target_s2 = Q[2, 3] / ((target_s2 - Q[3, 3])*Q[3,2])

    target_s3 = target_s2[:, ::2, ::2]
    valid_idx_s3 = target_s3 != 0
    valid_num_s3 = torch.count_nonzero(valid_idx_s3)
    output_s3 = output_s2[:, ::2, ::2]
    depth_target_s3 = Q[2, 3] / ((target_s3 - Q[3, 3])*Q[3,2])

    if torch.cuda.is_available():
        R_k = torch.zeros(target.shape).cuda()
        ssim_target = torch.zeros(target.shape).cuda()
        ssim_output = torch.zeros(target.shape).cuda()
    else:
        R_k = torch.zeros(target.shape)
        ssim_target = torch.zeros(target.shape)
        ssim_output = torch.zeros(target.shape)
    ssim_target[valid_idx] = log_depth_target[valid_idx]
    ssim_output[valid_idx] = depth_output[valid_idx]
    ssim_val = get_ssim_loss(ssim_target.unsqueeze(0),ssim_output.unsqueeze(0))
    R_k[valid_idx] = log_depth_target[valid_idx] - depth_output[valid_idx]
    loss_invar = ((1 / valid_num) * torch.sum(R_k ** 2)) - (
        ((1 / valid_num) ** 2) * (torch.sum(R_k) ** 2)
    )
    grad_loss = multi_grad_loss(depth_output, depth_target, valid_idx, valid_num)
    grad_loss_s1 = multi_grad_loss(output_s1, depth_target_s1, valid_idx_s1, valid_num_s1)
    grad_loss_s2 = multi_grad_loss(output_s2, depth_target_s2, valid_idx_s2, valid_num_s2)
    grad_loss_s3 = multi_grad_loss(output_s3, depth_target_s3, valid_idx_s3, valid_num_s3)
    # print(Q.device)
    # print(output.device)
    # print(valid_idx.device)
    # print(valid_num.device)
    # print(depth_output.device)
    # print(depth_target.device)
    # print(R_k.device)
    # print(loss_val.device)
    loss_val = loss_invar + 0.5 * (grad_loss + grad_loss_s1 + grad_loss_s2 + grad_loss_s3) + 0.03*ssim_val
    return loss_val


def get_log_depth_gt(depth_target, valid_idx, Dmax=80, alpha=3.7):
    invalid_idx=~valid_idx
    depth_target1 = torch.clamp(depth_target,0,80)
    depth_target1[invalid_idx] = 0
    depth_target1 = depth_target1/torch.amax(torch.amax(depth_target1,1,keepdims=True),2,keepdims=True)
    depth_target1 *= 80
    if torch.cuda.is_available():
        log_depth_target = torch.zeros(depth_target1.shape).cuda()
        log_depth_target[valid_idx] = (torch.log((depth_target1[valid_idx] / Dmax)) / alpha) + 1
    else:
        log_depth_target = torch.zeros(depth_target1.shape)
        log_depth_target[valid_idx] = (torch.log((depth_target1[valid_idx] / Dmax)) / alpha) + 1
    return log_depth_target


def multi_grad_loss(log_depth_output, depth_target, valid_idx, valid_num):
    log_depth_target = get_log_depth_gt(depth_target, valid_idx)
    log_d_diff = log_depth_output - log_depth_target
    log_d_diff = torch.mul(log_d_diff, valid_idx)
    # print(valid_num)
    v_gradient = torch.abs(log_d_diff[:, 0:-2, :] - log_d_diff[:, 2:, :])
    v_mask = torch.mul(valid_idx[:, 0:-2, :], valid_idx[:, 2:, :])
    v_gradient = torch.mul(v_gradient, v_mask)

    h_gradient = torch.abs(log_d_diff[:, :, 0:-2] - log_d_diff[:, :, 2:])
    h_mask = torch.mul(valid_idx[:, :, 0:-2], valid_idx[:, :, 2:])
    h_gradient = torch.mul(h_gradient, h_mask)

    gradient_loss = torch.sum(h_gradient) + torch.sum(v_gradient)
    grad_loss = gradient_loss / valid_num
    return grad_loss

def from_log_to_depth(input_log_image,Dmax=80, alpha=3.7):
    depth_image=Dmax*torch.exp(-alpha*(1-input_log_image))
    return depth_image
def get_ssim_loss(img1,img2):
    ssim_loss = model.pytorch_ssim.SSIM()
    ssim_val = 1-ssim_loss(img1, img2)
    return ssim_val