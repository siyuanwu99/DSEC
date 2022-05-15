import torch
import math

def lg10(x):
    return torch.div(torch.log(x), math.log(10))

def maxOfTwo(x, y):
    z = x.clone()
    maskYLarger = torch.lt(x, y)
    z[maskYLarger.detach()] = y[maskYLarger.detach()]
    return z


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def mean_square_error(output, target):
    with torch.no_grad():
        diffMatrix = torch.abs(output - target)
        return torch.sum(torch.pow(diffMatrix, 2)) / diffMatrix.numel()


def mean_absolute_error(output, target):
    with torch.no_grad():
        diffMatrix = torch.abs(output - target)
        return torch.sum(diffMatrix) / diffMatrix.numel()


def abs_rel_error(output, target):
    with torch.no_grad():
        diffMatrix = torch.abs(output - target)
        realMatrix = torch.div(diffMatrix, target)
        return torch.sum(realMatrix) / realMatrix.numel()

def lg10_error(output, target):
    with torch.no_grad():
        LG10Matrix = torch.abs(lg10(output) - lg10(target))
        return torch.sum(LG10Matrix) / LG10Matrix.numel()
    
def delta1_error(output, target):
    with torch.no_grad():
        yOverZ = torch.div(output, target)
        zOverY = torch.div(target, output)
        maxRatio = maxOfTwo(yOverZ, zOverY)

        return torch.sum(torch.le(maxRatio, 1.25).float()) / maxRatio.numel()

def delta2_error(output, target):
    with torch.no_grad():
        yOverZ = torch.div(output, target)
        zOverY = torch.div(target, output)
        maxRatio = maxOfTwo(yOverZ, zOverY)

        return torch.sum(torch.le(maxRatio, math.pow(1.25, 2)).float()) / maxRatio.numel()


def delta3_error(output, target):
    with torch.no_grad():
        yOverZ = torch.div(output, target)
        zOverY = torch.div(target, output)
        maxRatio = maxOfTwo(yOverZ, zOverY)

        return torch.sum(torch.le(maxRatio, math.pow(1.25, 3)).float()) / maxRatio.numel()
