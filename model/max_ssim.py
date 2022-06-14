import pytorch_ssim
import torch
from torch.autograd import Variable
from torch import optim
import cv2
import numpy as np

from ignite.engine import *
from ignite.handlers import *
from ignite.metrics import *
from ignite.utils import *
from ignite.contrib.metrics.regression import *
from ignite.contrib.metrics import *

npImg1 = cv2.imread("einstein.png")
def eval_step(engine, batch):
    return batch
img1 = torch.from_numpy(np.rollaxis(npImg1, 2)).float().unsqueeze(0)/255.0
img2 = torch.rand(img1.size())

if torch.cuda.is_available():
    img1 = img1.cuda()
    img2 = img2.cuda()


img1 = Variable( img1,  requires_grad=False)
img2 = Variable( img2, requires_grad = True)


# Functional: pytorch_ssim.ssim(img1, img2, window_size = 11, size_average = True)
ssim_value = pytorch_ssim.ssim(img1, img2)
print("Initial ssim:", ssim_value.item())

########################################################################################
default_evaluator = Engine(eval_step)
metric = SSIM(data_range=1.0)
metric.attach(default_evaluator, 'ssim')
preds = img1
target = img2
state = default_evaluator.run([[preds, target]])
print('initial ign ssim=',state.metrics['ssim'])
########################################################################################
# Module: pytorch_ssim.SSIM(window_size = 11, size_average = True)
ssim_loss = pytorch_ssim.SSIM()

optimizer = optim.Adam([img2], lr=0.01)

while ssim_value < 0.95:
    optimizer.zero_grad()
    ssim_out = -ssim_loss(img1, img2)
    ssim_value = - ssim_out.item()
    print("py_ssim=",ssim_value)
    state = default_evaluator.run([[img1, img2]])
    print('ign ssim=',state.metrics['ssim'])
    ssim_out.backward()
    optimizer.step()
