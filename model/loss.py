from re import X
from turtle import forward
from typing import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel) -> None:
        super(DoubleConv, self).__init__()
        self.F = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.F(x)


class UNet(nn.Module):
    def __init__(self, n_channels) -> None:
        super(UNet, self).__init__()
        self.n_channels = n_channels
        
        self.conv1 = DoubleConv(n_channels, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(64, 128, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(128, 256, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(256, 512, 3, padding=1)
        self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6 = DoubleConv(512, 256, 3, padding=1)
        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7 = DoubleConv(256, 128, 3, padding=1)
        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8 = DoubleConv(128, 64, 3, padding=1)
        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9 = DoubleConv(64, 32, 3, padding=1)
        self.up10 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.conv10 = DoubleConv(32, 1, 3, padding=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.pool4(x)
        x = self.conv5(x)
        x = self.up6(x)
        x = self.conv6(x)
        x = self.up7(x)
        x = self.conv7(x)
        x = self.up8(x)
        x = self.conv8(x)
        x = self.up9(x)
        x = self.conv9(x)
        x = self.up10(x)
        y = self.conv10(x)
        return y
        
        