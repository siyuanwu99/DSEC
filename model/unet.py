from stringprep import c7_set
from turtle import forward
from matplotlib.pyplot import axis
import torch
import torch.nn as nn
import torch.nn.functional as F

class DebugNet(nn.Module):
    def __init__(self, n_channels) -> None:
        super(DebugNet, self).__init__()
        self.F = nn.Sequential(
            nn.Conv2d(n_channels, 1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        y = self.F(x)
        return torch.sum(y, axis=1)

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
        
        self.conv1 = DoubleConv(n_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(256, 512)
        self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6 = DoubleConv(512, 256)
        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7 = DoubleConv(256, 128)
        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8 = DoubleConv(128, 64)
        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9 = DoubleConv(64, 32)
        self.conv10 = nn.Conv2d(32, 1, 1)
        
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
        x = self.up7(x)
        x = self.up8(x)
        x = self.up9(x)
        y = self.conv10(x)
        return y

class UNetConnect(UNet):
    # def __init__(self, n_channels) -> None:
    #     super(UNetConnect, self).__init__(n_channels)
        
    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(self.pool1(c1))
        c3 = self.conv3(self.pool2(c2))
        c4 = self.conv4(self.pool3(c3))
        c5 = self.conv5(self.pool4(c4))
        u6 = self.up6(c5)
        c6 = self.conv6(torch.cat([u6, c4], dim=1))
        u7 = self.up7(c6)
        c7 = self.conv7(torch.cat([u7, c3], dim=1))
        u8 = self.up8(c7)
        c8 = self.conv8(torch.cat([u8, c2], dim=1))
        u9 = self.up9(c8)
        c9 = self.conv9(torch.cat([u9, c1], dim=1))
        c10 = self.conv10(c9)
        y = nn.Sigmoid()(c10)
        return y
        
        