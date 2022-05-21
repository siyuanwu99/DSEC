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
            nn.Sigmoid(),
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
            nn.ReLU(inplace=True),
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


class ConvLSTM(nn.Module):
    """
    Based on https://github.com/ndrplz/ConvLSTM_pytorch
    """

    def __init__(self, input_channels, output_channels, kernel_size) -> None:
        super(ConvLSTM, self).__init__()

        self.in_ch = input_channels
        self.out_ch = output_channels
        self.padding = kernel_size // 2
        self.zero_tensors = {}
        self.Gates = nn.Conv2d(
            input_channels + output_channels, 4 * output_channels, kernel_size, padding=self.padding, stride=1
        )

    def forward(self, x, z_prev=None):
        batch_size = x.data.size()[0]
        spatial_size = x.data.size()[2:]

        if z_prev is None:
            state_size = tuple([batch_size, self.out_ch] + list(spatial_size))
            if state_size not in self.zero_tensors:
                self.zero_tensors[state_size] = (
                    torch.zeros(state_size).to(x.device),
                    torch.zeros(state_size).to(x.device),
                )
            z_prev = self.zero_tensors[state_size]

        x_hidden, x_cell = z_prev

        x = torch.cat((x, x_hidden), dim=1)
        gates = self.Gates(x)
        gate_in, gate_forget, gate_out, gate_cell = gates.chunk(4, dim=1)

        gate_in = torch.sigmoid(gate_in)
        gate_forget = torch.sigmoid(gate_forget)
        gate_out = torch.sigmoid(gate_out)
        gate_cell = torch.tanh(gate_cell)

        y_cell = gate_forget * x_cell + gate_in * gate_cell
        y_hidden = gate_out * torch.tanh(y_cell)
        return y_hidden, y_cell


class EncoderLayer(nn.Module):
    def __init__(self, in_chn, out_chn) -> None:
        super(EncoderLayer, self).__init__()
        self.in_chn = in_chn
        self.out_chn = out_chn
        self.conv = nn.Conv2d(in_chn, out_chn, kernel_size=5, stride=2, padding=1)
        self.conv_lstm = ConvLSTM(out_chn, out_chn, kernel_size=3)

    def forward(self, x, z_prev=None):
        x = self.conv(x)
        z = self.conv_lstm(x, z_prev)
        y = z[0]
        return y, z


class ResidualLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, down_sample=None):
        super(ResidualLayer, self).__init__()
        self.down_sample = down_sample
        self.F = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        residual = x
        out = self.F(x)

        if self.down_sample:
            residual = self.down_sample(x)
        out += residual
        out = nn.ReLU()(out)
        return out


class MonoDepthNet(nn.Module):
    def __init__(self, n_channels) -> None:
        super(MonoDepthNet, self).__init__()
        Nb = 32
        Ne = 3
        self.Ne = Ne
        Nr = 2

        # Header
        self.H = nn.Sequential(nn.Conv2d(n_channels, Nb, 3), nn.ReLU(inplace=True))

        # Prediction layer
        self.P = nn.Sequential(nn.Conv2d(Nb, 1, 3), nn.Sigmoid())

        # Encoders
        self.E = nn.ModuleList()
        for i in range(Ne):
            self.E.append(EncoderLayer(Nb * (2 ** i), Nb * (2 ** (i + 1))))

        Cr = Nb * (2 ** Ne)

        # Residual blocks
        self.R = nn.ModuleList()
        for i in range(Nr):
            self.R.append(ResidualLayer(Cr, Cr))

        # Decoders
        self.D = nn.ModuleList()
        for i in range(Ne):
            j = Ne - i
            self.D.append(
                nn.Sequential(
                    # nn.ConvTranspose2d(Nb * (2 ** j), Nb * (2 ** (j - 1)), kernel_size=3, stride=2, padding=1),
                    nn.UpsamplingBilinear2d(scale_factor=2),
                    nn.Conv2d(Nb * (2 ** j), Nb * (2 ** (j - 1)), kernel_size=3, padding=2, stride=1),
                    nn.BatchNorm2d(Nb * (2 ** (j - 1))),
                    nn.ReLU()
                )
            )

    def forward(self, x, z):
        x = self.H(x)
        head = x.clone()

        if z is None:
            z = [None] * self.Ne
        print(x.shape)
        blocks = []
        states = []
        for i, encoder in enumerate(self.E):
            x, z_ = encoder(x, z[i])
            print("Encoder {}: {}".format(i, x.shape))
            blocks.append(x)
            states.append(z_)
        
        for i, residual in enumerate(self.R):
            print("Residual {}: {}".format(i, x.shape))
            x = residual(x)

        for i, decoder in enumerate(self.D):
            print("Decoder {}: {}".format(i, x.shape))
            x = decoder(x + blocks[self.Ne - i - 1])

        x = self.P(x + head)
        return x, states
