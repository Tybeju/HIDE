import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, strides=1):
        super(ConvBlock, self).__init__()
        self.strides = strides
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channel, out_channel, kernel_size=3, stride=strides, padding=1
            ),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(
                out_channel, out_channel, kernel_size=3, stride=strides, padding=1
            ),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(inplace=True),
        )
        self.conv11 = nn.Conv2d(
            in_channel, out_channel, kernel_size=1, stride=strides, padding=0
        )

    def forward(self, x):
        out1 = self.block(x)
        out2 = self.conv11(x)
        out = out1 + out2
        return out


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class AttentionBlock(nn.Module):
    """Attention block with learnable parameters"""

    def __init__(self, F_g, F_l, n_coefficients):
        """
        :param F_g: number of feature maps (channels) in previous layer
        :param F_l: number of feature maps in corresponding encoder layer, transferred via skip connection
        :param n_coefficients: number of learnable multi-dimensional attention coefficients
        """
        super(AttentionBlock, self).__init__()

        self.W_gate = nn.Sequential(
            nn.Conv2d(
                F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True
            ),
            nn.BatchNorm2d(n_coefficients),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(
                F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True
            ),
            nn.BatchNorm2d(n_coefficients),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):
        """
        :param gate: gating signal from previous layer
        :param skip_connection: activation from corresponding encoder layer
        :return: output activations
        """
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = skip_connection * psi
        return out


class AttentionUNet(nn.Module):
    def __init__(self, block=ConvBlock, dim=64):
        super(AttentionUNet, self).__init__()

        self.dim = dim
        self.ConvBlock1 = ConvBlock(3, dim, strides=1)
        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.pool1 = nn.Conv2d(dim, dim, kernel_size=4, stride=2, padding=1)

        self.ConvBlock2 = block(dim, dim * 2, strides=1)
        self.pool2 = nn.Conv2d(dim * 2, dim * 2, kernel_size=4, stride=2, padding=1)

        self.ConvBlock3 = block(dim * 2, dim * 4, strides=1)
        self.pool3 = nn.Conv2d(dim * 4, dim * 4, kernel_size=4, stride=2, padding=1)

        self.ConvBlock4 = block(dim * 4, dim * 8, strides=1)
        self.pool4 = nn.Conv2d(dim * 8, dim * 8, kernel_size=4, stride=2, padding=1)

        self.ConvBlock5 = block(dim * 8, dim * 16, strides=1)

        self.upv6 = UpConv(1024, 512)
        self.ConvBlock6 = block(dim * 16, dim * 8, strides=1)

        self.upv7 = UpConv(512, 256)
        self.ConvBlock7 = block(dim * 8, dim * 4, strides=1)

        self.upv8 = UpConv(256, 128)
        self.ConvBlock8 = block(dim * 4, dim * 2, strides=1)

        self.upv9 = UpConv(128, 64)
        self.ConvBlock9 = block(dim * 2, dim, strides=1)

        self.conv10 = nn.Conv2d(dim, 3, kernel_size=3, stride=1, padding=1)

        self.Att1 = AttentionBlock(F_g=512, F_l=512, n_coefficients=256)
        self.Att2 = AttentionBlock(F_g=256, F_l=256, n_coefficients=128)
        self.Att3 = AttentionBlock(F_g=128, F_l=128, n_coefficients=64)
        self.Att4 = AttentionBlock(F_g=64, F_l=64, n_coefficients=32)

    def forward(self, x):
        conv1 = self.ConvBlock1(x)
        pool1 = self.MaxPool(conv1)

        conv2 = self.ConvBlock2(pool1)
        pool2 = self.MaxPool(conv2)

        conv3 = self.ConvBlock3(pool2)
        pool3 = self.MaxPool(conv3)

        conv4 = self.ConvBlock4(pool3)
        pool4 = self.MaxPool(conv4)

        conv5 = self.ConvBlock5(pool4)

        up6 = self.upv6(conv5)
        conv4 = self.Att1(gate=up6, skip_connection=conv4)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.ConvBlock6(up6)

        up7 = self.upv7(conv6)
        conv3 = self.Att2(gate=up7, skip_connection=conv3)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.ConvBlock7(up7)

        up8 = self.upv8(conv7)
        conv2 = self.Att3(gate=up8, skip_connection=conv2)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.ConvBlock8(up8)

        up9 = self.upv9(conv8)
        conv1 = self.Att4(gate=up9, skip_connection=conv1)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.ConvBlock9(up9)

        conv10 = self.conv10(conv9)
        out = x + conv10

        return out
