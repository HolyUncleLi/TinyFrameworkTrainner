import torch
from torch import nn
import torch.nn.functional as f
import math
from models.CBAM import CBAM1d,CBAM2d

'''
2D大核卷积块
'''

class SEBlock(nn.Module):
    def __init__(self, in_dim, reduction=16):
        super().__init__()
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_dim, in_dim // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_dim // reduction, in_dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        weights = self.layers(x)
        weights = weights.unsqueeze(-1).unsqueeze(-1)
        return x * weights.expand_as(x)


class LayerNorm(nn.Module):

    def __init__(self, channels, eps=1e-6, data_format="channels_last"):
        super(LayerNorm, self).__init__()
        self.norm = nn.Layernorm(channels)

    def forward(self, x):

        B, M, D, N = x.shape
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(B * M, N, D)
        x = self.norm(x)
        x = x.reshape(B, M, N, D)
        x = x.permute(0, 1, 3, 2)
        return x


def get_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=bias)


def get_bn(channels):
    return nn.BatchNorm2d(channels)


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1,bias=False,isFTConv=True):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module('conv', get_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))
    result.add_module('bn', get_bn(out_channels))
    return result


class ReparamLargeKernelConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, groups,
                 small_kernel=5,
                 nvars=1):
        super(ReparamLargeKernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel

        padding = kernel_size // 2

        self.lkb_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                    stride=stride, padding=padding, dilation=1, groups=groups,bias=False)

        self.small_conv = conv_bn(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=small_kernel,
                                    stride=stride, padding=small_kernel // 2, groups=groups, dilation=1,bias=False)

    def forward(self, inputs):

        out = self.lkb_origin(inputs)
        out += self.small_conv(inputs)
        return out


class Block2(nn.Module):
    def __init__(self, large_size, in_dim, out_dim, dff, drop=0.05):

        super(Block2, self).__init__()

        self.dw = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=large_size, stride=1, padding=large_size//2, groups=in_dim)
        # self.dw = ReparamLargeKernelConv(in_channels=in_dim, out_channels=out_dim, kernel_size=large_size, stride=1, groups=in_dim)
        self.norm = nn.BatchNorm2d(out_dim)
        self.cbam = CBAM2d(channel=out_dim, reduction=16, kernel_size=7)

        #convffn1
        self.ffn1pw1 = nn.Conv2d(in_channels=out_dim, out_channels=dff, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=1)
        self.ffn1act = nn.GELU()
        self.ffn1pw2 = nn.Conv2d(in_channels=dff, out_channels=out_dim, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=1)
        self.ffn1drop1 = nn.Dropout(drop)
        self.ffn1drop2 = nn.Dropout(drop)

        #convffn2
        self.ffn2pw1 = nn.Conv2d(in_channels=out_dim, out_channels=dff, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=out_dim)
        self.ffn2act = nn.GELU()
        self.ffn2pw2 = nn.Conv2d(in_channels=dff, out_channels=out_dim, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=out_dim)
        self.ffn2drop1 = nn.Dropout(drop)
        self.ffn2drop2 = nn.Dropout(drop)

    def forward(self, x):

        x = self.dw(x)
        x = self.norm(x)
        x = self.cbam(x)

        x = self.ffn1drop1(self.ffn1pw1(x))
        x = self.ffn1act(x)
        x = self.ffn1drop2(self.ffn1pw2(x))

        return x


class Stage2(nn.Module):
    def __init__(self, ffn_ratio, large_size, dmodels, drop=0.1):
        super(Stage2, self).__init__()
        self.num_blocks = len(dmodels)
        self.stem = nn.Sequential(
            nn.Conv2d(128, dmodels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dmodels[0])
        )
        blks = []
        for i in range(self.num_blocks):
            blk = Block2(large_size=large_size[i], in_dim=dmodels[i], out_dim=dmodels[i], dff=dmodels[i]*ffn_ratio, drop=drop)
            blks.append(blk)
            if i <(self.num_blocks - 1):
                 blks.append(nn.Sequential(
                     nn.Conv2d(dmodels[i], dmodels[i+1], kernel_size=3, stride=1, padding=1),
                     nn.BatchNorm2d(dmodels[i+1])
                 ))
        self.blocks = nn.ModuleList(blks)

    def forward(self, x):
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(x)
        return x

