import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.FTConv_2 import *

class FTReConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size1=51, kernel_size2=5,kernel_size3=5, stride=1, groups=1, padding=0):
        super(FTReConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size1 = kernel_size1
        self.kernel_size2 = kernel_size2
        self.kernel_size3 = kernel_size3
        self.groups = groups
        self.identity = nn.Identity()
        self.relu = nn.ReLU()

        self.embed_layer = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=8,
            stride=4,
            padding=3
        )

        self.rep_conv = nn.Conv1d(self.in_channels, self.out_channels, kernel_size1, stride=stride, padding=kernel_size1//2,groups=self.groups)

        self.rbr_identity = nn.BatchNorm1d(num_features=self.in_channels) if self.in_channels == self.out_channels and stride == 1 else None
        self.conv_large = nn.Conv1d(self.in_channels, self.out_channels, kernel_size1, stride, padding=kernel_size1//2,groups=self.groups)
        self.conv_small = nn.Conv1d(self.in_channels, self.out_channels, kernel_size2, stride, padding=kernel_size2//2,groups=self.groups)
        self.ftconv = FTConv1d(1, self.out_channels, kernel_size3, padding=kernel_size3//2,stride=stride,featureDim=3000 + kernel_size3//2 * 2)

    def forward(self, x, signal=None, deploy=False):
        if deploy == True:

            self.rep_conv.weight.data = self.conv_large.weight.data + F.pad(self.conv_small.weight.data, [
                (self.kernel_size1 - self.kernel_size2) // 2, (self.kernel_size1 - self.kernel_size2) // 2])
            self.rep_conv.bias.data = self.conv_large.bias.data + self.conv_small.bias.data
            return self.relu(self.identity(x) + self.rep_conv(x) + self.embed_layer(self.ftconv(signal)))

        else:
            out1 = self.conv_large(x)
            out2 = self.conv_small(x)
            out3 = self.embed_layer(self.ftconv(signal))
            out4 = self.identity(x)
            return self.relu(out1 + out2 + out3 + out4)


class ReConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size1=51, kernel_size2=5, stride=1, groups=1, padding=0):
        super(ReConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size1 = kernel_size1
        self.kernel_size2 = kernel_size2
        self.groups = groups
        self.identity = nn.Identity()
        self.relu = nn.ReLU()

        self.rep_conv = nn.Conv1d(self.in_channels, self.out_channels, kernel_size1, stride=stride, padding=kernel_size1//2,groups=self.groups)

        self.rbr_identity = nn.BatchNorm1d(num_features=self.in_channels) if self.in_channels == self.out_channels and stride == 1 else None
        self.conv_large = nn.Conv1d(self.in_channels, self.out_channels, kernel_size1, stride, padding=kernel_size1//2,groups=self.groups)
        self.conv_small = nn.Conv1d(self.in_channels, self.out_channels, kernel_size2, stride, padding=kernel_size2//2,groups=self.groups)

    def forward(self, x, signal=None, deploy=False):

        # 推理
        if deploy == True:

            self.rep_conv.weight.data = self.conv_large.weight.data + F.pad(self.conv_small.weight.data, [
                (self.kernel_size1 - self.kernel_size2) // 2, (self.kernel_size1 - self.kernel_size2) // 2])
            self.rep_conv.bias.data = self.conv_large.bias.data + self.conv_small.bias.data

            return self.relu(self.identity(x) + self.rep_conv(x))
        else:
            # 训练
            out1 = self.conv_large(x)
            out2 = self.conv_small(x)
            out3 = self.identity(x)
            return self.relu(out1 + out2 + out3)
'''
model = ReConvBlock(128, 128)
x = torch.randn(64, 128, 750)
s = torch.randn(64, 128, 3000)
print(model(x,s,deploy=False).shape)
print(model(x,s,deploy=True).shape)
'''
