import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn.utils import weight_norm

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=2):
        super(ResBlock, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size

        self.conv = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, stride=stride,
                      padding=int((kernel_size - 1) / stride), bias=False),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(),
            nn.Conv1d(out_channel, out_channel, kernel_size=kernel_size, stride=stride,
                      padding=int((kernel_size - 1) / stride), bias=False),
            nn.BatchNorm1d(out_channel),
        )

        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, stride=stride,
                      padding=int((kernel_size - 1) / 2), bias=False),
            nn.BatchNorm1d(out_channel),
        )

    def forward(self, x):
        out = self.conv(x)
        out += self.shortcut(x)
        out = f.relu(out)
        return out


class softThresholds(nn.Module):
    def __init__(self):
        super(softThresholds, self).__init__()

    def abs(self, x):
        return torch.abs(x)

    def sign(self, x):
        x_sign = torch.div(x, abs(x))
        x_sign[torch.isnan(x_sign)] = 0
        return x_sign.clone().detach().requires_grad_(True)

    def forward(self, x, T):
        output = self.sign(x) * torch.max(torch.tensor(0), abs(x) - T)
        return output


class Attention(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Attention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.channel_atten = nn.Sequential(
            nn.Linear(out_channel, out_channel // 4),
            nn.BatchNorm1d(out_channel // 4),
            nn.ReLU(),
            nn.Linear(out_channel // 4, out_channel),
            nn.Sigmoid(),
        )
        self.weight1 = nn.Conv1d(out_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False)

        self.spatial_atten = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid(),
        )
        self.weight2 = nn.Conv1d(out_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False)

        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(out_channel)
        )
        self.softThresholds = softThresholds()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):  # x: [batchsize, 128, 4500]
        x0 = x  # x0: [batchsize, 256, 2250]
        x1 = torch.abs(x0)
        x11 = self.avg_pool(x1)  # x1: [batchsize, 256, 1]
        x11 = x11.reshape(x11.size(0), x11.size(1))

        x12 = self.channel_atten(x11)  # [batchsize, 256]
        x0_weight = self.weight1(x0)  # [batchsize, 256, 2250]
        T = x11 * x12  # T : [batchsize, 256]
        x1 = self.softThresholds(x0_weight, T.reshape(T.size(0), T.size(1), 1))
        x1 += x0  # [batchsize, 256, 2250]

        x2_avg = self.avg_pool(x1)  # [batchsize, 256, 1]
        x2_max = self.max_pool(x1)

        x2_avg = x2_avg.reshape(x2_avg.size(0), 1, x2_avg.size(1))  # [batchsize, 1, 256]
        x2_max = x2_max.reshape(x2_max.size(0), 1, x2_max.size(1))

        x2 = torch.cat((x2_avg, x2_max), 1)  # [batchsize, 2, 256]
        x2 = self.spatial_atten(x2)  # [batchsize, 1, 256]
        x2 = x2.reshape(x2.size(0), x2.size(-1), -1)  # [batchsize, 256, 1]
        x1_weight = self.weight2(x1)  # [batchsize, 256, 2250]
        x2 = x1_weight * x2.expand_as(x1)  # [batchsize, 256, 2250]
        return self.dropout(f.relu(self.shortcut(x) + x2))



class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):

        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):

        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # 经过conv1，输出的size其实是(Batch, input_channel, seq_len + padding)
        self.chomp1 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
        参数初始化
o
        :return:
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        :param x: size of (Batch, input_channel, seq_len)
        :return:
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, dilation_sizes, kernel_size=2, dropout=0.2):

        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = dilation_sizes[i]
            in_channels = num_inputs if i == 0 else num_channels[i - 1]  # 确定每一层的输入通道数
            out_channels = num_channels[i]  # 确定每一层的输出通道数
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):

        return self.network(x)
