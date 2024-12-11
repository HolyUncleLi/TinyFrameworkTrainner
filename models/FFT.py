import torch
import torch.nn.functional as F
import numpy as np


def FFT0(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = torch.repeat_interleave(top_list.unsqueeze(0), x.shape[0], dim=0)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


def FFT1(x):
    # [B, T * C]
    xf = torch.fft.rfft(x.view(x.shape[0], -1), dim=1)
    frequency_list = abs(xf)

    signal_len = x.view(x.shape[0], -1).shape[1]
    frequence_resolution = 100 / signal_len

    top_list = torch.ones([x.shape[0],1])
    weight = torch.ones([x.shape[0],1])

    top_list[:, 0] = torch.topk(frequency_list[:, 1:int(32/frequence_resolution)], 1)[1].squeeze() + 1
    top_list = top_list.detach().cpu().numpy()

    period = x.view(x.shape[0], -1).shape[1] // top_list

    weight[:, 0] = torch.mean(frequency_list[:, 1:int(32/frequence_resolution)], dim=1)

    return period, F.softmax(weight, dim=1)


def FFT2(x):
    # [B, T * C]
    xf = torch.fft.rfft(x.view(x.shape[0], -1), dim=1)
    frequency_list = abs(xf)

    signal_len = x.view(x.shape[0], -1).shape[1]
    frequence_resolution = 100 / signal_len

    top_list = torch.ones([x.shape[0],5])
    weight = torch.ones([x.shape[0],5])

    top_list[:, 0] = torch.topk(frequency_list[:, 1:int(4/frequence_resolution)], 1)[1].squeeze() + 1
    top_list[:, 1] = torch.topk(frequency_list[:, int(4/frequence_resolution):int(8/frequence_resolution)], 1)[1].squeeze() + int(4/frequence_resolution)
    top_list[:, 2] = torch.topk(frequency_list[:, int(8/frequence_resolution):int(12/frequence_resolution)], 1)[1].squeeze() + int(8/frequence_resolution)
    top_list[:, 3] = torch.topk(frequency_list[:, int(12/frequence_resolution):int(16/frequence_resolution)], 1)[1].squeeze() + int(12/frequence_resolution)
    top_list[:, 4] = torch.topk(frequency_list[:, int(16/frequence_resolution):int(32/frequence_resolution)], 1)[1].squeeze() + int(16/frequence_resolution)

    top_list = top_list.detach().cpu().numpy()
    # print(top_list[0])
    # print(torch.topk(frequency_list[:, 1:frequency_list.shape[1]], 5)[1])
    period = x.view(x.shape[0], -1).shape[1] // top_list

    weight[:, 0] = torch.mean(frequency_list[:, 1:int(4/frequence_resolution)], dim=1)
    weight[:, 1] = torch.mean(frequency_list[:, int(4/frequence_resolution):int(8/frequence_resolution)], dim=1)
    weight[:, 2] = torch.mean(frequency_list[:, int(8/frequence_resolution):int(12/frequence_resolution)], dim=1)
    weight[:, 3] = torch.mean(frequency_list[:, int(12/frequence_resolution):int(16/frequence_resolution)], dim=1)
    weight[:, 4] = torch.mean(frequency_list[:, int(16/frequence_resolution):int(32/frequence_resolution)], dim=1)
    # print(period[0])
    return period, F.softmax(weight, dim=1)


def FFT3(x):
    # [B, T * C]
    xf = torch.fft.rfft(x.view(x.shape[0], -1), dim=1)
    frequency_list = abs(xf)

    signal_len = x.view(x.shape[0], -1).shape[1]
    frequence_resolution = 100 / signal_len

    top_list = torch.ones([x.shape[0],5])
    weight = torch.ones([x.shape[0],5])

    top_list = torch.topk(frequency_list[:, 1:frequency_list.shape[1]], 5)[1]

    top_list = top_list.detach().cpu().numpy()

    period = x.view(x.shape[0], -1).shape[1] // top_list

    weight[:, 0] = torch.mean(frequency_list[:, 1:int(4/frequence_resolution)], dim=1)
    weight[:, 1] = torch.mean(frequency_list[:, int(4/frequence_resolution):int(8/frequence_resolution)], dim=1)
    weight[:, 2] = torch.mean(frequency_list[:, int(8/frequence_resolution):int(12/frequence_resolution)], dim=1)
    weight[:, 3] = torch.mean(frequency_list[:, int(12/frequence_resolution):int(16/frequence_resolution)], dim=1)
    weight[:, 4] = torch.mean(frequency_list[:, int(16/frequence_resolution):int(32/frequence_resolution)], dim=1)

    return period, F.softmax(weight, dim=1)


def FFT4(x):
    # [B, T * C]
    xf = torch.fft.rfft(x.view(x.shape[0],-1), dim=1)
    frequency_list = abs(xf)
    top_list = torch.ones([x.shape[0],5])
    weight = torch.ones([x.shape[0],5])
    top_list[:, 0] = 512 * 29
    top_list[:, 1] = 512 * 23
    top_list[:, 2] = 512 * 16
    top_list[:, 3] = 512 * 7
    top_list[:, 4] = 512 * 3

    top_list = top_list.detach().cpu().numpy()
    period = xf.shape[1] // top_list

    weight[:, 0] = torch.mean(frequency_list[:, 1:122], dim=1)
    weight[:, 1] = torch.mean(frequency_list[:, 122:244], dim=1)
    weight[:, 2] = torch.mean(frequency_list[:, 244:366], dim=1)
    weight[:, 3] = torch.mean(frequency_list[:, 366:488], dim=1)
    weight[:, 4] = torch.mean(frequency_list[:, 488:972], dim=1)

    return period, F.softmax(weight, dim=1)