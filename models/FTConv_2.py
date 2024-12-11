from functools import partial
from typing import Iterable, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as f
from torch import Tensor, nn
from torch.fft import irfftn, rfftn
from math import ceil, floor

'''

'''

def complex_matmul(a: Tensor, b: Tensor, groups: int = 1) -> Tensor:
    """Multiplies two complex-valued tensors."""
    # Scalar matrix multiplication of two tensors, over only the first channel
    # dimensions. Dimensions 3 and higher will have the same shape after multiplication.
    # We also allow for "grouped" multiplications, where multiple sections of channels
    # are multiplied independently of one another (required for group convolutions).
    a = a.view(a.size(0), groups, -1, *a.shape[2:])
    b = b.view(groups, -1, *b.shape[1:])

    a = torch.movedim(a, 2, a.dim() - 1).unsqueeze(-2)
    b = torch.movedim(b, (1, 2), (b.dim() - 1, b.dim() - 2))

    # complex value matrix multiplication
    real = a.real @ b.real - a.imag @ b.imag
    imag = a.imag @ b.real + a.real @ b.imag
    real = torch.movedim(real, real.dim() - 1, 2).squeeze(-1)
    imag = torch.movedim(imag, imag.dim() - 1, 2).squeeze(-1)
    c = torch.zeros(real.shape, dtype=torch.complex64, device=a.device)
    c.real, c.imag = real, imag

    return c.view(c.size(0), -1, *c.shape[3:])


def to_ntuple(val: Union[int, Iterable[int]], n: int) -> Tuple[int, ...]:
    if isinstance(val, Iterable):
        out = tuple(val)
        if len(out) == n:
            return out
        else:
            raise ValueError(f"Cannot cast tuple of length {len(out)} to length {n}.")
    else:
        return n * (val,)


def initMAT(dim, start=0,end=0):

    fourier_basis = np.fft.rfft(np.eye(dim))

    for i in range(0, dim//2 + 1):
        if i < start or i > end:
            fourier_basis[:, i] = 0

    return torch.tensor(fourier_basis).to(torch.complex64)


def fft_conv(
        signal: Tensor,
        kernel: Tensor,
        bias: Tensor = None,
        padding: Union[int, Iterable[int], str] = 0,
        padding_mode: str = "constant",
        stride: Union[int, Iterable[int]] = 1,
        dilation: Union[int, Iterable[int]] = 1,
        groups: int = 1,
        fourier_basis: Tensor = None,

        fourier_basis_1: Tensor = None,
        fourier_basis_2: Tensor = None,
        fourier_basis_3: Tensor = None,
        fourier_basis_4: Tensor = None,
        fourier_basis_5: Tensor = None,
) -> Tensor:
    # Cast padding, stride & dilation to tuples.
    n = signal.ndim - 2
    stride_ = to_ntuple(stride, n=n)
    dilation_ = to_ntuple(dilation, n=n)
    if isinstance(padding, str):
        if padding == "same":
            if stride != 1 or dilation != 1:
                raise ValueError("stride must be 1 for padding='same'.")
            padding_ = [(k - 1) / 2 for k in kernel.shape[2:]]
        else:
            raise ValueError(f"Padding mode {padding} not supported.")
    else:
        padding_ = to_ntuple(padding, n=n)

    # internal dilation offsets
    offset = torch.zeros(1, 1, *dilation_, device=signal.device, dtype=signal.dtype)
    offset[(slice(None), slice(None), *((0,) * n))] = 1.0

    # correct the kernel by cutting off unwanted dilation trailing zeros
    cutoff = tuple(slice(None, -d + 1 if d != 1 else None) for d in dilation_)

    # pad the kernel internally according to the dilation parameters
    kernel = torch.kron(kernel, offset)[(slice(None), slice(None)) + cutoff]

    # Pad the input signal & kernel tensors (round to support even sized convolutions)
    signal_padding = [r(p) for p in padding_[::-1] for r in (floor, ceil)]
    signal = f.pad(signal, signal_padding, mode=padding_mode)

    # Because PyTorch computes a *one-sided* FFT, we need the final dimension to
    # have *even* length.  Just pad with one more zero if the final dimension is odd.
    signal_size = signal.size()  # original signal size without padding to even
    if signal.size(-1) % 2 != 0:
        signal = f.pad(signal, [0, 1])

    kernel_padding = [
        pad
        for i in reversed(range(2, signal.ndim))
        for pad in [0, signal.size(i) - kernel.size(i)]
    ]
    padded_kernel = f.pad(kernel, kernel_padding)

    # signal_fr = torch.matmul(signal.to(torch.complex64), fourier_basis)
    signal_fr_1 = torch.matmul(signal.to(torch.complex64), fourier_basis_1)
    signal_fr_2 = torch.matmul(signal.to(torch.complex64), fourier_basis_2)
    signal_fr_3 = torch.matmul(signal.to(torch.complex64), fourier_basis_3)
    signal_fr_4 = torch.matmul(signal.to(torch.complex64), fourier_basis_4)
    signal_fr_5 = torch.matmul(signal.to(torch.complex64), fourier_basis_5)

    kernel_fr = rfftn(padded_kernel.float(), dim=tuple(range(2, signal.ndim)))


    # 频域乘法结果做逆变换
    kernel_fr.imag *= -1
    output_fr_1 = complex_matmul(signal_fr_1, kernel_fr[0:kernel_fr.shape[0]//5 * 1], groups=groups)
    output_fr_2 = complex_matmul(signal_fr_2, kernel_fr[kernel_fr.shape[0] // 5 * 1:kernel_fr.shape[0] // 5 * 2], groups=groups)
    output_fr_3 = complex_matmul(signal_fr_3, kernel_fr[kernel_fr.shape[0] // 5 * 2:kernel_fr.shape[0] // 5 * 3], groups=groups)
    output_fr_4 = complex_matmul(signal_fr_4, kernel_fr[kernel_fr.shape[0] // 5 * 3:kernel_fr.shape[0] // 5 * 4], groups=groups)
    output_fr_5 = complex_matmul(signal_fr_5, kernel_fr[kernel_fr.shape[0] // 5 * 4:kernel_fr.shape[0]], groups=groups)

    output_fr = torch.cat((output_fr_1, output_fr_2), dim=1)
    output_fr = torch.cat((output_fr, output_fr_3), dim=1)
    output_fr = torch.cat((output_fr, output_fr_4), dim=1)
    output_fr = torch.cat((output_fr, output_fr_5), dim=1)

    output = irfftn(output_fr, dim=tuple(range(2, signal.ndim)))

    # 移除padding
    crop_slices = [slice(None), slice(None)] + [
        slice(0, (signal_size[i] - kernel.size(i) + 1), stride_[i - 2])
        for i in range(2, signal.ndim)
    ]
    output = output[crop_slices].contiguous()

    # 添加bias
    if bias is not None:
        bias_shape = tuple([1, -1] + (signal.ndim - 2) * [1])
        output += bias.view(bias_shape)

    return output


class _FTConv(nn.Module):
    """Base class for PyTorch FFT convolution layers."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Iterable[int]],
            padding: int,
            padding_mode: str = "constant",
            stride: Union[int, Iterable[int]] = 1,
            dilation: Union[int, Iterable[int]] = 1,
            groups: int = 1,
            bias: bool = True,
            ndim: int = 1,
            featureDim: int = 3000,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.padding_mode = padding_mode
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.use_bias = bias

        self.fourier_basis_1 = initMAT(featureDim, start=0, end=122).cuda()
        self.fourier_basis_2 = initMAT(featureDim, start=122, end=244).cuda()
        self.fourier_basis_3 = initMAT(featureDim, start=244, end=366).cuda()
        self.fourier_basis_4 = initMAT(featureDim, start=366, end=488).cuda()
        self.fourier_basis_5 = initMAT(featureDim, start=488, end=972).cuda()

        if in_channels % groups != 0:
            raise ValueError(
                "'in_channels' must be divisible by 'groups'."
                f"Found: in_channels={in_channels}, groups={groups}."
            )
        if out_channels % groups != 0:
            raise ValueError(
                "'out_channels' must be divisible by 'groups'."
                f"Found: out_channels={out_channels}, groups={groups}."
            )

        kernel_size = to_ntuple(kernel_size, ndim)
        weight = torch.randn(out_channels, in_channels // groups, *kernel_size)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None

    def forward(self, signal):

        return fft_conv(
            signal,
            self.weight,
            bias=self.bias,
            padding=self.padding,
            padding_mode=self.padding_mode,
            stride=self.stride,
            dilation=self.dilation,
            groups=self.groups,

            fourier_basis_1=self.fourier_basis_1,
            fourier_basis_2=self.fourier_basis_2,
            fourier_basis_3=self.fourier_basis_3,
            fourier_basis_4=self.fourier_basis_4,
            fourier_basis_5=self.fourier_basis_5,
        )

FTConv1d = partial(_FTConv, ndim=1)

'''
x = torch.rand([64,1,750])
a = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=5, stride=1, groups=1,padding=2)
b = FTConv1d(in_channels=1, out_channels=128, kernel_size=5, stride=1, groups=1,padding=2, featureDim=754)

print(a.weight.data.shape,a.bias.data.shape)
print(b.weight.data.shape,b.bias.data.shape)
b(x)
'''