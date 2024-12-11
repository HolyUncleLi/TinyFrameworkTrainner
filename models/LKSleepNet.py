from models.ReConv_2 import *
from models.TimesNet_LK import getmodel
from models.embed import *
from models.CBAM import CBAM1d,CBAM2d
from models.CRF import CRF
import numpy as np
'''
TimesNet 上下文时序编码
'''

class SEBlock(nn.Module):
    def __init__(self, in_dim, reduction=16):
        super().__init__()
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_dim, in_dim // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_dim // reduction, in_dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        weights = self.layers(x)
        weights = weights.unsqueeze(-1)
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


def get_conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    return nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=bias)


def get_bn(channels):
    return nn.BatchNorm1d(channels)


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1,bias=False,isFTConv=True):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module('conv', get_conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))
    result.add_module('bn', get_bn(out_channels))
    return result


def fuse_bn(conv, bn):

    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std


class ReparamLargeKernelConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, groups,
                 small_kernel,
                 small_kernel_merged=False, nvars=7):
        super(ReparamLargeKernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel

        padding = kernel_size // 2
        if small_kernel_merged:
            self.lkb_reparam = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=1, groups=groups, bias=True)
        else:
            self.lkb_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                        stride=stride, padding=padding, dilation=1, groups=groups,bias=False)
            if small_kernel is not None:
                assert small_kernel <= kernel_size, 'The kernel size for re-param cannot be larger than the large kernel!'
                self.small_conv = conv_bn(in_channels=in_channels, out_channels=out_channels,
                                            kernel_size=small_kernel,
                                            stride=stride, padding=small_kernel // 2, groups=groups, dilation=1,bias=False)

    def forward(self, inputs):

        if hasattr(self, 'lkb_reparam'):
            out = self.lkb_reparam(inputs)
        else:
            out = self.lkb_origin(inputs)
            if hasattr(self, 'small_conv'):
                out += self.small_conv(inputs)
        return out

    def PaddingTwoEdge1d(self, x, pad_length_left, pad_length_right, pad_values=0):

        D_out, D_in, ks = x.shape
        if pad_values ==0:
            pad_left = torch.zeros(D_out,D_in,pad_length_left).cuda()
            pad_right = torch.zeros(D_out,D_in,pad_length_right).cuda()
        else:
            pad_left = torch.ones(D_out, D_in, pad_length_left).cuda() * pad_values
            pad_right = torch.ones(D_out, D_in, pad_length_right).cuda() * pad_values

        x = torch.cat((pad_left, x), dim=-1)
        x = torch.cat((x, pad_right), dim=-1)
        return x

    def get_equivalent_kernel_bias(self):
        eq_k, eq_b = fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)
        if hasattr(self, 'small_conv'):
            small_k, small_b = fuse_bn(self.small_conv.conv, self.small_conv.bn)
            eq_b += small_b
            eq_k += self.PaddingTwoEdge1d(small_k, (self.kernel_size - self.small_kernel) // 2,
                                          (self.kernel_size - self.small_kernel) // 2, 0)
        return eq_k, eq_b

    def merge_kernel(self):
        eq_k, eq_b = self.get_equivalent_kernel_bias()
        self.lkb_reparam = nn.Conv1d(in_channels=self.lkb_origin.conv.in_channels,
                                     out_channels=self.lkb_origin.conv.out_channels,
                                     kernel_size=self.lkb_origin.conv.kernel_size, stride=self.lkb_origin.conv.stride,
                                     padding=self.lkb_origin.conv.padding, dilation=self.lkb_origin.conv.dilation,
                                     groups=self.lkb_origin.conv.groups, bias=True)
        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        self.__delattr__('lkb_origin')
        if hasattr(self, 'small_conv'):
            self.__delattr__('small_conv')


class Block(nn.Module):
    def __init__(self, large_size, small_size, dmodel, dff, nvars, small_kernel_merged=False, drop=0.05):

        super(Block, self).__init__()

        self.dw = ReparamLargeKernelConv(in_channels=nvars * dmodel, out_channels=nvars * dmodel,
                                         kernel_size=large_size, stride=1, groups=nvars * dmodel,
                                         small_kernel=small_size, small_kernel_merged=small_kernel_merged, nvars=nvars)
        self.norm = nn.BatchNorm1d(dmodel)
        self.se = SEBlock(in_dim=dmodel)
        self.cbam = CBAM1d(channel=dmodel, reduction=16, kernel_size=7)

        #convffn1
        self.ffn1pw1 = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * dff, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=nvars)
        # self.ffn1act1 = nn.GELU()
        self.ffn1act1 = nn.PReLU()
        self.ffn1norm1 = nn.BatchNorm1d(nvars * dff)
        self.ffn1pw2 = nn.Conv1d(in_channels=nvars * dff, out_channels=nvars * dmodel, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=nvars)
        self.ffn1norm2 = nn.BatchNorm1d(nvars * dmodel)
        # self.ffn1act2 = nn.GELU()
        self.ffn1act2 = nn.PReLU()
        self.ffn1drop1 = nn.Dropout(drop)
        self.ffn1drop2 = nn.Dropout(drop)

        self.ffn_ratio = dff//dmodel
        self.shortcut = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * dmodel, kernel_size=1, stride=1,
                                 padding=0, dilation=1)

    def forward(self, x):

        input = x
        B, M, D, N = x.shape
        x = x.reshape(B, M*D, N)

        x = self.dw(x)
        x = x.reshape(B, M, D, N)
        x = x.reshape(B*M, D, N)
        x = self.norm(x)
        x = x.reshape(B, M, D, N)
        x = x.reshape(B, M * D, N)
        x = self.cbam(x)

        x = self.ffn1drop1(self.ffn1pw1(x))
        x = self.ffn1act1(x)
        x = self.ffn1drop2(self.ffn1pw2(x))

        x = x.reshape(B, M, D, N)
        x = input + x
        return x


class Stage(nn.Module):
    def __init__(self, ffn_ratio, num_blocks, large_size, small_size, dmodel, dw_model, nvars,
                 small_kernel_merged=False, drop=0.1):

        super(Stage, self).__init__()
        d_ffn = dmodel * ffn_ratio
        blks = []
        for i in range(num_blocks):
            blk = Block(large_size=large_size, small_size=small_size, dmodel=dmodel, dff=d_ffn, nvars=nvars, small_kernel_merged=small_kernel_merged, drop=drop)
            blks.append(blk)

        self.blocks = nn.ModuleList(blks)

    def forward(self, x):

        for blk in self.blocks:
            x = blk(x)

        return x


class ModernTCN(nn.Module):
    def __init__(self,task_name,patch_size,patch_stride, stem_ratio, downsample_ratio, ffn_ratio, num_blocks, large_size, small_size, dims, dw_dims,
                 nvars, small_kernel_merged=False, backbone_dropout=0.1, head_dropout=0.1, use_multi_scale=True, revin=True, affine=True,
                 subtract_last=False, freq=None, seq_len=512, c_in=7, individual=False, target_window=96, class_drop=0.,class_num = 10,
                 ftconv_layer=None):

        super(ModernTCN, self).__init__()

        self.task_name = task_name
        self.class_drop = class_drop
        self.batchsize = 15
        self.seq_len = 20
        self.cnndim = 1024
        self.featuredim = 128
        self.class_num = class_num

        # stem layer & down sampling layers
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv1d(1, dims[0], kernel_size=patch_size, stride=patch_stride),
            nn.BatchNorm1d(dims[0])
        )
        self.downsample_layers.append(stem)

        self.num_stage = len(num_blocks)
        if self.num_stage > 1:
            for i in range(self.num_stage - 1):
                downsample_layer = nn.Sequential(
                    nn.BatchNorm1d(dims[i]),
                    nn.Conv1d(dims[i], dims[i + 1], kernel_size=downsample_ratio * 2, stride=downsample_ratio),
                )
                self.downsample_layers.append(downsample_layer)

        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.downsample_ratio = downsample_ratio


        # FTCNN
        self.ftcnn_channels = dims[0]
        self.ftcnn = FTConv1d(in_channels=1, out_channels=self.ftcnn_channels, kernel_size=9, stride=1,
                                 padding=4, featureDim=3016)
        self.ftcnn_downsample_layer = nn.Sequential(
            nn.BatchNorm1d(self.ftcnn_channels),
            nn.Conv1d(self.ftcnn_channels, self.ftcnn_channels, kernel_size=patch_size, stride=patch_stride),
        )


        # cnn backbone
        self.num_stage = len(num_blocks)
        self.stages = nn.ModuleList()
        for stage_idx in range(self.num_stage):
            layer = Stage(ffn_ratio, num_blocks[stage_idx], large_size[stage_idx], small_size[stage_idx], dmodel=dims[stage_idx],
                          dw_model=dw_dims[stage_idx], nvars=nvars, small_kernel_merged=small_kernel_merged, drop=backbone_dropout)
            self.stages.append(layer)
        self.maxpool = nn.AdaptiveMaxPool1d(8)
        self.flatten = nn.Flatten()

        # times backbone
        self.embed = ARFEmbedding(128, 16)
        # self.embed = CBAMEmbedding(128, 16)
        self.times_drop = nn.Dropout(0.5)
        self.timesNet = getmodel()

        # head
        self.n_vars = c_in
        self.individual = individual

        if self.task_name == 'classification':
            self.act_class = F.gelu
            self.class_dropout = nn.Dropout(0.5)
            self.head_class2 = nn.Linear(self.featuredim, self.class_num)

        self.mask = torch.ones([self.batchsize, self.seq_len]).to(bool)
        self.crf = CRF(5)

    def forward_feature(self, x, te=None):
        B, M, L = x.shape
        ftcnn_res = torch.rand([]).cuda()
        x = x.unsqueeze(-2)

        for i in range(self.num_stage):
            B, M, D, N = x.shape
            x = x.reshape(B * M, D, N)
            if i == 0:
                if self.patch_size != self.patch_stride:
                    # stem layer padding
                    pad_len = self.patch_size - self.patch_stride
                    pad = x[:,:,-1:].repeat(1,1, pad_len)
                    x = torch.cat([x,pad],dim=-1)
                ftcnn_res = self.ftcnn(x)
                ftcnn_res = self.ftcnn_downsample_layer(ftcnn_res).unsqueeze(1)

            else:
                if N % self.downsample_ratio != 0:
                    pad_len = self.downsample_ratio - (N % self.downsample_ratio)
                    x = torch.cat([x, x[:, :, -pad_len:]], dim=-1)

            x = self.downsample_layers[i](x)

            _, D_, N_ = x.shape
            x = x.reshape(B, M, D_, N_)

            if i == 0:
                x += ftcnn_res

            x = self.stages[i](x)
        return x

    def classification1(self, x):
        x = self.forward_feature(x, te=None).squeeze()
        x = self.act_class(x)
        x = self.class_dropout(x)
        x = self.maxpool(x)
        x = x.contiguous().view(self.batchsize * self.seq_len, self.cnndim)
        x = self.head_class1(x)
        return x

    def classification2(self, x, tags=None):
        # cnn
        x = self.forward_feature(x, te=None).squeeze()
        # x = self.act_class(x)
        # x = self.class_dropout(x)
        x = self.maxpool(x)
        # x = self.flatten(x)
        # x = x.view(self.batchsize, self.seq_len, self.cnndim)
        x = self.embed(x, None, stage=2)
        cnn_out = x

        # get period
        x = self.timesNet(x, None, None, None)
        x = self.times_drop(x)
        x = x + cnn_out

        # head
        x = x.view(self.batchsize * self.seq_len, self.featuredim)
        x = self.head_class2(x)
        '''
        x = x.reshape(self.batchsize, self.seq_len, 5)
        if tags is not None:
            x = self.crf.forward(x, tags.reshape(self.batchsize, self.seq_len), self.mask)
            x = (-torch.mean(x))
        else:
            x = self.crf.viterbi_decode(x, self.mask)
            x = np.array(x).reshape(-1)
        '''
        return x

    def forward(self, x, tags=None, pre_stage=2):
        if pre_stage == 1:
            x = self.classification1(x)
        elif pre_stage == 2:
            x = self.classification2(x, tags=tags)
        return x

    def structural_reparam(self):
        for m in self.modules():
            if hasattr(m, 'merge_kernel'):
                m.merge_kernel()


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        # hyper param
        self.task_name = configs.task_name
        self.stem_ratio = configs.stem_ratio
        self.downsample_ratio = configs.downsample_ratio
        self.ffn_ratio = configs.ffn_ratio
        self.num_blocks = configs.num_blocks
        self.large_size = configs.large_size
        self.small_size = configs.small_size
        self.dims = configs.dims
        self.dw_dims = configs.dw_dims

        self.nvars = configs.enc_in
        self.small_kernel_merged = configs.small_kernel_merged
        self.drop_backbone = configs.dropout
        self.drop_head = configs.head_dropout
        self.use_multi_scale = configs.use_multi_scale
        self.revin = configs.revin
        self.affine = configs.affine
        self.subtract_last = configs.subtract_last

        self.freq = configs.freq
        self.seq_len = configs.seq_len
        self.c_in = self.nvars,
        self.individual = configs.individual
        self.target_window = configs.pred_len

        self.kernel_size = configs.kernel_size
        self.patch_size = configs.patch_size
        self.patch_stride = configs.patch_stride

        #classification
        self.class_dropout = configs.class_dropout
        self.class_num = configs.num_class

        self.model = ModernTCN(task_name=self.task_name,patch_size=self.patch_size, patch_stride=self.patch_stride, stem_ratio=self.stem_ratio,
                           downsample_ratio=self.downsample_ratio, ffn_ratio=self.ffn_ratio, num_blocks=self.num_blocks,
                           large_size=self.large_size, small_size=self.small_size, dims=self.dims, dw_dims=self.dw_dims,
                           nvars=self.nvars, small_kernel_merged=self.small_kernel_merged,
                           backbone_dropout=self.drop_backbone, head_dropout=self.drop_head,
                           use_multi_scale=self.use_multi_scale, revin=self.revin, affine=self.affine,
                           subtract_last=self.subtract_last, freq=self.freq, seq_len=self.seq_len, c_in=self.c_in,
                           individual=self.individual, target_window=self.target_window,
                            class_drop = self.class_dropout, class_num = self.class_num,ftconv_layer=[False,False])

    def forward(self, x, tags=None, pre_stage=2, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # x = x.permute(0, 2, 1)
        x = self.model(x, tags, pre_stage)
        return x
