import os
import torch
import argparse
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from torch.optim import lr_scheduler
# from TCN_Block_15_RevINtimes import *
# from TCN_Block_14_tiny2 import *
# from TCN_Block_14_times_tiny import *
# from TCN_Block_14_tiny_MTGODE import *

from models.LKSleepNet import *
# from LKSleepNet_tiny import *

parser = argparse.ArgumentParser(description='ModernTCN')

# random seed
parser.add_argument('--random_seed', type=int, default=2024, help='random seed')

# basic config
parser.add_argument('--is_training', type=int, default=1, help='status')
parser.add_argument('--model_id', type=str, default='test', help='model id')
parser.add_argument('--model', type=str,  default='ModernTCN',
                    help='model name, options: [ModernTCN]')

# data loader
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')



# forecasting task
parser.add_argument('--seq_len', type=int, default=10000, help='input sequence length')
parser.add_argument('--label_len', type=int, default=1, help='start token length')
parser.add_argument('--pred_len', type=int, default=64, help='prediction sequence length')
parser.add_argument('--num_class', type=int, default=5, help='num of class')

# ModernTCN base
parser.add_argument('--stem_ratio', type=int, default=1000000, help='stem ratio')
parser.add_argument('--downsample_ratio', type=int, default=4, help='downsample_ratio')
parser.add_argument('--ffn_ratio', type=int, default=4, help='ffn_ratio')
parser.add_argument('--patch_size', type=int, default=16, help='the patch size')
parser.add_argument('--patch_stride', type=int, default=8, help='the patch stride')

parser.add_argument('--num_blocks', nargs='+', type=int, default=[1, 1], help='num_blocks in each stage')
parser.add_argument('--large_size', nargs='+', type=int, default=[51, 31], help='big kernel size')
parser.add_argument('--small_size', nargs='+', type=int, default=[5, 5], help='small kernel size for structral reparam')
parser.add_argument('--dims', nargs='+', type=int, default=[64, 128], help='dmodels in each stage')
parser.add_argument('--dw_dims', nargs='+',type=int, default=[64, 128], help='dw dims in dw conv in each stage')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout')

parser.add_argument('--small_kernel_merged', type=bool, default=False, help='small_kernel has already merged or not')
parser.add_argument('--call_structural_reparam', type=bool, default=False, help='structural_reparam after training')
parser.add_argument('--use_multi_scale', type=bool, default=True, help='use_multi_scale fusion')


# PatchTST
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

# Formers
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
parser.add_argument('--c_out', type=int, default=1000, help='output size')
parser.add_argument('--d_model', type=int, default=256, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=5, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=1024, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)

parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

# multi task
parser.add_argument('--task_name', type=str, default='classification',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
# classfication task
parser.add_argument('--class_dropout', type=float, default=0.5, help='classfication dropout')

args = parser.parse_args()

setting = '{}_{}_sl{}_pl{}_dim{}_nb{}_lk{}_sk{}_ffr{}_ps{}_str{}_multi{}_merged_{}_{}'.format(
            args.model_id,
            args.model,
            args.seq_len,
            args.pred_len,
            args.dims[0],

            args.num_blocks[0],
            args.large_size[0],
            args.small_size[0],
            args.ffn_ratio,
            args.patch_size,
            args.patch_stride,
            args.small_kernel_merged,
            args.des,
            0)


def getmodel():
    return Model(args)

'''
model = Model(args)
# print(model(torch.rand(64, 3000, 1).cuda(),None, None, None).shape)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")
'''