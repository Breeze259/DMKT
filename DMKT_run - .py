import argparse
import copy
import csv
import os
# 处理数据集
import random
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
# 导入所需要的函数
from sklearn.metrics import roc_curve, mean_squared_error, mean_absolute_error, accuracy_score, auc
from torch import eye, zeros, tensor, detach
from torch.utils.data import DataLoader
from pytorch_tools import EarlyStopping


import torch, math
import torch.nn as nn
import torch.nn.functional as F


def get_activation_fn(activation):
    if activation == "relu": return nn.ReLU()
    elif activation == "gelu": return nn.GELU()
    else: return activation()

class SublayerConnection(nn.Module):

    def __init__(self, enable_res_parameter, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.enable = enable_res_parameter
        if enable_res_parameter:
            self.a = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, out_x):
        if not self.enable:
            return x + self.dropout(out_x) 
        else:
            return x + self.dropout(self.a * out_x)  

class _ConvEncoderLayer(nn.Module):
    def __init__(self, kernel_size, d_model, d_ff=256, dropout=0.1, activation="relu", 
                 enable_res_param=True, norm='batch', small_ks=3, re_param=True, device='cuda:0', enable_multi_kernel=True):
        super(_ConvEncoderLayer, self).__init__()

        self.norm_tp = norm
        self.re_param = re_param
        self.enable_multi_kernel = enable_multi_kernel

        # 多尺度卷积核配置（对应不同工作记忆容量）
        if self.enable_multi_kernel:
            self.kernel_sizes = [5,11,21,29,39,51]  # 不同尺度的卷积核
            self.multi_kernel_convs = nn.ModuleList([
                nn.Conv1d(d_model, d_model, ks, stride=1, padding='same', groups=d_model)
                for ks in self.kernel_sizes
            ])
            self.kernel_fusion = nn.Conv1d(d_model * len(self.kernel_sizes), d_model, 1)
        else:
            # 原有单核实现
            if self.re_param:
                self.large_ks = kernel_size
                self.small_ks = small_ks
                self.DW_conv_large = nn.Conv1d(d_model, d_model, self.large_ks, stride=1, padding='same', groups=d_model)
                self.DW_conv_small = nn.Conv1d(d_model, d_model, self.small_ks, stride=1, padding='same', groups=d_model)
                self.DW_infer = nn.Conv1d(d_model, d_model, self.large_ks, stride=1, padding='same', groups=d_model)
            else:
                self.DW_conv = nn.Conv1d(d_model, d_model, kernel_size, stride=1, padding='same', groups=d_model)

        self.dw_act = get_activation_fn(activation)

        self.sublayerconnect1 = SublayerConnection(enable_res_param, dropout)
        self.dw_norm = nn.BatchNorm1d(d_model) if norm == 'batch' else nn.LayerNorm(d_model)

    def _get_merge_param(self):
        left_pad = (self.large_ks - self.small_ks) // 2
        right_pad = (self.large_ks - self.small_ks) - left_pad
        module_output = copy.deepcopy(self.DW_conv_large)
        
        # 避免in-place操作，使用clone()和detach()
        small_weight_padded = F.pad(self.DW_conv_small.weight.detach(), (left_pad, right_pad), value=0)
        module_output.weight = nn.Parameter(module_output.weight.detach() + small_weight_padded)
        module_output.bias = nn.Parameter(module_output.bias.detach() + self.DW_conv_small.bias.detach())
        
        self.DW_infer = module_output

    def forward(self, src): # [B, C, L]
        if self.enable_multi_kernel:
            # 多核模式：并行计算多个卷积核的输出
            kernel_outputs = []
            for conv in self.multi_kernel_convs:
                kernel_output = conv(src)
                kernel_outputs.append(kernel_output)
            
            # 融合多核输出
            fused_output = torch.cat(kernel_outputs, dim=1)  # [B, C*num_kernels, L]
            fused_output = self.kernel_fusion(fused_output)  # [B, C, L]
            
            # 应用激活函数和残差连接
            src = self.sublayerconnect1(src, self.dw_act(fused_output))
            
            # 归一化
            src = src.permute(0, 2, 1) if self.norm_tp != 'batch' else src
            src = self.dw_norm(src)      
            src = src.permute(0, 2, 1) if self.norm_tp != 'batch' else src
            
            # 返回主输出和多核输出
            return src, kernel_outputs
        else:
            # 原有单核实现
            if not self.re_param:
                src = self.DW_conv(src)
            else:
                if self.training: # training phase
                    large_out, small_out = self.DW_conv_large(src), self.DW_conv_small(src)
                    src = self.sublayerconnect1(src, self.dw_act(large_out+small_out))
                else: # testing phase
                    self._get_merge_param()
                    merge_out = self.DW_infer(src)
                    src = self.sublayerconnect1(src, self.dw_act(merge_out))

            src = src.permute(0, 2, 1) if self.norm_tp != 'batch' else src
            src = self.dw_norm(src)      
            src = src.permute(0, 2, 1) if self.norm_tp != 'batch' else src
            
            return src


class _ConvEncoder(nn.Module):
    def __init__(self, d_model, d_ff, kernel_size=[5,11,21,29,39,51], dropout=0.1, activation='gelu', 
                 n_layers=6, enable_res_param=True, norm='batch', re_param=True, device='cuda:0', enable_multi_kernel=True):
        
        super(_ConvEncoder, self).__init__()
        self.layers = nn.ModuleList([_ConvEncoderLayer(kernel_size[i], d_model, d_ff=d_ff, dropout=dropout, 
                                                        activation=activation, enable_res_param=enable_res_param, norm=norm, 
                                                        re_param=re_param, device=device, enable_multi_kernel=enable_multi_kernel) 
                                                        for i in range(n_layers)])

    def forward(self, src):
        output = src
        multi_kernel_outputs = []
        
        for mod in self.layers:
            if self.layers[0].enable_multi_kernel:  # 检查是否启用多核模式
                output, layer_kernel_outputs = mod(output)
                multi_kernel_outputs.extend(layer_kernel_outputs)
            else:
                output = mod(output)
        
        if self.layers[0].enable_multi_kernel:
            return output, multi_kernel_outputs
        else:
            return output

class BoxCoder(nn.Module):
    def __init__(self, patch_count, stride, patch_size, seq_len, in_feats):
        super().__init__()

        self.seq_len = seq_len
        self.in_feats = in_feats
        self.patch_size = patch_size
        self.patch_count = patch_count
        self.stride = stride
        
    # compute the center points. idx: [0 ~ seq_len - 1]
    def _generate_anchor(self):
        anchors = []
        self.S_bias = (self.patch_size - 1) / 2
        
        for i in range(self.patch_count):
            x = i * self.stride + 0.5 * (self.patch_size - 1)
            anchors.append(x)

        anchors = torch.as_tensor(anchors)
        # print(f"BoxCoder anchor count: {len(anchors)}")
        return anchors

    def forward(self, boxes):
        self.bound = self.decode(boxes) # (bs, patch_count, channel, 2)
        points = self.meshgrid(self.bound)

        return points, self.bound

    def decode(self, rel_codes):  # Input: (B, patch_count, channel, 2)
        # 每次decode时都重新生成anchor，确保与当前patch_count匹配
        boxes = self._generate_anchor()
        
        # 确保anchor在与输入相同的设备上
        device = rel_codes.device
        boxes = boxes.to(device)

        dx = rel_codes[:, :, :, 0]
        ds = torch.relu(rel_codes[:, :, :, 1] + self.S_bias)

        # 打印张量形状用于调试
        # print(f"dx shape: {dx.shape}")
        # print(f"ds shape: {ds.shape}")
        # print(f"boxes shape: {boxes.shape}")

        pred_boxes = torch.zeros_like(rel_codes)
        ref_x = boxes.view(1, boxes.shape[0], 1, 1).to(rel_codes.device)  # 调整为 (1, patch_count, 1, 1)并确保设备匹配

        # dx, ds: (bs, patch_count, channel, 1)
        # 确保所有张量形状匹配
        dx = dx.unsqueeze(-1)  # 添加最后一个维度
        ds = ds.unsqueeze(-1)  # 添加最后一个维度
        
        # print(f"dx after unsqueeze shape: {dx.shape}")
        # print(f"ds after unsqueeze shape: {ds.shape}")
        # print(f"ref_x shape: {ref_x.shape}")
        
        # 调整ref_x的形状以匹配dx和ds
        ref_x = ref_x.expand(dx.shape[0], -1, -1, -1)  # 扩展到 [32, 49, 1, 1]
        # print(f"ref_x after expand shape: {ref_x.shape}")
        
        pred_boxes[:, :, :, 0] = (dx + ref_x - ds).squeeze(-1)
        pred_boxes[:, :, :, 1] = (dx + ref_x + ds).squeeze(-1)
        pred_boxes /= (self.seq_len - 1)

        pred_boxes = pred_boxes.clamp_(min=0., max=1.)

        # pred_boxes: each of the patch's left-bound & right-bound. norm to [0, 1]
        return pred_boxes   
   
    def meshgrid(self, boxes): # Input: pred_boxes. To get the sampling location
        B, patch_count, C = boxes.shape[0], boxes.shape[1], boxes.shape[2]
        device = boxes.device
        channel_boxes = torch.zeros((boxes.shape[0], boxes.shape[1], 2), device=device)
        channel_boxes[:, :, 1] = 1.0
        xs = boxes.view(B*patch_count, C, 2)
        xs = torch.nn.functional.interpolate(xs, size=self.patch_size, mode='linear', align_corners=True)
        ys = torch.nn.functional.interpolate(channel_boxes, size=self.in_feats, mode='linear', align_corners=True)

        # xs: [bs, patch_count, channel, patch_size]   ys: [bs, patch_count, channels(also feats)]
  
        xs = xs.view(B, patch_count, C, self.patch_size, 1)
        ys = ys.unsqueeze(3).expand(B, patch_count, C, self.patch_size).unsqueeze(-1)
  
        grid = torch.stack([xs, ys], dim = -1)
        return grid # [bs, patch_count, channel, patch_size, 2]

def zero_init(m):
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        m.weight.data.fill_(0)
        m.bias.data.fill_(0)

class OffsetPredictor(nn.Module):
    def __init__(self, in_feats, patch_size, stride, use_zero_init=True):
        """
        Note: decoupling on channel-dim !
        """
        super().__init__()
        self.stride = stride
        self.channel = in_feats
        self.patch_size = patch_size

        self.offset_predictor = nn.Sequential(
            nn.Conv1d(1, 64, patch_size, stride=stride, padding=0), 
            nn.GELU(),
            nn.Conv1d(64, 2, 1, 1, padding=0) 
        )

        if use_zero_init:
            self.offset_predictor.apply(zero_init)
        
    def forward(self, X): # Input: (bs, channel, seq_len)
        
        # print(f"Input X shape: {X.shape}")
        # print(f"patch_size: {self.patch_size}, channel: {self.channel}, stride: {self.stride}")
        
        patch_X = X.unsqueeze(1).permute(0, 1, 3, 2)
        # print(f"After unsqueeze and permute: {patch_X.shape}")
        
        # 检查输入尺寸是否足够进行unfold操作
        B, _, H, W = patch_X.shape
        if H < self.patch_size or W < self.channel:
            # 如果输入尺寸小于kernel_size，需要进行填充
            pad_h = max(0, self.patch_size - H)
            pad_w = max(0, self.channel - W)
            patch_X = F.pad(patch_X, (0, pad_w, 0, pad_h))
            print(f"After padding: {patch_X.shape}")
        
        patch_X = F.unfold(patch_X, kernel_size=(self.patch_size, self.channel), stride=self.stride).permute(0, 2, 1) # (B, patch_count, patch_size*channel)
        #print(f"After unfold and permute: {patch_X.shape}")

        # decoupling
        B, patch_count = patch_X.shape[0], patch_X.shape[1] 
        #print(f"OffsetPredictor patch_count: {patch_count}")
        patch_X = patch_X.contiguous().view(B, patch_count, self.patch_size, self.channel)
        patch_X = patch_X.permute(0, 1, 3, 2)

        # patch_X: (B, patch_count, channel, patchsize)
        patch_X = patch_X.contiguous().view(B*patch_count*self.channel, 1, self.patch_size)

        # calculate the bias throughout 2 Conv1d
        pred_offset = self.offset_predictor(patch_X)
        pred_offset = pred_offset.view(B, patch_count, self.channel, 2).contiguous()

        # For each of the patch block and it's channel, there exists a bias（dx, ds）
        # pred_offset: (B, patch_count, channel, 2)
        return pred_offset 

# Input: (B, C, L)  Output: (B, C, patch_num * patch_len)
class DepatchSampling(nn.Module):
    def __init__(self, in_feats, seq_len, patch_size, stride):   
        super(DepatchSampling, self).__init__()
        self.in_feats = in_feats
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.stride = stride

        # 不在初始化时计算patch_count，在forward时根据实际输入动态计算
        self.patch_count = None
  
        self.dropout = nn.Dropout(0.1)
  
        # offset predictor
        self.offset_predictor = OffsetPredictor(in_feats, patch_size, stride)

        # 不在初始化时创建box_coder，在forward时根据实际patch_count创建
        self.box_coder = None
  
    def get_sampling_location(self, X): # Input: (bs, channel, window)
        """
        Input shape: (bs, channel, window) ;
        Sampling location  shape: [bs, patch_count, C, self.patch_size, 2]. range = [0, 1] ; 
        """
        # get offset
        pred_offset = self.offset_predictor(X)

        sampling_locations, bound = self.box_coder(pred_offset)
        return sampling_locations, bound
    
    def forward(self, X, return_bound=False): # Input: (bs, channel, window)
        # 动态计算patch_count
        actual_seq_len = X.shape[2]  # 获取实际的序列长度
        self.patch_count = max(1, (actual_seq_len - self.patch_size) // self.stride + 1)
        #print(f"DepatchSampling actual_seq_len: {actual_seq_len}, patch_count: {self.patch_count}")
        
        # 动态创建box_coder（如果尚未创建或patch_count发生变化）
        if self.box_coder is None or self.box_coder.patch_count != self.patch_count:
            self.box_coder = BoxCoder(self.patch_count, self.stride, self.patch_size, actual_seq_len, self.in_feats)
            # print(f"BoxCoder created with patch_count: {self.patch_count}")

        # Consider the X as a img. shape: (B, C, H, W) <--> (bs, 1, channel, padded_window)
        img = X.unsqueeze(1)
        B = img.shape[0]

        sampling_locations, bound = self.get_sampling_location(X) # sampling_locations: [bs, patch_count, channel, patch_size, 2]
        sampling_locations = sampling_locations.view(B, self.patch_count*self.in_feats, self.patch_size, 2)

        # print('sampling_locations: ', sampling_locations.shape)

        sampling_locations = (sampling_locations - 0.5) * 2 # location map: [-1, 1]
        output = F.grid_sample(img, sampling_locations, align_corners=True) 
        output = output.view(B, self.patch_count, self.in_feats, self.patch_size)
        output = output.permute(0, 2, 1, 3).contiguous()
        return output # (B, C, patch_count, patch_size)

# argparse 模块  处理命令行参数
parser = argparse.ArgumentParser(description='DKT')  # 创建解析器
# base information 添加参数
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')  # 使用的GPU
parser.add_argument('--resume', default='./model_info_store', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--dataset_path', default='./dataset/assistments09', type=str,
                    help='path to dataset')  # 数据集路径
parser.add_argument('--dataset_name', default='assistments09', type=str,
                    help='name of dataset')  # 数据集名称
parser.add_argument('--dataset_seed', default=0, type=int,
                    help='seed of dataset')  # 设置seed
parser.add_argument('--cv_num', default=3, type=int,
                    help='K fold Cross-Validation (default:- 5)')  # 交叉验证，3次
parser.add_argument('--lr', default=0.001, type=float,
                    help='initial learning rate', dest='lr')  # 学习率
parser.add_argument('--batch-size', default=64, type=int,
                    help='mini-batch size (default: 32)')  # 批量大小
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')  # 重启时 手动设置epoch
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')  # 初始epoch值200
parser.add_argument('--seq_len', default=200, type=int,
                    help='the max length of the learning process')  # 交互序列（练习序列）的最大长度
parser.add_argument('--optim', default='adamax', type=str,
                    choices=['adam', 'rmsprop', 'sgd', 'adamax', 'adagrad', 'adadelta'],
                    help='type of optimizer')  # 优化器的选择
parser.add_argument('--dropout', default=0.25, type=float,
                    help='dropout parameter(default: 0.75)')  # 减少过拟合，正则化
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')  # SGD中的 上一次梯度值的权重
parser.add_argument('--wd', '--weight-decay', default=0.0001, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',  # 权重衰退
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

# 对抗训练参数
# parser.add_argument('--adversarial_training', action='store_true', default=True,
#                     help='enable adversarial training')
# parser.add_argument('--epsilon', default=0.01, type=float,
#                     help='epsilon for adversarial perturbation (default: 0.01)')

# model specific configs（与特定模型相关的配置参数）:
parser.add_argument('--input_size', default=128, type=int,
                    help='the size of rnn input(embedding vector)')  # 输入大小
parser.add_argument('--hidden_size', default=128, type=int,
                    help='the size of hidden layers')  # 隐藏层大小
parser.add_argument('--embedding_size', default=128, type=int,
                    help='the dim of embedding')  # 嵌入的维度
parser.add_argument('--num_hidden_layers', default=2, type=int,
                    help='number of hidden layers')  # 隐藏层的数量
# knowledge tracing specific configs:
parser.add_argument('--total_num_concept', default=101, type=int,
                    help='number of knowledge concepts')  # 知识概念的数量
parser.add_argument('--total_num_student', default=13692, type=int,
                    help='total number of train_valid dataset student learning process')  # 训练_验证过程数据集数据数量(学生)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 匿名函数lambda 传入参数：返回值
cuda = lambda o: o.cuda() if torch.cuda.is_available() else 1
# cuda上tensor的定义
tensor = lambda o: cuda(torch.tensor(o))
# 初始化RNN隐藏状态
# 生成一个单位矩阵（对角线为1，其余都为0） torch.eye(n)，生成一个n×n的单位矩阵
eye = lambda d: cuda(torch.eye(d))
zeros = lambda *args: cuda(torch.zeros(*args))
# detach : 分离梯度信息,减少存储
detach = lambda o: o.cpu().detach().numpy().tolist()



class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 全局平均池化
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )
        self.register_buffer('attention_weights', None)  # 用于保存注意力权重
        
    def forward(self, x):  # x: [B, C, L]
        B, C, L = x.shape
        avg = self.avg_pool(x).view(B, C)  # 通道统计量
        channel_weights = self.fc(avg).view(B, C, 1)  # 通道权重
        self.attention_weights = channel_weights  # 保存注意力权重
        return x * channel_weights  # 通道重标定

class WorkingMemoryTransfer(nn.Module):
    def __init__(self, hidden_size, num_kernels):
        """
        WMT模块：模拟工作记忆到长时记忆的转移过程
        - hidden_size: 隐藏层大小
        - num_kernels: 卷积核数量（对应不同工作记忆容量）
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_kernels = num_kernels
        
        # 可学习的衰减率参数，每个卷积核对应一个γ
        # 初始值设为0.1，对应中等遗忘速度
        self.gamma = nn.Parameter(torch.ones(num_kernels) * 0.1)
    
    def _create_distance_matrix(self, seq_len, device):
        """动态创建距离矩阵 d(t,τ) = |t-τ|"""
        t_indices = torch.arange(seq_len, device=device).unsqueeze(1)  # (L, 1)
        tau_indices = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, L)
        distance = torch.abs(t_indices - tau_indices)  # (L, L)
        causal_mask = tau_indices > t_indices  # (L, L)，上三角为True（不含对角线）
        distance.masked_fill_(causal_mask, float('inf'))
        return distance
    
    def forward(self, conv_outputs, actual_seq_len):
        """
        conv_outputs: list of tensors, 每个元素形状为 (B, hidden_size, L)
        actual_seq_len: 实际序列长度
        """
        if actual_seq_len <= 0:
            return conv_outputs  # 处理边界情况
            
        B, hidden_size, L = conv_outputs[0].shape
        device = conv_outputs[0].device
        
        #print(f"WMT模块输入: conv_outputs长度={len(conv_outputs)}, 第一个张量形状={conv_outputs[0].shape}")
        #print(f"WMT模块参数: actual_seq_len={actual_seq_len}, B={B}, hidden_size={hidden_size}, L={L}")
        
        # 动态创建距离矩阵
        distance_matrix = self._create_distance_matrix(actual_seq_len, device)  # (actual_seq_len, actual_seq_len)
        #print(f"距离矩阵形状: {distance_matrix.shape}")
        
        ltm_outputs = []
        for i, (gamma, wm_sequence) in enumerate(zip(self.gamma, conv_outputs)):
            #print(f"\n处理第{i}个卷积核输出:")
            #print(f"gamma值: {gamma}")
            #print(f"wm_sequence原始形状: {wm_sequence.shape}")
            
            # wm_sequence: (B, hidden_size, L)
            # 确保序列长度匹配
            if L != actual_seq_len:
                #print(f"序列长度不匹配: L={L}, actual_seq_len={actual_seq_len}")
                # 如果长度不匹配，进行截断或填充
                if L > actual_seq_len:
                    wm_sequence = wm_sequence[:, :, :actual_seq_len]
                    #print(f"截断后wm_sequence形状: {wm_sequence.shape}")
                else:
                    padding = torch.zeros(B, hidden_size, actual_seq_len - L, device=device)
                    wm_sequence = torch.cat([wm_sequence, padding], dim=2)
                    #print(f"填充后wm_sequence形状: {wm_sequence.shape}")
            
            # 计算衰减权重 ρτ = exp(-γ·d(t,τ))
            # 确保数值稳定性：避免指数爆炸
            decay_weights = torch.exp(-gamma * distance_matrix)  # (actual_seq_len, actual_seq_len)
            #print(f"衰减权重形状: {decay_weights.shape}")
            
            # 归一化：确保每个时间步的权重和为1
            decay_weights = decay_weights / decay_weights.sum(dim=1, keepdim=True)
            
            # 扩展维度以支持批量计算
            decay_weights = decay_weights.unsqueeze(0).unsqueeze(0)  # (1, 1, actual_seq_len, actual_seq_len)
            #print(f"扩展后衰减权重形状: {decay_weights.shape}")
            
            # 加权求和：LTM_t = Σ(ρτ ⊙ WM_τ)
            # 使用高效的矩阵乘法
            wm_sequence_t = wm_sequence.permute(0, 2, 1)  # (B, actual_seq_len, hidden_size)
            #print(f"转置后wm_sequence_t形状: {wm_sequence_t.shape}")
            
            # 将衰减权重扩展到批量维度
            decay_weights = decay_weights.expand(B, 1, actual_seq_len, actual_seq_len)  # (B, 1, actual_seq_len, actual_seq_len)
            #print(f"批量扩展后衰减权重形状: {decay_weights.shape}")
            
            # 正确调整维度顺序：将hidden_size维度放在最后
            wm_sequence_t_unsqueeze = wm_sequence_t.unsqueeze(1)  # (B, 1, actual_seq_len, hidden_size)
            #print(f"unsqueeze后wm_sequence_t形状: {wm_sequence_t_unsqueeze.shape}")
            
            #print(f"准备矩阵乘法: decay_weights形状={decay_weights.shape}, wm_sequence_t_unsqueeze形状={wm_sequence_t_unsqueeze.shape}")
            
            # 矩阵乘法：decay_weights (B, 1, actual_seq_len, actual_seq_len) @ wm_sequence_t_unsqueeze (B, 1, actual_seq_len, hidden_size)
            # 结果形状应为 (B, 1, actual_seq_len, hidden_size)
            ltm_sequence = torch.matmul(decay_weights, wm_sequence_t_unsqueeze)  # (B, 1, actual_seq_len, hidden_size)
            #print(f"矩阵乘法结果形状: {ltm_sequence.shape}")
            
            # 移除多余的维度：从 (B, 1, actual_seq_len, hidden_size) 到 (B, actual_seq_len, hidden_size)
            ltm_sequence = ltm_sequence.squeeze(1)  # (B, actual_seq_len, hidden_size)
            #print(f"squeeze后形状: {ltm_sequence.shape}")
            
            ltm_sequence = ltm_sequence.permute(0, 2, 1)  # (B, hidden_size, actual_seq_len)
            #print(f"最终ltm_sequence形状: {ltm_sequence.shape}")
            
            ltm_outputs.append(ltm_sequence)
        
        return ltm_outputs

class DKT(nn.Module):
    def __init__(self, total_num_concept, embedding_size, hidden_size, seq_len=200, enable_wmt=True):
        super(DKT, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.enable_wmt = enable_wmt
        '''
        total_num_concept: 概念的总数，q.num
        embedding_size: the dim of embedding 嵌入的大小
        hidden_size: the size of hidden layers 隐藏层单元大小
        '''
        # 使用embedding层替代onehot编码
        self.embedding = nn.Embedding(2 * total_num_concept, embedding_dim=self.embedding_size)
        # 初始化embedding权重
        nn.init.xavier_uniform_(self.embedding.weight)
        
        # DePatch Sampling module
        self.depatch_sampling = DepatchSampling(
            in_feats=embedding_size,
            seq_len=seq_len,  # 使用实际的序列长度
            patch_size=8,   # 调整为更小的patch size以匹配199的序列长度
            stride=4         # 调整为更小的stride
        )
        
        # 通道注意力机制
        self.channel_attention = ChannelAttention(in_channels=embedding_size, reduction_ratio=8)
        
        # ConvTimeNet encoder (替换RNN)
        self.conv_encoder = _ConvEncoder(
            d_model=hidden_size,         # 模型维度
            d_ff=256,                    # 前馈网络维度
            kernel_size=[5,11,21,29,39,51],     # 大核卷积尺寸
            dropout=0.1,                 # dropout率
            activation="gelu",           # 激活函数
            n_layers=6,                  # 6级卷积金字塔
            enable_res_param=True,      # 不使用残差参数
            norm='batch',                # 归一化类型
            re_param=True,              # 使用重参数化
            enable_multi_kernel=enable_wmt  # 启用多核模式以支持WMT
        )
        
        # WMT模块（工作记忆转移）
        if enable_wmt:
            self.wmt = WorkingMemoryTransfer(hidden_size, num_kernels=6)  # 6个卷积核对应6种工作记忆容量
        
        # LSTM序列建模层（在ConvTimeNet之后）
        self.lstm = nn.LSTM(
            input_size=hidden_size,      # 输入维度与ConvTimeNet输出匹配
            hidden_size=hidden_size//2,  # 隐藏层维度减半以控制参数量
            num_layers=1,               # 单层LSTM
            batch_first=True,           # 输入形状为(B, L, C)
            bidirectional=False         # 单方向以保持轻量
        )
        
        # LSTM输出投影层，将维度恢复到hidden_size
        self.lstm_projection = nn.Linear(hidden_size//2, hidden_size)
        
        # ConvTimeNet的输入嵌入层
        self.conv_embed = nn.Linear(embedding_size, hidden_size)
        
        self.FCs = nn.Sequential(
            nn.Tanh(),  # 激活函数
            nn.Linear(hidden_size, total_num_concept),  # 线性层
            nn.Sigmoid()  # 激活函数
        )

    def forward(self, X):
        size, length = X.shape # B L
        #print(f"Input X shape: {X.shape}")
        # 使用embedding替代onehot编码
        X_embed = self.embedding(X)  # (B, L, embedding_size)
        #print(f"After embedding X shape: {X_embed.shape}")
        
        # 应用DePatch Sampling
        # 首先将形状从 (B, L, C) 转换为 (B, C, L) 以适应DePatch Sampling
        X_depatch = X_embed.permute(0, 2, 1)  # (B, embedding_size, L)
        #print(f"After permute X_depatch shape: {X_depatch.shape}")
        X_depatch = self.depatch_sampling(X_depatch)  # (B, C, patch_count, patch_size)
        #print(f"After DePatch Sampling shape: {X_depatch.shape}")
        
        # 将DePatch Sampling的输出重新整形为ConvTimeNet输入形状
        B, C, patch_count, patch_size = X_depatch.shape
        
        # 将patch特征展平为序列形式 (B, C, patch_count * patch_size)
        X_depatch = X_depatch.reshape(B, C, patch_count * patch_size)
        #print(f"After reshape X_depatch shape: {X_depatch.shape}")
        
        # 应用通道注意力机制进行跨通道增强
        X_ca = self.channel_attention(X_depatch)  # 通道重标定
        #print(f"After channel attention X_ca shape: {X_ca.shape}")
        
        # 如果长度不匹配，进行截断或填充
        if X_ca.shape[2] > length:
            X_ca = X_ca[:, :, :length]
        elif X_ca.shape[2] < length:
            padding = torch.zeros(B, C, length - X_ca.shape[2], device=X_ca.device)
            X_ca = torch.cat([X_ca, padding], dim=2)
        #print(f"After padding/truncation X_ca shape: {X_ca.shape}")
        
        # 使用ConvTimeNet处理通道注意力增强后的特征
        # ConvTimeNet期望输入形状: (B, C, L)
        #print(f"ConvTimeNet input shape: {X_ca.shape}")
        
        # 使用ConvTimeNet encoder处理序列特征
        # 首先进行输入嵌入
        X_embed = self.conv_embed(X_ca.transpose(2,1))  # (B, L, hidden_size)
        #print(f"After ConvTimeNet embedding shape: {X_embed.shape}")
        
        # 通过encoder层
        if self.enable_wmt:
            # 启用WMT模式：获取多核输出
            X_conv, multi_kernel_outputs = self.conv_encoder(X_embed.transpose(2,1).contiguous())
            #print(f"ConvTimeNet encoder output shape: {X_conv.shape}")
            #print(f"Multi-kernel outputs count: {len(multi_kernel_outputs)}")
            
            # 获取ConvTimeNet输出的实际序列长度
            conv_seq_len = X_conv.shape[2]
            
            # 应用WMT模块进行工作记忆转移
            ltm_outputs = self.wmt(multi_kernel_outputs, conv_seq_len)
            #print(f"LTM outputs count: {len(ltm_outputs)}")
            
            # 融合长时记忆输出（这里简单取平均）
            if ltm_outputs:
                ltm_fused = torch.stack(ltm_outputs, dim=0).mean(dim=0)  # [B, hidden_size, L]
                # 与原始卷积输出进行残差连接
                X_conv = X_conv + ltm_fused
        else:
            # 原始模式
            X_conv = self.conv_encoder(X_embed.transpose(2,1).contiguous())  # 输出形状: (B, hidden_size, L)
            #print(f"ConvTimeNet encoder output shape: {X_conv.shape}")
        
        # 转置为序列形式 (B, L, hidden_size)
        X_conv = X_conv.permute(0, 2, 1)
        #print(f"After permute X_conv shape: {X_conv.shape}")
        
        # 应用LSTM进行序列建模
        X_lstm, (h_n, c_n) = self.lstm(X_conv)  # LSTM输出形状: (B, L, hidden_size//2)
        #print(f"After LSTM X_lstm shape: {X_lstm.shape}")
        
        # 将LSTM输出投影回hidden_size维度
        X_lstm_proj = self.lstm_projection(X_lstm)  # (B, L, hidden_size)
        #print(f"After LSTM projection shape: {X_lstm_proj.shape}")
        
        # 可选：将ConvTimeNet输出与LSTM输出进行融合（残差连接）
        X_final = X_conv + X_lstm_proj  # 残差连接
        #print(f"After fusion X_final shape: {X_final.shape}")
        
        # 保存知识状态（使用X_final作为知识状态表示）
        self.knowledge_state = X_final
        
        P = self.FCs(X_final)
        #print(f"经过ConvTimeNet+LSTM和FC后：{P.shape}")
        return P



def train(model, data, optimizer, batch_size):
    """
    model: 用于训练的模型
    data: 训练数据
    optimizer: 训练优化器
    batch_size: 训练批量大小
    """
    model.train(mode=True)

    criterion = nn.BCELoss()
    for X, Y, S, Q in DataLoader(
            dataset=data,
            batch_size=batch_size,
            collate_fn=lambda batch: collate(batch, data.total_num_concept),
            shuffle=False
    ):
        # 标准训练：计算原始损失
        P = model(X) # b*l* cn
        # 由于模型内部已经使用embedding，不再需要onehot编码
        # 直接使用Q作为索引，模型会处理embedding
        Q, P, Y, S = Q[:, 1:], P[:, :-1], Y[:, 1:], S[:, 1:]  # P去除最后一列，Q, Y，S去除0列
        #print('Q', Q.shape)
        #print('Y', Y.shape)
        #print('S', S.shape)
        #print('P', P.shape)
        # 使用gather从P中提取对应Q的预测概率
        P = torch.gather(P, 2, Q.unsqueeze(2)).squeeze(2)  # (B, L-1)
        # 按 2 维度求和
        """P.shape =  torch.Size([32, 199])"""

        '''
            只提取真实数据 作loss运算 填充数据不算
            所以通过S转换index矩阵（true,false) ，再P[index]去除。
        '''
        # 计算损失函数
        index = S == 1
        # 注意：模型FCs层已经包含sigmoid激活函数，这里不需要再次应用
        loss = criterion(P[index], Y[index].float())
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate(model, data, batch_size, dataset_name='unknown', save_knowledge_state=False):
    """
    model: the model used for evaluating
    data: the data used for evaluating
    batch_size: the size of evaluate batch
    dataset_name: name of the dataset
    save_knowledge_state: whether to save knowledge state for visualization
    """
    model.eval()  # 加上此函数
    criterion = nn.BCELoss()
    y_pred, y_true = [], []
    loss = 0.0
    
    # 用于保存知识状态和相关数据
    if save_knowledge_state:
        all_Q = []
        all_C = []  # 这里使用Q作为概念标签
        all_R = []
        all_R_pred = []
        all_knowledge_state = []
    
    for X, Y, S, Q in DataLoader(
            dataset=data,
            batch_size=batch_size,
            collate_fn=lambda batch: collate(batch, data.total_num_concept)
    ):
        P = model(X)
        
        # 由于模型内部已经使用embedding，不再需要onehot编码
        Q_original, P_original, Y_original, S_original = Q[:, 1:], P[:, :-1], Y[:, 1:], S[:, 1:]  # 错位
        # 使用gather从P中提取对应Q的预测概率
        P_gathered = torch.gather(P_original, 2, Q_original.unsqueeze(2)).squeeze(2)

        index = S_original == 1
        P_filtered, Y_filtered = P_gathered[index], Y_original[index].float()

        y_pred += detach(P_filtered)
        y_true += detach(Y_filtered)
        # 注意：模型FCs层已经包含sigmoid激活函数，这里不需要再次应用
        # 因为每个batch里真实样本的个数不一样, 所以乘以一个真实样本的个数相当于权重
        loss += detach(criterion(P_filtered, Y_filtered) * P_filtered.shape[0])  # 之后求平均
        
        # 保存知识状态和相关数据
        if save_knowledge_state:
            # 获取知识状态
            knowledge_state = model.knowledge_state
            
            # 对于每个样本，只保存实际有效的部分（根据S的值）
            for i in range(Q.shape[0]):
                # 找到实际有效的长度
                actual_length = int(torch.sum(S[i]).item())
                if actual_length > 0:
                    # 保存实际有效的数据
                    all_Q.append(Q[i, :actual_length].cpu().detach().numpy())
                    all_C.append(Q[i, :actual_length].cpu().detach().numpy())  # 使用Q作为概念标签
                    all_R.append(Y[i, :actual_length].cpu().detach().numpy())
                    all_R_pred.append(P[i, :actual_length].cpu().detach().numpy())
                    all_knowledge_state.append(knowledge_state[i, :actual_length].cpu().detach().numpy())
    
    '''
    fpr 特异性 检测出确实为0的能力 真阳性
    tpr 敏感性 检测出确实为1的能力 真阴性
    thres 阈值
    '''
    fpr, tpr, thres = roc_curve(y_true, y_pred, pos_label=1)
    '''
    mse 均方误差 
    mae 平均绝对误差
    越小越好
    '''
    mse_value = mean_squared_error(y_true, y_pred)
    mae_value = mean_absolute_error(y_true, y_pred)
    bi_y_pred = [1 if i >= 0.5 else 0 for i in y_pred]  # >0.5就是1
    acc_value = accuracy_score(y_true, bi_y_pred)
    
    # 保存知识状态到pickle文件（供julei.py使用）
    if save_knowledge_state:
        import pickle
        if all_knowledge_state:
            # 对于知识状态，我们需要特殊处理，因为每个样本的长度可能不同
            # 我们将保存为列表而不是数组
            save_data = {
                'Q': all_Q,
                'C': all_C,
                'R': all_R,
                'R_pred': all_R_pred,
                'Knowledge_state': all_knowledge_state
            }
            
            # 使用脚本所在目录的绝对路径
            import os
            script_dir = os.path.dirname(os.path.abspath(__file__))
            knowledge_state_save_path = os.path.join(script_dir, f'test_knowledge_state_{dataset_name}.pkl')
            
            with open(knowledge_state_save_path, 'wb') as f:
                pickle.dump(save_data, f)
            print(f"知识状态已保存到 {knowledge_state_save_path}")
    
    return auc(fpr, tpr), loss / len(y_true), mse_value, mae_value, acc_value


if __name__ == '__main__':
    args = parser.parse_args()  # 解析命令行参数，并将解析结果存储在 args 变量中

    ngpus = torch.cuda.device_count()  # ngpus = 2

    if ngpus > 1:
        print("=> Use {} GPUs for training".format(ngpus))
    else:
        print("=> Use GPU: {} for training".format(args.gpu))

    print('Using device:', device)

    if (args.dataset_name in ["assistments09", "assistments12", "assistments15", "assistments17", "Ednet", "eedi", "statics11"]):
        test_data_file = args.dataset_path + '/' + args.dataset_name + '_test_0.csv'
        train_valid_data_file = args.dataset_path + '/' + args.dataset_name + '_train_0.csv'
    elif (args.dataset_name in ["junyi_test", "junyi_all"]):
        test_data_file = args.dataset_path + '/' + args.dataset_name + '/test.csv'
        train_valid_data_file = args.dataset_path + '/' + args.dataset_name + '/train_valid.csv'

    print('=> Loading test_dataset...')
    test_data = Data(open(test_data_file, 'r'), args.seq_len, False, True, None)
    print('>>> test_data.total_num_student <<<', test_data.total_num_student)
    print('>>> test_data.total_num_concept <<<', test_data.total_num_concept)
    print('>>> test_data.total_len_LH <<<', test_data.total_len_LH)

    seed = args.dataset_seed
    set_seed(seed)

    # Store information
    path = './DMKT/result_embedding_dkt/%s' % ('{0:%Y-%m-%d-%H-%M-%S}'.format(datetime.now()))
    os.makedirs(path)
    info_file = open('%s/info.txt' % path, 'w+')
    params_list = (
        'Base Information\n',
        'dataset = %s\n' % args.dataset_name,
        'seed = %s\n' % args.dataset_seed,
        'cv_num = %d\n' % args.cv_num,
        'learning_rate = %f\n' % args.lr,
        'batch_size = %d\n' % args.batch_size,
        'epochs = %d\n' % args.epochs,
        'sequence_length = %d\n' % args.seq_len,
        'optimizer = %s\n' % args.optim,
        'dropout = %f\n' % args.dropout,
        '\nModel Specific Hyperparameters\n',
        'model_type = %s\n' % args.model_type,
        'rnn_type = %s\n' % args.rnn_type,
        'embedding_size = %d\n' % args.embedding_size,
        'hidden_size = %d\n' % args.hidden_size,
        'num_hidden_layers = %d\n' % args.num_hidden_layers,
        '\nKT Specific Hyperparameters\n',
        'total_num_concept = %d\n' % args.total_num_concept,
        'total_num_student = %d\n' % args.total_num_student,
    )
    info_file.write('file_name = DKT-Base\n')
    info_file.write('%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s' % params_list)

    model_list = []
    print("=> Start {}-fold cross validation.".format(args.cv_num))
    #  开始实验 重复  cv_num次
    for cv in range(args.cv_num):
        origin_list = [i for i in range(args.total_num_student)]
        random.seed(cv + 1000)
        index_split = random.sample(origin_list, int(0.25 * len(origin_list)))  # random.sample 随机截取
        random.seed(0)

        print('=> Loading train_dataset...')
        train_data = Data(open(train_valid_data_file, 'r'), args.seq_len, True, False, index_split)
        print('>>> train_data.total_num_student <<<', train_data.total_num_student)
        print('>>> train_data.total_num_concept <<<', train_data.total_num_concept)
        print('>>> train_data.total_len_LH <<<', train_data.total_len_LH)
        print('=> Loading valid_dataset...')
        valid_data = Data(open(train_valid_data_file, 'r'), args.seq_len, False, False, index_split)
        print('>>> valid_data.total_num_student <<<', valid_data.total_num_student)
        print('>>> valid_data.total_num_concept <<<', valid_data.total_num_concept)
        print('>>> valid_data.total_len_LH <<<', valid_data.total_len_LH)

        max_auc = 0.0
        # create models
        print("=> Creating the models...")
        print("Use Model: {} for training".format(args.model_type))
        model = cuda(DKT(train_data.total_num_concept, args.embedding_size, args.hidden_size, args.seq_len))
        print(model)

        # select the optimizer
        print("=> Selecting the optimizer...")
        print("Use optimizer: {} for training".format(args.optim))
        if args.optim == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), args.lr, eps=1e-9, betas=[0.9, 0.98],
                                         weight_decay=args.weight_decay)  # 0.0001
        elif args.optim == 'adamax':
            optimizer = torch.optim.Adamax(model.parameters(), args.lr, eps=1e-9, betas=[0.9, 0.98],
                                           weight_decay=args.weight_decay)  # 0.0001
        elif args.optim == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum,
                                        weight_decay=args.weight_decay)  # 0.01
        elif args.optim == 'rmsprop':
            optimizer = torch.optim.RMSprop(model.parameters(), args.lr, momentum=0.9, eps=1e-10)  # 0.0001
        elif args.optim == 'adagrad':
            optimizer = torch.optim.Adagrad(model.parameters(), args.lr)
        elif args.optim == 'adadelta':
            optimizer = torch.optim.Adadelta(model.parameters(), args.lr)
        '''
            LambdaLR模块提供了一些根据epoch训练次数来调整学习率（learning rate）的方法。
            一般情况下我们会设置随着epoch的增大而逐渐减小学习率从而达到更好的训练效果。
        '''
        lambda1 = lambda epoch: epoch // 30
        lambda2 = lambda epoch: 0.95 ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda2)

        # early stopping
        early_stopping = EarlyStopping(patience=20, verbose=True)

        print('=> Starting {} epochs training...'.format(args.epochs))
        for epoch in range(1, args.epochs + 1):
            time_start = time.time()

            train(model, train_data, optimizer, args.batch_size)
            train_auc, train_loss, train_mse, train_mae, train_acc = evaluate(model, train_data, args.batch_size, dataset_name=args.dataset_name)
            valid_auc, valid_loss, valid_mse, valid_mae, valid_acc = evaluate(model, valid_data, args.batch_size, dataset_name=args.dataset_name)

            time_end = time.time()

            if max_auc < valid_auc:
                max_auc = valid_auc
                torch.save(model.state_dict(), '%s/model_%s' % ('%s' % path, '%d' % cv))
                current_max_model = model
                print('Saving model ...')

            early_stopping(np.average(valid_loss), model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            print_list = (
                'cv:%-3d' % cv,
                'epoch:%-3d' % epoch,
                'max_auc:%-8.4f' % max_auc,
                'valid_auc:%-8.4f' % valid_auc,
                'valid_loss:%-8.4f' % valid_loss,
                'valid_mse:%-8.4f' % valid_mse,
                'valid_mae:%-8.4f' % valid_mae,
                'valid_acc:%-8.4f' % valid_acc,
                'train_auc:%-8.4f' % train_auc,
                'train_loss:%-8.4f' % train_loss,
                'train_mse:%-8.4f' % train_mse,
                'train_mae:%-8.4f' % train_mae,
                'train_acc:%-8.4f' % train_acc,
                'time:%-6.2fs' % (time_end - time_start)
            )

            print('%s %s %s %s %s %s %s %s %s %s %s %s %s %s' % print_list)

            info_file.write('%s %s %s %s %s %s %s %s %s %s %s %s %s %s\n' % print_list)
        model_list.append(current_max_model)
    train_list = []
    auc_list = []
    mse_list = []
    mae_list = []
    acc_list = []
    loss_list = []
    for cv, model_item in enumerate(model_list):
        train_auc, train_loss, train_mse, train_mae, train_acc = evaluate(model_item, train_data, args.batch_size, dataset_name=args.dataset_name)
        # 在测试评估时保存知识状态数据
        test_auc, test_loss, test_mse, test_mae, test_acc = evaluate(model_item, test_data, args.batch_size, 
                                                                     dataset_name=args.dataset_name, save_knowledge_state=True)

        train_list.append(train_auc)
        auc_list.append(test_auc)
        mse_list.append(test_mse)
        mae_list.append(test_mae)
        acc_list.append(test_acc)
        loss_list.append(test_loss)
        print_list_test = (
            'cv:%-3d' % cv,
            'train_auc:%-8.4f' % train_auc,
            'test_auc:%-8.4f' % test_auc,
            'test_mse:%-8.4f' % test_mse,
            'test_mae:%-8.4f' % test_mae,
            'test_acc:%-8.4f' % test_acc,
            'test_loss:%-8.4f' % test_loss
        )

        print('%s %s %s %s %s %s %s\n' % print_list_test)
        info_file.write('%s %s %s %s %s %s %s\n' % print_list_test)
        # 取平均
    average_train_auc = sum(train_list) / len(train_list)
    average_test_auc = sum(auc_list) / len(auc_list)
    average_test_mse = sum(mse_list) / len(mse_list)
    average_test_mae = sum(mae_list) / len(mae_list)
    average_test_acc = sum(acc_list) / len(acc_list)
    average_test_loss = sum(loss_list) / len(loss_list)
    print_result = (
        'average_train_auc:%-8.4f' % average_train_auc,
        'average_test_auc:%-8.4f' % average_test_auc,
        'average_test_mse:%-8.4f' % average_test_mse,
        'average_test_mae:%-8.4f' % average_test_mae,
        'average_test_acc:%-8.4f' % average_test_acc,
        'average_test_loss:%-8.4f' % average_test_loss
    )
    print('%s %s %s %s %s %s\n' % print_result)
    info_file.write('%s %s %s %s %s %s\n' % print_result)
