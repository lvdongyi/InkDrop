import torch
import torch.nn as nn
import torch.nn.functional as F
import time


def get_network(model, channel, num_classes, net_norm=None, img_size=(32, 32), embedding_dim=None, num_layers=None):
    """
    获取网络模型
    
    Args:
        model: 模型名称
        channel: 输入通道数
        num_classes: 类别数
        net_norm: 归一化方式
        img_size: 图像尺寸
        embedding_dim: CCT模型的嵌入维度（可选，默认256）
        num_layers: CCT模型的Transformer层数（可选，默认7）
    """
    torch.random.manual_seed(int(time.time() * 1000) % 100000)
    net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()

    if model == 'MLP':
        net = MLP(channel=channel, num_classes=num_classes)
    elif model == 'ConvNet':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                      net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=img_size)
    elif model == 'ConvNetMask':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                        net_act=net_act, net_norm='maskbatchnorm', net_pooling=net_pooling, im_size=img_size)
    elif model == 'LeNet':
        net = LeNet(channel=channel, num_classes=num_classes)
    elif model == 'AlexNet':
        net = AlexNet(channel=channel, num_classes=num_classes)
    elif model == 'AlexNetBN':
        net = AlexNetBN(channel=channel, num_classes=num_classes, image_size=img_size)
    elif model == 'VGG11':
        net = VGG11(channel=channel, num_classes=num_classes, image_size=img_size)
    elif model == 'VGG11BN':
        net = VGG11BN(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18':
        net = ResNet18(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18BN_AP':
        net = ResNet18BN_AP(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18BN':
        net = ResNet18BN(channel=channel, num_classes=num_classes)

    elif model == 'ConvNetD1':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=1, net_act=net_act,
                      net_norm=net_norm, net_pooling=net_pooling, im_size=img_size)
    elif model == 'ConvNetD2':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=2, net_act=net_act,
                      net_norm=net_norm, net_pooling=net_pooling, im_size=img_size)
    elif model == 'ConvNetD3':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=3, net_act=net_act,
                      net_norm=net_norm, net_pooling=net_pooling, im_size=img_size)
    elif model == 'ConvNetD4':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=4, net_act=net_act,
                      net_norm=net_norm, net_pooling=net_pooling, im_size=img_size)

    elif model == 'ConvNetW32':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=32, net_depth=net_depth, net_act=net_act,
                      net_norm=net_norm, net_pooling=net_pooling, im_size=img_size)
    elif model == 'ConvNetW64':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=64, net_depth=net_depth, net_act=net_act,
                      net_norm=net_norm, net_pooling=net_pooling, im_size=img_size)
    elif model == 'ConvNetW128':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=128, net_depth=net_depth, net_act=net_act,
                      net_norm=net_norm, net_pooling=net_pooling, im_size=img_size)
    elif model == 'ConvNetW256':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=256, net_depth=net_depth, net_act=net_act,
                      net_norm=net_norm, net_pooling=net_pooling, im_size=img_size)

    elif model == 'ConvNetAS':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                      net_act='sigmoid', net_norm=net_norm, net_pooling=net_pooling, im_size=img_size)
    elif model == 'ConvNetAR':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                      net_act='relu', net_norm=net_norm, net_pooling=net_pooling, im_size=img_size)
    elif model == 'ConvNetAL':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                      net_act='leakyrelu', net_norm=net_norm, net_pooling=net_pooling, im_size=img_size)
    elif model == 'ConvNetASwish':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                      net_act='swish', net_norm=net_norm, net_pooling=net_pooling, im_size=img_size)
    elif model == 'ConvNetASwishBN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                      net_act='swish', net_norm='batchnorm', net_pooling=net_pooling, im_size=img_size)

    elif model == 'ConvNetNN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                      net_act=net_act, net_norm='none', net_pooling=net_pooling, im_size=img_size)
    elif model == 'ConvNetBN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                      net_act=net_act, net_norm='batchnorm', net_pooling=net_pooling, im_size=img_size)
    elif model == 'ConvNetLN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                      net_act=net_act, net_norm='layernorm', net_pooling=net_pooling, im_size=img_size)
    elif model == 'ConvNetIN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                      net_act=net_act, net_norm='instancenorm', net_pooling=net_pooling, im_size=img_size)
    elif model == 'ConvNetGN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                      net_act=net_act, net_norm='groupnorm', net_pooling=net_pooling, im_size=img_size)

    elif model == 'ConvNetNP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                      net_act=net_act, net_norm=net_norm, net_pooling='none', im_size=img_size)
    elif model == 'ConvNetMP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                      net_act=net_act, net_norm=net_norm, net_pooling='maxpooling', im_size=img_size)
    elif model == 'ConvNetAP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                      net_act=net_act, net_norm=net_norm, net_pooling='avgpooling', im_size=img_size)
    elif model == 'CCT':
        # 使用传入的参数，如果未指定则使用默认值
        cct_embedding_dim = embedding_dim if embedding_dim is not None else 256
        cct_num_layers = num_layers if num_layers is not None else 7
        net = CCT(n_input_channels=channel, num_classes=num_classes, img_size=img_size[0],
                  embedding_dim=cct_embedding_dim, num_layers=cct_num_layers)

    else:
        net = None
        exit('unknown model: %s' % model)

    return net

# This code is based on:
# https://github.com/csdongxian/ANP_backdoor/blob/main/models/anp_batchnorm.py

from torch import Tensor
import torch.nn.init as init
from torch.nn.parameter import Parameter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import math

# ==========================================
# 1. CCT 模型定义 (Model Definition)
# ==========================================

class Tokenizer(nn.Module):
    """
    卷积 Tokenizer：替代 ViT 中的 Patch Embedding。
    使用卷积层提取低级特征，引入归纳偏置，适合小数据集。
    """
    def __init__(self, kernel_size, stride, padding, pooling_kernel, pooling_stride,
                 pooling_padding, n_conv_layers, n_input_channels, n_output_channels,
                 in_planes=64, activation=None, max_pool=True):
        super(Tokenizer, self).__init__()

        n_filter_list = [n_input_channels] + \
                        [in_planes for _ in range(n_conv_layers - 1)] + \
                        [n_output_channels]

        self.conv_layers = nn.Sequential()
        for i in range(n_conv_layers):
            self.conv_layers.add_module(
                f'conv_{i}',
                nn.Conv2d(n_filter_list[i], n_filter_list[i + 1],
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding, bias=False)
            )
            self.conv_layers.add_module(f'relu_{i}', nn.ReLU())
            if max_pool:
                self.conv_layers.add_module(
                    f'maxpool_{i}',
                    nn.MaxPool2d(kernel_size=pooling_kernel,
                                 stride=pooling_stride,
                                 padding=pooling_padding)
                )

        self.flattener = nn.Flatten(2, 3) # (B, C, H, W) -> (B, C, H*W)

    def forward(self, x):
        return self.flattener(self.conv_layers(x)).transpose(1, 2) # 输出 (B, Seq_Len, Dim)


class Attention(nn.Module):
    """标准的 Multi-Head Self-Attention，使用分块计算以节省内存"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., chunk_size=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.chunk_size = chunk_size  # 分块大小，None表示不分块

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # qkv shape: (B, N, 3, Heads, Head_Dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 如果序列长度较大，使用分块注意力以节省内存
        if self.chunk_size is not None and N > self.chunk_size:
            # 分块计算注意力
            chunk_size = self.chunk_size
            num_chunks = (N + chunk_size - 1) // chunk_size
            outputs = []
            
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, N)
                q_chunk = q[:, :, start_idx:end_idx, :]  # (B, num_heads, chunk_size, head_dim)
                
                # 计算注意力分数: (B, num_heads, chunk_size, N)
                attn_chunk = (q_chunk @ k.transpose(-2, -1)) * self.scale
                attn_chunk = attn_chunk.softmax(dim=-1)
                attn_chunk = self.attn_drop(attn_chunk)
                
                # 应用注意力到value: (B, num_heads, chunk_size, head_dim)
                out_chunk = attn_chunk @ v
                outputs.append(out_chunk)
            
            # 合并所有chunk: (B, num_heads, N, head_dim)
            x = torch.cat(outputs, dim=2)
        else:
            # 原始计算方式（当序列长度较小时）
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerEncoderLayer(nn.Module):
    """Transformer Encoder Block: Attention + MLP"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., chunk_size=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, chunk_size=chunk_size)
        self.norm2 = nn.LayerNorm(dim)
        
        hidden_features = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_features),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_features, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# class CCT(nn.Module):
#     def __init__(self, 
#                  img_size=32, 
#                  embedding_dim=256, 
#                  n_input_channels=3, 
#                  n_conv_layers=1, 
#                  kernel_size=3, 
#                  stride=1, 
#                  padding=1, 
#                  pooling_kernel=3, 
#                  pooling_stride=2, 
#                  pooling_padding=1,
#                  num_layers=7, 
#                  num_heads=4, 
#                  mlp_ratio=2., 
#                  num_classes=10,
#                  positional_embedding='learnable'):
#         super(CCT, self).__init__()

#         # 1. Tokenizer (卷积层)
#         self.tokenizer = Tokenizer(n_conv_layers=n_conv_layers,
#                                    n_input_channels=n_input_channels,
#                                    n_output_channels=embedding_dim,
#                                    kernel_size=kernel_size,
#                                    stride=stride,
#                                    padding=padding,
#                                    pooling_kernel=pooling_kernel,
#                                    pooling_stride=pooling_stride,
#                                    pooling_padding=pooling_padding)

#         # 2. Positional Embedding
#         # 计算 Tokenizer 输出后的序列长度
#         # 简单估算：对于 CIFAR-10 (32x32)，经过一次 stride=2 的池化，变为 16x16 = 256
#         self.sequence_length = self._get_sequence_length(img_size, n_input_channels)
        
#         # 根据序列长度自动设置chunk_size以节省内存
#         # 对于大图像（如STL-10的96x96），序列长度可能超过2000，使用分块注意力
#         if self.sequence_length > 1000:
#             # 设置chunk_size为512，这样可以显著减少内存使用
#             attn_chunk_size = 512
#         elif self.sequence_length > 500:
#             attn_chunk_size = 256
#         else:
#             # 小图像不需要分块
#             attn_chunk_size = None
        
#         if positional_embedding == 'learnable':
#             self.position_embedding = nn.Parameter(torch.zeros(1, self.sequence_length, embedding_dim))
#             nn.init.trunc_normal_(self.position_embedding, std=0.02)
#         else:
#             # 也可以实现正弦位置编码，这里简化为可学习
#             self.position_embedding = nn.Parameter(torch.zeros(1, self.sequence_length, embedding_dim))

#         self.dropout = nn.Dropout(p=0.1)

#         # 3. Transformer Encoder
#         self.blocks = nn.ModuleList([
#             TransformerEncoderLayer(dim=embedding_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, chunk_size=attn_chunk_size)
#             for _ in range(num_layers)
#         ])
        
#         self.norm = nn.LayerNorm(embedding_dim)

#         # 4. Sequence Pooling (CCT 特有的分类头)
#         # 将所有 Token 加权平均，而不是只取 [CLS] Token
#         self.attention_pool = nn.Linear(embedding_dim, 1)
#         self.fc = nn.Linear(embedding_dim, num_classes)

#     def _get_sequence_length(self, img_size, ch):
#         # 辅助函数：通过跑一次假数据来确定序列长度
#         dummy = torch.zeros(1, ch, img_size, img_size)
#         out = self.tokenizer(dummy)
#         return out.shape[1]

#     def forward(self, x):
#         # x: (B, C, H, W)
#         x = self.tokenizer(x)  # (B, N, Dim)
        
#         # Add Position Embedding
#         x = x + self.position_embedding
#         x = self.dropout(x)

#         # Transformer Blocks
#         for blk in self.blocks:
#             x = blk(x)
        
#         x = self.norm(x)

#         # Sequence Pooling
#         # x shape: (B, N, Dim)
#         # attention_pool 输出: (B, N, 1) -> softmax -> (B, N, 1)
#         attn_weights = F.softmax(self.attention_pool(x), dim=1)
#         # 加权平均: (B, N, 1)^T * (B, N, Dim) -> (B, 1, N) * (B, N, Dim) -> (B, 1, Dim)
#         x = torch.matmul(attn_weights.transpose(1, 2), x).squeeze(1) # (B, Dim)

#         # Classifier
#         x = self.fc(x)
#         return x
#     def embed(self, x):
#         """
#         提取特征嵌入（不经过分类头）
#         返回: (B, embedding_dim) 特征向量
#         """
#         # x: (B, C, H, W)
#         x = self.tokenizer(x)  # (B, N, Dim)
        
#         # Add Position Embedding
#         x = x + self.position_embedding
#         x = self.dropout(x)

#         # Transformer Blocks
#         for blk in self.blocks:
#             x = blk(x)
        
#         x = self.norm(x)

#         # Sequence Pooling（与 forward 相同，但不经过分类头）
#         attn_weights = F.softmax(self.attention_pool(x), dim=1)
#         x = torch.matmul(attn_weights.transpose(1, 2), x).squeeze(1) # (B, Dim)
        
#         return x  # 返回 (B, embedding_dim)，不经过 self.fc

class CCT(nn.Module):
    def __init__(self, 
                 img_size=32, 
                 embedding_dim=128, 
                 n_input_channels=3, 
                 n_conv_layers=2, 
                 kernel_size=3, 
                 stride=1, 
                 padding=1, 
                 pooling_kernel=3, 
                 pooling_stride=2, 
                 pooling_padding=1,
                 num_layers=4, 
                 num_heads=2, 
                 mlp_ratio=2, 
                 num_classes=10,
                 positional_embedding='learnable'):
        super(CCT, self).__init__()

        # 1. Tokenizer (卷积层)
        self.tokenizer = Tokenizer(n_conv_layers=n_conv_layers,
                                   n_input_channels=n_input_channels,
                                   n_output_channels=embedding_dim,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   pooling_kernel=pooling_kernel,
                                   pooling_stride=pooling_stride,
                                   pooling_padding=pooling_padding)

        # 2. Positional Embedding
        # 计算 Tokenizer 输出后的序列长度
        # 简单估算：对于 CIFAR-10 (32x32)，经过一次 stride=2 的池化，变为 16x16 = 256
        self.sequence_length = self._get_sequence_length(img_size, n_input_channels)
        
        # 根据序列长度自动设置chunk_size以节省内存
        # 对于大图像（如STL-10的96x96），序列长度可能超过2000，使用分块注意力
        if self.sequence_length > 1000:
            # 设置chunk_size为512，这样可以显著减少内存使用
            attn_chunk_size = 512
        elif self.sequence_length > 500:
            attn_chunk_size = 256
        else:
            # 小图像不需要分块
            attn_chunk_size = None
        
        if positional_embedding == 'learnable':
            self.position_embedding = nn.Parameter(torch.zeros(1, self.sequence_length, embedding_dim))
            nn.init.trunc_normal_(self.position_embedding, std=0.02)
        else:
            # 也可以实现正弦位置编码，这里简化为可学习
            self.position_embedding = nn.Parameter(torch.zeros(1, self.sequence_length, embedding_dim))

        self.dropout = nn.Dropout(p=0.1)

        # 3. Transformer Encoder
        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(dim=embedding_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, chunk_size=attn_chunk_size)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embedding_dim)

        # 4. Sequence Pooling (CCT 特有的分类头)
        # 将所有 Token 加权平均，而不是只取 [CLS] Token
        self.attention_pool = nn.Linear(embedding_dim, 1)
        self.fc = nn.Linear(embedding_dim, num_classes)

    def _get_sequence_length(self, img_size, ch):
        # 辅助函数：通过跑一次假数据来确定序列长度
        dummy = torch.zeros(1, ch, img_size, img_size)
        out = self.tokenizer(dummy)
        return out.shape[1]

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.tokenizer(x)  # (B, N, Dim)
        
        # Add Position Embedding
        x = x + self.position_embedding
        x = self.dropout(x)

        # Transformer Blocks
        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)

        # Sequence Pooling
        # x shape: (B, N, Dim)
        # attention_pool 输出: (B, N, 1) -> softmax -> (B, N, 1)
        attn_weights = F.softmax(self.attention_pool(x), dim=1)
        # 加权平均: (B, N, 1)^T * (B, N, Dim) -> (B, 1, N) * (B, N, Dim) -> (B, 1, Dim)
        x = torch.matmul(attn_weights.transpose(1, 2), x).squeeze(1) # (B, Dim)

        # Classifier
        x = self.fc(x)
        return x
    def embed(self, x):
        """
        提取特征嵌入（不经过分类头）
        返回: (B, embedding_dim) 特征向量
        """
        # x: (B, C, H, W)
        x = self.tokenizer(x)  # (B, N, Dim)
        
        # Add Position Embedding
        x = x + self.position_embedding
        x = self.dropout(x)

        # Transformer Blocks
        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)

        # Sequence Pooling（与 forward 相同，但不经过分类头）
        attn_weights = F.softmax(self.attention_pool(x), dim=1)
        x = torch.matmul(attn_weights.transpose(1, 2), x).squeeze(1) # (B, Dim)
        
        return x  # 返回 (B, embedding_dim)，不经过 self.fc

class MaskBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(MaskBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.neuron_mask = Parameter(torch.Tensor(num_features))
        self.neuron_noise = Parameter(torch.Tensor(num_features))
        self.neuron_noise_bias = Parameter(torch.Tensor(num_features))
        init.ones_(self.neuron_mask)

    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        assert self.running_mean is None or isinstance(self.running_mean, torch.Tensor)
        assert self.running_var is None or isinstance(self.running_var, torch.Tensor)

        coeff_weight = self.neuron_mask
        coeff_bias = 1.0

        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight * coeff_weight, self.bias * coeff_bias,
            bn_training, exponential_average_factor, self.eps)



def get_default_convnet_setting():
    net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
    return net_width, net_depth, net_act, net_norm, net_pooling


"""
MLP    ConvNet    LeNet    AlexNet    AlexNetBN    VGG    ResNet
"""


# Swish activation
class Swish(nn.Module):  # Swish(x) = x∗σ(x)
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.sigmoid(input)


# MLP
class MLP(nn.Module):
    def __init__(self, channel, num_classes):
        super(MLP, self).__init__()
        self.fc_1 = nn.Linear(28 * 28 * 1 if channel == 1 else 32 * 32 * 3, 128)
        self.fc_2 = nn.Linear(128, 128)
        self.fc_3 = nn.Linear(128, num_classes)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = F.relu(self.fc_1(out))
        out = F.relu(self.fc_2(out))
        out = self.fc_3(out)
        return out


# ConvNet
class ConvNet(nn.Module):
    def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size=(32, 32)):
        super(ConvNet, self).__init__()

        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling,
                                                      im_size)
        num_feat = shape_feat[0] * shape_feat[1] * shape_feat[2]
        # self.classifier = nn.Linear(num_feat, num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(num_feat, 192),
            nn.ReLU(inplace=True),
            nn.Linear(192, num_classes),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.reshape(out.size(0), -1)
        out = self.classifier(out)
        return out

    def embed(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out
    
    def embedding(self,x):
        out = self.features(x)  # [B, C, H, W]
        out = out.view(out.size(0), -1)  # [B, C]
        return out

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        elif net_act == 'swish':
            return Swish()
        else:
            exit('unknown activation function: %s' % net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s' % net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'maskbatchnorm':
            return MaskBatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s' % net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2

        return nn.Sequential(*layers), shape_feat


# LeNet
class LeNet(nn.Module):
    def __init__(self, channel, num_classes):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channel, 6, kernel_size=5, padding=2 if channel == 1 else 0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_1 = nn.Linear(16 * 5 * 5, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = self.fc_3(x)
        return x


# AlexNet
class AlexNet(nn.Module):
    def __init__(self, channel, num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channel, 128, kernel_size=5, stride=1, padding=4 if channel == 1 else 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(192 * 4 * 4, 192),
            nn.ReLU(inplace=True),
            nn.Linear(192, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def embed(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


# AlexNetBN
class AlexNetBN(nn.Module):
    def __init__(self, channel, num_classes, image_size=None):
        super(AlexNetBN, self).__init__()
        
        if image_size is None:
            image_size = (28, 28) if channel == 1 else (32, 32)

        self.features = nn.Sequential(
            nn.Conv2d(channel, 128, kernel_size=5, stride=1, padding=4 if channel == 1 else 2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, channel, *image_size)
            dummy_output = self.features(dummy_input)
            flattened_size = dummy_output.view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Linear(flattened_size, 192),
            nn.ReLU(),
            nn.Linear(192, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

    def embed(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        return x


# VGG
cfg_vgg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, channel, num_classes, image_size=(32, 32), norm='instancenorm'):
        super(VGG, self).__init__()
        self.channel = channel
        self.features = self._make_layers(cfg_vgg[vgg_name], norm)
        
        # 使用 dummy 输入自动推断 features 输出维度
        with torch.no_grad():
            dummy_input = torch.zeros(1, channel, *image_size)
            out = self.features(dummy_input)
            self.feature_dim = out.view(1, -1).size(1)

        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def embed(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

    def _make_layers(self, cfg, norm):
        layers = []
        in_channels = self.channel
        for ic, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=3 if self.channel == 1 and ic == 0 else 1),
                    nn.GroupNorm(x, x, affine=True) if norm == 'instancenorm' else nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True)
                ]
                in_channels = x
        layers += [nn.AdaptiveAvgPool2d((1, 1))]  # 更泛化的池化方式
        return nn.Sequential(*layers)


def VGG11(channel, num_classes, image_size):
    return VGG('VGG11', channel, num_classes, image_size, norm='instancenorm')


def VGG11BN(channel, num_classes):
    return VGG('VGG11', channel, num_classes, norm='batchnorm')


def VGG13(channel, num_classes):
    return VGG('VGG13', channel, num_classes)


def VGG16(channel, num_classes):
    return VGG('VGG16', channel, num_classes)


def VGG19(channel, num_classes):
    return VGG('VGG19', channel, num_classes)


# ResNet_AP
# The conv(stride=2) is replaced by conv(stride=1) + avgpool(kernel_size=2, stride=2)
class BasicBlock_AP(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        super(BasicBlock_AP, self).__init__()
        self.norm = norm
        self.stride = stride
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)  # modification
        self.bn1 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=1, bias=False),
                nn.AvgPool2d(kernel_size=2, stride=2),  # modification
                nn.GroupNorm(self.expansion * planes, self.expansion * planes,
                             affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.stride != 1:  # modification
            out = F.avg_pool2d(out, kernel_size=2, stride=2)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck_AP(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        super(Bottleneck_AP, self).__init__()
        self.norm = norm
        self.stride = stride
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)  # modification
        self.bn2 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(self.expansion * planes, self.expansion * planes,
                                affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=1, bias=False),
                nn.AvgPool2d(kernel_size=2, stride=2),  # modification
                nn.GroupNorm(self.expansion * planes, self.expansion * planes,
                             affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        if self.stride != 1:  # modification
            out = F.avg_pool2d(out, kernel_size=2, stride=2)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_AP(nn.Module):
    def __init__(self, block, num_blocks, channel=3, num_classes=10, norm='instancenorm'):
        super(ResNet_AP, self).__init__()
        self.in_planes = 64
        self.norm = norm

        self.conv1 = nn.Conv2d(channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(64, 64, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.classifier = nn.Linear(512 * block.expansion * 3 * 3 if channel == 1 else 512 * block.expansion * 4 * 4,
                                    num_classes)  # modification

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.norm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, kernel_size=1, stride=1)  # modification
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def embed(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, kernel_size=1, stride=1)  # modification
        out = out.view(out.size(0), -1)
        return out


def ResNet18BN_AP(channel, num_classes):
    return ResNet_AP(BasicBlock_AP, [2, 2, 2, 2], channel=channel, num_classes=num_classes, norm='batchnorm')


def ResNet18_AP(channel, num_classes):
    return ResNet_AP(BasicBlock_AP, [2, 2, 2, 2], channel=channel, num_classes=num_classes)


# ResNet
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        super(BasicBlock, self).__init__()
        self.norm = norm
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(self.expansion * planes, self.expansion * planes,
                             affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        super(Bottleneck, self).__init__()
        self.norm = norm
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(self.expansion * planes, self.expansion * planes,
                                affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(self.expansion * planes, self.expansion * planes,
                             affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, channel=3, num_classes=10, norm='instancenorm'):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.norm = norm

        self.conv1 = nn.Conv2d(channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(64, 64, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.classifier = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.norm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        out = F.adaptive_avg_pool2d(out, (1,1))
        feature = out.view(out.size(0), -1)
        out = self.classifier(feature)
        return out

    def embed(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        out = F.adaptive_avg_pool2d(out, (1,1))
        feature = out.view(out.size(0), -1)
        return feature


def ResNet18BN(channel, num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], channel=channel, num_classes=num_classes, norm='batchnorm')


def ResNet18(channel, num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], channel=channel, num_classes=num_classes)


def ResNet34(channel, num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3], channel=channel, num_classes=num_classes)


def ResNet50(channel, num_classes):
    return ResNet(Bottleneck, [3, 4, 6, 3], channel=channel, num_classes=num_classes)


def ResNet101(channel, num_classes):
    return ResNet(Bottleneck, [3, 4, 23, 3], channel=channel, num_classes=num_classes)


def ResNet152(channel, num_classes):
    return ResNet(Bottleneck, [3, 8, 36, 3], channel=channel, num_classes=num_classes)

if __name__ == '__main__':
    net = get_network('ConvNet', 3, 10,img_size=(96, 96))
    sd = torch.load('/home/user009/CODE/FedDOGE/results/dm/stl10/0606112702/consumer_model_mask_9000.pth')
    
    net.load_state_dict(sd)
    print(net)
    x = torch.randn(256, 3, 96, 96)
    print(net(x).shape)