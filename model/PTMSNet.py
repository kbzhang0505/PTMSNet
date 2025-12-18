## Rethinking progressive low-light image enhancement: A frequency-aware tripartite multi-scale network
## Yingjian Li, Kaibing Zhang, Xuan Zhou, Zhouqiang Zhang, Sheng Hu
## https://doi.org/10.1016/j.neunet.2025.108351
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

import einops
from einops import rearrange
import numpy as np

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
import os

from torchvision.ops import DeformConv2d


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


### Axis-based Multi-head Self-Attention

class NextAttentionImplZ(nn.Module):
    def __init__(self, num_dims, num_heads, bias) -> None:
        super().__init__()
        self.num_dims = num_dims
        self.num_heads = num_heads
        self.q1 = nn.Conv2d(num_dims, num_dims * 3, kernel_size=1, bias=bias)
        self.q2 = nn.Conv2d(num_dims * 3, num_dims * 3, kernel_size=3, padding=1, groups=num_dims * 3, bias=bias)
        self.q3 = nn.Conv2d(num_dims * 3, num_dims * 3, kernel_size=3, padding=1, groups=num_dims * 3, bias=bias)

        self.fac = nn.Parameter(torch.ones(1))
        self.fin = nn.Conv2d(num_dims, num_dims, kernel_size=1, bias=bias)
        return

    def forward(self, x):
        # x: [n, c, h, w]
        n, c, h, w = x.size()
        n_heads, dim_head = self.num_heads, c // self.num_heads
        reshape = lambda x: einops.rearrange(x, "n (nh dh) h w -> (n nh h) w dh", nh=n_heads, dh=dim_head)
        # reshape中的n为batch_size,nh为n_heads注意力头的个数，dh为dim_heads每个头的维度，h，w为高和宽
        qkv = self.q3(self.q2(self.q1(x)))
        q, k, v = map(reshape, qkv.chunk(3, dim=1))
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # fac = dim_head ** -0.5
        res = k.transpose(-2, -1)
        res = torch.matmul(q, res) * self.fac
        res = torch.softmax(res, dim=-1)

        res = torch.matmul(res, v)
        res = einops.rearrange(res, "(n nh h) w dh -> n (nh dh) h w", nh=n_heads, dh=dim_head, n=n, h=h)
        res = self.fin(res)

        return res


### Axis-based Multi-head Self-Attention (row and col attention)
class NextAttentionZ(nn.Module):
    def __init__(self, num_dims, num_heads=1, bias=True) -> None:
        super().__init__()
        assert num_dims % num_heads == 0
        self.num_dims = num_dims
        self.num_heads = num_heads
        self.row_att = NextAttentionImplZ(num_dims, num_heads, bias)
        self.col_att = NextAttentionImplZ(num_dims, num_heads, bias)
        return

    def forward(self, x: torch.Tensor):
        assert len(x.size()) == 4

        x = self.row_att(x)
        x = x.transpose(-2, -1)
        x = self.col_att(x)
        x = x.transpose(-2, -1)

        return x


###### Dual Gated Feed-Forward Networ
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x2) * x1 + F.gelu(x1) * x2
        x = self.project_out(x)
        return x


#########################################################################
##----------------Multi-Head Channel Self-Attention----------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), x.size(1)))
        max_out = self.fc(self.max_pool(x).view(x.size(0), x.size(1)))
        out = avg_out + max_out
        out = self.sigmoid(out).unsqueeze(2).unsqueeze(3)
        return out


class MultiHeadChannelSelfAttention(nn.Module):
    def __init__(self, in_channels, num_heads):
        super(MultiHeadChannelSelfAttention, self).__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        assert self.in_channels % self.num_heads == 0

        self.projection_dim = self.in_channels // self.num_heads

        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.combine_heads = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def separate_heads(self, x, batch_size):
        x = x.view(batch_size, self.num_heads, self.projection_dim, x.size(2), x.size(3))
        return x.permute(0, 1, 3, 4, 2)

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention = torch.matmul(query, key.transpose(-2, -1)) / (self.in_channels ** 0.5)
        attention = F.softmax(attention, dim=-1)

        output = torch.matmul(attention, value)
        output = output.permute(0, 1, 3, 4, 2).contiguous()
        output = output.view(batch_size, self.num_heads, height, width, self.projection_dim)
        output = output.permute(0, 4, 2, 3, 1).contiguous()
        output = output.view(batch_size, self.in_channels, height, width)

        output = self.combine_heads(output)

        return output


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class CNNBlock(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias=True):
        super(CNNBlock, self).__init__()
        self.conv_aspp_layer = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1, bias=bias),
            LayerNorm(dim * 2, LayerNorm_type='WithBias'),
            nn.GELU(),
            DepthwiseSeparableConv2d(dim * 2, dim * 4),
            DepthwiseSeparableConv2d(dim * 4, dim * 4, 5, 1, 2),
            DepthwiseSeparableConv2d(dim * 4, dim * 2),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, 1, bias=bias),
            LayerNorm(dim, LayerNorm_type='WithBias'),
        )

    def forward(self, x):
        x = F.gelu(x + self.conv_aspp_layer(x))
        return x

def match_size(SNR, tensor):
    target_size = tensor.shape[-2:]
    if SNR.shape[-2:] == target_size:
        return SNR

    if SNR.shape[-2] < target_size[0] or SNR.shape[-1] < target_size[1]:
        SNR_resized = F.interpolate(SNR, size=target_size, mode='bilinear', align_corners=False)
    else:
        SNR_resized = F.interpolate(SNR, size=target_size, mode='bilinear', align_corners=False)

    return SNR_resized


######  Parallel Hybrid Module (PHM) 
class PHM(nn.Module):
    def __init__(self, dim, num_heads=1, ffn_expansion_factor=2.66, bias=True, LayerNorm_type='WithBias'):
        super(PHM, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = NextAttentionZ(dim, num_heads)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.conv = CNNBlock(dim, ffn_expansion_factor, bias)

    def forward(self, x, SNR):
        t = x + self.attn(self.norm1(x))
        t = t + self.ffn(self.norm2(t))
        c = self.conv(x)
        SNR = match_size(SNR, c)
        SNR_expanded = SNR.expand(-1, c.shape[1], -1, -1)
        y = SNR_expanded * c + (1 - SNR_expanded) * t
        return y


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)




#### Cross-layer Attention Fusion Block
class LAM_Module_v2(nn.Module):
    """ Layer attention module"""

    def __init__(self, in_dim, bias=True):
        super(LAM_Module_v2, self).__init__()
        self.chanel_in = in_dim

        self.temperature = nn.Parameter(torch.ones(1))

        self.qkv = nn.Conv2d(self.chanel_in, self.chanel_in * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(self.chanel_in * 3, self.chanel_in * 3, kernel_size=3, stride=1, padding=1,
                                    groups=self.chanel_in * 3, bias=bias)
        self.project_out = nn.Conv2d(self.chanel_in, self.chanel_in, kernel_size=1, bias=bias)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, N, C, height, width = x.size()

        x_input = x.view(m_batchsize, N * C, height, width)
        qkv = self.qkv_dwconv(self.qkv(x_input))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(m_batchsize, N, -1)
        k = k.view(m_batchsize, N, -1)
        v = v.view(m_batchsize, N, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out_1 = (attn @ v)
        out_1 = out_1.view(m_batchsize, -1, height, width)

        out_1 = self.project_out(out_1)
        out_1 = out_1.view(m_batchsize, N, C, height, width)

        out = out_1 + x
        out = out.view(m_batchsize, -1, height, width)
        return out


##############################################################################
##--------------------gamma-----------------------
class Mlp(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class query_Attention(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Parameter(torch.ones((1, 1, dim)), requires_grad=True)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = self.q.expand(B, -1, -1).view(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class query_SABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = query_Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class conv_embedding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_embedding, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(out_channels // 2),
            nn.GELU(),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.proj(x)
        return x


##########################################################################
##---------- PTMSNet -----------------------
class PTMSNet(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=16,
                 heads=[1, 2, 4, 8, 16],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 attention=True,
                 ):
        super(PTMSNet, self).__init__()

        self.coefficient = nn.Parameter(torch.Tensor(np.ones((4, 2, int(int(dim * 2 * 4))))),
                                        requires_grad=attention)

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_1 = PHM(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                          bias=bias,
                                          LayerNorm_type=LayerNorm_type)

        self.encoder_2 = PHM(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                          bias=bias,
                                          LayerNorm_type=LayerNorm_type)

        self.encoder_3 = PHM(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                          bias=bias,
                                          LayerNorm_type=LayerNorm_type)


        self.layer_fussion = LAM_Module_v2(in_dim=int(dim * 3))
        self.conv_fuss = nn.Conv2d(int(dim * 3), int(dim), kernel_size=1, bias=bias)

        self.trans_1_1 = PHM(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                          bias=bias, LayerNorm_type=LayerNorm_type)
        self.down_2_1 = Downsample(int(dim))
        self.trans_2_1 = PHM(dim=int(int(dim * 2)), num_heads=heads[1],
                                          ffn_expansion_factor=ffn_expansion_factor,
                                          bias=bias, LayerNorm_type=LayerNorm_type)
        self.coefficient_1_1 = nn.Parameter(torch.Tensor(np.ones((2, int(dim)))), requires_grad=attention)
        self.up_2_1 = Upsample(int(dim * 2))
        self.trans_1_2 = PHM(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                          bias=bias, LayerNorm_type=LayerNorm_type)
        self.down_1_1 = Downsample(int(dim))

        self.down_3_1 = Downsample(int(dim * 2))
        self.trans_3_1 = PHM(dim=int(int(dim * 2 * 2)), num_heads=heads[2],
                                          ffn_expansion_factor=ffn_expansion_factor,
                                          bias=bias, LayerNorm_type=LayerNorm_type)
        self.up_3_1 = Upsample(int(dim * 2 * 2))
        self.coefficient_2_1 = nn.Parameter(torch.Tensor(np.ones((3, int(dim * 2)))), requires_grad=attention)
        self.trans_2_2 = PHM(dim=int(int(dim * 2)), num_heads=heads[1],
                                          ffn_expansion_factor=ffn_expansion_factor,
                                          bias=bias, LayerNorm_type=LayerNorm_type)
        self.up_2_2 = Upsample(int(dim * 2))
        self.trans_1_3 = PHM(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                          bias=bias, LayerNorm_type=LayerNorm_type)
        self.coefficient_1_2 = nn.Parameter(torch.Tensor(np.ones((2, int(dim)))), requires_grad=attention)
        self.down_4_1 = Downsample(int(dim * 2 * 2))
        self.trans_4_11 = PHM(dim=int(int(dim * 2 * 2 * 2)), num_heads=heads[3],
                                          ffn_expansion_factor=ffn_expansion_factor,
                                          bias=bias, LayerNorm_type=LayerNorm_type)
        self.trans_4_12 = PHM(dim=int(int(dim * 2 * 2 * 2)), num_heads=heads[3],
                                          ffn_expansion_factor=ffn_expansion_factor,
                                          bias=bias, LayerNorm_type=LayerNorm_type)
        self.trans_4_13 = PHM(dim=int(int(dim * 2 * 2 * 2)), num_heads=heads[3],
                                          ffn_expansion_factor=ffn_expansion_factor,
                                          bias=bias, LayerNorm_type=LayerNorm_type)
        self.up_4_1 = Upsample(int(dim * 2 * 2 * 2))
        self.coefficient_3_1 = nn.Parameter(torch.Tensor(np.ones((3, int(dim * 2 * 2)))), requires_grad=attention)
        self.down_2_2 = Downsample(int(dim * 2))

        self.trans_3_2 = PHM(dim=int(int(dim * 2 * 2)), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
        self.up_3_2 = Upsample(int(dim * 2 * 2))
        self.coefficient_2_2 = nn.Parameter(torch.Tensor(np.ones((3, int(dim * 2)))), requires_grad=attention)
        self.down_1_2 = Downsample(int(dim))
        self.trans_2_3 = PHM(dim=int(int(dim * 2)), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
        self.up_2_3 = Upsample(int(dim * 2))
        self.coefficient_1_3 = nn.Parameter(torch.Tensor(np.ones((2, int(dim)))), requires_grad=attention)
        self.down_2_3 = Downsample(int(dim * 2))
        self.down_1_3 = Downsample(int(dim))
        self.coefficient_3_2 = nn.Parameter(torch.Tensor(np.ones((2, int(dim * 2 * 2)))), requires_grad=attention)
        self.trans_1_4 = PHM(dim=int(int(dim)), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
        self.down_4_2 = Downsample(int(dim * 2 * 2))
        self.coefficient_2_3 = nn.Parameter(torch.Tensor(np.ones((2, int(dim * 2)))), requires_grad=attention)
        self.coefficient_4_1 = nn.Parameter(torch.Tensor(np.ones((2, int(dim * 2 * 2 * 2)))), requires_grad=attention)

        self.up_out_1 = Upsample(int(dim * 2))
        self.conv_out_1 = nn.Sequential(
            LAM_Module_v2(in_dim=int(dim * 2)),
            nn.Conv2d(int(dim * 2), int(dim), 1)
        )
        self.up_out_2 = Upsample(int(dim * 2 * 2))
        self.conv_out_2 = nn.Sequential(
            LAM_Module_v2(in_dim=int(dim * 2 * 2)),
            nn.Conv2d(int(dim * 2 * 2), int(dim * 2), 1)
        )
        self.up_out_3 = Upsample(int(dim * 2 * 2 * 2))
        self.conv_out_3 = nn.Sequential(
            LAM_Module_v2(in_dim=int(dim * 2 * 2 * 2)),
            nn.Conv2d(int(dim * 2 * 2 * 2), int(dim * 2 * 2), 1)
        )

        self.layer_fussion_out = LAM_Module_v2(in_dim=int(dim * 2))
        self.conv_fuss_out = nn.Conv2d(int(dim * 2), int(dim), 3, 1, 1, bias=bias)
        self.conv_fuss_out_end = nn.Sequential(nn.Conv2d(int(dim * 3), int(dim), 1, bias=bias),
                                               nn.Conv2d(int(dim), out_channels, 3, 1, 1, bias=bias))

        self.dencoder_1 = PHM(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                          bias=bias, LayerNorm_type=LayerNorm_type)

        self.dencoder_2 = PHM(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                          bias=bias, LayerNorm_type=LayerNorm_type)

        self.dencoder_3 = PHM(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                          bias=bias, LayerNorm_type=LayerNorm_type)
        self.layer_fussion_end_out = LAM_Module_v2(in_dim=int(dim * 3))

    def forward(self, inp_img):
        x, snr = inp_img[0], inp_img[1]
        inp_enc_encoder1 = self.patch_embed(x)
        out_enc_encoder1 = self.encoder_1(inp_enc_encoder1, snr)
        out_enc_encoder2 = self.encoder_2(out_enc_encoder1, snr)
        out_enc_encoder3 = self.encoder_3(out_enc_encoder2, snr)

        inp_fusion_123 = torch.cat(
            [out_enc_encoder1.unsqueeze(1), out_enc_encoder2.unsqueeze(1), out_enc_encoder3.unsqueeze(1)], dim=1)

        out_fusion_123 = self.layer_fussion(inp_fusion_123)
        out_fusion_123 = self.conv_fuss(out_fusion_123)

        sub_network_1_1 = self.trans_1_1(out_fusion_123, snr)
        sub_network_2_in = self.down_2_1(out_fusion_123)
        sub_network_2_1 = self.trans_2_1(sub_network_2_in, snr)
        sub_network_1_1 = self.coefficient_1_1[0, :][None, :, None, None] * sub_network_1_1 + self.coefficient_1_1[1,
                                                                                              :][None, :, None,
                                                                                              None] * self.up_2_1(
            sub_network_2_1)
        sub_network_1_2 = self.trans_1_2(sub_network_1_1, snr)

        sub_network_3_in = self.down_3_1(sub_network_2_in)
        sub_network_3_1 = self.trans_3_1(sub_network_3_in, snr)
        sub_network_2_1 = self.coefficient_2_1[0, :][None, :, None, None] * self.down_1_1(
            sub_network_1_2) + self.coefficient_2_1[1, :][None,
                               :, None, None] * sub_network_2_1 + self.coefficient_2_1[2, :][None,
                                                                  :, None, None] * self.up_3_1(sub_network_3_1)
        sub_network_2_2 = self.trans_2_2(sub_network_2_1, snr)
        sub_network_1_2 = self.coefficient_1_2[0, :][None, :, None, None] * sub_network_1_2 + self.coefficient_1_2[1,
                                                                                              :][None, :, None,
                                                                                              None] * self.up_2_2(
            sub_network_2_2)
        sub_network_1_3 = self.trans_1_3(sub_network_1_2, snr)

        sub_network_4_in = self.down_4_1(sub_network_3_in)
        sub_network_4_11 = self.trans_4_11(sub_network_4_in, snr)
        sub_network_4_12 = self.trans_4_12(sub_network_4_11, snr)
        sub_network_4_13 = self.trans_4_13(sub_network_4_12, snr)
        sub_network_3_1 = self.coefficient_3_1[0, :][None, :, None, None] * self.down_2_2(
            sub_network_2_2) + self.coefficient_3_1[1, :][None, :, None, None] * sub_network_3_1 + self.coefficient_3_1[
                                                                                                   2, :][None, :, None,
                                                                                                   None] * self.up_4_1(
            sub_network_4_13)
        sub_network_3_2 = self.trans_3_2(sub_network_3_1, snr)
        sub_network_2_2 = self.coefficient_2_2[0, :][None, :, None, None] * self.down_1_2(
            sub_network_1_3) + self.coefficient_2_2[1, :][None, :, None, None] * sub_network_2_2 + self.coefficient_2_2[
                                                                                                   2, :][None, :, None,
                                                                                                   None] * self.up_3_2(
            sub_network_3_2)
        sub_network_2_3 = self.trans_2_3(sub_network_2_2, snr)
        sub_network_1_3 = self.coefficient_1_3[0, :][None, :, None, None] * sub_network_1_3 + self.coefficient_1_3[1,
                                                                                              :][None, :, None,
                                                                                              None] * self.up_2_3(
            sub_network_2_3)
        sub_network_1_out = self.trans_1_4(sub_network_1_3, snr)
        sub_network_2_out = self.coefficient_2_3[0, :][None, :, None, None] * self.down_1_3(
            sub_network_1_out) + self.coefficient_2_3[1, :][None, :, None, None] * sub_network_2_3
        sub_network_4_out = self.coefficient_4_1[0, :][None, :, None, None] * self.down_4_2(
            sub_network_3_2) + self.coefficient_4_1[1, :][None, :, None, None] * sub_network_4_13
        sub_network_3_out = self.coefficient_3_2[0, :][None, :, None, None] * self.down_2_3(
            sub_network_2_3) + self.coefficient_3_2[1, :][None, :, None, None] * sub_network_3_2

        out_fusion_3 = torch.cat([self.up_out_3(sub_network_4_out).unsqueeze(1), sub_network_3_out.unsqueeze(1)], dim=1)
        out_fusion_3 = self.conv_out_3(out_fusion_3)
        out_fusion_2 = self.conv_out_2(
            torch.cat([self.up_out_2(out_fusion_3).unsqueeze(1), sub_network_2_out.unsqueeze(1)], dim=1))
        out_fusion_1 = self.conv_out_1(
            torch.cat([self.up_out_1(out_fusion_2).unsqueeze(1), sub_network_1_out.unsqueeze(1)], dim=1))

        out_fusion = torch.cat(
            [out_fusion_1.unsqueeze(1), out_fusion_123.unsqueeze(1)], dim=1)
        out_fusion = self.layer_fussion_out(out_fusion)
        out = self.conv_fuss_out(out_fusion)

        out_denc_dencoder1 = self.dencoder_1(out, snr)
        out_denc_dencoder2 = self.dencoder_2(out_denc_dencoder1, snr)
        out_denc_dencoder3 = self.dencoder_3(out_denc_dencoder2, snr)

        inp_fusion_end = torch.cat(
            [out_denc_dencoder1.unsqueeze(1), out_denc_dencoder2.unsqueeze(1), out_denc_dencoder3.unsqueeze(1)], dim=1)
        out_fusion_end = self.layer_fussion_end_out(inp_fusion_end)
        out = self.conv_fuss_out_end(out_fusion_end)

        return out




if __name__ == '__main__':
    batch_size = 4
    channels = 3
    height = 256
    width = 256
    image_tensor = torch.rand((batch_size, channels, height, width)).to('cuda')

