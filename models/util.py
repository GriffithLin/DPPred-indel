#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/8/3 21:54
# @Author : fhh
# @FileName: util.py
# @Software: PyCharm

import math
import torch
from torch import nn
from torch.nn.utils import weight_norm


class AddNorm(nn.Module):
    """残差连接后进行层归一化"""

    def __init__(self, normalized, dropout):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized)

    def forward(self, X, y):
        return self.ln(self.dropout(y) + X)


class PositionWiseFFN(nn.Module):
    """基于位置的前馈⽹络"""

    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs):
        super(PositionWiseFFN, self).__init__()
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.mish = nn.Mish()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.mish(self.dense1(X)))


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, num_hiddens, dropout, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(0)
        # 创建⼀个⾜够⻓的P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000, torch.arange(0, num_hiddens, 2,
                                                                                                      dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


class AttentionEncode(nn.Module):
    def __init__(self, dropout, embedding_size, num_heads, max_seqlen):
        super(AttentionEncode, self).__init__()
        self.dropout = dropout
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.at1 = nn.MultiheadAttention(embed_dim=self.embedding_size,
                                         num_heads=num_heads,
                                         dropout=0.6
                                         )
        # self.at1 = MultiHeadAttention(key_size=self.embedding_size,
        #                               query_size=self.embedding_size,
        #                               value_size=self.embedding_size,
        #                               num_hiddens=self.embedding_size,
        #                               num_heads=self.num_heads,
        #                               dropout=self.dropout)
        self.addNorm1 = AddNorm(normalized=[max_seqlen, self.embedding_size], dropout=self.dropout)
        # self.addNorm2 = AddNorm(normalized=[max_seqlen, self.embedding_size], dropout=self.dropout)
        self.FFN = PositionWiseFFN(ffn_num_input=embedding_size, ffn_num_hiddens=embedding_size * 2, ffn_num_outputs=embedding_size)

    def forward(self, x, y=None):
        Multi, _ = self.at1(x, x, x)
        # Multi = self.at1(x, x, x, y)
        Multi_encode = self.addNorm1(x, Multi)
        # return Multi_encode
        encode_output = self.addNorm1(Multi_encode, self.FFN(Multi_encode))
        return encode_output



class FAN_encode(nn.Module):
    def __init__(self, dropout, shape):
        super(FAN_encode, self).__init__()
        self.dropout = dropout
        self.addNorm = AddNorm(normalized=[shape], dropout=dropout)
        self.FFN = PositionWiseFFN(ffn_num_input=shape, ffn_num_hiddens=(2*shape), ffn_num_outputs=shape)

    def forward(self, x):
        encode_output = self.addNorm(x, self.FFN(x))

        return encode_output


def sequence_mask(X, valid_len, value=0.):
    """在序列中屏蔽不相关的项"""
    valid_len = valid_len.float()
    MaxLen = X.size(1)
    mask = torch.arange(MaxLen, dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None].to(X.device)
    X[~mask] = value
    return X


def masked_softmax(X, valid_lens):
    """通过在最后⼀个轴上掩蔽元素来执⾏softmax操作"""
    # X:3D张量，valid_lens:1D或2D张量
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
    if valid_lens.dim() == 1:
        valid_lens = torch.repeat_interleave(valid_lens, shape[1])
    else:
        valid_lens = valid_lens.reshape(-1)  # 最后⼀轴上被掩蔽的元素使⽤⼀个⾮常⼤的负值替换，从⽽其softmax输出为0
    X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
    return nn.functional.softmax(X.reshape(shape), dim=-1)


# class AdditiveAttention(nn.Module):
#     """加性注意⼒"""
#
#     def __init__(self, key_size, query_size, num_hiddens, dropout):
#         super(AdditiveAttention, self).__init__()
#         self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
#         self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
#         self.w_v = nn.Linear(num_hiddens, 1, bias=False)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, queries, keys, values, valid_lens):
#         queries, keys = self.W_q(queries), self.W_k(keys)
#         # 在维度扩展后，
#         # queries的形状：(batch_size，查询的个数，1，num_hidden)
#         # key的形状：(batch_size，1，“键－值”对的个数，num_hiddens)
#         # 使⽤⼴播⽅式进⾏求和
#         features = queries.unsqueeze(2) + keys.unsqueeze(1)
#         features = torch.tanh(features)
#         # self.w_v仅有⼀个输出，因此从形状中移除最后那个维度。
#         # scores的形状：(batch_size，查询的个数，“键-值”对的个数)
#         scores = self.w_v(features).squeeze(-1)
#         attention_weights = masked_softmax(scores, valid_lens)
#         # values的形状：(batch_size，“键－值”对的个数，值的维度)
#         return torch.bmm(self.dropout(attention_weights), values)


class MaskedAttention(nn.Module):
    """加性注意⼒"""

    def __init__(self, input_size, value_size, num_hiddens, dropout):
        super(MaskedAttention, self).__init__()
        self.W_k = nn.Linear(input_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(input_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(input_size, num_hiddens, bias=False)
        self.w_o = nn.Linear(50, value_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        queries, keys = self.W_q(queries), self.W_k(keys)
        d = queries.shape[-1]

        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        scores = self.w_o(scores).permute(0, 2, 1)
        attention_weights = 10 * masked_softmax(scores, valid_lens)

        # attention_weights = nn.Softmax(dim=1)(scores)
        values = self.w_v(values)
        # values = torch.transpose(values, 1, 2)
        # values的形状：(batch_size，“键－值”对的个数，值的维度)
        return torch.bmm(self.dropout(attention_weights), values), attention_weights


class MultiHeadAttention(nn.Module):
    """多头注意力"""

    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens=None):
        # queries，keys，values的形状:
        # (batch_size，查询或者“键－值”对的个数，num_hiddens)
        # valid_lens　的形状:
        # (batch_size，)或(batch_size，查询的个数)
        # 经过变换后，输出的queries，keys，values　的形状:
        # (batch_size*num_heads，查询或者“键－值”对的个数， num_hiddens/num_heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 在轴0，将第一项（标量或者矢量）复制num_heads次，
            # 然后如此复制第二项，然后诸如此类。
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)

        # output的形状:(batch_size*num_heads，查询的个数，num_hiddens/num_heads)
        output = self.attention(queries, keys, values, valid_lens)

        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状"""
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数, num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)

    # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数, num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


class DotProductAttention(nn.Module):
    """缩放点积注意力"""

    def __init__(self, dropout):
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(attention_weights), values)


class MASK_AttentionEncode(nn.Module):
    def __init__(self, dropout, embedding_size, num_heads):
        super(MASK_AttentionEncode, self).__init__()
        self.dropout = dropout
        self.embedding_size = embedding_size
        self.num_heads = num_heads

        self.at1 = MultiHeadAttention(key_size=self.embedding_size,
                                      query_size=self.embedding_size,
                                      value_size=self.embedding_size,
                                      num_hiddens=self.embedding_size,
                                      num_heads=self.num_heads,
                                      dropout=self.dropout)
        self.addNorm = AddNorm(normalized=[50, self.embedding_size], dropout=self.dropout)

        self.FFN = PositionWiseFFN(ffn_num_input=64, ffn_num_hiddens=192, ffn_num_outputs=64)

    def forward(self, x, y=None):
        # Multi, _ = self.at1(x, x, x)
        Multi = self.at1(x, x, x, y)
        Multi_encode = self.addNorm(x, Multi)

        # encode_output = self.addNorm(Multi_encode, self.FFN(Multi_encode))

        return Multi_encode


class transformer_encode(nn.Module):
    def __init__(self, dropout, embedding, num_heads, seq_length):
        super(transformer_encode, self).__init__()
        self.dropout = dropout
        self.embedding_size = embedding
        self.num_heads = num_heads

        self.at1 = MultiHeadAttention(key_size=self.embedding_size,
                                      query_size=self.embedding_size,
                                      value_size=self.embedding_size,
                                      num_hiddens=self.embedding_size,
                                      num_heads=self.num_heads,
                                      dropout=self.dropout)

        self.addNorm = AddNorm(normalized=[seq_length, self.embedding_size], dropout=self.dropout)

        self.ffn = PositionWiseFFN(ffn_num_input=self.embedding_size, ffn_num_hiddens=2*self.embedding_size,
                                   ffn_num_outputs=self.embedding_size)

    def forward(self, x, valid=None):
        # Multi, _ = self.attention(x, x, x)
        Multi = self.at1(x, x, x, valid)
        Multi_encode = self.addNorm(x, Multi)

        encode_output = self.addNorm(Multi_encode, self.ffn(Multi_encode))

        return encode_output


# CBAM
class ChannelAttention(nn.Module):
    # 通道注意力
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool1d(1)  # 最大池化 目标输出大小为1
        self.avgpool = nn.AdaptiveAvgPool1d(1)  # 平均池化 目标输出大小为1
        self.se = nn.Sequential(
            nn.Conv1d(channel, channel // reduction, 1, bias=False),
            nn.Mish(),
            nn.Conv1d(channel // reduction, channel, 1, bias=False)
        )  # 两层感知机
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)  # batch, channel, 1

        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.softmax(max_out + avg_out)
        return output


class SpatialAttention(nn.Module):
    # 空间注意力
    def __init__(self, kernel_size=1):
        super().__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding='same')
        self.softmax = nn.Softmax(2)

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)  # dim =1 返回维度1中最大的那个元素
        avg_result = torch.mean(x, dim=1, keepdim=True)  # 返回维度1中所有元素的平均值
        result = torch.cat([max_result, avg_result], 1)  # 192, 2, 10

        output = self.conv(result)  # 192 1 10
        output = self.softmax(output)
        return output


class CBAMBlock(nn.Module):

    def __init__(self, channel=512, reduction=16, kernel_size=2):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)  # 通道注意力
        self.sa = SpatialAttention(kernel_size=kernel_size)  # 空间注意力

    def forward(self, x):
        b, c, _ = x.size()
        residual = x
        out = x * self.ca(x)  # 192, 64, 10
        out = out * self.sa(out)
        return out + residual


# T-CNN
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
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TemporalRNNNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalRNNNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout),
                       nn.GRU(input_size=out_channels, hidden_size=out_channels, batch_first=True)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
