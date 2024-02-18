#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""


import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorly.decomposition import tucker
import tensorly as tl

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature


class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


class DGCNN(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args.k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims*2+128, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)
        self.classifier = nn.Sequential(
            nn.Linear(3*1024, 128),
        )
    def forward(self, x):
        device=x.device
        a, b, c = x.shape
        X = tl.tensor(x.cpu().reshape(a, b, c))
        core, factors = tucker(X, rank=[a, b, c])
        xx = torch.from_numpy(core).to(device)

        xx=xx.reshape(a,b*c)
       
        xx=self.classifier(xx)
       
   

        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        x = torch.cat((x, xx), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

def calc_flops(model, input):
    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups) * (
            2 if multiply_adds else 1)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width

        list_conv.append(flops)

    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1
        num_steps = input[0].size(0)
        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement() if self.bias is not None else 0

        flops = batch_size * (weight_ops + bias_ops)
        flops *= num_steps
        list_linear.append(flops)

    def fsmn_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.filter.nelement() * (2 if multiply_adds else 1)
        num_steps = input[0].size(0)
        flops = num_steps * weight_ops
        flops *= batch_size
        list_fsmn.append(flops)

    def gru_cell(input_size, hidden_size, bias=True):
        total_ops = 0
        # r = \sigma(W_{ir} x + b_{ir} + W_{hr} h + b_{hr}) \\
        # z = \sigma(W_{iz} x + b_{iz} + W_{hz} h + b_{hz}) \\
        state_ops = (hidden_size + input_size) * hidden_size + hidden_size
        if bias:
            state_ops += hidden_size * 2
        total_ops += state_ops * 2

        # n = \tanh(W_{in} x + b_{in} + r * (W_{hn} h + b_{hn})) \\
        total_ops += (hidden_size + input_size) * hidden_size + hidden_size
        if bias:
            total_ops += hidden_size * 2
        # r hadamard : r * (~)
        total_ops += hidden_size

        # h' = (1 - z) * n + z * h
        # hadamard hadamard add
        total_ops += hidden_size * 3

        return total_ops

    def gru_hook(self, input, output):

        batch_size = input[0].size(0) if input[0].dim() == 2 else 1
        if self.batch_first:
            batch_size = input[0].size(0)
            num_steps = input[0].size(1)
        else:
            batch_size = input[0].size(1)
            num_steps = input[0].size(0)
        total_ops = 0
        bias = self.bias
        input_size = self.input_size
        hidden_size = self.hidden_size
        num_layers = self.num_layers
        total_ops = 0
        total_ops += gru_cell(input_size, hidden_size, bias)
        for i in range(num_layers - 1):
            total_ops += gru_cell(hidden_size, hidden_size, bias)
        total_ops *= batch_size
        total_ops *= num_steps

        list_lstm.append(total_ops)

    def lstm_cell(input_size, hidden_size, bias):
        total_ops = 0
        state_ops = (input_size + hidden_size) * hidden_size + hidden_size
        if bias:
            state_ops += hidden_size * 2
        total_ops += state_ops * 4
        total_ops += hidden_size * 3
        total_ops += hidden_size
        return total_ops

    def lstm_hook(self, input, output):

        batch_size = input[0].size(0) if input[0].dim() == 2 else 1
        if self.batch_first:
            batch_size = input[0].size(0)
            num_steps = input[0].size(1)
        else:
            batch_size = input[0].size(1)
            num_steps = input[0].size(0)
        total_ops = 0
        bias = self.bias
        input_size = self.input_size
        hidden_size = self.hidden_size
        num_layers = self.num_layers
        total_ops = 0
        total_ops += lstm_cell(input_size, hidden_size, bias)
        for i in range(num_layers - 1):
            total_ops += lstm_cell(hidden_size, hidden_size, bias)
        total_ops *= batch_size
        total_ops *= num_steps

        list_lstm.append(total_ops)

    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement())

    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width

        list_pooling.append(flops)

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            print(net)
            if isinstance(net, torch.nn.Conv2d) or isinstance(net, torch.nn.ConvTranspose2d):
                net.register_forward_hook(conv_hook)
                # print('conv_hook_ready')
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
                # print('linear_hook_ready')
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
                # print('batch_norm_hook_ready')
            if isinstance(net, torch.nn.ReLU) or isinstance(net, torch.nn.PReLU):
                net.register_forward_hook(relu_hook)
                # print('relu_hook_ready')
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(pooling_hook)
                # print('pooling_hook_ready')
            if isinstance(net, torch.nn.LSTM):
                net.register_forward_hook(lstm_hook)
                # print('lstm_hook_ready')
            if isinstance(net, torch.nn.GRU):
                net.register_forward_hook(gru_hook)

            # if isinstance(net, FSMNZQ):
            #     net.register_forward_hook(fsmn_hook)
            # print('fsmn_hook_ready')
            return
        for c in childrens:
            foo(c)

    multiply_adds = False
    list_conv, list_bn, list_relu, list_linear, list_pooling, list_lstm, list_fsmn = [], [], [], [], [], [], []
    foo(model)

    _ = model(input)

    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling) + sum(
        list_lstm) + sum(list_fsmn))
    fsmn_flops = (sum(list_fsmn) + sum(list_linear))
    lstm_flops = sum(list_lstm)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('The network has {} params.'.format(params))

    print(total_flops, fsmn_flops, lstm_flops)
    print('  + Number of FLOPs: %.2f M' % (total_flops / 1000 ** 2))
    return total_flops
