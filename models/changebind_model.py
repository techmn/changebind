#!/usr/bin/env python3

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm import create_model
from timm.models.layers import DropPath, trunc_normal_
from models.cd_modules import Decoder


class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x



class DifferenceEncoder(nn.Module):
    """
    param dims: list, number of channels of encoder stages e.g. [256, 512, 1024, 2048] for Resnet50
    param kernel_size: int, kernel size of convolution for merging pre/post features
    param heads: list, number of heads for encoder features
    param embedding_dim: int, output channels for input to decoder
    """
    def __init__(self, dims=[256, 512, 1024, 2048], kernel_size=3, heads=[4, 8, 16, 32], embedding_dim=256):
        super().__init__()

        # convolution layers to merge pre/post features
        self.conv1 = nn.Conv2d(in_channels=dims[0]*2, out_channels=dims[0], kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dims[1]*2, out_channels=dims[1], kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=True)
        self.conv3 = nn.Conv2d(in_channels=dims[2]*2, out_channels=dims[2], kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=True)
        self.conv4 = nn.Conv2d(in_channels=dims[3]*2, out_channels=dims[3], kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=True)

        # convolution layers to merge local and global contextual features
        self.proj1 = nn.Conv2d(in_channels=dims[0]*3, out_channels=dims[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.proj2 = nn.Conv2d(in_channels=dims[1]*3, out_channels=dims[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.proj3 = nn.Conv2d(in_channels=dims[2]*3, out_channels=dims[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.proj4 = nn.Conv2d(in_channels=dims[3]*3, out_channels=dims[0], kernel_size=3, stride=1, padding=1, bias=False)

        # multi-head attention to mix pre/post features
        self.attn1 = nn.MultiheadAttention(dims[0]*2, heads[0], dropout=0.1, bias=True, batch_first=True)
        self.attn2 = nn.MultiheadAttention(dims[1]*2, heads[1], dropout=0.1, bias=True, batch_first=True)
        self.attn3 = nn.MultiheadAttention(dims[2]*2, heads[2], dropout=0.1, bias=True, batch_first=True)
        self.attn4 = nn.MultiheadAttention(dims[3]*2, heads[3], dropout=0.1, bias=True, batch_first=True)

        self.norm1 = LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        self.norm2 = LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        self.norm3 = LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        self.norm4 = LayerNorm(dims[0], eps=1e-6, data_format="channels_first")

        # final convolution layer to merge multi-scale local and global contextual features
        self.multiscale_fusion = nn.Conv2d(in_channels=dims[0]*4, out_channels=embedding_dim, kernel_size=3, stride=1, padding=1, bias=True)
        self.multiscale_norm = nn.BatchNorm2d(embedding_dim)

        self.relu = nn.ReLU()
        self.gelu = nn.GELU()

    def forward(self, pre, post):
        x1, x2, x3, x4 = pre
        y1, y2, y3, y4 = post

        B, C, H, W = x1.shape
        d1 = torch.cat([x1, y1], dim=1)
        d2 = torch.cat([x2, y2], dim=1)
        d3 = torch.cat([x3, y3], dim=1)
        d4 = torch.cat([x4, y4], dim=1)

        spat1 = self.gelu(self.conv1(d1))
        d1 = d1.flatten(2).transpose(1,2)
        attn1, _ = self.attn1(d1, d1, d1)
        attn1 = self.gelu(attn1).transpose(1,2).reshape([spat1.shape[0], spat1.shape[1]*2, spat1.shape[2], spat1.shape[3]]).contiguous()
        f1 = self.norm1(self.relu(self.proj1(torch.cat([spat1, attn1], dim=1))))

        spat2 = self.gelu(self.conv2(d2))
        d2 = d2.flatten(2).transpose(1,2)
        attn2, _ = self.attn2(d2, d2, d2)
        attn2 = self.gelu(attn2).transpose(1,2).reshape([spat2.shape[0], spat2.shape[1]*2, spat2.shape[2], spat2.shape[3]]).contiguous()
        f2 = self.norm2(self.relu(self.proj2(torch.cat([spat2, attn2], dim=1))))

        spat3 = self.gelu(self.conv3(d3))
        d3 = d3.flatten(2).transpose(1,2)
        attn3, _ = self.attn3(d3, d3, d3)
        attn3 = self.gelu(attn3).transpose(1,2).reshape([spat3.shape[0], spat3.shape[1]*2, spat3.shape[2], spat3.shape[3]]).contiguous()
        f3 = self.norm3(self.relu(self.proj3(torch.cat([spat3, attn3], dim=1))))

        spat4 = self.gelu(self.conv4(d4))
        d4 = d4.flatten(2).transpose(1,2)
        attn4, _ = self.attn4(d4, d4, d4)
        attn4 = self.gelu(attn4).transpose(1,2).reshape([spat4.shape[0], spat4.shape[1]*2, spat4.shape[2], spat4.shape[3]]).contiguous()
        f4 = self.norm4(self.relu(self.proj4(torch.cat([spat4, attn4], dim=1))))

        f2 = F.interpolate(f2, size=(H, W), mode='bilinear')
        f3 = F.interpolate(f3, size=(H, W), mode='bilinear')
        f4 = F.interpolate(f4, size=(H, W), mode='bilinear')

        x = torch.cat((f1,f2,f3,f4), dim=1)
        x = self.multiscale_fusion(x)
        x = self.multiscale_norm(self.relu(x))

        return x


class ChangeBindModel(nn.Module):
    def __init__(self, embed_dim=256, encoder_type='resnet50', encoder_dims=[256, 512, 1024, 2048], freeze_backbone=False):
        super().__init__()

        self.visual_encoder = create_model(encoder_type, pretrained=False, features_only=True)
        self.difference_encoder = DifferenceEncoder(dims=encoder_dims, embedding_dim=embed_dim)
        self.decoder = Decoder(embedding_dim=embed_dim)
        
        self.apply(self._init_weights)

        if freeze_backbone:
            for param in self.visual_encoder.parameters():
                param.requires_grad = False


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward_visual_features(self, x):
        _, x1, x2, x3, x4 = self.visual_encoder(x)
        return x1, x2, x3, x4

    def encode_difference_features(self, pre_feats, post_feats):
        x = self.difference_encoder(pre_feats, post_feats)
        return x

    def forward(self, pre_img, post_img):
        # extract visual features
        x1, x2, x3, x4 = self.forward_visual_features(pre_img)
        y1, y2, y3, y4 = self.forward_visual_features(post_img)

        # extract difference features
        diff_feats = self.encode_difference_features([x1,x2,x3,x4], [y1,y2,y3,y4])

        pred = self.decoder(diff_feats)

        return pred


if __name__ == "__main__":
    model = ChangeBindModel()

    from fvcore.nn import FlopCountAnalysis, flop_count_table
    input_res = (3, 256, 256)
    input = torch.ones(()).new_empty((1, *input_res), dtype=next(model.parameters()).dtype,
                                     device=next(model.parameters()).device)
    #model.eval()
    flops = FlopCountAnalysis(model, (input, input))
    print(flop_count_table(flops))
