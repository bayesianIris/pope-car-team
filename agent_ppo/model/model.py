#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Neural network model for Gorge Chase PPO.
峡谷追猎 PPO 神经网络模型。
"""

import math

import torch
import torch.nn as nn

from agent_ppo.conf.conf import Config


def make_fc_layer(in_features, out_features, gain=math.sqrt(2.0)):
    """Create a linear layer with orthogonal initialization.

    创建正交初始化的线性层。
    """
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight.data, gain=gain)
    nn.init.zeros_(fc.bias.data)
    return fc


def make_conv_layer(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    """Create a conv2d layer with orthogonal initialization.

    创建正交初始化的卷积层。
    """
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
    nn.init.orthogonal_(conv.weight.data, gain=math.sqrt(2.0))
    if conv.bias is not None:
        nn.init.zeros_(conv.bias.data)
    return conv


class ResidualBlock(nn.Module):
    """Simple residual block for map feature extraction.

    用于地图特征提取的轻量残差块。
    """

    def __init__(self, channels):
        super().__init__()
        self.conv1 = make_conv_layer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = make_conv_layer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.act2 = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.act1(self.conv1(x))
        out = self.conv2(out)
        return self.act2(out + identity)


class Model(nn.Module):
    """Enhanced PPO model with map residual CNN and dual towers.

    增强版 PPO 网络：残差地图 CNN + 向量编码 + Actor/Critic 分塔。
    """

    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_lite"
        self.device = device

        self.vector_dim = Config.FEATURE_VECTOR_LEN
        self.map_channels, self.map_height, self.map_width = Config.FEATURE_IMAGE_SHAPE
        self.global_map_channels, self.global_map_height, self.global_map_width = Config.GLOBAL_MAP_SHAPE
        map_hidden_dim = 96
        global_map_hidden_dim = 64
        vector_hidden_dim = 128
        fusion_hidden_dim = 256
        tower_hidden_dim = 128
        action_num = Config.ACTION_NUM
        value_num = Config.VALUE_NUM

        # Map encoder / 地图编码器
        self.map_encoder = nn.Sequential(
            make_conv_layer(self.map_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            ResidualBlock(32),
            make_conv_layer(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            ResidualBlock(64),
            make_conv_layer(64, map_hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Global Map encoder
        self.global_map_encoder = nn.Sequential(
            make_conv_layer(self.global_map_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            make_conv_layer(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            make_conv_layer(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            make_conv_layer(64, global_map_hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Vector encoder / 标量编码器
        self.vector_encoder = nn.Sequential(
            make_fc_layer(self.vector_dim, vector_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(vector_hidden_dim),
        )

        # Fusion backbone / 融合骨干网络
        self.fusion_backbone = nn.Sequential(
            make_fc_layer(vector_hidden_dim + map_hidden_dim + global_map_hidden_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(fusion_hidden_dim),
        )

        # Actor/Critic towers / 策略与价值分塔
        self.actor_tower = nn.Sequential(
            make_fc_layer(fusion_hidden_dim, tower_hidden_dim),
            nn.ReLU(),
        )

        self.critic_tower = nn.Sequential(
            make_fc_layer(fusion_hidden_dim, tower_hidden_dim),
            nn.ReLU(),
        )

        # Actor head / 策略头
        self.actor_head = make_fc_layer(tower_hidden_dim, action_num, gain=0.01)

        # Critic head / 价值头
        self.critic_head = make_fc_layer(tower_hidden_dim, value_num, gain=1.0)

    def forward(self, obs, inference=False):
        vector_obs = obs[:, : self.vector_dim]
        
        map_offset = self.vector_dim + self.map_channels * self.map_height * self.map_width
        map_obs = obs[:, self.vector_dim : map_offset].view(-1, self.map_channels, self.map_height, self.map_width)
        global_map_obs = obs[:, map_offset : map_offset + self.global_map_channels * self.global_map_height * self.global_map_width].view(-1, self.global_map_channels, self.global_map_height, self.global_map_width)

        vector_hidden = self.vector_encoder(vector_obs)
        map_hidden = self.map_encoder(map_obs).flatten(start_dim=1)
        global_map_hidden = self.global_map_encoder(global_map_obs).flatten(start_dim=1)
        
        hidden = self.fusion_backbone(torch.cat([vector_hidden, map_hidden, global_map_hidden], dim=1))

        actor_hidden = self.actor_tower(hidden)
        critic_hidden = self.critic_tower(hidden)

        logits = self.actor_head(actor_hidden)
        value = self.critic_head(critic_hidden)
        return logits, value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
