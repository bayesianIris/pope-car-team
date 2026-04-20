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

import torch
import torch.nn as nn

from agent_ppo.conf.conf import Config


def make_fc_layer(in_features, out_features):
    """Create a linear layer with orthogonal initialization.

    创建正交初始化的线性层。
    """
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight.data)
    nn.init.zeros_(fc.bias.data)
    return fc


def make_conv_layer(in_channels, out_channels, kernel_size, stride=1, padding=0):
    """Create a conv layer with orthogonal initialization.

    创建正交初始化的卷积层。
    """
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
    nn.init.orthogonal_(conv.weight.data)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias.data)
    return conv


class Model(nn.Module):
    """Map CNN encoder + tabular MLP encoder + Actor/Critic dual heads.

    地图CNN编码 + 非地图MLP编码 + Actor/Critic 双头。
    """

    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_lite"
        self.device = device

        input_dim = Config.DIM_OF_OBSERVATION
        # Features layout: [4, 5, 5, 15, 5, 81, 16, 2]
        # 第6段为局部地图特征(9x9=81)
        self.map_feat_idx = 5
        self.map_feat_start = sum(Config.FEATURES[: self.map_feat_idx])
        self.map_feat_dim = Config.FEATURES[self.map_feat_idx]
        self.map_feat_end = self.map_feat_start + self.map_feat_dim

        map_side = int(self.map_feat_dim ** 0.5)
        if map_side * map_side != self.map_feat_dim:
            raise ValueError(
                f"Map feature dim should be a square number, got {self.map_feat_dim}."
            )
        self.map_side = map_side

        non_map_dim = input_dim - self.map_feat_dim
        action_num = Config.ACTION_NUM
        value_num = Config.VALUE_NUM

        # Map branch: 1x9x9 -> CNN embedding / 地图分支
        self.map_encoder = nn.Sequential(
            make_conv_layer(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            make_conv_layer(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            make_fc_layer(32 * self.map_side * self.map_side, 64),
            nn.ReLU(),
        )

        # Non-map branch / 非地图分支
        self.non_map_encoder = nn.Sequential(
            make_fc_layer(non_map_dim, 64),
            nn.ReLU(),
        )

        # Fusion trunk / 融合骨干
        self.fusion = nn.Sequential(
            make_fc_layer(128, 64),
            nn.ReLU(),
        )

        # Actor head / 策略头
        self.actor_head = make_fc_layer(64, action_num)

        # Critic head / 价值头
        self.critic_head = make_fc_layer(64, value_num)

    def forward(self, obs, inference=False):
        map_feat = obs[:, self.map_feat_start : self.map_feat_end]
        map_feat = map_feat.view(-1, 1, self.map_side, self.map_side)

        non_map_feat = torch.cat(
            [obs[:, : self.map_feat_start], obs[:, self.map_feat_end :]], dim=1
        )

        map_hidden = self.map_encoder(map_feat)
        non_map_hidden = self.non_map_encoder(non_map_feat)
        hidden = self.fusion(torch.cat([map_hidden, non_map_hidden], dim=1))

        logits = self.actor_head(hidden)
        value = self.critic_head(hidden)
        return logits, value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
