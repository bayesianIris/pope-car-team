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


class Model(nn.Module):
    """CNN map encoder + MLP vector encoder + Actor/Critic dual heads.

    地图 CNN 编码 + 向量 MLP 编码 + Actor/Critic 双头。
    """

    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_cnn"
        self.device = device

        self.map_channels = 4
        map_size = int((Config.FEATURES[5] // self.map_channels) ** 0.5)
        self.map_h = map_size
        self.map_w = map_size
        self.map_dim = Config.FEATURES[5]
        self.vec_dim = Config.DIM_OF_OBSERVATION - self.map_dim

        vec_hidden_dim = 128
        fused_dim = 128
        action_num = Config.ACTION_NUM
        value_num = Config.VALUE_NUM

        # Vector branch / 向量分支
        self.vec_encoder = nn.Sequential(
            make_fc_layer(self.vec_dim, vec_hidden_dim),
            nn.ReLU(),
            make_fc_layer(vec_hidden_dim, vec_hidden_dim),
            nn.ReLU(),
        )

        # Map branch / 地图分支
        self.map_encoder = nn.Sequential(
            nn.Conv2d(self.map_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        with torch.no_grad():
            dummy_map = torch.zeros(1, self.map_channels, self.map_h, self.map_w)
            map_out_dim = int(self.map_encoder(dummy_map).flatten(start_dim=1).shape[1])
        self.map_proj = nn.Sequential(
            make_fc_layer(map_out_dim, vec_hidden_dim),
            nn.ReLU(),
        )

        # Fusion backbone / 融合骨干
        self.fusion = nn.Sequential(
            make_fc_layer(vec_hidden_dim * 2, fused_dim),
            nn.ReLU(),
        )

        # Actor head / 策略头
        self.actor_head = make_fc_layer(fused_dim, action_num)

        # Critic head / 价值头
        self.critic_head = make_fc_layer(fused_dim, value_num)

    def _split_obs(self, obs):
        """Split flattened observation into vector part and local map part.

        将展平观测拆分为向量部分和局部地图部分。
        """
        # Feature layout: [4,6,6,4,4,4*HxW,16,2], map is the 6th segment.
        map_start = sum(Config.FEATURES[:5])
        map_end = map_start + self.map_dim

        vec_left = obs[:, :map_start]
        map_flat = obs[:, map_start:map_end]
        vec_right = obs[:, map_end:]
        vec = torch.cat([vec_left, vec_right], dim=1)
        map_tensor = map_flat.view(-1, self.map_channels, self.map_h, self.map_w)
        return vec, map_tensor

    def forward(self, obs, inference=False):
        vec_obs, map_obs = self._split_obs(obs)

        vec_hidden = self.vec_encoder(vec_obs)

        map_hidden = self.map_encoder(map_obs)
        map_hidden = map_hidden.flatten(start_dim=1)
        map_hidden = self.map_proj(map_hidden)

        hidden = self.fusion(torch.cat([vec_hidden, map_hidden], dim=1))

        logits = self.actor_head(hidden)
        value = self.critic_head(hidden)
        return logits, value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
