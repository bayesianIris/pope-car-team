#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Enhanced neural network model for DIY algorithm.
DIY算法的增强神经网络模型。
"""

import torch
import torch.nn as nn
import numpy as np
from agent_diy.conf.conf import Config


def make_fc_layer(in_features, out_features):
    """Create a linear layer with orthogonal initialization.
    
    创建正交初始化的线性层。
    """
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight.data)
    nn.init.zeros_(fc.bias.data)
    return fc


class Model(nn.Module):
    """Enhanced MLP backbone with larger capacity + Actor/Critic dual heads.
    
    增强的MLP骨干网络（更大容量）+ Actor/Critic双头。
    
    Architecture:
    Input (107D) → FC(256) → ReLU → FC(128) → ReLU → FC(64) → ReLU
                                                           ↙                           ↘
                                                      Actor(64→16)              Critic(64→1)
    """

    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_enhanced_diy"
        self.device = device

        input_dim = Config.DIM_OF_OBSERVATION
        hidden_dims = Config.HIDDEN_DIMS  # [256, 128, 64]
        action_num = Config.ACTION_NUM  # 16
        value_num = Config.VALUE_NUM  # 1

        # Shared backbone with larger capacity / 更大容量的共享骨干网络
        self.backbone = nn.Sequential(
            make_fc_layer(input_dim, hidden_dims[0]),
            nn.ReLU(),
            make_fc_layer(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            make_fc_layer(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
        )

        # Actor head (16 actions: 8 move + 8 flash) / 策略头（16个动作）
        self.actor_head = make_fc_layer(hidden_dims[2], action_num)

        # Critic head (value function) / 价值头
        self.critic_head = make_fc_layer(hidden_dims[2], value_num)

    def forward(self, obs, inference=False):
        """Forward pass.
        
        Args:
            obs: Input observation tensor (batch_size, 107)
            inference: Whether in inference mode (unused, for compatibility)
        
        Returns:
            logits: Action logits (batch_size, 16)
            value: State value (batch_size, 1)
        """
        hidden = self.backbone(obs)
        logits = self.actor_head(hidden)
        value = self.critic_head(hidden)
        return logits, value

    def set_train_mode(self):
        """Set model to training mode."""
        self.train()

    def set_eval_mode(self):
        """Set model to evaluation mode."""
        self.eval()

