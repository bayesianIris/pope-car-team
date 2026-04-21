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


class Model(nn.Module):
    """Tiny one-layer actor-critic heads.

    保留强化学习接口，但网络本体极简，仅作为贪心策略的微弱扰动项。
    """

    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_greedy_tiny"
        self.device = device
        self.net_input_dim = Config.NET_INPUT_DIM
        action_num = Config.ACTION_NUM
        value_num = Config.VALUE_NUM

        # One-layer tiny heads / 一层极简头部
        self.actor_head = nn.Linear(self.net_input_dim, action_num)
        self.critic_head = nn.Linear(self.net_input_dim, value_num)

        nn.init.normal_(self.actor_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.actor_head.bias)
        nn.init.normal_(self.critic_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.critic_head.bias)

    def forward(self, obs, inference=False):
        x = obs[:, : self.net_input_dim]
        logits = self.actor_head(x)
        value = self.critic_head(x)
        return logits, value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
