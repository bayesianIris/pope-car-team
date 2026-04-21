#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Configuration for Gorge Chase PPO.
峡谷追猎 PPO 配置。
"""


class Config:

    # Action space / 动作空间：16维
    ACTION_NUM = 16

    # Compact feature dimensions / 轻量特征维度（共49维）
    # [core(17), dir_open(8), dir_blocked(8), greedy_action_scores(16)]
    FEATURES = [17, 8, 8, 16]
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURE_SPLIT_SHAPE)
    DIM_OF_OBSERVATION = FEATURE_LEN

    # Prefix used by the tiny one-layer network / 极简一层网络输入前缀
    NET_INPUT_DIM = 17

    # Greedy score slice in feature / 特征中贪心评分切片
    GREEDY_SCORE_START = FEATURE_LEN - ACTION_NUM
    GREEDY_SCORE_END = FEATURE_LEN

    # Fusion ratio: net only contributes 0.01% / 融合比率：网络仅贡献0.01%
    NET_LOGIT_RATIO = 1e-4

    # Value head / 价值头：单头生存奖励
    VALUE_NUM = 1

    # PPO hyperparameters / PPO 超参数
    GAMMA = 0.99
    LAMDA = 0.95
    INIT_LEARNING_RATE_START = 0.0003
    BETA_START = 0.001
    CLIP_PARAM = 0.2
    VF_COEF = 1.0
    GRAD_CLIP_RANGE = 0.5
