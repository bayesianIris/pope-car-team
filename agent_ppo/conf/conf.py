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

    # Feature dimensions / 特征维度（共366维）
    # [hero4, monster1_6, monster2_6, nearest_treasure4, nearest_buff4,
    #  local_maps_4x9x9, legal_action16, progress2]
    FEATURES = [
        4,
        6,
        6,
        4,
        4,
        4 * 9 * 9,
        16,
        2,
    ]
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURE_SPLIT_SHAPE)
    DIM_OF_OBSERVATION = FEATURE_LEN

    # Action space / 动作空间：16维（8个移动 + 8个闪现）
    ACTION_NUM = 16

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
