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

    # Scalar feature dimensions / 标量特征维度
    FEATURES = [
        4,
        5,
        5,
        8,
        16,
        2,
    ]
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_VECTOR_LEN = sum(FEATURE_SPLIT_SHAPE)

    # Map CNN input / 地图 CNN 输入
    MAP_VIEW_RADIUS = 10
    MAP_VIEW_SIZE = MAP_VIEW_RADIUS * 2 + 1
    MAP_CHANNELS = 3
    MAP_DEFAULT_VALUE = 0.0
    FEATURE_IMAGE_SHAPE = (MAP_CHANNELS, MAP_VIEW_SIZE, MAP_VIEW_SIZE)
    FEATURE_IMAGE_LEN = MAP_CHANNELS * MAP_VIEW_SIZE * MAP_VIEW_SIZE

    DIM_OF_OBSERVATION = FEATURE_VECTOR_LEN + FEATURE_IMAGE_LEN

    # Action space / 动作空间：16维（8移动 + 8闪现）
    ACTION_NUM = 16

    # Action penalties / 动作惩罚
    ACTION_FAIL_PENALTY = 0.08
    ILLEGAL_ACTION_PENALTY = 0.15

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
