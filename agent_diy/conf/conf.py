#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Enhanced configuration for DIY algorithm - Improved Gorge Chase Agent.
DIY算法的增强配置 - 改进的峡谷追猎智能体。
"""

import numpy as np


# Configuration for enhanced DIY algorithm
# 增强DIY算法的配置
class Config:

    # ============ Feature Dimensions / 特征维度 ============
    # Enhanced feature composition (107D total):
    # - Hero self (6D): pos_x, pos_z, flash_cd, buff_remaining, buff_active, speed
    # - Monster 1 (7D): is_in_view, pos_x, pos_z, speed, distance, direction, is_threat
    # - Monster 2 (7D): same as Monster 1
    # - Treasures (12D): up to 3 nearest treasures with (is_in_view, pos_x, pos_z, distance, direction) each
    # - Buffs (8D): up to 2 nearest buffs with (is_in_view, pos_x, pos_z, distance, direction) each
    # - Local Map (49D): 7×7 center map patch
    # - Legal Actions (16D): 16-dim action mask
    # - Progress (2D): step normalized, survival ratio
    
    FEATURE_DIMS = [6, 7, 7, 12, 8, 49, 16, 2]
    DIM_OF_OBSERVATION = sum(FEATURE_DIMS)  # 107D

    # Action space: 16 (8 move + 8 flash)
    # 动作空间：16维（8个移动 + 8个闪现）
    ACTION_NUM = 16

    # Value output: single value head (survival value)
    # 价值头：单个价值输出（生存价值）
    VALUE_NUM = 1

    # ============ PPO Hyperparameters / PPO超参数 ============
    # Discount factor for future rewards
    # 未来奖励的折扣因子
    GAMMA = 0.99

    # GAE lambda parameter
    # GAE lambda参数
    LAMDA = 0.95

    # Initial learning rate
    # 初始学习率
    INIT_LEARNING_RATE_START = 0.0003

    # Entropy regularization coefficient
    # 熵正则化系数
    BETA_START = 0.001

    # PPO clip parameter
    # PPO裁剪参数
    CLIP_PARAM = 0.2

    # Value function loss coefficient
    # 价值函数损失系数
    VF_COEF = 1.0

    # Gradient clip range
    # 梯度裁剪范围
    GRAD_CLIP_RANGE = 0.5

    # ============ Network Architecture / 网络架构 ============
    # Input dimension
    INPUT_DIM = DIM_OF_OBSERVATION
    # Hidden dimensions
    HIDDEN_DIMS = [256, 128, 64]
    # Actor output dimension
    ACTOR_OUT_DIM = ACTION_NUM
    # Critic output dimension
    CRITIC_OUT_DIM = VALUE_NUM

    # ============ Reward Design / 奖励设计参数 ============
    # 注意：奖励设计已调整以匹配游戏评分比例
    # 游戏评分: 步数得分(60%) vs 宝箱得分(40%)
    # 学习奖励应该类似比例: 生存激励 vs 收集激励
    
    # Base survival reward per step.
    # 每步的基础生存奖励。
    # 1000步可获得150奖励，与10个宝箱的100奖励保持接近60:40的比例。
    SURVIVE_REWARD = 0.15

    # Treasure collection reward - aligned with official score ratio.
    # 宝箱收集奖励 - 按官方评分比例对齐。
    # 10个宝箱可获100奖励，和1000步生存奖励(150)处于同一量级。
    TREASURE_REWARD = 10.0

    # Buff collection reward - INCREASED proportionally
    # Buff收集奖励 - 相应提升 (0.3 → 2.0)
    BUFF_REWARD = 2.0

    # Flash escape reward (when distance increases significantly)
    # 闪现逃脱奖励（距离显著增加时）- INCREASED (0.5 → 1.0)
    FLASH_ESCAPE_REWARD = 1.0

    # Distance shaping coefficient
    # 距离塑形系数 - 保持不变
    DISTANCE_SHAPING_COEF = 0.15

    # Direction preference reward (moving away from monsters)
    # 方向偏好奖励（远离怪物）- 保持不变
    DIRECTION_REWARD = 0.05

    # Failure penalty (when caught by monsters)
    # 失败惩罚（被怪物抓住）- 保持不变
    FAILURE_PENALTY = -1.0
