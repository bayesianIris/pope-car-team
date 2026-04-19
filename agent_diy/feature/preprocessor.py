#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Enhanced feature preprocessor and reward design for DIY algorithm.
DIY算法的增强特征预处理与奖励设计。
"""

import numpy as np
from agent_diy.conf.conf import Config

# Map size / 地图尺寸（128×128）
MAP_SIZE = 128.0
# Max monster speed / 最大怪物速度
MAX_MONSTER_SPEED = 5.0
# Max distance bucket / 距离桶最大值
MAX_DIST_BUCKET = 5.0
# Max flash cooldown / 最大闪现冷却步数
MAX_FLASH_CD = 2000.0
# Max buff duration / buff最大持续时间
MAX_BUFF_DURATION = 50.0


def _norm(v, v_max, v_min=0.0):
    """Normalize value to [0, 1].
    
    将值归一化到 [0, 1]。
    """
    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0


class Preprocessor:
    def __init__(self):
        self.reset()

    def reset(self):
        """Reset per-episode state."""
        self.step_no = 0
        self.max_step = 1000
        self.last_min_monster_dist = 1.0
        self.treasures_collected = 0
        self.buffs_collected = 0
        self.hero_pos = None
        self.hero_speed = 1

    def feature_process(self, env_obs, last_action):
        """Process env_obs into feature vector, legal_action mask, and reward.
        
        将 env_obs 转换为特征向量、合法动作掩码和即时奖励。
        Feature: 107D
        - Hero self (6D): pos_x, pos_z, flash_cd, buff_remaining, buff_active, speed
        - Monster 1 (7D): is_in_view, pos_x, pos_z, speed, distance, direction, is_threat
        - Monster 2 (7D): same as Monster 1
        - Treasures (12D): up to 3 treasures
        - Buffs (8D): up to 2 buffs
        - Local Map (49D): 7×7 center patch
        - Legal Actions (16D): action mask
        - Progress (2D): step_norm, survival_ratio
        """
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        map_info = observation["map_info"]
        legal_act_raw = observation["legal_action"]

        self.step_no = observation["step_no"]
        self.max_step = env_info.get("max_step", 1000)

        # ===== Hero self features (6D) =====
        hero = frame_state["heroes"]
        hero_pos = hero["pos"]
        self.hero_pos = (hero_pos["x"], hero_pos["z"])
        self.hero_speed = hero.get("speed", 1)
        
        hero_x_norm = _norm(hero_pos["x"], MAP_SIZE)
        hero_z_norm = _norm(hero_pos["z"], MAP_SIZE)
        flash_cd_norm = _norm(hero.get("flash_cooldown", 0), MAX_FLASH_CD)
        buff_remain_norm = _norm(hero.get("buff_remaining_time", 0), MAX_BUFF_DURATION)
        buff_active = float(hero.get("buff_remaining_time", 0) > 0)
        speed_norm = _norm(self.hero_speed, 2.0)  # max speed is 2 with buff

        hero_feat = np.array(
            [hero_x_norm, hero_z_norm, flash_cd_norm, buff_remain_norm, buff_active, speed_norm],
            dtype=np.float32
        )

        # ===== Monster features (7D x 2) =====
        monsters = frame_state.get("monsters", [])
        monster_feats = []
        min_monster_dist = 1.0
        
        for i in range(2):
            if i < len(monsters):
                m = monsters[i]
                is_in_view = float(m.get("is_in_view", 0))
                m_pos = m["pos"]
                
                if is_in_view:
                    m_x_norm = _norm(m_pos["x"], MAP_SIZE)
                    m_z_norm = _norm(m_pos["z"], MAP_SIZE)
                    m_speed_norm = _norm(m.get("speed", 1), MAX_MONSTER_SPEED)
                    
                    # Euclidean distance / 欧式距离
                    raw_dist = np.sqrt((hero_pos["x"] - m_pos["x"]) ** 2 + (hero_pos["z"] - m_pos["z"]) ** 2)
                    dist_norm = _norm(raw_dist, MAP_SIZE * 1.41)
                    min_monster_dist = min(min_monster_dist, dist_norm)
                    
                    # Direction from hero to monster / 英雄到怪物的方向
                    direction = m.get("hero_relative_direction", 0) / 8.0  # normalize to [0, 1]
                    
                    # Is threat: monster is close and in view
                    # 威胁判断：怪物接近且可见
                    is_threat = float(dist_norm < 0.3)
                else:
                    m_x_norm = 0.0
                    m_z_norm = 0.0
                    m_speed_norm = 0.0
                    dist_norm = 1.0
                    direction = 0.0
                    is_threat = 0.0
                
                monster_feats.append(
                    np.array([is_in_view, m_x_norm, m_z_norm, m_speed_norm, dist_norm, direction, is_threat],
                             dtype=np.float32)
                )
            else:
                monster_feats.append(np.zeros(7, dtype=np.float32))

        # ===== Treasure features (12D) - up to 3 nearest treasures =====
        organs = frame_state.get("organs", [])
        treasures = [o for o in organs if o.get("sub_type") == 1 and o.get("status", 0) == 1]
        treasures = sorted(
            treasures,
            key=lambda item: np.sqrt(
                (hero_pos["x"] - item["pos"]["x"]) ** 2 + (hero_pos["z"] - item["pos"]["z"]) ** 2
            ),
        )
        
        treasure_feats = np.zeros(12, dtype=np.float32)
        for idx, t in enumerate(treasures[:3]):  # Only consider first 3 nearest treasures
            offset = idx * 4
            is_in_view = float(t.get("status", 0) == 1)  # status=1 means can be collected
            t_pos = t["pos"]
            
            if is_in_view:
                t_x_norm = _norm(t_pos["x"], MAP_SIZE)
                t_z_norm = _norm(t_pos["z"], MAP_SIZE)
                
                raw_dist = np.sqrt((hero_pos["x"] - t_pos["x"]) ** 2 + (hero_pos["z"] - t_pos["z"]) ** 2)
                dist_norm = _norm(raw_dist, MAP_SIZE * 1.41)
                direction = t.get("hero_relative_direction", 0) / 8.0
                
                treasure_feats[offset:offset+4] = [t_x_norm, t_z_norm, dist_norm, direction]

        # ===== Buff features (8D) - up to 2 nearest buffs =====
        buffs = [o for o in organs if o.get("sub_type") == 2 and o.get("status", 0) == 1]
        buffs = sorted(
            buffs,
            key=lambda item: np.sqrt(
                (hero_pos["x"] - item["pos"]["x"]) ** 2 + (hero_pos["z"] - item["pos"]["z"]) ** 2
            ),
        )
        
        buff_feats = np.zeros(8, dtype=np.float32)
        for idx, b in enumerate(buffs[:2]):  # Only consider first 2 nearest buffs
            offset = idx * 4
            is_in_view = float(b.get("status", 0) == 1)  # status=1 means can be collected
            b_pos = b["pos"]
            
            if is_in_view:
                b_x_norm = _norm(b_pos["x"], MAP_SIZE)
                b_z_norm = _norm(b_pos["z"], MAP_SIZE)
                
                raw_dist = np.sqrt((hero_pos["x"] - b_pos["x"]) ** 2 + (hero_pos["z"] - b_pos["z"]) ** 2)
                dist_norm = _norm(raw_dist, MAP_SIZE * 1.41)
                direction = b.get("hero_relative_direction", 0) / 8.0
                
                buff_feats[offset:offset+4] = [b_x_norm, b_z_norm, dist_norm, direction]

        # ===== Local map features (49D) =====
        # Use a 7×7 patch so the hero is at the exact center and distances are symmetric.
        map_feat = np.zeros(49, dtype=np.float32)
        if map_info is not None and len(map_info) >= 13:
            center = len(map_info) // 2
            flat_idx = 0
            for row in range(center - 3, center + 4):
                for col in range(center - 3, center + 4):
                    if 0 <= row < len(map_info) and 0 <= col < len(map_info[0]):
                        map_feat[flat_idx] = float(map_info[row][col] != 0)
                    flat_idx += 1

        # ===== Legal action mask (16D) =====
        legal_action = [1] * 16
        if isinstance(legal_act_raw, list) and legal_act_raw:
            if isinstance(legal_act_raw[0], bool):
                for j in range(min(16, len(legal_act_raw))):
                    legal_action[j] = int(legal_act_raw[j])
            else:
                valid_set = {int(a) for a in legal_act_raw if int(a) < 16}
                legal_action = [1 if j in valid_set else 0 for j in range(16)]

        if sum(legal_action) == 0:
            legal_action = [1] * 16

        # ===== Progress features (2D) =====
        step_norm = _norm(self.step_no, self.max_step)
        survival_ratio = step_norm
        progress_feat = np.array([step_norm, survival_ratio], dtype=np.float32)

        # ===== Concatenate all features =====
        feature = np.concatenate([
            hero_feat,
            monster_feats[0],
            monster_feats[1],
            treasure_feats,
            buff_feats,
            map_feat,
            np.array(legal_action, dtype=np.float32),
            progress_feat,
        ])

        # ===== Reward computation =====
        # Track previous state for reward
        treasures_collected_now = env_info.get("treasures_collected", 0)
        buffs_collected_now = env_info.get("collected_buff", 0)
        
        reward = self._compute_reward(
            min_monster_dist=min_monster_dist,
            treasures_collected_now=treasures_collected_now,
            buffs_collected_now=buffs_collected_now,
            legal_actions=legal_action,
            hero_speed=self.hero_speed,
        )

        self.last_min_monster_dist = min_monster_dist
        self.treasures_collected = treasures_collected_now
        self.buffs_collected = buffs_collected_now

        return feature, legal_action, [reward]

    def _compute_reward(self, min_monster_dist, treasures_collected_now, buffs_collected_now,
                        legal_actions, hero_speed):
        """Compute enhanced multi-component reward.
        
        计算增强的多分量奖励。
        
        奖励设计已调整以匹配游戏的实际评分比例：
        - 游戏评分：步数得分(60%) vs 宝箱得分(40%)
        - 学习奖励：生存激励 vs 收集激励
        
        1000步生存可获得150奖励，10个宝箱收集可获得100奖励
        这样两个目标的学习信号接近官方评分的60:40比例
        """
        reward = 0.0

        # 1. Base survival reward / 基础生存奖励
        # 1000步可获150奖励
        reward += Config.SURVIVE_REWARD

        # 2. Distance shaping / 距离塑形
        dist_change = min_monster_dist - self.last_min_monster_dist
        reward += Config.DISTANCE_SHAPING_COEF * dist_change

        # 3. Treasure collection reward / 宝箱收集奖励
        # 10个宝箱可获100奖励，与官方得分比例对齐
        new_treasures = treasures_collected_now - self.treasures_collected
        reward += Config.TREASURE_REWARD * new_treasures

        # 4. Buff collection reward / Buff收集奖励
        new_buffs = buffs_collected_now - self.buffs_collected
        reward += Config.BUFF_REWARD * new_buffs

        # 5. Flash escape reward / 闪现逃脱奖励
        # If distance increased significantly and we had flash available / 如果距离明显增加且闪现可用
        if dist_change > 0.1 and len(legal_actions) > 8 and any(legal_actions[8:]):
            reward += Config.FLASH_ESCAPE_REWARD

        # 6. Direction preference reward / 方向偏好奖励
        # Encourage moving away from monsters / 鼓励远离怪物
        if dist_change > 0:
            reward += Config.DIRECTION_REWARD

        return reward
