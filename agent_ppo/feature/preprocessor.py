#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Feature preprocessor and reward design for Gorge Chase PPO.
峡谷追猎 PPO 特征预处理与奖励设计。
"""

from collections import deque

import numpy as np
from agent_ppo.conf.conf import Config

# Map size / 地图尺寸（128×128）
MAP_SIZE = 128.0
# World map side length / 全局地图边长
WORLD_MAP_SIZE = 128
# Local view map side length / 局部视野边长
LOCAL_VIEW_SIZE = 21
# Max monster speed / 最大怪物速度
MAX_MONSTER_SPEED = 5.0
# Max relative direction enum / 最大相对方向枚举
MAX_REL_DIR = 8.0
# Max flash cooldown / 最大闪现冷却步数
MAX_FLASH_CD = 2000.0
# Max buff duration / buff最大持续时间
MAX_BUFF_DURATION = 50.0
# Action dimension / 动作维度
ACTION_DIM = 16

# Direction offsets for action protocol / 动作协议方向偏移量
DIRECTION_OFFSETS = (
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
    (1, 0),
    (1, 1),
)

# Reward for each newly collected treasure / 每新增一个宝箱奖励
TREASURE_INC_REWARD = 0.66
# Reward for each newly collected buff / 每新增一个buff奖励
BUFF_INC_REWARD = 0.4
# Small exploration reward coefficient / 探索奖励系数（小）
EXPLORE_REWARD_PER_CELL = 0.002
# Flash-escape bonus / 闪现拉开距离奖励
FLASH_ESCAPE_REWARD = 0.03
# Penalty when action does not move hero position / 动作未发生位移时惩罚
ACTION_FAIL_PENALTY = 0.02
# Wander waste penalty: min steps to achieve net displacement, waste = 10 - min_steps
# 浪费步数惩罚：净位移需要的最少移动步数，浪费步数 = 10 - 最少步数
WASTE_STEP_THRESHOLD = 4     # wasted steps ≤ this → no penalty / 浪费 ≤ 4 不罚（容忍网格量化损失）
WASTE_STEP_PENALTY = 0.015   # penalty per wasted step / 每浪费一步的惩罚
WANDER_WINDOW_SIZE = 10

# Potential-based shaping for organs (slower decay than reference)
# 参考实现半衰期约1，这里调慢到半衰期约4
ORGAN_POTENTIAL_DECAY = np.log(2.0) / 4.0
ORGAN_TREASURE_ONE_STEP_REWARD = 0.1
ORGAN_POTENTIAL_SCALE = ORGAN_TREASURE_ONE_STEP_REWARD / (
    Config.GAMMA - np.exp(-ORGAN_POTENTIAL_DECAY)
)
BUFF_SHAPING_MULT = BUFF_INC_REWARD / TREASURE_INC_REWARD

# Nonlinear monster-distance shaping (also slow decay)
MONSTER_POTENTIAL_DECAY = np.log(2.0) / 4.0
MONSTER_POTENTIAL_SCALE = 0.12


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
        self.step_no = 0
        self.max_step = 200
        self.last_treasures_collected = 0
        self.last_collected_buff = 0
        self.last_organ_potential = None
        self.last_monster_potential = None
        # -1: 未探索, 0: 障碍, 1: 可通行
        self.global_map = np.full((WORLD_MAP_SIZE, WORLD_MAP_SIZE), -1.0, dtype=np.float32)
        self.explored_cell_count = 0
        self.last_hero_pos = None
        self.position_window = deque(maxlen=WANDER_WINDOW_SIZE)

    def feature_process(self, env_obs, last_action):
        """Process env_obs into feature vector, legal_action mask, and reward.

        将 env_obs 转换为特征向量、合法动作掩码和即时奖励。
        """
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        map_info = observation["map_info"]
        legal_act_raw = observation["legal_action"]

        self.step_no = observation["step_no"]
        self.max_step = env_info.get("max_step", 200)
        cur_treasures_collected = int(env_info.get("treasures_collected", 0))
        cur_collected_buff = int(env_info.get("collected_buff", 0))

        # Hero self features (4D) / 英雄自身特征
        hero = frame_state["heroes"]
        hero_pos = hero["pos"]
        hero_x = int(hero_pos["x"])
        hero_z = int(hero_pos["z"])
        current_pos = (hero_x, hero_z)
        hero_x_norm = _norm(hero_x, MAP_SIZE)
        hero_z_norm = _norm(hero_z, MAP_SIZE)
        flash_cd_norm = _norm(hero.get("flash_cooldown", 0), MAX_FLASH_CD)
        buff_remain_norm = _norm(hero.get("buff_remaining_time", 0), MAX_BUFF_DURATION)

        hero_feat = np.array([hero_x_norm, hero_z_norm, flash_cd_norm, buff_remain_norm], dtype=np.float32)

        # Monster features (7D x 2) / 怪物特征（含桶距离）
        monsters = frame_state.get("monsters", [])
        monster_feats = []
        visible_monster_min_dist = MAP_SIZE * 1.41
        for i in range(2):
            if i < len(monsters):
                m = monsters[i]
                is_in_view = float(m.get("is_in_view", 1))
                m_pos = m.get("pos", {})
                # 获取环境返回的桶距离和相对方向（无论是否可见都会返回）
                hero_l2_distance_bucket = int(m.get("hero_l2_distance", 0))  # 0-5
                rel_dir = int(m.get("hero_relative_direction", 0))  # 0-8
                
                if is_in_view:
                    mx = float(m_pos.get("x", 0))
                    mz = float(m_pos.get("z", 0))
                    m_x_norm = _norm(mx, MAP_SIZE)
                    m_z_norm = _norm(mz, MAP_SIZE)
                    m_speed_norm = _norm(m.get("speed", 1), MAX_MONSTER_SPEED)
                    raw_dist = np.sqrt((hero_x - mx) ** 2 + (hero_z - mz) ** 2)
                    dist_norm = _norm(raw_dist, MAP_SIZE * 1.41)
                    visible_monster_min_dist = min(visible_monster_min_dist, raw_dist)
                else:
                    m_x_norm = 0.0
                    m_z_norm = 0.0
                    m_speed_norm = 0.0
                    dist_norm = 1.0
                
                # 标准化桶距离（0-5 -> 0-1）和相对方向（0-8 -> 0-1）
                hero_l2_distance_norm = _norm(hero_l2_distance_bucket, 5.0)
                rel_dir_norm = _norm(rel_dir, 8.0)
                
                monster_feats.append(
                    np.array([is_in_view, m_x_norm, m_z_norm, m_speed_norm, dist_norm, hero_l2_distance_norm, rel_dir_norm], dtype=np.float32)
                )
            else:
                monster_feats.append(np.zeros(7, dtype=np.float32))

        organs = frame_state.get("organs", [])
        treasures = [o for o in organs if int(o.get("sub_type", 0)) == 1 and int(o.get("status", 0)) == 1]
        buffs = [o for o in organs if int(o.get("sub_type", 0)) == 2 and int(o.get("status", 0)) == 1]

        # Nearest treasure feature (4D): x, z, dist, direction
        nearest_treasure_feat = self._nearest_organ_feature(treasures, hero_x, hero_z)

        # Nearest buff feature (4D): x, z, dist, direction
        nearest_buff_feat = self._nearest_organ_feature(buffs, hero_x, hero_z)

        # Local map channels: terrain + treasure + buff + monster / 4x21x21
        terrain_map = self._extract_terrain_local_map(map_info)
        treasure_map = self._build_entity_local_map(treasures, hero_x, hero_z)
        buff_map = self._build_entity_local_map(buffs, hero_x, hero_z)
        monster_map = self._build_entity_local_map(monsters, hero_x, hero_z, require_in_view=True)
        multi_map_feat = np.concatenate(
            [
                terrain_map.reshape(-1),
                treasure_map.reshape(-1),
                buff_map.reshape(-1),
                monster_map.reshape(-1),
            ]
        ).astype(np.float32)

        # Update global explored map and get exploration reward
        newly_explored = self._update_global_map(map_info, hero_x, hero_z)
        exploration_reward = EXPLORE_REWARD_PER_CELL * newly_explored

        # Legal action mask (16D) / 合法动作掩码
        legal_action = [1] * ACTION_DIM
        if isinstance(legal_act_raw, list) and legal_act_raw:
            if isinstance(legal_act_raw[0], bool):
                for j in range(min(ACTION_DIM, len(legal_act_raw))):
                    legal_action[j] = int(legal_act_raw[j])
            else:
                valid_set = {int(a) for a in legal_act_raw if int(a) < ACTION_DIM}
                legal_action = [1 if j in valid_set else 0 for j in range(ACTION_DIM)]

        if sum(legal_action) == 0:
            legal_action = [1] * ACTION_DIM

        legal_action = self._apply_directional_terrain_mask(legal_action, terrain_map)

        if sum(legal_action) == 0:
            legal_action = [1] * ACTION_DIM

        # Progress features (2D) / 进度特征
        step_norm = _norm(self.step_no, self.max_step)
        survival_ratio = step_norm
        progress_feat = np.array([step_norm, survival_ratio], dtype=np.float32)

        # 10 steps ago position features (2D) / 10步前位置特征
        pos_10steps_ago_x_norm = 0.0
        pos_10steps_ago_z_norm = 0.0
        if len(self.position_window) == WANDER_WINDOW_SIZE:
            pos_10_steps_ago = self.position_window[0]
            pos_10steps_ago_x_norm = _norm(pos_10_steps_ago[0], MAP_SIZE)
            pos_10steps_ago_z_norm = _norm(pos_10_steps_ago[1], MAP_SIZE)
        pos_10steps_ago_feat = np.array([pos_10steps_ago_x_norm, pos_10steps_ago_z_norm], dtype=np.float32)

        # Concatenate features / 拼接特征
        feature = np.concatenate(
            [
                hero_feat,
                monster_feats[0],
                monster_feats[1],
                nearest_treasure_feat,
                nearest_buff_feat,
                multi_map_feat,
                np.array(legal_action, dtype=np.float32),
                progress_feat,
                pos_10steps_ago_feat,
            ]
        )

        # Organ potential shaping / 物件势能塑形（缓衰减）
        organ_potential = ORGAN_POTENTIAL_SCALE * (
            cur_treasures_collected + BUFF_SHAPING_MULT * cur_collected_buff
        )
        for organ in organs:
            if int(organ.get("status", 0)) != 1:
                continue
            o_pos = organ.get("pos", {})
            ox = float(o_pos.get("x", 0))
            oz = float(o_pos.get("z", 0))
            raw_dist = np.sqrt((hero_x - ox) ** 2 + (hero_z - oz) ** 2)
            subtype = int(organ.get("sub_type", 0))
            type_weight = BUFF_SHAPING_MULT if subtype == 2 else 1.0
            organ_potential += type_weight * ORGAN_POTENTIAL_SCALE * np.exp(-ORGAN_POTENTIAL_DECAY * raw_dist)

        if self.last_organ_potential is None:
            organ_shaping = 0.0
        else:
            organ_shaping = Config.GAMMA * organ_potential - self.last_organ_potential

        # Nonlinear monster-distance shaping / 怪物距离非线性塑形
        if visible_monster_min_dist >= MAP_SIZE * 1.41:
            visible_monster_min_dist = MAP_SIZE * 1.41
        monster_potential = -MONSTER_POTENTIAL_SCALE * np.exp(
            -MONSTER_POTENTIAL_DECAY * visible_monster_min_dist
        )
        if self.last_monster_potential is None:
            monster_shaping = 0.0
        else:
            monster_shaping = Config.GAMMA * monster_potential - self.last_monster_potential

        treasure_inc = max(0, cur_treasures_collected - self.last_treasures_collected)
        buff_inc = max(0, cur_collected_buff - self.last_collected_buff)
        collection_reward = TREASURE_INC_REWARD * treasure_inc + BUFF_INC_REWARD * buff_inc

        # Flash action index [8, 15]
        flash_escape_reward = (
            FLASH_ESCAPE_REWARD
            if (
                last_action is not None
                and int(last_action) >= 8
                and int(last_action) < ACTION_DIM
                and monster_shaping > 0.0
            )
            else 0.0
        )

        action_fail_penalty = 0.0
        if (
            last_action is not None
            and self.last_hero_pos is not None
            and 0 <= int(last_action) < ACTION_DIM
            and self.last_hero_pos[0] == hero_x
            and self.last_hero_pos[1] == hero_z
        ):
            action_fail_penalty = ACTION_FAIL_PENALTY

        self.position_window.append(current_pos)
        wander_penalty = 0.0
        if len(self.position_window) == WANDER_WINDOW_SIZE:
            start_x, start_z = self.position_window[0]
            dx = self.position_window[-1][0] - start_x
            dz = self.position_window[-1][1] - start_z
            # Minimum steps to achieve net displacement (diagonals cover both axes)
            min_steps = max(abs(dx), abs(dz))
            wasted_steps = WANDER_WINDOW_SIZE - min_steps
            if wasted_steps > WASTE_STEP_THRESHOLD:
                wander_penalty = WASTE_STEP_PENALTY * (wasted_steps - WASTE_STEP_THRESHOLD)

        survive_reward = 0.01

        self.last_treasures_collected = cur_treasures_collected
        self.last_collected_buff = cur_collected_buff
        self.last_organ_potential = organ_potential
        self.last_monster_potential = monster_potential
        self.last_hero_pos = current_pos

        reward = [
            survive_reward
            + collection_reward
            + organ_shaping
            + monster_shaping
            + exploration_reward
            # - action_fail_penalty
            - wander_penalty
            # + flash_escape_reward
        ]

        return feature, legal_action, reward

    def _nearest_organ_feature(self, organs, hero_x, hero_z):
        feat = np.zeros(4, dtype=np.float32)
        if not organs:
            return feat

        nearest = min(
            organs,
            key=lambda item: np.sqrt(
                (hero_x - float(item.get("pos", {}).get("x", 0))) ** 2
                + (hero_z - float(item.get("pos", {}).get("z", 0))) ** 2
            ),
        )
        pos = nearest.get("pos", {})
        ox = float(pos.get("x", 0))
        oz = float(pos.get("z", 0))
        raw_dist = np.sqrt((hero_x - ox) ** 2 + (hero_z - oz) ** 2)
        feat[:] = np.array(
            [
                _norm(ox, MAP_SIZE),
                _norm(oz, MAP_SIZE),
                _norm(raw_dist, MAP_SIZE * 1.41),
                _norm(nearest.get("hero_relative_direction", 0), MAX_REL_DIR),
            ],
            dtype=np.float32,
        )
        return feat

    def _extract_terrain_local_map(self, map_info):
        terrain = np.zeros((LOCAL_VIEW_SIZE, LOCAL_VIEW_SIZE), dtype=np.float32)
        if not isinstance(map_info, list) or not map_info or not isinstance(map_info[0], list):
            return terrain

        src_h = len(map_info)
        src_w = len(map_info[0]) if src_h > 0 else 0
        if src_w <= 0:
            return terrain

        row_start = max((src_h - LOCAL_VIEW_SIZE) // 2, 0)
        col_start = max((src_w - LOCAL_VIEW_SIZE) // 2, 0)
        for r in range(LOCAL_VIEW_SIZE):
            src_r = row_start + r
            if src_r < 0 or src_r >= src_h:
                continue
            for c in range(LOCAL_VIEW_SIZE):
                src_c = col_start + c
                if src_c < 0 or src_c >= src_w:
                    continue
                terrain[r, c] = float(map_info[src_r][src_c] != 0)
        return terrain

    def _build_entity_local_map(self, entities, hero_x, hero_z, require_in_view=False):
        entity_map = np.zeros((LOCAL_VIEW_SIZE, LOCAL_VIEW_SIZE), dtype=np.float32)
        center = LOCAL_VIEW_SIZE // 2
        for item in entities:
            if require_in_view and float(item.get("is_in_view", 1)) <= 0:
                continue
            pos = item.get("pos", {})
            ox = int(pos.get("x", 0))
            oz = int(pos.get("z", 0))
            row = center + (oz - hero_z)
            col = center + (ox - hero_x)
            if 0 <= row < LOCAL_VIEW_SIZE and 0 <= col < LOCAL_VIEW_SIZE:
                entity_map[row, col] = 1.0
        return entity_map

    def _apply_directional_terrain_mask(self, legal_action, terrain_map):
        if terrain_map.size == 0:
            return legal_action

        center = LOCAL_VIEW_SIZE // 2
        if terrain_map[center, center] <= 0.5:
            return legal_action

        masked_action = list(legal_action)
        for action_idx in range(min(ACTION_DIM, 16)):
            direction_idx = action_idx % 8
            dr, dc = DIRECTION_OFFSETS[direction_idx]
            next_row = center + dr
            next_col = center + dc

            if not (0 <= next_row < LOCAL_VIEW_SIZE and 0 <= next_col < LOCAL_VIEW_SIZE):
                masked_action[action_idx] = 0
                continue

            if masked_action[action_idx] == 0:
                continue

            if terrain_map[next_row, next_col] <= 0.5:
                masked_action[action_idx] = 0

        return masked_action

    def _update_global_map(self, map_info, hero_x, hero_z):
        if not isinstance(map_info, list) or not map_info or not isinstance(map_info[0], list):
            return 0

        src_h = len(map_info)
        src_w = len(map_info[0]) if src_h > 0 else 0
        if src_h <= 0 or src_w <= 0:
            return 0

        view_size = min(src_h, src_w, LOCAL_VIEW_SIZE)
        center = view_size // 2
        row_start = max((src_h - view_size) // 2, 0)
        col_start = max((src_w - view_size) // 2, 0)

        newly_explored = 0
        for r in range(view_size):
            for c in range(view_size):
                src_r = row_start + r
                src_c = col_start + c
                gx = hero_x + (c - center)
                gz = hero_z + (r - center)
                if gx < 0 or gx >= WORLD_MAP_SIZE or gz < 0 or gz >= WORLD_MAP_SIZE:
                    continue

                cell_state = 1.0 if map_info[src_r][src_c] != 0 else 0.0
                if self.global_map[gz, gx] < 0:
                    newly_explored += 1
                self.global_map[gz, gx] = cell_state

        self.explored_cell_count += newly_explored
        return newly_explored
