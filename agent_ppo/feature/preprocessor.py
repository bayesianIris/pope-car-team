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

import numpy as np

from agent_ppo.conf.conf import Config

# Map size / 地图尺寸（128×128）
MAP_SIZE = 128.0
MAX_MAP_DISTANCE = MAP_SIZE * 1.41
# Max monster speed / 最大怪物速度
MAX_MONSTER_SPEED = 5.0
# Max distance bucket / 距离桶最大值
MAX_DIST_BUCKET = 5.0
# Max flash cooldown / 最大闪现冷却步数
MAX_FLASH_CD = 2000.0
# Max buff duration / buff最大持续时间
MAX_BUFF_DURATION = 50.0
# Pickup rewards / 拾取奖励系数
TREASURE_PICKUP_REWARD = 2.0
BUFF_PICKUP_REWARD = 0.8
# Proximity shaping weights / 接近目标奖励系数
TREASURE_PROXIMITY_WEIGHT = 0.5
TREASURE_PROXIMITY_LAMBDA = 0.5
BUFF_PROXIMITY_WEIGHT = TREASURE_PROXIMITY_WEIGHT * 0.8 / 2
BUFF_PROXIMITY_LAMBDA = 0.5
MONSTER_PROXIMITY_WEIGHT = 2.0  # Increased base weight
MONSTER_PROXIMITY_LAMBDA = 0.85 # Fades much slower, giving strong signal from afar
MONSTER_UNSEEN_DISTANCE = 12.0


def _norm(v, v_max, v_min=0.0):
    """Normalize value to [0, 1].

    将值归一化到 [0, 1]。
    """
    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0


def _signed_norm(v, v_abs_max):
    """Normalize signed value to [-1, 1].

    将带符号的值归一化到 [-1, 1]。
    """
    v_abs_max = float(v_abs_max)
    if v_abs_max <= 1e-6:
        return 0.0
    return float(np.clip(v / v_abs_max, -1.0, 1.0))


class Preprocessor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.step_no = 0
        self.max_step = 200
        self.last_min_monster_dist = MONSTER_UNSEEN_DISTANCE
        self.last_nearest_treasure_dist = MAX_MAP_DISTANCE
        self.last_nearest_buff_dist = MAX_MAP_DISTANCE
        self.last_treasures_collected = None
        self.last_buffs_collected = None
        self.prev_hero_pos = None
        self.prev_legal_action = None
        self.global_map = np.full((128, 128), -1.0, dtype=np.float32)

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

        # Hero self features (4D) / 英雄自身特征
        hero = frame_state["heroes"]
        hero_pos = hero["pos"]
        hero_x_norm = _norm(hero_pos["x"], MAP_SIZE)
        hero_z_norm = _norm(hero_pos["z"], MAP_SIZE)
        flash_cd_norm = _norm(hero["flash_cooldown"], MAX_FLASH_CD)
        buff_remain_norm = _norm(hero["buff_remaining_time"], MAX_BUFF_DURATION)

        hero_feat = np.array([hero_x_norm, hero_z_norm, flash_cd_norm, buff_remain_norm], dtype=np.float32)

        # Monster features (5D x 2) / 怪物特征
        monsters = frame_state.get("monsters", [])
        monster_feats = []
        cur_min_monster_dist = MONSTER_UNSEEN_DISTANCE
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
                    dist_norm = _norm(raw_dist, MAX_MAP_DISTANCE)
                    cur_min_monster_dist = min(cur_min_monster_dist, float(np.clip(raw_dist, 0.0, MONSTER_UNSEEN_DISTANCE)))
                else:
                    m_x_norm = 0.0
                    m_z_norm = 0.0
                    m_speed_norm = 0.0
                    dist_norm = 1.0
                monster_feats.append(
                    np.array([is_in_view, m_x_norm, m_z_norm, m_speed_norm, dist_norm], dtype=np.float32)
                )
            else:
                monster_feats.append(np.zeros(5, dtype=np.float32))

        # Organ features (treasure + buff, 8D) / 物件特征（宝箱+buff）
        organs = frame_state.get("organs", [])
        nearest_treasure_dist_norm, nearest_treasure_dist, nearest_treasure_dir_norm, treasure_available = self._nearest_organ_feature(
            organs, hero_pos, sub_type=1
        )
        nearest_buff_dist_norm, nearest_buff_dist, nearest_buff_dir_norm, buff_available = self._nearest_organ_feature(
            organs, hero_pos, sub_type=2
        )

        total_treasure = max(1, int(env_info.get("total_treasure", 1)))
        total_buff = max(1, int(env_info.get("total_buff", 1)))
        treasures_collected = int(env_info.get("treasures_collected", hero.get("treasure_collected_count", 0)))
        buffs_collected = int(env_info.get("collected_buff", 0))

        treasure_progress = _norm(treasures_collected, total_treasure)
        buff_progress = _norm(buffs_collected, total_buff)

        organ_feat = np.array(
            [
                nearest_treasure_dist_norm,
                nearest_treasure_dir_norm,
                float(treasure_available),
                nearest_buff_dist_norm,
                nearest_buff_dir_norm,
                float(buff_available),
                treasure_progress,
                buff_progress,
            ],
            dtype=np.float32,
        )

        # Full map view for CNN / 地图全视野 CNN 输入
        map_feat = self._build_map_feature(map_info, hero_pos, frame_state)
        
        # Directional fog features / 四方向迷雾占比特征
        global_map_feat, newly_explored = self._update_directional_fog_feature(map_info, hero_pos)

        # Legal action mask (16D) / 合法动作掩码
        legal_action = [1] * Config.ACTION_NUM
        if isinstance(legal_act_raw, list) and legal_act_raw:
            if isinstance(legal_act_raw[0], bool):
                for j in range(min(Config.ACTION_NUM, len(legal_act_raw))):
                    legal_action[j] = int(legal_act_raw[j])
            else:
                valid_set = {int(a) for a in legal_act_raw if 0 <= int(a) < Config.ACTION_NUM}
                legal_action = [1 if j in valid_set else 0 for j in range(Config.ACTION_NUM)]

        if sum(legal_action) == 0:
            legal_action = [1] * Config.ACTION_NUM

        # Progress features (2D) / 进度特征
        # DEBUG：这俩为啥一样
        step_norm = _norm(self.step_no, self.max_step)
        survival_ratio = step_norm
        progress_feat = np.array([step_norm, survival_ratio], dtype=np.float32)

        # Concatenate features / 拼接特征
        feature = np.concatenate(
            [
                hero_feat,
                monster_feats[0],
                monster_feats[1],
                organ_feat,
                np.array(legal_action, dtype=np.float32),
                progress_feat,
                map_feat,
                global_map_feat,
            ]
        )

        # Step reward / 即时奖励
        survive_reward = 0.03
        
        # 根据需求只保留躲避怪物和探索地图任务的奖励
        monster_dist_reward = (
            MONSTER_PROXIMITY_WEIGHT * (MONSTER_PROXIMITY_LAMBDA ** (self.last_min_monster_dist - 1.0))
            - Config.GAMMA * MONSTER_PROXIMITY_WEIGHT * (MONSTER_PROXIMITY_LAMBDA ** (cur_min_monster_dist - 1.0))
        )
        
        # 每探索到一个全新的可行走/障碍格子，给予 0.002 分
        explore_reward = newly_explored * 0.002

        # Reward for moving closer to targets / 接近宝箱和buff奖励
        # treasure_close_reward = (
        #     Config.GAMMA * TREASURE_PROXIMITY_WEIGHT * (TREASURE_PROXIMITY_LAMBDA ** (nearest_treasure_dist - 1.0))
        #     - TREASURE_PROXIMITY_WEIGHT
        #     * (TREASURE_PROXIMITY_LAMBDA ** (self.last_nearest_treasure_dist - 1.0))
        # )
        # buff_close_reward = (
        #     Config.GAMMA * BUFF_PROXIMITY_WEIGHT * (BUFF_PROXIMITY_LAMBDA ** (nearest_buff_dist - 1.0))
        #     - BUFF_PROXIMITY_WEIGHT * (BUFF_PROXIMITY_LAMBDA ** (self.last_nearest_buff_dist - 1.0))
        # )

        if self.last_treasures_collected is None:
            self.last_treasures_collected = treasures_collected
        if self.last_buffs_collected is None:
            self.last_buffs_collected = buffs_collected

        picked_treasures = max(0, treasures_collected - self.last_treasures_collected)
        picked_buffs = max(0, buffs_collected - self.last_buffs_collected)
        pickup_reward = picked_treasures * TREASURE_PICKUP_REWARD + picked_buffs * BUFF_PICKUP_REWARD

        # Action failure penalties / 动作失败惩罚
        action_penalty = 0.0
        if self.prev_hero_pos is not None and int(last_action) >= 0:
            last_action_int = int(last_action)
            was_legal = True
            if self.prev_legal_action is not None and 0 <= last_action_int < len(self.prev_legal_action):
                was_legal = bool(self.prev_legal_action[last_action_int])

            moved = (
                int(hero_pos.get("x", 0)) != int(self.prev_hero_pos[0])
                or int(hero_pos.get("z", 0)) != int(self.prev_hero_pos[1])
            )

            if not was_legal:
                action_penalty -= Config.ILLEGAL_ACTION_PENALTY
            elif not moved:
                action_penalty -= Config.ACTION_FAIL_PENALTY

        self.last_min_monster_dist = cur_min_monster_dist
        self.last_nearest_treasure_dist = nearest_treasure_dist
        self.last_nearest_buff_dist = nearest_buff_dist
        self.last_treasures_collected = treasures_collected
        self.last_buffs_collected = buffs_collected
        self.prev_hero_pos = (int(hero_pos.get("x", 0)), int(hero_pos.get("z", 0)))
        self.prev_legal_action = list(legal_action)

        reward = [
            survive_reward + monster_dist_reward + explore_reward + pickup_reward + action_penalty
        ]

        return feature, legal_action, reward

    def _nearest_organ_feature(self, organs, hero_pos, sub_type):
        """Get nearest organ distance/direction feature by sub type.

        按类型提取最近物件的距离和方向特征。
        """
        nearest_dist_norm = 1.0
        nearest_dist = MAX_MAP_DISTANCE
        nearest_dir_norm = 0.0
        available = False

        for organ in organs:
            if int(organ.get("sub_type", 0)) != int(sub_type):
                continue
            if int(organ.get("status", 0)) != 1:
                continue

            available = True
            organ_pos = organ.get("pos", {})
            raw_dist = np.sqrt(
                (float(hero_pos.get("x", 0)) - float(organ_pos.get("x", 0))) ** 2
                + (float(hero_pos.get("z", 0)) - float(organ_pos.get("z", 0))) ** 2
            )
            dist = float(raw_dist)

            if dist < nearest_dist:
                nearest_dist = dist
                nearest_dist_norm = _norm(dist, MAX_MAP_DISTANCE)
                nearest_dir_norm = _norm(organ.get("hero_relative_direction", 0), 8.0)

        return nearest_dist_norm, nearest_dist, nearest_dir_norm, available

    def _build_map_feature(self, map_info, hero_pos, frame_state):
        """Build a fixed-size map image feature.

        构造以英雄为中心的固定尺寸地图图像特征；看不到的位置使用默认值填充。
        """
        view_size = Config.MAP_VIEW_SIZE
        view_radius = Config.MAP_VIEW_RADIUS
        default_value = np.float32(Config.MAP_DEFAULT_VALUE)
        # Channel 0: passable map, Channel 1: monsters in view, Channel 2: treasures.
        map_feat = np.full((Config.MAP_CHANNELS, view_size, view_size), default_value, dtype=np.float32)

        if map_info is None:
            return map_feat.reshape(-1)

        try:
            src_h = len(map_info)
        except TypeError:
            return map_feat.reshape(-1)

        src_w = len(map_info[0]) if src_h > 0 else 0
        if src_h == 0 or src_w == 0:
            return map_feat.reshape(-1)

        center_row = int(round(float(hero_pos["z"])))
        center_col = int(round(float(hero_pos["x"])))

        for row in range(view_size):
            src_row = center_row - view_radius + row
            if not (0 <= src_row < src_h):
                continue
            for col in range(view_size):
                src_col = center_col - view_radius + col
                if 0 <= src_col < src_w:
                    cell = map_info[src_row][src_col]
                    map_feat[0, row, col] = float(cell != 0)

        # Monster channel / 怪物通道（可见怪物）
        monsters = frame_state.get("monsters", []) if isinstance(frame_state, dict) else []
        hero_x = int(round(float(hero_pos.get("x", 0))))
        hero_z = int(round(float(hero_pos.get("z", 0))))
        for monster in monsters:
            if float(monster.get("is_in_view", 0)) <= 0:
                continue
            m_pos = monster.get("pos", {})
            m_col = int(round(float(m_pos.get("x", 0)))) - hero_x + view_radius
            m_row = int(round(float(m_pos.get("z", 0)))) - hero_z + view_radius
            if 0 <= m_row < view_size and 0 <= m_col < view_size:
                map_feat[1, m_row, m_col] = 1.0

        # Treasure channel / 宝箱通道（status=1 可拾取）
        organs = frame_state.get("organs", []) if isinstance(frame_state, dict) else []
        for organ in organs:
            if int(organ.get("sub_type", 0)) != 1:
                continue
            if int(organ.get("status", 0)) != 1:
                continue
            t_pos = organ.get("pos", {})
            t_col = int(round(float(t_pos.get("x", 0)))) - hero_x + view_radius
            t_row = int(round(float(t_pos.get("z", 0)))) - hero_z + view_radius
            if 0 <= t_row < view_size and 0 <= t_col < view_size:
                map_feat[2, t_row, t_col] = 1.0

        return map_feat.reshape(-1)

    def _update_global_map(self, map_info, hero_pos):
        if map_info is None:
            return self.global_map.reshape(-1), 0

        try:
            src_h = len(map_info)
        except TypeError:
            return self.global_map.reshape(-1), 0

        src_w = len(map_info[0]) if src_h > 0 else 0
        if src_h == 0 or src_w == 0:
            return self.global_map.reshape(-1), 0

        view_radius = Config.MAP_VIEW_RADIUS
        center_row = int(round(float(hero_pos["z"])))
        center_col = int(round(float(hero_pos["x"])))
        
        newly_explored = 0
        for row in range(src_h):
            global_row = center_row - view_radius + row
            if not (0 <= global_row < 128):
                continue
            for col in range(src_w):
                global_col = center_col - view_radius + col
                if 0 <= global_col < 128:
                    if self.global_map[global_row, global_col] == -1.0:
                        newly_explored += 1
                    cell = map_info[row][col]
                    self.global_map[global_row, global_col] = float(cell != 0)

        return self.global_map.reshape(-1), newly_explored

    def _update_directional_fog_feature(self, map_info, hero_pos):
        """Update explored map cache and build four directional fog ratios.

        更新已探索地图缓存，并构造上下左右四个方向的迷雾占比特征。
        """
        if map_info is None:
            return np.zeros(4, dtype=np.float32), 0

        try:
            src_h = len(map_info)
        except TypeError:
            return np.zeros(4, dtype=np.float32), 0

        src_w = len(map_info[0]) if src_h > 0 else 0
        if src_h == 0 or src_w == 0:
            return np.zeros(4, dtype=np.float32), 0

        view_radius = Config.MAP_VIEW_RADIUS
        hero_x = int(round(float(hero_pos.get("x", 0))))
        hero_z = int(round(float(hero_pos.get("z", 0))))

        # Update explored cache with current visible map / 用当前可见地图更新已探索缓存
        newly_explored = 0
        for row in range(src_h):
            global_row = hero_z - view_radius + row
            if not (0 <= global_row < 128):
                continue
            for col in range(src_w):
                global_col = hero_x - view_radius + col
                if 0 <= global_col < 128:
                    if self.global_map[global_row, global_col] == -1.0:
                        newly_explored += 1
                    cell = map_info[row][col]
                    self.global_map[global_row, global_col] = float(cell != 0)

        def fog_ratio(row_start, row_end, col_start, col_end):
            row_start = max(0, row_start)
            row_end = min(128, row_end)
            col_start = max(0, col_start)
            col_end = min(128, col_end)
            if row_end <= row_start or col_end <= col_start:
                return 0.0
            area = self.global_map[row_start:row_end, col_start:col_end]
            return float(np.mean(area == -1.0))

        top = hero_z - view_radius
        bottom = hero_z + view_radius + 1
        left = hero_x - view_radius
        right = hero_x + view_radius + 1

        fog_feat = np.array(
            [
                fog_ratio(0, top, left, right),
                fog_ratio(bottom, 128, left, right),
                fog_ratio(top, bottom, 0, left),
                fog_ratio(top, bottom, right, 128),
            ],
            dtype=np.float32,
        )

        return fog_feat, newly_explored
