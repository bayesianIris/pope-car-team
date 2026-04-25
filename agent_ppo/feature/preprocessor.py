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

# Reward for each newly collected treasure / 每新增一个宝箱奖励
TREASURE_INC_REWARD = 0.66
# Reward for each newly collected buff / 每新增一个buff奖励
BUFF_INC_REWARD = 0.4
# Small exploration reward coefficient / 探索奖励系数（小）
EXPLORE_REWARD_PER_CELL = 0.001
# Revisit penalty per extra visit / 重复走位惩罚系数
REVISIT_PENALTY_PER_STEP = 0.005
# Flash use penalty / 闪现基础使用惩罚
FLASH_USE_PENALTY = 0.03
# Flash wall-pass compensation / 闪现穿墙补偿
FLASH_WALL_COMPENSATION = FLASH_USE_PENALTY + 0.1
# Penalty when action does not move hero position / 动作未发生位移时惩罚
ACTION_FAIL_PENALTY = 0.05
# Penalty when 10-step window Manhattan progress is too small / 10步窗口曼哈顿进展过小时惩罚
WANDER_PENALTY = 0.02
WANDER_WINDOW_SIZE = 10
WANDER_MANHATTAN_THRESHOLD = 4
FRONTIER_RADIUS = 10
PASSABILITY_RADIUS = 2
MAX_CONE_OPEN_DIRECTIONS = 8.0
SAFETY_MARGIN_SCALE = 20.0
MIN_SPEED_EPS = 0.1

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
        self.unexplored_cell_count = WORLD_MAP_SIZE * WORLD_MAP_SIZE
        self.visit_count_map = np.zeros((WORLD_MAP_SIZE, WORLD_MAP_SIZE), dtype=np.int32)
        self.last_hero_pos = None
        self.position_window = deque(maxlen=WANDER_WINDOW_SIZE)
        self.buff_remain = 0
        self.last_flash_count = None
        self.flash_cd_remain = 0.0

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

        # Monster features (7D x 2) / 怪物特征：is_in_view, x, z, speed, dist, relative_direction, l2_distance_bucket
        monsters = frame_state.get("monsters", [])
        monster_feats = []
        visible_monster_min_dist = MAP_SIZE * 1.41
        
        for i in range(2):
            if i < len(monsters):
                m = monsters[i]
                is_in_view = float(m.get("is_in_view", 1))
                m_pos = m.get("pos", {})
                
                # Always provide hero_relative_direction and hero_l2_distance regardless of in_view
                rel_dir_norm = _norm(m.get("hero_relative_direction", 0), MAX_REL_DIR)
                hero_l2_dist_norm = _norm(m.get("hero_l2_distance", 0), 5.0)
                
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
                
                monster_feats.append(
                    np.array([is_in_view, m_x_norm, m_z_norm, m_speed_norm, dist_norm, rel_dir_norm, hero_l2_dist_norm], dtype=np.float32)
                )
            else:
                monster_feats.append(np.zeros(7, dtype=np.float32))
                # print("monster_teats:", monster_feats)

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
        newly_explored, revisit_penalty = self._update_global_map(map_info, hero_x, hero_z)
        exploration_reward = EXPLORE_REWARD_PER_CELL * newly_explored
        exploration_ratio = float(self.explored_cell_count) / float(WORLD_MAP_SIZE * WORLD_MAP_SIZE)

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

        # Progress features (2D) / 进度特征
        step_norm = _norm(self.step_no, self.max_step)
        survival_ratio = step_norm
        progress_feat = np.array([step_norm, survival_ratio], dtype=np.float32)


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
        flash_use_penalty = 0.0
        flash_wall_compensation = 0.0
        if (
            last_action is not None
            and int(last_action) >= 8
            and int(last_action) < ACTION_DIM
        ):
            flash_use_penalty = FLASH_USE_PENALTY
            if self._flash_passed_wall(terrain_map):
                flash_wall_compensation = FLASH_WALL_COMPENSATION

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
            manhattan_dist = abs(hero_x - start_x) + abs(hero_z - start_z)
            if manhattan_dist < WANDER_MANHATTAN_THRESHOLD:
                wander_penalty = WANDER_PENALTY

        survive_reward = 0.01

        self.last_treasures_collected = cur_treasures_collected
        self.last_collected_buff = cur_collected_buff
        self.last_organ_potential = organ_potential
        self.last_monster_potential = monster_potential
        self.last_hero_pos = current_pos

        # /二级特征
        # 危险度特征
        if buff_inc > 0:
            self.buff_remain = MAX_BUFF_DURATION
        else:
            self.buff_remain = max(0, self.buff_remain - 1)
        danger = 0.0
        for i in range(2):
            if i >= len(monsters):
                continue
            m = monsters[i]
            if float(m.get("is_in_view", 1)) <= 0:
                continue

            m_pos = m.get("pos", {})
            mx = float(m_pos.get("x", 0))
            mz = float(m_pos.get("z", 0))
            raw_dist = np.sqrt((hero_x - mx) ** 2 + (hero_z - mz) ** 2)

            # Danger uses potential-like distance decay, scaled by monster speed.
            m_danger = (
                MONSTER_POTENTIAL_SCALE
                * np.exp(-MONSTER_POTENTIAL_DECAY * raw_dist)
                * float(m.get("speed", 1))
            )
            if self.buff_remain > 0:
                m_danger /= 2.0

            danger = max(danger, m_danger)
        danger = float(danger)

        # Flash cooldown feature / 闪现冷却就绪特征
        flash_count = int(hero.get("flash_count", 0))
        flash_cd_config = float(hero.get("flash_cooldown", MAX_FLASH_CD))
        if self.last_flash_count is None:
            # Align internal cooldown state with the first frame observation.
            self.flash_cd_remain = max(0.0, flash_cd_config)
        elif flash_count != self.last_flash_count:
            # Re-fire detected by flash_count change, reset cooldown.
            self.flash_cd_remain = max(0.0, flash_cd_config)
        else:
            self.flash_cd_remain = max(0.0, self.flash_cd_remain - 1.0)
        self.last_flash_count = flash_count
        flash_ready = 1.0 if self.flash_cd_remain <= 1e-6 else 0.0

        # Safety margin: time-to-monster minus time-to-treasure (normalized)
        nearest_treasure_dist = MAP_SIZE * 1.41
        if treasures:
            nearest_treasure_dist = min(
                np.sqrt(
                    (hero_x - float(item.get("pos", {}).get("x", 0))) ** 2
                    + (hero_z - float(item.get("pos", {}).get("z", 0))) ** 2
                )
                for item in treasures
            )
        hero_speed = max(float(hero.get("speed", 1.0)), MIN_SPEED_EPS)
        tta_treasure = nearest_treasure_dist / hero_speed

        tta_monster = MAP_SIZE * 1.41 / MIN_SPEED_EPS
        for i in range(2):
            if i >= len(monsters):
                continue
            m = monsters[i]
            if float(m.get("is_in_view", 1)) <= 0:
                continue
            m_pos = m.get("pos", {})
            mx = float(m_pos.get("x", 0))
            mz = float(m_pos.get("z", 0))
            m_dist = np.sqrt((hero_x - mx) ** 2 + (hero_z - mz) ** 2)
            m_speed = max(float(m.get("speed", 1.0)), MIN_SPEED_EPS)
            tta_monster = min(tta_monster, m_dist / m_speed)

        safety_margin = float(np.tanh((tta_monster - tta_treasure) / SAFETY_MARGIN_SCALE))
        frontier_density = self._frontier_density(hero_x, hero_z, FRONTIER_RADIUS)
        passability_width = self._passability_width(terrain_map, PASSABILITY_RADIUS)
        cone_open_dirs = self._cone_open_directions(terrain_map)
        cone_open_dirs_norm = _norm(cone_open_dirs, MAX_CONE_OPEN_DIRECTIONS)

        # 10-step-old position feature for anti-wander context.
        if len(self.position_window) == WANDER_WINDOW_SIZE:
            past_x, past_z = self.position_window[0]
        else:
            past_x, past_z = current_pos
        past10_x_norm = _norm(past_x, MAP_SIZE)
        past10_z_norm = _norm(past_z, MAP_SIZE)

        secondary_feat = np.array(
            [
                danger,
                flash_ready,
                safety_margin,
                frontier_density,
                passability_width,
                cone_open_dirs_norm,
                past10_x_norm,
                past10_z_norm,
                exploration_ratio,
            ],
            dtype=np.float32,
        )
        
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
                secondary_feat,
            ]
        )


        reward = [
            survive_reward
            + collection_reward
            + organ_shaping
            + monster_shaping
            + exploration_reward
            - revisit_penalty
            - action_fail_penalty
            - wander_penalty
            - flash_use_penalty
            + flash_wall_compensation
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

    def _update_global_map(self, map_info, hero_x, hero_z):
        if not isinstance(map_info, list) or not map_info or not isinstance(map_info[0], list):
            return 0, 0.0

        src_h = len(map_info)
        src_w = len(map_info[0]) if src_h > 0 else 0
        if src_h <= 0 or src_w <= 0:
            return 0, 0.0

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
        self.unexplored_cell_count = max(WORLD_MAP_SIZE * WORLD_MAP_SIZE - self.explored_cell_count, 0)

        visit_penalty = 0.0
        if 0 <= hero_x < WORLD_MAP_SIZE and 0 <= hero_z < WORLD_MAP_SIZE:
            self.visit_count_map[hero_z, hero_x] += 1
            visit_count = int(self.visit_count_map[hero_z, hero_x])
            if visit_count > 1:
                visit_penalty = REVISIT_PENALTY_PER_STEP * float(visit_count - 1)

        return newly_explored, visit_penalty

    def _frontier_density(self, hero_x, hero_z, radius):
        total = 0
        frontier = 0
        for gz in range(max(0, hero_z - radius), min(WORLD_MAP_SIZE, hero_z + radius + 1)):
            for gx in range(max(0, hero_x - radius), min(WORLD_MAP_SIZE, hero_x + radius + 1)):
                if self.global_map[gz, gx] != -1:
                    continue
                total += 1
                for nz, nx in ((gz - 1, gx), (gz + 1, gx), (gz, gx - 1), (gz, gx + 1)):
                    if nz < 0 or nz >= WORLD_MAP_SIZE or nx < 0 or nx >= WORLD_MAP_SIZE:
                        continue
                    if self.global_map[nz, nx] >= 0:
                        frontier += 1
                        break
        return float(frontier) / float(total) if total > 0 else 0.0

    def _passability_width(self, terrain_map, radius):
        if terrain_map.size == 0:
            return 0.0
        center = LOCAL_VIEW_SIZE // 2
        r0 = max(0, center - radius)
        r1 = min(LOCAL_VIEW_SIZE, center + radius + 1)
        c0 = max(0, center - radius)
        c1 = min(LOCAL_VIEW_SIZE, center + radius + 1)
        patch = terrain_map[r0:r1, c0:c1]
        if patch.size == 0:
            return 0.0
        return float(np.mean(patch > 0.5))

    def _cone_open_directions(self, terrain_map):
        if terrain_map.size == 0:
            return 0

        center = LOCAL_VIEW_SIZE // 2
        if terrain_map[center, center] <= 0.5:
            return 0

        # Clockwise: N, NE, E, SE, S, SW, W, NW
        dirs = [
            (-1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
            (1, 0),
            (1, -1),
            (0, -1),
            (-1, -1),
        ]

        open_count = 0
        for i in range(8):
            cone_steps = [dirs[(i - 1) % 8], dirs[i], dirs[(i + 1) % 8]]

            visited = np.zeros((LOCAL_VIEW_SIZE, LOCAL_VIEW_SIZE), dtype=np.bool_)
            queue = deque([(center, center)])
            visited[center, center] = True
            reached_edge = False

            while queue:
                row, col = queue.popleft()
                if row == 0 or row == LOCAL_VIEW_SIZE - 1 or col == 0 or col == LOCAL_VIEW_SIZE - 1:
                    reached_edge = True
                    break

                for dr, dc in cone_steps:
                    next_row = row + dr
                    next_col = col + dc
                    if not (0 <= next_row < LOCAL_VIEW_SIZE and 0 <= next_col < LOCAL_VIEW_SIZE):
                        continue
                    if visited[next_row, next_col]:
                        continue
                    if terrain_map[next_row, next_col] <= 0.5:
                        continue
                    visited[next_row, next_col] = True
                    queue.append((next_row, next_col))

            if reached_edge:
                open_count += 1

        return open_count

    def _flash_passed_wall(self, terrain_map):
        if terrain_map.size == 0:
            return False

        center = LOCAL_VIEW_SIZE // 2
        if terrain_map[center, center] > 0.5:
            return True

        visited = np.zeros((LOCAL_VIEW_SIZE, LOCAL_VIEW_SIZE), dtype=np.bool_)
        queue = deque([(center, center)])
        visited[center, center] = True

        while queue:
            row, col = queue.popleft()
            if terrain_map[row, col] <= 0.5:
                return False

            for next_row, next_col in (
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            ):
                if not (0 <= next_row < LOCAL_VIEW_SIZE and 0 <= next_col < LOCAL_VIEW_SIZE):
                    continue
                if visited[next_row, next_col]:
                    continue
                if terrain_map[next_row, next_col] > 0.5:
                    continue
                visited[next_row, next_col] = True
                queue.append((next_row, next_col))

        return True
