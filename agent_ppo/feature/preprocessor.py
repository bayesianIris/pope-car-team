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
# Max monster speed / 最大怪物速度
MAX_MONSTER_SPEED = 5.0
# Max distance bucket / 距离桶最大值
MAX_DIST_BUCKET = 5.0
# Max relative direction enum / 最大相对方向枚举值
MAX_REL_DIR = 8.0
# Max flash cooldown / 最大闪现冷却步数
MAX_FLASH_CD = 2000.0
# Max buff duration / buff最大持续时间
MAX_BUFF_DURATION = 50.0
# Action dimension / 动作维度
ACTION_DIM = 16
# Local map window size (player centered) / 局部地图窗口边长（玩家居中）
LOCAL_MAP_WIN_SIZE = 9
# Treasure slot config / 宝箱槽位配置
TREASURE_SLOT_NUM = 3
TREASURE_FEAT_DIM = 5
# Keep only nearest one buff with 5D feature / buff仅保留最近1个，5维特征（含status）
BUFF_SLOT_NUM = 1
BUFF_FEAT_DIM = 5
# Reward for each newly collected treasure / 每新增一个宝箱的奖励
TREASURE_INC_REWARD = 0.66
# Reward for each newly collected buff / 每新增一个buff的奖励
BUFF_INC_REWARD = 0.4
# Flash-escape bonus with conservative PPO scale / 小尺度闪现逃脱奖励
FLASH_ESCAPE_REWARD = 0.03
# Direction preference reward with conservative PPO scale / 小尺度远离怪物方向奖励
DIRECTION_REWARD = 0.01
# Potential decay factor for organ shaping / 物件势能衰减系数（非线性，距离越远影响越小）
ORGAN_POTENTIAL_DECAY = np.log(2.0)
# Calibrate treasure shaping: when distance changes by 1 near target, reward ~= 0.1
ORGAN_TREASURE_ONE_STEP_REWARD = 0.1
ORGAN_POTENTIAL_SCALE = ORGAN_TREASURE_ONE_STEP_REWARD / (
    Config.GAMMA - np.exp(-ORGAN_POTENTIAL_DECAY)
) # DEBUG: AI写的，为啥乘以这个我不太明白，但是actually数值上影响不大，或许是为了凑0.1的整数
BUFF_SHAPING_MULT = BUFF_INC_REWARD / TREASURE_INC_REWARD


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
        self.last_min_monster_dist_norm = 0.5
        self.last_treasures_collected = 0
        self.last_collected_buff = 0
        self.last_organ_potential = None

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
        hero_x_norm = _norm(hero_pos["x"], MAP_SIZE)
        hero_z_norm = _norm(hero_pos["z"], MAP_SIZE)
        flash_cd_norm = _norm(hero["flash_cooldown"], MAX_FLASH_CD)
        buff_remain_norm = _norm(hero["buff_remaining_time"], MAX_BUFF_DURATION)

        hero_feat = np.array([hero_x_norm, hero_z_norm, flash_cd_norm, buff_remain_norm], dtype=np.float32)

        # Monster features (5D x 2) / 怪物特征
        monsters = frame_state.get("monsters", [])
        monster_feats = []
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

        # Treasure/Buff features / 宝箱与buff特征
        organs = frame_state.get("organs", [])
        treasures = [o for o in organs if int(o.get("sub_type", 0)) == 1 and int(o.get("status", 0)) == 1]
        treasures = sorted(
            treasures,
            key=lambda item: np.sqrt(
                (hero_pos["x"] - float(item.get("pos", {}).get("x", 0))) ** 2
                + (hero_pos["z"] - float(item.get("pos", {}).get("z", 0))) ** 2
            ),
        )

        treasure_feat = np.zeros(Config.FEATURES[3], dtype=np.float32)
        for idx, t in enumerate(treasures[:TREASURE_SLOT_NUM]):
            t_pos = t.get("pos", {})
            tx = float(t_pos.get("x", 0))
            tz = float(t_pos.get("z", 0))
            raw_dist = np.sqrt((hero_pos["x"] - tx) ** 2 + (hero_pos["z"] - tz) ** 2)
            base = idx * TREASURE_FEAT_DIM
            treasure_feat[base : base + TREASURE_FEAT_DIM] = np.array(
                [
                    1.0,
                    _norm(tx, MAP_SIZE),
                    _norm(tz, MAP_SIZE),
                    _norm(raw_dist, MAP_SIZE * 1.41),
                    _norm(t.get("hero_relative_direction", 0), MAX_REL_DIR),
                ],
                dtype=np.float32,
            )

        buffs = [o for o in organs if int(o.get("sub_type", 0)) == 2 and int(o.get("status", 0)) == 1]
        buffs = sorted(
            buffs,
            key=lambda item: np.sqrt(
                (hero_pos["x"] - float(item.get("pos", {}).get("x", 0))) ** 2
                + (hero_pos["z"] - float(item.get("pos", {}).get("z", 0))) ** 2
            ),
        )

        buff_feat = np.zeros(Config.FEATURES[4], dtype=np.float32)
        for idx, b in enumerate(buffs[:BUFF_SLOT_NUM]):
            b_pos = b.get("pos", {})
            bx = float(b_pos.get("x", 0))
            bz = float(b_pos.get("z", 0))
            raw_dist = np.sqrt((hero_pos["x"] - bx) ** 2 + (hero_pos["z"] - bz) ** 2)
            base = idx * BUFF_FEAT_DIM
            buff_feat[base : base + BUFF_FEAT_DIM] = np.array(
                [
                    1.0,
                    _norm(bx, MAP_SIZE),
                    _norm(bz, MAP_SIZE),
                    _norm(raw_dist, MAP_SIZE * 1.41),
                    _norm(b.get("hero_relative_direction", 0), MAX_REL_DIR),
                ],
                dtype=np.float32,
            )

        # Organ potential-based shaping / 物件势能差分奖励
        # F(s,a,s') = gamma * Phi(s') - Phi(s)
        # Phi(s) = sum_i [w_i * A * exp(-k * d_i)]
        # Treasure: w=1; Buff: w=TREASURE_INC_REWARD/BUFF_INC_REWARD
        organ_potential = 0.0
        # Monotonic collected-count baseline keeps potential continuous at pickup,
        # while staying in PBRS form and reducing pickup-time penalty artifacts.
        organ_potential += ORGAN_POTENTIAL_SCALE * (
            cur_treasures_collected + BUFF_SHAPING_MULT * cur_collected_buff
        )
        for organ in organs:
            if int(organ.get("status", 0)) != 1:
                continue
            o_pos = organ.get("pos", {})
            ox = float(o_pos.get("x", 0))
            oz = float(o_pos.get("z", 0))
            raw_dist = np.sqrt((hero_pos["x"] - ox) ** 2 + (hero_pos["z"] - oz) ** 2)
            subtype = int(organ.get("sub_type", 0))
            type_weight = BUFF_SHAPING_MULT if subtype == 2 else 1.0
            organ_potential += type_weight * ORGAN_POTENTIAL_SCALE * np.exp(-ORGAN_POTENTIAL_DECAY * raw_dist)

        if self.last_organ_potential is None:
            organ_shaping = 0.0
        else:
            organ_shaping = Config.GAMMA * organ_potential - self.last_organ_potential

        # Local map features (9x9=81D) / 局部地图特征
        map_feat = np.zeros(LOCAL_MAP_WIN_SIZE * LOCAL_MAP_WIN_SIZE, dtype=np.float32)
        if map_info is not None and len(map_info) > 0:
            center = len(map_info) // 2
            radius = LOCAL_MAP_WIN_SIZE // 2
            flat_idx = 0
            for row in range(center - radius, center + radius + 1):
                for col in range(center - radius, center + radius + 1):
                    if 0 <= row < len(map_info) and 0 <= col < len(map_info[0]):
                        map_feat[flat_idx] = float(map_info[row][col] != 0)
                    flat_idx += 1

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

        # Concatenate features / 拼接特征
        feature = np.concatenate(
            [
                hero_feat,
                monster_feats[0],
                monster_feats[1],
                treasure_feat,
                buff_feat,
                map_feat,
                np.array(legal_action, dtype=np.float32),
                progress_feat,
            ]
        )

        # Step reward / 即时奖励
        cur_min_dist_norm = 1.0
        for m_feat in monster_feats:
            if m_feat[0] > 0:
                cur_min_dist_norm = min(cur_min_dist_norm, m_feat[4])

        survive_reward = 0.01
        dist_change = cur_min_dist_norm - self.last_min_monster_dist_norm
        dist_shaping = 0.1 * dist_change
        treasure_inc = max(0, cur_treasures_collected - self.last_treasures_collected)
        buff_inc = max(0, cur_collected_buff - self.last_collected_buff)
        collection_reward = TREASURE_INC_REWARD * treasure_inc + BUFF_INC_REWARD * buff_inc
        flash_escape_reward = (
            FLASH_ESCAPE_REWARD
            if (last_action is not None and int(last_action) >= 8 and dist_change > 0.05)
            else 0.0
        )
        # direction_reward = DIRECTION_REWARD if dist_change > 0 else 0.0

        self.last_min_monster_dist_norm = cur_min_dist_norm
        self.last_treasures_collected = cur_treasures_collected
        self.last_collected_buff = cur_collected_buff
        self.last_organ_potential = organ_potential

        reward = [
            survive_reward
            + dist_shaping
            + collection_reward
            + organ_shaping
            + flash_escape_reward
            # + direction_reward
        ]

        return feature, legal_action, reward
