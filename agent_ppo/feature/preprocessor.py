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
from collections import deque
from agent_ppo.conf.conf import Config

MAP_SIZE = 128.0
MAP_GRID_SIZE = 128
UNKNOWN_CELL = -1
MAX_MONSTER_SPEED = 5.0
MAX_FLASH_CD = 2000.0
MAX_BUFF_DURATION = 50.0
ACTION_DIM = 16
MAX_VIEW_DIST = MAP_SIZE * 1.42

# 8方向: E, NE, N, NW, W, SW, S, SE
DIR_VEC = np.array(
    [
        [1, 0],
        [1, -1],
        [0, -1],
        [-1, -1],
        [-1, 0],
        [-1, 1],
        [0, 1],
        [1, 1],
    ],
    dtype=np.int32,
)


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
        self.max_step = 1000
        self.last_min_monster_dist = 12.0
        self.last_treasures_collected = 0
        self.last_collected_buff = 0
        self.last_target_dist = None

        # Episode-scoped global occupancy map:
        # -1 unknown, 0 blocked, 1 passable
        self.global_map = np.full((MAP_GRID_SIZE, MAP_GRID_SIZE), UNKNOWN_CELL, dtype=np.int8)
        self.global_seen = np.zeros((MAP_GRID_SIZE, MAP_GRID_SIZE), dtype=np.uint8)
        self.global_map_revision = 0

        # Cache of shortest-path distance fields keyed by source grid cell.
        self._sp_cache_revision = -1
        self._sp_cache = {}

    def _world_to_grid(self, hero_pos):
        """Project continuous world position (x, z) to global grid (row, col)."""
        hx = int(np.clip(np.rint(float(hero_pos[0])), 0, MAP_GRID_SIZE - 1))
        hz = int(np.clip(np.rint(float(hero_pos[1])), 0, MAP_GRID_SIZE - 1))
        return hz, hx

    def _update_global_map(self, hero_pos, map_info):
        """Fuse local map_info into global 128x128 map with O(k^2) patch update.

        map_info is expected to be the hero-centered local occupancy (typically 21x21).
        """
        if map_info is None or len(map_info) == 0:
            return 0

        local = np.asarray(map_info, dtype=np.int8)
        if local.ndim != 2:
            return 0

        local_bin = (local != 0).astype(np.int8)
        h, w = int(local_bin.shape[0]), int(local_bin.shape[1])
        if h <= 0 or w <= 0:
            return 0

        hero_r, hero_c = self._world_to_grid(hero_pos)
        center_r, center_c = h // 2, w // 2

        r0, c0 = hero_r - center_r, hero_c - center_c
        r1, c1 = r0 + h, c0 + w

        gr0, gc0 = max(0, r0), max(0, c0)
        gr1, gc1 = min(MAP_GRID_SIZE, r1), min(MAP_GRID_SIZE, c1)
        if gr0 >= gr1 or gc0 >= gc1:
            return 0

        lr0, lc0 = gr0 - r0, gc0 - c0
        lr1, lc1 = lr0 + (gr1 - gr0), lc0 + (gc1 - gc0)
        patch = local_bin[lr0:lr1, lc0:lc1]

        target = self.global_map[gr0:gr1, gc0:gc1]
        changed = int(np.count_nonzero(target != patch))
        if changed > 0:
            target[...] = patch
            self.global_seen[gr0:gr1, gc0:gc1] = 1
            self.global_map_revision += 1

        return changed

    def _is_global_cell_observable_passable(self, r, c):
        if r < 0 or c < 0 or r >= MAP_GRID_SIZE or c >= MAP_GRID_SIZE:
            return False
        return bool(self.global_seen[r, c] > 0 and self.global_map[r, c] == 1)

    def _neighbor8(self, r, c):
        for dz in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dz == 0 and dx == 0:
                    continue
                nr, nc = r + dz, c + dx
                if 0 <= nr < MAP_GRID_SIZE and 0 <= nc < MAP_GRID_SIZE:
                    yield nr, nc

    def _get_distance_field(self, source_pos):
        """Return cached BFS distance field from source on observed-passable cells."""
        if source_pos is None:
            return None

        if self._sp_cache_revision != self.global_map_revision:
            self._sp_cache_revision = self.global_map_revision
            self._sp_cache.clear()

        src_r, src_c = self._world_to_grid(source_pos)
        key = (src_r, src_c)
        if key in self._sp_cache:
            return self._sp_cache[key]

        if not self._is_global_cell_observable_passable(src_r, src_c):
            return None

        dist = np.full((MAP_GRID_SIZE, MAP_GRID_SIZE), -1, dtype=np.int16)
        q = deque()
        dist[src_r, src_c] = 0
        q.append((src_r, src_c))

        while q:
            r, c = q.popleft()
            d = int(dist[r, c])
            for nr, nc in self._neighbor8(r, c):
                if dist[nr, nc] >= 0:
                    continue
                if not self._is_global_cell_observable_passable(nr, nc):
                    continue
                dist[nr, nc] = d + 1
                q.append((nr, nc))

        # Keep cache small but effective for repeated monster/target queries.
        if len(self._sp_cache) >= 8:
            self._sp_cache.pop(next(iter(self._sp_cache)))
        self._sp_cache[key] = dist
        return dist

    def _shortest_path_dist_observed(self, from_pos, to_pos):
        """Observed-map shortest path distance. Return None if path is not observable."""
        if from_pos is None or to_pos is None:
            return None

        dist_field = self._get_distance_field(to_pos)
        if dist_field is None:
            return None

        fr, fc = self._world_to_grid(from_pos)
        if fr < 0 or fc < 0 or fr >= MAP_GRID_SIZE or fc >= MAP_GRID_SIZE:
            return None

        d = int(dist_field[fr, fc])
        if d < 0:
            return None
        return float(d)

    def _path_dist_or_euclid(self, p1, p2):
        d_sp = self._shortest_path_dist_observed(p1, p2)
        if d_sp is not None:
            return d_sp
        return self._dist(p1, p2)

    def _simulate_flash_delta(self, action, hero_speed, map_info):
        """Flash simulation that may cross a wall segment if landing cell is passable."""
        d_idx = self._dir_from_action(action)
        dx, dz = int(DIR_VEC[d_idx][0]), int(DIR_VEC[d_idx][1])
        step_len = self._move_len(action, hero_speed)
        center = self._to_map_center(map_info)

        if center is None:
            return dx * step_len, dz * step_len, step_len, False

        moved = 0
        seen_block = False
        wall_crossed = False
        for i in range(1, step_len + 1):
            r = center + dz * i
            c = center + dx * i
            if self._cell_passable(map_info, r, c):
                moved = i
                if seen_block:
                    wall_crossed = True
            else:
                seen_block = True

        return dx * moved, dz * moved, moved, wall_crossed

    def _dir_from_action(self, action):
        return int(action % 8)

    def _move_len(self, action, hero_speed):
        if action >= 8:
            return 10 if (action % 8) in (0, 2, 4, 6) else 8
        return int(max(1, hero_speed))

    def _to_map_center(self, map_info):
        if map_info is None or len(map_info) == 0:
            return None
        return len(map_info) // 2

    def _cell_passable(self, map_info, r, c):
        if map_info is None or len(map_info) == 0:
            return True
        if r < 0 or c < 0 or r >= len(map_info) or c >= len(map_info[0]):
            return False
        return bool(map_info[r][c] != 0)

    def _simulate_delta(self, action, hero_speed, map_info):
        """Simulate movement/flash with local obstacle clipping."""
        d_idx = self._dir_from_action(action)
        dx, dz = int(DIR_VEC[d_idx][0]), int(DIR_VEC[d_idx][1])
        step_len = self._move_len(action, hero_speed)
        center = self._to_map_center(map_info)

        if center is None:
            return dx * step_len, dz * step_len, step_len

        moved = 0
        for i in range(1, step_len + 1):
            r = center + dz * i
            c = center + dx * i
            if not self._cell_passable(map_info, r, c):
                break
            moved = i

        return dx * moved, dz * moved, moved

    def _dir_local_space(self, map_info):
        """Compute openness / immediate blocked flags for 8 directions."""
        open_scores = np.zeros(8, dtype=np.float32)
        blocked = np.zeros(8, dtype=np.float32)
        center = self._to_map_center(map_info)

        if center is None:
            open_scores.fill(1.0)
            return open_scores, blocked

        max_probe = 4
        for i in range(8):
            dx, dz = int(DIR_VEC[i][0]), int(DIR_VEC[i][1])
            free_cnt = 0
            for step in range(1, max_probe + 1):
                r = center + dz * step
                c = center + dx * step
                if not self._cell_passable(map_info, r, c):
                    break
                free_cnt += 1
            open_scores[i] = free_cnt / float(max_probe)

            r1 = center + dz
            c1 = center + dx
            blocked[i] = 0.0 if self._cell_passable(map_info, r1, c1) else 1.0

        return open_scores, blocked

    def _dist(self, p1, p2):
        return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))

    def _rel_dir_norm(self, rel_dir):
        rd = float(rel_dir)
        rd = np.clip(rd, 0.0, 8.0)
        return rd / 8.0

    def _nearest_organ(self, organs, sub_type, hero_pos):
        candidates = [o for o in organs if int(o.get("sub_type", 0)) == sub_type and int(o.get("status", 0)) == 1]
        if not candidates:
            return None, None
        best = min(
            candidates,
            key=lambda o: self._dist(hero_pos, (float(o.get("pos", {}).get("x", 0)), float(o.get("pos", {}).get("z", 0)))),
        )
        p = best.get("pos", {})
        bpos = (float(p.get("x", 0)), float(p.get("z", 0)))
        return best, bpos

    def _visible_monsters(self, monsters):
        out = []
        for m in monsters[:2]:
            if float(m.get("is_in_view", 0)) <= 0:
                continue
            p = m.get("pos", {})
            out.append(
                {
                    "pos": (float(p.get("x", 0)), float(p.get("z", 0))),
                    "speed": float(m.get("speed", 1)),
                    "dist_bucket": float(m.get("hero_l2_distance", 5)),
                    "rel_dir": float(m.get("hero_relative_direction", 0)),
                }
            )
        return out

    def _nearest_monster_dist(self, hero_pos, visible_monsters):
        if not visible_monsters:
            return 30.0
        return min(self._path_dist_or_euclid(hero_pos, m["pos"]) for m in visible_monsters)

    def _greedy_scores(
        self,
        hero_pos,
        hero_speed,
        flash_cd_norm,
        buff_remain_norm,
        map_info,
        legal_action,
        visible_monsters,
        nearest_treasure_pos,
        nearest_buff_pos,
        dir_open,
        dir_blocked,
        last_action,
    ):
        scores = np.full(ACTION_DIM, -1e6, dtype=np.float32)
        nearest_before = self._nearest_monster_dist(hero_pos, visible_monsters)

        for a in range(ACTION_DIM):
            if int(legal_action[a]) == 0:
                continue

            wall_crossed = False
            if a >= 8 and flash_cd_norm <= 1e-4:
                ddx, ddz, moved, wall_crossed = self._simulate_flash_delta(a, hero_speed, map_info)
            else:
                ddx, ddz, moved = self._simulate_delta(a, hero_speed, map_info)
            new_pos = (hero_pos[0] + ddx, hero_pos[1] + ddz)
            a_dir = self._dir_from_action(a)
            avec = DIR_VEC[a_dir].astype(np.float32)
            anorm = np.linalg.norm(avec)
            if anorm > 1e-6:
                avec = avec / anorm

            danger = 0.0
            collision_penalty = 0.0
            away_vec = np.zeros(2, dtype=np.float32)
            nearest_after = 30.0
            for m in visible_monsters:
                d_after = self._path_dist_or_euclid(new_pos, m["pos"])
                nearest_after = min(nearest_after, d_after)

                threat = (1.2 + 0.3 * np.clip(m["speed"], 0.0, MAX_MONSTER_SPEED))
                danger += threat * np.exp(-d_after / 6.0)
                if d_after <= 1.5:
                    collision_penalty += 40.0

                vec = np.array([hero_pos[0] - m["pos"][0], hero_pos[1] - m["pos"][1]], dtype=np.float32)
                vnorm = np.linalg.norm(vec) + 1e-6
                away_vec += (vec / vnorm) * (1.0 / (vnorm + 1.0))

            score = -3.5 * danger - collision_penalty

            away_norm = np.linalg.norm(away_vec)
            if away_norm > 1e-6:
                away_unit = away_vec / away_norm
                score += 1.4 * float(np.dot(away_unit, avec))

            target_pos = None
            target_weight = 0.0
            if nearest_buff_pos is not None and buff_remain_norm < 0.05:
                d_buff = self._path_dist_or_euclid(hero_pos, nearest_buff_pos)
                d_treasure = (
                    self._path_dist_or_euclid(hero_pos, nearest_treasure_pos)
                    if nearest_treasure_pos is not None
                    else 999.0
                )
                if nearest_before > 8.0 or d_buff <= d_treasure * 0.8:
                    target_pos = nearest_buff_pos
                    target_weight = 1.6

            if target_pos is None and nearest_treasure_pos is not None:
                target_pos = nearest_treasure_pos
                target_weight = 1.2

            if target_pos is not None:
                before = self._path_dist_or_euclid(hero_pos, target_pos)
                after = self._path_dist_or_euclid(new_pos, target_pos)
                score += target_weight * (before - after)
                if after <= 1.2:
                    score += 0.8

            score += 0.35 * float(dir_open[a_dir])
            score -= 0.8 * float(dir_blocked[a_dir])

            if last_action is not None and int(last_action) >= 0 and self._dir_from_action(int(last_action)) == a_dir:
                score += 0.08

            if a >= 8:
                score -= 0.6
                if flash_cd_norm > 1e-4:
                    score -= 2.0
                else:
                    safety_gain = nearest_after - nearest_before
                    danger_now = danger + (2.0 if nearest_before <= 6.0 else 0.0)
                    if wall_crossed and safety_gain > 1.0 and danger_now >= 1.8:
                        # High bonus when dangerous and flash crosses wall while increasing safety.
                        score += 3.2 + 0.6 * safety_gain
                    if safety_gain > 2.0:
                        score += 2.0 + 0.3 * safety_gain
                    if nearest_before > 12.0 and target_pos is not None:
                        score -= 0.4
            else:
                if moved <= 0:
                    score -= 1.2

            scores[a] = score

        return scores

    def feature_process(self, env_obs, last_action):
        """Return compact features + legal mask + reward.

        观测中携带贪心动作评分，供策略端直接融合输出。
        """
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        map_info = observation["map_info"]
        legal_act_raw = observation["legal_action"]

        self.step_no = observation["step_no"]
        self.max_step = env_info.get("max_step", 1000)
        cur_treasures_collected = int(env_info.get("treasures_collected", 0))
        cur_collected_buff = int(env_info.get("collected_buff", 0))

        hero = frame_state["heroes"]
        hero_pos = hero["pos"]
        hero_xy = (float(hero_pos.get("x", 0)), float(hero_pos.get("z", 0)))

        # Real-time global map update from local observation window.
        self._update_global_map(hero_xy, map_info)

        hero_speed = int(max(1, hero.get("speed", 1)))
        flash_cd = float(hero.get("flash_cooldown", env_info.get("flash_cooldown", 0)))
        buff_remain = float(hero.get("buff_remaining_time", 0))
        flash_cd_norm = _norm(flash_cd, MAX_FLASH_CD)
        buff_remain_norm = _norm(buff_remain, MAX_BUFF_DURATION)

        monsters = frame_state.get("monsters", [])
        visible_monsters = self._visible_monsters(monsters)

        monster_core = []
        for i in range(2):
            if i < len(monsters):
                m = monsters[i]
                in_view = float(m.get("is_in_view", 0))
                dist_bucket = float(m.get("hero_l2_distance", 5))
                dist_approx = min(MAX_VIEW_DIST, dist_bucket * 30.0 + 15.0)
                rel_dir_norm = self._rel_dir_norm(m.get("hero_relative_direction", 0))
                speed_norm = _norm(m.get("speed", 1), MAX_MONSTER_SPEED)
                monster_core.extend([in_view, _norm(dist_approx, MAX_VIEW_DIST), rel_dir_norm, speed_norm])
            else:
                monster_core.extend([0.0, 1.0, 0.0, 0.0])

        organs = frame_state.get("organs", [])
        treasure_obj, nearest_treasure_pos = self._nearest_organ(organs, 1, hero_xy)
        buff_obj, nearest_buff_pos = self._nearest_organ(organs, 2, hero_xy)

        if treasure_obj is None:
            treasure_core = [1.0, 0.0, 0.0]
        else:
            tdist = self._dist(hero_xy, nearest_treasure_pos)
            treasure_core = [
                _norm(tdist, MAX_VIEW_DIST),
                self._rel_dir_norm(treasure_obj.get("hero_relative_direction", 0)),
                1.0,
            ]

        if buff_obj is None:
            buff_core = [1.0, 0.0, 0.0]
        else:
            bdist = self._dist(hero_xy, nearest_buff_pos)
            buff_core = [
                _norm(bdist, MAX_VIEW_DIST),
                self._rel_dir_norm(buff_obj.get("hero_relative_direction", 0)),
                1.0,
            ]

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

        step_norm = _norm(self.step_no, self.max_step)
        core = np.array(
            [flash_cd_norm, buff_remain_norm, step_norm] + monster_core + treasure_core + buff_core,
            dtype=np.float32,
        )

        dir_open, dir_blocked = self._dir_local_space(map_info)
        greedy_scores = self._greedy_scores(
            hero_pos=hero_xy,
            hero_speed=hero_speed,
            flash_cd_norm=flash_cd_norm,
            buff_remain_norm=buff_remain_norm,
            map_info=map_info,
            legal_action=legal_action,
            visible_monsters=visible_monsters,
            nearest_treasure_pos=nearest_treasure_pos,
            nearest_buff_pos=nearest_buff_pos,
            dir_open=dir_open,
            dir_blocked=dir_blocked,
            last_action=last_action,
        )

        feature = np.concatenate(
            [
                core,
                dir_open,
                dir_blocked,
                greedy_scores,
            ]
        )

        nearest_monster_dist = self._nearest_monster_dist(hero_xy, visible_monsters)
        survive_reward = 0.01
        dist_change = nearest_monster_dist - self.last_min_monster_dist
        dist_shaping = 0.04 * np.clip(dist_change, -5.0, 5.0)

        target_dist = None
        if buff_remain_norm < 0.05 and nearest_buff_pos is not None:
            target_dist = self._dist(hero_xy, nearest_buff_pos)
        elif nearest_treasure_pos is not None:
            target_dist = self._dist(hero_xy, nearest_treasure_pos)

        target_shaping = 0.0
        if target_dist is not None and self.last_target_dist is not None:
            target_shaping = 0.02 * np.clip(self.last_target_dist - target_dist, -4.0, 4.0)

        treasure_inc = max(0, cur_treasures_collected - self.last_treasures_collected)
        buff_inc = max(0, cur_collected_buff - self.last_collected_buff)
        collection_reward = 0.8 * treasure_inc + 0.45 * buff_inc
        flash_escape_reward = (
            0.03
            if (last_action is not None and int(last_action) >= 8 and dist_change > 1.2)
            else 0.0
        )

        self.last_min_monster_dist = nearest_monster_dist
        self.last_treasures_collected = cur_treasures_collected
        self.last_collected_buff = cur_collected_buff
        self.last_target_dist = target_dist

        reward = [survive_reward + dist_shaping + target_shaping + collection_reward + flash_escape_reward]

        return feature, legal_action, reward
