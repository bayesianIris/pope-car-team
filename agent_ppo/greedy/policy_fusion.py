#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import numpy as np
import torch

from agent_ppo.conf.conf import Config


def _center_rows_np(arr):
    return arr - np.mean(arr, axis=-1, keepdims=True)


def _center_rows_torch(arr):
    return arr - torch.mean(arr, dim=1, keepdim=True)


def fuse_logits_np(net_logits, feature):
    """Fuse greedy action scores with tiny network logits (numpy).

    Greedy dominates the decision; network contributes 0.01%.
    """
    greedy_logits = np.asarray(feature[Config.GREEDY_SCORE_START : Config.GREEDY_SCORE_END], dtype=np.float32)
    greedy_logits = _center_rows_np(greedy_logits.reshape(1, -1))[0]
    return (1.0 - Config.NET_LOGIT_RATIO) * greedy_logits + Config.NET_LOGIT_RATIO * net_logits


def fuse_logits_torch(net_logits, obs):
    """Fuse greedy action scores with tiny network logits (torch).

    `obs` shape: [B, DIM_OF_OBSERVATION].
    """
    greedy_logits = obs[:, Config.GREEDY_SCORE_START : Config.GREEDY_SCORE_END]
    greedy_logits = _center_rows_torch(greedy_logits)
    return (1.0 - Config.NET_LOGIT_RATIO) * greedy_logits + Config.NET_LOGIT_RATIO * net_logits
