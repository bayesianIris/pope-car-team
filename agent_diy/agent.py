#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Enhanced Agent class for Gorge Chase DIY.
峡谷追猎 DIY 增强Agent 主类。
"""

import torch
import os

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import numpy as np
from kaiwudrl.interface.agent import BaseAgent

from agent_diy.algorithm.algorithm import Algorithm
from agent_diy.conf.conf import Config
from agent_diy.feature.definition import ActData, ObsData
from agent_diy.feature.preprocessor import Preprocessor
from agent_diy.model.model import Model


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        torch.manual_seed(0)
        self.device = device
        self.model = Model(device).to(self.device)
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=Config.INIT_LEARNING_RATE_START,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        self.algorithm = Algorithm(self.model, self.optimizer, self.device, logger, monitor)
        self.preprocessor = Preprocessor()
        self.last_action = -1
        self.logger = logger
        self.monitor = monitor
        super().__init__(agent_type, device, logger, monitor)

    def reset(self, env_obs=None):
        """Reset per-episode state.
        
        每局开始时重置状态。
        """
        self.preprocessor.reset()
        self.last_action = -1

    def observation_process(self, env_obs):
        """Convert raw env_obs to ObsData and remain_info.
        
        将原始观测转换为 ObsData 和 remain_info。
        """
        feature, legal_action, reward = self.preprocessor.feature_process(env_obs, self.last_action)
        obs_data = ObsData(
            feature=list(feature),
            legal_action=legal_action,
        )
        remain_info = {"reward": reward}
        return obs_data, remain_info

    def predict(self, list_obs_data):
        """Stochastic inference for training (exploration).
        
        训练时随机采样动作（探索）。
        """
        feature = list_obs_data[0].feature
        legal_action = list_obs_data[0].legal_action

        logits, value, prob = self._run_model(feature, legal_action)

        action = self._legal_sample(prob, use_max=False)
        d_action = self._legal_sample(prob, use_max=True)

        return [
            ActData(
                action=[action],
                d_action=[d_action],
                prob=list(prob),
                value=value,
            )
        ]

    def exploit(self, env_obs):
        """Greedy inference for evaluation.
        
        评估时贪心选择动作（利用）。
        """
        obs_data, _ = self.observation_process(env_obs)
        act_data = self.predict([obs_data])
        return self.action_process(act_data[0], is_stochastic=False)

    def learn(self, list_sample_data):
        """Train the model.
        
        训练模型。
        """
        if not list_sample_data:
            return

        self.algorithm.learn(list_sample_data)

    def save_model(self, path=None, id="1"):
        """Save model checkpoint.
        
        保存模型检查点。
        """
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl" if path else f"model.ckpt-{str(id)}.pkl"
        state_dict_cpu = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
        torch.save(state_dict_cpu, model_file_path)
        if self.logger:
            self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        """Load model checkpoint.
        
        加载模型检查点。
        """
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl" if path else f"model.ckpt-{str(id)}.pkl"

        # First training run may not have a latest checkpoint yet.
        if not os.path.exists(model_file_path):
            if self.logger:
                self.logger.warning(f"model file {model_file_path} not found, skip loading")
            return

        self.model.load_state_dict(torch.load(model_file_path, map_location=self.device))
        if self.logger:
            self.logger.info(f"load model {model_file_path} successfully")

    def _run_model(self, feature, legal_action):
        """Run model inference.
        
        运行模型推理。
        """
        # Convert to tensors
        if isinstance(feature, list):
            feature = np.array(feature, dtype=np.float32)
        if isinstance(legal_action, list):
            legal_action = np.array(legal_action, dtype=np.float32)

        feature = torch.from_numpy(feature).unsqueeze(0).to(self.device)
        legal_action = torch.from_numpy(legal_action).unsqueeze(0).to(self.device)

        with torch.no_grad():
            self.model.set_eval_mode()
            logits, value = self.model(feature)

        # Apply masked softmax
        illegal_mask = (1.0 - legal_action) * -1e9
        masked_logits = logits + illegal_mask
        prob = torch.softmax(masked_logits, dim=1)
        prob = prob * legal_action
        prob = prob / (prob.sum(dim=1, keepdim=True) + 1e-9)

        prob = prob.cpu().numpy()[0]
        value = value.cpu().numpy()[0, 0]

        return logits, value, prob

    def _legal_sample(self, prob, use_max=False):
        """Sample action from legal probability distribution.
        
        从合法概率分布中采样动作。
        """
        if use_max:
            action = np.argmax(prob)
        else:
            action = np.random.choice(len(prob), p=prob)
        
        self.last_action = action
        return action

    def action_process(self, act_data, is_stochastic=True):
        """Convert ActData to final action.
        
        将ActData转换为最终动作。
        """
        if is_stochastic:
            action = act_data.action[0]
        else:
            action = act_data.d_action[0]

        self.last_action = int(action)
        return int(action)
