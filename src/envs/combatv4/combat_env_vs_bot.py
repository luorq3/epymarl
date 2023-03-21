import random

import numpy as np

from envs.multiagentenv import MultiAgentEnv
from envs.combatv4.combat_env_core import CombatEnvCore
from envs.combatv4.agent.base import CAMPS_TYPES as CAMPS


class CombatEnv(MultiAgentEnv):
    def __init__(self, config):
        super(CombatEnv, self).__init__()
        self.camp = config.camp
        self.camp_id = CAMPS[self.camp]
        self._env = CombatEnvCore(config)
        # self._init()

    def get_unit_by_unique_id(self, unique_id):
        return self._env.units_dict.get(unique_id)

    def _init(self):
        # self.agent_idx_dict = {agent_id: unique_id for agent_id, unique_id in enumerate(sorted(self._env.agents_unique_idx))}
        self.agent_idx_dict = {}
        for agent_id, unique_id in enumerate(sorted(self._env.agents_unique_idx)):
            agent = self._env.get_agent(unique_id)
            agent.agent_id = agent_id
            self.agent_idx_dict[agent_id] = unique_id
        self.num_agents = len(self.agent_idx_dict)
        self.reward_type = self._env.reward_type

    def step(self, actions):
        if isinstance(actions, np.ndarray):
            actions = actions.tolist()
        if not isinstance(actions, list):
            actions = [actions]
        action_dict = {agent_id: action for agent_id, action in zip(self.agent_idx_dict.values(), actions)}
        rewards, done, info = self._env.step(action_dict)
        if self.reward_type == "dense":
            reward = sum(rewards.values())
        else:
            reward = rewards.get(self.camp)
        return reward, done, info

    def get_obs(self):
        obs = []
        obs_dict = self._env.get_obs()
        for unique_id in self.agent_idx_dict.values():
            obs.append(obs_dict.get(unique_id))
        return np.array(obs, dtype=np.float32)

    def get_obs_agent(self, agent_id):
        return self._env.get_obs_unique_id(self.agent_idx_dict.get(agent_id))

    def get_obs_size(self):
        obs_dims = []
        obs_dim_dict = self._env.get_obs_dim()
        for unique_id in self.agent_idx_dict.values():
            obs_dims.append(obs_dim_dict.get(unique_id))
        return obs_dims

    def get_state(self):
        return self._env.get_state()

    def get_state_size(self):
        return self._env.state_dims[self.camp]

    def get_avail_actions(self):
        avail_acts = []
        avail_acts_dict = self._env.get_avail_actions()
        for unique_id in self.agent_idx_dict.values():
            avail_acts.append(avail_acts_dict.get(unique_id))
        return avail_acts

    def get_avail_agent_actions(self, agent_id):
        return self._env.get_agent_avail_actions(self.agent_idx_dict.get(agent_id))

    def get_total_actions(self):
        pass

    def reset(self):
        self._env.reset()
        self._init()

    def render(self):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        if seed is None:
            seed = 1
        random.seed(seed)
        np.random.seed(seed)

    def save_replay(self):
        pass
