# firm_env.py

import gymnasium as gym
import numpy as np
from model_rl import LaborMarketModel


class LaborMarketEnv(gym.Env):

    def __init__(self):

        self.model = LaborMarketModel()

        self.rl_firm_id = 0
        self.rl_firm = self.model.firms[self.rl_firm_id]
        self.current_step = 0
        self.max_step = 2000

        self.action_space = gym.spaces.Discrete(5)
        
        self.observation_space = gym.spaces.Box(
            low=-5,
            high=5,
            shape=(6,),
            dtype=np.float32
        )

    # ---------------- Observation ----------------

    def observe(self):

        firm = self.rl_firm
        
        avg_other_wage = np.mean([
            f.monthly_wage for i,f in enumerate(self.model.firms)
            if i != self.rl_firm_id
        ])

        current_worker = len(firm.current_workers)
        
        obs = np.array([
            np.log1p(avg_other_wage) / 20,
            firm.monthly_wage / 15000,
            np.log1p(firm.capital) / 15,
            np.arcsinh(firm.profit / 1000) / 5.0,
            firm.vacancies,
            current_worker/40
        ], dtype=np.float32)
        
        # print(obs)
        
        return obs

    # ---------------- Step ----------------

    def step(self, action):

        self.model.rl_action = action

        self.model.step()

        reward = self.rl_firm.reward
        self.current_step += 1

        obs = self.observe()

        terminated = False
        truncated = False
        
        if self.current_step >= self.max_step:
            truncated = True
            self.current_step = 0
        

        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):

        self.model = LaborMarketModel()
        self.rl_firm = self.model.firms[self.rl_firm_id]
        return self.observe(), {}