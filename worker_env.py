# labor_env.py

import gymnasium as gym
import numpy as np
from model_rl import LaborMarketModel


class LaborMarketEnv(gym.Env):

    def __init__(self):

        self.model = LaborMarketModel()

        self.n_workers = len(self.model.workers)
        self.n_firms = len(self.model.firms)

        self.worker_action_space = gym.spaces.Discrete(3)
        self.firm_action_space = gym.spaces.Discrete(9)
        

        self.worker_observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(4,),
            dtype=np.float32
        )
        
        self.firm_observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(4,),
            dtype=np.float32
        )

    # ---------------- Observation ----------------

    def observe(self):

        employed_workers = [w for w in self.model.workers if w.employed]

        avg_wage = np.mean([w.monthly_wage for w in employed_workers] or [0])
        employment = len(employed_workers) / self.n_workers
        avg_firm_wage = np.mean([f.monthly_wage for f in self.model.firms])

        avg_profit = np.mean([f.profit for f in self.model.firms])
        avg_capital = np.mean([f.capital for f in self.model.firms])
        avg_utility = np.mean([w.last_utility for w in self.model.workers])
        avg_firm_capital = np.mean([f.capital for f in self.model.firms])

        return np.array([
            avg_wage/8000,
            employment,
            avg_utility/3000,
            avg_profit/3000,
            avg_capital/200,
            avg_firm_wage/8000,
            avg_firm_capital/100,
            0,0,0
        ], dtype=np.float32)

    # ---------------- Step ----------------

    def step(self, action):

        self.worker_actions = action["workers"]
        self.firm_actions = action["firms"]

        self.model.step()

        reward_workers = np.mean([w.reward for w in self.model.workers])/10000
        reward_firms = np.mean([f.reward for f in self.model.firms])/10000
        reward = {
                "workers": reward_workers,
                "firms": reward_firms
            }

        obs = self.observe()

        terminated = False
        truncated = False   

        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):

        self.model = LaborMarketModel()
        return self.observe(), {}