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
        self.max_step = 360          # ~30 years; long enough to see capital cycles (12-step) and deficit exit (24-step)

        # --- tracking variables for obs (can't read from firm.last_profit after step, it's already updated) ---
        self.prev_profit = 0.0
        self.prev_workers = len(self.rl_firm.current_workers)

        # 7 actions:
        #   0 = hold
        #   1 = wage +300  (aggressive raise – attract workers fast)
        #   2 = wage +100  (soft raise)
        #   3 = wage -100  (soft cut – save cost)
        #   4 = wage -300  (aggressive cut – when clearly overpaying)
        #   5 = post 1 vacancy
        #   6 = fire 1 worker
        self.action_space = gym.spaces.Discrete(7)

        # 8 features, all normalised to roughly [-1, 1]
        self.observation_space = gym.spaces.Box(
            low=-1.5,
            high=1.5,
            shape=(8,),
            dtype=np.float32
        )

    # ------------------------------------------------------------------ #
    #  Observation                                                         #
    # ------------------------------------------------------------------ #

    def observe(self):
        firm  = self.rl_firm
        model = self.model

        # ---- profit signal ----
        profit_signal        = float(np.tanh(firm.profit / 20_000))
        profit_change        = firm.profit - self.prev_profit
        profit_change_signal = float(np.tanh(profit_change / 5_000))

        # ---- VMPL gap: positive → underpaying (hire / raise), negative → overpaying (fire / cut) ----
        labor = len(firm.current_workers)
        if labor > 0:
            mpl      = firm.marginal_product_labor(firm.productivity, labor, firm.alpha)
            vmpl     = mpl * firm.output_price
            vmpl_gap = float(np.tanh((vmpl - firm.monthly_wage) / max(firm.monthly_wage, 1.0)))
        else:
            vmpl_gap = 1.0   # no workers ⟹ MPL → ∞, always worth hiring

        # ---- wage vs market: positive → paying above market ----
        other_wages  = [f.monthly_wage for i, f in enumerate(model.firms) if i != self.rl_firm_id]
        market_wage  = float(np.mean(other_wages)) if other_wages else firm.monthly_wage
        wage_vs_mkt  = float(np.tanh((firm.monthly_wage - market_wage) / max(market_wage, 1.0)))

        # ---- workforce size ----
        labor_ratio   = labor / 40.0                           # assume max ~40 workers per firm
        vacancy_ratio = min(firm.vacancies, 5) / 5.0           # cap at 5 to avoid extreme values

        # ---- recent workforce change (did we gain or lose workers this step?) ----
        worker_change = float(np.tanh((labor - self.prev_workers) / 3.0))

        # ---- wage-decision clock: rises 0→1 across each 12-step cycle ----
        # Tells the agent how close it is to the next wage-change opportunity.
        wage_clock = (self.current_step % 12) / 11.0

        obs = np.array([
            profit_signal,        # tanh  ∈ (-1, 1)  — own profit level
            profit_change_signal, # tanh  ∈ (-1, 1)  — own profit trend
            vmpl_gap,             # tanh  ∈ (-1, 1)  — own VMPL vs own wage
            wage_vs_mkt,          # tanh  ∈ (-1, 1)  — own wage vs public market avg
            labor_ratio,          # [0, ~1]           — own workforce size
            vacancy_ratio,        # [0,  1]           — own open vacancies
            worker_change,        # tanh  ∈ (-1, 1)  — own workforce change this step
            wage_clock,           # [0,  1]           — position in 12-step wage cycle
        ], dtype=np.float32)

        return np.clip(obs, -1.5, 1.5)

    # ------------------------------------------------------------------ #
    #  Action mask (for MaskablePPO)                                       #
    # ------------------------------------------------------------------ #

    def action_masks(self) -> np.ndarray:
        wage_step = (self.current_step % 12 == 0)
        return np.array([
            True,       # 0: hold        — always valid
            wage_step,  # 1: wage +300   — annual only
            wage_step,  # 2: wage +100   — annual only
            wage_step,  # 3: wage -100   — annual only
            wage_step,  # 4: wage -300   — annual only
            True,       # 5: post vacancy — always valid
            True,       # 6: fire worker  — always valid
        ], dtype=bool)

    # ------------------------------------------------------------------ #
    #  Step                                                                #
    # ------------------------------------------------------------------ #

    def step(self, action):

        # snapshot BEFORE the step so observe() can compute changes correctly
        self.prev_profit  = self.rl_firm.profit
        self.prev_workers = len(self.rl_firm.current_workers)

        self.model.rl_action = action
        self.model.step()

        reward = float(self.rl_firm.reward)
        self.current_step += 1

        obs = self.observe()

        terminated = False
        truncated  = self.current_step >= self.max_step
        if truncated:
            self.current_step = 0

        return obs, reward, terminated, truncated, {}

    # ------------------------------------------------------------------ #
    #  Reset                                                               #
    # ------------------------------------------------------------------ #

    def reset(self, seed=None, options=None):
        self.model        = LaborMarketModel()
        self.rl_firm      = self.model.firms[self.rl_firm_id]
        self.current_step = 0
        self.prev_profit  = 0.0
        self.prev_workers = len(self.rl_firm.current_workers)
        return self.observe(), {}
