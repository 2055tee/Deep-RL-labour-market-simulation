# solo/firm_env.py  — 1 RL firm (F0) vs heuristic firms
#
# Observation (12 features):
#   0  profit_signal       — tanh(profit/20k)           own profit level
#   1  profit_change       — tanh(Δprofit/5k)           own profit trend
#   2  vmpl_gap            — tanh((VMPL-wage)/wage)      over/underpaying vs MPL
#   3  wage_vs_mkt         — tanh((own-mkt)/mkt)        wage premium vs all firms
#   4  labor_ratio         — n_workers/40               workforce size
#   5  vacancy_ratio       — min(vac,5)/5               open slots
#   6  worker_change       — tanh(Δworkers/3)           recent headcount Δ
#   7  wage_clock          — step%12/11                 position in wage cycle
#   8  prod_vs_mkt         — tanh((A-avgA)/avgA)        own productivity vs peers
#   9  cap_vs_mkt          — tanh((K-avgK)/avgK)        own capital vs peers
#  10  survival_signal     — tanh(deficit_months/12)    urgency to turn profitable
#  11  market_employment   — n_employed/n_workers       tight vs loose labour market

import gymnasium as gym
import numpy as np
from model_rl import LaborMarketModel


class LaborMarketEnv(gym.Env):

    def __init__(self):
        self.model        = LaborMarketModel()
        self.rl_firm_id   = 0
        self.rl_firm      = self.model.firms[self.rl_firm_id]
        self.current_step = 0
        self.max_step     = 360

        self.prev_profit  = 0.0
        self.prev_workers = len(self.rl_firm.current_workers)

        # 7 actions: hold / wage±300 / wage±100 / post_vacancy / fire_worker
        self.action_space = gym.spaces.Discrete(7)

        self.observation_space = gym.spaces.Box(
            low=-1.5, high=1.5, shape=(12,), dtype=np.float32
        )

    # ------------------------------------------------------------------ #
    #  Observation                                                         #
    # ------------------------------------------------------------------ #

    def observe(self):
        firm  = self.rl_firm
        model = self.model

        # --- own performance ---
        profit_signal        = float(np.tanh(firm.profit / 20_000))
        profit_change_signal = float(np.tanh((firm.profit - self.prev_profit) / 5_000))

        # --- VMPL gap: positive = underpaying, negative = overpaying ---
        labor = len(firm.current_workers)
        if labor > 0:
            mpl      = firm.marginal_product_labor(firm.productivity, labor, firm.alpha)
            vmpl     = mpl * firm.output_price
            vmpl_gap = float(np.tanh((vmpl - firm.monthly_wage) / max(firm.monthly_wage, 1.0)))
        else:
            vmpl_gap = 1.0

        # --- wage position vs market ---
        other_wages = [f.monthly_wage for f in model.firms if f is not firm]
        market_wage = float(np.mean(other_wages)) if other_wages else firm.monthly_wage
        wage_vs_mkt = float(np.tanh((firm.monthly_wage - market_wage) / max(market_wage, 1.0)))

        # --- workforce signals ---
        labor_ratio   = labor / 40.0
        vacancy_ratio = min(firm.vacancies, 5) / 5.0
        worker_change = float(np.tanh((labor - self.prev_workers) / 3.0))
        wage_clock    = (self.current_step % 12) / 11.0

        # --- inherent capacity (reduces bad-luck variance) ---
        avg_prod    = float(np.mean([f.productivity for f in model.firms]))
        avg_cap     = float(np.mean([f.capital      for f in model.firms]))
        prod_vs_mkt = float(np.tanh((firm.productivity - avg_prod) / max(avg_prod, 1.0)))
        cap_vs_mkt  = float(np.tanh((firm.capital      - avg_cap)  / max(avg_cap,  1.0)))

        # --- survival urgency: rises toward 1 as deficit_months → 24 ---
        survival_signal = float(np.tanh(firm.deficit_months / 12.0))

        # --- labour market tightness ---
        employed          = sum(1 for w in model.workers if w.employed)
        market_employment = employed / len(model.workers) if model.workers else 0.0

        obs = np.array([
            profit_signal,
            profit_change_signal,
            vmpl_gap,
            wage_vs_mkt,
            labor_ratio,
            vacancy_ratio,
            worker_change,
            wage_clock,
            prod_vs_mkt,
            cap_vs_mkt,
            survival_signal,
            market_employment,
        ], dtype=np.float32)

        return np.clip(obs, -1.5, 1.5)

    # ------------------------------------------------------------------ #
    #  Action mask                                                         #
    # ------------------------------------------------------------------ #

    def action_masks(self) -> np.ndarray:
        wage_step = (self.current_step % 12 == 0)
        return np.array(
            [True, wage_step, wage_step, wage_step, wage_step, True, True],
            dtype=bool,
        )

    # ------------------------------------------------------------------ #
    #  Step / Reset                                                        #
    # ------------------------------------------------------------------ #

    def step(self, action):
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

    def reset(self, seed=None, options=None):
        self.model        = LaborMarketModel()
        self.rl_firm      = self.model.firms[self.rl_firm_id]
        self.current_step = 0
        self.prev_profit  = 0.0
        self.prev_workers = len(self.rl_firm.current_workers)
        return self.observe(), {}
