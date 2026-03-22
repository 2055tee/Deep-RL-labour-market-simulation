# competitive/firm_env.py
#
# Multi-agent env: N_RL_FIRMS firms compete against each other.
# COMPETITIVE mode: reward = own_reward - 0.5 * mean(other RL rewards).
# Each firm is incentivised to beat its RL peers, not just earn profit.
#
# Observation (13 features):
#   0  profit_signal       — tanh(profit/20k)             own profit level
#   1  profit_change       — tanh(Δprofit/5k)             own profit trend
#   2  vmpl_gap            — tanh((VMPL-wage)/wage)        over/underpaying vs MPL
#   3  wage_vs_mkt         — tanh((own-mkt)/mkt)          wage vs ALL firms' avg
#   4  labor_ratio         — n_workers/40                 workforce size
#   5  vacancy_ratio       — min(vac,5)/5                 open slots
#   6  worker_change       — tanh(Δworkers/3)             recent headcount Δ
#   7  wage_clock          — step%12/11                   position in wage cycle
#   8  prod_vs_mkt         — tanh((A-avgA)/avgA)          own productivity vs ALL
#   9  cap_vs_mkt          — tanh((K-avgK)/avgK)          own capital vs ALL
#  10  profit_vs_peers     — tanh((own-RL_avg_profit)/20k) am I beating peers?
#  11  wage_vs_peers       — tanh((own-RL_avg_wage)/RL_avg) am I outbidding peers?
#  12  workers_vs_peers    — tanh((own-RL_avg_workers)/5)  am I winning talent war?

import gymnasium as gym
import numpy as np
from model_rl import LaborMarketModel

N_RL_FIRMS = 3


class CompFirmEnv(gym.Env):

    def __init__(self):
        self.model        = LaborMarketModel(n_rl_firms=N_RL_FIRMS)
        self.rl_firms     = self.model.firms[:N_RL_FIRMS]
        self.current_idx  = 0
        self.current_step = 0
        self.max_step     = 360

        self.prev_profit  = {f.uid: 0.0                    for f in self.rl_firms}
        self.prev_workers = {f.uid: len(f.current_workers) for f in self.rl_firms}

        self.action_space = gym.spaces.Discrete(7)
        self.observation_space = gym.spaces.Box(
            low=-1.5, high=1.5, shape=(13,), dtype=np.float32
        )

    # ------------------------------------------------------------------ #
    #  Observation for a single RL firm                                    #
    # ------------------------------------------------------------------ #

    def _observe(self, idx):
        firm  = self.rl_firms[idx]
        model = self.model

        # --- own performance ---
        profit_signal        = float(np.tanh(firm.profit / 20_000))
        profit_change_signal = float(np.tanh(
            (firm.profit - self.prev_profit[firm.uid]) / 5_000
        ))

        # --- VMPL gap ---
        labor = len(firm.current_workers)
        if labor > 0:
            mpl      = firm.marginal_product_labor(firm.productivity, labor, firm.alpha)
            vmpl     = mpl * firm.output_price
            vmpl_gap = float(np.tanh((vmpl - firm.monthly_wage) / max(firm.monthly_wage, 1.0)))
        else:
            vmpl_gap = 1.0

        # --- wage vs ALL firms ---
        all_wages   = [f.monthly_wage for f in model.firms]
        market_wage = float(np.mean(all_wages))
        wage_vs_mkt = float(np.tanh((firm.monthly_wage - market_wage) / max(market_wage, 1.0)))

        # --- workforce signals ---
        labor_ratio   = labor / 40.0
        vacancy_ratio = min(firm.vacancies, 5) / 5.0
        worker_change = float(np.tanh(
            (labor - self.prev_workers[firm.uid]) / 3.0
        ))
        wage_clock = (self.current_step % 12) / 11.0

        # --- own capacity vs market ---
        avg_prod    = float(np.mean([f.productivity for f in model.firms]))
        avg_cap     = float(np.mean([f.capital      for f in model.firms]))
        prod_vs_mkt = float(np.tanh((firm.productivity - avg_prod) / max(avg_prod, 1.0)))
        cap_vs_mkt  = float(np.tanh((firm.capital      - avg_cap)  / max(avg_cap,  1.0)))

        # --- peer comparison signals (critical for competitive strategy) ---
        peers = [f for f in self.rl_firms if f is not firm]

        peer_avg_profit  = float(np.mean([f.profit            for f in peers])) if peers else 0.0
        peer_avg_wage    = float(np.mean([f.monthly_wage       for f in peers])) if peers else firm.monthly_wage
        peer_avg_workers = float(np.mean([len(f.current_workers) for f in peers])) if peers else labor

        # am I beating them on profit?
        profit_vs_peers  = float(np.tanh((firm.profit - peer_avg_profit) / 20_000))
        # am I outbidding them for workers?
        wage_vs_peers    = float(np.tanh(
            (firm.monthly_wage - peer_avg_wage) / max(peer_avg_wage, 1.0)
        ))
        # am I winning the talent war?
        workers_vs_peers = float(np.tanh((labor - peer_avg_workers) / 5.0))

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
            profit_vs_peers,
            wage_vs_peers,
            workers_vs_peers,
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
    #  Step                                                                #
    # ------------------------------------------------------------------ #

    def step(self, action):
        firm = self.rl_firms[self.current_idx]

        if self.current_idx == 0:
            for f in self.rl_firms:
                self.prev_profit[f.uid]  = f.profit
                self.prev_workers[f.uid] = len(f.current_workers)

        firm.rl_action = action

        self.current_idx = (self.current_idx + 1) % N_RL_FIRMS

        if self.current_idx == 0:
            self.model.step()
            self.current_step += 1

        # Competitive: own reward minus half of peers' average reward
        peers_reward = float(np.mean([
            f.reward for f in self.rl_firms if f is not firm
        ])) if N_RL_FIRMS > 1 else 0.0
        reward = firm.reward - 0.5 * peers_reward

        obs = self._observe(self.current_idx)

        terminated = False
        truncated  = (self.current_idx == 0) and (self.current_step >= self.max_step)
        if truncated:
            self.current_step = 0

        return obs, reward, terminated, truncated, {}

    # ------------------------------------------------------------------ #
    #  Reset                                                               #
    # ------------------------------------------------------------------ #

    def reset(self, seed=None, options=None):
        self.model        = LaborMarketModel(n_rl_firms=N_RL_FIRMS)
        self.rl_firms     = self.model.firms[:N_RL_FIRMS]
        self.current_idx  = 0
        self.current_step = 0
        self.prev_profit  = {f.uid: 0.0                    for f in self.rl_firms}
        self.prev_workers = {f.uid: len(f.current_workers) for f in self.rl_firms}
        return self._observe(0), {}
