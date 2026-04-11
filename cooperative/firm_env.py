# cooperative/firm_env.py
#
# Multi-agent env: N_RL_FIRMS firms share the same simulation.
# COOPERATIVE mode: reward = average profit of all RL firms.
# Uses round-robin: each gym.step() acts for one RL firm in turn.
#
# Observation (13 features):
#   0  profit_signal       — tanh(profit/5k)             own profit level
#   1  profit_change       — tanh(Δprofit/2k)            own profit trend
#   2  vmpl_gap            — tanh((VMPL-wage)/wage)       over/underpaying vs MPL
#   3  wage_vs_mkt         — tanh((own-mkt)/mkt)         wage vs ALL firms' avg
#   4  labor_ratio         — n_workers/40                workforce size
#   5  vacancy_ratio       — min(vac,5)/5                open slots
#   6  worker_change       — tanh(Δworkers/3)            recent headcount Δ
#   7  wage_clock          — step%12/11                  position in wage cycle
#   8  prod_vs_mkt         — tanh((A-avgA)/avgA)         own productivity vs peers
#   9  cap_vs_mkt          — tanh((K-avgK)/avgK)         own capital vs peers
#  10  survival_signal     — tanh(deficit_months/12)     own survival urgency
#  11  team_profit_signal  — tanh(mean(RL profits)/5k)   collective team health
#  12  at_risk_fraction    — workers w/ months_below_mkt>=patience//2 / max(labor,1)

import gymnasium as gym
import numpy as np
from model_rl import LaborMarketModel

N_RL_FIRMS = 3


class CoopFirmEnv(gym.Env):

    def __init__(self, n_workers=100, n_firms=10):
        self.n_workers    = n_workers
        self.n_firms      = n_firms
        self.model        = LaborMarketModel(N_workers=n_workers, N_firms=n_firms,
                                             n_rl_firms=N_RL_FIRMS, equal_terms=True)
        self.rl_firms     = self.model.firms[:N_RL_FIRMS]
        self.current_idx  = 0
        self.current_step = 0
        self.max_step     = 360

        self.prev_profit  = {f.uid: 0.0                    for f in self.rl_firms}
        self.prev_workers = {f.uid: len(f.current_workers) for f in self.rl_firms}

        self.action_space = gym.spaces.Discrete(8)
        self.observation_space = gym.spaces.Box(
            low=-1.5, high=1.5, shape=(13,), dtype=np.float32
        )

    # ── Observation for one RL firm ──────────────────────────────────

    def _observe(self, idx):
        firm  = self.rl_firms[idx]
        model = self.model

        profit_signal        = float(np.tanh(firm.profit / 5_000))
        profit_change_signal = float(np.tanh(
            (firm.profit - self.prev_profit[firm.uid]) / 2_000
        ))

        labor = len(firm.current_workers)
        if labor > 0:
            mpl      = firm.marginal_product_labor(firm.productivity, labor, firm.alpha)
            vmpl     = mpl * firm.output_price
            vmpl_gap = float(np.tanh((vmpl - firm.monthly_wage) / max(firm.monthly_wage, 1.0)))
        else:
            vmpl_gap = 1.0

        all_wages   = [f.monthly_wage for f in model.firms]
        market_wage = float(np.mean(all_wages))
        wage_vs_mkt = float(np.tanh((firm.monthly_wage - market_wage) / max(market_wage, 1.0)))

        labor_ratio   = labor / 40.0
        vacancy_ratio = min(firm.vacancies, 5) / 5.0
        worker_change = float(np.tanh(
            (labor - self.prev_workers[firm.uid]) / 3.0
        ))
        wage_clock = (self.current_step % 12) / 11.0

        avg_prod    = float(np.mean([f.productivity for f in model.firms]))
        avg_cap     = float(np.mean([f.capital      for f in model.firms]))
        prod_vs_mkt = float(np.tanh((firm.productivity - avg_prod) / max(avg_prod, 1.0)))
        cap_vs_mkt  = float(np.tanh((firm.capital      - avg_cap)  / max(avg_cap,  1.0)))

        survival_signal = float(np.tanh(firm.deficit_months / 12.0))

        peer_profits       = [f.profit for f in self.rl_firms]
        team_profit_signal = float(np.tanh(float(np.mean(peer_profits)) / 5_000))

        patience = self.model.market_quit_patience
        at_risk  = sum(1 for w in firm.current_workers
                       if w.months_below_mkt >= patience // 2)
        at_risk_fraction = at_risk / max(labor, 1)

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
            team_profit_signal,
            at_risk_fraction,
        ], dtype=np.float32)
        return np.clip(obs, -1.5, 1.5)

    # ── Action mask ──────────────────────────────────────────────────

    def action_masks(self) -> np.ndarray:
        wage_step = (self.current_step % 12 == 0)
        return np.array(
            [True, wage_step, wage_step, wage_step, wage_step, True, True, True],
            dtype=bool,
        )

    # ── Step ─────────────────────────────────────────────────────────

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

        # Cooperative: shared reward = mean profit of all RL firms
        reward = float(np.mean([f.reward for f in self.rl_firms]))

        obs = self._observe(self.current_idx)

        terminated = False
        truncated  = (self.current_idx == 0) and (self.current_step >= self.max_step)
        if truncated:
            self.current_step = 0

        return obs, reward, terminated, truncated, {}

    # ── Reset ────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        import random as _random
        env_seed = seed if seed is not None else int(np.random.randint(0, 2**31))
        _random.seed(env_seed)
        np.random.seed(env_seed)

        self.model        = LaborMarketModel(N_workers=self.n_workers, N_firms=self.n_firms,
                                             n_rl_firms=N_RL_FIRMS,
                                             use_wage_gap_prob=True,
                                             equal_terms=True,
                                             seed=env_seed)
        self.rl_firms     = self.model.firms[:N_RL_FIRMS]
        self.current_idx  = 0
        self.current_step = 0
        self.prev_profit  = {f.uid: 0.0                    for f in self.rl_firms}
        self.prev_workers = {f.uid: len(f.current_workers) for f in self.rl_firms}
        return self._observe(0), {}
