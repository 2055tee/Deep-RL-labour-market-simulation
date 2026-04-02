# reformed/firm_env.py
#
# Gymnasium env: 1 RL firm vs heuristic firms.
# Key differences from solo/firm_env.py:
#   - Uses reformed/model.py (market-quit, Option 3+5 always on)
#   - use_wage_gap_prob=True (Option 4 active during training)
#   - Vacancy cap MAX_VACANCIES=5 enforced in model.rl_decision
#   - 13-feature obs: standard 12 + at_risk_fraction (quit warning signal)
#
# Observation (13 features):
#   0  profit_signal       — tanh(profit/5k)            own profit level
#   1  profit_change       — tanh(dp/2k)                own profit trend
#   2  vmpl_gap            — tanh((VMPL-wage)/wage)      over/underpaying vs MPL
#   3  wage_vs_mkt         — tanh((own-mkt)/mkt)         wage premium vs all
#   4  labor_ratio         — n_workers/40               workforce size
#   5  vacancy_ratio       — min(vac,5)/5               open slots
#   6  worker_change       — tanh(dworkers/3)           recent headcount delta
#   7  wage_clock          — step%12/11                 position in wage cycle
#   8  prod_vs_mkt         — tanh((A-avgA)/avgA)        own productivity vs peers
#   9  cap_vs_mkt          — tanh((K-avgK)/avgK)        own capital vs peers
#  10  survival_signal     — tanh(deficit_months/12)    urgency to turn profitable
#  11  market_employment   — n_employed/n_workers       tight vs loose labour market
#  12  at_risk_fraction    — workers w/ months_below_mkt>=2 / max(labor,1)  quit warning

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import gymnasium as gym
import numpy as np
from model import LaborMarketModel


class ReformedFirmEnv(gym.Env):

    # Number of firms in the market — used to randomise which slot is the RL firm
    N_FIRMS = 10

    def __init__(self, N_workers=100, N_firms=10, min_wage=7700,
                 market_quit_patience=None, market_quit_threshold=None,
                 equal_terms=True):
        self.rl_firm_id   = "F0"
        self._init_kwargs = dict(
            N_workers=N_workers, N_firms=N_firms, min_wage=min_wage,
            market_quit_patience=market_quit_patience,
            market_quit_threshold=market_quit_threshold,
            equal_terms=equal_terms,
        )
        self.model        = LaborMarketModel(use_wage_gap_prob=True,
                                             rl_firm_id=self.rl_firm_id,
                                             **self._init_kwargs)
        self.rl_firm      = self._find_rl_firm()
        self.current_step = 0
        self.max_step     = 360

        self.prev_profit  = 0.0
        self.prev_workers = len(self.rl_firm.current_workers)

        # 8 actions: hold / wage+300 / wage+100 / wage-100 / wage-300 / post_vacancy / fire_worker / snap_to_market
        self.action_space = gym.spaces.Discrete(8)
        self.observation_space = gym.spaces.Box(
            low=-1.5, high=1.5, shape=(13,), dtype=np.float32
        )

    def _find_rl_firm(self):
        for f in self.model.firms:
            if f.uid == self.rl_firm_id:
                return f
        raise RuntimeError(f"RL firm {self.rl_firm_id!r} not found")

    # ── Observation ──────────────────────────────────────────────────

    def observe(self):
        firm  = self.rl_firm
        model = self.model

        profit_signal        = float(np.tanh(firm.profit / 5_000))
        profit_change_signal = float(np.tanh((firm.profit - self.prev_profit) / 2_000))

        labor = len(firm.current_workers)
        if labor > 0:
            mpl      = firm.marginal_product_labor(firm.productivity, labor, firm.alpha)
            vmpl     = mpl * firm.output_price
            vmpl_gap = float(np.tanh((vmpl - firm.monthly_wage) / max(firm.monthly_wage, 1.0)))
        else:
            vmpl_gap = 1.0

        other_wages = [f.monthly_wage for f in model.firms if f is not firm]
        market_wage = float(np.mean(other_wages)) if other_wages else firm.monthly_wage
        wage_vs_mkt = float(np.tanh((firm.monthly_wage - market_wage) / max(market_wage, 1.0)))

        labor_ratio   = labor / 40.0
        vacancy_ratio = min(firm.vacancies, 5) / 5.0
        worker_change = float(np.tanh((labor - self.prev_workers) / 3.0))
        wage_clock    = (self.current_step % 12) / 11.0

        avg_prod    = float(np.mean([f.productivity for f in model.firms]))
        avg_cap     = float(np.mean([f.capital      for f in model.firms]))
        prod_vs_mkt = float(np.tanh((firm.productivity - avg_prod) / max(avg_prod, 1.0)))
        cap_vs_mkt  = float(np.tanh((firm.capital      - avg_cap)  / max(avg_cap,  1.0)))

        survival_signal = float(np.tanh(firm.deficit_months / 12.0))

        employed          = sum(1 for w in model.workers if w.employed)
        market_employment = employed / len(model.workers) if model.workers else 0.0

        # Quit warning: fraction of own workers who've been below market wage
        # for >= half the patience period. Lets the policy pre-empt mass
        # market-quit by snapping/raising wages in time.
        patience = self.model.market_quit_patience
        at_risk = sum(1 for w in firm.current_workers
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
            market_employment,
            at_risk_fraction,
        ], dtype=np.float32)

        return np.clip(obs, -1.5, 1.5)

    # ── Action mask ──────────────────────────────────────────────────

    def action_masks(self) -> np.ndarray:
        wage_step = (self.current_step % 12 == 0)
        # action 7 (snap_to_market) available every step — same as post_vacancy/fire
        return np.array(
            [True, wage_step, wage_step, wage_step, wage_step, True, True, True],
            dtype=bool,
        )

    # ── Step / Reset ─────────────────────────────────────────────────

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
        # Fresh random seed each episode → diverse market configurations.
        # rl_firm_id stays "F0" so the policy specialises on one market
        # position (different starting capital/productivity per seed still
        # gives meaningful diversity without making the problem too hard).
        env_seed = seed if seed is not None else int(np.random.randint(0, 2**31))

        self.model        = LaborMarketModel(use_wage_gap_prob=True,
                                             rl_firm_id=self.rl_firm_id,
                                             seed=env_seed,
                                             **self._init_kwargs)
        self.rl_firm      = self._find_rl_firm()
        self.current_step = 0
        self.prev_profit  = 0.0
        self.prev_workers = len(self.rl_firm.current_workers)
        return self.observe(), {}
