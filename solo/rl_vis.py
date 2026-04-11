import shutil
import numpy as np
from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback

ACTION_NAMES = {
    0: "hold",
    1: "wage_up_300",
    2: "wage_up_100",
    3: "wage_dn_100",
    4: "wage_dn_300",
    5: "post_vacancy",
    6: "fire_worker",
}

class LaborMetricsCallback(BaseCallback):

    def __init__(self, log_dir="./tensorboard_logs", algo_name="MaskablePPO",
                 keep_runs=3, verbose=0):
        super().__init__(verbose)
        self.log_dir    = Path(log_dir)
        self.algo_name  = algo_name
        self.keep_runs  = keep_runs
        self.action_counts = np.zeros(len(ACTION_NAMES), dtype=np.int64)

    # ------------------------------------------------------------------ #
    #  On training start: delete all but the last `keep_runs` log folders #
    # ------------------------------------------------------------------ #

    def _on_training_start(self):
        try:
            runs = []
            for d in self.log_dir.iterdir():
                if d.is_dir() and d.name.startswith(self.algo_name + "_"):
                    try:
                        num = int(d.name.rsplit("_", 1)[-1])
                        runs.append((num, d))
                    except ValueError:
                        pass
            runs.sort(key=lambda x: x[0])
            to_delete = runs[: -self.keep_runs] if len(runs) > self.keep_runs else []
            for _, run_dir in to_delete:
                shutil.rmtree(run_dir)
                print(f"[cleanup] removed old run: {run_dir.name}")
        except Exception:
            pass  # never crash training over housekeeping

    # ------------------------------------------------------------------ #
    #  Every step                                                          #
    # ------------------------------------------------------------------ #

    def _on_step(self) -> bool:
        env   = self.training_env.envs[0].env   # unwrap ActionMasker → LaborMarketEnv
        sim   = env.model
        firm  = env.rl_firm

        # ---- track actions ----
        actions = self.locals.get("actions")
        if actions is not None:
            self.action_counts[int(actions[0])] += 1

        # ================================================================
        # 1. RL FIRM — key profit-maximisation signals
        # ================================================================
        labor = len(firm.current_workers)

        self.logger.record("firm/profit",        firm.profit)
        self.logger.record("firm/monthly_wage",  firm.monthly_wage)
        self.logger.record("firm/n_workers",     labor)
        self.logger.record("firm/vacancies",     firm.vacancies)
        self.logger.record("firm/reward",        firm.reward)
        self.logger.record("firm/capital",       firm.capital)

        # VMPL gap: positive = underpaying (should hire/raise wage)
        if labor > 0:
            mpl  = firm.marginal_product_labor(firm.productivity, labor, firm.alpha)
            vmpl = mpl * firm.output_price
            self.logger.record("firm/vmpl",      vmpl)
            self.logger.record("firm/vmpl_gap",  vmpl - firm.monthly_wage)

        # ================================================================
        # 2. ECONOMY — market context
        # ================================================================
        employed = [w for w in sim.workers if w.employed]

        market_wage  = np.mean([f.monthly_wage for f in sim.firms] or [0])
        avg_profit   = np.mean([f.profit       for f in sim.firms] or [0])
        employ_rate  = len(employed) / len(sim.workers) if sim.workers else 0

        self.logger.record("economy/market_avg_wage",   market_wage)
        self.logger.record("economy/avg_profit_all",    avg_profit)
        self.logger.record("economy/employment_rate",   employ_rate)

        # How the RL firm compares to the market
        if market_wage > 0:
            self.logger.record("economy/firm_wage_premium",
                               (firm.monthly_wage - market_wage) / market_wage)
        self.logger.record("economy/firm_vs_avg_profit",
                           firm.profit - avg_profit)

        # ================================================================
        # 3. SURVIVAL & MARKET TIGHTNESS
        # ================================================================
        self.logger.record("firm/deficit_months",    firm.deficit_months)
        self.logger.record("economy/employment_rate", employ_rate)

        # ================================================================
        # 4. ACTION DISTRIBUTION — what is the agent learning to do?
        # ================================================================
        total = self.action_counts.sum()
        if total > 0:
            for idx, name in ACTION_NAMES.items():
                self.logger.record(f"actions/{name}",
                                   self.action_counts[idx] / total)

        return True
