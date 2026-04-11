# competitive/rl_vis.py — training callback for competitive scenario

import shutil
import numpy as np
from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback

ACTION_NAMES = {
    0: "hold", 1: "wage_up_300", 2: "wage_up_100",
    3: "wage_dn_100", 4: "wage_dn_300", 5: "post_vacancy",
    6: "fire_worker", 7: "snap_to_market",
}


class LaborMetricsCallback(BaseCallback):

    def __init__(self, log_dir="./tensorboard_logs", algo_name="Comp_MaskablePPO",
                 keep_runs=3, verbose=0):
        super().__init__(verbose)
        self.log_dir       = Path(log_dir)
        self.algo_name     = algo_name
        self.keep_runs     = keep_runs
        self.action_counts = np.zeros(len(ACTION_NAMES), dtype=np.int64)

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
        except Exception:
            pass

    def _on_step(self) -> bool:
        env = self.training_env.envs[0].env
        sim = env.model

        actions = self.locals.get("actions")
        if actions is not None:
            self.action_counts[int(actions[0])] += 1

        rl_profits  = [f.profit       for f in env.rl_firms]
        rl_wages    = [f.monthly_wage for f in env.rl_firms]
        rl_workers  = [len(f.current_workers) for f in env.rl_firms]

        self.logger.record("comp/avg_rl_profit",    float(np.mean(rl_profits)))
        self.logger.record("comp/max_rl_profit",    float(np.max(rl_profits)))
        self.logger.record("comp/min_rl_profit",    float(np.min(rl_profits)))
        self.logger.record("comp/profit_spread",    float(np.max(rl_profits) - np.min(rl_profits)))
        self.logger.record("comp/avg_rl_wage",      float(np.mean(rl_wages)))
        self.logger.record("comp/wage_spread",      float(np.max(rl_wages) - np.min(rl_wages)))
        self.logger.record("comp/avg_rl_workers",   float(np.mean(rl_workers)))
        self.logger.record("comp/worker_spread",    float(np.max(rl_workers) - np.min(rl_workers)))

        employed    = [w for w in sim.workers if w.employed]
        employ_rate = len(employed) / len(sim.workers) if sim.workers else 0
        avg_profit  = float(np.mean([f.profit for f in sim.firms]))
        market_wage = float(np.mean([f.monthly_wage for f in sim.firms]))

        self.logger.record("economy/employment_rate", employ_rate)
        self.logger.record("economy/avg_profit_all",  avg_profit)
        self.logger.record("economy/market_avg_wage", market_wage)

        heuristic = [f for f in sim.firms if f not in env.rl_firms]
        if heuristic:
            h_avg = float(np.mean([f.profit for f in heuristic]))
            self.logger.record("comp/rl_vs_heuristic", float(np.mean(rl_profits)) - h_avg)

        total = self.action_counts.sum()
        if total > 0:
            for idx, name in ACTION_NAMES.items():
                self.logger.record(f"actions/{name}", self.action_counts[idx] / total)

        return True
