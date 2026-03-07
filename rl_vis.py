from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class LaborMetricsCallback(BaseCallback):

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self):

        env = self.training_env.envs[0]   # access your environment
        model = env.model

        employed_workers = [w for w in model.workers if w.employed]

        avg_wage = np.mean([w.monthly_wage for w in employed_workers] or [0])
        employment = len(employed_workers) / len(model.workers)
        avg_profit = np.mean([f.profit for f in model.firms])

        # log custom metrics
        self.logger.record("economy/avg_wage", avg_wage)
        self.logger.record("economy/employment_rate", employment)
        self.logger.record("economy/avg_profit", avg_profit)

        return True