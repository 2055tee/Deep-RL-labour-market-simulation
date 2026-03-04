# run_trained_model.py

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from rl_env import LaborMarketEnv
import numpy as np

# -----------------------
# Create environment
# -----------------------
env = DummyVecEnv([lambda: LaborMarketEnv()])

# -----------------------
# Load trained model
# -----------------------
model = PPO.load("labor_market_rl", env=env)

# -----------------------
# Run simulation
# -----------------------
obs = env.reset()

n_steps = 600   # how long you want to simulate

for step in range(n_steps):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    avg_wage = obs[0][0] * 50000
    employment = obs[0][1]
    unemployment = obs[0][2]
    avg_profit = obs[0][3] * 100000

    print(f"Step {step}")
    print(f"Avg Wage: {avg_wage:.2f}")
    print(f"Employment Rate: {employment:.2f}")
    print(f"Unemployment Rate: {unemployment:.2f}")
    print(f"Avg Firm Profit: {avg_profit:.2f}")
    print(f"Reward: {reward}")
    print("-"*40)