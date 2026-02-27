# train_rl.py

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from rl_env import LaborMarketEnv

env = DummyVecEnv([lambda: LaborMarketEnv()])

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=256,
    gamma=0.99
)

model.learn(total_timesteps=200_000)

model.save("labor_market_rl")