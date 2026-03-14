# train_firm.py

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv
from firm_env import LaborMarketEnv
from rl_vis import LaborMetricsCallback


def mask_fn(env):
    return env.action_masks()


N_ENVS = 4
env = DummyVecEnv([
    lambda: ActionMasker(LaborMarketEnv(), mask_fn)
    for _ in range(N_ENVS)
])

model = MaskablePPO(
    "MlpPolicy",
    env,
    verbose=1,

    # --- rollout ---
    n_steps=512,       # per env; 512 × 4 envs = 2048 total per update
    batch_size=256,
    n_epochs=10,

    # --- discount & advantage ---
    gamma=0.95,
    gae_lambda=0.95,

    # --- policy gradient ---
    learning_rate=3e-4,
    clip_range=0.2,
    max_grad_norm=0.5,

    # --- entropy bonus: encourages exploring all valid actions ---
    ent_coef=0.02,
    vf_coef=0.5,

    tensorboard_log="./tensorboard_logs/",
    device="auto"
)

callback = LaborMetricsCallback(
    log_dir="./tensorboard_logs",
    algo_name="MaskablePPO",
    keep_runs=3,          # keeps the 3 most recent runs, deletes the rest
)

model.learn(
    total_timesteps=1_000_000,
    callback=callback
)

model.save("firm_rl_2")
