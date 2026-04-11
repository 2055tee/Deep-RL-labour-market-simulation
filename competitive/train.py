# competitive/train.py  — train 3 RL firms with competitive (relative) reward
#
# Reformed rules: market-quit, Options 3+4+5, snap action (7), equal_terms
# Hyperparams aligned with reformed/train.py:
#   ent_coef 0.02 -> 0.08  (break hold-dominance plateau)
#   n_steps  1024 -> 2048
#   total_timesteps 2M -> 3M

import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from firm_env import CompFirmEnv
from rl_vis import LaborMetricsCallback


def mask_fn(env):
    return env.action_masks()


def linear_schedule(initial, final=1e-5):
    def fn(progress_remaining):
        return final + progress_remaining * (initial - final)
    return fn


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on: {device}")

N_ENVS = 4
raw_env = DummyVecEnv([
    lambda: ActionMasker(CompFirmEnv(), mask_fn)
    for _ in range(N_ENVS)
])

env = VecNormalize(raw_env, norm_obs=False, norm_reward=True, clip_reward=5.0)

model = MaskablePPO(
    "MlpPolicy",
    env,
    verbose=1,
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    learning_rate=linear_schedule(3e-4, 1e-5),
    clip_range=0.2,
    max_grad_norm=0.5,
    ent_coef=0.08,
    vf_coef=0.5,
    policy_kwargs=dict(net_arch=[256, 256]),
    tensorboard_log="./tensorboard_logs/",
    device=device
)

callback = LaborMetricsCallback(
    log_dir="./tensorboard_logs",
    algo_name="Comp_Reformed_MaskablePPO",
    keep_runs=3,
)

model.learn(
    total_timesteps=3_000_000,
    callback=callback
)

model.save("comp_model_longrun")
env.save("comp_vecnorm_longrun.pkl")
print("Saved: comp_model_longrun.zip  comp_vecnorm_longrun.pkl")
