# cooperative/train.py  — train 3 RL firms with shared (cooperative) reward

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from firm_env import CoopFirmEnv
from rl_vis import LaborMetricsCallback


def mask_fn(env):
    return env.action_masks()


N_ENVS = 4
raw_env = DummyVecEnv([
    lambda: ActionMasker(CoopFirmEnv(), mask_fn)
    for _ in range(N_ENVS)
])

env = VecNormalize(raw_env, norm_obs=False, norm_reward=True, clip_reward=5.0)

model = MaskablePPO(
    "MlpPolicy",
    env,
    verbose=1,
    n_steps=512,
    batch_size=256,
    n_epochs=10,
    gamma=0.95,
    gae_lambda=0.95,
    learning_rate=3e-4,
    clip_range=0.2,
    max_grad_norm=0.5,
    ent_coef=0.02,
    vf_coef=0.5,
    tensorboard_log="./tensorboard_logs/",
    device="auto"
)

callback = LaborMetricsCallback(
    log_dir="./tensorboard_logs",
    algo_name="Coop_MaskablePPO",
    keep_runs=3,
)

model.learn(
    total_timesteps=1_000_000,
    callback=callback
)

model.save("coop_model")
env.save("coop_vecnorm.pkl")
print("Saved: coop_model.zip  coop_vecnorm.pkl")
