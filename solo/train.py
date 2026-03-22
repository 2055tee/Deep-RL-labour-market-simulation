# solo/train.py  — train 1 RL firm vs heuristic
#
# Uses VecNormalize(norm_reward=True) to stabilize training across episodes
# where the RL firm gets different random capital/productivity each reset.
# The reward running-mean/std is saved alongside the model.

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from firm_env import LaborMarketEnv
from rl_vis import LaborMetricsCallback


def mask_fn(env):
    return env.action_masks()


N_ENVS = 4
raw_env = DummyVecEnv([
    lambda: ActionMasker(LaborMarketEnv(), mask_fn)
    for _ in range(N_ENVS)
])

# norm_obs=False  — obs is already normalised via tanh / manual scaling
# norm_reward=True — normalise reward to unit variance so bad-luck episodes
#                    (low capital/productivity) don't drown out good ones
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
    algo_name="Solo_MaskablePPO",
    keep_runs=3,
)

model.learn(
    total_timesteps=1_000_000,
    callback=callback
)

model.save("solo_model")
env.save("solo_vecnorm.pkl")   # save reward stats so run.py can optionally load them
print("Saved: solo_model.zip  solo_vecnorm.pkl")
