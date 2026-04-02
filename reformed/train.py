# reformed/train.py — train RL firm in reformed environment
#
# Uses reformed/model.py which has:
#   - Market-quit mechanism (structural fix for worker hoarding)
#   - Option 3+5 always on, Option 4 (wage-gap prob) on during training
#   - Vacancy cap MAX_VACANCIES=5
#   - Firm replacement on exit
#
# Run: python reformed/train.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from firm_env import ReformedFirmEnv
from rl_vis import ReformedMetricsCallback

OUT_DIR = Path(__file__).parent


def mask_fn(env):
    return env.action_masks()


def linear_schedule(initial, final=1e-5):
    def fn(progress_remaining):
        return final + progress_remaining * (initial - final)
    return fn


N_ENVS = 4
raw_env = DummyVecEnv([
    lambda: ActionMasker(ReformedFirmEnv(), mask_fn)
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
    ent_coef=0.08,   # push hard on exploration to break hold-dominance plateau
    vf_coef=0.5,
    policy_kwargs=dict(net_arch=[256, 256]),
    tensorboard_log=str(OUT_DIR / "tensorboard_logs"),
    device="auto",
)

callback = ReformedMetricsCallback(
    log_dir=str(OUT_DIR / "tensorboard_logs"),
    algo_name="Reformed_MaskablePPO",
    keep_runs=3,
)

model.learn(
    total_timesteps=3_000_000,
    callback=callback,
)

model.save(str(OUT_DIR / "reformed_model"))
env.save(str(OUT_DIR / "reformed_vecnorm.pkl"))
print(f"Saved: {OUT_DIR/'reformed_model.zip'}  {OUT_DIR/'reformed_vecnorm.pkl'}")
