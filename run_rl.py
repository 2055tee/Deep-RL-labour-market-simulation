# run_rl.py — 1 RL firm (F0), rest heuristic

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv
from firm_env import LaborMarketEnv

MODEL_PATH = "firm_rl_2"
N_STEPS    = 500

ACTION_LABELS = {
    0: "hold     ",
    1: "wage+300 ",
    2: "wage+100 ",
    3: "wage-100 ",
    4: "wage-300 ",
    5: "post_vac ",
    6: "fire     ",
}

def mask_fn(env):
    return env.action_masks()

def vmpl_gap(firm):
    labor = len(firm.current_workers)
    if labor > 0:
        mpl = firm.marginal_product_labor(firm.productivity, labor, firm.alpha)
        return mpl * firm.output_price - firm.monthly_wage
    return float("nan")

vec_env = DummyVecEnv([lambda: ActionMasker(LaborMarketEnv(), mask_fn)])
policy  = MaskablePPO.load(MODEL_PATH, env=vec_env)
obs     = vec_env.reset()

SEP  = "=" * 100
SEP2 = "-" * 100

for step in range(N_STEPS):
    masks  = np.array(vec_env.env_method("action_masks"))
    action, _ = policy.predict(obs, deterministic=True, action_masks=masks)
    obs, reward, done, info = vec_env.step(action)

    inner = vec_env.envs[0].env        # LaborMarketEnv
    sim   = inner.model
    rl    = inner.rl_firm

    employed    = sum(1 for w in sim.workers if w.employed)
    employ_rate = employed / len(sim.workers)
    heuristic   = [f for f in sim.firms if f is not rl]
    h_profits   = [f.profit for f in heuristic]

    # ------------------------------------------------------------------ #
    print(SEP)
    print(f"  Step {step:>3}  |  employment: {employ_rate:.1%}  |  "
          f"action: {ACTION_LABELS[int(action[0])]}")
    print(SEP)

    # header
    print(f"  {'Firm':<6}  {'Type':<9}  {'Profit':>10}  {'Wage':>7}  "
          f"{'Workers':>7}  {'Vac':>3}  {'VMPL gap':>10}  {'Capital':>9}")
    print(SEP2)

    # RL firm row (always first)
    gap = vmpl_gap(rl)
    gap_str = f"{gap:>10,.0f}" if not np.isnan(gap) else f"{'n/a':>10}"
    print(f"  {'F0 [RL]':<6}  {'RL':<9}  {rl.profit:>10,.0f}  {rl.monthly_wage:>7,}  "
          f"{len(rl.current_workers):>7}  {rl.vacancies:>3}  {gap_str}  {rl.capital:>9,.1f}")

    print(SEP2)

    # heuristic firms
    for i, f in enumerate(heuristic, start=1):
        gap = vmpl_gap(f)
        gap_str = f"{gap:>10,.0f}" if not np.isnan(gap) else f"{'n/a':>10}"
        print(f"  {f'F{i}':<6}  {'heuristic':<9}  {f.profit:>10,.0f}  {f.monthly_wage:>7,}  "
              f"{len(f.current_workers):>7}  {f.vacancies:>3}  {gap_str}  {f.capital:>9,.1f}")

    # summary comparison
    print(SEP2)
    avg_h_profit = np.mean(h_profits) if h_profits else 0
    avg_h_wage   = np.mean([f.monthly_wage for f in heuristic]) if heuristic else 0
    avg_h_work   = np.mean([len(f.current_workers) for f in heuristic]) if heuristic else 0
    rl_rank      = sorted([f.profit for f in sim.firms], reverse=True).index(rl.profit) + 1

    print(f"  {'AVG heur':<6}  {'':<9}  {avg_h_profit:>10,.0f}  {avg_h_wage:>7,.0f}  "
          f"{avg_h_work:>7.1f}")
    print(f"\n  RL firm profit rank: {rl_rank} / {len(sim.firms)}   "
          f"({'ABOVE' if rl.profit > avg_h_profit else 'BELOW'} heuristic avg by "
          f"{abs(rl.profit - avg_h_profit):,.0f})")

    if done[0]:
        print(f"\n{'  [episode ended — resetting]':^100}")
        obs = vec_env.reset()
