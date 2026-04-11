# solo/run.py  — evaluate 1 RL firm vs 9 heuristic

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv
from firm_env import LaborMarketEnv

MODEL_PATH = "solo_model"
N_STEPS    = 360

ACTION_LABELS = {
    0: "hold     ", 1: "wage+300 ", 2: "wage+100 ",
    3: "wage-100 ", 4: "wage-300 ", 5: "post_vac ", 6: "fire     ",
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

rl_profits, h_profits_all = [], []

for step in range(N_STEPS):
    masks  = np.array(vec_env.env_method("action_masks"))
    action, _ = policy.predict(obs, deterministic=True, action_masks=masks)
    obs, reward, done, info = vec_env.step(action)

    inner = vec_env.envs[0].env
    sim   = inner.model
    rl    = inner.rl_firm

    employed    = sum(1 for w in sim.workers if w.employed)
    employ_rate = employed / len(sim.workers)
    heuristic   = [f for f in sim.firms if f is not rl]
    h_profits   = [f.profit for f in heuristic]

    rl_profits.append(rl.profit)
    h_profits_all.append(np.mean(h_profits))

    print(SEP)
    print(f"  Step {step:>3}  |  employment: {employ_rate:.1%}  |  action: {ACTION_LABELS[int(action[0])]}")
    print(SEP)
    print(f"  {'Firm':<8}  {'Type':<9}  {'Profit':>10}  {'Wage':>7}  {'Workers':>7}  {'Vac':>3}  {'VMPL gap':>10}  {'Capital':>9}")
    print(SEP2)

    gap = vmpl_gap(rl)
    gap_str = f"{gap:>10,.0f}" if not np.isnan(gap) else f"{'n/a':>10}"
    print(f"  {'F0 [RL]':<8}  {'RL':<9}  {rl.profit:>10,.0f}  {rl.monthly_wage:>7,}  "
          f"{len(rl.current_workers):>7}  {rl.vacancies:>3}  {gap_str}  {rl.capital:>9,.1f}")
    print(SEP2)

    for i, f in enumerate(heuristic, start=1):
        gap = vmpl_gap(f)
        gap_str = f"{gap:>10,.0f}" if not np.isnan(gap) else f"{'n/a':>10}"
        print(f"  {f'F{i}':<8}  {'heuristic':<9}  {f.profit:>10,.0f}  {f.monthly_wage:>7,}  "
              f"{len(f.current_workers):>7}  {f.vacancies:>3}  {gap_str}  {f.capital:>9,.1f}")

    print(SEP2)
    avg_h_profit = np.mean(h_profits) if h_profits else 0
    rl_rank      = sorted([f.profit for f in sim.firms], reverse=True).index(rl.profit) + 1
    print(f"  AVG heuristic profit: {avg_h_profit:>10,.0f}")
    print(f"  RL profit rank: {rl_rank}/{len(sim.firms)}  "
          f"({'ABOVE' if rl.profit > avg_h_profit else 'BELOW'} avg by {abs(rl.profit - avg_h_profit):,.0f})")

    if done[0]:
        print(f"\n{'  [episode ended — resetting]':^100}")
        obs = vec_env.reset()

print(f"\n{'='*60}")
print(f"  SOLO SUMMARY ({N_STEPS} steps)")
print(f"{'='*60}")
print(f"  Avg RL profit:        {np.mean(rl_profits):>10,.0f}")
print(f"  Avg heuristic profit: {np.mean(h_profits_all):>10,.0f}")
delta = np.mean(rl_profits) - np.mean(h_profits_all)
print(f"  RL vs heuristic:      {delta:>+10,.0f}  ({'BETTER' if delta > 0 else 'WORSE'})")
