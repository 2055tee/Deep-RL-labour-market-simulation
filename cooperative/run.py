# cooperative/run.py  — evaluate 3 cooperative RL firms vs 7 heuristic

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv
from firm_env import CoopFirmEnv, N_RL_FIRMS

MODEL_PATH = "coop_model"
N_STEPS    = 360   # model steps

ACTION_LABELS = {
    0: "hold", 1: "+300", 2: "+100",
    3: "-100", 4: "-300", 5: "vac",  6: "fire",
}

def mask_fn(env):
    return env.action_masks()

def vmpl_gap(firm):
    labor = len(firm.current_workers)
    if labor > 0:
        mpl = firm.marginal_product_labor(firm.productivity, labor, firm.alpha)
        return mpl * firm.output_price - firm.monthly_wage
    return float("nan")

# Use a fresh CoopFirmEnv in eval mode (deterministic policy)
vec_env = DummyVecEnv([lambda: ActionMasker(CoopFirmEnv(), mask_fn)])
policy  = MaskablePPO.load(MODEL_PATH, env=vec_env)

# We run the evaluation manually to properly cycle through all RL firms
from model_rl import LaborMarketModel

policy_bare = MaskablePPO.load(MODEL_PATH)

sim = LaborMarketModel(n_rl_firms=N_RL_FIRMS)
rl_firms   = sim.firms[:N_RL_FIRMS]
heuristic  = sim.firms[N_RL_FIRMS:]

prev_profit  = {f.uid: 0.0                    for f in rl_firms}
prev_workers = {f.uid: len(f.current_workers) for f in rl_firms}

SEP  = "=" * 110
SEP2 = "-" * 110

rl_profit_log, h_profit_log = [], []


def observe_firm(firm, prev_p, prev_w, step):
    model = sim
    profit_signal        = float(np.tanh(firm.profit / 20_000))
    profit_change_signal = float(np.tanh((firm.profit - prev_p) / 5_000))
    labor = len(firm.current_workers)
    if labor > 0:
        mpl      = firm.marginal_product_labor(firm.productivity, labor, firm.alpha)
        vmpl     = mpl * firm.output_price
        vmpl_gap = float(np.tanh((vmpl - firm.monthly_wage) / max(firm.monthly_wage, 1.0)))
    else:
        vmpl_gap = 1.0
    all_wages   = [f.monthly_wage for f in model.firms]
    market_wage = float(np.mean(all_wages))
    wage_vs_mkt = float(np.tanh((firm.monthly_wage - market_wage) / max(market_wage, 1.0)))
    labor_ratio   = labor / 40.0
    vacancy_ratio = min(firm.vacancies, 5) / 5.0
    worker_change = float(np.tanh((labor - prev_w) / 3.0))
    wage_clock    = (step % 12) / 11.0
    avg_prod    = float(np.mean([f.productivity for f in model.firms]))
    avg_cap     = float(np.mean([f.capital      for f in model.firms]))
    prod_vs_mkt = float(np.tanh((firm.productivity - avg_prod) / max(avg_prod, 1.0)))
    cap_vs_mkt  = float(np.tanh((firm.capital      - avg_cap)  / max(avg_cap,  1.0)))
    obs = np.array([profit_signal, profit_change_signal, vmpl_gap, wage_vs_mkt,
                    labor_ratio, vacancy_ratio, worker_change, wage_clock,
                    prod_vs_mkt, cap_vs_mkt], dtype=np.float32)
    return np.clip(obs, -1.5, 1.5)


def action_mask(step):
    w = (step % 12 == 0)
    return np.array([True, w, w, w, w, True, True], dtype=bool)


for step in range(N_STEPS):
    mask = action_mask(step)
    firm_actions = {}

    for firm in rl_firms:
        obs    = observe_firm(firm, prev_profit[firm.uid], prev_workers[firm.uid], step)
        action, _ = policy_bare.predict(obs[np.newaxis], deterministic=True, action_masks=mask[np.newaxis])
        a = int(action[0])
        firm.rl_action = a
        firm_actions[firm.uid] = a

    # Snapshot before step
    for f in rl_firms:
        prev_profit[f.uid]  = f.profit
        prev_workers[f.uid] = len(f.current_workers)

    sim.step()

    employed    = sum(1 for w in sim.workers if w.employed)
    employ_rate = employed / len(sim.workers)
    avg_rl_p    = float(np.mean([f.profit for f in rl_firms]))
    avg_h_p     = float(np.mean([f.profit for f in heuristic])) if heuristic else 0
    rl_profit_log.append(avg_rl_p)
    h_profit_log.append(avg_h_p)

    print(SEP)
    print(f"  Step {step:>3}  |  employment: {employ_rate:.1%}  |  "
          f"avg RL profit: {avg_rl_p:,.0f}  |  avg heuristic profit: {avg_h_p:,.0f}")
    print(SEP)
    print(f"  {'Firm':<8}  {'Type':<9}  {'Action':<7}  {'Profit':>10}  "
          f"{'Wage':>7}  {'Workers':>7}  {'Vac':>3}  {'VMPL gap':>10}")
    print(SEP2)

    for i, f in enumerate(rl_firms):
        gap = vmpl_gap(f)
        gs  = f"{gap:>10,.0f}" if not np.isnan(gap) else f"{'n/a':>10}"
        print(f"  {f'F{i} [RL]':<8}  {'RL':<9}  {ACTION_LABELS[firm_actions[f.uid]]:<7}  "
              f"{f.profit:>10,.0f}  {f.monthly_wage:>7,}  {len(f.current_workers):>7}  {f.vacancies:>3}  {gs}")

    print(SEP2)
    for i, f in enumerate(heuristic):
        gap = vmpl_gap(f)
        gs  = f"{gap:>10,.0f}" if not np.isnan(gap) else f"{'n/a':>10}"
        print(f"  {f'H{i}':<8}  {'heuristic':<9}  {'—':<7}  "
              f"{f.profit:>10,.0f}  {f.monthly_wage:>7,}  {len(f.current_workers):>7}  {f.vacancies:>3}  {gs}")
    print(SEP2)
    print(f"  RL vs heuristic: {avg_rl_p - avg_h_p:>+10,.0f}")

print(f"\n{'='*60}")
print(f"  COOPERATIVE SUMMARY ({N_STEPS} steps)")
print(f"{'='*60}")
print(f"  Avg RL profit (3 firms):   {np.mean(rl_profit_log):>10,.0f}")
print(f"  Avg heuristic profit:      {np.mean(h_profit_log):>10,.0f}")
delta = np.mean(rl_profit_log) - np.mean(h_profit_log)
print(f"  RL vs heuristic:           {delta:>+10,.0f}  ({'BETTER' if delta > 0 else 'WORSE'})")
