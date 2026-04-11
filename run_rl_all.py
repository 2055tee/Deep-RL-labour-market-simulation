# run_rl_all.py — ALL firms use the same trained policy

import numpy as np
from sb3_contrib import MaskablePPO
from model_rl import LaborMarketModel

MODEL_PATH = "firm_rl_2"
N_STEPS    = 360

ACTION_LABELS = {
    0: "hold", 1: "+300", 2: "+100",
    3: "-100",  4: "-300", 5: "vac",  6: "fire",
}

# ------------------------------------------------------------------ #
#  Helpers — mirror firm_env.py exactly so the policy sees            #
#  the same observation distribution it was trained on                #
# ------------------------------------------------------------------ #

def observe_firm(firm, prev_profit, prev_workers, step, market_wage):
    labor = len(firm.current_workers)

    profit_signal        = float(np.tanh(firm.profit / 20_000))
    profit_change_signal = float(np.tanh((firm.profit - prev_profit) / 5_000))

    if labor > 0:
        mpl      = firm.marginal_product_labor(firm.productivity, labor, firm.alpha)
        vmpl     = mpl * firm.output_price
        vmpl_gap = float(np.tanh((vmpl - firm.monthly_wage) / max(firm.monthly_wage, 1.0)))
    else:
        vmpl_gap = 1.0

    wage_vs_mkt   = float(np.tanh((firm.monthly_wage - market_wage) / max(market_wage, 1.0)))
    labor_ratio   = labor / 40.0
    vacancy_ratio = min(firm.vacancies, 5) / 5.0
    worker_change = float(np.tanh((labor - prev_workers) / 3.0))
    wage_clock    = (step % 12) / 11.0

    obs = np.array([
        profit_signal, profit_change_signal, vmpl_gap, wage_vs_mkt,
        labor_ratio, vacancy_ratio, worker_change, wage_clock,
    ], dtype=np.float32)
    return np.clip(obs, -1.5, 1.5)


def action_mask(step):
    wage_ok = (step % 12 == 0)
    return np.array([True, wage_ok, wage_ok, wage_ok, wage_ok, True, True], dtype=bool)


# ------------------------------------------------------------------ #
#  Setup                                                              #
# ------------------------------------------------------------------ #

policy = MaskablePPO.load(MODEL_PATH)

sim = LaborMarketModel()
# Disable the built-in rl_stage so it doesn't try to control only F0.
# We drive ALL firms manually below, then run the other stages ourselves.
sim.rl_firm_id = "__none__"

# Initialise per-firm tracking (mirrors firm_env reset state)
prev_profit  = {f.uid: 0.0                      for f in sim.firms}
prev_workers = {f.uid: len(f.current_workers)   for f in sim.firms}

SEP  = "=" * 100
SEP2 = "-" * 100

# ------------------------------------------------------------------ #
#  Simulation loop                                                    #
# ------------------------------------------------------------------ #

for step in range(N_STEPS):

    market_wage = float(np.mean([f.monthly_wage for f in sim.firms]))
    mask        = action_mask(step)

    # 1. RL decision for every firm
    firm_actions = {}
    for firm in sim.firms:
        obs    = observe_firm(firm, prev_profit[firm.uid], prev_workers[firm.uid],
                              step, market_wage)
        action, _ = policy.predict(
            obs[np.newaxis],
            deterministic=True,
            action_masks=mask[np.newaxis],
        )
        a = int(action[0])
        firm.rl_action = a
        firm_actions[firm.uid] = a
        firm.rl_decision()

    # 2. Run simulation stages manually
    #    — skip adjust_employment_step: RL controls vacancies & firing via actions 5/6
    agents = list(sim.schedule.agents)
    for agent in agents: agent.onboard_workers_step()
    for agent in agents: agent.job_search_step()
    for agent in agents: agent.hire_step()

    # snapshot workers AFTER hiring is settled (before step() changes profit)
    next_workers = {f.uid: len(f.current_workers) for f in sim.firms}

    for agent in agents: agent.step()   # computes production, profit, reward

    # 3. Update tracking for next iteration
    prev_profit  = {f.uid: f.profit for f in sim.firms}
    prev_workers = next_workers

    # 4. Print per-step table
    employed    = sum(1 for w in sim.workers if w.employed)
    avg_wage    = float(np.mean([f.monthly_wage         for f in sim.firms]))
    avg_profit  = float(np.mean([f.profit               for f in sim.firms]))
    avg_workers = float(np.mean([len(f.current_workers) for f in sim.firms]))

    profits_sorted = sorted([f.profit for f in sim.firms], reverse=True)

    print(SEP)
    print(f"  Step {step:>3}  |  employment: {employed/len(sim.workers):.1%}  |  "
          f"market_wage: {market_wage:,.0f}  |  avg_profit: {avg_profit:,.0f}  |  "
          f"avg_workers: {avg_workers:.1f}")
    print(SEP)
    print(f"  {'Firm':<6}  {'Action':<9}  {'Profit':>10}  {'Rank':>4}  "
          f"{'Wage':>7}  {'Workers':>7}  {'Vac':>3}  {'VMPL gap':>10}  {'Capital':>9}")
    print(SEP2)

    for i, firm in enumerate(sim.firms):
        labor = len(firm.current_workers)
        if labor > 0:
            mpl      = firm.marginal_product_labor(firm.productivity, labor, firm.alpha)
            gap      = mpl * firm.output_price - firm.monthly_wage
            gap_str  = f"{gap:>10,.0f}"
        else:
            gap_str  = f"{'n/a':>10}"

        rank    = profits_sorted.index(firm.profit) + 1
        act_lbl = ACTION_LABELS[firm_actions[firm.uid]]
        print(f"  {f'F{i}':<6}  {act_lbl:<9}  {firm.profit:>10,.0f}  {rank:>4}  "
              f"{firm.monthly_wage:>7,}  {labor:>7}  {firm.vacancies:>3}  "
              f"{gap_str}  {firm.capital:>9,.1f}")

    print(SEP2)
    best   = sim.firms[profits_sorted.index(max(profits_sorted))] if sim.firms else None
    print(f"  {'AVERAGE':<6}  {'':<9}  {avg_profit:>10,.0f}  {'':>4}  "
          f"{avg_wage:>7,.0f}  {avg_workers:>7.1f}")
