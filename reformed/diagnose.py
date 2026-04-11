# reformed/diagnose.py  — run 500 steps, trace worker-count drops
#
# Run:  python reformed/diagnose.py

import sys, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from model import LaborMarketModel, MARKET_QUIT_THRESHOLD, MARKET_QUIT_PATIENCE

try:
    from sb3_contrib import MaskablePPO
    _POLICY = MaskablePPO.load(str(Path(__file__).parent / "reformed_model"))
    print("Policy loaded.\n")
except Exception as e:
    _POLICY = None
    print(f"No policy: {e}\n")

ACTION_NAMES = ["hold","wage+300","wage+100","wage-100","wage-300","post_vac","fire","snap_mkt"]
N_STEPS  = 500
N_SEEDS  = 5
DROP_THR = 4   # flag any single-step drop >= this many workers

# ── Observation (mirrors firm_env.py) ─────────────────────────────────

def _obs(m, rl, prev_profit, prev_wkrs, step):
    firm  = rl
    labor = len(firm.current_workers)
    ps    = float(np.tanh(firm.profit / 20_000))
    pcs   = float(np.tanh((firm.profit - prev_profit) / 5_000))
    if labor > 0:
        mpl  = firm.marginal_product_labor(firm.productivity, labor, firm.alpha)
        vmpl = mpl * firm.output_price
        vg   = float(np.tanh((vmpl - firm.monthly_wage) / max(firm.monthly_wage, 1.0)))
    else:
        vg = 1.0
    ow       = [f.monthly_wage for f in m.firms if f is not firm]
    mw       = float(np.mean(ow)) if ow else firm.monthly_wage
    wvm      = float(np.tanh((firm.monthly_wage - mw) / max(mw, 1.0)))
    lr       = labor / 40.0
    vr       = min(firm.vacancies, 5) / 5.0
    wc       = float(np.tanh((labor - prev_wkrs) / 3.0))
    wclk     = (step % 12) / 11.0
    ap       = float(np.mean([f.productivity for f in m.firms]))
    ac       = float(np.mean([f.capital      for f in m.firms]))
    pvm      = float(np.tanh((firm.productivity - ap) / max(ap, 1.0)))
    cvm      = float(np.tanh((firm.capital      - ac) / max(ac, 1.0)))
    surv     = float(np.tanh(firm.deficit_months / 12.0))
    emp      = sum(1 for w in m.workers if w.employed)
    mkte     = emp / len(m.workers) if m.workers else 0.0
    af       = m.active_firms()
    avgw     = float(np.mean([len(f.current_workers) for f in af])) if af else labor
    wms      = float(np.tanh((labor - avgw) / 5.0))
    obs = np.array([ps,pcs,vg,wvm,lr,vr,wc,wclk,pvm,cvm,surv,mkte,wms], dtype=np.float32)
    return np.clip(obs, -1.5, 1.5)

def _mask(step):
    wo = (step % 12 == 0)
    return np.array([True, wo, wo, wo, wo, True, True, True], dtype=bool)

# ── Single run ─────────────────────────────────────────────────────────

def run_one(seed):
    m = LaborMarketModel(N_workers=100, N_firms=10,
                         use_wage_gap_prob=True, rl_firm_id="F0", seed=seed)
    rl = next(f for f in m.firms if f.uid == "F0")

    prev_profit = 0.0
    prev_wkrs   = len(rl.current_workers)
    history     = []   # (step, workers, wage, mkt_wage, action, profit)
    events      = []   # large drops

    for step in range(N_STEPS):
        wkrs_before = len(rl.current_workers)
        wage_before = rl.monthly_wage

        # Policy action
        if _POLICY:
            obs  = _obs(m, rl, prev_profit, prev_wkrs, step)
            mask = _mask(step)
            act, _ = _POLICY.predict(obs[np.newaxis], deterministic=True,
                                     action_masks=mask[np.newaxis])
            m.rl_action = int(act[0])
        else:
            m.rl_action = 0

        # Count workers about to hit market-quit
        active = m.active_firms()
        mkt_w  = float(np.mean([f.monthly_wage for f in active])) if active else 0
        at_risk = sum(
            1 for w in rl.current_workers
            if w.months_below_mkt >= MARKET_QUIT_PATIENCE - 1
            and w.monthly_wage < MARKET_QUIT_THRESHOLD * mkt_w
        )

        prev_profit = rl.profit
        prev_wkrs   = wkrs_before
        m.step()

        wkrs_after = len(rl.current_workers)
        drop       = wkrs_before - wkrs_after
        active2    = m.active_firms()
        mkt_w2     = float(np.mean([f.monthly_wage for f in active2])) if active2 else 0

        history.append((step, wkrs_after, rl.monthly_wage, mkt_w2,
                        m.rl_action, rl.profit))

        if drop >= DROP_THR:
            # Diagnose the cause
            threshold = MARKET_QUIT_THRESHOLD * mkt_w
            events.append({
                "step":         step,
                "drop":         drop,
                "before":       wkrs_before,
                "after":        wkrs_after,
                "action":       ACTION_NAMES[m.rl_action],
                "wage_before":  wage_before,
                "wage_after":   rl.monthly_wage,
                "mkt_wage":     round(mkt_w, 0),
                "threshold":    round(threshold, 0),
                "was_below":    wage_before < threshold,
                "at_risk_prev": at_risk,
                "profit":       round(rl.profit, 0),
                "deficit_mo":   rl.deficit_months,
            })

    return history, events


# ── Report ─────────────────────────────────────────────────────────────

SEP  = "=" * 65
SEP2 = "-" * 65

all_events = []

for seed in range(N_SEEDS):
    hist, events = run_one(seed)
    all_events.extend(events)

    workers_ts = [h[1] for h in hist]
    wages_ts   = [h[2] for h in hist]
    profits_ts = [h[5] for h in hist]

    print(SEP)
    print(f"  SEED {seed}  — {N_STEPS} steps")
    print(SEP2)
    print(f"  RL workers : min={min(workers_ts):3d}  max={max(workers_ts):3d}  "
          f"mean={np.mean(workers_ts):5.1f}  std={np.std(workers_ts):.1f}")
    print(f"  RL wage    : min={min(wages_ts):6,}  max={max(wages_ts):6,}  "
          f"mean={np.mean(wages_ts):7.0f}")
    print(f"  RL profit  : min={min(profits_ts):9,.0f}  max={max(profits_ts):9,.0f}  "
          f"mean={np.mean(profits_ts):9.0f}")
    print(f"  Big drops (>={DROP_THR} workers in 1 step): {len(events)}")
    for ev in events:
        cause = []
        if ev["was_below"]:
            cause.append(f"market-quit (wage {ev['wage_before']:,} < "
                         f"91% x mkt {ev['mkt_wage']:,} = {ev['threshold']:,})")
        if ev["action"] == "fire":
            cause.append("RL fired a worker")
        if ev["action"].startswith("wage"):
            cause.append(f"wage change ({ev['action']})")
        if not cause:
            cause.append("utility-quit or other")
        print(f"    step {ev['step']:3d}: {ev['before']} -> {ev['after']} workers "
              f"(-{ev['drop']})  action={ev['action']}  "
              f"cause: {', '.join(cause)}")

# ── Global summary ─────────────────────────────────────────────────────
print()
print(SEP)
print("  GLOBAL SUMMARY  (all seeds)")
print(SEP2)

mq_drops = sum(1 for e in all_events if e["was_below"])
fi_drops = sum(1 for e in all_events if e["action"] == "fire")
ot_drops = len(all_events) - mq_drops - fi_drops
print(f"  Total big drops : {len(all_events)}")
print(f"  Cause breakdown :")
print(f"    Market-quit  (wage < 91% mkt)  : {mq_drops}")
print(f"    RL fired worker                : {fi_drops}")
print(f"    Other (utility-quit etc.)      : {ot_drops}")
print()

if mq_drops > 0:
    print("  DIAGNOSIS: Market-quit is the main driver of sudden drops.")
    print(f"  Workers accumulate months_below_mkt over {MARKET_QUIT_PATIENCE} months,")
    print(f"  then all quit at once when threshold ({MARKET_QUIT_THRESHOLD:.0%}) is crossed.")
    print()
    print("  IS IT BAD? It depends:")
    print("  - GOOD if RL wage is genuinely below market (workers should leave).")
    print("  - BAD if the simultaneous mass-quit is unrealistically abrupt.")
    print()
    print("  FIX OPTIONS:")
    print("  A) Stagger quit: workers quit one-per-step (not all at once)")
    print("     -> in job_search_step, break after first quit per worker group")
    print("  B) Raise MARKET_QUIT_PATIENCE from 4 to 6-8 months")
    print("     -> workers tolerate low wages longer before leaving")
    print("  C) Add quit noise: quit prob = sigmoid(months_below / patience)")
    print("     -> gradual probabilistic drain instead of cliff")
if fi_drops > 0:
    print("  DIAGNOSIS: RL is actively firing workers (action=fire).")
    print("  This is GOOD — it means the agent learned to trim excess workforce.")
if ot_drops > 0:
    print("  DIAGNOSIS: Some drops from utility-quit (outside option > work utility).")
    print("  These are rare and economically correct.")

print(SEP)
