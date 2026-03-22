#!/usr/bin/env python
# compare_all.py
#
# Loads all three trained policies and evaluates them in fresh simulations.
# Prints a side-by-side performance comparison table at the end.
#
# Uses each scenario's own env class so observations are always in sync
# with the trained policy — no manual feature engineering needed here.
#
# Usage:
#   python compare_all.py

import sys
import os
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.resolve()

N_STEPS    = 360   # model-level steps per evaluation episode
N_RL_FIRMS = 3     # firms used in coop / comp

# ------------------------------------------------------------------ #
#  Helper: safely reload a scenario's modules                         #
# ------------------------------------------------------------------ #

def _load_scenario(name):
    d = ROOT / name
    if str(d) in sys.path:
        sys.path.remove(str(d))
    sys.path.insert(0, str(d))
    for mod in ["model_rl", "firm_env", "rl_vis"]:
        sys.modules.pop(mod, None)
    return d


def _unload_scenario(name):
    d = str(ROOT / name)
    if d in sys.path:
        sys.path.remove(d)


# ------------------------------------------------------------------ #
#  Solo evaluation                                                    #
# ------------------------------------------------------------------ #

def run_solo():
    d = _load_scenario("solo")

    from sb3_contrib import MaskablePPO
    from firm_env import LaborMarketEnv

    model_path = d / "solo_model.zip"
    if not model_path.exists():
        print(f"  [solo] Model not found — skipping.")
        _unload_scenario("solo")
        return None

    policy = MaskablePPO.load(str(d / "solo_model"))
    env    = LaborMarketEnv()
    obs, _ = env.reset()

    rl_log, h_log = [], []

    for _ in range(N_STEPS):
        mask          = env.action_masks()
        action, _     = policy.predict(obs[np.newaxis], deterministic=True,
                                       action_masks=mask[np.newaxis])
        obs, _, _, _, _ = env.step(int(action[0]))

        rl_log.append(env.rl_firm.profit)
        heuristic = [f for f in env.model.firms if f is not env.rl_firm]
        h_log.append(float(np.mean([f.profit for f in heuristic])) if heuristic else 0.0)

    _unload_scenario("solo")
    return {"rl": np.array(rl_log), "heuristic": np.array(h_log)}


# ------------------------------------------------------------------ #
#  Cooperative evaluation                                             #
# ------------------------------------------------------------------ #

def run_cooperative():
    d = _load_scenario("cooperative")

    from sb3_contrib import MaskablePPO
    from firm_env import CoopFirmEnv

    model_path = d / "coop_model.zip"
    if not model_path.exists():
        print(f"  [cooperative] Model not found — skipping.")
        _unload_scenario("cooperative")
        return None

    policy = MaskablePPO.load(str(d / "coop_model"))
    env    = CoopFirmEnv()
    obs, _ = env.reset()

    rl_log, h_log = [], []

    # Round-robin: N_RL_FIRMS gym steps = 1 model step
    for _ in range(N_STEPS * N_RL_FIRMS):
        mask          = env.action_masks()
        action, _     = policy.predict(obs[np.newaxis], deterministic=True,
                                       action_masks=mask[np.newaxis])
        obs, _, _, _, _ = env.step(int(action[0]))

        # Record once per completed model step (after all firms acted)
        if env.current_idx == 0:
            rl_log.append(float(np.mean([f.profit for f in env.rl_firms])))
            heuristic = [f for f in env.model.firms if f not in env.rl_firms]
            h_log.append(float(np.mean([f.profit for f in heuristic])) if heuristic else 0.0)

    _unload_scenario("cooperative")
    return {"rl": np.array(rl_log), "heuristic": np.array(h_log)}


# ------------------------------------------------------------------ #
#  Competitive evaluation                                             #
# ------------------------------------------------------------------ #

def run_competitive():
    d = _load_scenario("competitive")

    from sb3_contrib import MaskablePPO
    from firm_env import CompFirmEnv

    model_path = d / "comp_model.zip"
    if not model_path.exists():
        print(f"  [competitive] Model not found — skipping.")
        _unload_scenario("competitive")
        return None

    policy = MaskablePPO.load(str(d / "comp_model"))
    env    = CompFirmEnv()
    obs, _ = env.reset()

    rl_log, h_log = [], []

    for _ in range(N_STEPS * N_RL_FIRMS):
        mask          = env.action_masks()
        action, _     = policy.predict(obs[np.newaxis], deterministic=True,
                                       action_masks=mask[np.newaxis])
        obs, _, _, _, _ = env.step(int(action[0]))

        if env.current_idx == 0:
            rl_log.append(float(np.mean([f.profit for f in env.rl_firms])))
            heuristic = [f for f in env.model.firms if f not in env.rl_firms]
            h_log.append(float(np.mean([f.profit for f in heuristic])) if heuristic else 0.0)

    _unload_scenario("competitive")
    return {"rl": np.array(rl_log), "heuristic": np.array(h_log)}


# ------------------------------------------------------------------ #
#  Run all three                                                      #
# ------------------------------------------------------------------ #

print("\nRunning evaluations...")
print("  [1/3] Solo        (1 RL firm  vs 9 heuristic)...")
solo_res = run_solo()

print("  [2/3] Cooperative (3 RL firms vs 7 heuristic, shared reward)...")
coop_res = run_cooperative()

print("  [3/3] Competitive (3 RL firms vs 7 heuristic, relative reward)...")
comp_res = run_competitive()


# ------------------------------------------------------------------ #
#  Comparison table                                                   #
# ------------------------------------------------------------------ #

SEP = "=" * 85

def fmt(val):
    return f"{val:>12,.0f}" if val is not None else f"{'N/A':>12}"

print(f"\n{SEP}")
print(f"  {'SCENARIO COMPARISON  (avg over 360 model steps)':^81}")
print(SEP)
print(f"  {'Metric':<35}  {'Solo':>12}  {'Cooperative':>12}  {'Competitive':>12}")
print(f"  {'-'*35}  {'-'*12}  {'-'*12}  {'-'*12}")

metrics = {}
for name, res in [("solo", solo_res), ("coop", coop_res), ("comp", comp_res)]:
    if res is not None:
        metrics[name] = {
            "avg_rl":   float(np.mean(res["rl"])),
            "avg_h":    float(np.mean(res["heuristic"])),
            "delta":    float(np.mean(res["rl"])) - float(np.mean(res["heuristic"])),
            "peak_rl":  float(np.max(res["rl"])),
            "final_rl": float(res["rl"][-1]),
        }
    else:
        metrics[name] = None


def row(label, key):
    vals = {n: (m[key] if m else None) for n, m in metrics.items()}
    print(f"  {label:<35}  {fmt(vals['solo'])}  {fmt(vals['coop'])}  {fmt(vals['comp'])}")

row("Avg RL profit  (whole run)",   "avg_rl")
row("Avg heuristic profit",         "avg_h")
row("RL vs heuristic  (delta)",     "delta")
row("Peak RL profit  (best step)",  "peak_rl")
row("Final RL profit (last step)",  "final_rl")
print(SEP)

# ── Winner per metric ──────────────────────────────────────────────
print("\n  WINNER BY METRIC:")
for label, key in [
    ("Avg RL profit",         "avg_rl"),
    ("RL vs heuristic edge",  "delta"),
    ("Peak RL profit",        "peak_rl"),
]:
    candidates = {k: v[key] for k, v in metrics.items() if v is not None}
    if candidates:
        winner = max(candidates, key=candidates.get)
        print(f"    {label:<28}  ->  {winner.upper():<12}  ({candidates[winner]:+,.0f})")

print(f"\n{SEP}\n")
