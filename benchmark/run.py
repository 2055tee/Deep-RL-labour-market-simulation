#!/usr/bin/env python
# benchmark/run.py
#
# Systematic benchmark: RL firm vs all-heuristic baseline across many seeds and
# parameter configurations.  Results are saved to benchmark/results.csv for
# analysis with benchmark/analyse.py.
#
# Run: python benchmark/run.py
#
# Design:
#   BASE sweep  — 100 seeds, default params. Main "RL better than heuristic?"
#   PARAM sweeps — 30 seeds each, vary one parameter at a time:
#       N_firms:              [5, 10, 15]
#       N_workers:            [50, 100, 150]
#       min_wage:             [6000, 7700, 9000]
#       market_quit_patience: [2, 4, 8]
#       market_quit_threshold:[0.85, 0.91, 0.95]
#       equal_terms:          [True, False]
#
# Each "run" records final-step stats for both the RL market and the
# all-heuristic baseline (same seed, same params).

import sys
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "reformed"))

import csv
import random
import numpy as np
import time
from model import LaborMarketModel

# ── RL model ────────────────────────────────────────────────────────
try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from firm_env import ReformedFirmEnv
    _HAS_RL = True
except Exception as _e:
    _HAS_RL = False
    print(f"[benchmark] WARNING: RL deps not available: {_e}")

MODEL_PATH = ROOT / "reformed" / "reformed_model.zip"
NORM_PATH  = ROOT / "reformed" / "reformed_vecnorm.pkl"
N_STEPS    = 350   # match eval.py

OUT_CSV    = Path(__file__).parent / "results.csv"

CSV_FIELDS = [
    "sweep", "seed",
    "N_workers", "N_firms", "min_wage",
    "market_quit_patience", "market_quit_threshold", "equal_terms",
    # RL run metrics
    "rl_final_profit", "rl_avg_profit", "rl_final_workers",
    "rl_final_wage", "rl_employment_pct",
    "rl_market_avg_profit", "rl_active_firms",
    # Heuristic run metrics
    "base_final_profit", "base_avg_profit", "base_final_workers",
    "base_final_wage", "base_employment_pct",
    "base_market_avg_profit", "base_active_firms",
]


def _mask_fn(env):
    return env.action_masks()


def run_one_rl(seed, params):
    """Single RL evaluation run. Returns dict of metrics."""
    if not _HAS_RL or not MODEL_PATH.exists():
        return None

    random.seed(seed)
    np.random.seed(seed)

    rl_model = MaskablePPO.load(str(MODEL_PATH))

    env_kwargs = dict(
        N_workers=params["N_workers"],
        N_firms=params["N_firms"],
        min_wage=params["min_wage"],
        market_quit_patience=params["market_quit_patience"],
        market_quit_threshold=params["market_quit_threshold"],
        equal_terms=params["equal_terms"],
    )
    raw = DummyVecEnv([lambda: ActionMasker(ReformedFirmEnv(**env_kwargs), _mask_fn)])
    vec = VecNormalize.load(str(NORM_PATH), raw)
    vec.training    = False
    vec.norm_reward = False

    obs   = vec.reset()
    inner = vec.envs[0].env  # ReformedFirmEnv

    profits, wages, workers = [], [], []

    for _ in range(N_STEPS):
        masks  = np.array(vec.env_method("action_masks"))
        action, _ = rl_model.predict(obs, action_masks=masks, deterministic=True)
        obs, _, done, _ = vec.step(action)
        if done[0]:
            obs = vec.reset()

        m    = inner.model
        firm = inner.rl_firm
        profits.append(firm.profit)
        wages.append(firm.monthly_wage)
        workers.append(len(firm.current_workers))

    active   = m.active_firms()
    employed = sum(1 for w in m.workers if w.employed)
    emp_pct  = 100.0 * employed / max(len(m.workers), 1)
    mkt_avg  = float(np.mean([f.profit for f in active])) if active else 0.0

    vec.close()
    return {
        "rl_final_profit":      profits[-1],
        "rl_avg_profit":        float(np.mean(profits)),
        "rl_final_workers":     workers[-1],
        "rl_final_wage":        wages[-1],
        "rl_employment_pct":    emp_pct,
        "rl_market_avg_profit": mkt_avg,
        "rl_active_firms":      len(active),
    }


def run_one_heuristic(seed, params):
    """Single heuristic baseline run. Returns dict of metrics."""
    random.seed(seed)
    np.random.seed(seed)

    m = LaborMarketModel(
        N_workers=params["N_workers"],
        N_firms=params["N_firms"],
        use_wage_gap_prob=True,
        rl_firm_id=None,
        equal_terms=params["equal_terms"],
        min_wage=params["min_wage"],
        market_quit_patience=params["market_quit_patience"],
        market_quit_threshold=params["market_quit_threshold"],
        seed=seed,
    )
    f0 = m.firms[0]

    profits, wages, workers = [], [], []

    for _ in range(N_STEPS):
        m.step()
        profits.append(f0.profit if f0.active else 0.0)
        wages.append(f0.monthly_wage if f0.active else 0)
        workers.append(len(f0.current_workers) if f0.active else 0)

    active   = m.active_firms()
    employed = sum(1 for w in m.workers if w.employed)
    emp_pct  = 100.0 * employed / max(len(m.workers), 1)
    mkt_avg  = float(np.mean([f.profit for f in active])) if active else 0.0

    return {
        "base_final_profit":      profits[-1],
        "base_avg_profit":        float(np.mean(profits)),
        "base_final_workers":     workers[-1],
        "base_final_wage":        wages[-1],
        "base_employment_pct":    emp_pct,
        "base_market_avg_profit": mkt_avg,
        "base_active_firms":      len(active),
    }


def _worker(task):
    """Worker function for ProcessPoolExecutor."""
    sweep_name, seed, params = task
    row = {"sweep": sweep_name, "seed": seed, **params}

    h_res = run_one_heuristic(seed, params)
    row.update(h_res)

    if _HAS_RL and MODEL_PATH.exists():
        rl_res = run_one_rl(seed, params)
        if rl_res:
            row.update(rl_res)
        else:
            for k in ["rl_final_profit", "rl_avg_profit", "rl_final_workers",
                      "rl_final_wage", "rl_employment_pct",
                      "rl_market_avg_profit", "rl_active_firms"]:
                row[k] = float("nan")
    else:
        for k in ["rl_final_profit", "rl_avg_profit", "rl_final_workers",
                  "rl_final_wage", "rl_employment_pct",
                  "rl_market_avg_profit", "rl_active_firms"]:
            row[k] = float("nan")

    return row


# ─────────────────────────────────────────────────────────────────────
# Parameter grid
# ─────────────────────────────────────────────────────────────────────

DEFAULT = dict(
    N_workers=100, N_firms=10, min_wage=7700,
    market_quit_patience=4, market_quit_threshold=0.91, equal_terms=True,
)
BASE_SEEDS   = list(range(100))
PARAM_SEEDS  = list(range(30))


def _build_tasks():
    tasks = []

    # --- BASE sweep ---
    for seed in BASE_SEEDS:
        tasks.append(("base", seed, dict(DEFAULT)))

    # --- PARAM sweeps (vary one dimension at a time) ---
    for n_firms in [5, 15]:
        p = dict(DEFAULT); p["N_firms"] = n_firms
        for seed in PARAM_SEEDS:
            tasks.append((f"N_firms={n_firms}", seed, dict(p)))

    for n_workers in [50, 150]:
        p = dict(DEFAULT); p["N_workers"] = n_workers
        for seed in PARAM_SEEDS:
            tasks.append((f"N_workers={n_workers}", seed, dict(p)))

    for mw in [6000, 9000]:
        p = dict(DEFAULT); p["min_wage"] = mw
        for seed in PARAM_SEEDS:
            tasks.append((f"min_wage={mw}", seed, dict(p)))

    for patience in [2, 8]:
        p = dict(DEFAULT); p["market_quit_patience"] = patience
        for seed in PARAM_SEEDS:
            tasks.append((f"patience={patience}", seed, dict(p)))

    for thresh in [0.85, 0.95]:
        p = dict(DEFAULT); p["market_quit_threshold"] = thresh
        for seed in PARAM_SEEDS:
            tasks.append((f"threshold={thresh}", seed, dict(p)))

    # equal_terms=False (wide spread)
    p = dict(DEFAULT); p["equal_terms"] = False
    for seed in PARAM_SEEDS:
        tasks.append(("equal_terms=False", seed, dict(p)))

    return tasks


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not MODEL_PATH.exists():
        print(f"WARNING: {MODEL_PATH} not found — RL columns will be NaN.")

    tasks = _build_tasks()
    print(f"Total tasks: {len(tasks)}")

    # Sequential execution — ProcessPoolExecutor hangs on Windows due to
    # spawn context + module-level Gym warnings. Sequential is reliable and
    # still fast enough for 340 tasks (heuristic is <1s each; RL ~3s each).
    rows  = []
    total = len(tasks)

    for i, task in enumerate(tasks, 1):
        try:
            row = _worker(task)
            rows.append(row)
        except Exception as exc:
            print(f"  ERROR in {task[0]} seed={task[1]}: {exc}")
        if i % 50 == 0 or i == total:
            print(f"  {i}/{total} done")

    # Sort for reproducibility
    rows.sort(key=lambda r: (r["sweep"], r["seed"]))

    # Write CSV
    OUT_CSV.parent.mkdir(exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows -> {OUT_CSV}")
    print("Run: python benchmark/analyse.py")
