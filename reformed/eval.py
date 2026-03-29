# reformed/eval.py — evaluate trained RL firm vs heuristic baseline
#
# Compares:
#   A — RL firm (trained reformed_model.zip)
#   B — All-heuristic baseline (same model, no RL firm)
#
# Runs N_SEEDS seeds each, plots mean +/- std, prints verdict.
#
# Run: python reformed/eval.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from model import LaborMarketModel
from firm_env import ReformedFirmEnv

OUT_DIR    = Path(__file__).parent
MODEL_PATH = OUT_DIR / "reformed_model.zip"
NORM_PATH  = OUT_DIR / "reformed_vecnorm.pkl"

N_STEPS = 120
SEEDS   = [42, 123, 456, 789, 1000, 7, 13, 99, 2025, 314]


def _gini(values):
    w = np.array(values, dtype=float)
    if w.sum() == 0 or len(w) < 2:
        return 0.0
    w = np.sort(w)
    n = len(w)
    return (2 * np.dot(np.arange(1, n + 1), w) - (n + 1) * w.sum()) / (n * w.sum())


# ── Heuristic baseline ────────────────────────────────────────────────

def run_heuristic(seeds=SEEDS, n_steps=N_STEPS) -> dict:
    emp_h = []; profit_h = []; wage_h = []; firms_h = []; gini_h = []
    rl_workers_h = []; rl_profit_h = []; rl_wage_h = []

    for seed in seeds:
        m = LaborMarketModel(use_wage_gap_prob=False, rl_firm_id=None, seed=seed)
        emp_r = []; profit_r = []; wage_r = []; firms_r = []; gini_r = []
        rl_w = []; rl_p = []; rl_wg = []

        f0 = m.firms[0]  # track first firm as reference

        for _ in range(n_steps):
            m.step()
            active   = m.active_firms()
            employed = sum(1 for w in m.workers if w.employed)
            emp_r.append(100.0 * employed / len(m.workers))
            profit_r.append(float(np.mean([f.profit for f in active])) if active else 0.0)
            wages = [f.monthly_wage for f in active] if active else [0]
            wage_r.append(float(np.mean(wages)))
            firms_r.append(len(active))
            gini_r.append(_gini(wages))
            rl_w.append(len(f0.current_workers) if f0.active else 0)
            rl_p.append(f0.profit if f0.active else 0.0)
            rl_wg.append(f0.monthly_wage if f0.active else 0)

        emp_h.append(emp_r); profit_h.append(profit_r)
        wage_h.append(wage_r); firms_h.append(firms_r); gini_h.append(gini_r)
        rl_workers_h.append(rl_w); rl_profit_h.append(rl_p); rl_wage_h.append(rl_wg)

    return dict(
        employment  = np.array(emp_h),
        avg_profit  = np.array(profit_h),
        avg_wage    = np.array(wage_h),
        n_firms     = np.array(firms_h),
        wage_gini   = np.array(gini_h),
        rl_workers  = np.array(rl_workers_h),
        rl_profit   = np.array(rl_profit_h),
        rl_wage     = np.array(rl_wage_h),
    )


# ── RL evaluation ─────────────────────────────────────────────────────

def run_rl(rl_model, seeds=SEEDS, n_steps=N_STEPS) -> dict:
    emp_h = []; profit_h = []; wage_h = []; firms_h = []; gini_h = []
    rl_workers_h = []; rl_profit_h = []; rl_wage_h = []

    def mask_fn(env):
        return env.action_masks()

    for seed in seeds:
        raw = DummyVecEnv([lambda: ActionMasker(ReformedFirmEnv(), mask_fn)])
        vec = VecNormalize.load(str(NORM_PATH), raw)
        vec.training = False
        vec.norm_reward = False

        obs = vec.reset()
        env_inner = vec.envs[0].env  # ReformedFirmEnv

        # Re-seed
        import random
        random.seed(seed)
        np.random.seed(seed)

        emp_r = []; profit_r = []; wage_r = []; firms_r = []; gini_r = []
        rl_w = []; rl_p = []; rl_wg = []

        for step in range(n_steps):
            action_masks = np.array(vec.env_method("action_masks"))
            action, _ = rl_model.predict(obs, action_masks=action_masks, deterministic=True)
            obs, _, done, _ = vec.step(action)
            if done[0]:
                obs = vec.reset()

            m    = env_inner.model
            firm = env_inner.rl_firm
            active   = m.active_firms()
            employed = sum(1 for w in m.workers if w.employed)
            emp_r.append(100.0 * employed / len(m.workers))
            profit_r.append(float(np.mean([f.profit for f in active])) if active else 0.0)
            wages = [f.monthly_wage for f in active] if active else [0]
            wage_r.append(float(np.mean(wages)))
            firms_r.append(len(active))
            gini_r.append(_gini(wages))
            rl_w.append(len(firm.current_workers))
            rl_p.append(firm.profit)
            rl_wg.append(firm.monthly_wage)

        vec.close()
        emp_h.append(emp_r); profit_h.append(profit_r)
        wage_h.append(wage_r); firms_h.append(firms_r); gini_h.append(gini_r)
        rl_workers_h.append(rl_w); rl_profit_h.append(rl_p); rl_wage_h.append(rl_wg)

    return dict(
        employment  = np.array(emp_h),
        avg_profit  = np.array(profit_h),
        avg_wage    = np.array(wage_h),
        n_firms     = np.array(firms_h),
        wage_gini   = np.array(gini_h),
        rl_workers  = np.array(rl_workers_h),
        rl_profit   = np.array(rl_profit_h),
        rl_wage     = np.array(rl_wage_h),
    )


# ── Plots ─────────────────────────────────────────────────────────────

def plot_eval(res_rl: dict, res_base: dict, path: Path):
    steps = np.arange(1, N_STEPS + 1)
    COL_RL   = "#4fc3f7"
    COL_BASE = "#ffb74d"

    metrics = [
        ("employment", "Employment Rate (%)"),
        ("avg_profit",  "Avg Firm Profit (THB)"),
        ("avg_wage",    "Avg Market Wage (THB)"),
        ("n_firms",     "Active Firms"),
        ("wage_gini",   "Wage Gini (0=equal)"),
        ("rl_workers",  "RL/F0 Workers"),
        ("rl_profit",   "RL/F0 Profit (THB)"),
        ("rl_wage",     "RL/F0 Wage (THB)"),
    ]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(11, 3.5 * len(metrics)))
    fig.suptitle("Reformed RL vs Heuristic Baseline\n"
                 "(mean +/- 1 std across 10 seeds, 120 months)",
                 fontsize=13, fontweight="bold", y=1.002)

    for ax, (key, title) in zip(axes, metrics):
        for res, label, col in [
            (res_rl,   "RL firm",         COL_RL),
            (res_base, "All heuristic",   COL_BASE),
        ]:
            m = res[key].mean(axis=0)
            s = res[key].std(axis=0)
            ax.plot(steps, m, color=col, lw=2.2, label=label)
            ax.fill_between(steps, m - s, m + s, color=col, alpha=0.18)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Month", fontsize=8)
        ax.set_ylabel(title, fontsize=8)
        ax.legend(fontsize=8, framealpha=0.7)
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=8)

    plt.tight_layout(pad=0.5)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Summary ───────────────────────────────────────────────────────────

def _summarise(res: dict, label: str):
    print("\n" + "-" * 54)
    print(f"  {label}")
    print("-" * 54)
    for key in ["employment", "avg_profit", "avg_wage", "n_firms",
                "rl_workers", "rl_profit", "rl_wage"]:
        final = res[key][:, -1]
        early = res[key][:, :12].mean(axis=1)
        trend = final.mean() - early.mean()
        print(f"  {key:<22}  {final.mean():>10.1f}  std={final.std():>7.1f}  trend={trend:>+8.1f}")


def _verdict(res_rl: dict, res_base: dict):
    print("\n" + "=" * 54)
    print("  VERDICT — RL firm vs All-Heuristic")
    print("=" * 54)
    scorecard = [
        ("Employment %",   "employment", True),
        ("Market Wage",    "avg_wage",   True),
        ("Firms Alive",    "n_firms",    True),
        ("Wage Gini",      "wage_gini",  False),
        ("RL Workers",     "rl_workers", False),  # fewer = less hoarding
        ("RL Profit",      "rl_profit",  True),
    ]
    wins_rl = wins_base = 0
    for name, key, higher_better in scorecard:
        v_rl   = res_rl[key][:, -1].mean()
        v_base = res_base[key][:, -1].mean()
        diff   = v_rl - v_base
        if higher_better:
            winner = "RL  " if diff > 0 else "Base"
            wins_rl += (diff > 0); wins_base += (diff <= 0)
        else:
            winner = "RL  " if diff < 0 else "Base"
            wins_rl += (diff < 0); wins_base += (diff >= 0)
        sign = "+" if diff > 0 else ""
        print(f"  {name:<20}  RL={v_rl:8.1f}  Base={v_base:8.1f}  d={sign}{diff:.1f}  [{winner}]")
    print(f"\n  Score:  RL={wins_rl}   Heuristic={wins_base}")
    print("=" * 54 + "\n")


# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not MODEL_PATH.exists():
        print(f"ERROR: {MODEL_PATH} not found. Run train.py first.")
        sys.exit(1)

    print("Loading RL model ...")
    rl_model = MaskablePPO.load(str(MODEL_PATH))

    print("Running RL evaluation ...")
    res_rl = run_rl(rl_model)

    print("Running all-heuristic baseline ...")
    res_base = run_heuristic()

    plot_eval(res_rl, res_base, OUT_DIR / "eval_results.png")
    _summarise(res_rl,   "RL firm in market")
    _summarise(res_base, "All-heuristic baseline")
    _verdict(res_rl, res_base)
