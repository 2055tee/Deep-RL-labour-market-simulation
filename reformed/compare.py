#!/usr/bin/env python
# reformed/compare.py
#
# Compares two variants:
#   A — Options 5 + 3 + 4  (wage-gap probability ON)
#   B — Options 5 + 3      (wage-gap probability OFF)
#
# Runs N_SEEDS seeds each, plots mean ± std across 5 metrics,
# then prints a text verdict.
#
# Run:  python reformed/compare.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import LaborMarketModel

# ── Configuration ────────────────────────────────────────────────────
N_STEPS = 120
SEEDS   = [42, 123, 456, 789, 1000, 7, 13, 99, 2025, 314]

OUT_DIR = Path(__file__).parent
# ─────────────────────────────────────────────────────────────────────

def _gini(values):
    """Gini coefficient of a list (0 = equal, 1 = maximally unequal)."""
    w = np.array(values, dtype=float)
    if w.sum() == 0 or len(w) < 2:
        return 0.0
    w = np.sort(w)
    n = len(w)
    return (2 * np.dot(np.arange(1, n + 1), w) - (n + 1) * w.sum()) / (n * w.sum())


def run(use_wage_gap_prob: bool, seeds=SEEDS, n_steps=N_STEPS) -> dict:
    """Run all seeds for one variant; return dict of (n_seeds × n_steps) arrays."""
    emp_hist    = []
    profit_hist = []
    wage_hist   = []
    firms_hist  = []
    gini_hist   = []
    turnover_hist = []

    for seed in seeds:
        m = LaborMarketModel(N_workers=100, N_firms=10,
                             use_wage_gap_prob=use_wage_gap_prob, seed=seed)
        emp_row = []; profit_row = []; wage_row = []
        firms_row = []; gini_row = []; turn_row = []

        for _ in range(n_steps):
            m.step()
            active  = m.active_firms()
            employed = sum(1 for w in m.workers if w.employed)

            emp_row.append(100.0 * employed / len(m.workers))
            profit_row.append(float(np.mean([f.profit        for f in active])) if active else 0.0)
            wages = [f.monthly_wage for f in active] if active else [0]
            wage_row.append(float(np.mean(wages)))
            firms_row.append(len(active))
            gini_row.append(_gini(wages))
            # turnover proxy: fraction of workers who changed employer this step
            turn_row.append(float(np.mean([f.quits_last_month for f in active])) if active else 0.0)

        emp_hist.append(emp_row);    profit_hist.append(profit_row)
        wage_hist.append(wage_row);  firms_hist.append(firms_row)
        gini_hist.append(gini_row);  turnover_hist.append(turn_row)

    return dict(
        employment = np.array(emp_hist),
        avg_profit = np.array(profit_hist),
        avg_wage   = np.array(wage_hist),
        n_firms    = np.array(firms_hist),
        wage_gini  = np.array(gini_hist),
        turnover   = np.array(turnover_hist),
    )


def plot_comparison(res_a: dict, res_b: dict, path: Path):
    steps = np.arange(1, N_STEPS + 1)

    metrics = [
        ("employment", "Employment Rate (%)",        "%"),
        ("avg_profit",  "Avg Firm Profit (THB)",      "THB"),
        ("avg_wage",    "Avg Market Wage (THB)",       "THB"),
        ("n_firms",     "Active Firms",               ""),
        ("wage_gini",   "Wage Gini  (0=equal)",       ""),
        ("turnover",    "Avg Quits per Firm / month",  ""),
    ]

    COL_A = "#4fc3f7"   # blue  — 5+3+4
    COL_B = "#ffb74d"   # amber — 5+3

    fig, axes = plt.subplots(len(metrics), 1, figsize=(11, 3.8 * len(metrics)))
    fig.suptitle("Options 5+3+4  vs  Options 5+3\n"
                 "(mean ± 1 std across 10 seeds, 120 months)",
                 fontsize=13, fontweight="bold", y=1.002)

    for ax, (key, title, unit) in zip(axes, metrics):
        for res, label, col in [
            (res_a, "5+3+4  (wage-gap prob ON)",  COL_A),
            (res_b, "5+3    (wage-gap prob OFF)", COL_B),
        ]:
            m = res[key].mean(axis=0)
            s = res[key].std(axis=0)
            ax.plot(steps, m, color=col, lw=2.2, label=label)
            ax.fill_between(steps, m - s, m + s, color=col, alpha=0.18)

        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Month", fontsize=8)
        ax.set_ylabel(unit if unit else title, fontsize=8)
        ax.legend(fontsize=8, framealpha=0.7)
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=8)

    plt.tight_layout(pad=0.5)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def _summarise(res: dict, label: str):
    print("\n" + "-"*54)
    print(f"  {label}")
    print("-"*54)
    header = f"  {'Metric':<22}  {'Final mean':>10}  {'Std':>7}  {'Trend':>10}"
    print(header)
    for key in ["employment", "avg_profit", "avg_wage", "n_firms", "wage_gini"]:
        final   = res[key][:, -1]
        early   = res[key][:, :12].mean(axis=1)
        trend   = final.mean() - early.mean()
        print(f"  {key:<22}  {final.mean():>10.1f}  {final.std():>7.1f}  {trend:>+10.1f}")


def _verdict(res_a: dict, res_b: dict):
    print("\n" + "="*54)
    print("  VERDICT")
    print("="*54)

    scorecard = [
        ("Employment %",    "employment", True),
        ("Avg Profit",      "avg_profit",  True),
        ("Firms Surviving", "n_firms",     True),
        ("Wage Gini",       "wage_gini",   False),
    ]

    wins_a = wins_b = 0
    for name, key, higher_better in scorecard:
        v_a = res_a[key][:, -1].mean()
        v_b = res_b[key][:, -1].mean()
        diff = v_a - v_b
        if higher_better:
            winner = "5+3+4" if diff > 0 else "5+3  "
            wins_a += (diff > 0); wins_b += (diff <= 0)
        else:
            winner = "5+3+4" if diff < 0 else "5+3  "
            wins_a += (diff < 0); wins_b += (diff >= 0)
        sign = "+" if diff > 0 else ""
        print(f"  {name:<22}  5+3+4={v_a:7.1f}  5+3={v_b:7.1f}  d={sign}{diff:.1f}  winner={winner}")

    print(f"\n  Score:  5+3+4 = {wins_a}   5+3 = {wins_b}")
    overall = "Options 5+3+4  (wage-gap prob ON)" if wins_a >= wins_b \
              else "Options 5+3   (wage-gap prob OFF)"
    print(f"  Overall winner: {overall}")
    print("="*54 + "\n")


# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Seeds: {SEEDS}")
    print(f"Steps per run: {N_STEPS}\n")

    print("Running Options 5+3+4  (wage-gap prob ON) ...")
    res_a = run(use_wage_gap_prob=True)

    print("Running Options 5+3    (wage-gap prob OFF) ...")
    res_b = run(use_wage_gap_prob=False)

    plot_comparison(res_a, res_b, OUT_DIR / "comparison.png")

    _summarise(res_a, "Options 5+3+4  (wage-gap prob ON)")
    _summarise(res_b, "Options 5+3    (wage-gap prob OFF)")
    _verdict(res_a, res_b)
