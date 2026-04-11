#!/usr/bin/env python
# benchmark/analyse.py
#
# Statistical analysis of benchmark/results.csv produced by benchmark/run.py.
#
# For each sweep:
#   - Paired t-test: RL profit vs heuristic profit
#   - Win rate: fraction of seeds where RL firm > heuristic F0
#   - Cohen's d: effect size
#   - Box plots
#
# Run: python benchmark/analyse.py

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

OUT_DIR  = Path(__file__).parent
CSV_PATH = OUT_DIR / "results.csv"
PLOT_DIR = OUT_DIR / "plots"

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def cohens_d(a, b):
    """Paired Cohen's d (mean diff / pooled std)."""
    diff = np.asarray(a) - np.asarray(b)
    return diff.mean() / (diff.std(ddof=1) + 1e-12)


def analyse_sweep(df_sweep, label):
    """
    Compare RL firm profit vs heuristic baseline profit for one sweep.
    Returns a dict of stats and prints a summary.
    """
    valid = df_sweep.dropna(subset=["rl_final_profit", "base_final_profit"])
    n     = len(valid)
    if n < 2:
        print(f"  {label}: not enough valid rows ({n})")
        return None

    rl   = valid["rl_final_profit"].values
    base = valid["base_final_profit"].values

    t_stat, p_val = stats.ttest_rel(rl, base)
    d             = cohens_d(rl, base)
    win_rate      = (rl > base).mean()

    # Also compare RL profit vs its OWN market average (same run) —
    # this is the cleaner within-run comparison used in eval.py
    rl_vs_mkt_win = (valid["rl_final_profit"] > valid["rl_market_avg_profit"]).mean()

    result = dict(
        sweep=label, n=n,
        rl_mean=rl.mean(),      rl_std=rl.std(),
        base_mean=base.mean(),  base_std=base.std(),
        mean_diff=rl.mean()-base.mean(),
        t_stat=t_stat, p_val=p_val,
        cohens_d=d,
        win_rate_vs_base=win_rate,
        win_rate_vs_own_mkt=rl_vs_mkt_win,
    )

    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    verdict = "RL BETTER" if (rl.mean() > base.mean() and p_val < 0.05) else \
              "HEURISTIC BETTER" if (rl.mean() < base.mean() and p_val < 0.05) else "no sig diff"

    print(f"\n{'-'*60}")
    print(f"  {label}  (n={n})")
    print(f"{'-'*60}")
    print(f"  RL   profit:  {rl.mean():>10,.0f}  +/- {rl.std():,.0f}")
    print(f"  Base profit:  {base.mean():>10,.0f}  +/- {base.std():,.0f}")
    print(f"  Delta mean:       {result['mean_diff']:>+10,.0f}")
    print(f"  t={t_stat:.3f}  p={p_val:.4f} {sig}  Cohen's d={d:.3f}")
    print(f"  Win rate vs base:    {win_rate:.1%}")
    print(f"  Win rate vs own mkt: {rl_vs_mkt_win:.1%}")
    print(f"  > {verdict}")

    return result


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------

def box_plot_sweep(df_sweep, label, ax):
    valid = df_sweep.dropna(subset=["rl_final_profit", "base_final_profit"])
    if valid.empty:
        return
    data   = [valid["rl_final_profit"].values, valid["base_final_profit"].values]
    bp     = ax.boxplot(data, patch_artist=True, widths=0.5,
                        medianprops=dict(color="white", lw=2))
    colors = ["#4fc3f7", "#ffb74d"]
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.8)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["RL", "Heuristic"], fontsize=9)
    ax.set_title(label, fontsize=9, fontweight="bold")
    ax.set_ylabel("Final Profit (THB)", fontsize=8)
    ax.axhline(0, color="gray", lw=0.7, linestyle="--")
    ax.grid(alpha=0.25, lw=0.5)


def plot_win_rates(results_list, path):
    labels    = [r["sweep"]              for r in results_list if r]
    wins_base = [r["win_rate_vs_base"]   for r in results_list if r]
    wins_mkt  = [r["win_rate_vs_own_mkt"] for r in results_list if r]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.9), 5))
    ax.bar(x - w/2, [v*100 for v in wins_base], w, label="vs heuristic F0", color="#4fc3f7", alpha=0.85)
    ax.bar(x + w/2, [v*100 for v in wins_mkt],  w, label="vs own mkt avg",  color="#ce93d8", alpha=0.85)
    ax.axhline(50, color="white", lw=1, linestyle="--", label="50% line")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("RL Win Rate (%)", fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_title("RL Firm Win Rate by Sweep", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, axis="y")
    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_profit_distribution(df, path):
    sweeps = df["sweep"].unique()
    n_sw   = len(sweeps)
    ncols  = min(4, n_sw)
    nrows  = (n_sw + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.5))
    axes_flat = np.array(axes).flatten()

    for i, sw in enumerate(sweeps):
        box_plot_sweep(df[df["sweep"] == sw], sw, axes_flat[i])

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Profit Distribution: RL vs Heuristic by Sweep", fontsize=12, fontweight="bold")
    plt.tight_layout(pad=1.0)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_param_sensitivity(df, path):
    """For each non-base sweep, show mean RL profit vs mean base profit as grouped bars."""
    param_df = df[df["sweep"] != "base"]
    if param_df.empty:
        return

    sweeps = param_df["sweep"].unique()
    rl_m   = [param_df[param_df["sweep"]==s]["rl_final_profit"].mean()   for s in sweeps]
    base_m = [param_df[param_df["sweep"]==s]["base_final_profit"].mean() for s in sweeps]

    x = np.arange(len(sweeps))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(sweeps) * 0.9), 5))
    ax.bar(x - w/2, rl_m,   w, label="RL",        color="#4fc3f7", alpha=0.85)
    ax.bar(x + w/2, base_m, w, label="Heuristic",  color="#ffb74d", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(sweeps, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Mean Final Profit (THB)", fontsize=10)
    ax.set_title("Parameter Sensitivity — Mean Final Profit", fontsize=12, fontweight="bold")
    ax.axhline(0, color="gray", lw=0.7)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, axis="y")
    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found. Run benchmark/run.py first.")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows from {CSV_PATH}")
    print(f"Sweeps: {sorted(df['sweep'].unique())}")

    PLOT_DIR.mkdir(exist_ok=True)
    results_list = []

    print("\n" + "="*60)
    print("  STATISTICAL ANALYSIS")
    print("="*60)

    for sweep in sorted(df["sweep"].unique()):
        res = analyse_sweep(df[df["sweep"] == sweep], sweep)
        results_list.append(res)

    # Summary CSV
    summary_df = pd.DataFrame([r for r in results_list if r])
    summary_df.to_csv(PLOT_DIR / "summary.csv", index=False, float_format="%.4f")
    print(f"\nSaved: {PLOT_DIR/'summary.csv'}")

    # Plots
    plot_win_rates(results_list, PLOT_DIR / "win_rates.png")
    plot_profit_distribution(df, PLOT_DIR / "profit_distributions.png")
    plot_param_sensitivity(df, PLOT_DIR / "param_sensitivity.png")

    # Overall verdict
    base_res = next((r for r in results_list if r and r["sweep"] == "base"), None)
    if base_res:
        print("\n" + "="*60)
        print("  OVERALL VERDICT (base sweep, 100 seeds)")
        print("="*60)
        verdict = (
            "RL IS SIGNIFICANTLY BETTER" if base_res["p_val"] < 0.05 and base_res["mean_diff"] > 0 else
            "HEURISTIC IS SIGNIFICANTLY BETTER" if base_res["p_val"] < 0.05 else
            "NO SIGNIFICANT DIFFERENCE"
        )
        print(f"  {verdict}")
        print(f"  p={base_res['p_val']:.4f}  Cohen's d={base_res['cohens_d']:.3f}"
              f"  win_rate={base_res['win_rate_vs_base']:.1%}")
        print("="*60 + "\n")
