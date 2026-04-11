#!/usr/bin/env python
# viz_shortlong_compare.py
#
# Compares short-run (gamma=0.95, [64,64]) vs long-run (gamma=0.99, [256,256])
# trained models across all three scenarios (solo, cooperative, competitive).
#
# Runs each model over 5 seeds, then plots mean ± std for both on the same axes.
#
# Output folder: visualizations/shortlong/
#   solo_profit.png       coop_profit.png       comp_profit.png
#   solo_wage.png         coop_wage.png         comp_wage.png
#   solo_workers.png      coop_workers.png      comp_workers.png
#   final_profit_bar.png  -- bar chart of mean final-step profit across all 6 models

import sys
import os
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

SEEDS = [42, 7, 13, 99, 2025]
ROOT    = Path(__file__).parent.resolve()
VIZ_DIR = ROOT / "visualizations" / "shortlong"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ #
#  Style                                                              #
# ------------------------------------------------------------------ #

BG    = "#0f0f1a"
PANEL = "#16213e"
GRID  = "#1e2a4a"
TEXT  = "#d0d0e8"

SHORT_COLOR = "#ff8a65"   # warm orange  — short-run
LONG_COLOR  = "#4fc3f7"   # cool blue    — long-run


def styled_fig(w=16, h=7):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=11)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    return fig, ax


def save(fig, name):
    path = VIZ_DIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved: shortlong/{path.name}")


def wage_cycle_lines(ax, alpha=0.2):
    for x in range(0, 361, 12):
        ax.axvline(x, color=GRID, lw=0.7, alpha=alpha, zorder=1)


# ------------------------------------------------------------------ #
#  Episode runners — one per scenario                                 #
# ------------------------------------------------------------------ #

def run_solo(policy, seed):
    random.seed(seed); np.random.seed(seed)
    env = _solo_env()
    obs, _ = env.reset()
    profits, wages, workers = [], [], []
    for _ in range(360):
        mask = env.action_masks()
        act, _ = policy.predict(obs[np.newaxis], deterministic=True,
                                action_masks=mask[np.newaxis])
        obs, _, _, _, _ = env.step(int(act[0]))
        profits.append(float(env.rl_firm.profit))
        wages.append(int(env.rl_firm.monthly_wage))
        workers.append(len(env.rl_firm.current_workers))
    return profits, wages, workers


def run_coop(policy, seed):
    random.seed(seed); np.random.seed(seed)
    env = _coop_env()
    obs, _ = env.reset()
    profits, wages, workers = [], [], []
    for _ in range(360 * env.n_rl_firms):
        mask = env.action_masks()
        act, _ = policy.predict(obs[np.newaxis], deterministic=True,
                                action_masks=mask[np.newaxis])
        obs, _, _, _, _ = env.step(int(act[0]))
        if env.current_idx == 0:
            profits.append(float(np.mean([f.profit for f in env.rl_firms])))
            wages.append(float(np.mean([f.monthly_wage for f in env.rl_firms])))
            workers.append(float(np.mean([len(f.current_workers) for f in env.rl_firms])))
    return profits, wages, workers


def run_comp(policy, seed):
    random.seed(seed); np.random.seed(seed)
    env = _comp_env()
    obs, _ = env.reset()
    profits, wages, workers = [], [], []
    for _ in range(360 * env.n_rl_firms):
        mask = env.action_masks()
        act, _ = policy.predict(obs[np.newaxis], deterministic=True,
                                action_masks=mask[np.newaxis])
        obs, _, _, _, _ = env.step(int(act[0]))
        if env.current_idx == 0:
            profits.append(float(np.mean([f.profit for f in env.rl_firms])))
            wages.append(float(np.mean([f.monthly_wage for f in env.rl_firms])))
            workers.append(float(np.mean([len(f.current_workers) for f in env.rl_firms])))
    return profits, wages, workers


# ------------------------------------------------------------------ #
#  Lazy env constructors (import after sys.path manipulation)         #
# ------------------------------------------------------------------ #

def _solo_env():
    return _SOLO_ENV_CLS()

def _coop_env():
    env = _COOP_ENV_CLS()
    env.n_rl_firms = len(env.rl_firms)
    return env

def _comp_env():
    env = _COMP_ENV_CLS()
    env.n_rl_firms = len(env.rl_firms)
    return env


# ------------------------------------------------------------------ #
#  Run all seeds for a scenario, return (n_seeds, 360) arrays        #
# ------------------------------------------------------------------ #

def collect(runner, policy):
    all_p, all_w, all_k = [], [], []
    for seed in SEEDS:
        p, w, k = runner(policy, seed)
        all_p.append(p)
        all_w.append(w)
        all_k.append(k)
    return np.array(all_p), np.array(all_w), np.array(all_k)


# ------------------------------------------------------------------ #
#  Plot helper: two models on same axes                               #
# ------------------------------------------------------------------ #

def plot_comparison(ax, short_vals, long_vals, ylabel, zero_line=False):
    steps = np.arange(short_vals.shape[1])
    wage_cycle_lines(ax)

    if zero_line:
        ax.axhline(0, color="#445566", lw=1.0, ls="--", zorder=2)

    for vals, color, label in [
        (short_vals, SHORT_COLOR, "Short-run  (γ=0.95, [64,64])"),
        (long_vals,  LONG_COLOR,  "Long-run   (γ=0.99, [256,256])"),
    ]:
        mean = vals.mean(axis=0)
        std  = vals.std(axis=0)
        ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.15)
        ax.plot(steps, mean, color=color, lw=2.2, label=label)

    ax.set_ylabel(ylabel, color=TEXT, fontsize=11)
    ax.set_xlim(0, steps[-1])
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=10)


def save_scenario_charts(tag, title_prefix, short_p, short_w, short_k,
                         long_p, long_w, long_k):
    steps_label = "Model Step  (vertical lines = annual wage-review points)"

    # Profit
    fig, ax = styled_fig()
    plot_comparison(ax, short_p, long_p, "Monthly Profit  (THB)", zero_line=True)
    ax.set_title(f"{title_prefix} — Profit: Short-run vs Long-run  (mean ± 1 std, 5 seeds)",
                 color="white", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(steps_label, color=TEXT, fontsize=11)
    save(fig, f"{tag}_profit.png")

    # Wage
    fig, ax = styled_fig(16, 6)
    plot_comparison(ax, short_w, long_w, "Monthly Wage  (THB)")
    ax.set_title(f"{title_prefix} — Wage: Short-run vs Long-run",
                 color="white", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Model Step", color=TEXT, fontsize=11)
    save(fig, f"{tag}_wage.png")

    # Workers
    fig, ax = styled_fig(16, 6)
    plot_comparison(ax, short_k, long_k, "Workers (mean across RL firms)")
    ax.set_title(f"{title_prefix} — Headcount: Short-run vs Long-run",
                 color="white", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Model Step", color=TEXT, fontsize=11)
    save(fig, f"{tag}_workers.png")


# ------------------------------------------------------------------ #
#  Final profit bar chart across all 6 models                        #
# ------------------------------------------------------------------ #

def chart_final_bar(results):
    """
    results: list of (label, color, mean_final_profit, std_final_profit)
    """
    fig, ax = plt.subplots(figsize=(14, 6), facecolor=BG)
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.tick_params(colors=TEXT, labelsize=10)

    labels = [r[0] for r in results]
    colors = [r[1] for r in results]
    means  = [r[2] for r in results]
    stds   = [r[3] for r in results]

    x = np.arange(len(labels))
    bars = ax.bar(x, means, color=colors, edgecolor=GRID, linewidth=0.5,
                  width=0.55, yerr=stds, capsize=5,
                  error_kw=dict(ecolor="white", elinewidth=1.5, capthick=1.5))
    ax.axhline(0, color="#445566", lw=1.0, ls="--")

    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2,
                (mean + std + abs(mean) * 0.03) if mean >= 0 else (mean - std - abs(mean)*0.03),
                f"{mean:,.0f}", ha="center",
                va="bottom" if mean >= 0 else "top",
                color="white", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=TEXT, fontsize=10)
    ax.set_ylabel("Mean Final-Step Profit  (THB)", color=TEXT, fontsize=11)
    ax.set_title("Final Profit Comparison — All 6 Models  (mean ± 1 std across 5 seeds)",
                 color="white", fontsize=14, fontweight="bold", pad=12)

    handles = [
        mpatches.Patch(color=SHORT_COLOR, label="Short-run  (γ=0.95)"),
        mpatches.Patch(color=LONG_COLOR,  label="Long-run   (γ=0.99)"),
    ]
    ax.legend(handles=handles, facecolor=PANEL, edgecolor=GRID,
              labelcolor=TEXT, fontsize=10)
    save(fig, "final_profit_bar.png")


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    from sb3_contrib import MaskablePPO

    # ── Solo ──────────────────────────────────────────────────────
    print("Loading solo models...")
    sys.path.insert(0, str(ROOT / "solo"))
    for mod in ["model_rl", "firm_env", "rl_vis"]:
        sys.modules.pop(mod, None)
    from firm_env import LaborMarketEnv as _SOLO_ENV_CLS  # noqa: F811

    solo_short = MaskablePPO.load(str(ROOT / "solo" / "solo_model"))
    solo_long  = MaskablePPO.load(str(ROOT / "solo" / "solo_model_longrun"))

    print("  Running solo short-run..."); solo_s_p, solo_s_w, solo_s_k = collect(run_solo, solo_short)
    print("  Running solo long-run...");  solo_l_p, solo_l_w, solo_l_k = collect(run_solo, solo_long)
    save_scenario_charts("solo", "Solo", solo_s_p, solo_s_w, solo_s_k,
                                         solo_l_p, solo_l_w, solo_l_k)

    # ── Cooperative ───────────────────────────────────────────────
    print("Loading cooperative models...")
    sys.path.insert(0, str(ROOT / "cooperative"))
    for mod in ["model_rl", "firm_env", "rl_vis"]:
        sys.modules.pop(mod, None)
    from firm_env import CoopFirmEnv as _COOP_ENV_CLS  # noqa: F811

    coop_short = MaskablePPO.load(str(ROOT / "cooperative" / "coop_model"))
    coop_long  = MaskablePPO.load(str(ROOT / "cooperative" / "coop_model_longrun"))

    print("  Running coop short-run..."); coop_s_p, coop_s_w, coop_s_k = collect(run_coop, coop_short)
    print("  Running coop long-run...");  coop_l_p, coop_l_w, coop_l_k = collect(run_coop, coop_long)
    save_scenario_charts("coop", "Cooperative", coop_s_p, coop_s_w, coop_s_k,
                                                 coop_l_p, coop_l_w, coop_l_k)

    # ── Competitive ───────────────────────────────────────────────
    print("Loading competitive models...")
    sys.path.insert(0, str(ROOT / "competitive"))
    for mod in ["model_rl", "firm_env", "rl_vis"]:
        sys.modules.pop(mod, None)
    from firm_env import CompFirmEnv as _COMP_ENV_CLS  # noqa: F811

    comp_short = MaskablePPO.load(str(ROOT / "competitive" / "comp_model"))
    comp_long  = MaskablePPO.load(str(ROOT / "competitive" / "comp_model_longrun"))

    print("  Running comp short-run..."); comp_s_p, comp_s_w, comp_s_k = collect(run_comp, comp_short)
    print("  Running comp long-run...");  comp_l_p, comp_l_w, comp_l_k = collect(run_comp, comp_long)
    save_scenario_charts("comp", "Competitive", comp_s_p, comp_s_w, comp_s_k,
                                                 comp_l_p, comp_l_w, comp_l_k)

    # ── Final profit bar ──────────────────────────────────────────
    print("Generating final profit bar chart...")
    results = [
        ("Solo\nShort",  SHORT_COLOR, solo_s_p[:, -1].mean(),  solo_s_p[:, -1].std()),
        ("Solo\nLong",   LONG_COLOR,  solo_l_p[:, -1].mean(),  solo_l_p[:, -1].std()),
        ("Coop\nShort",  SHORT_COLOR, coop_s_p[:, -1].mean(),  coop_s_p[:, -1].std()),
        ("Coop\nLong",   LONG_COLOR,  coop_l_p[:, -1].mean(),  coop_l_p[:, -1].std()),
        ("Comp\nShort",  SHORT_COLOR, comp_s_p[:, -1].mean(),  comp_s_p[:, -1].std()),
        ("Comp\nLong",   LONG_COLOR,  comp_l_p[:, -1].mean(),  comp_l_p[:, -1].std()),
    ]
    chart_final_bar(results)

    print("\nDone. All charts saved to: visualizations/shortlong/")
    print("  solo_profit.png    solo_wage.png    solo_workers.png")
    print("  coop_profit.png    coop_wage.png    coop_workers.png")
    print("  comp_profit.png    comp_wage.png    comp_workers.png")
    print("  final_profit_bar.png")
