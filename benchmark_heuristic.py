#!/usr/bin/env python
# benchmark_heuristic.py
#
# Pure Rule-Based Benchmark  +  Comparison vs RL Models
#
# Runs the original min_wage_model.py (all heuristic firms, no AI) across
# 3 market sizes x 100 random seeds, then compares against RL benchmark results.
#
# Output:
#   benchmark/heuristic/          Pure rule-based market charts
#     profit.png                  Avg firm profit over time
#     employment.png              Employment rate over time
#     wages.png                   Avg wages over time
#     workers.png                 Avg workforce per firm over time
#     scorecard.png               Summary bar chart
#
#   benchmark/comparison/         Head-to-head: Heuristic vs all RL models
#     profit_overview.png         Profit per market size  (all models on one chart)
#     employment_overview.png     Employment per market size
#     wages_overview.png          Wages per market size
#     scorecard_overview.png      Grouped bar chart - the big picture
#
# Usage:
#   1. Run compare_all.py first  (creates benchmark/.rl_cache.pkl)
#   2. python benchmark_heuristic.py

import sys
import contextlib
import io
import pickle
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Force UTF-8 stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT  = Path(__file__).parent.resolve()
BENCH = ROOT / "benchmark"
OUT_H = BENCH / "heuristic"
OUT_C = BENCH / "comparison"
OUT_H.mkdir(parents=True, exist_ok=True)
OUT_C.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────
# Configuration  (must match compare_all.py so charts are comparable)
# ─────────────────────────────────────────────────────────────────────

N_STEPS = 360
N_SEEDS = 100

CONFIGS = [
    {"label": "Small Market",  "subtitle": "50 workers  6 firms",  "tag": "small",  "n_workers": 50,  "n_firms": 6},
    {"label": "Medium Market", "subtitle": "100 workers  10 firms", "tag": "medium", "n_workers": 100, "n_firms": 10},
    {"label": "Large Market",  "subtitle": "150 workers  14 firms", "tag": "large",  "n_workers": 150, "n_firms": 14},
]

# ─────────────────────────────────────────────────────────────────────
# Colour palette  (dark theme, consistent with compare_all.py)
# ─────────────────────────────────────────────────────────────────────

BG    = "#0f0f1a"
PANEL = "#16213e"
GRID  = "#1e2a4a"
TEXT  = "#d0d0e8"
DIM   = "#888899"
WHITE = "#ffffff"

COL = {
    "heuristic":   "#aaaaaa",   # grey  — pure rule-based baseline
    "solo":        "#4fc3f7",   # blue  — solo RL
    "cooperative": "#66bb6a",   # green — cooperative RL
    "competitive": "#ef5350",   # red   — competitive RL
    "h_in_mixed":  "#ffb74d",   # orange — heuristic firms inside RL markets
}

LABEL = {
    "heuristic":   "Rule-Based Only",
    "solo":        "Solo AI",
    "cooperative": "Cooperative AI",
    "competitive": "Competitive AI",
    "h_in_mixed":  "Rule-Based (vs AI)",
}

# ─────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────

def _ax(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(WHITE)
    ax.grid(color=GRID, alpha=0.3, lw=0.5)


def _band(ax, x, mean, std, color, label, lw=2.0, alpha=0.15, ls="-"):
    ax.plot(x, mean, color=color, lw=lw, label=label, ls=ls)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=alpha)


def _note(ax, text, color=TEXT, size=8.5):
    ax.text(0.03, 0.97, text,
            transform=ax.transAxes, va="top", ha="left",
            color=color, fontsize=size,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=BG,
                      edgecolor=GRID, alpha=0.88))


def _save(fig, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"    Saved: {path.relative_to(ROOT)}")


# ─────────────────────────────────────────────────────────────────────
# Heuristic episode collector  (no policy — just run the model)
# ─────────────────────────────────────────────────────────────────────

def _run_heuristic_episode(n_workers, n_firms, seed):
    """
    Run one full episode of the original rule-based model.
    Suppresses the model's print statements to keep output clean.
    Returns dict metric -> np.array(N_STEPS).
    """
    sys.path.insert(0, str(ROOT))
    from min_wage_model import LaborMarketModel

    buf = {k: [] for k in ["profit", "employ_pct", "mkt_wage", "workers", "active_n", "avg_worker_utility"]}

    with contextlib.redirect_stdout(io.StringIO()):
        model = LaborMarketModel(N_workers=n_workers, N_firms=n_firms,
                                 min_wage=7700, seed=seed)
        for _ in range(N_STEPS):
            model.step()
            firms  = model.firms
            n_act  = len(firms)
            buf["profit"].append(
                float(np.mean([f.profit for f in firms])) if firms else 0.0)
            buf["employ_pct"].append(model.compute_employment_rate() * 100.0)
            buf["mkt_wage"].append(model.get_avg_firm_wage())
            buf["workers"].append(
                float(np.mean([len(f.current_workers) for f in firms])) if firms else 0.0)
            buf["active_n"].append(n_act)

            utils = []
            for w in model.workers:
                if w.employed:
                    utils.append(w.utility_if_work(w.monthly_wage))
                else:
                    utils.append(w.utility_if_not_work())
            buf["avg_worker_utility"].append(float(np.mean(utils)) if utils else 0.0)

    return {k: np.array(v[:N_STEPS]) for k, v in buf.items()}


def _aggregate(episodes):
    keys    = list(episodes[0].keys())
    stacked = {k: np.vstack([e[k] for e in episodes]) for k in keys}
    mean_ts = {k: stacked[k].mean(axis=0) for k in keys}
    std_ts  = {k: stacked[k].std(axis=0)  for k in keys}
    scalars = {
        k: {
            "mean": float(mean_ts[k].mean()),
            "std":  float(np.array([e[k].mean() for e in episodes]).std()),
        }
        for k in keys
    }
    return mean_ts, std_ts, scalars


def run_heuristic():
    results = {}
    for cfg in CONFIGS:
        print(f"  [heuristic] {cfg['label']}  ({N_SEEDS} seeds) ...", flush=True)
        episodes = []
        for s in range(N_SEEDS):
            episodes.append(_run_heuristic_episode(cfg["n_workers"], cfg["n_firms"], seed=s))
            if (s + 1) % 25 == 0:
                print(f"    {s+1}/{N_SEEDS} done", end="\r", flush=True)
        print(f"    {N_SEEDS}/{N_SEEDS} done  [done]")
        results[cfg["tag"]] = _aggregate(episodes)
    return results


# ─────────────────────────────────────────────────────────────────────
# Heuristic-only charts  (benchmark/heuristic/)
# ─────────────────────────────────────────────────────────────────────

def _3panel_chart(h_res, metric_key, ylabel, title, subtitle, outname, color=COL["heuristic"]):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), facecolor=BG)
    x = np.arange(1, N_STEPS + 1)
    for ax, cfg in zip(axes, CONFIGS):
        tag = cfg["tag"]
        if tag not in h_res:
            continue
        mean, std, sc = h_res[tag]
        _ax(ax)
        _band(ax, x, mean[metric_key], std[metric_key], color, "Rule-Based", lw=2.2)
        ax.set_title(f"{cfg['label']}\n{cfg['subtitle']}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Month", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel(ylabel, fontsize=9)
        avg = sc[metric_key]["mean"]
        _note(ax, f"Average: {avg:,.1f}")
    fig.suptitle(
        f"RULE-BASED ONLY  -  {title}\n{subtitle}",
        color=WHITE, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout(pad=1.2)
    _save(fig, OUT_H, outname)


def chart_heuristic_profit(h_res):
    _3panel_chart(h_res, "profit", "Monthly Profit (THB)",
                  "Monthly Profit per Firm",
                  f"All firms follow rule-based wage strategy  -  mean across {N_SEEDS} simulations",
                  "profit.png")


def chart_heuristic_employment(h_res):
    _3panel_chart(h_res, "employ_pct", "Workers Employed (%)",
                  "Employment Rate: What Percentage of Workers Have Jobs?",
                  "Higher = more workers employed in the market",
                  "employment.png")


def chart_heuristic_wages(h_res):
    _3panel_chart(h_res, "mkt_wage", "Monthly Wage (THB)",
                  "Wages: What Are Rule-Based Firms Paying Workers?",
                  "Wages adjust annually based on profit and vacancy pressure",
                  "wages.png")


def chart_heuristic_workers(h_res):
    _3panel_chart(h_res, "workers", "Avg Employees per Firm",
                  "Workforce Size: How Many Employees Does Each Firm Hire?",
                  "Reflects how aggressively firms post vacancies",
                  "workers.png")


def chart_heuristic_scorecard(h_res):
    """Summary bar chart for heuristic-only results across market sizes."""
    metrics = [
        ("Monthly Profit (THB)", "profit"),
        ("Employment Rate (%)",  "employ_pct"),
        ("Monthly Wage (THB)",   "mkt_wage"),
        ("Employees / Firm",     "workers"),
        ("Active Firms",         "active_n"),
    ]
    tags   = [cfg["tag"] for cfg in CONFIGS if cfg["tag"] in h_res]
    labels = [cfg["label"] for cfg in CONFIGS if cfg["tag"] in h_res]
    x      = np.arange(len(tags))

    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 5.5), facecolor=BG)
    for ax, (title, key) in zip(axes, metrics):
        _ax(ax)
        vals = [h_res[t][2][key]["mean"] for t in tags]
        errs = [h_res[t][2][key]["std"]  for t in tags]
        bars = ax.bar(x, vals, 0.55, color=COL["heuristic"],
                      edgecolor=GRID, lw=0.5,
                      yerr=errs, error_kw=dict(ecolor=DIM, capsize=3, lw=1.1))
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(errs) * 0.05,
                    f"{val:,.0f}", ha="center", va="bottom",
                    color=TEXT, fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, color=TEXT, fontsize=8)
        ax.set_title(title, color=WHITE, fontsize=9, fontweight="bold")

    fig.suptitle(
        f"RULE-BASED ONLY  -  Performance Scorecard\n"
        f"Averaged over {N_SEEDS} simulations x {N_STEPS} months  -  no AI firms present",
        color=WHITE, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout(pad=1.4)
    _save(fig, OUT_H, "scorecard.png")


# ─────────────────────────────────────────────────────────────────────
# Comparison charts  (benchmark/comparison/)
# Shows all models in one chart so the impact of AI is clear
# ─────────────────────────────────────────────────────────────────────

def chart_profit_overview(h_res, rl_cache):
    """
    3 panels (one per market size).
    Each panel: Rule-Based Only vs AI firm profits for all RL models.
    Answers: "Does AI help or hurt profitability?"
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), facecolor=BG)
    x = np.arange(1, N_STEPS + 1)

    for ax, cfg in zip(axes, CONFIGS):
        tag = cfg["tag"]
        _ax(ax)

        # Heuristic baseline (all firms rule-based)
        if tag in h_res:
            hm, hs, hsc = h_res[tag]
            _band(ax, x, hm["profit"], hs["profit"],
                  COL["heuristic"], LABEL["heuristic"], lw=2.0)

        # Each RL model — show AI firm profit
        for key in ("solo", "cooperative", "competitive"):
            r = rl_cache.get(key)
            if r and tag in r:
                rm, rs, rsc = r[tag]
                _band(ax, x, rm["rl_profit"], rs["rl_profit"],
                      COL[key], LABEL[key], lw=1.8)

        ax.axhline(0, color=DIM, lw=0.8, ls="--", alpha=0.5)
        ax.set_title(f"{cfg['label']}\n{cfg['subtitle']}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Month", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("Monthly Profit per Firm (THB)", fontsize=9)
        ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    fig.suptitle(
        "PROFIT COMPARISON  -  Rule-Based Only vs AI Firms Across All Models\n"
        f"Shaded area = variability across {N_SEEDS} simulations",
        color=WHITE, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout(pad=1.2)
    _save(fig, OUT_C, "profit_overview.png")


def chart_employment_overview(h_res, rl_cache):
    """Employment rate: pure heuristic market vs markets with AI."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), facecolor=BG)
    x = np.arange(1, N_STEPS + 1)

    for ax, cfg in zip(axes, CONFIGS):
        tag = cfg["tag"]
        _ax(ax)

        if tag in h_res:
            hm, hs, _ = h_res[tag]
            _band(ax, x, hm["employ_pct"], hs["employ_pct"],
                  COL["heuristic"], LABEL["heuristic"], lw=2.0)

        for key in ("solo", "cooperative", "competitive"):
            r = rl_cache.get(key)
            if r and tag in r:
                rm, rs, _ = r[tag]
                _band(ax, x, rm["employ_pct"], rs["employ_pct"],
                      COL[key], LABEL[key], lw=1.8)

        ax.set_ylim(0, 105)
        ax.set_title(f"{cfg['label']}\n{cfg['subtitle']}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Month", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("Workers Employed (%)", fontsize=9)
        ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    fig.suptitle(
        "EMPLOYMENT COMPARISON  -  Does AI Change How Many Workers Get Jobs?\n"
        "Higher = more workers employed in the market",
        color=WHITE, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout(pad=1.2)
    _save(fig, OUT_C, "employment_overview.png")


def chart_wages_overview(h_res, rl_cache):
    """Market wages: pure heuristic vs markets with AI firms."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), facecolor=BG)
    x = np.arange(1, N_STEPS + 1)

    for ax, cfg in zip(axes, CONFIGS):
        tag = cfg["tag"]
        _ax(ax)

        if tag in h_res:
            hm, hs, _ = h_res[tag]
            _band(ax, x, hm["mkt_wage"], hs["mkt_wage"],
                  COL["heuristic"], LABEL["heuristic"], lw=2.0)

        for key in ("solo", "cooperative", "competitive"):
            r = rl_cache.get(key)
            if r and tag in r:
                rm, rs, _ = r[tag]
                # Show market wage in mixed-AI market
                _band(ax, x, rm["mkt_wage"], rs["mkt_wage"],
                      COL[key], LABEL[key], lw=1.8)

        ax.set_title(f"{cfg['label']}\n{cfg['subtitle']}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Month", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("Market Average Wage (THB)", fontsize=9)
        ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    fig.suptitle(
        "WAGE COMPARISON  -  How Does AI Presence Affect Worker Wages?\n"
        "Shows the market-wide average wage in each scenario",
        color=WHITE, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout(pad=1.2)
    _save(fig, OUT_C, "wages_overview.png")


def chart_utility_overview(h_res, rl_cache):
    """Market-wide worker utility: pure heuristic vs markets with AI firms."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), facecolor=BG)
    x = np.arange(1, N_STEPS + 1)

    for ax, cfg in zip(axes, CONFIGS):
        tag = cfg["tag"]
        _ax(ax)

        if tag in h_res:
            hm, hs, _ = h_res[tag]
            _band(ax, x, hm["avg_worker_utility"], hs["avg_worker_utility"],
                  COL["heuristic"], LABEL["heuristic"], lw=2.0)

        for key in ("solo", "cooperative", "competitive"):
            r = rl_cache.get(key)
            if r and tag in r:
                rm, rs, _ = r[tag]
                _band(ax, x, rm["avg_worker_utility"], rs["avg_worker_utility"],
                      COL[key], LABEL[key], lw=1.8)

        ax.set_title(f"{cfg['label']}\n{cfg['subtitle']}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Month", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("Avg Worker Utility (Cobb-Douglas)", fontsize=9)
        ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    fig.suptitle(
        "WORKER UTILITY COMPARISON  -  How Does AI Presence Affect Worker Wellbeing?\n"
        "Shows market-wide average Cobb-Douglas utility (employed + unemployed workers)",
        color=WHITE, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout(pad=1.2)
    _save(fig, OUT_C, "utility_overview.png")


def chart_scorecard_overview(h_res, rl_cache):
    """
    The big-picture comparison: grouped bar chart.
    For each metric, 4 bars side by side: Heuristic / Solo AI / Coop AI / Comp AI.
    Averaged across all market sizes for clarity.
    """
    metrics = [
        ("Avg Profit / Firm\n(THB/month)",    "profit",      "rl_profit"),
        ("Employment Rate\n(%)",               "employ_pct",  "employ_pct"),
        ("Market Wage\n(THB/month)",           "mkt_wage",    "mkt_wage"),
        ("Avg Workers\nper Firm",              "workers",     "rl_workers"),
    ]

    scenarios = [
        ("heuristic",   LABEL["heuristic"],   COL["heuristic"]),
        ("solo",        LABEL["solo"],         COL["solo"]),
        ("cooperative", LABEL["cooperative"],  COL["cooperative"]),
        ("competitive", LABEL["competitive"],  COL["competitive"]),
    ]

    n_metrics  = len(metrics)
    n_scenario = len(scenarios)
    w          = 0.18
    x          = np.arange(n_metrics)

    fig, ax = plt.subplots(figsize=(16, 6), facecolor=BG)
    _ax(ax)

    for i, (key, label, color) in enumerate(scenarios):
        vals, errs = [], []
        for _, h_key, rl_key in metrics:
            # Average across all 3 market sizes
            if key == "heuristic":
                v_list = [h_res[t][2][h_key]["mean"] for t in ("small", "medium", "large") if t in h_res]
                e_list = [h_res[t][2][h_key]["std"]  for t in ("small", "medium", "large") if t in h_res]
            else:
                r = rl_cache.get(key)
                if r:
                    v_list = [r[t][2][rl_key]["mean"] for t in ("small", "medium", "large") if t in r]
                    e_list = [r[t][2][rl_key]["std"]  for t in ("small", "medium", "large") if t in r]
                else:
                    v_list, e_list = [0], [0]
            vals.append(float(np.mean(v_list)) if v_list else 0.0)
            errs.append(float(np.mean(e_list)) if e_list else 0.0)

        offset = (i - n_scenario / 2 + 0.5) * w
        bars = ax.bar(x + offset, vals, w,
                      color=color, edgecolor=GRID, lw=0.5, label=label,
                      yerr=errs, error_kw=dict(ecolor=DIM, capsize=2, lw=1.0))
        for bar, val in zip(bars, vals):
            if abs(val) > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(errs) * 0.02,
                        f"{val:,.0f}", ha="center", va="bottom",
                        color=color, fontsize=7.5, fontweight="bold",
                        rotation=45)

    ax.axhline(0, color=DIM, lw=0.8, ls="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metrics], color=TEXT, fontsize=10)
    ax.legend(fontsize=10, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT,
              loc="upper right")
    ax.set_title(
        "Overall Scorecard: Rule-Based Only vs AI Models\n"
        f"Averaged across all 3 market sizes and {N_SEEDS} simulations each",
        color=WHITE, fontsize=13, fontweight="bold")

    plt.tight_layout(pad=1.4)
    _save(fig, OUT_C, "scorecard_overview.png")


def chart_impact_of_ai(h_res, rl_cache):
    """
    Shows the IMPACT of introducing AI firms:
    - How much more does an AI firm earn vs a pure rule-based firm?
    - How does market employment change?
    - How do wages change for workers?
    Shown as % difference from pure heuristic baseline.
    """
    tags = [cfg["tag"] for cfg in CONFIGS]
    short = {"small": "Small", "medium": "Medium", "large": "Large"}

    metrics = [
        ("Firm Profit",      "profit",              "rl_profit"),
        ("Employment Rate",  "employ_pct",          "employ_pct"),
        ("Market Wage",      "mkt_wage",            "mkt_wage"),
        ("Worker Utility",   "avg_worker_utility",  "avg_worker_utility"),
    ]

    rl_keys = [
        ("solo",        LABEL["solo"],        COL["solo"]),
        ("cooperative", LABEL["cooperative"], COL["cooperative"]),
        ("competitive", LABEL["competitive"], COL["competitive"]),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(24, 5.5), facecolor=BG)

    for ax, (m_label, h_key, rl_key) in zip(axes, metrics):
        _ax(ax)
        x = np.arange(len(tags))
        w = 0.25

        for i, (key, label, color) in enumerate(rl_keys):
            r = rl_cache.get(key)
            pct_vals = []
            for tag in tags:
                h_base = h_res[tag][2][h_key]["mean"] if tag in h_res else None
                ai_val = r[tag][2][rl_key]["mean"] if (r and tag in r) else None
                if h_base and ai_val and abs(h_base) > 1:
                    pct_vals.append((ai_val - h_base) / abs(h_base) * 100)
                else:
                    pct_vals.append(0.0)

            offset = (i - 1) * w
            bars = ax.bar(x + offset, pct_vals, w,
                          color=color, edgecolor=GRID, lw=0.5, label=label)
            for bar, val in zip(bars, pct_vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (0.5 if val >= 0 else -2.5),
                        f"{val:+.0f}%", ha="center",
                        va="bottom" if val >= 0 else "top",
                        color=color, fontsize=8, fontweight="bold")

        ax.axhline(0, color=DIM, lw=1.2, ls="--", alpha=0.7, label="Heuristic baseline (0%)")
        ax.set_xticks(x)
        ax.set_xticklabels([short[t] for t in tags], color=TEXT, fontsize=9)
        ax.set_title(m_label, color=WHITE, fontsize=11, fontweight="bold")
        ax.set_ylabel("Change vs Pure Rule-Based (%)", fontsize=9)
        ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    fig.suptitle(
        "IMPACT OF AI FIRMS  -  % Change Compared to a Pure Rule-Based Market\n"
        "Positive = AI improves the metric vs a market with no AI firms",
        color=WHITE, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout(pad=1.2)
    _save(fig, OUT_C, "impact_of_ai.png")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

print(f"\n{'='*68}")
print("  Rule-Based Benchmark  +  Comparison vs RL Models")
print(f"  {N_SEEDS} seeds  x  {N_STEPS} months  x  {len(CONFIGS)} market sizes")
print(f"{'='*68}\n")

# Step 1: Run pure heuristic benchmark
print("[1/2] Running pure rule-based benchmark ...")
h_res = run_heuristic()

print("\n  Generating heuristic charts ...")
chart_heuristic_profit(h_res)
chart_heuristic_employment(h_res)
chart_heuristic_wages(h_res)
chart_heuristic_workers(h_res)
chart_heuristic_scorecard(h_res)

# Step 2: Load RL cache and generate comparison
print("\n[2/2] Loading RL benchmark results ...")
cache_path = BENCH / ".rl_cache.pkl"
if not cache_path.exists():
    print("  WARNING: benchmark/.rl_cache.pkl not found.")
    print("  Run 'python compare_all.py' first to generate RL results.")
    print("  Skipping comparison charts.\n")
    rl_cache = {}
else:
    with open(cache_path, "rb") as f:
        rl_cache = pickle.load(f)
    found = [k for k in ("solo", "cooperative", "competitive") if k in rl_cache and rl_cache[k]]
    print(f"  Loaded: {', '.join(found)}")

    print("\n  Generating comparison charts ...")
    chart_profit_overview(h_res, rl_cache)
    chart_employment_overview(h_res, rl_cache)
    chart_wages_overview(h_res, rl_cache)
    chart_utility_overview(h_res, rl_cache)
    chart_scorecard_overview(h_res, rl_cache)
    chart_impact_of_ai(h_res, rl_cache)

# Print summary table
print(f"\n{'='*72}")
print(f"  {'RULE-BASED ONLY SUMMARY':^68}")
print(f"{'='*72}")
print(f"  {'Metric':<30}  {'Small':>10}  {'Medium':>10}  {'Large':>10}")
print(f"  {'-'*30}  {'-'*10}  {'-'*10}  {'-'*10}")
for label, key in [
    ("Avg profit / firm (THB)",  "profit"),
    ("Employment rate (%)",      "employ_pct"),
    ("Market wage (THB)",        "mkt_wage"),
    ("Avg workers / firm",       "workers"),
    ("Active firms",             "active_n"),
]:
    vals = [h_res.get(t, (None, None, {}))[2].get(key, {}).get("mean", float("nan"))
            for t in ("small", "medium", "large")]
    row = "  ".join(f"{v:>10,.1f}" if not (v != v) else f"{'N/A':>10}" for v in vals)
    print(f"  {label:<30}  {row}")

print(f"{'='*72}")
print(f"\n  Heuristic charts:   benchmark/heuristic/")
print(f"  Comparison charts:  benchmark/comparison/\n")
