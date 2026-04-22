#!/usr/bin/env python
# compare_all.py
#
# Benchmark: AI (Reinforcement Learning) firms vs Rule-Based (Heuristic) firms
#
# Tests each model across 3 market sizes, 100 random simulations each.
# Results are saved to benchmark/ organized by model type:
#
#   benchmark/solo/          1 AI firm competing against rule-based firms
#   benchmark/cooperative/   3 AI firms working together as a team
#   benchmark/competitive/   3 AI firms competing against each other
#
# Each folder contains:
#   profit.png       Monthly profit over time: AI vs rule-based firms
#   employment.png   % of workers employed in the market
#   wages.png        What firms are paying workers over time
#   workers.png      How many employees each firm type hires
#   scorecard.png    Overall summary: who wins, and by how much
#   wage_spread.png  [coop/comp only] how much AI firms differ in wage strategy
#
# Usage:  python compare_all.py

import sys
import random
import pickle
import numpy as np

# Force UTF-8 stdout so special characters don't crash on Windows cp874
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT  = Path(__file__).parent.resolve()
BENCH = ROOT / "benchmark"
BENCH.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

N_STEPS    = 360    # Steps per episode  (1 step = 1 month → 30 years)
N_SEEDS    = 100    # Random simulations per environment config
N_RL_FIRMS = 3      # AI firms in cooperative / competitive scenarios

# Three market sizes to evaluate generalisation
CONFIGS = [
    {
        "label":    "Small Market",
        "subtitle": "50 workers · 6 firms",
        "tag":      "small",
        "n_workers": 50,
        "n_firms":   6,
    },
    {
        "label":    "Medium Market",
        "subtitle": "100 workers · 10 firms",
        "tag":      "medium",
        "n_workers": 100,
        "n_firms":   10,
    },
    {
        "label":    "Large Market",
        "subtitle": "150 workers · 14 firms",
        "tag":      "large",
        "n_workers": 150,
        "n_firms":   14,
    },
]

# ─────────────────────────────────────────────────────────────────────
# Colour palette  (dark theme)
# ─────────────────────────────────────────────────────────────────────

BG    = "#0f0f1a"
PANEL = "#16213e"
GRID  = "#1e2a4a"
TEXT  = "#d0d0e8"
DIM   = "#888899"
WHITE = "#ffffff"

AI_COL = {
    "solo":        "#4fc3f7",   # blue
    "cooperative": "#66bb6a",   # green
    "competitive": "#ef5350",   # red
}
H_COL   = "#ffb74d"   # rule-based / heuristic — orange
MKT_COL = "#ce93d8"   # market average — purple

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


def _save(fig, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"    Saved: benchmark/{folder.name}/{name}")


def _note(ax, text, color=TEXT, size=9, loc="upper left"):
    """Annotation text box inside a panel."""
    x = 0.03 if "left" in loc else 0.97
    ha = "left" if "left" in loc else "right"
    y = 0.97 if "upper" in loc else 0.03
    va = "top" if "upper" in loc else "bottom"
    ax.text(x, y, text,
            transform=ax.transAxes, va=va, ha=ha,
            color=color, fontsize=size,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=BG,
                      edgecolor=GRID, alpha=0.88))


def _win_badge(ax, win_pct, ai_color):
    """Win-rate badge in the upper-right corner of a panel."""
    color = ai_color if win_pct >= 50 else H_COL
    ax.text(0.97, 0.97,
            f"AI wins {win_pct:.0f}% of months",
            transform=ax.transAxes, va="top", ha="right",
            color=color, fontsize=8.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor=BG,
                      edgecolor=color, alpha=0.92))


# ─────────────────────────────────────────────────────────────────────
# Namespace management
# loads model_rl / firm_env from the right subfolder without collisions
# ─────────────────────────────────────────────────────────────────────

def _load_ns(name):
    d = ROOT / name
    if str(d) in sys.path:
        sys.path.remove(str(d))
    sys.path.insert(0, str(d))
    for m in ["model_rl", "firm_env", "model"]:
        sys.modules.pop(m, None)
    return d


def _unload_ns(name):
    d = str(ROOT / name)
    if d in sys.path:
        sys.path.remove(d)
    for m in ["model_rl", "firm_env", "model"]:
        sys.modules.pop(m, None)


# ─────────────────────────────────────────────────────────────────────
# Episode collection
# ─────────────────────────────────────────────────────────────────────

METRICS = [
    "rl_profit", "h_profit",
    "employ_pct",
    "mkt_wage", "rl_wage", "h_wage",
    "rl_workers", "h_workers",
    "active_n",
    "rl_f0_profit", "rl_f1_profit", "rl_f2_profit",
    "rl_wage_spread",
    "ai_wins",   # 1.0 if AI profit > heuristic profit this step
    "avg_worker_utility",  # average utility across all workers (employed + unemployed)
]


def _collect(env, policy, is_multi, seed):
    """Run one full episode. Returns {metric: np.array(N_STEPS)}."""
    try:
        obs, _ = env.reset(seed=seed)
    except TypeError:
        obs, _ = env.reset()

    buf   = {k: [] for k in METRICS}
    total = N_STEPS * (N_RL_FIRMS if is_multi else 1)

    for _ in range(total):
        mask   = env.action_masks()
        act, _ = policy.predict(obs[np.newaxis], deterministic=True,
                                action_masks=mask[np.newaxis])
        obs, _, _, _, _ = env.step(int(act[0]))

        # For multi-firm envs, only record once all RL firms have acted
        if is_multi and env.current_idx != 0:
            continue

        model    = env.model
        rl_firms = env.rl_firms if is_multi else [env.rl_firm]
        all_f    = model.firms
        h_firms  = [f for f in all_f if f not in rl_firms]
        active   = [f for f in all_f if getattr(f, "active", True)]

        rl_p = float(np.mean([f.profit for f in rl_firms]))
        h_p  = float(np.mean([f.profit for f in h_firms])) if h_firms else 0.0

        buf["rl_profit"].append(rl_p)
        buf["h_profit"].append(h_p)
        buf["employ_pct"].append(
            100.0 * sum(1 for w in model.workers if w.employed)
            / max(len(model.workers), 1))
        buf["mkt_wage"].append(
            float(np.mean([f.monthly_wage for f in active])) if active else 0.0)
        buf["rl_wage"].append(
            float(np.mean([f.monthly_wage for f in rl_firms])))
        buf["h_wage"].append(
            float(np.mean([f.monthly_wage for f in h_firms])) if h_firms else 0.0)
        buf["rl_workers"].append(
            float(np.mean([len(f.current_workers) for f in rl_firms])))
        buf["h_workers"].append(
            float(np.mean([len(f.current_workers) for f in h_firms])) if h_firms else 0.0)
        buf["active_n"].append(len(active))

        rl_pf = [f.profit for f in rl_firms]
        buf["rl_f0_profit"].append(rl_pf[0] if len(rl_pf) > 0 else 0.0)
        buf["rl_f1_profit"].append(rl_pf[1] if len(rl_pf) > 1 else 0.0)
        buf["rl_f2_profit"].append(rl_pf[2] if len(rl_pf) > 2 else 0.0)

        rl_wages = [f.monthly_wage for f in rl_firms]
        buf["rl_wage_spread"].append(
            float(max(rl_wages) - min(rl_wages)) if len(rl_wages) > 1 else 0.0)
        buf["ai_wins"].append(1.0 if rl_p > h_p else 0.0)

        # Average worker utility across all workers (employed + unemployed)
        utils = []
        for w in model.workers:
            if w.employed:
                utils.append(w.utility_if_work(w.monthly_wage))
            else:
                utils.append(w.utility_if_not_work())
        buf["avg_worker_utility"].append(float(np.mean(utils)) if utils else 0.0)

    return {k: np.array(v[:N_STEPS]) for k, v in buf.items()}


def _aggregate(episodes):
    """Mean, std, and scalar summary across a list of episode dicts."""
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


# ─────────────────────────────────────────────────────────────────────
# Scenario runners
# ─────────────────────────────────────────────────────────────────────

def _run_scenario(scenario_key, model_file, env_class_name, is_multi, size_kwargs="lower"):
    """
    Load the model once, run all CONFIGS x N_SEEDS, unload.
    size_kwargs: "lower" -> n_workers/n_firms,  "upper" -> N_workers/N_firms
    Returns dict: tag -> (mean_ts, std_ts, scalars)  or  None if model missing.
    """
    d = _load_ns(scenario_key)
    from sb3_contrib import MaskablePPO
    import firm_env as fe

    if not (d / model_file).exists():
        print(f"  [{scenario_key}] Model not found - skipping.")
        _unload_ns(scenario_key)
        return None

    EnvClass = getattr(fe, env_class_name)
    policy   = MaskablePPO.load(str(d / model_file.replace(".zip", "")))
    results  = {}

    for cfg in CONFIGS:
        print(f"  [{scenario_key}] {cfg['label']}  ({N_SEEDS} seeds) ...", flush=True)
        episodes = []
        if size_kwargs == "upper":
            env = EnvClass(N_workers=cfg["n_workers"], N_firms=cfg["n_firms"])
        else:
            env = EnvClass(n_workers=cfg["n_workers"], n_firms=cfg["n_firms"])
        for s in range(N_SEEDS):
            episodes.append(_collect(env, policy, is_multi=is_multi, seed=s))
            if (s + 1) % 25 == 0:
                print(f"    {s+1}/{N_SEEDS} seeds done", end="\r", flush=True)
        print(f"    {N_SEEDS}/{N_SEEDS} seeds done  [done]")
        results[cfg["tag"]] = _aggregate(episodes)

    _unload_ns(scenario_key)
    return results


def run_solo():
    # Uses the reformed model and environment (stronger rules, snap action, market-quit)
    return _run_scenario("reformed", "reformed_model.zip", "ReformedFirmEnv",
                         is_multi=False, size_kwargs="upper")


def run_cooperative():
    return _run_scenario("cooperative", "coop_model_longrun.zip", "CoopFirmEnv", is_multi=True)


def run_competitive():
    return _run_scenario("competitive", "comp_model_longrun.zip", "CompFirmEnv", is_multi=True)


# ─────────────────────────────────────────────────────────────────────
# Chart: Monthly Profit
# ─────────────────────────────────────────────────────────────────────

def chart_profit(results, outdir, scenario_name, ai_color):
    """One panel per market size — AI vs rule-based monthly profit."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), facecolor=BG)
    x = np.arange(1, N_STEPS + 1)

    for ax, cfg in zip(axes, CONFIGS):
        tag = cfg["tag"]
        if tag not in results:
            continue
        mean, std, sc = results[tag]
        _ax(ax)

        _band(ax, x, mean["rl_profit"], std["rl_profit"], ai_color, "AI Strategy",    lw=2.2)
        _band(ax, x, mean["h_profit"],  std["h_profit"],  H_COL,    "Rule-Based",      lw=1.8)
        ax.axhline(0, color=DIM, lw=0.8, ls="--", alpha=0.5)

        ax.set_title(f"{cfg['label']}\n{cfg['subtitle']}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Month", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("Monthly Profit (THB)", fontsize=9)
        ax.legend(fontsize=9, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

        ai_avg = sc["rl_profit"]["mean"]
        h_avg  = sc["h_profit"]["mean"]
        diff   = ai_avg - h_avg
        pct    = diff / max(abs(h_avg), 1) * 100
        sign   = "+" if diff >= 0 else ""
        note   = (f"AI avg:       {ai_avg:>9,.0f} THB\n"
                  f"Rule-Based: {h_avg:>9,.0f} THB\n"
                  f"Difference:  {sign}{diff:,.0f} ({sign}{pct:.1f}%)")
        _note(ax, note, color=ai_color if diff >= 0 else H_COL)
        _win_badge(ax, sc["ai_wins"]["mean"] * 100, ai_color)

    fig.suptitle(
        f"{scenario_name.upper()}  ·  Monthly Profit: AI vs Rule-Based Firms\n"
        f"Mean ± 1 std across {N_SEEDS} simulations — shaded band shows variability",
        color=WHITE, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout(pad=1.2)
    _save(fig, outdir, "profit.png")


# ─────────────────────────────────────────────────────────────────────
# Chart: Employment Rate
# ─────────────────────────────────────────────────────────────────────

def chart_employment(results, outdir, scenario_name, ai_color):
    """Employment rate over time — how well is the market absorbing workers?"""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), facecolor=BG)
    x = np.arange(1, N_STEPS + 1)

    for ax, cfg in zip(axes, CONFIGS):
        tag = cfg["tag"]
        if tag not in results:
            continue
        mean, std, sc = results[tag]
        _ax(ax)

        _band(ax, x, mean["employ_pct"], std["employ_pct"], ai_color, "Employment Rate", lw=2.2)
        ax.set_ylim(0, 105)
        ax.axhline(100, color=DIM, lw=0.8, ls="--", alpha=0.4)

        ax.set_title(f"{cfg['label']}\n{cfg['subtitle']}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Month", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("Workers Employed (%)", fontsize=9)

        avg = sc["employ_pct"]["mean"]
        note = f"Average: {avg:.1f}% of workers\nhave jobs in this market"
        _note(ax, note, color=TEXT)

    fig.suptitle(
        f"{scenario_name.upper()}  ·  Employment Rate: What Percentage of Workers Have Jobs?\n"
        f"Higher = more workers employed — mean across {N_SEEDS} simulations",
        color=WHITE, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout(pad=1.2)
    _save(fig, outdir, "employment.png")


# ─────────────────────────────────────────────────────────────────────
# Chart: Wages
# ─────────────────────────────────────────────────────────────────────

def chart_wages(results, outdir, scenario_name, ai_color):
    """What are firms paying workers? AI vs rule-based vs market average."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), facecolor=BG)
    x = np.arange(1, N_STEPS + 1)

    for ax, cfg in zip(axes, CONFIGS):
        tag = cfg["tag"]
        if tag not in results:
            continue
        mean, std, sc = results[tag]
        _ax(ax)

        _band(ax, x, mean["rl_wage"],  std["rl_wage"],  ai_color,  "AI Firms",       lw=2.2)
        _band(ax, x, mean["h_wage"],   std["h_wage"],   H_COL,     "Rule-Based Firms", lw=1.8)
        _band(ax, x, mean["mkt_wage"], std["mkt_wage"], MKT_COL,   "Market Average",  lw=1.3, alpha=0.08)

        ax.set_title(f"{cfg['label']}\n{cfg['subtitle']}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Month", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("Monthly Wage (THB)", fontsize=9)
        ax.legend(fontsize=8.5, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

        ai_w = sc["rl_wage"]["mean"]
        h_w  = sc["h_wage"]["mean"]
        gap  = ai_w - h_w
        sign = "+" if gap >= 0 else ""
        note = (f"AI wage:       {ai_w:>9,.0f} THB\n"
                f"Rule-Based:  {h_w:>9,.0f} THB\n"
                f"Gap:           {sign}{gap:,.0f} THB")
        _note(ax, note, color=TEXT)

    fig.suptitle(
        f"{scenario_name.upper()}  ·  Monthly Wages: What Are Firms Paying Workers?\n"
        f"Higher wages attract more applicants; lower wages boost profit margins",
        color=WHITE, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout(pad=1.2)
    _save(fig, outdir, "wages.png")


# ─────────────────────────────────────────────────────────────────────
# Chart: Workforce Size
# ─────────────────────────────────────────────────────────────────────

def chart_workers(results, outdir, scenario_name, ai_color):
    """Average number of employees per firm over time."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), facecolor=BG)
    x = np.arange(1, N_STEPS + 1)

    for ax, cfg in zip(axes, CONFIGS):
        tag = cfg["tag"]
        if tag not in results:
            continue
        mean, std, sc = results[tag]
        _ax(ax)

        _band(ax, x, mean["rl_workers"], std["rl_workers"], ai_color, "AI Firms",         lw=2.2)
        _band(ax, x, mean["h_workers"],  std["h_workers"],  H_COL,    "Rule-Based Firms",  lw=1.8)

        ax.set_title(f"{cfg['label']}\n{cfg['subtitle']}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Month", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("Avg Employees per Firm", fontsize=9)
        ax.legend(fontsize=9, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

        ai_w = sc["rl_workers"]["mean"]
        h_w  = sc["h_workers"]["mean"]
        note = f"AI firms:      {ai_w:.1f} employees avg\nRule-Based: {h_w:.1f} employees avg"
        _note(ax, note, color=TEXT)

    fig.suptitle(
        f"{scenario_name.upper()}  ·  Workforce Size: How Many Employees Does Each Firm Hire?\n"
        f"Reflects how aggressively firms post vacancies and retain workers",
        color=WHITE, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout(pad=1.2)
    _save(fig, outdir, "workers.png")


# ─────────────────────────────────────────────────────────────────────
# Chart: Wage Spread  (cooperative / competitive only)
# ─────────────────────────────────────────────────────────────────────

def chart_wage_spread(results, outdir, scenario_name, ai_color, mode):
    if mode == "cooperative":
        subtitle = ("AI firms in cooperative mode should converge on similar wages "
                    "— lower spread means better coordination")
        ylabel   = "Wage Spread (Max − Min AI Wage, THB)"
    else:
        subtitle = ("AI firms in competitive mode are expected to differentiate wages "
                    "— higher spread means stronger rivalry")
        ylabel   = "Wage Spread (Max − Min AI Wage, THB)"

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), facecolor=BG)
    x = np.arange(1, N_STEPS + 1)

    for ax, cfg in zip(axes, CONFIGS):
        tag = cfg["tag"]
        if tag not in results:
            continue
        mean, std, sc = results[tag]
        _ax(ax)

        _band(ax, x, mean["rl_wage_spread"], std["rl_wage_spread"],
              ai_color, "Wage Spread", lw=2.2)
        ax.axhline(0, color=DIM, lw=0.8, ls="--", alpha=0.5)

        ax.set_title(f"{cfg['label']}\n{cfg['subtitle']}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Month", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel(ylabel, fontsize=9)

        avg = sc["rl_wage_spread"]["mean"]
        note = f"Average spread: {avg:,.0f} THB"
        _note(ax, note, color=TEXT)

    fig.suptitle(
        f"{scenario_name.upper()}  ·  Wage Spread Among AI Firms\n{subtitle}",
        color=WHITE, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout(pad=1.2)
    _save(fig, outdir, "wage_spread.png")


# ─────────────────────────────────────────────────────────────────────
# Chart: Scorecard  (summary bar chart)
# ─────────────────────────────────────────────────────────────────────

def chart_scorecard(results, outdir, scenario_name, ai_color):
    """
    Summary bar chart — quick visual answer to "did AI do better?"
    Panels: Profit | Employment | Wage | Workforce | Win Rate
    """
    panels = [
        # (title,           ai_key,       h_key,        scale, unit)
        ("Monthly Profit",   "rl_profit",  "h_profit",   1,     "THB"),
        ("Employment Rate",  "employ_pct", "employ_pct", 1,     "%"),
        ("Monthly Wage",     "rl_wage",    "h_wage",     1,     "THB"),
        ("Employees / Firm", "rl_workers", "h_workers",  1,     ""),
        ("AI Win Rate",      "ai_wins",    None,         100,   "%"),
    ]

    tags   = [cfg["tag"]              for cfg in CONFIGS if cfg["tag"] in results]
    labels = [cfg["label"].split("\n")[0] for cfg in CONFIGS if cfg["tag"] in results]
    x      = np.arange(len(tags))
    w      = 0.36

    fig, axes = plt.subplots(1, len(panels), figsize=(21, 6), facecolor=BG)

    for ax, (title, ai_key, h_key, scale, unit) in zip(axes, panels):
        _ax(ax)

        ai_vals = [results[t][2][ai_key]["mean"] * scale for t in tags]
        ai_errs = [results[t][2][ai_key]["std"]  * scale for t in tags]

        if h_key and h_key != ai_key:
            h_vals = [results[t][2][h_key]["mean"] * scale for t in tags]
            h_errs = [results[t][2][h_key]["std"]  * scale for t in tags]

            b1 = ax.bar(x - w / 2, ai_vals, w,
                        color=ai_color, edgecolor=GRID, lw=0.5, label="AI Strategy",
                        yerr=ai_errs, error_kw=dict(ecolor=DIM, capsize=3, lw=1.1))
            b2 = ax.bar(x + w / 2, h_vals,  w,
                        color=H_COL,   edgecolor=GRID, lw=0.5, label="Rule-Based",
                        yerr=h_errs, error_kw=dict(ecolor=DIM, capsize=3, lw=1.1))

            for bar, val in zip(b1, ai_vals):
                if val != 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + max(ai_errs) * 0.05,
                            f"{val:,.0f}", ha="center", va="bottom",
                            color=ai_color, fontsize=7.5, fontweight="bold")
            for bar, val in zip(b2, h_vals):
                if val != 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + max(h_errs) * 0.05,
                            f"{val:,.0f}", ha="center", va="bottom",
                            color=H_COL, fontsize=7.5)

            ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

        else:
            # Win-rate panel: single bars + 50% breakeven line
            ax.set_ylim(0, 108)
            bars = ax.bar(x, ai_vals, w * 1.6,
                          color=ai_color, edgecolor=GRID, lw=0.5,
                          yerr=ai_errs, error_kw=dict(ecolor=DIM, capsize=3, lw=1.1))
            ax.axhline(50, color=H_COL, lw=1.5, ls="--", alpha=0.8,
                       label="50 % (breakeven)")
            for bar, val in zip(bars, ai_vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 1.5,
                        f"{val:.0f}%", ha="center", va="bottom",
                        color=ai_color, fontsize=10, fontweight="bold")
            ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, color=TEXT, fontsize=8.5)
        ax.set_title(title, color=WHITE, fontsize=10, fontweight="bold")
        if unit:
            ax.set_ylabel(unit, fontsize=9)

    fig.suptitle(
        f"{scenario_name.upper()}  ·  Overall Scorecard: AI vs Rule-Based Firms\n"
        f"Averaged over {N_SEEDS} simulations x {N_STEPS} months · three market sizes",
        color=WHITE, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout(pad=1.4)
    _save(fig, outdir, "scorecard.png")


# ─────────────────────────────────────────────────────────────────────
# Chart: Workers Overview  (cross-scenario comparison → benchmark/comparison/)
# ─────────────────────────────────────────────────────────────────────

def chart_comparison_utility(all_results, outdir):
    """
    Cross-scenario comparison: average worker utility across all AI models.
    Saved to benchmark/comparison/utility_overview.png
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), facecolor=BG)
    x = np.arange(1, N_STEPS + 1)

    SCENARIO_ORDER = [
        ("solo",        "Solo AI"),
        ("cooperative", "Cooperative AI"),
        ("competitive", "Competitive AI"),
    ]

    for ax, cfg in zip(axes, CONFIGS):
        tag = cfg["tag"]
        _ax(ax)

        for key, label in SCENARIO_ORDER:
            if key not in all_results or tag not in all_results[key]:
                continue
            mean, std, _ = all_results[key][tag]
            _band(ax, x, mean["avg_worker_utility"], std["avg_worker_utility"],
                  AI_COL[key], label, lw=2.0)

        ax.set_title(f"{cfg['label']}\n{cfg['subtitle']}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Month", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("Average Worker Utility", fontsize=9)
        ax.legend(fontsize=8.5, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

        lines = []
        for key, label in SCENARIO_ORDER:
            if key in all_results and tag in all_results[key]:
                avg = all_results[key][tag][2]["avg_worker_utility"]["mean"]
                lines.append(f"{label.split()[0]}: {avg:.3f}")
        if lines:
            _note(ax, "\n".join(lines), color=TEXT)

    fig.suptitle(
        "ALL MODELS  ·  Average Worker Utility\n"
        f"Cobb-Douglas utility for all workers (employed + unemployed)  —  "
        f"mean ± 1 std across {N_SEEDS} simulations",
        color=WHITE, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout(pad=1.2)
    _save(fig, outdir, "utility_overview.png")


def chart_comparison_profit(all_results, outdir):
    """
    Cross-scenario comparison: monthly profit for all AI models + rule-based baseline.
    Saved to benchmark/comparison/profit_overview.png
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), facecolor=BG)
    x = np.arange(1, N_STEPS + 1)

    SCENARIO_ORDER = [
        ("solo",        "Solo AI"),
        ("cooperative", "Cooperative AI"),
        ("competitive", "Competitive AI"),
    ]

    for ax, cfg in zip(axes, CONFIGS):
        tag = cfg["tag"]
        _ax(ax)

        for key, label in SCENARIO_ORDER:
            if key not in all_results or tag not in all_results[key]:
                continue
            mean, std, _ = all_results[key][tag]
            _band(ax, x, mean["rl_profit"], std["rl_profit"], AI_COL[key], label, lw=2.0)

        for key, _ in SCENARIO_ORDER:
            if key in all_results and tag in all_results[key]:
                mean, std, _ = all_results[key][tag]
                _band(ax, x, mean["h_profit"], std["h_profit"],
                      H_COL, "Rule-Based", lw=1.6, ls="--", alpha=0.12)
                break

        ax.axhline(0, color=DIM, lw=0.8, ls="--", alpha=0.5)
        ax.set_title(f"{cfg['label']}\n{cfg['subtitle']}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Month", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("Monthly Profit (THB)", fontsize=9)
        ax.legend(fontsize=8.5, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

        lines = []
        for key, label in SCENARIO_ORDER:
            if key in all_results and tag in all_results[key]:
                avg = all_results[key][tag][2]["rl_profit"]["mean"]
                lines.append(f"{label.split()[0]}: {avg:,.0f} THB")
        for key, _ in SCENARIO_ORDER:
            if key in all_results and tag in all_results[key]:
                h_avg = all_results[key][tag][2]["h_profit"]["mean"]
                lines.append(f"Rule-Based: {h_avg:,.0f} THB")
                break
        if lines:
            _note(ax, "\n".join(lines), color=TEXT)

    fig.suptitle(
        "ALL MODELS  ·  Monthly Profit Comparison\n"
        f"Solo vs Cooperative vs Competitive AI vs Rule-Based  —  "
        f"mean ± 1 std across {N_SEEDS} simulations",
        color=WHITE, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout(pad=1.2)
    _save(fig, outdir, "profit_overview.png")


def chart_comparison_employment(all_results, outdir):
    """
    Cross-scenario comparison: market employment rate across all AI models.
    Saved to benchmark/comparison/employment_overview.png
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), facecolor=BG)
    x = np.arange(1, N_STEPS + 1)

    SCENARIO_ORDER = [
        ("solo",        "Solo AI"),
        ("cooperative", "Cooperative AI"),
        ("competitive", "Competitive AI"),
    ]

    for ax, cfg in zip(axes, CONFIGS):
        tag = cfg["tag"]
        _ax(ax)

        for key, label in SCENARIO_ORDER:
            if key not in all_results or tag not in all_results[key]:
                continue
            mean, std, _ = all_results[key][tag]
            _band(ax, x, mean["employ_pct"], std["employ_pct"], AI_COL[key], label, lw=2.0)

        ax.set_ylim(0, 105)
        ax.axhline(100, color=DIM, lw=0.8, ls="--", alpha=0.4)
        ax.set_title(f"{cfg['label']}\n{cfg['subtitle']}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Month", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("Workers Employed (%)", fontsize=9)
        ax.legend(fontsize=8.5, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

        lines = []
        for key, label in SCENARIO_ORDER:
            if key in all_results and tag in all_results[key]:
                avg = all_results[key][tag][2]["employ_pct"]["mean"]
                lines.append(f"{label.split()[0]}: {avg:.1f}%")
        if lines:
            _note(ax, "\n".join(lines), color=TEXT)

    fig.suptitle(
        "ALL MODELS  ·  Market Employment Rate\n"
        f"Percentage of workers with jobs across all model types  —  "
        f"mean ± 1 std across {N_SEEDS} simulations",
        color=WHITE, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout(pad=1.2)
    _save(fig, outdir, "employment_overview.png")


def chart_comparison_wages(all_results, outdir):
    """
    Cross-scenario comparison: AI firm wages vs rule-based vs market average.
    Saved to benchmark/comparison/wages_overview.png
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), facecolor=BG)
    x = np.arange(1, N_STEPS + 1)

    SCENARIO_ORDER = [
        ("solo",        "Solo AI"),
        ("cooperative", "Cooperative AI"),
        ("competitive", "Competitive AI"),
    ]

    for ax, cfg in zip(axes, CONFIGS):
        tag = cfg["tag"]
        _ax(ax)

        for key, label in SCENARIO_ORDER:
            if key not in all_results or tag not in all_results[key]:
                continue
            mean, std, _ = all_results[key][tag]
            _band(ax, x, mean["rl_wage"], std["rl_wage"], AI_COL[key], label, lw=2.0)

        for key, _ in SCENARIO_ORDER:
            if key in all_results and tag in all_results[key]:
                mean, std, _ = all_results[key][tag]
                _band(ax, x, mean["h_wage"], std["h_wage"],
                      H_COL, "Rule-Based", lw=1.6, ls="--", alpha=0.12)
                _band(ax, x, mean["mkt_wage"], std["mkt_wage"],
                      MKT_COL, "Market Average", lw=1.2, alpha=0.08)
                break

        ax.set_title(f"{cfg['label']}\n{cfg['subtitle']}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Month", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("Monthly Wage (THB)", fontsize=9)
        ax.legend(fontsize=8.5, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

        lines = []
        for key, label in SCENARIO_ORDER:
            if key in all_results and tag in all_results[key]:
                avg = all_results[key][tag][2]["rl_wage"]["mean"]
                lines.append(f"{label.split()[0]}: {avg:,.0f} THB")
        for key, _ in SCENARIO_ORDER:
            if key in all_results and tag in all_results[key]:
                h_avg = all_results[key][tag][2]["h_wage"]["mean"]
                lines.append(f"Rule-Based: {h_avg:,.0f} THB")
                break
        if lines:
            _note(ax, "\n".join(lines), color=TEXT)

    fig.suptitle(
        "ALL MODELS  ·  Monthly Wages Paid to Workers\n"
        f"Solo vs Cooperative vs Competitive AI vs Rule-Based  —  "
        f"mean ± 1 std across {N_SEEDS} simulations",
        color=WHITE, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout(pad=1.2)
    _save(fig, outdir, "wages_overview.png")


def chart_comparison_workers(all_results, outdir):
    """
    Compare average employees per firm across all AI scenarios and rule-based,
    one panel per market size.  Saved to benchmark/comparison/workers_overview.png
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), facecolor=BG)
    x = np.arange(1, N_STEPS + 1)

    SCENARIO_ORDER = [
        ("solo",        "Solo AI"),
        ("cooperative", "Cooperative AI"),
        ("competitive", "Competitive AI"),
    ]

    for ax, cfg in zip(axes, CONFIGS):
        tag = cfg["tag"]
        _ax(ax)

        # Draw each AI scenario
        for key, label in SCENARIO_ORDER:
            if key not in all_results or tag not in all_results[key]:
                continue
            mean, std, _ = all_results[key][tag]
            _band(ax, x, mean["rl_workers"], std["rl_workers"],
                  AI_COL[key], label, lw=2.0)

        # Draw rule-based baseline (same heuristic firms — use first available scenario)
        for key, _ in SCENARIO_ORDER:
            if key in all_results and tag in all_results[key]:
                mean, std, _ = all_results[key][tag]
                _band(ax, x, mean["h_workers"], std["h_workers"],
                      H_COL, "Rule-Based", lw=1.6, ls="--", alpha=0.12)
                break

        ax.set_title(f"{cfg['label']}\n{cfg['subtitle']}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Month", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("Avg Employees per Firm", fontsize=9)
        ax.legend(fontsize=8.5, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

        # Summary note: avg workers per scenario
        lines = []
        for key, label in SCENARIO_ORDER:
            if key in all_results and tag in all_results[key]:
                avg = all_results[key][tag][2]["rl_workers"]["mean"]
                lines.append(f"{label.split()[0]}: {avg:.1f} workers")
        for key, _ in SCENARIO_ORDER:
            if key in all_results and tag in all_results[key]:
                h_avg = all_results[key][tag][2]["h_workers"]["mean"]
                lines.append(f"Rule-Based: {h_avg:.1f} workers")
                break
        if lines:
            _note(ax, "\n".join(lines), color=TEXT)

    fig.suptitle(
        "ALL MODELS  ·  Workforce Size: Average Employees per Firm\n"
        f"Solo vs Cooperative vs Competitive AI vs Rule-Based  —  "
        f"mean ± 1 std across {N_SEEDS} simulations",
        color=WHITE, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout(pad=1.2)
    _save(fig, outdir, "workers_overview.png")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'='*68}")
    print(f"  AI vs Rule-Based Benchmark")
    print(f"  {N_SEEDS} seeds  x  {N_STEPS} months  x  {len(CONFIGS)} market sizes")
    print(f"{'='*68}\n")

    print("[1/3] Solo  (1 AI firm, reformed rules: market-quit + snap wage)")
    solo_res = run_solo()

    print("\n[2/3] Cooperative  (3 AI firms working as a team)")
    coop_res = run_cooperative()

    print("\n[3/3] Competitive  (3 AI firms competing against each other)")
    comp_res = run_competitive()

    SCENARIOS = [
        ("solo",        "Solo",        solo_res, AI_COL["solo"]),
        ("cooperative", "Cooperative", coop_res, AI_COL["cooperative"]),
        ("competitive", "Competitive", comp_res, AI_COL["competitive"]),
    ]

    print(f"\n{'='*68}")
    print("  Generating charts...")
    print(f"{'='*68}")

    for key, name, res, color in SCENARIOS:
        if res is None:
            print(f"\n  [{key}] No results - skipped.")
            continue
        outdir = BENCH / key
        print(f"\n  {name}:")
        chart_profit(res, outdir, name, color)
        chart_employment(res, outdir, name, color)
        chart_wages(res, outdir, name, color)
        chart_workers(res, outdir, name, color)
        chart_scorecard(res, outdir, name, color)
        if key in ("cooperative", "competitive"):
            chart_wage_spread(res, outdir, name, color, mode=key)

    all_results = {k: r for k, _, r, _ in SCENARIOS if r is not None}

    # ── Cross-scenario comparison charts ────────────────────────────
    comparison_dir = BENCH / "comparison"
    print(f"\n  Cross-scenario comparison (benchmark/comparison/):")
    chart_comparison_profit(all_results, comparison_dir)
    chart_comparison_employment(all_results, comparison_dir)
    chart_comparison_wages(all_results, comparison_dir)
    chart_comparison_workers(all_results, comparison_dir)
    chart_comparison_utility(all_results, comparison_dir)

    # ── Text summary table ───────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  {'SUMMARY TABLE':^76}")
    print(f"{'='*80}")
    cfg_tags    = [c["tag"] for c in CONFIGS]
    cfg_short   = {"small": "Small", "medium": "Medium", "large": "Large"}

    if all_results:
        print(f"\n  {'Metric':<32}  " + "  ".join(
            f"{'--- ' + cfg_short[t] + ' ---':>20}" for t in cfg_tags))
        print(f"  {'':32}  " + "  ".join(
            f"{'AI Firms':>9}  {'Rule-Based':>9}" for _ in cfg_tags))

        def _row(scenario_key, label, metric_ai, metric_h, fmt="{:,.0f}"):
            r = all_results.get(scenario_key)
            if r is None:
                return
            parts = []
            for t in cfg_tags:
                if t in r:
                    ai_v = r[t][2][metric_ai]["mean"]
                    h_v  = r[t][2][metric_h]["mean"] if metric_h else None
                    ai_s = fmt.format(ai_v)
                    h_s  = fmt.format(h_v) if h_v is not None else "N/A"
                else:
                    ai_s = h_s = "N/A"
                parts.append(f"{ai_s:>9}  {h_s:>9}")
            print(f"  {label:<32}  " + "  ".join(parts))

        for scenario_key, name, _, _ in SCENARIOS:
            if scenario_key not in all_results:
                continue
            print(f"\n  [{name}]")
            _row(scenario_key, "Avg monthly profit (THB)",  "rl_profit",  "h_profit")
            _row(scenario_key, "Avg employees / firm",      "rl_workers", "h_workers", "{:.1f}")
            _row(scenario_key, "Avg wage paid (THB)",       "rl_wage",    "h_wage")
            _row(scenario_key, "Employment rate (%)",       "employ_pct", "employ_pct", "{:.1f}")
            _row(scenario_key, "AI win rate",               "ai_wins",    None, "{:.1%}")

    print(f"\n{'='*80}")
    print(f"\n  Charts saved to:  benchmark/\n")

    # Save results cache so benchmark_heuristic.py can load RL data for comparison
    cache_path = BENCH / ".rl_cache.pkl"
    with open(cache_path, "wb") as _f:
        pickle.dump(all_results, _f)
    print(f"  RL results cached to: {cache_path.name}\n")
