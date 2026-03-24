#!/usr/bin/env python
# viz_all_models.py
#
# Runs all three trained scenarios and saves one PNG per metric per model.
#
# Output folder structure:
#   visualizations/
#     solo/
#       profit.png         RL vs heuristic profit  (green/red fill)
#       profit_delta.png   RL - heuristic profit gap over time
#       workers.png        RL vs heuristic worker headcount
#       wage.png           RL vs heuristic firm wage
#       employment.png     economy-wide employment rate
#       scorecard.png      RL % advantage per metric (horizontal bars)
#     cooperative/         same set of 6 charts
#     competitive/         same set of 6 charts
#     comparison/
#       summary.png        grouped bar chart — all models side by side
#       scorecard.png      RL % edge per metric, one panel per model
#
# Green fill / bar = RL doing better than heuristic
# Red   fill / bar = RL doing worse  than heuristic
#
# Usage:
#   python viz_all_models.py

import sys
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT    = Path(__file__).parent.resolve()
VIZ_DIR = ROOT / "visualizations"

N_STEPS    = 360
N_RL_FIRMS = 3

# Fixed seed so every run produces the same episode and all charts are comparable.
# Change this number to explore a different episode.
EVAL_SEED = 42

# ------------------------------------------------------------------ #
#  Shared colour palette                                              #
# ------------------------------------------------------------------ #

BG      = "#0f0f1a"
PANEL   = "#16213e"
GRID    = "#1e2a4a"
TEXT    = "#d0d0e8"
DIM     = "#888899"
BETTER  = "#00c853"   # green — RL better
WORSE   = "#ff1744"   # red   — RL worse
RL_COL  = "#4fc3f7"   # blue  — RL line
H_COL   = "#ffb74d"   # orange — heuristic line
MAC_COL = "#ce93d8"   # purple — macro / single-line metrics


# ------------------------------------------------------------------ #
#  Scenario loading helpers                                           #
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


def _collect(env, is_multi, rl_firms_attr=None):
    """Collect one step's worth of metrics from env after stepping."""
    if is_multi:
        rl_firms  = env.rl_firms
        heuristic = [f for f in env.model.firms if f not in rl_firms]
        rl_profit  = float(np.mean([f.profit          for f in rl_firms]))
        rl_wage    = float(np.mean([f.monthly_wage    for f in rl_firms]))
        rl_workers = float(np.mean([len(f.current_workers) for f in rl_firms]))
    else:
        rl_firms  = [env.rl_firm]
        heuristic = [f for f in env.model.firms if f is not env.rl_firm]
        rl_profit  = float(env.rl_firm.profit)
        rl_wage    = float(env.rl_firm.monthly_wage)
        rl_workers = float(len(env.rl_firm.current_workers))

    employed = sum(1 for w in env.model.workers if w.employed)

    return {
        "rl_profit":       rl_profit,
        "h_profit":        float(np.mean([f.profit for f in heuristic]))          if heuristic else 0.0,
        "employment_rate": employed / max(len(env.model.workers), 1),
        "market_wage":     float(np.mean([f.monthly_wage for f in env.model.firms])),
        "rl_wage":         rl_wage,
        "h_wage":          float(np.mean([f.monthly_wage for f in heuristic]))    if heuristic else 0.0,
        "rl_workers":      rl_workers,
        "h_workers":       float(np.mean([len(f.current_workers) for f in heuristic])) if heuristic else 0.0,
    }


# ------------------------------------------------------------------ #
#  Scenario runners                                                   #
# ------------------------------------------------------------------ #

def run_solo():
    random.seed(EVAL_SEED)
    np.random.seed(EVAL_SEED)
    d = _load_scenario("solo")
    try:
        from sb3_contrib import MaskablePPO
        from firm_env import LaborMarketEnv
    except ImportError as e:
        print(f"  [solo] import error: {e}")
        _unload_scenario("solo")
        return None

    if not (d / "solo_model.zip").exists():
        print("  [solo] model not found — skipping.")
        _unload_scenario("solo")
        return None

    policy = MaskablePPO.load(str(d / "solo_model"))
    env    = LaborMarketEnv()
    obs, _ = env.reset()
    rows   = []

    for _ in range(N_STEPS):
        mask      = env.action_masks()
        act, _    = policy.predict(obs[np.newaxis], deterministic=True,
                                   action_masks=mask[np.newaxis])
        obs, _, _, _, _ = env.step(int(act[0]))
        rows.append(_collect(env, is_multi=False))

    _unload_scenario("solo")
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def run_cooperative():
    random.seed(EVAL_SEED)
    np.random.seed(EVAL_SEED)
    d = _load_scenario("cooperative")
    try:
        from sb3_contrib import MaskablePPO
        from firm_env import CoopFirmEnv
    except ImportError as e:
        print(f"  [cooperative] import error: {e}")
        _unload_scenario("cooperative")
        return None

    if not (d / "coop_model.zip").exists():
        print("  [cooperative] model not found — skipping.")
        _unload_scenario("cooperative")
        return None

    policy = MaskablePPO.load(str(d / "coop_model"))
    env    = CoopFirmEnv()
    obs, _ = env.reset()
    rows   = []

    for _ in range(N_STEPS * N_RL_FIRMS):
        mask      = env.action_masks()
        act, _    = policy.predict(obs[np.newaxis], deterministic=True,
                                   action_masks=mask[np.newaxis])
        obs, _, _, _, _ = env.step(int(act[0]))
        if env.current_idx == 0:
            rows.append(_collect(env, is_multi=True))

    _unload_scenario("cooperative")
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def run_competitive():
    random.seed(EVAL_SEED)
    np.random.seed(EVAL_SEED)
    d = _load_scenario("competitive")
    try:
        from sb3_contrib import MaskablePPO
        from firm_env import CompFirmEnv
    except ImportError as e:
        print(f"  [competitive] import error: {e}")
        _unload_scenario("competitive")
        return None

    if not (d / "comp_model.zip").exists():
        print("  [competitive] model not found — skipping.")
        _unload_scenario("competitive")
        return None

    policy = MaskablePPO.load(str(d / "comp_model"))
    env    = CompFirmEnv()
    obs, _ = env.reset()
    rows   = []

    for _ in range(N_STEPS * N_RL_FIRMS):
        mask      = env.action_masks()
        act, _    = policy.predict(obs[np.newaxis], deterministic=True,
                                   action_masks=mask[np.newaxis])
        obs, _, _, _, _ = env.step(int(act[0]))
        if env.current_idx == 0:
            rows.append(_collect(env, is_multi=True))

    _unload_scenario("competitive")
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


# ------------------------------------------------------------------ #
#  Drawing primitives                                                 #
# ------------------------------------------------------------------ #

def _new_fig(w=14, h=6):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=10)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color("white")
    ax.grid(color=GRID, alpha=0.35, lw=0.5, zorder=0)
    return fig, ax


def _edge_label(ax, rl, h, better_higher=True):
    """Avg RL-vs-H edge label in bottom-right corner."""
    delta  = float(np.mean(rl - h))
    rl_win = (delta > 0) == better_higher
    col    = BETTER if rl_win else WORSE
    sign   = "+" if delta > 0 else ""
    ax.text(0.98, 0.04, f"Avg RL edge: {sign}{delta:,.1f}",
            transform=ax.transAxes, ha="right", va="bottom",
            color=col, fontsize=9, fontweight="bold",
            bbox=dict(facecolor=BG, alpha=0.6, edgecolor="none", pad=3))


def _save(fig, folder, filename):
    out = VIZ_DIR / folder
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved: {folder}/{filename}")


# ------------------------------------------------------------------ #
#  Per-model chart builders                                           #
# ------------------------------------------------------------------ #

def chart_profit(folder, label, res):
    steps = np.arange(len(res["rl_profit"]))
    rl, h = res["rl_profit"], res["h_profit"]

    fig, ax = _new_fig()
    ax.plot(steps, h,  color=H_COL,  lw=1.5, alpha=0.85, label="Heuristic (avg)", zorder=3)
    ax.plot(steps, rl, color=RL_COL, lw=2.0,              label="RL (avg)",        zorder=4)
    ax.fill_between(steps, rl, h, where=(rl >= h), color=BETTER, alpha=0.22, zorder=2)
    ax.fill_between(steps, rl, h, where=(rl <  h), color=WORSE,  alpha=0.22, zorder=2)
    _edge_label(ax, rl, h)

    ax.set_title(f"Profit — RL vs Heuristic\n{label}\nGreen = RL earning more  |  Red = RL earning less",
                 fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Model Step", fontsize=10)
    ax.set_ylabel("Profit (THB)", fontsize=10)
    ax.set_xlim(0, len(steps) - 1)
    ax.legend(fontsize=9, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT,
              loc="upper left", framealpha=0.7)
    _save(fig, folder, "profit.png")


def chart_profit_delta(folder, label, res):
    steps = np.arange(len(res["rl_profit"]))
    delta = res["rl_profit"] - res["h_profit"]

    fig, ax = _new_fig()
    ax.axhline(0, color=DIM, lw=1.2, ls="--", zorder=3)
    ax.fill_between(steps, delta, 0, where=(delta >= 0), color=BETTER, alpha=0.40, zorder=2)
    ax.fill_between(steps, delta, 0, where=(delta <  0), color=WORSE,  alpha=0.40, zorder=2)
    ax.plot(steps, delta, color=TEXT, lw=1.0, alpha=0.55, zorder=4)

    avg = float(np.mean(delta))
    col = BETTER if avg >= 0 else WORSE
    ax.text(0.98, 0.04, f"Avg: {avg:+,.1f} THB",
            transform=ax.transAxes, ha="right", va="bottom",
            color=col, fontsize=9, fontweight="bold",
            bbox=dict(facecolor=BG, alpha=0.6, edgecolor="none", pad=3))

    ax.set_title(f"Profit Gap — RL minus Heuristic\n{label}\nAbove zero (green) = RL winning  |  Below zero (red) = RL losing",
                 fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Model Step", fontsize=10)
    ax.set_ylabel("Profit Gap (THB)", fontsize=10)
    ax.set_xlim(0, len(steps) - 1)
    _save(fig, folder, "profit_delta.png")


def chart_workers(folder, label, res):
    steps = np.arange(len(res["rl_workers"]))
    rl, h = res["rl_workers"], res["h_workers"]

    fig, ax = _new_fig()
    ax.plot(steps, h,  color=H_COL,  lw=1.5, alpha=0.85, label="Heuristic (avg)", zorder=3)
    ax.plot(steps, rl, color=RL_COL, lw=2.0,              label="RL (avg)",        zorder=4)
    ax.fill_between(steps, rl, h, where=(rl >= h), color=BETTER, alpha=0.22, zorder=2)
    ax.fill_between(steps, rl, h, where=(rl <  h), color=WORSE,  alpha=0.22, zorder=2)
    _edge_label(ax, rl, h)

    ax.set_title(f"Worker Headcount — RL vs Heuristic\n{label}\nGreen = RL employing more workers  |  Red = RL employing fewer",
                 fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Model Step", fontsize=10)
    ax.set_ylabel("Avg Workers per Firm", fontsize=10)
    ax.set_xlim(0, len(steps) - 1)
    ax.legend(fontsize=9, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT,
              loc="upper left", framealpha=0.7)
    _save(fig, folder, "workers.png")


def chart_wage(folder, label, res):
    steps = np.arange(len(res["rl_wage"]))
    rl, h = res["rl_wage"], res["h_wage"]
    mkt   = res["market_wage"]

    fig, ax = _new_fig(w=14, h=7)
    ax.plot(steps, h,   color=H_COL,   lw=1.5, alpha=0.85, label="Heuristic avg wage", zorder=3)
    ax.plot(steps, rl,  color=RL_COL,  lw=2.0,              label="RL avg wage",        zorder=4)
    ax.plot(steps, mkt, color=MAC_COL, lw=1.2, ls=":",
            alpha=0.6, label="Market avg wage (all firms)", zorder=3)
    ax.fill_between(steps, rl, h, where=(rl >= h), color=BETTER, alpha=0.22, zorder=2)
    ax.fill_between(steps, rl, h, where=(rl <  h), color=WORSE,  alpha=0.22, zorder=2)
    _edge_label(ax, rl, h)

    ax.set_title(f"Monthly Wage — RL vs Heuristic\n{label}\nGreen = RL paying higher wage  |  Red = RL paying lower wage\nDotted purple = full market average",
                 fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Model Step", fontsize=10)
    ax.set_ylabel("Monthly Wage (THB)", fontsize=10)
    ax.set_xlim(0, len(steps) - 1)
    ax.legend(fontsize=9, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT,
              loc="best", framealpha=0.7)
    _save(fig, folder, "wage.png")


def chart_employment(folder, label, res):
    steps = np.arange(len(res["employment_rate"]))
    emp   = res["employment_rate"] * 100

    fig, ax = _new_fig()
    ax.fill_between(steps, emp, alpha=0.15, color=BETTER)
    ax.plot(steps, emp, color=BETTER, lw=2.0)
    ax.axhline(70, color=DIM, lw=1, ls="--", alpha=0.7)
    ax.text(steps[-1] * 0.02, 71.5, "70% reference", color=DIM, fontsize=8)

    avg = float(np.mean(emp))
    col = BETTER if avg >= 70 else WORSE
    ax.text(0.98, 0.04, f"Avg: {avg:.1f}%",
            transform=ax.transAxes, ha="right", va="bottom",
            color=col, fontsize=9, fontweight="bold",
            bbox=dict(facecolor=BG, alpha=0.6, edgecolor="none", pad=3))

    ax.set_title(f"Economy-Wide Employment Rate\n{label}\nGreen fill = above 70% (healthy)  |  Dashed line = 70% reference",
                 fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Model Step", fontsize=10)
    ax.set_ylabel("Employment Rate (%)", fontsize=10)
    ax.set_xlim(0, len(steps) - 1)
    ax.set_ylim(0, 105)
    _save(fig, folder, "employment.png")


def chart_scorecard(folder, label, res):
    """Horizontal bar chart — RL % advantage per metric."""
    specs = [
        ("rl_profit",  "h_profit",  "Profit",   True),
        ("rl_workers", "h_workers", "Workers",  True),
        ("rl_wage",    "h_wage",    "Wage",      True),
    ]

    metric_names, pcts, colors = [], [], []
    for rk, hk, mname, hi in specs:
        ra = float(np.mean(res[rk]))
        ha = float(np.mean(res[hk]))
        pct = ((ra - ha) / abs(ha) * 100) if ha != 0 else 0.0
        if not hi:
            pct = -pct
        metric_names.append(mname)
        pcts.append(pct)
        colors.append(BETTER if pct >= 0 else WORSE)

    fig, ax = _new_fig(w=9, h=5)
    y    = np.arange(len(metric_names))
    bars = ax.barh(y, pcts, color=colors, height=0.45,
                   edgecolor=GRID, linewidth=0.5)
    ax.axvline(0, color=DIM, lw=1.5, zorder=5)

    xlim = max(abs(p) for p in pcts) * 1.45 if pcts else 10
    for bar, val in zip(bars, pcts):
        ha  = "left"  if val >= 0 else "right"
        off = xlim * 0.025
        ax.text(val + (off if val >= 0 else -off),
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.1f}%", ha=ha, va="center",
                color="white", fontsize=10, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(metric_names, color=TEXT, fontsize=11)
    ax.set_xlabel("RL advantage vs heuristic (%)", fontsize=10)
    ax.set_xlim(-xlim, xlim)
    ax.axvspan(   0, xlim, color=BETTER, alpha=0.05)
    ax.axvspan(-xlim, 0,   color=WORSE,  alpha=0.05)

    ax.set_title(f"RL Effect Scorecard\n{label}\nGreen = positive RL effect  |  Red = negative RL effect",
                 fontsize=12, fontweight="bold", pad=8)
    _save(fig, folder, "scorecard.png")


# ------------------------------------------------------------------ #
#  Cross-model comparison charts (saved to comparison/)              #
# ------------------------------------------------------------------ #

def chart_comparison_summary(results, active_names, model_labels):
    """Grouped bar chart — RL vs Heuristic for each metric, one group per model."""
    specs = [
        ("rl_profit",  "h_profit",  "Avg Profit per Firm (THB)"),
        ("rl_workers", "h_workers", "Avg Workers per Firm"),
        ("rl_wage",    "h_wage",    "Avg Monthly Wage (THB)"),
    ]
    labels = [n.upper() for n in active_names]
    x      = np.arange(len(active_names))
    bar_w  = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(21, 7), facecolor=BG)
    fig.suptitle("Cross-Model Summary  —  Averages Over 360 Steps\n"
                 "Blue = RL firms  |  Orange = Heuristic firms",
                 color="white", fontsize=15, fontweight="bold", y=1.02)

    for ax, (rk, hk, ylabel) in zip(axes, specs):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT, labelsize=10)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)
        ax.grid(axis="y", color=GRID, alpha=0.35, lw=0.5)

        rl_vals = [float(np.mean(results[n][rk])) if results.get(n) is not None else 0.0
                   for n in active_names]
        h_vals  = [float(np.mean(results[n][hk])) if results.get(n) is not None else 0.0
                   for n in active_names]

        bars_rl = ax.bar(x - bar_w / 2, rl_vals, bar_w,
                         color=RL_COL, label="RL", edgecolor=GRID, linewidth=0.5)
        bars_h  = ax.bar(x + bar_w / 2, h_vals,  bar_w,
                         color=H_COL,  label="Heuristic", edgecolor=GRID, linewidth=0.5)

        ref = max(abs(v) for v in rl_vals + h_vals) * 0.012
        for bar, val in zip(bars_rl, rl_vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ref,
                    f"{val:,.0f}", ha="center", va="bottom",
                    color=TEXT, fontsize=8, fontweight="bold")
        for bar, val in zip(bars_h, h_vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ref,
                    f"{val:,.0f}", ha="center", va="bottom",
                    color=TEXT, fontsize=8)

        # "RL wins" badge above RL bar
        for i, (rv, hv) in enumerate(zip(rl_vals, h_vals)):
            if rv > hv:
                top = max(rv, hv) * 1.08
                ax.text(x[i] - bar_w / 2, top, "[RL wins]",
                        ha="center", va="bottom",
                        color=BETTER, fontsize=8, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, color=TEXT, fontsize=11)
        ax.set_ylabel(ylabel, color=TEXT, fontsize=10)
        ax.yaxis.label.set_color(TEXT)
        ax.legend(fontsize=9, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    plt.tight_layout()
    _save(fig, "comparison", "summary.png")


def chart_comparison_scorecard(results, active_names, model_labels):
    """Side-by-side scorecards for all models in one figure."""
    specs = [
        ("rl_profit",  "h_profit",  "Profit",  True),
        ("rl_workers", "h_workers", "Workers", True),
        ("rl_wage",    "h_wage",    "Wage",     True),
    ]

    fig, axes = plt.subplots(1, len(active_names),
                             figsize=(7 * len(active_names), 6), facecolor=BG)
    if len(active_names) == 1:
        axes = [axes]
    fig.suptitle("RL Effect Scorecard  —  All Models\n"
                 "Green = RL has positive effect vs heuristic  |  Red = negative effect",
                 color="white", fontsize=14, fontweight="bold", y=1.04)

    for ax, name in zip(axes, active_names):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT, labelsize=10)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)

        res = results.get(name)
        if res is None:
            ax.set_visible(False)
            continue

        m_names, pcts, cols = [], [], []
        for rk, hk, mname, hi in specs:
            ra  = float(np.mean(res[rk]))
            ha  = float(np.mean(res[hk]))
            pct = ((ra - ha) / abs(ha) * 100) if ha != 0 else 0.0
            if not hi:
                pct = -pct
            m_names.append(mname)
            pcts.append(pct)
            cols.append(BETTER if pct >= 0 else WORSE)

        y    = np.arange(len(m_names))
        bars = ax.barh(y, pcts, color=cols, height=0.45,
                       edgecolor=GRID, linewidth=0.5)
        ax.axvline(0, color=DIM, lw=1.5, zorder=5)

        xlim = max(abs(p) for p in pcts) * 1.45 if pcts else 10
        for bar, val in zip(bars, pcts):
            ha  = "left"  if val >= 0 else "right"
            off = xlim * 0.025
            ax.text(val + (off if val >= 0 else -off),
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:+.1f}%", ha=ha, va="center",
                    color="white", fontsize=10, fontweight="bold")

        ax.set_yticks(y)
        ax.set_yticklabels(m_names, color=TEXT, fontsize=11)
        ax.set_xlabel("RL advantage vs heuristic (%)", color=TEXT, fontsize=10)
        ax.set_xlim(-xlim, xlim)
        ax.axvspan(   0, xlim, color=BETTER, alpha=0.05)
        ax.axvspan(-xlim, 0,   color=WORSE,  alpha=0.05)
        ax.set_title(model_labels[name], color="white", fontsize=10,
                     fontweight="bold", pad=8)
        ax.grid(axis="x", color=GRID, alpha=0.35, lw=0.5)

    plt.tight_layout()
    _save(fig, "comparison", "scorecard.png")


# ------------------------------------------------------------------ #
#  Folder name map                                                    #
# ------------------------------------------------------------------ #

FOLDER_NAME = {
    "solo": "solo",
    "coop": "cooperative",
    "comp": "competitive",
}

MODEL_LABELS = {
    "solo": "Solo  (1 RL firm vs 9 heuristic)",
    "coop": "Cooperative  (3 RL firms, shared reward)",
    "comp": "Competitive  (3 RL firms, relative reward)",
}


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    print("\nRunning evaluations...")
    print("  [1/3] Solo        (1 RL firm vs 9 heuristic)...")
    solo = run_solo()

    print("  [2/3] Cooperative (3 RL firms, shared reward)...")
    coop = run_cooperative()

    print("  [3/3] Competitive (3 RL firms, relative reward)...")
    comp = run_competitive()

    results = {"solo": solo, "coop": coop, "comp": comp}
    available = [k for k, v in results.items() if v is not None]

    if not available:
        print("No models found. Exiting.")
        sys.exit(1)

    print(f"\n  Models loaded: {', '.join(available)}")
    print(f"  Saving charts to: {VIZ_DIR}\n")

    for key in available:
        res    = results[key]
        folder = FOLDER_NAME[key]
        label  = MODEL_LABELS[key]
        print(f"  [{label}]")
        chart_profit(folder, label, res)
        chart_profit_delta(folder, label, res)
        chart_workers(folder, label, res)
        chart_wage(folder, label, res)
        chart_employment(folder, label, res)
        chart_scorecard(folder, label, res)

    print("\n  [comparison]")
    active_labels = {n: MODEL_LABELS[n] for n in available}
    chart_comparison_summary(results, available, active_labels)
    chart_comparison_scorecard(results, available, active_labels)

    print("\nDone. Folder structure:")
    print("  visualizations/")
    for key in available:
        folder = FOLDER_NAME[key]
        print(f"    {folder}/")
        for f in ["profit.png", "profit_delta.png", "workers.png",
                  "wage.png", "employment.png", "scorecard.png"]:
            print(f"      {f}")
    print("    comparison/")
    print("      summary.png")
    print("      scorecard.png")
