#!/usr/bin/env python
# viz_avg_seeds.py
#
# Runs all three scenarios across multiple random seeds and saves two versions
# of every chart:
#
#   *_seeds.png  — one coloured line per seed so you can see individual runs
#   *_avg.png    — mean line with +/- 1 std band (overall average picture)
#
# Output: visualizations_temp/<model>/
#   profit_seeds.png       profit_avg.png
#   profit_delta_seeds.png profit_delta_avg.png
#   workers_seeds.png      workers_avg.png
#   wage_seeds.png         wage_avg.png
#   employment_seeds.png   employment_avg.png
#   scorecard.png          (average only — already has error bars per metric)
#
#   visualizations_temp/comparison/
#   summary.png   scorecard.png
#
# Usage:
#   python viz_avg_seeds.py

import sys
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT    = Path(__file__).parent.resolve()
VIZ_DIR = ROOT / "visualizations_temp"

N_STEPS    = 360
N_RL_FIRMS = 3

# ------------------------------------------------------------------ #
#  Seeds — one coloured line per seed in the _seeds charts           #
# ------------------------------------------------------------------ #

SEEDS = [42, 123, 456, 789, 1000]

# One distinct colour per seed (used for the individual-run lines)
SEED_COLORS = [
    "#4fc3f7",   # seed 42   — sky blue
    "#ef5350",   # seed 123  — red
    "#66bb6a",   # seed 456  — green
    "#ffa726",   # seed 789  — amber
    "#ab47bc",   # seed 1000 — purple
]

# ------------------------------------------------------------------ #
#  Shared colour palette                                              #
# ------------------------------------------------------------------ #

BG      = "#0f0f1a"
PANEL   = "#16213e"
GRID    = "#1e2a4a"
TEXT    = "#d0d0e8"
DIM     = "#888899"
BETTER  = "#00c853"
WORSE   = "#ff1744"
RL_COL  = "#4fc3f7"
H_COL   = "#ffb74d"
MAC_COL = "#ce93d8"
MEAN_COL = "#ffffff"   # bold white — mean RL line on seeds charts


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


def _collect(env, is_multi):
    if is_multi:
        rl_firms  = env.rl_firms
        heuristic = [f for f in env.model.firms if f not in rl_firms]
        rl_profit  = float(np.mean([f.profit               for f in rl_firms]))
        rl_wage    = float(np.mean([f.monthly_wage         for f in rl_firms]))
        rl_workers = float(np.mean([len(f.current_workers) for f in rl_firms]))
    else:
        heuristic  = [f for f in env.model.firms if f is not env.rl_firm]
        rl_profit  = float(env.rl_firm.profit)
        rl_wage    = float(env.rl_firm.monthly_wage)
        rl_workers = float(len(env.rl_firm.current_workers))

    employed = sum(1 for w in env.model.workers if w.employed)
    return {
        "rl_profit":       rl_profit,
        "h_profit":        float(np.mean([f.profit               for f in heuristic])) if heuristic else 0.0,
        "employment_rate": employed / max(len(env.model.workers), 1),
        "market_wage":     float(np.mean([f.monthly_wage         for f in env.model.firms])),
        "rl_wage":         rl_wage,
        "h_wage":          float(np.mean([f.monthly_wage         for f in heuristic])) if heuristic else 0.0,
        "rl_workers":      rl_workers,
        "h_workers":       float(np.mean([len(f.current_workers) for f in heuristic])) if heuristic else 0.0,
    }


# ------------------------------------------------------------------ #
#  Single-seed runners                                                #
# ------------------------------------------------------------------ #

def _run_solo_seed(seed):
    random.seed(seed); np.random.seed(seed)
    d = _load_scenario("solo")
    from sb3_contrib import MaskablePPO
    from firm_env import LaborMarketEnv
    policy = MaskablePPO.load(str(d / "solo_model"))
    env    = LaborMarketEnv()
    obs, _ = env.reset()
    rows   = []
    for _ in range(N_STEPS):
        mask   = env.action_masks()
        act, _ = policy.predict(obs[np.newaxis], deterministic=True,
                                action_masks=mask[np.newaxis])
        obs, _, _, _, _ = env.step(int(act[0]))
        rows.append(_collect(env, is_multi=False))
    _unload_scenario("solo")
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def _run_coop_seed(seed):
    random.seed(seed); np.random.seed(seed)
    d = _load_scenario("cooperative")
    from sb3_contrib import MaskablePPO
    from firm_env import CoopFirmEnv
    policy = MaskablePPO.load(str(d / "coop_model"))
    env    = CoopFirmEnv()
    obs, _ = env.reset()
    rows   = []
    for _ in range(N_STEPS * N_RL_FIRMS):
        mask   = env.action_masks()
        act, _ = policy.predict(obs[np.newaxis], deterministic=True,
                                action_masks=mask[np.newaxis])
        obs, _, _, _, _ = env.step(int(act[0]))
        if env.current_idx == 0:
            rows.append(_collect(env, is_multi=True))
    _unload_scenario("cooperative")
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def _run_comp_seed(seed):
    random.seed(seed); np.random.seed(seed)
    d = _load_scenario("competitive")
    from sb3_contrib import MaskablePPO
    from firm_env import CompFirmEnv
    policy = MaskablePPO.load(str(d / "comp_model"))
    env    = CompFirmEnv()
    obs, _ = env.reset()
    rows   = []
    for _ in range(N_STEPS * N_RL_FIRMS):
        mask   = env.action_masks()
        act, _ = policy.predict(obs[np.newaxis], deterministic=True,
                                action_masks=mask[np.newaxis])
        obs, _, _, _, _ = env.step(int(act[0]))
        if env.current_idx == 0:
            rows.append(_collect(env, is_multi=True))
    _unload_scenario("competitive")
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


# ------------------------------------------------------------------ #
#  Multi-seed aggregator                                              #
# ------------------------------------------------------------------ #

def run_multi_seed(scenario_key):
    runner_map = {
        "solo": ("solo",        "solo_model.zip",  _run_solo_seed),
        "coop": ("cooperative", "coop_model.zip",  _run_coop_seed),
        "comp": ("competitive", "comp_model.zip",  _run_comp_seed),
    }
    folder, model_file, runner = runner_map[scenario_key]
    if not (ROOT / folder / model_file).exists():
        print(f"  [{scenario_key}] model not found — skipping.")
        return None

    all_runs = []
    for i, seed in enumerate(SEEDS):
        print(f"    seed {seed}  ({i+1}/{len(SEEDS)})")
        try:
            all_runs.append(runner(seed))
        except Exception as e:
            print(f"    seed {seed} failed: {e}")

    if not all_runs:
        return None

    keys   = list(all_runs[0].keys())
    result = {}
    for k in keys:
        stack = np.stack([r[k] for r in all_runs], axis=0)  # (n_seeds, n_steps)
        result[k] = {
            "mean":  np.mean(stack, axis=0),
            "std":   np.std(stack,  axis=0),
            "seeds": [r[k] for r in all_runs],   # list of (n_steps,) arrays
        }
    return result


# ------------------------------------------------------------------ #
#  Drawing primitives                                                 #
# ------------------------------------------------------------------ #

def _new_fig(w=16, h=7):
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


def _save(fig, folder, filename):
    out = VIZ_DIR / folder
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved: {folder}/{filename}")


def _edge_label(ax, rl_mean, h_mean):
    delta  = float(np.mean(rl_mean - h_mean))
    col    = BETTER if delta >= 0 else WORSE
    sign   = "+" if delta > 0 else ""
    ax.text(0.98, 0.04, f"Avg RL edge: {sign}{delta:,.1f}  ({len(SEEDS)} seeds)",
            transform=ax.transAxes, ha="right", va="bottom",
            color=col, fontsize=9, fontweight="bold",
            bbox=dict(facecolor=BG, alpha=0.6, edgecolor="none", pad=3))


def _seed_legend(ax):
    """Add a compact seed colour legend."""
    handles = [plt.Line2D([0], [0], color=SEED_COLORS[i], lw=1.5,
                          label=f"Seed {SEEDS[i]}")
               for i in range(len(SEEDS))]
    handles += [
        plt.Line2D([0], [0], color=MEAN_COL, lw=2.5, label="RL mean (all seeds)"),
        plt.Line2D([0], [0], color=H_COL,    lw=2.0, ls="--", label="Heuristic mean"),
    ]
    ax.legend(handles=handles, fontsize=8, facecolor=PANEL, edgecolor=GRID,
              labelcolor=TEXT, loc="upper left", framealpha=0.75, ncol=2)


# ------------------------------------------------------------------ #
#  Per-model chart builders — SEEDS version (*_seeds.png)            #
# ------------------------------------------------------------------ #

def chart_profit_seeds(folder, label, res):
    steps  = np.arange(N_STEPS)
    rl_m   = res["rl_profit"]["mean"]
    h_m    = res["h_profit"]["mean"]

    fig, ax = _new_fig()
    # Individual RL seed lines
    for i, seed_data in enumerate(res["rl_profit"]["seeds"]):
        ax.plot(steps, seed_data, color=SEED_COLORS[i], lw=1.2, alpha=0.60, zorder=3)
    # Heuristic mean reference
    ax.plot(steps, h_m,  color=H_COL,   lw=2.0, ls="--", alpha=0.85, zorder=4)
    # RL mean on top
    ax.plot(steps, rl_m, color=MEAN_COL, lw=2.5, alpha=0.90, zorder=5)
    # Green/red fill between means
    ax.fill_between(steps, rl_m, h_m, where=(rl_m >= h_m), color=BETTER, alpha=0.15, zorder=2)
    ax.fill_between(steps, rl_m, h_m, where=(rl_m <  h_m), color=WORSE,  alpha=0.15, zorder=2)

    _edge_label(ax, rl_m, h_m)
    _seed_legend(ax)
    ax.set_title(f"Profit — Individual Seed Runs\n{label}\n"
                 f"Each colour = one seed's RL profit  |  White = mean  |  Dashed orange = heuristic mean",
                 fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Model Step", fontsize=10)
    ax.set_ylabel("Profit (THB)", fontsize=10)
    ax.set_xlim(0, N_STEPS - 1)
    _save(fig, folder, "profit_seeds.png")


def chart_profit_avg(folder, label, res):
    steps  = np.arange(N_STEPS)
    rl_m   = res["rl_profit"]["mean"]
    rl_s   = res["rl_profit"]["std"]
    h_m    = res["h_profit"]["mean"]
    h_s    = res["h_profit"]["std"]

    fig, ax = _new_fig()
    ax.fill_between(steps, h_m - h_s,  h_m + h_s,  color=H_COL,  alpha=0.15, zorder=2)
    ax.fill_between(steps, rl_m - rl_s, rl_m + rl_s, color=RL_COL, alpha=0.15, zorder=2)
    ax.plot(steps, h_m,  color=H_COL,  lw=1.8, alpha=0.85, label="Heuristic mean", zorder=3)
    ax.plot(steps, rl_m, color=RL_COL, lw=2.2, label="RL mean", zorder=4)
    ax.fill_between(steps, rl_m, h_m, where=(rl_m >= h_m), color=BETTER, alpha=0.18, zorder=1)
    ax.fill_between(steps, rl_m, h_m, where=(rl_m <  h_m), color=WORSE,  alpha=0.18, zorder=1)

    _edge_label(ax, rl_m, h_m)
    ax.set_title(f"Profit — Average Across {len(SEEDS)} Seeds\n{label}\n"
                 f"Solid line = mean  |  Shaded band = +/-1 std  |  Green fill = RL ahead  |  Red = behind",
                 fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Model Step", fontsize=10)
    ax.set_ylabel("Profit (THB)", fontsize=10)
    ax.set_xlim(0, N_STEPS - 1)
    ax.legend(fontsize=9, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT,
              loc="upper left", framealpha=0.7)
    _save(fig, folder, "profit_avg.png")


def chart_profit_delta_seeds(folder, label, res):
    steps = np.arange(N_STEPS)

    fig, ax = _new_fig()
    ax.axhline(0, color=DIM, lw=1.2, ls="--", zorder=3)

    for i, (rl_d, h_d) in enumerate(zip(res["rl_profit"]["seeds"], res["h_profit"]["seeds"])):
        delta = rl_d - h_d
        ax.plot(steps, delta, color=SEED_COLORS[i], lw=1.2, alpha=0.60, zorder=3)

    mean_delta = res["rl_profit"]["mean"] - res["h_profit"]["mean"]
    ax.plot(steps, mean_delta, color=MEAN_COL, lw=2.5, alpha=0.90, zorder=5,
            label="Mean gap (all seeds)")
    ax.fill_between(steps, mean_delta, 0, where=(mean_delta >= 0), color=BETTER, alpha=0.20, zorder=2)
    ax.fill_between(steps, mean_delta, 0, where=(mean_delta <  0), color=WORSE,  alpha=0.20, zorder=2)

    _seed_legend(ax)
    avg = float(np.mean(mean_delta))
    col = BETTER if avg >= 0 else WORSE
    ax.text(0.98, 0.04, f"Mean avg gap: {avg:+,.1f} THB",
            transform=ax.transAxes, ha="right", va="bottom",
            color=col, fontsize=9, fontweight="bold",
            bbox=dict(facecolor=BG, alpha=0.6, edgecolor="none", pad=3))

    ax.set_title(f"Profit Gap (RL - Heuristic) — Individual Seed Runs\n{label}\n"
                 f"Each colour = one seed  |  White = mean gap  |  Above 0 = RL winning that seed",
                 fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Model Step", fontsize=10)
    ax.set_ylabel("Profit Gap (THB)", fontsize=10)
    ax.set_xlim(0, N_STEPS - 1)
    _save(fig, folder, "profit_delta_seeds.png")


def chart_profit_delta_avg(folder, label, res):
    steps      = np.arange(N_STEPS)
    deltas     = np.stack([rl - h for rl, h in
                           zip(res["rl_profit"]["seeds"], res["h_profit"]["seeds"])], axis=0)
    mean = np.mean(deltas, axis=0)
    std  = np.std(deltas,  axis=0)

    fig, ax = _new_fig()
    ax.axhline(0, color=DIM, lw=1.2, ls="--", zorder=3)
    ax.fill_between(steps, mean - std, mean + std, color=TEXT, alpha=0.10, zorder=2, label="+/-1 std")
    ax.fill_between(steps, mean, 0, where=(mean >= 0), color=BETTER, alpha=0.35, zorder=3)
    ax.fill_between(steps, mean, 0, where=(mean <  0), color=WORSE,  alpha=0.35, zorder=3)
    ax.plot(steps, mean, color=MEAN_COL, lw=2.0, alpha=0.80, zorder=4, label="Mean gap")

    avg = float(np.mean(mean))
    col = BETTER if avg >= 0 else WORSE
    ax.text(0.98, 0.04, f"Overall avg: {avg:+,.1f} THB",
            transform=ax.transAxes, ha="right", va="bottom",
            color=col, fontsize=9, fontweight="bold",
            bbox=dict(facecolor=BG, alpha=0.6, edgecolor="none", pad=3))
    ax.set_title(f"Profit Gap — Average Across {len(SEEDS)} Seeds\n{label}\n"
                 f"Green = RL winning on average  |  Shaded band = spread across seeds",
                 fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Model Step", fontsize=10)
    ax.set_ylabel("Profit Gap (THB)", fontsize=10)
    ax.set_xlim(0, N_STEPS - 1)
    ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, framealpha=0.7)
    _save(fig, folder, "profit_delta_avg.png")


def chart_workers_seeds(folder, label, res):
    steps = np.arange(N_STEPS)
    rl_m  = res["rl_workers"]["mean"]
    h_m   = res["h_workers"]["mean"]

    fig, ax = _new_fig()
    for i, seed_data in enumerate(res["rl_workers"]["seeds"]):
        ax.plot(steps, seed_data, color=SEED_COLORS[i], lw=1.2, alpha=0.60, zorder=3)
    ax.plot(steps, h_m,  color=H_COL,   lw=2.0, ls="--", alpha=0.85, zorder=4)
    ax.plot(steps, rl_m, color=MEAN_COL, lw=2.5, alpha=0.90, zorder=5)
    ax.fill_between(steps, rl_m, h_m, where=(rl_m >= h_m), color=BETTER, alpha=0.15, zorder=2)
    ax.fill_between(steps, rl_m, h_m, where=(rl_m <  h_m), color=WORSE,  alpha=0.15, zorder=2)

    _edge_label(ax, rl_m, h_m)
    _seed_legend(ax)
    ax.set_title(f"Workers — Individual Seed Runs\n{label}\n"
                 f"Each colour = one seed's RL headcount  |  White = mean  |  Dashed orange = heuristic mean",
                 fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Model Step", fontsize=10)
    ax.set_ylabel("Avg Workers per Firm", fontsize=10)
    ax.set_xlim(0, N_STEPS - 1)
    _save(fig, folder, "workers_seeds.png")


def chart_workers_avg(folder, label, res):
    steps  = np.arange(N_STEPS)
    rl_m, rl_s = res["rl_workers"]["mean"], res["rl_workers"]["std"]
    h_m,  h_s  = res["h_workers"]["mean"],  res["h_workers"]["std"]

    fig, ax = _new_fig()
    ax.fill_between(steps, h_m - h_s,   h_m + h_s,   color=H_COL,  alpha=0.15, zorder=2)
    ax.fill_between(steps, rl_m - rl_s, rl_m + rl_s, color=RL_COL, alpha=0.15, zorder=2)
    ax.plot(steps, h_m,  color=H_COL,  lw=1.8, alpha=0.85, label="Heuristic mean", zorder=3)
    ax.plot(steps, rl_m, color=RL_COL, lw=2.2, label="RL mean", zorder=4)
    ax.fill_between(steps, rl_m, h_m, where=(rl_m >= h_m), color=BETTER, alpha=0.18, zorder=1)
    ax.fill_between(steps, rl_m, h_m, where=(rl_m <  h_m), color=WORSE,  alpha=0.18, zorder=1)

    _edge_label(ax, rl_m, h_m)
    ax.set_title(f"Workers — Average Across {len(SEEDS)} Seeds\n{label}\n"
                 f"Solid = mean  |  Shaded band = +/-1 std  |  Green = RL employing more  |  Red = fewer",
                 fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Model Step", fontsize=10)
    ax.set_ylabel("Avg Workers per Firm", fontsize=10)
    ax.set_xlim(0, N_STEPS - 1)
    ax.legend(fontsize=9, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT,
              loc="upper left", framealpha=0.7)
    _save(fig, folder, "workers_avg.png")


def chart_wage_seeds(folder, label, res):
    steps = np.arange(N_STEPS)
    rl_m  = res["rl_wage"]["mean"]
    h_m   = res["h_wage"]["mean"]
    mkt   = res["market_wage"]["mean"]

    fig, ax = _new_fig(w=16, h=7)
    for i, seed_data in enumerate(res["rl_wage"]["seeds"]):
        ax.plot(steps, seed_data, color=SEED_COLORS[i], lw=1.2, alpha=0.60, zorder=3)
    ax.plot(steps, h_m,  color=H_COL,   lw=2.0, ls="--", alpha=0.85, zorder=4)
    ax.plot(steps, mkt,  color=MAC_COL, lw=1.2, ls=":",  alpha=0.65, zorder=4,
            label="Market avg (all firms)")
    ax.plot(steps, rl_m, color=MEAN_COL, lw=2.5, alpha=0.90, zorder=5)
    ax.fill_between(steps, rl_m, h_m, where=(rl_m >= h_m), color=BETTER, alpha=0.15, zorder=2)
    ax.fill_between(steps, rl_m, h_m, where=(rl_m <  h_m), color=WORSE,  alpha=0.15, zorder=2)

    _edge_label(ax, rl_m, h_m)
    _seed_legend(ax)
    ax.set_title(f"Wage — Individual Seed Runs\n{label}\n"
                 f"Each colour = one seed's RL wage  |  White = mean  |  Dashed orange = heuristic mean  |  Dotted purple = market",
                 fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Model Step", fontsize=10)
    ax.set_ylabel("Monthly Wage (THB)", fontsize=10)
    ax.set_xlim(0, N_STEPS - 1)
    _save(fig, folder, "wage_seeds.png")


def chart_wage_avg(folder, label, res):
    steps  = np.arange(N_STEPS)
    rl_m, rl_s = res["rl_wage"]["mean"], res["rl_wage"]["std"]
    h_m,  h_s  = res["h_wage"]["mean"],  res["h_wage"]["std"]
    mkt        = res["market_wage"]["mean"]

    fig, ax = _new_fig(w=16, h=7)
    ax.fill_between(steps, h_m - h_s,   h_m + h_s,   color=H_COL,  alpha=0.15, zorder=2)
    ax.fill_between(steps, rl_m - rl_s, rl_m + rl_s, color=RL_COL, alpha=0.15, zorder=2)
    ax.plot(steps, h_m,  color=H_COL,   lw=1.8, alpha=0.85, label="Heuristic mean", zorder=3)
    ax.plot(steps, mkt,  color=MAC_COL, lw=1.2, ls=":", alpha=0.65, label="Market avg (all firms)", zorder=3)
    ax.plot(steps, rl_m, color=RL_COL,  lw=2.2, label="RL mean", zorder=4)
    ax.fill_between(steps, rl_m, h_m, where=(rl_m >= h_m), color=BETTER, alpha=0.18, zorder=1)
    ax.fill_between(steps, rl_m, h_m, where=(rl_m <  h_m), color=WORSE,  alpha=0.18, zorder=1)

    _edge_label(ax, rl_m, h_m)
    ax.set_title(f"Wage — Average Across {len(SEEDS)} Seeds\n{label}\n"
                 f"Solid = mean  |  Shaded band = +/-1 std  |  Green = RL paying more  |  Red = less",
                 fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Model Step", fontsize=10)
    ax.set_ylabel("Monthly Wage (THB)", fontsize=10)
    ax.set_xlim(0, N_STEPS - 1)
    ax.legend(fontsize=9, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT,
              loc="best", framealpha=0.7)
    _save(fig, folder, "wage_avg.png")


def chart_employment_seeds(folder, label, res):
    steps = np.arange(N_STEPS)
    mean  = res["employment_rate"]["mean"] * 100

    fig, ax = _new_fig()
    for i, seed_data in enumerate(res["employment_rate"]["seeds"]):
        ax.plot(steps, seed_data * 100, color=SEED_COLORS[i], lw=1.2, alpha=0.60, zorder=3)
    ax.plot(steps, mean, color=MEAN_COL, lw=2.5, alpha=0.90, zorder=5, label="Mean employment rate")
    ax.axhline(70, color=DIM, lw=1, ls="--", alpha=0.7)
    ax.text(steps[-1] * 0.02, 71.5, "70% reference", color=DIM, fontsize=8)

    handles_extra = [plt.Line2D([0], [0], color=SEED_COLORS[i], lw=1.5,
                                label=f"Seed {SEEDS[i]}") for i in range(len(SEEDS))]
    handles_extra.append(plt.Line2D([0], [0], color=MEAN_COL, lw=2.5, label="Mean"))
    ax.legend(handles=handles_extra, fontsize=8, facecolor=PANEL, edgecolor=GRID,
              labelcolor=TEXT, loc="lower right", framealpha=0.75, ncol=2)

    avg = float(np.mean(mean))
    col = BETTER if avg >= 70 else WORSE
    ax.text(0.98, 0.04, f"Mean avg: {avg:.1f}%",
            transform=ax.transAxes, ha="right", va="bottom",
            color=col, fontsize=9, fontweight="bold",
            bbox=dict(facecolor=BG, alpha=0.6, edgecolor="none", pad=3))

    ax.set_title(f"Employment Rate — Individual Seed Runs\n{label}\n"
                 f"Each colour = one seed  |  White = mean  |  Dashed = 70% reference",
                 fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Model Step", fontsize=10)
    ax.set_ylabel("Employment Rate (%)", fontsize=10)
    ax.set_xlim(0, N_STEPS - 1)
    ax.set_ylim(0, 105)
    _save(fig, folder, "employment_seeds.png")


def chart_employment_avg(folder, label, res):
    steps = np.arange(N_STEPS)
    mean  = res["employment_rate"]["mean"] * 100
    std   = res["employment_rate"]["std"]  * 100

    fig, ax = _new_fig()
    ax.fill_between(steps, mean - std, mean + std, color=BETTER, alpha=0.15, zorder=2,
                    label="+/-1 std across seeds")
    ax.fill_between(steps, mean, alpha=0.10, color=BETTER)
    ax.plot(steps, mean, color=BETTER, lw=2.2, label=f"Mean ({len(SEEDS)} seeds)", zorder=4)
    ax.axhline(70, color=DIM, lw=1, ls="--", alpha=0.7)
    ax.text(steps[-1] * 0.02, 71.5, "70% reference", color=DIM, fontsize=8)

    avg = float(np.mean(mean))
    col = BETTER if avg >= 70 else WORSE
    ax.text(0.98, 0.04, f"Avg: {avg:.1f}%",
            transform=ax.transAxes, ha="right", va="bottom",
            color=col, fontsize=9, fontweight="bold",
            bbox=dict(facecolor=BG, alpha=0.6, edgecolor="none", pad=3))
    ax.set_title(f"Employment Rate — Average Across {len(SEEDS)} Seeds\n{label}\n"
                 f"Shaded band = spread across seeds  |  Dashed = 70% reference",
                 fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Model Step", fontsize=10)
    ax.set_ylabel("Employment Rate (%)", fontsize=10)
    ax.set_xlim(0, N_STEPS - 1)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, framealpha=0.7)
    _save(fig, folder, "employment_avg.png")


def chart_scorecard(folder, label, res):
    specs = [
        ("rl_profit",  "h_profit",  "Profit",  True),
        ("rl_workers", "h_workers", "Workers", True),
        ("rl_wage",    "h_wage",    "Wage",    True),
    ]
    m_names, pcts, errs, colors = [], [], [], []
    for rk, hk, mname, hi in specs:
        ra  = float(np.mean(res[rk]["mean"]))
        ha  = float(np.mean(res[hk]["mean"]))
        pct = ((ra - ha) / abs(ha) * 100) if ha != 0 else 0.0
        delta_seeds = [np.mean(rl - h) for rl, h in zip(res[rk]["seeds"], res[hk]["seeds"])]
        err = float(np.std(delta_seeds) / abs(ha) * 100) if ha != 0 else 0.0
        if not hi:
            pct = -pct
        m_names.append(mname)
        pcts.append(pct)
        errs.append(err)
        colors.append(BETTER if pct >= 0 else WORSE)

    fig, ax = _new_fig(w=10, h=5)
    y    = np.arange(len(m_names))
    bars = ax.barh(y, pcts, color=colors, height=0.45,
                   edgecolor=GRID, linewidth=0.5, xerr=errs,
                   error_kw=dict(ecolor=TEXT, capsize=5, elinewidth=1.5))
    ax.axvline(0, color=DIM, lw=1.5, zorder=5)

    xlim = max(abs(p) + e for p, e in zip(pcts, errs)) * 1.5 if pcts else 10
    for bar, val, err in zip(bars, pcts, errs):
        ha  = "left"  if val >= 0 else "right"
        off = xlim * 0.025
        ax.text(val + (off if val >= 0 else -off),
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.1f}%  (+/-{err:.1f}%)", ha=ha, va="center",
                color="white", fontsize=9, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(m_names, color=TEXT, fontsize=11)
    ax.set_xlabel(f"RL advantage vs heuristic (%)  —  error bars = std across {len(SEEDS)} seeds",
                  fontsize=9, color=TEXT)
    ax.set_xlim(-xlim, xlim)
    ax.axvspan(   0, xlim, color=BETTER, alpha=0.05)
    ax.axvspan(-xlim, 0,   color=WORSE,  alpha=0.05)
    ax.set_title(f"RL Effect Scorecard  (avg over {len(SEEDS)} seeds)\n{label}\n"
                 f"Green = positive RL effect  |  Red = negative  |  Error bars = variability",
                 fontsize=11, fontweight="bold", pad=8)
    _save(fig, folder, "scorecard.png")


# ------------------------------------------------------------------ #
#  Cross-model comparison charts                                      #
# ------------------------------------------------------------------ #

def chart_comparison_summary(results, active_names):
    specs = [
        ("rl_profit",  "h_profit",  "Avg Profit per Firm (THB)"),
        ("rl_workers", "h_workers", "Avg Workers per Firm"),
        ("rl_wage",    "h_wage",    "Avg Monthly Wage (THB)"),
    ]
    labels = [n.upper() for n in active_names]
    x      = np.arange(len(active_names))
    bar_w  = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(21, 7), facecolor=BG)
    fig.suptitle(f"Cross-Model Summary  —  Mean over {len(SEEDS)} Seeds x 360 Steps\n"
                 "Blue = RL  |  Orange = Heuristic  |  Error bars = std across seeds",
                 color="white", fontsize=15, fontweight="bold", y=1.02)

    for ax, (rk, hk, ylabel) in zip(axes, specs):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT, labelsize=10)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)
        ax.grid(axis="y", color=GRID, alpha=0.35, lw=0.5)

        rl_vals, h_vals, rl_errs, h_errs = [], [], [], []
        for name in active_names:
            res = results.get(name)
            if res is not None:
                rls = [float(np.mean(s)) for s in res[rk]["seeds"]]
                hs  = [float(np.mean(s)) for s in res[hk]["seeds"]]
                rl_vals.append(float(np.mean(rls))); rl_errs.append(float(np.std(rls)))
                h_vals.append(float(np.mean(hs)));   h_errs.append(float(np.std(hs)))
            else:
                rl_vals.append(0.0); h_vals.append(0.0)
                rl_errs.append(0.0); h_errs.append(0.0)

        ekw = dict(elinewidth=1.5, capsize=4, ecolor=TEXT)
        bars_rl = ax.bar(x - bar_w/2, rl_vals, bar_w, color=RL_COL,
                         label="RL", edgecolor=GRID, linewidth=0.5,
                         yerr=rl_errs, error_kw=ekw)
        bars_h  = ax.bar(x + bar_w/2, h_vals,  bar_w, color=H_COL,
                         label="Heuristic", edgecolor=GRID, linewidth=0.5,
                         yerr=h_errs, error_kw=ekw)

        ref = max(abs(v) for v in rl_vals + h_vals) * 0.015
        for bar, val in zip(bars_rl, rl_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ref,
                    f"{val:,.0f}", ha="center", va="bottom",
                    color=TEXT, fontsize=8, fontweight="bold")
        for bar, val in zip(bars_h, h_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ref,
                    f"{val:,.0f}", ha="center", va="bottom", color=TEXT, fontsize=8)
        for i, (rv, hv) in enumerate(zip(rl_vals, h_vals)):
            if rv > hv:
                ax.text(x[i] - bar_w/2, max(rv, hv) * 1.10, "[RL wins]",
                        ha="center", va="bottom", color=BETTER, fontsize=8, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, color=TEXT, fontsize=11)
        ax.set_ylabel(ylabel, color=TEXT, fontsize=10)
        ax.yaxis.label.set_color(TEXT)
        ax.legend(fontsize=9, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    plt.tight_layout()
    _save(fig, "comparison", "summary.png")


def chart_comparison_scorecard(results, active_names, model_labels):
    specs = [
        ("rl_profit",  "h_profit",  "Profit",  True),
        ("rl_workers", "h_workers", "Workers", True),
        ("rl_wage",    "h_wage",    "Wage",    True),
    ]
    fig, axes = plt.subplots(1, len(active_names),
                             figsize=(7 * len(active_names), 6), facecolor=BG)
    if len(active_names) == 1:
        axes = [axes]
    fig.suptitle(f"RL Effect Scorecard — All Models  (avg over {len(SEEDS)} seeds)\n"
                 "Green = positive  |  Red = negative  |  Error bars = variability across seeds",
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

        m_names, pcts, errs, cols = [], [], [], []
        for rk, hk, mname, hi in specs:
            ra  = float(np.mean(res[rk]["mean"]))
            ha  = float(np.mean(res[hk]["mean"]))
            pct = ((ra - ha) / abs(ha) * 100) if ha != 0 else 0.0
            delta_seeds = [np.mean(rl - h) for rl, h in zip(res[rk]["seeds"], res[hk]["seeds"])]
            err = float(np.std(delta_seeds) / abs(ha) * 100) if ha != 0 else 0.0
            if not hi:
                pct = -pct
            m_names.append(mname); pcts.append(pct); errs.append(err)
            cols.append(BETTER if pct >= 0 else WORSE)

        y    = np.arange(len(m_names))
        ax.barh(y, pcts, color=cols, height=0.45, edgecolor=GRID, linewidth=0.5,
                xerr=errs, error_kw=dict(ecolor=TEXT, capsize=5, elinewidth=1.5))
        ax.axvline(0, color=DIM, lw=1.5, zorder=5)

        xlim = max(abs(p) + e for p, e in zip(pcts, errs)) * 1.5 if pcts else 10
        for p, e, yi in zip(pcts, errs, y):
            ha = "left" if p >= 0 else "right"
            off = xlim * 0.025
            ax.text(p + (off if p >= 0 else -off), yi,
                    f"{p:+.1f}%", ha=ha, va="center",
                    color="white", fontsize=10, fontweight="bold")

        ax.set_yticks(y)
        ax.set_yticklabels(m_names, color=TEXT, fontsize=11)
        ax.set_xlabel("RL advantage (%)", color=TEXT, fontsize=10)
        ax.set_xlim(-xlim, xlim)
        ax.axvspan(   0, xlim, color=BETTER, alpha=0.05)
        ax.axvspan(-xlim, 0,   color=WORSE,  alpha=0.05)
        ax.set_title(model_labels[name], color="white", fontsize=10,
                     fontweight="bold", pad=8)
        ax.grid(axis="x", color=GRID, alpha=0.35, lw=0.5)

    plt.tight_layout()
    _save(fig, "comparison", "scorecard.png")


# ------------------------------------------------------------------ #
#  Folder / label maps                                                #
# ------------------------------------------------------------------ #

FOLDER_NAME = {"solo": "solo", "coop": "cooperative", "comp": "competitive"}
MODEL_LABELS = {
    "solo": "Solo  (1 RL firm vs 9 heuristic)",
    "coop": "Cooperative  (3 RL firms, shared reward)",
    "comp": "Competitive  (3 RL firms, relative reward)",
}
MODEL_NAMES = ["solo", "coop", "comp"]


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    print(f"\nSeeds: {SEEDS}  ({len(SEEDS)} runs per scenario)")
    print(f"Seed colours: {dict(zip(SEEDS, SEED_COLORS))}")
    print(f"Output: {VIZ_DIR}\n")

    results = {}
    for key in MODEL_NAMES:
        print(f"[{MODEL_LABELS[key]}]")
        results[key] = run_multi_seed(key)

    available = [k for k, v in results.items() if v is not None]
    if not available:
        print("No models found. Exiting.")
        sys.exit(1)

    print(f"\nGenerating charts for: {', '.join(available)}\n")

    for key in available:
        res    = results[key]
        folder = FOLDER_NAME[key]
        label  = MODEL_LABELS[key]
        print(f"  [{label}]")
        chart_profit_seeds(folder, label, res)
        chart_profit_avg(folder, label, res)
        chart_profit_delta_seeds(folder, label, res)
        chart_profit_delta_avg(folder, label, res)
        chart_workers_seeds(folder, label, res)
        chart_workers_avg(folder, label, res)
        chart_wage_seeds(folder, label, res)
        chart_wage_avg(folder, label, res)
        chart_employment_seeds(folder, label, res)
        chart_employment_avg(folder, label, res)
        chart_scorecard(folder, label, res)

    print("\n  [comparison]")
    active_labels = {n: MODEL_LABELS[n] for n in available}
    chart_comparison_summary(results, available)
    chart_comparison_scorecard(results, available, active_labels)

    print(f"\nDone. Charts saved to: {VIZ_DIR}")
    print(f"\nPer model folder (x3): 10 charts + scorecard")
    print(f"  *_seeds.png  — individual coloured runs  (sky blue=42, red=123, green=456, amber=789, purple=1000)")
    print(f"  *_avg.png    — mean +/- 1 std band")
    print(f"\ncomparison/: summary.png  scorecard.png")
