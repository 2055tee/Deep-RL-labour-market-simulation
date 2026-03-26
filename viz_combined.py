#!/usr/bin/env python
# viz_combined.py
#
# All-in-one multi-seed visualization for all three RL scenarios.
# Change N to run any number of seeds.
#
# Per-model output  (visualizations/<model>/):
#   {metric}_seeds.png  — 1 big figure with N subplots, one panel per seed (RL vs heuristic)
#   {metric}_avg.png    — mean ± 1-std across all N seeds
#   scorecard.png
#
# Solo model only (action-level analysis):
#   actions_rl.png, actions_heuristic.png, divergence_map.png, outcome_summary.png
#
# Cross-model (visualizations/comparison/):
#   summary.png, scorecard.png

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
N = 5          # ← number of seeds; change to any positive integer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import sys
import random
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

ROOT    = Path(__file__).parent.resolve()
VIZ_DIR = ROOT / "visualizations"

N_STEPS    = 360
N_RL_FIRMS = 3

# ── Seed pool (first N are used) ─────────────────────────────────────
_SEED_POOL = [42, 123, 456, 789, 1000, 7, 13, 99, 2025, 314,
              271, 161, 1337, 500, 999, 888, 77, 333, 4242, 9999]
SEEDS = (_SEED_POOL + list(range(10000, 10000 + max(0, N - len(_SEED_POOL)))))[:N]

# ── Colour palette ───────────────────────────────────────────────────
BG       = "#0f0f1a"
PANEL    = "#16213e"
GRID     = "#1e2a4a"
TEXT     = "#d0d0e8"
DIM      = "#888899"
BETTER   = "#00c853"
WORSE    = "#ff1744"
RL_COL   = "#4fc3f7"
H_COL    = "#ffb74d"
MAC_COL  = "#ce93d8"
MEAN_COL = "#ffffff"

_COLOR_POOL = [
    "#4fc3f7", "#ef5350", "#66bb6a", "#ffa726", "#ab47bc",
    "#26c6da", "#ec407a", "#9ccc65", "#ff7043", "#5c6bc0",
    "#26a69a", "#d4e157", "#ff8a65", "#7e57c2", "#29b6f6",
    "#f06292", "#aed581", "#80deea", "#ffcc02", "#a78bfa",
]

def _seed_colors(n):
    if n <= len(_COLOR_POOL):
        return _COLOR_POOL[:n]
    cmap = plt.get_cmap("tab20", n)
    return [matplotlib.colors.to_hex(cmap(i)) for i in range(n)]

SEED_COLORS = _seed_colors(N)

# ── Action maps (solo model only) ────────────────────────────────────
ACTION_NAMES  = {0: "Hold", 1: "Wage +300", 2: "Wage +100",
                 3: "Wage -100", 4: "Wage -300", 5: "Post Vacancy", 6: "Fire Worker"}
ACTION_COLORS = {0: "#555577", 1: "#1565c0", 2: "#64b5f6",
                 3: "#ffa726", 4: "#e53935", 5: "#43a047", 6: "#8e24aa"}
OUT_BETTER  = "#00e676"
OUT_WORSE   = "#ff1744"
OUT_NEUTRAL = "#ffd740"
OUT_AGREE   = "#333355"


# ── Scenario loading helpers ─────────────────────────────────────────

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


# ── Step data collector ──────────────────────────────────────────────

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


# ── Single-seed runners ──────────────────────────────────────────────

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


# ── Multi-seed aggregator ────────────────────────────────────────────

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
        print(f"    seed {seed}  ({i + 1}/{N})")
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
            "seeds": [r[k] for r in all_runs],
        }
    return result


# ── Heuristic action shadow (solo model only) ────────────────────────

def heuristic_action(firm, step):
    current_profit = firm.compute_profit()
    if firm.current_workers:
        if firm.compute_profit(labor_override=len(firm.current_workers) - 1) > current_profit:
            return 6
    labor = len(firm.current_workers)
    mpl   = firm.marginal_product_labor(firm.productivity, labor + 1, firm.alpha)
    if firm.output_price * mpl >= firm.monthly_wage:
        return 5
    if step % 12 == 0:
        if firm.vacancies > 0 and firm.vacancy_duration > 0:
            return 1
        profit_up = firm.compute_profit(wage=max(int(firm.monthly_wage * 1.02), firm.monthly_wage + 1))
        profit_dn = firm.compute_profit(wage=max(int(firm.monthly_wage * 0.98), firm.wage_floor()))
        if profit_up > current_profit and profit_up >= profit_dn:
            return 2
        if profit_dn > current_profit:
            return 3
    return 0


def run_solo_episode_with_actions(seed):
    """Run one solo episode tracking both RL and heuristic actions per step."""
    random.seed(seed); np.random.seed(seed)
    d = _load_scenario("solo")
    from sb3_contrib import MaskablePPO
    from firm_env import LaborMarketEnv
    policy = MaskablePPO.load(str(d / "solo_model"))
    env    = LaborMarketEnv()
    obs, _ = env.reset()
    records = []
    for step in range(N_STEPS):
        firm        = env.rl_firm
        pre_profit  = float(firm.profit)
        pre_wage    = int(firm.monthly_wage)
        pre_workers = len(firm.current_workers)
        mask       = env.action_masks()
        act_arr, _ = policy.predict(obs[np.newaxis], deterministic=True,
                                    action_masks=mask[np.newaxis])
        rl_act = int(act_arr[0])
        h_act  = heuristic_action(firm, step)
        obs, _, _, _, _ = env.step(rl_act)
        records.append({
            "step":      step,
            "rl_action": rl_act,
            "h_action":  h_act,
            "diverge":   rl_act != h_act,
            "profit":    float(env.rl_firm.profit),
            "delta":     float(env.rl_firm.profit) - pre_profit,
            "wage":      pre_wage,
            "workers":   pre_workers,
        })
    _unload_scenario("solo")
    return records


def classify_actions(records):
    agree_deltas = [r["delta"] for r in records if not r["diverge"]]
    med = float(np.median(agree_deltas)) if agree_deltas else 0.0
    std = float(np.std(agree_deltas))    if agree_deltas else 1.0
    thr = max(std * 0.5, 300.0)
    for r in records:
        if not r["diverge"]:
            r["outcome"] = "agree"
        elif r["delta"] > med + thr:
            r["outcome"] = "better"
        elif r["delta"] < med - thr:
            r["outcome"] = "worse"
        else:
            r["outcome"] = "neutral"


# ── Drawing primitives ───────────────────────────────────────────────

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


def _style_ax(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color("white")
    ax.grid(color=GRID, alpha=0.30, lw=0.5, zorder=0)


def _save(fig, folder, filename):
    out = VIZ_DIR / folder
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved: {folder}/{filename}")


def _edge_label(ax, rl_mean, h_mean):
    delta = float(np.mean(rl_mean - h_mean))
    col   = BETTER if delta >= 0 else WORSE
    sign  = "+" if delta > 0 else ""
    ax.text(0.98, 0.04, f"Avg RL edge: {sign}{delta:,.1f}  ({N} seeds)",
            transform=ax.transAxes, ha="right", va="bottom",
            color=col, fontsize=9, fontweight="bold",
            bbox=dict(facecolor=BG, alpha=0.6, edgecolor="none", pad=3))


def _grid_layout(n):
    """Compact (nrows, ncols) grid for n subplots."""
    if n <= 3: return 1, n
    if n <= 4: return 2, 2
    if n <= 6: return 2, 3
    if n <= 9: return 3, 3
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    return nrows, ncols


# ── Seeds chart: one big figure, N panels (one per seed) ────────────

def chart_metric_seeds(folder, label, res, rl_key, h_key, ylabel, filename,
                       extra_line=None):
    """1 figure × N subplots.  Each panel = one seed's RL vs heuristic."""
    nrows, ncols = _grid_layout(N)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(7 * ncols, 4.5 * nrows),
                             facecolor=BG, squeeze=False)
    fig.suptitle(f"{ylabel} — RL vs Heuristic per Seed\n"
                 f"{label}  ({N} seeds × {N_STEPS} steps)",
                 color="white", fontsize=13, fontweight="bold", y=1.02)

    steps = np.arange(N_STEPS)
    for idx in range(nrows * ncols):
        r, c = divmod(idx, ncols)
        ax   = axes[r][c]
        if idx < N:
            _style_ax(ax)
            rl = res[rl_key]["seeds"][idx]
            h  = res[h_key]["seeds"][idx]
            ax.plot(steps, h,  color=H_COL,           lw=1.6, ls="--", alpha=0.85, label="Heuristic")
            ax.plot(steps, rl, color=SEED_COLORS[idx], lw=1.8, label=f"RL")
            if extra_line:
                ax.plot(steps, res[extra_line]["seeds"][idx],
                        color=MAC_COL, lw=1.0, ls=":", alpha=0.65, label="Market avg")
            ax.fill_between(steps, rl, h, where=(rl >= h), color=BETTER, alpha=0.15)
            ax.fill_between(steps, rl, h, where=(rl <  h), color=WORSE,  alpha=0.15)
            delta = float(np.mean(rl - h))
            ax.set_title(f"Seed {SEEDS[idx]}  (edge: {delta:+,.1f})",
                         color="white", fontsize=9, pad=4)
            ax.set_xlabel("Step", fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.set_xlim(0, N_STEPS - 1)
            ax.legend(fontsize=7, facecolor=PANEL, edgecolor=GRID,
                      labelcolor=TEXT, loc="best", framealpha=0.7)
        else:
            ax.set_visible(False)

    plt.tight_layout()
    _save(fig, folder, filename)


# ── Average chart: mean ± std ────────────────────────────────────────

def chart_metric_avg(folder, label, res, rl_key, h_key, ylabel, filename,
                     extra_line=None):
    steps  = np.arange(N_STEPS)
    rl_m, rl_s = res[rl_key]["mean"], res[rl_key]["std"]
    h_m,  h_s  = res[h_key]["mean"],  res[h_key]["std"]

    fig, ax = _new_fig()
    ax.fill_between(steps, h_m  - h_s,  h_m  + h_s,  color=H_COL,  alpha=0.15, zorder=2)
    ax.fill_between(steps, rl_m - rl_s, rl_m + rl_s, color=RL_COL, alpha=0.15, zorder=2)
    ax.plot(steps, h_m,  color=H_COL,  lw=1.8, alpha=0.85, label="Heuristic mean", zorder=3)
    if extra_line:
        ax.plot(steps, res[extra_line]["mean"],
                color=MAC_COL, lw=1.2, ls=":", alpha=0.65,
                label="Market avg (all firms)", zorder=3)
    ax.plot(steps, rl_m, color=RL_COL, lw=2.2, label="RL mean", zorder=4)
    ax.fill_between(steps, rl_m, h_m, where=(rl_m >= h_m), color=BETTER, alpha=0.18, zorder=1)
    ax.fill_between(steps, rl_m, h_m, where=(rl_m <  h_m), color=WORSE,  alpha=0.18, zorder=1)

    _edge_label(ax, rl_m, h_m)
    ax.set_title(f"{ylabel} — Average Across {N} Seeds\n{label}\n"
                 f"Solid = mean  |  Shaded = ±1 std  |  Green = RL ahead  |  Red = behind",
                 fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Model Step", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xlim(0, N_STEPS - 1)
    ax.legend(fontsize=9, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT,
              loc="best", framealpha=0.7)
    _save(fig, folder, filename)


# ── Profit delta ─────────────────────────────────────────────────────

def chart_profit_delta_seeds(folder, label, res):
    nrows, ncols = _grid_layout(N)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(7 * ncols, 4.5 * nrows),
                             facecolor=BG, squeeze=False)
    fig.suptitle(f"Profit Gap (RL − Heuristic) — Each Seed\n{label}",
                 color="white", fontsize=13, fontweight="bold", y=1.02)

    steps = np.arange(N_STEPS)
    for idx in range(nrows * ncols):
        r, c = divmod(idx, ncols)
        ax   = axes[r][c]
        if idx < N:
            _style_ax(ax)
            delta = res["rl_profit"]["seeds"][idx] - res["h_profit"]["seeds"][idx]
            ax.axhline(0, color=DIM, lw=1.0, ls="--")
            ax.plot(steps, delta, color=SEED_COLORS[idx], lw=1.8)
            ax.fill_between(steps, delta, 0, where=(delta >= 0), color=BETTER, alpha=0.25)
            ax.fill_between(steps, delta, 0, where=(delta <  0), color=WORSE,  alpha=0.25)
            avg = float(np.mean(delta))
            ax.set_title(f"Seed {SEEDS[idx]}  (avg: {avg:+,.1f})",
                         color="white", fontsize=9, pad=4)
            ax.set_xlabel("Step", fontsize=8)
            ax.set_ylabel("Profit Gap (THB)", fontsize=8)
            ax.set_xlim(0, N_STEPS - 1)
        else:
            ax.set_visible(False)

    plt.tight_layout()
    _save(fig, folder, "profit_delta_seeds.png")


def chart_profit_delta_avg(folder, label, res):
    steps  = np.arange(N_STEPS)
    deltas = np.stack([rl - h for rl, h in
                       zip(res["rl_profit"]["seeds"], res["h_profit"]["seeds"])], axis=0)
    mean = np.mean(deltas, axis=0)
    std  = np.std(deltas,  axis=0)

    fig, ax = _new_fig()
    ax.axhline(0, color=DIM, lw=1.2, ls="--", zorder=3)
    ax.fill_between(steps, mean - std, mean + std, color=TEXT, alpha=0.10, zorder=2, label="±1 std")
    ax.fill_between(steps, mean, 0, where=(mean >= 0), color=BETTER, alpha=0.35, zorder=3)
    ax.fill_between(steps, mean, 0, where=(mean <  0), color=WORSE,  alpha=0.35, zorder=3)
    ax.plot(steps, mean, color=MEAN_COL, lw=2.0, alpha=0.80, zorder=4, label="Mean gap")

    avg = float(np.mean(mean))
    col = BETTER if avg >= 0 else WORSE
    ax.text(0.98, 0.04, f"Overall avg: {avg:+,.1f} THB",
            transform=ax.transAxes, ha="right", va="bottom",
            color=col, fontsize=9, fontweight="bold",
            bbox=dict(facecolor=BG, alpha=0.6, edgecolor="none", pad=3))
    ax.set_title(f"Profit Gap (RL − Heuristic) — Average Across {N} Seeds\n{label}",
                 fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Model Step", fontsize=10)
    ax.set_ylabel("Profit Gap (THB)", fontsize=10)
    ax.set_xlim(0, N_STEPS - 1)
    ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, framealpha=0.7)
    _save(fig, folder, "profit_delta_avg.png")


# ── Employment ───────────────────────────────────────────────────────

def chart_employment_seeds(folder, label, res):
    nrows, ncols = _grid_layout(N)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(7 * ncols, 4.5 * nrows),
                             facecolor=BG, squeeze=False)
    fig.suptitle(f"Employment Rate — Each Seed\n{label}",
                 color="white", fontsize=13, fontweight="bold", y=1.02)

    steps = np.arange(N_STEPS)
    for idx in range(nrows * ncols):
        r, c = divmod(idx, ncols)
        ax   = axes[r][c]
        if idx < N:
            _style_ax(ax)
            emp = res["employment_rate"]["seeds"][idx] * 100
            ax.plot(steps, emp, color=SEED_COLORS[idx], lw=1.8)
            ax.axhline(70, color=DIM, lw=1.0, ls="--", alpha=0.7)
            ax.text(N_STEPS * 0.02, 72, "70%", color=DIM, fontsize=7)
            avg = float(np.mean(emp))
            ax.set_title(f"Seed {SEEDS[idx]}  (avg: {avg:.1f}%)",
                         color="white", fontsize=9, pad=4)
            ax.set_xlabel("Step", fontsize=8)
            ax.set_ylabel("Employment Rate (%)", fontsize=8)
            ax.set_xlim(0, N_STEPS - 1)
            ax.set_ylim(0, 105)
        else:
            ax.set_visible(False)

    plt.tight_layout()
    _save(fig, folder, "employment_seeds.png")


def chart_employment_avg(folder, label, res):
    steps = np.arange(N_STEPS)
    mean  = res["employment_rate"]["mean"] * 100
    std   = res["employment_rate"]["std"]  * 100

    fig, ax = _new_fig()
    ax.fill_between(steps, mean - std, mean + std, color=BETTER, alpha=0.15, zorder=2,
                    label="±1 std across seeds")
    ax.fill_between(steps, mean, alpha=0.10, color=BETTER)
    ax.plot(steps, mean, color=BETTER, lw=2.2, label=f"Mean ({N} seeds)", zorder=4)
    ax.axhline(70, color=DIM, lw=1, ls="--", alpha=0.7)
    ax.text(N_STEPS * 0.02, 71.5, "70% reference", color=DIM, fontsize=8)

    avg = float(np.mean(mean))
    col = BETTER if avg >= 70 else WORSE
    ax.text(0.98, 0.04, f"Avg: {avg:.1f}%",
            transform=ax.transAxes, ha="right", va="bottom",
            color=col, fontsize=9, fontweight="bold",
            bbox=dict(facecolor=BG, alpha=0.6, edgecolor="none", pad=3))
    ax.set_title(f"Employment Rate — Average Across {N} Seeds\n{label}",
                 fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Model Step", fontsize=10)
    ax.set_ylabel("Employment Rate (%)", fontsize=10)
    ax.set_xlim(0, N_STEPS - 1)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, framealpha=0.7)
    _save(fig, folder, "employment_avg.png")


# ── Scorecard ────────────────────────────────────────────────────────

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
        m_names.append(mname); pcts.append(pct); errs.append(err)
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
                f"{val:+.1f}%  (±{err:.1f}%)", ha=ha, va="center",
                color="white", fontsize=9, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(m_names, color=TEXT, fontsize=11)
    ax.set_xlabel(f"RL advantage vs heuristic (%)  —  error bars = std across {N} seeds",
                  fontsize=9, color=TEXT)
    ax.set_xlim(-xlim, xlim)
    ax.axvspan(   0, xlim, color=BETTER, alpha=0.05)
    ax.axvspan(-xlim, 0,   color=WORSE,  alpha=0.05)
    ax.set_title(f"RL Effect Scorecard  (avg over {N} seeds)\n{label}",
                 fontsize=11, fontweight="bold", pad=8)
    _save(fig, folder, "scorecard.png")


# ── Solo action charts (from viz_compare) ────────────────────────────

def chart_action_strip(records, agent_key, label, filename, ref_seed):
    actions = [r[agent_key] for r in records]
    divs    = {r["step"] for r in records if r["diverge"]}

    fig, ax = plt.subplots(figsize=(20, 3), facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=10)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)

    for s, act in enumerate(actions):
        ax.bar(s, 1, color=ACTION_COLORS[act], width=1.0, align="edge")
    for s in divs:
        ax.bar(s, 1, color="none", width=1.0, align="edge",
               edgecolor="white", linewidth=1.0)

    ax.set_xlim(0, N_STEPS)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel(f"Model Step  (white border = differs from other agent) — seed {ref_seed}",
                  color=TEXT, fontsize=11)
    ax.set_title(f"Action at Every Step — {label}",
                 color="white", fontsize=13, fontweight="bold", pad=10)

    patches = [mpatches.Patch(color=ACTION_COLORS[k], label=ACTION_NAMES[k]) for k in range(7)]
    ax.legend(handles=patches, ncol=7, facecolor=PANEL, edgecolor=GRID,
              labelcolor=TEXT, fontsize=9, loc="upper center",
              bbox_to_anchor=(0.5, -0.35))
    _save(fig, "solo", filename)


def chart_divergence_map(records, ref_seed):
    COLS, ROWS = 60, 6
    outcome_color = {"agree": OUT_AGREE, "better": OUT_BETTER,
                     "worse": OUT_WORSE, "neutral": OUT_NEUTRAL}

    fig, ax = plt.subplots(figsize=(16, 5), facecolor=BG)
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])

    pad = 0.06
    for r in records:
        s   = r["step"]
        col = s % COLS
        row = ROWS - 1 - (s // COLS)
        x, y = col * 1.0, row * 1.0
        rect = plt.Rectangle((x + pad, y + pad), 1 - 2 * pad, 1 - 2 * pad,
                              color=outcome_color[r["outcome"]])
        ax.add_patch(rect)
        if s % 10 == 0:
            ax.text(x + 0.5, y + 0.5, str(s), ha="center", va="center",
                    fontsize=6.5, color="white", alpha=0.7)

    ax.set_xlim(0, COLS); ax.set_ylim(0, ROWS)
    ax.set_aspect("equal")
    for row_i in range(ROWS):
        start = (ROWS - 1 - row_i) * COLS
        ax.text(-1.5, row_i + 0.5, f"Steps\n{start}-{start + COLS - 1}",
                ha="right", va="center", color=TEXT, fontsize=8)

    ax.set_title(f"Divergence Outcome Map — Seed {ref_seed}",
                 color="white", fontsize=14, fontweight="bold", pad=12)
    handles = [
        mpatches.Patch(color=OUT_AGREE,   label="Agreed with rule"),
        mpatches.Patch(color=OUT_BETTER,  label="Diverged → better"),
        mpatches.Patch(color=OUT_WORSE,   label="Diverged → worse"),
        mpatches.Patch(color=OUT_NEUTRAL, label="Diverged → neutral"),
    ]
    ax.legend(handles=handles, facecolor=BG, edgecolor=GRID, labelcolor=TEXT,
              fontsize=10, loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=4)
    _save(fig, "solo", "divergence_map.png")


def chart_outcome_summary(all_action_records):
    from collections import Counter
    flat = [r for recs in all_action_records for r in recs]
    divs = [r for r in flat if r["diverge"]]
    rl_c = Counter(r["rl_action"] for r in flat)
    h_c  = Counter(r["h_action"]  for r in flat)

    outcomes = {
        "Agreed\nwith rule": sum(1 for r in flat if not r["diverge"]),
        "Diverged\nBetter":  sum(1 for r in divs if r["outcome"] == "better"),
        "Diverged\nNeutral": sum(1 for r in divs if r["outcome"] == "neutral"),
        "Diverged\nWorse":   sum(1 for r in divs if r["outcome"] == "worse"),
    }
    colors_out = [OUT_AGREE, OUT_BETTER, OUT_NEUTRAL, OUT_WORSE]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=BG)
    fig.suptitle(f"Decision Summary — RL vs Heuristic  ({N} seeds × {N_STEPS} steps)",
                 color="white", fontsize=15, fontweight="bold", y=1.01)

    for ax in axes:
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)
        ax.tick_params(colors=TEXT, labelsize=9)

    bars = axes[0].bar(outcomes.keys(), outcomes.values(), color=colors_out,
                       width=0.55, edgecolor=GRID, linewidth=0.5)
    for bar, val in zip(bars, outcomes.values()):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                     str(val), ha="center", va="bottom", color="white",
                     fontsize=12, fontweight="bold")
    axes[0].set_title("Divergence Outcomes\n(all seeds combined)",
                      color="white", fontsize=12, pad=10)
    axes[0].set_ylabel("Steps", color=TEXT, fontsize=10)
    axes[0].set_ylim(0, max(outcomes.values()) * 1.2)

    acts = [ACTION_NAMES[k] for k in range(7)]
    cols = [ACTION_COLORS[k] for k in range(7)]

    rl_v  = [rl_c.get(k, 0) for k in range(7)]
    bars2 = axes[1].bar(acts, rl_v, color=cols, edgecolor=GRID, linewidth=0.5)
    for bar, val in zip(bars2, rl_v):
        if val > 0:
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                         str(val), ha="center", va="bottom", color="white", fontsize=9)
    axes[1].set_title("RL Agent — Action Frequency\n(all seeds)",
                      color="white", fontsize=12, pad=10)
    axes[1].set_ylabel("Times Chosen", color=TEXT, fontsize=10)
    plt.setp(axes[1].get_xticklabels(), rotation=30, ha="right")

    h_v   = [h_c.get(k, 0) for k in range(7)]
    bars3 = axes[2].bar(acts, h_v, color=cols, edgecolor=GRID, linewidth=0.5)
    for bar, val in zip(bars3, h_v):
        if val > 0:
            axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                         str(val), ha="center", va="bottom", color="white", fontsize=9)
    axes[2].set_title("Heuristic Rule — Action Frequency\n(all seeds)",
                      color="white", fontsize=12, pad=10)
    axes[2].set_ylabel("Times Chosen", color=TEXT, fontsize=10)
    plt.setp(axes[2].get_xticklabels(), rotation=30, ha="right")

    plt.tight_layout(pad=2.0)
    _save(fig, "solo", "outcome_summary.png")


# ── Cross-model comparison ───────────────────────────────────────────

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
    fig.suptitle(f"Cross-Model Summary  —  Mean over {N} Seeds × {N_STEPS} Steps\n"
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
        bars_rl = ax.bar(x - bar_w / 2, rl_vals, bar_w, color=RL_COL,
                         label="RL", edgecolor=GRID, linewidth=0.5,
                         yerr=rl_errs, error_kw=ekw)
        bars_h  = ax.bar(x + bar_w / 2, h_vals,  bar_w, color=H_COL,
                         label="Heuristic", edgecolor=GRID, linewidth=0.5,
                         yerr=h_errs, error_kw=ekw)

        ref = max(abs(v) for v in rl_vals + h_vals) * 0.015
        for bar, val in zip(bars_rl, rl_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ref,
                    f"{val:,.0f}", ha="center", va="bottom",
                    color=TEXT, fontsize=8, fontweight="bold")
        for bar, val in zip(bars_h, h_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ref,
                    f"{val:,.0f}", ha="center", va="bottom", color=TEXT, fontsize=8)
        for i, (rv, hv) in enumerate(zip(rl_vals, h_vals)):
            if rv > hv:
                ax.text(x[i] - bar_w / 2, max(rv, hv) * 1.10, "[RL wins]",
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
    fig.suptitle(f"RL Effect Scorecard — All Models  (avg over {N} seeds)\n"
                 "Green = positive  |  Red = negative  |  Error bars = variability",
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

        y = np.arange(len(m_names))
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


# ── Folder / label maps ──────────────────────────────────────────────

FOLDER_NAME = {"solo": "solo", "coop": "cooperative", "comp": "competitive"}
MODEL_LABELS = {
    "solo": "Solo  (1 RL firm vs 9 heuristic)",
    "coop": "Cooperative  (3 RL firms, shared reward)",
    "comp": "Competitive  (3 RL firms, relative reward)",
}
MODEL_NAMES = ["solo", "coop", "comp"]


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\nN = {N} seeds: {SEEDS}")
    print(f"Output: {VIZ_DIR}\n")

    # ── Metric charts for all models ─────────────────────────────────
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

        # Profit
        chart_metric_seeds(folder, label, res,
                           "rl_profit", "h_profit", "Profit (THB)", "profit_seeds.png")
        chart_metric_avg(folder, label, res,
                         "rl_profit", "h_profit", "Profit (THB)", "profit_avg.png")
        chart_profit_delta_seeds(folder, label, res)
        chart_profit_delta_avg(folder, label, res)

        # Workers
        chart_metric_seeds(folder, label, res,
                           "rl_workers", "h_workers", "Avg Workers per Firm", "workers_seeds.png")
        chart_metric_avg(folder, label, res,
                         "rl_workers", "h_workers", "Avg Workers per Firm", "workers_avg.png")

        # Wage (with market-average reference line)
        chart_metric_seeds(folder, label, res,
                           "rl_wage", "h_wage", "Monthly Wage (THB)", "wage_seeds.png",
                           extra_line="market_wage")
        chart_metric_avg(folder, label, res,
                         "rl_wage", "h_wage", "Monthly Wage (THB)", "wage_avg.png",
                         extra_line="market_wage")

        # Employment rate
        chart_employment_seeds(folder, label, res)
        chart_employment_avg(folder, label, res)

        # Scorecard
        chart_scorecard(folder, label, res)

    # ── Solo action analysis ──────────────────────────────────────────
    if "solo" in available and (ROOT / "solo" / "solo_model.zip").exists():
        print("\n  [Solo — action analysis]")
        all_action_records = []
        for seed in SEEDS:
            print(f"    action episode seed {seed}...")
            recs = run_solo_episode_with_actions(seed)
            classify_actions(recs)
            all_action_records.append(recs)
            divs = [r for r in recs if r["diverge"]]
            print(f"      divergences={len(divs)}  "
                  f"better={sum(1 for r in divs if r['outcome']=='better')}  "
                  f"worse={sum(1 for r in divs if r['outcome']=='worse')}  "
                  f"neutral={sum(1 for r in divs if r['outcome']=='neutral')}")

        ref = all_action_records[0]
        chart_action_strip(ref, "rl_action", "RL Agent",       "actions_rl.png",        SEEDS[0])
        chart_action_strip(ref, "h_action",  "Heuristic Rule", "actions_heuristic.png", SEEDS[0])
        chart_divergence_map(ref, SEEDS[0])
        chart_outcome_summary(all_action_records)

    # ── Cross-model comparison ────────────────────────────────────────
    print("\n  [comparison]")
    active_labels = {n: MODEL_LABELS[n] for n in available}
    chart_comparison_summary(results, available)
    chart_comparison_scorecard(results, available, active_labels)

    print(f"\nDone. All charts saved to: {VIZ_DIR}")
    print(f"\nPer model  (×{len(available)} models):")
    print(f"  profit_seeds.png        profit_avg.png")
    print(f"  profit_delta_seeds.png  profit_delta_avg.png")
    print(f"  workers_seeds.png       workers_avg.png")
    print(f"  wage_seeds.png          wage_avg.png")
    print(f"  employment_seeds.png    employment_avg.png")
    print(f"  scorecard.png")
    if "solo" in available:
        print(f"\nSolo only:")
        print(f"  actions_rl.png  actions_heuristic.png  divergence_map.png  outcome_summary.png")
    print(f"\ncomparison/: summary.png  scorecard.png")
