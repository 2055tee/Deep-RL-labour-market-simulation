#!/usr/bin/env python
# viz_compare.py
#
# Runs one solo RL episode, compares every decision to the heuristic rule,
# and saves each chart to visualizations/solo/ as a separate PNG.
#
# Output files (visualizations/solo/):
#   profit_detail.png      -- profit over time + divergence markers
#   wage_detail.png        -- wage decisions & when RL differed
#   workers_detail.png     -- headcount & hire/fire divergences
#   actions_rl.png         -- what the RL agent chose each step
#   actions_heuristic.png  -- what the heuristic would have chosen
#   divergence_map.png     -- step-by-step divergence outcome grid
#   outcome_summary.png    -- bar chart: better / worse / neutral
#
# Usage:
#   python viz_compare.py

import sys
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Must match EVAL_SEED in viz_all_models.py so both scripts show the same episode.
EVAL_SEED = 42

ROOT    = Path(__file__).parent.resolve()
VIZ_DIR = ROOT / "visualizations" / "solo"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "solo"))
for mod in ["model_rl", "firm_env", "rl_vis"]:
    sys.modules.pop(mod, None)

from sb3_contrib import MaskablePPO
from firm_env import LaborMarketEnv

# ------------------------------------------------------------------ #
#  Shared style                                                       #
# ------------------------------------------------------------------ #

BG    = "#0f0f1a"
PANEL = "#16213e"
GRID  = "#1e2a4a"
TEXT  = "#d0d0e8"
DIM   = "#888899"

ACTION_NAMES = {
    0: "Hold",
    1: "Wage +300",
    2: "Wage +100",
    3: "Wage -100",
    4: "Wage -300",
    5: "Post Vacancy",
    6: "Fire Worker",
}

ACTION_COLORS = {
    0: "#555577",
    1: "#1565c0",
    2: "#64b5f6",
    3: "#ffa726",
    4: "#e53935",
    5: "#43a047",
    6: "#8e24aa",
}

OUT_BETTER  = "#00e676"
OUT_WORSE   = "#ff1744"
OUT_NEUTRAL = "#ffd740"
OUT_AGREE   = "#333355"


def styled_fig(w=14, h=7):
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
    print(f"  saved: solo/{path.name}")


def wage_cycle_lines(ax, alpha=0.25):
    for x in range(0, 361, 12):
        ax.axvline(x, color=GRID, lw=0.8, alpha=alpha, zorder=1)


# ------------------------------------------------------------------ #
#  Heuristic shadow                                                   #
# ------------------------------------------------------------------ #

def heuristic_action(firm, step):
    current_profit = firm.compute_profit()

    # Employment (every step): fire first, then hire
    if firm.current_workers:
        if firm.compute_profit(labor_override=len(firm.current_workers) - 1) > current_profit:
            return 6  # fire worker

    labor = len(firm.current_workers)
    mpl   = firm.marginal_product_labor(firm.productivity, labor + 1, firm.alpha)
    if firm.output_price * mpl >= firm.monthly_wage:
        return 5  # post vacancy

    # Wage (annual)
    if step % 12 == 0:
        if firm.vacancies > 0 and firm.vacancy_duration > 0:
            return 1  # wage +300 (vacancy pressure)
        profit_up = firm.compute_profit(wage=max(int(firm.monthly_wage * 1.02), firm.monthly_wage + 1))
        profit_dn = firm.compute_profit(wage=max(int(firm.monthly_wage * 0.98), firm.wage_floor()))
        if profit_up > current_profit and profit_up >= profit_dn:
            return 2
        if profit_dn > current_profit:
            return 3

    return 0  # hold


# ------------------------------------------------------------------ #
#  Episode runner                                                     #
# ------------------------------------------------------------------ #

def run_episode():
    random.seed(EVAL_SEED)
    np.random.seed(EVAL_SEED)
    policy = MaskablePPO.load(str(ROOT / "solo" / "solo_model"))
    env    = LaborMarketEnv()
    obs, _ = env.reset()
    records = []

    for step in range(360):
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
            "step":       step,
            "rl_action":  rl_act,
            "h_action":   h_act,
            "diverge":    rl_act != h_act,
            "pre_profit": pre_profit,
            "profit":     float(env.rl_firm.profit),
            "delta":      float(env.rl_firm.profit) - pre_profit,
            "wage":       pre_wage,
            "workers":    pre_workers,
        })

    return records


def classify(records):
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


# ------------------------------------------------------------------ #
#  Chart 1 — Profit timeline                                         #
# ------------------------------------------------------------------ #

def chart_profit(records):
    steps   = [r["step"]   for r in records]
    profits = [r["profit"] for r in records]
    divs    = [r for r in records if r["diverge"]]

    fig, ax = styled_fig(16, 7)
    wage_cycle_lines(ax)

    ax.axhline(0, color="#445566", lw=1.2, ls="--", zorder=2)
    ax.fill_between(steps, profits, 0,
                    where=[p > 0 for p in profits],
                    color="#00897b", alpha=0.15, zorder=2)
    ax.fill_between(steps, profits, 0,
                    where=[p <= 0 for p in profits],
                    color="#c62828", alpha=0.15, zorder=2)
    ax.plot(steps, profits, color="#4fc3f7", lw=2, zorder=3)

    for r in divs:
        s, p = r["step"], r["profit"]
        if r["outcome"] == "better":
            ax.scatter(s, p, color=OUT_BETTER,  s=120, zorder=6,
                       marker="^", edgecolors="white", linewidths=0.5)
        elif r["outcome"] == "worse":
            ax.scatter(s, p, color=OUT_WORSE,   s=120, zorder=6,
                       marker="v", edgecolors="white", linewidths=0.5)
        else:
            ax.scatter(s, p, color=OUT_NEUTRAL, s=50,  zorder=5, marker="o")

    ax.set_title("Profit Over Time — RL Firm (Solo Scenario)",
                 color="white", fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Model Step  (vertical lines = annual wage-review points)", color=TEXT, fontsize=11)
    ax.set_ylabel("Monthly Profit  (THB)", color=TEXT, fontsize=11)
    ax.set_xlim(0, 359)

    handles = [
        plt.Line2D([0],[0], color="#4fc3f7", lw=2, label="RL firm profit"),
        mpatches.Patch(color="#00897b", alpha=0.5, label="Profitable zone"),
        mpatches.Patch(color="#c62828", alpha=0.5, label="Loss zone"),
        mpatches.Patch(color=OUT_BETTER,  label="Diverged from rule -> better outcome  (triangle up)"),
        mpatches.Patch(color=OUT_WORSE,   label="Diverged from rule -> worse outcome   (triangle down)"),
        mpatches.Patch(color=OUT_NEUTRAL, label="Diverged -> similar outcome  (circle)"),
    ]
    ax.legend(handles=handles, facecolor=PANEL, edgecolor=GRID,
              labelcolor=TEXT, fontsize=9, loc="upper left")

    save(fig, "profit_detail.png")


# ------------------------------------------------------------------ #
#  Chart 2 — Wage timeline                                           #
# ------------------------------------------------------------------ #

def chart_wage(records):
    steps = [r["step"] for r in records]
    wages = [r["wage"] for r in records]

    # Identify wage-change steps and divergences on wage decisions
    wage_divs = [r for r in records
                 if r["diverge"] and r["step"] % 12 == 0
                 and r["rl_action"] in (1, 2, 3, 4)
                    or r["h_action"] in (1, 2, 3, 4)]

    fig, ax = styled_fig(16, 6)
    wage_cycle_lines(ax, alpha=0.5)

    ax.plot(steps, wages, color="#ce93d8", lw=2, drawstyle="steps-post")

    # Mark wage change steps (every 12)
    for r in records:
        if r["step"] % 12 == 0:
            w = r["wage"]
            if r["diverge"] and (r["rl_action"] in (1, 2, 3, 4) or
                                  r["h_action"] in (1, 2, 3, 4)):
                col = OUT_BETTER if r["outcome"] == "better" else \
                      OUT_WORSE  if r["outcome"] == "worse"  else OUT_NEUTRAL
                ax.scatter(r["step"], w, color=col, s=180, zorder=5,
                           marker="D", edgecolors="white", linewidths=0.5)
            else:
                ax.scatter(r["step"], w, color=DIM, s=40, zorder=4, marker="|")

    ax.set_title("Monthly Wage Over Time — RL Firm\n"
                 "Diamonds = wage-review steps where RL and heuristic disagreed",
                 color="white", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Model Step", color=TEXT, fontsize=11)
    ax.set_ylabel("Monthly Wage  (THB)", color=TEXT, fontsize=11)
    ax.set_xlim(0, 359)

    handles = [
        plt.Line2D([0],[0], color="#ce93d8", lw=2, label="RL monthly wage"),
        mpatches.Patch(color=DIM,         label="Annual wage review (agreed with rule)"),
        mpatches.Patch(color=OUT_BETTER,  label="Diverged -> better outcome"),
        mpatches.Patch(color=OUT_WORSE,   label="Diverged -> worse outcome"),
        mpatches.Patch(color=OUT_NEUTRAL, label="Diverged -> neutral outcome"),
    ]
    ax.legend(handles=handles, facecolor=PANEL, edgecolor=GRID,
              labelcolor=TEXT, fontsize=9, loc="best")

    save(fig, "wage_detail.png")


# ------------------------------------------------------------------ #
#  Chart 3 — Worker headcount                                        #
# ------------------------------------------------------------------ #

def chart_workers(records):
    steps   = [r["step"]    for r in records]
    workers = [r["workers"] for r in records]

    hire_fire_divs = [r for r in records
                      if r["diverge"] and
                      (r["rl_action"] in (5, 6) or r["h_action"] in (5, 6))]

    fig, ax = styled_fig(16, 6)
    wage_cycle_lines(ax)

    ax.fill_between(steps, workers, alpha=0.15, color="#a5d6a7", step="post")
    ax.step(steps, workers, color="#a5d6a7", lw=2, where="post")

    for r in hire_fire_divs:
        col = OUT_BETTER if r["outcome"] == "better" else \
              OUT_WORSE  if r["outcome"] == "worse"  else OUT_NEUTRAL
        shape = "^" if r["rl_action"] == 5 else "v"
        ax.scatter(r["step"], r["workers"], color=col, s=130, zorder=5,
                   marker=shape, edgecolors="white", linewidths=0.5)

    ax.set_title("Worker Headcount Over Time — RL Firm\n"
                 "Triangle up = RL posted vacancy, Triangle down = RL fired  (coloured when rule disagreed)",
                 color="white", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Model Step", color=TEXT, fontsize=11)
    ax.set_ylabel("Number of Workers", color=TEXT, fontsize=11)
    ax.set_xlim(0, 359)

    handles = [
        plt.Line2D([0],[0], color="#a5d6a7", lw=2, label="RL worker count"),
        mpatches.Patch(color=OUT_BETTER,  label="RL diverged -> better outcome"),
        mpatches.Patch(color=OUT_WORSE,   label="RL diverged -> worse outcome"),
        mpatches.Patch(color=OUT_NEUTRAL, label="RL diverged -> neutral outcome"),
    ]
    ax.legend(handles=handles, facecolor=PANEL, edgecolor=GRID,
              labelcolor=TEXT, fontsize=9, loc="best")

    save(fig, "workers_detail.png")


# ------------------------------------------------------------------ #
#  Chart 4 — RL action strip (full episode)                          #
# ------------------------------------------------------------------ #

def chart_action_strip(records, agent_key, label, filename):
    actions = [r[agent_key] for r in records]
    divs    = {r["step"] for r in records if r["diverge"]}

    fig, ax = plt.subplots(figsize=(20, 3), facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=10)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)

    for s, act in enumerate(actions):
        ax.bar(s, 1, color=ACTION_COLORS[act], width=1.0, align="edge")

    # White border on diverged steps
    for s in divs:
        ax.bar(s, 1, color="none", width=1.0, align="edge",
               edgecolor="white", linewidth=1.0)

    ax.set_xlim(0, 360)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Model Step  (white border = differs from the other agent)", color=TEXT, fontsize=11)
    ax.set_title(f"Action at Every Step — {label}\n"
                 f"Each bar is one model step, colour shows which action was chosen",
                 color="white", fontsize=13, fontweight="bold", pad=10)

    patches = [mpatches.Patch(color=ACTION_COLORS[k], label=ACTION_NAMES[k])
               for k in range(7)]
    ax.legend(handles=patches, ncol=7, facecolor=PANEL, edgecolor=GRID,
              labelcolor=TEXT, fontsize=9, loc="upper center",
              bbox_to_anchor=(0.5, -0.35))

    save(fig, filename)


# ------------------------------------------------------------------ #
#  Chart 6 — Divergence outcome map                                  #
# ------------------------------------------------------------------ #

def chart_divergence_map(records):
    """
    A step-by-step grid where each cell is coloured by outcome:
    agree (dark), better (green), worse (red), neutral (yellow).
    Rows wrap every 60 steps so the grid is readable.
    """
    COLS = 60
    ROWS = 360 // COLS   # 6 rows

    outcome_color = {
        "agree":   OUT_AGREE,
        "better":  OUT_BETTER,
        "worse":   OUT_WORSE,
        "neutral": OUT_NEUTRAL,
    }

    fig, ax = plt.subplots(figsize=(16, 5), facecolor=BG)
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    cell_w, cell_h = 1.0, 1.0
    pad = 0.06

    for r in records:
        s = r["step"]
        col = s % COLS
        row = ROWS - 1 - (s // COLS)   # top row = steps 0-59
        x = col * cell_w
        y = row * cell_h
        color = outcome_color[r["outcome"]]
        rect  = plt.Rectangle((x + pad, y + pad),
                               cell_w - 2*pad, cell_h - 2*pad,
                               color=color)
        ax.add_patch(rect)

        # Step number inside cell (every 10th step)
        if s % 10 == 0:
            ax.text(x + cell_w/2, y + cell_h/2, str(s),
                    ha="center", va="center", fontsize=6.5,
                    color="white", alpha=0.7)

    ax.set_xlim(0, COLS * cell_w)
    ax.set_ylim(0, ROWS * cell_h)
    ax.set_aspect("equal")

    # Row labels
    for row_i in range(ROWS):
        start = (ROWS - 1 - row_i) * COLS
        ax.text(-1.5, row_i * cell_h + cell_h/2,
                f"Steps\n{start}-{start+COLS-1}",
                ha="right", va="center", color=TEXT, fontsize=8)

    ax.set_title("Divergence Outcome Map — Every Step at a Glance\n"
                 "Each cell = 1 step. Step numbers shown every 10 steps.",
                 color="white", fontsize=14, fontweight="bold", pad=12)

    handles = [
        mpatches.Patch(color=OUT_AGREE,   label="Agreed with rule (no divergence)"),
        mpatches.Patch(color=OUT_BETTER,  label="Diverged -> better outcome"),
        mpatches.Patch(color=OUT_WORSE,   label="Diverged -> worse outcome"),
        mpatches.Patch(color=OUT_NEUTRAL, label="Diverged -> neutral outcome"),
    ]
    ax.legend(handles=handles, facecolor=BG, edgecolor=GRID,
              labelcolor=TEXT, fontsize=10,
              loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=4)

    save(fig, "divergence_map.png")


# ------------------------------------------------------------------ #
#  Chart 7 — Outcome summary bar chart                               #
# ------------------------------------------------------------------ #

def chart_outcome_summary(records):
    divs = [r for r in records if r["diverge"]]

    # By action type that RL chose
    from collections import Counter
    rl_action_counts = Counter(r["rl_action"] for r in records)
    h_action_counts  = Counter(r["h_action"]  for r in records)

    outcomes = {
        "Agreed\nwith rule":    sum(1 for r in records if not r["diverge"]),
        "Diverged -\nBetter":   sum(1 for r in divs if r["outcome"] == "better"),
        "Diverged -\nNeutral":  sum(1 for r in divs if r["outcome"] == "neutral"),
        "Diverged -\nWorse":    sum(1 for r in divs if r["outcome"] == "worse"),
    }
    colors_out = [OUT_AGREE, OUT_BETTER, OUT_NEUTRAL, OUT_WORSE]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=BG)
    fig.suptitle("Decision Summary — RL vs Heuristic (360 Steps)",
                 color="white", fontsize=15, fontweight="bold", y=1.01)

    # ── Left: divergence outcomes ─────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.tick_params(colors=TEXT, labelsize=10)

    bars = ax.bar(outcomes.keys(), outcomes.values(), color=colors_out,
                  width=0.55, edgecolor=GRID, linewidth=0.5)
    for bar, val in zip(bars, outcomes.values()):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 3,
                str(val), ha="center", va="bottom",
                color="white", fontsize=12, fontweight="bold")
    ax.set_title("How Often Did RL Diverge\nand What Was the Result?",
                 color="white", fontsize=12, pad=10)
    ax.set_ylabel("Number of Steps", color=TEXT, fontsize=10)
    ax.set_ylim(0, max(outcomes.values()) * 1.2)

    # ── Middle: RL action distribution ────────────────────────────
    ax = axes[1]
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.tick_params(colors=TEXT, labelsize=9)

    acts  = [ACTION_NAMES[k] for k in range(7)]
    rl_v  = [rl_action_counts.get(k, 0) for k in range(7)]
    cols  = [ACTION_COLORS[k] for k in range(7)]
    bars2 = ax.bar(acts, rl_v, color=cols, edgecolor=GRID, linewidth=0.5)
    for bar, val in zip(bars2, rl_v):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 1,
                    str(val), ha="center", va="bottom",
                    color="white", fontsize=10)
    ax.set_title("RL Agent — Action Frequency\n(total 360 steps)",
                 color="white", fontsize=12, pad=10)
    ax.set_ylabel("Times Chosen", color=TEXT, fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    # ── Right: Heuristic action distribution ─────────────────────
    ax = axes[2]
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.tick_params(colors=TEXT, labelsize=9)

    h_v   = [h_action_counts.get(k, 0) for k in range(7)]
    bars3 = ax.bar(acts, h_v, color=cols, edgecolor=GRID, linewidth=0.5)
    for bar, val in zip(bars3, h_v):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 1,
                    str(val), ha="center", va="bottom",
                    color="white", fontsize=10)
    ax.set_title("Heuristic Rule — Action Frequency\n(what it would have done)",
                 color="white", fontsize=12, pad=10)
    ax.set_ylabel("Times Chosen", color=TEXT, fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    plt.tight_layout(pad=2.0)
    save(fig, "outcome_summary.png")


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    print("Running solo episode...")
    records = run_episode()
    classify(records)

    divs = [r for r in records if r["diverge"]]
    print(f"  Divergences : {len(divs)} / 360  ({len(divs)/3.6:.0f}%)")
    print(f"  Better      : {sum(1 for r in divs if r['outcome'] == 'better')}")
    print(f"  Worse       : {sum(1 for r in divs if r['outcome'] == 'worse')}")
    print(f"  Neutral     : {sum(1 for r in divs if r['outcome'] == 'neutral')}")
    print()
    print(f"Saving charts to: {VIZ_DIR}")

    chart_profit(records)
    chart_wage(records)
    chart_workers(records)
    chart_action_strip(records, "rl_action", "RL Agent",       "actions_rl.png")
    chart_action_strip(records, "h_action",  "Heuristic Rule", "actions_heuristic.png")
    chart_divergence_map(records)
    chart_outcome_summary(records)

    print()
    print("Done. Charts saved to: visualizations/solo/")
    print("  profit_detail.png      wage_detail.png     workers_detail.png")
    print("  actions_rl.png         actions_heuristic.png")
    print("  divergence_map.png     outcome_summary.png")
