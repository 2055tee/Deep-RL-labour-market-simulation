#!/usr/bin/env python
# demo/demo_solo.py
#
# Interactive Mesa demo — Solo scenario: 1 RL firm vs 9 heuristic firms.
# Run with:  solara run demo/demo_solo.py
#
# ⭐ = slider critically affects RL performance (model was trained on fixed values).
#     Parameters far outside the training range produce a yellow warning banner.
#
# Training defaults (safe zone):
#   output_price      = 100      ⭐  (safe: 70–150)
#   productivity_scale= 1.0      ⭐  (safe: 0.7–1.5)
#   labor share α     = 0.65         (safe: 0.50–0.80)
#   min wage          = 7700         (safe: 5000–12000)
#   workers           = 100          (safe: 60–160)

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT / "solo"))

import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import solara
from mesa import DataCollector
from mesa.visualization import SolaraViz, Slider, make_plot_component
from mesa.visualization.utils import update_counter

# ── Policy (loaded once at module level) ─────────────────────────────
try:
    from sb3_contrib import MaskablePPO
    _POLICY = MaskablePPO.load(str(ROOT / "solo" / "solo_model"))
    print("[demo_solo] Policy loaded.")
except Exception as _e:
    _POLICY = None
    print(f"[demo_solo] Could not load solo model: {_e}")

from model_rl import LaborMarketModel  # solo/model_rl.py

# ── Colours ───────────────────────────────────────────────────────────
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

ACT_COLORS = {
    0: "#555577", 1: "#1565c0", 2: "#64b5f6",
    3: "#ffa726", 4: "#e53935", 5: "#43a047", 6: "#8e24aa",
}
ACT_NAMES = ["Hold", "Wage+300", "Wage+100", "Wage-100", "Wage-300", "Post Vac", "Fire"]

# Training defaults & OOD safe ranges
DEFAULTS = dict(output_price=100.0, productivity_scale=1.0,
                alpha_param=0.65, min_wage_val=7700, n_workers=100)
OOD_BOUNDS = dict(
    output_price       =(70,    150),
    productivity_scale =(0.7,   1.5),
    alpha_param        =(0.50,  0.80),
    min_wage_val       =(5000,  12000),
    n_workers          =(60,    160),
)
OOD_LABELS = dict(
    output_price       ="⭐ Output Price",
    productivity_scale ="⭐ Productivity Scale",
    alpha_param        ="Labor Share α",
    min_wage_val       ="Min Wage",
    n_workers          ="Workers",
)


# ─────────────────────────────────────────────────────────────────────
# Demo Model
# ─────────────────────────────────────────────────────────────────────

class SoloDemoModel(LaborMarketModel):
    """
    Solo LaborMarketModel with inline RL policy.
    SolaraViz re-instantiates this whenever a slider changes.
    """

    def __init__(self,
                 output_price       = DEFAULTS["output_price"],
                 productivity_scale = DEFAULTS["productivity_scale"],
                 alpha_param        = DEFAULTS["alpha_param"],
                 min_wage_val       = DEFAULTS["min_wage_val"],
                 n_workers          = DEFAULTS["n_workers"]):

        super().__init__(N_workers=int(n_workers), N_firms=10)

        # Store for OOD display
        self.params = dict(
            output_price=output_price,
            productivity_scale=productivity_scale,
            alpha_param=alpha_param,
            min_wage_val=int(min_wage_val),
            n_workers=int(n_workers),
        )

        # Apply custom parameters to every firm
        self.min_wage = int(min_wage_val)
        for firm in self.firms:
            firm.output_price  = output_price
            firm.productivity  = firm.productivity * productivity_scale
            firm.alpha         = alpha_param
        # Recompute wages so they reflect the new economics
        for firm in self.firms:
            firm.set_initial_wage(gamma=0.8)

        self._policy       = _POLICY
        self.rl_firm       = self.firms[0]       # F0 is always the RL firm
        self._step         = 0
        self._prev_profit  = 0.0
        self._prev_workers = len(self.rl_firm.current_workers)

        # Action history (for ActionBar strip)
        self.actions = []

        # Mesa DataCollector — drives make_plot_component charts
        self.datacollector = DataCollector(model_reporters={
            "RL Profit":         lambda m: m.rl_firm.profit,
            "Heuristic Profit":  lambda m: float(np.mean([f.profit         for f in m.firms if f is not m.rl_firm])),
            "RL Workers":        lambda m: len(m.rl_firm.current_workers),
            "Heuristic Workers": lambda m: float(np.mean([len(f.current_workers) for f in m.firms if f is not m.rl_firm])),
            "RL Wage":           lambda m: m.rl_firm.monthly_wage,
            "Heuristic Wage":    lambda m: float(np.mean([f.monthly_wage   for f in m.firms if f is not m.rl_firm])),
            "Market Wage":       lambda m: float(np.mean([f.monthly_wage   for f in m.firms])),
            "Employment %":      lambda m: 100.0 * sum(1 for w in m.workers if w.employed) / max(len(m.workers), 1),
        })

    # ── Observation (mirrors solo/firm_env.py exactly) ────────────────

    def _obs(self):
        firm  = self.rl_firm
        labor = len(firm.current_workers)

        profit_signal        = float(np.tanh(firm.profit / 20_000))
        profit_change_signal = float(np.tanh((firm.profit - self._prev_profit) / 5_000))

        if labor > 0:
            mpl      = firm.marginal_product_labor(firm.productivity, labor, firm.alpha)
            vmpl     = mpl * firm.output_price
            vmpl_gap = float(np.tanh((vmpl - firm.monthly_wage) / max(firm.monthly_wage, 1.0)))
        else:
            vmpl_gap = 1.0

        other_wages = [f.monthly_wage for f in self.firms if f is not firm]
        mkt_wage    = float(np.mean(other_wages)) if other_wages else firm.monthly_wage
        wage_vs_mkt = float(np.tanh((firm.monthly_wage - mkt_wage) / max(mkt_wage, 1.0)))

        labor_ratio   = labor / 40.0
        vacancy_ratio = min(firm.vacancies, 5) / 5.0
        worker_change = float(np.tanh((labor - self._prev_workers) / 3.0))
        wage_clock    = (self._step % 12) / 11.0

        avg_prod    = float(np.mean([f.productivity for f in self.firms]))
        avg_cap     = float(np.mean([f.capital      for f in self.firms]))
        prod_vs_mkt = float(np.tanh((firm.productivity - avg_prod) / max(avg_prod, 1.0)))
        cap_vs_mkt  = float(np.tanh((firm.capital      - avg_cap)  / max(avg_cap,  1.0)))

        survival_signal = float(np.tanh(firm.deficit_months / 12.0))

        employed       = sum(1 for w in self.workers if w.employed)
        mkt_employment = employed / len(self.workers) if self.workers else 0.0

        obs = np.array([
            profit_signal, profit_change_signal, vmpl_gap, wage_vs_mkt,
            labor_ratio, vacancy_ratio, worker_change, wage_clock,
            prod_vs_mkt, cap_vs_mkt, survival_signal, mkt_employment,
        ], dtype=np.float32)
        return np.clip(obs, -1.5, 1.5)

    def _action_mask(self):
        wage_ok = self._step % 12 == 0
        return np.array([True, wage_ok, wage_ok, wage_ok, wage_ok, True, True], dtype=bool)

    # ── Mesa step ─────────────────────────────────────────────────────

    def step(self):
        self._prev_profit  = self.rl_firm.profit
        self._prev_workers = len(self.rl_firm.current_workers)

        if self._policy is not None:
            obs  = self._obs()
            mask = self._action_mask()
            act, _ = self._policy.predict(
                obs[np.newaxis], deterministic=True,
                action_masks=mask[np.newaxis],
            )
            self.rl_action = int(act[0])
        else:
            self.rl_action = 0

        super().step()
        self._step += 1
        self.actions.append(self.rl_action)
        self.datacollector.collect(self)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _ood_msgs(params):
    msgs = []
    for k, (lo, hi) in OOD_BOUNDS.items():
        v = params[k]
        if v < lo or v > hi:
            msgs.append(
                f"⚠  {OOD_LABELS[k]} = {v}  is outside the training range [{lo} – {hi}]. "
                f"Policy may behave sub-optimally."
            )
    return msgs


def _ax(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color("white")
    ax.grid(color=GRID, alpha=0.28, lw=0.5)



# ─────────────────────────────────────────────────────────────────────
# Solara visualisation components
# ─────────────────────────────────────────────────────────────────────

@solara.component
def InfoBanner(model):
    """Training-default reminder + OOD warnings."""
    update_counter.get()
    solara.Info(
        "Training defaults — ⭐ Output Price: 100  |  ⭐ Productivity Scale: 1.0  |  "
        "α: 0.65  |  Min Wage: 7700  |  Workers: 100  "
        "(reset sliders to these values to restore baseline behaviour)"
    )
    for msg in _ood_msgs(model.params):
        solara.Warning(msg)



@solara.component
def ActionBar(model):
    """Colour strip of the last 60 RL actions."""
    update_counter.get()
    last = model.actions[-60:] if model.actions else []
    if not last:
        return
    fig, ax = plt.subplots(figsize=(10, 1.4), facecolor=BG)
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    for i, act in enumerate(last):
        ax.bar(i, 1, color=ACT_COLORS[act], width=1.0, align="edge")
    ax.set_xlim(0, len(last))
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel(f"Last {len(last)} months  (current month: {model._step})", color=TEXT, fontsize=8)
    ax.set_title("RL Firm — Recent Actions", color="white", fontsize=9, fontweight="bold", pad=4)
    patches = [mpatches.Patch(color=ACT_COLORS[k], label=ACT_NAMES[k]) for k in range(7)]
    ax.legend(handles=patches, ncol=7, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT,
              fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.55))
    plt.tight_layout(pad=0.3)
    solara.FigureMatplotlib(fig)
    plt.close(fig)


@solara.component
def FirmTable(model):
    """Live per-firm snapshot (sortable)."""
    update_counter.get()
    firms   = model.firms
    rl_firm = model.rl_firm
    df = pd.DataFrame([
        {
            "Firm":         f"F{i}" + (" [RL]" if f is rl_firm else ""),
            "Type":         "RL" if f is rl_firm else "Heuristic",
            "Profit":       round(f.profit, 0),
            "Workers":      len(f.current_workers),
            "Wage":         round(f.monthly_wage, 0),
            "Capital":      round(f.capital, 0),
            "Productivity": round(f.productivity, 3),
            "α":            round(f.alpha, 2),
            "Output Price": round(f.output_price, 0),
            "Vacancies":    f.vacancies,
            "Deficit Mo":   getattr(f, "deficit_months", 0),
        }
        for i, f in enumerate(firms)
    ])
    solara.Text(f"All Firms — month {model._step}")
    solara.DataFrame(df)


@solara.component
def WorkerTable(model):
    """Live worker snapshot (employed/unemployed, wage, utility)."""
    update_counter.get()
    df = pd.DataFrame([
        {
            "Worker":   w.unique_id,
            "Employed": w.employed,
            "Wage":     round(w.monthly_wage, 0) if w.employed else 0,
            "Utility":  round(
                w.utility_if_work(w.monthly_wage) if w.employed else w.utility_if_not_work(),
                3,
            ),
        }
        for w in model.workers
    ])
    solara.Text(f"All Workers — month {model._step}")
    solara.DataFrame(df)


@solara.component
def Scorecard(model):
    """Cumulative RL advantage vs heuristic baseline."""
    update_counter.get()
    df = model.datacollector.get_model_vars_dataframe()
    if df.empty:
        return
    metrics = {
        "Profit":  ("RL Profit",  "Heuristic Profit"),
        "Workers": ("RL Workers", "Heuristic Workers"),
        "Wage":    ("RL Wage",    "Heuristic Wage"),
    }
    labels, pcts, colors = [], [], []
    for name, (rl_col, h_col) in metrics.items():
        rl_m = float(df[rl_col].mean())
        h_m  = float(df[h_col].mean())
        pct  = (rl_m - h_m) / max(abs(h_m), 1.0) * 100
        labels.append(name)
        pcts.append(pct)
        colors.append(BETTER if pct >= 0 else WORSE)

    fig, ax = plt.subplots(figsize=(6, 2.8), facecolor=BG)
    _ax(ax)
    y    = np.arange(len(labels))
    bars = ax.barh(y, pcts, color=colors, height=0.45, edgecolor=GRID, linewidth=0.5)
    ax.axvline(0, color=DIM, lw=1.5)
    xlim = max(abs(p) for p in pcts) * 1.7 if any(pcts) else 10
    for bar, p in zip(bars, pcts):
        ha  = "left"  if p >= 0 else "right"
        off = xlim * 0.03
        ax.text(p + (off if p >= 0 else -off),
                bar.get_y() + bar.get_height() / 2,
                f"{p:+.1f}%", ha=ha, va="center",
                color="white", fontsize=10, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=TEXT, fontsize=10)
    ax.set_xlim(-xlim, xlim)
    ax.axvspan( 0,  xlim, color=BETTER, alpha=0.05)
    ax.axvspan(-xlim, 0,  color=WORSE,  alpha=0.05)
    ax.set_title(f"Cumulative RL Scorecard  (month {model._step})",
                 color="white", fontsize=10, fontweight="bold")
    ax.set_xlabel("RL advantage vs heuristic (%)", color=TEXT, fontsize=9)
    plt.tight_layout(pad=0.4)
    solara.FigureMatplotlib(fig)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Distribution / scatter charts  (adapted from viz_wage.py)
# RL firm shown in red, heuristic firms in blue.
# ─────────────────────────────────────────────────────────────────────

_COL_RL = "#e74c3c"
_COL_H  = "#3498db"


@solara.component
def FirmSizeHistogram(model):
    update_counter.get()
    firms  = model.firms
    sizes  = [len(f.current_workers) for f in firms]
    rl_sz  = len(model.rl_firm.current_workers)
    fig, ax = plt.subplots(figsize=(8, 4))
    _ax(ax)
    if sizes:
        ax.hist(sizes, bins=20, color=_COL_H, edgecolor=GRID, label="Heuristic")
    ax.axvline(rl_sz, color=_COL_RL, lw=2, linestyle="--",
               label=f"RL firm ({rl_sz} workers)")
    ax.set_title("Distribution of Firm Sizes")
    ax.set_xlabel("Workers"); ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    plt.tight_layout()
    solara.FigureMatplotlib(fig); plt.close(fig)


@solara.component
def FirmWageHistogram(model):
    update_counter.get()
    firms = model.firms
    wages = [f.monthly_wage for f in firms if f.monthly_wage]
    rl_w  = model.rl_firm.monthly_wage
    fig, ax = plt.subplots(figsize=(8, 4))
    _ax(ax)
    if wages:
        ax.hist(wages, bins=20, color=_COL_H, edgecolor=GRID, label="Heuristic")
        if len(set(wages)) == 1:
            ax.set_xlim(wages[0] - 1, wages[0] + 1); ax.set_xticks(wages)
    ax.axvline(rl_w, color=_COL_RL, lw=2, linestyle="--",
               label=f"RL firm ({rl_w:,})")
    ax.set_title("Distribution of Firm Wages")
    ax.set_xlabel("Monthly Wage (THB)"); ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    plt.tight_layout()
    solara.FigureMatplotlib(fig); plt.close(fig)


@solara.component
def FirmProfitHistogram(model):
    update_counter.get()
    firms   = model.firms
    profits = [f.profit for f in firms]
    rl_p    = model.rl_firm.profit
    fig, ax = plt.subplots(figsize=(8, 4))
    _ax(ax)
    if profits:
        ax.hist(profits, bins=20, color=_COL_H, edgecolor=GRID, label="Heuristic")
    ax.axvline(rl_p, color=_COL_RL, lw=2, linestyle="--",
               label=f"RL firm ({rl_p:,.0f})")
    ax.set_title("Distribution of Firm Profits")
    ax.set_xlabel("Profit (THB)"); ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    plt.tight_layout()
    solara.FigureMatplotlib(fig); plt.close(fig)


@solara.component
def FirmCapitalHistogram(model):
    update_counter.get()
    firms    = model.firms
    capitals = [f.capital for f in firms]
    rl_k     = model.rl_firm.capital
    fig, ax = plt.subplots(figsize=(8, 4))
    _ax(ax)
    if capitals:
        ax.hist(capitals, bins=20, color=_COL_H, edgecolor=GRID, label="Heuristic")
    ax.axvline(rl_k, color=_COL_RL, lw=2, linestyle="--",
               label=f"RL firm ({rl_k:.1f})")
    ax.set_title("Distribution of Firm Capital")
    ax.set_xlabel("Capital"); ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    plt.tight_layout()
    solara.FigureMatplotlib(fig); plt.close(fig)


@solara.component
def WageVsMPLScatter(model):
    update_counter.get()
    wages, vmpls, colors = [], [], []
    for f in model.firms:
        if not f.monthly_wage:
            continue
        labor = len(f.current_workers)
        mpl   = f.marginal_product_labor(f.productivity, labor, f.alpha)
        wages.append(f.monthly_wage)
        vmpls.append(mpl * f.output_price)
        colors.append(_COL_RL if f is model.rl_firm else _COL_H)
    fig, ax = plt.subplots(figsize=(8, 4))
    _ax(ax)
    if wages:
        ax.scatter(vmpls, wages, c=colors, edgecolors=GRID, alpha=0.85, s=60, zorder=3)
        lo, hi = min(min(vmpls), min(wages)), max(max(vmpls), max(wages))
        ax.plot([lo, hi], [lo, hi], color=DIM, linestyle="--", lw=1, label="wage = VMPL")
    ax.set_title("Wage vs Value of MPL")
    ax.set_xlabel("VMPL (price x MPL)"); ax.set_ylabel("Monthly Wage (THB)")
    handles = [mpatches.Patch(color=_COL_RL, label="RL firm"),
               mpatches.Patch(color=_COL_H,  label="Heuristic")]
    ax.legend(handles=handles, fontsize=8)
    plt.tight_layout()
    solara.FigureMatplotlib(fig); plt.close(fig)


@solara.component
def CapitalVsProfitScatter(model):
    update_counter.get()
    caps, profits, colors = [], [], []
    for f in model.firms:
        caps.append(f.capital)
        profits.append(f.profit)
        colors.append(_COL_RL if f is model.rl_firm else _COL_H)
    fig, ax = plt.subplots(figsize=(8, 4))
    _ax(ax)
    if caps:
        ax.scatter(caps, profits, c=colors, edgecolors=GRID, alpha=0.85, s=60)
    ax.axhline(0, color=DIM, linestyle="--", lw=1)
    ax.set_title("Capital vs Profit")
    ax.set_xlabel("Capital"); ax.set_ylabel("Profit (THB)")
    handles = [mpatches.Patch(color=_COL_RL, label="RL firm"),
               mpatches.Patch(color=_COL_H,  label="Heuristic")]
    ax.legend(handles=handles, fontsize=8)
    plt.tight_layout()
    solara.FigureMatplotlib(fig); plt.close(fig)


@solara.component
def WorkerUtilityHistogram(model):
    update_counter.get()
    utils = [
        w.utility_if_work(w.monthly_wage) if w.employed else w.utility_if_not_work()
        for w in model.workers
    ]
    fig, ax = plt.subplots(figsize=(9, 4))
    _ax(ax)
    if utils:
        ax.hist(utils, bins=20, color="#9b59b6", edgecolor=GRID)
    ax.set_title("Distribution of Worker Utility")
    ax.set_xlabel("Utility"); ax.set_ylabel("Count")
    plt.tight_layout()
    solara.FigureMatplotlib(fig); plt.close(fig)


@solara.component
def WorkerWageHistogram(model):
    update_counter.get()
    wages = [w.monthly_wage for w in model.workers if w.employed and w.monthly_wage > 0]
    fig, ax = plt.subplots(figsize=(9, 4))
    _ax(ax)
    if wages:
        ax.hist(wages, bins=20, color="#1abc9c", edgecolor=GRID)
        if len(set(wages)) == 1:
            ax.set_xlim(wages[0] - 1, wages[0] + 1); ax.set_xticks(wages)
    ax.set_title("Distribution of Worker Wages")
    ax.set_xlabel("Monthly Wage (THB)"); ax.set_ylabel("Count")
    plt.tight_layout()
    solara.FigureMatplotlib(fig); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# SolaraViz app
# ─────────────────────────────────────────────────────────────────────

chart_profit     = make_plot_component({"RL Profit": RL_COL, "Heuristic Profit": H_COL})
chart_workers    = make_plot_component({"RL Workers": RL_COL, "Heuristic Workers": H_COL})
chart_wage       = make_plot_component({"RL Wage": RL_COL, "Heuristic Wage": H_COL, "Market Wage": MAC_COL})
chart_employment = make_plot_component("Employment %")

model_params = {
    "output_price":        Slider("⭐ Output Price  (default 100)",         100,   50,    300,    10),
    "productivity_scale":  Slider("⭐ Productivity Scale  (default 1.0)",   1.0,   0.3,   3.0,    0.1),
    "alpha_param":         Slider("Labor Share α  (default 0.65)",          0.65,  0.30,  0.90,   0.05),
    "min_wage_val":        Slider("Min Wage THB  (default 7700)",           7700,  3000,  20000,  500),
    "n_workers":           Slider("Workers  (default 100)",                 100,   30,    200,    10),
}

page = SolaraViz(
    SoloDemoModel(),
    components=[InfoBanner, FirmTable, WorkerTable,
                chart_profit, chart_workers, chart_wage, chart_employment,
                ActionBar, Scorecard,
                FirmSizeHistogram, FirmWageHistogram, FirmProfitHistogram, FirmCapitalHistogram,
                WageVsMPLScatter, CapitalVsProfitScatter,
                WorkerUtilityHistogram, WorkerWageHistogram],
    model_params=model_params,
    name="RL Labor Market — Solo Demo",
)
