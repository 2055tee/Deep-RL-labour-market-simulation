#!/usr/bin/env python
# reformed/demo.py
#
# Interactive Solara demo — Reformed model: 1 RL firm vs 9 heuristic firms.
# Run with:  solara run reformed/demo.py
#
# Structural fixes vs original model:
#   - Market-quit: workers leave after 4 months below 91% of market wage
#   - Option 3: workers see 25% of firms per search step
#   - Option 5: utility-proportional switching cost
#   - Option 4: wage-gap probability boost (enabled during RL training)
#   - RL firm vacancy cap: MAX_VACANCIES = 5
#   - Firm replacement: new firms spawn on exit

import sys
from pathlib import Path

ROOT     = Path(__file__).parent.parent.resolve()
REF_DIR  = Path(__file__).parent.resolve()
sys.path.insert(0, str(REF_DIR))

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

from model import LaborMarketModel

# ── Visualisation components from rl_vis.py ──────────────────────────
from rl_vis import (
    FirmHistogram, FirmWageHistogram, FirmProfitHistogram, FirmCapitalHistogram,
    WageVsMPLScatter, CapitalVsProfitScatter,
    WorkerUtilityHistogram, WorkerWageHistogram,
)

# ── Load trained policy ───────────────────────────────────────────────
try:
    from sb3_contrib import MaskablePPO
    _POLICY = MaskablePPO.load(str(REF_DIR / "reformed_model"))
    print("[demo] Reformed policy loaded.")
except Exception as _e:
    _POLICY = None
    print(f"[demo] Could not load reformed model: {_e}")

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

RL_FIRM_ID = "F0"


# ─────────────────────────────────────────────────────────────────────
# Demo Model wrapper
# ─────────────────────────────────────────────────────────────────────

class ReformedDemoModel(LaborMarketModel):
    """
    Reformed LaborMarketModel with inline RL policy.
    SolaraViz re-instantiates this whenever a slider changes.
    """

    def __init__(self,
                 output_price        = 100.0,
                 productivity_scale  = 1.0,
                 min_wage_val        = 7700,
                 n_workers           = 100):

        super().__init__(
            N_workers        = int(n_workers),
            N_firms          = 10,
            use_wage_gap_prob= True,
            rl_firm_id       = RL_FIRM_ID,
        )

        self.min_wage = int(min_wage_val)
        for f in self.firms:
            f.output_price = output_price
            f.productivity = f.productivity * productivity_scale
        # Re-set wages with updated economics
        for f in self.firms:
            if f.uid != RL_FIRM_ID:
                f.set_initial_wage(gamma=0.8)
            # RL firm keeps min_wage init from model.__init__

        self.rl_firm      = next(f for f in self.firms if f.uid == RL_FIRM_ID)
        self._policy      = _POLICY
        self._step        = 0
        self._prev_profit = 0.0
        self._prev_wkrs   = len(self.rl_firm.current_workers)
        self.actions      = []

        # DataCollector drives make_plot_component charts
        self.datacollector = DataCollector(model_reporters={
            "RL Profit":         lambda m: m.rl_firm.profit,
            "Heuristic Profit":  lambda m: float(np.mean(
                [f.profit for f in m.active_firms() if f is not m.rl_firm] or [0]
            )),
            "RL Workers":        lambda m: len(m.rl_firm.current_workers),
            "Heuristic Workers": lambda m: float(np.mean(
                [len(f.current_workers) for f in m.active_firms() if f is not m.rl_firm] or [0]
            )),
            "RL Wage":           lambda m: m.rl_firm.monthly_wage,
            "Heuristic Wage":    lambda m: float(np.mean(
                [f.monthly_wage for f in m.active_firms() if f is not m.rl_firm] or [0]
            )),
            "Market Wage":       lambda m: float(np.mean(
                [f.monthly_wage for f in m.active_firms()] or [0]
            )),
            "Employment %":      lambda m: 100.0 * sum(1 for w in m.workers if w.employed) / max(len(m.workers), 1),
        })

    # ── Observation — mirrors reformed/firm_env.py exactly ────────────

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
        worker_change = float(np.tanh((labor - self._prev_wkrs) / 3.0))
        wage_clock    = (self._step % 12) / 11.0

        avg_prod    = float(np.mean([f.productivity for f in self.firms]))
        avg_cap     = float(np.mean([f.capital      for f in self.firms]))
        prod_vs_mkt = float(np.tanh((firm.productivity - avg_prod) / max(avg_prod, 1.0)))
        cap_vs_mkt  = float(np.tanh((firm.capital      - avg_cap)  / max(avg_cap,  1.0)))

        survival_signal = float(np.tanh(firm.deficit_months / 12.0))

        employed       = sum(1 for w in self.workers if w.employed)
        mkt_employment = employed / len(self.workers) if self.workers else 0.0

        # Anti-hoarding signal (feature 13)
        active_f      = self.active_firms()
        avg_workers   = float(np.mean([len(f.current_workers) for f in active_f])) if active_f else labor
        worker_mkt_sh = float(np.tanh((labor - avg_workers) / 5.0))

        obs = np.array([
            profit_signal, profit_change_signal, vmpl_gap, wage_vs_mkt,
            labor_ratio, vacancy_ratio, worker_change, wage_clock,
            prod_vs_mkt, cap_vs_mkt, survival_signal, mkt_employment,
            worker_mkt_sh,
        ], dtype=np.float32)
        return np.clip(obs, -1.5, 1.5)

    def _action_mask(self):
        wage_ok = self._step % 12 == 0
        return np.array([True, wage_ok, wage_ok, wage_ok, wage_ok, True, True], dtype=bool)

    # ── Mesa step ─────────────────────────────────────────────────────

    def step(self):
        self._prev_profit = self.rl_firm.profit
        self._prev_wkrs   = len(self.rl_firm.current_workers)

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
# Custom Solara components
# ─────────────────────────────────────────────────────────────────────

def _ax(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color("white")
    ax.grid(color=GRID, alpha=0.28, lw=0.5)


@solara.component
def InfoBanner(model):
    update_counter.get()
    solara.Info(
        "Reformed model  |  Market-quit: workers leave after 4 months < 91% market wage  |  "
        "RL firm vacancy cap: 5  |  Firm replacement on exit  |  "
        "Option 3 + 5 always on  |  Option 4 active"
    )


@solara.component
def FirmTableRL(model):
    """Live per-firm snapshot with RL firm highlighted."""
    update_counter.get()
    active = model.active_firms()
    df = pd.DataFrame([
        {
            "Firm":       f.uid + (" [RL]" if f.uid == RL_FIRM_ID else ""),
            "Type":       "RL" if f.uid == RL_FIRM_ID else "Heuristic",
            "Profit":     round(f.profit, 0),
            "Workers":    len(f.current_workers),
            "Wage":       round(f.monthly_wage, 0),
            "Capital":    round(f.capital, 1),
            "Vacancies":  f.vacancies,
            "Deficit Mo": f.deficit_months,
        }
        for f in active
    ])
    solara.Text(f"All Firms — month {model._step}  |  Active: {len(active)}")
    solara.DataFrame(df)


@solara.component
def WorkerTableRL(model):
    """Live worker snapshot with market-quit counter."""
    update_counter.get()
    df = pd.DataFrame([
        {
            "Worker":    w.uid,
            "Employed":  w.employed,
            "Wage":      w.monthly_wage if w.employed else 0,
            "Utility":   round(
                w.utility_if_work(w.monthly_wage) if w.employed
                else w.utility_if_not_work(), 3
            ),
            "Mkt-quit":  w.months_below_mkt,
        }
        for w in model.workers
    ])
    employed = df["Employed"].sum()
    solara.Text(f"Workers — {employed} employed / {len(df)} total  |  month {model._step}")
    solara.DataFrame(df)


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
    ax.set_xlabel(f"Last {len(last)} months  (month {model._step})", color=TEXT, fontsize=8)
    ax.set_title("RL Firm — Recent Actions", color="white", fontsize=9, fontweight="bold", pad=4)
    patches = [mpatches.Patch(color=ACT_COLORS[k], label=ACT_NAMES[k]) for k in range(7)]
    ax.legend(handles=patches, ncol=7, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT,
              fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.55))
    plt.tight_layout(pad=0.3)
    solara.FigureMatplotlib(fig)
    plt.close(fig)


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
    bars = ax.barh(y, pcts, color=colors, height=0.45, edgecolor=GRID, lw=0.5)
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
# make_plot_component line charts
# ─────────────────────────────────────────────────────────────────────

chart_profit     = make_plot_component({"RL Profit": RL_COL, "Heuristic Profit": H_COL})
chart_workers    = make_plot_component({"RL Workers": RL_COL, "Heuristic Workers": H_COL})
chart_wage       = make_plot_component({"RL Wage": RL_COL, "Heuristic Wage": H_COL, "Market Wage": MAC_COL})
chart_employment = make_plot_component("Employment %")

# ─────────────────────────────────────────────────────────────────────
# SolaraViz app
# ─────────────────────────────────────────────────────────────────────

model_params = {
    "output_price":       Slider("Output Price  (default 100)",        100,  50,   300,   10),
    "productivity_scale": Slider("Productivity Scale  (default 1.0)",  1.0,  0.3,  3.0,   0.1),
    "min_wage_val":       Slider("Min Wage THB  (default 7700)",       7700, 3000, 20000, 500),
    "n_workers":          Slider("Workers  (default 100)",             100,  30,   200,   10),
}

page = SolaraViz(
    ReformedDemoModel(),
    components=[
        InfoBanner,
        FirmTableRL,
        WorkerTableRL,
        chart_profit,
        chart_workers,
        chart_wage,
        chart_employment,
        ActionBar,
        Scorecard,
        FirmHistogram,
        FirmWageHistogram,
        FirmProfitHistogram,
        FirmCapitalHistogram,
        WageVsMPLScatter,
        CapitalVsProfitScatter,
        WorkerUtilityHistogram,
        WorkerWageHistogram,
    ],
    model_params=model_params,
    name="Reformed RL Labor Market",
)
