#!/usr/bin/env python
# demo/demo_solo.py
#
# Interactive demo -- Solo: 1 RL firm vs 9 heuristic firms (original solo model).
# Run with:  solara run demo/demo_solo.py
#
# ⭐ = slider critically affects RL performance (model trained on fixed values).
#     Parameters far outside the training range produce a warning banner.
#
# Training defaults: output_price=100, productivity_scale=1.0,
#                    alpha=0.65, min_wage=7700, workers=100

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT / "solo"))

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

try:
    from sb3_contrib import MaskablePPO
    _POLICY = MaskablePPO.load(str(ROOT / "solo" / "solo_model"))
    print("[demo_solo] Policy loaded.")
except Exception as _e:
    _POLICY = None
    print(f"[demo_solo] Could not load solo model: {_e}")

from model_rl import LaborMarketModel

# ── Palette ───────────────────────────────────────────────────────────
BG      = "#0f0f1a"
PANEL   = "#16213e"
GRID    = "#1e2a4a"
TEXT    = "#d0d0e8"
DIM     = "#888899"
BETTER  = "#00c853"
WORSE   = "#ff1744"
WARN    = "#ffab40"
RL_COL  = "#4fc3f7"
H_COL   = "#ffb74d"
MAC_COL = "#ce93d8"
AT_COL  = "#ef5350"

ACT_COLORS = {
    0: "#555577", 1: "#1565c0", 2: "#64b5f6",
    3: "#ffa726", 4: "#e53935", 5: "#43a047", 6: "#8e24aa",
}
ACT_NAMES = ["Hold", "Wage+300", "Wage+100", "Wage-100", "Wage-300", "Post Vac", "Fire"]

OBS_LABELS = [
    "profit_signal", "profit_change", "vmpl_gap", "wage_vs_mkt",
    "labor_ratio", "vacancy_ratio", "worker_change", "wage_clock",
    "prod_vs_mkt", "cap_vs_mkt", "survival_signal", "mkt_employment",
]

DEFAULTS   = dict(output_price=100.0, productivity_scale=1.0, alpha_param=0.65,
                  min_wage_val=7700, n_workers=100, n_firms=10,
                  rental_rate_val=500, worker_search_prob_val=10, seed_val=455)
OOD_BOUNDS = dict(output_price=(70,150), productivity_scale=(0.7,1.5),
                  alpha_param=(0.50,0.80), min_wage_val=(5000,12000), n_workers=(60,160))
OOD_LABELS = dict(output_price="Output Price", productivity_scale="Productivity Scale",
                  alpha_param="Labour Share alpha", min_wage_val="Min Wage", n_workers="Workers")


# ─────────────────────────────────────────────────────────────────────
# Demo model
# ─────────────────────────────────────────────────────────────────────

class SoloDemoModel(LaborMarketModel):

    def __init__(self,
                 output_price           = DEFAULTS["output_price"],
                 productivity_scale     = DEFAULTS["productivity_scale"],
                 alpha_param            = DEFAULTS["alpha_param"],
                 min_wage_val           = DEFAULTS["min_wage_val"],
                 n_workers              = DEFAULTS["n_workers"],
                 n_firms                = DEFAULTS["n_firms"],
                 rental_rate_val        = DEFAULTS["rental_rate_val"],
                 worker_search_prob_val = DEFAULTS["worker_search_prob_val"],
                 seed_val               = DEFAULTS["seed_val"]):
        import random as _random
        _random.seed(int(seed_val))
        np.random.seed(int(seed_val))

        super().__init__(N_workers=int(n_workers), N_firms=int(n_firms))

        self.params = dict(output_price=output_price, productivity_scale=productivity_scale,
                           alpha_param=alpha_param, min_wage_val=int(min_wage_val),
                           n_workers=int(n_workers))

        self.min_wage = int(min_wage_val)
        rr = float(rental_rate_val)
        sp = float(worker_search_prob_val) / 100.0

        for f in self.firms:
            f.output_price  = output_price
            f.productivity  = f.productivity * productivity_scale
            f.alpha         = alpha_param
            f.rental_rate   = rr
        for w in self.workers:
            w.job_search_prob = sp
        for f in self.firms:
            f.set_initial_wage(gamma=0.8)

        self._policy       = _POLICY
        self.rl_firm       = self.firms[0]
        self._step         = 0
        self._prev_profit  = 0.0
        self._prev_workers = len(self.rl_firm.current_workers)
        self.actions       = []
        self._last_obs     = np.zeros(12, dtype=np.float32)

        self.datacollector = DataCollector(model_reporters={
            "RL Profit":         lambda m: m.rl_firm.profit,
            "Heuristic Profit":  lambda m: float(np.mean([f.profit for f in m.firms if f is not m.rl_firm] or [0])),
            "RL Workers":        lambda m: len(m.rl_firm.current_workers),
            "Heuristic Workers": lambda m: float(np.mean([len(f.current_workers) for f in m.firms if f is not m.rl_firm] or [0])),
            "RL Wage":           lambda m: m.rl_firm.monthly_wage,
            "Heuristic Wage":    lambda m: float(np.mean([f.monthly_wage for f in m.firms if f is not m.rl_firm] or [0])),
            "Market Wage":       lambda m: float(np.mean([f.monthly_wage for f in m.firms] or [0])),
            "Wage Min":          lambda m: float(np.min([f.monthly_wage for f in m.firms] or [0])),
            "Wage Max":          lambda m: float(np.max([f.monthly_wage for f in m.firms] or [0])),
            "Employment %":      lambda m: 100.0 * sum(1 for w in m.workers if w.employed) / max(len(m.workers), 1),
            "Active Firms":      lambda m: len([f for f in m.firms if getattr(f, "active", True)]),
            "RL Deficit Months": lambda m: getattr(m.rl_firm, "deficit_months", 0),
        })

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

        survival_signal = float(np.tanh(getattr(firm, "deficit_months", 0) / 12.0))
        employed        = sum(1 for w in self.workers if w.employed)
        mkt_employment  = employed / len(self.workers) if self.workers else 0.0

        obs = np.array([
            profit_signal, profit_change_signal, vmpl_gap, wage_vs_mkt,
            labor_ratio, vacancy_ratio, worker_change, wage_clock,
            prod_vs_mkt, cap_vs_mkt, survival_signal, mkt_employment,
        ], dtype=np.float32)
        return np.clip(obs, -1.5, 1.5)

    def _action_mask(self):
        wage_ok = self._step % 12 == 0
        return np.array([True, wage_ok, wage_ok, wage_ok, wage_ok, True, True], dtype=bool)

    def step(self):
        self._prev_profit  = self.rl_firm.profit
        self._prev_workers = len(self.rl_firm.current_workers)

        if self._policy is not None:
            obs  = self._obs()
            mask = self._action_mask()
            act, _ = self._policy.predict(obs[np.newaxis], deterministic=True,
                                          action_masks=mask[np.newaxis])
            self.rl_action = int(act[0])
            self._last_obs = obs
        else:
            self.rl_action = 0

        super().step()
        self._step += 1
        self.actions.append(self.rl_action)
        self.datacollector.collect(self)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _ax(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    ax.xaxis.label.set_color(TEXT); ax.yaxis.label.set_color(TEXT)
    ax.title.set_color("white"); ax.grid(color=GRID, alpha=0.28, lw=0.5)

def _ood_msgs(params):
    msgs = []
    for k, (lo, hi) in OOD_BOUNDS.items():
        v = params.get(k)
        if v is not None and (v < lo or v > hi):
            msgs.append(f"WARNING: {OOD_LABELS[k]} = {v} is outside training range [{lo}-{hi}]. Policy may behave sub-optimally.")
    return msgs


# ─────────────────────────────────────────────────────────────────────
# Components
# ─────────────────────────────────────────────────────────────────────

@solara.component
def InfoBanner(model):
    update_counter.get()
    active   = [f for f in model.firms if getattr(f, "active", True)]
    employed = sum(1 for w in model.workers if w.employed)
    mkt_wage = float(np.mean([f.monthly_wage for f in active])) if active else 0
    solara.Info(
        f"Month {model._step}  |  Active firms: {len(active)}  |  "
        f"Employed: {employed}/{len(model.workers)}  |  "
        f"Market wage: {mkt_wage:,.0f} THB  |  "
        f"Training defaults: output_price=100, productivity=1.0, alpha=0.65, min_wage=7700, workers=100"
    )
    for msg in _ood_msgs(model.params):
        solara.Warning(msg)


@solara.component
def RLObsPanel(model):
    """Bar chart of the 12 observation values the RL policy currently sees."""
    update_counter.get()
    obs    = model._last_obs
    colors = [BETTER if v > 0.05 else (WORSE if v < -0.05 else DIM) for v in obs]

    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=BG)
    _ax(ax)
    y = np.arange(len(OBS_LABELS))
    ax.barh(y, obs, color=colors, height=0.65, edgecolor=GRID, lw=0.4)
    ax.axvline(0, color=DIM, lw=1.5)
    ax.set_yticks(y)
    ax.set_yticklabels(OBS_LABELS, fontsize=9, color=TEXT)
    ax.set_xlim(-1.6, 1.6)
    ax.set_xlabel("Observation value (clipped to +/-1.5)", fontsize=8, color=TEXT)
    act_name = ACT_NAMES[model.rl_action] if model.actions else "—"
    ax.set_title(f"RL Policy Observation  |  last action: {act_name}  (month {model._step})",
                 fontsize=10, fontweight="bold", color="white")
    for i, v in enumerate(obs):
        ha = "left" if v >= 0 else "right"
        ax.text(v + (0.04 if v >= 0 else -0.04), i, f"{v:.2f}",
                va="center", ha=ha, color=TEXT, fontsize=7.5)
    plt.tight_layout(pad=0.4)
    solara.FigureMatplotlib(fig); plt.close(fig)


@solara.component
def ActionBar(model):
    """Colour strip of the last 72 RL actions (6 wage cycles)."""
    update_counter.get()
    last = model.actions[-72:] if model.actions else []
    if not last:
        return
    fig, ax = plt.subplots(figsize=(11, 1.6), facecolor=BG)
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    for i, act in enumerate(last):
        ax.bar(i, 1, color=ACT_COLORS[act], width=1.0, align="edge")
    for t in range(0, len(last), 12):
        ax.axvline(t, color=DIM, lw=0.7, linestyle=":")
    ax.set_xlim(0, len(last)); ax.set_ylim(0, 1); ax.set_yticks([])
    ax.set_xlabel(f"Last {len(last)} months  (month {model._step})", color=TEXT, fontsize=8)
    ax.set_title("RL Firm -- Recent Actions  (dotted = annual wage window)",
                 color="white", fontsize=9, fontweight="bold", pad=4)
    patches = [mpatches.Patch(color=ACT_COLORS[k], label=ACT_NAMES[k]) for k in range(7)]
    ax.legend(handles=patches, ncol=7, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT,
              fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.6))
    plt.tight_layout(pad=0.3)
    solara.FigureMatplotlib(fig); plt.close(fig)


@solara.component
def ActionPieChart(model):
    update_counter.get()
    if not model.actions: return
    counts = np.zeros(7, dtype=int)
    for a in model.actions: counts[a] += 1
    nonzero = [(c, k) for k, c in enumerate(counts) if c > 0]
    if not nonzero: return
    vals   = [c for c, _ in nonzero]
    labels = [f"{ACT_NAMES[k]} ({c})" for c, k in nonzero]
    colors = [ACT_COLORS[k] for _, k in nonzero]
    fig, ax = plt.subplots(figsize=(5, 4.5), facecolor=BG)
    ax.set_facecolor(BG)
    wedges, _ = ax.pie(vals, colors=colors, startangle=90,
                       wedgeprops=dict(edgecolor=BG, linewidth=1.5))
    ax.legend(wedges, labels, loc="lower center", ncol=2,
              facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8,
              bbox_to_anchor=(0.5, -0.18))
    ax.set_title(f"Action Distribution  ({len(model.actions)} steps)",
                 color="white", fontsize=10, fontweight="bold")
    plt.tight_layout(pad=0.3)
    solara.FigureMatplotlib(fig); plt.close(fig)


@solara.component
def ProfitRank(model):
    update_counter.get()
    firms  = model.firms
    ranked = sorted(firms, key=lambda f: f.profit, reverse=True)
    rows   = [{"Rank": i+1,
               "Firm": f"F{firms.index(f)}" + (" [RL]" if f is model.rl_firm else ""),
               "Profit": f"{f.profit:,.0f}",
               "Wage": f.monthly_wage,
               "Workers": len(f.current_workers)}
              for i, f in enumerate(ranked)]
    rl_rank = next((r["Rank"] for r in rows if "[RL]" in r["Firm"]), "?")
    solara.Text(f"Profit Leaderboard  |  RL rank: {rl_rank}/{len(ranked)}  |  month {model._step}")
    solara.DataFrame(pd.DataFrame(rows))


@solara.component
def WageBandChart(model):
    update_counter.get()
    df = model.datacollector.get_model_vars_dataframe()
    if len(df) < 2: return
    steps = np.arange(1, len(df)+1)
    fig, ax = plt.subplots(figsize=(9, 3.2), facecolor=BG)
    _ax(ax)
    ax.fill_between(steps, df["Wage Min"], df["Wage Max"], color=H_COL, alpha=0.15, label="Market wage band")
    ax.plot(steps, df["Market Wage"], color=H_COL, lw=1.8, label="Market mean", linestyle="--")
    ax.plot(steps, df["RL Wage"],     color=RL_COL, lw=2.2, label="RL firm wage")
    ax.set_xlabel("Month", fontsize=8); ax.set_ylabel("Wage (THB)", fontsize=8)
    ax.set_title("RL Wage vs Market Band  (fill = min-max range)",
                 color="white", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
    plt.tight_layout(pad=0.4)
    solara.FigureMatplotlib(fig); plt.close(fig)


@solara.component
def Scorecard(model):
    update_counter.get()
    df = model.datacollector.get_model_vars_dataframe()
    if df.empty: return
    metrics = {"Profit":("RL Profit","Heuristic Profit"),
               "Workers":("RL Workers","Heuristic Workers"),
               "Wage":("RL Wage","Heuristic Wage")}
    labels, pcts, colors = [], [], []
    for name, (rc, hc) in metrics.items():
        rl_m = float(df[rc].mean()); h_m = float(df[hc].mean())
        pct  = (rl_m - h_m) / max(abs(h_m), 1.0) * 100
        labels.append(name); pcts.append(pct)
        colors.append(BETTER if pct >= 0 else WORSE)
    fig, ax = plt.subplots(figsize=(6, 2.8), facecolor=BG)
    _ax(ax)
    y    = np.arange(len(labels))
    bars = ax.barh(y, pcts, color=colors, height=0.45, edgecolor=GRID, lw=0.5)
    ax.axvline(0, color=DIM, lw=1.5)
    xlim = max(abs(p) for p in pcts) * 1.7 if any(pcts) else 10
    for bar, p in zip(bars, pcts):
        ha = "left" if p >= 0 else "right"; off = xlim * 0.03
        ax.text(p+(off if p>=0 else -off), bar.get_y()+bar.get_height()/2,
                f"{p:+.1f}%", ha=ha, va="center", color="white", fontsize=10, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels(labels, color=TEXT, fontsize=10)
    ax.set_xlim(-xlim, xlim)
    ax.axvspan(0, xlim, color=BETTER, alpha=0.05); ax.axvspan(-xlim, 0, color=WORSE, alpha=0.05)
    ax.set_title(f"Cumulative RL Scorecard  (month {model._step})",
                 color="white", fontsize=10, fontweight="bold")
    ax.set_xlabel("RL advantage vs heuristic (%)", color=TEXT, fontsize=9)
    plt.tight_layout(pad=0.4)
    solara.FigureMatplotlib(fig); plt.close(fig)


@solara.component
def FirmTable(model):
    update_counter.get()
    rows = []
    for i, f in enumerate(model.firms):
        labor = len(f.current_workers)
        mpl   = f.marginal_product_labor(f.productivity, labor, f.alpha) if labor > 0 else 0
        rows.append({"Firm": f"F{i}" + (" [RL]" if f is model.rl_firm else ""),
                     "Profit": round(f.profit, 0), "Workers": labor,
                     "Wage": round(f.monthly_wage, 0), "VMPL": round(mpl*f.output_price, 0),
                     "Capital": round(f.capital, 1), "Vacancies": f.vacancies,
                     "Deficit Mo": getattr(f, "deficit_months", 0)})
    solara.Text(f"All Firms -- month {model._step}")
    solara.DataFrame(pd.DataFrame(rows))


@solara.component
def WorkerTable(model):
    update_counter.get()
    rows = [{"Worker": w.unique_id, "Employed": w.employed,
              "Employer": getattr(w.employer, "uid", "--") if w.employer else "--",
              "Wage": round(w.monthly_wage, 0) if w.employed else 0,
              "Utility": round(w.utility_if_work(w.monthly_wage) if w.employed
                               else w.utility_if_not_work(), 3)}
             for w in model.workers]
    employed = sum(1 for w in model.workers if w.employed)
    solara.Text(f"Workers -- {employed} employed / {len(model.workers)} total  |  month {model._step}")
    solara.DataFrame(pd.DataFrame(rows))


@solara.component
def SurvivalChart(model):
    update_counter.get()
    df = model.datacollector.get_model_vars_dataframe()
    if len(df) < 2: return
    steps = np.arange(1, len(df)+1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 4), facecolor=BG,
                                    gridspec_kw={"height_ratios":[2,1]})
    _ax(ax1); _ax(ax2)
    ax1.plot(steps, df["Active Firms"], color=BETTER, lw=2.0)
    ax1.set_ylabel("Active Firms", fontsize=8)
    ax1.set_title("Market Health & RL Survival Pressure", color="white", fontsize=10, fontweight="bold")
    ax2.fill_between(steps, 0, df["RL Deficit Months"], color=WORSE, alpha=0.5)
    ax2.plot(steps, df["RL Deficit Months"], color=WORSE, lw=1.5)
    ax2.set_ylabel("RL Deficit\nMonths", fontsize=8); ax2.set_xlabel("Month", fontsize=8)
    plt.tight_layout(pad=0.4)
    solara.FigureMatplotlib(fig); plt.close(fig)


# Histogram / scatter components
_COL_RL = "#e74c3c"; _COL_H = "#3498db"

@solara.component
def FirmSizeHistogram(model):
    update_counter.get()
    sizes = [len(f.current_workers) for f in model.firms]
    rl_sz = len(model.rl_firm.current_workers)
    fig, ax = plt.subplots(figsize=(8,4)); _ax(ax)
    if sizes: ax.hist(sizes, bins=20, color=_COL_H, edgecolor=GRID, label="Heuristic")
    ax.axvline(rl_sz, color=_COL_RL, lw=2, linestyle="--", label=f"RL firm ({rl_sz} workers)")
    ax.set_title("Distribution of Firm Sizes"); ax.set_xlabel("Workers"); ax.set_ylabel("Count")
    ax.legend(fontsize=8); plt.tight_layout(); solara.FigureMatplotlib(fig); plt.close(fig)

@solara.component
def FirmWageHistogram(model):
    update_counter.get()
    wages = [f.monthly_wage for f in model.firms if f.monthly_wage]
    rl_w  = model.rl_firm.monthly_wage
    fig, ax = plt.subplots(figsize=(8,4)); _ax(ax)
    if wages: ax.hist(wages, bins=20, color=_COL_H, edgecolor=GRID, label="Heuristic")
    ax.axvline(rl_w, color=_COL_RL, lw=2, linestyle="--", label=f"RL firm ({rl_w:,})")
    ax.set_title("Distribution of Firm Wages"); ax.set_xlabel("Monthly Wage (THB)"); ax.set_ylabel("Count")
    ax.legend(fontsize=8); plt.tight_layout(); solara.FigureMatplotlib(fig); plt.close(fig)

@solara.component
def FirmProfitHistogram(model):
    update_counter.get()
    profits = [f.profit for f in model.firms]; rl_p = model.rl_firm.profit
    fig, ax = plt.subplots(figsize=(8,4)); _ax(ax)
    if profits: ax.hist(profits, bins=20, color=_COL_H, edgecolor=GRID, label="Heuristic")
    ax.axvline(rl_p, color=_COL_RL, lw=2, linestyle="--", label=f"RL firm ({rl_p:,.0f})")
    ax.set_title("Distribution of Firm Profits"); ax.set_xlabel("Profit (THB)"); ax.set_ylabel("Count")
    ax.legend(fontsize=8); plt.tight_layout(); solara.FigureMatplotlib(fig); plt.close(fig)

@solara.component
def WageVsMPLScatter(model):
    update_counter.get()
    wages, vmpls, colors = [], [], []
    for f in model.firms:
        if not f.monthly_wage: continue
        labor = len(f.current_workers)
        mpl   = f.marginal_product_labor(f.productivity, labor, f.alpha)
        wages.append(f.monthly_wage); vmpls.append(mpl * f.output_price)
        colors.append(_COL_RL if f is model.rl_firm else _COL_H)
    fig, ax = plt.subplots(figsize=(8,4)); _ax(ax)
    if wages:
        ax.scatter(vmpls, wages, c=colors, edgecolors=GRID, alpha=0.85, s=60, zorder=3)
        lo, hi = min(min(vmpls), min(wages)), max(max(vmpls), max(wages))
        ax.plot([lo,hi],[lo,hi], color=DIM, linestyle="--", lw=1, label="wage = VMPL")
    ax.set_title("Wage vs Value of MPL"); ax.set_xlabel("VMPL"); ax.set_ylabel("Wage (THB)")
    handles = [mpatches.Patch(color=_COL_RL, label="RL"), mpatches.Patch(color=_COL_H, label="Heuristic")]
    ax.legend(handles=handles, fontsize=8); plt.tight_layout(); solara.FigureMatplotlib(fig); plt.close(fig)

@solara.component
def WorkerUtilityHistogram(model):
    update_counter.get()
    utils = [w.utility_if_work(w.monthly_wage) if w.employed else w.utility_if_not_work() for w in model.workers]
    fig, ax = plt.subplots(figsize=(9,4)); _ax(ax)
    if utils: ax.hist(utils, bins=20, color="#9b59b6", edgecolor=GRID)
    ax.set_title("Distribution of Worker Utility"); ax.set_xlabel("Utility"); ax.set_ylabel("Count")
    plt.tight_layout(); solara.FigureMatplotlib(fig); plt.close(fig)

@solara.component
def WorkerWageHistogram(model):
    update_counter.get()
    wages = [w.monthly_wage for w in model.workers if w.employed and w.monthly_wage > 0]
    fig, ax = plt.subplots(figsize=(9,4)); _ax(ax)
    if wages: ax.hist(wages, bins=20, color="#1abc9c", edgecolor=GRID)
    ax.set_title("Distribution of Worker Wages"); ax.set_xlabel("Monthly Wage (THB)"); ax.set_ylabel("Count")
    plt.tight_layout(); solara.FigureMatplotlib(fig); plt.close(fig)


# ── make_plot_component charts ────────────────────────────────────────

chart_profit     = make_plot_component({"RL Profit": RL_COL, "Heuristic Profit": H_COL})
chart_workers    = make_plot_component({"RL Workers": RL_COL, "Heuristic Workers": H_COL})
chart_wage       = make_plot_component({"RL Wage": RL_COL, "Heuristic Wage": H_COL, "Market Wage": MAC_COL})
chart_employment = make_plot_component("Employment %")


# ─────────────────────────────────────────────────────────────────────
# SolaraViz app
# ─────────────────────────────────────────────────────────────────────

model_params = {
    # ── Market structure ──────────────────────────────────────────────
    "n_workers":              Slider("Workers  (default 100)",              100,   30,  300,   10),
    "n_firms":                Slider("Firms  (default 10)",                  10,    3,   20,    1),
    # ── Production economics ──────────────────────────────────────────
    "output_price":           Slider("Output Price  (default 100)",         100,   30,  400,   10),
    "productivity_scale":     Slider("Productivity Scale  (default 1.0)",   1.0,  0.2,  4.0,  0.1),
    "rental_rate_val":        Slider("Capital Rental Rate  (default 500)",  500,   50, 2000,   50),
    "alpha_param":            Slider("Labour Elasticity alpha (default 0.65)", 0.65, 0.20, 0.90, 0.05),
    # ── Labour market ─────────────────────────────────────────────────
    "min_wage_val":           Slider("Min Wage THB  (default 7700)",       7700, 3000,15000,  100),
    "worker_search_prob_val": Slider("Worker Search Prob %  (default 10)",   10,    1,   60,    1),
    "seed_val":               Slider("Random Seed  (default 455)",          455,    0,  999,    1),
}

page = SolaraViz(
    SoloDemoModel(),
    components=[
        InfoBanner,
        Scorecard,
        ProfitRank,
        RLObsPanel,
        ActionBar,
        ActionPieChart,
        chart_profit,
        chart_workers,
        WageBandChart,
        chart_employment,
        SurvivalChart,
        FirmTable,
        WorkerTable,
        FirmSizeHistogram,
        FirmWageHistogram,
        FirmProfitHistogram,
        WageVsMPLScatter,
        WorkerUtilityHistogram,
        WorkerWageHistogram,
    ],
    model_params=model_params,
    name="RL Labor Market -- Solo Demo",
)
