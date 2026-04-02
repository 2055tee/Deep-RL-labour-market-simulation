#!/usr/bin/env python
# reformed/demo.py
#
# Interactive Solara demo — Reformed model: 1 RL firm vs N heuristic firms.
# Run with:  solara run reformed/demo.py
#
# Sliders cover every interesting model parameter so you can explore:
#   Market structure, production economics, market-quit mechanism,
#   firm survival rules, and worker mobility — all live.

import sys
from pathlib import Path

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
    3: "#ffa726", 4: "#e53935", 5: "#43a047",
    6: "#8e24aa", 7: "#00bcd4",
}
ACT_NAMES  = ["Hold", "Wage+300", "Wage+100", "Wage-100", "Wage-300",
              "Post Vac", "Fire", "Snap"]

OBS_LABELS = [
    "profit_signal",
    "profit_change",
    "vmpl_gap",
    "wage_vs_mkt",
    "labor_ratio",
    "vacancy_ratio",
    "worker_change",
    "wage_clock",
    "prod_vs_mkt",
    "cap_vs_mkt",
    "survival_signal",
    "mkt_employment",
    "at_risk_fraction",
]

RL_FIRM_ID = "F0"


# ─────────────────────────────────────────────────────────────────────
# Demo Model
# ─────────────────────────────────────────────────────────────────────

class ReformedDemoModel(LaborMarketModel):
    """
    Wraps LaborMarketModel with live RL policy + DataCollector.
    SolaraViz re-instantiates this on every slider change.
    """

    def __init__(
        self,
        # ── Market structure ──────────────────────────────────────────
        n_workers                 = 100,
        n_firms                   = 10,
        # ── Production ───────────────────────────────────────────────
        output_price              = 100.0,
        productivity_scale        = 1.0,
        rental_rate_val           = 500,
        alpha_val                 = 65,     # labour elasticity ×100 → 0.65
        # ── Labour market ─────────────────────────────────────────────
        min_wage_val              = 7700,
        worker_search_prob_val    = 10,     # % → 0.10
        # ── Market-quit ───────────────────────────────────────────────
        market_quit_patience_val  = 4,
        market_quit_threshold_val = 91,     # % → 0.91
        # ── Firm rules ────────────────────────────────────────────────
        max_vacancies_val         = 5,
        deficit_exit_months_val   = 24,
        equal_terms_val           = 1,      # 1=narrow spread, 0=wide
        # ── Policy toggle ────────────────────────────────────────────
        use_rl_policy_val         = 1,      # 1=run RL, 0=run heuristic for F0
    ):
        self._use_rl_policy = bool(use_rl_policy_val)

        super().__init__(
            N_workers             = int(n_workers),
            N_firms               = int(n_firms),
            use_wage_gap_prob     = True,
            rl_firm_id            = RL_FIRM_ID if self._use_rl_policy else None,
            equal_terms           = bool(equal_terms_val),
            min_wage              = int(min_wage_val),
            market_quit_patience  = int(market_quit_patience_val),
            market_quit_threshold = float(market_quit_threshold_val) / 100.0,
            max_vacancies         = int(max_vacancies_val),
            deficit_exit_months   = int(deficit_exit_months_val),
        )

        alpha = float(alpha_val) / 100.0
        rr    = float(rental_rate_val)
        sp    = float(worker_search_prob_val) / 100.0

        # Override per-firm production params
        for f in self.firms:
            f.output_price  = float(output_price)
            f.productivity  = f.productivity * float(productivity_scale)
            f.rental_rate   = rr
            f.alpha         = alpha

        # Override per-worker search probability
        for w in self.workers:
            w.job_search_prob = sp

        # Re-initialise heuristic wages with updated economics
        for f in self.firms:
            if f.uid != RL_FIRM_ID:
                f.set_initial_wage(gamma=0.8)

        # RL firm snaps to updated market mean
        if self._use_rl_policy:
            rl = next((f for f in self.firms if f.uid == RL_FIRM_ID), None)
            if rl is not None:
                others   = [f.monthly_wage for f in self.firms if f is not rl]
                mkt_mean = int(round(float(np.mean(others)) / 100.0) * 100) if others else self.min_wage
                rl.fixed_wage_floor = self.min_wage
                rl.monthly_wage     = max(mkt_mean, self.min_wage)
                rl.daily_wage       = rl.monthly_wage / 20
                for w in rl.current_workers:
                    w.monthly_wage = rl.monthly_wage
                    w.daily_wage   = rl.daily_wage

        self.rl_firm      = next((f for f in self.firms if f.uid == RL_FIRM_ID),
                                  self.firms[0])
        self._policy      = _POLICY if self._use_rl_policy else None
        self._step        = 0
        self._prev_profit = 0.0
        self._prev_wkrs   = len(self.rl_firm.current_workers)
        self.actions      = []
        self._last_obs    = np.zeros(13, dtype=np.float32)
        self._market_quits_total = 0

        self.datacollector = DataCollector(model_reporters={
            "RL Profit":          lambda m: m.rl_firm.profit,
            "Heuristic Profit":   lambda m: float(np.mean(
                [f.profit for f in m.active_firms() if f is not m.rl_firm] or [0]
            )),
            "RL Workers":         lambda m: len(m.rl_firm.current_workers),
            "Heuristic Workers":  lambda m: float(np.mean(
                [len(f.current_workers) for f in m.active_firms() if f is not m.rl_firm] or [0]
            )),
            "RL Wage":            lambda m: m.rl_firm.monthly_wage,
            "Heuristic Wage":     lambda m: float(np.mean(
                [f.monthly_wage for f in m.active_firms() if f is not m.rl_firm] or [0]
            )),
            "Market Wage":        lambda m: float(np.mean(
                [f.monthly_wage for f in m.active_firms()] or [0]
            )),
            "Wage Min":           lambda m: float(np.min(
                [f.monthly_wage for f in m.active_firms()] or [0]
            )),
            "Wage Max":           lambda m: float(np.max(
                [f.monthly_wage for f in m.active_firms()] or [0]
            )),
            "Employment %":       lambda m: 100.0 * sum(1 for w in m.workers if w.employed)
                                             / max(len(m.workers), 1),
            "Active Firms":       lambda m: len(m.active_firms()),
            "RL Deficit Months":  lambda m: m.rl_firm.deficit_months,
            "At Risk Workers":    lambda m: sum(
                1 for w in m.rl_firm.current_workers
                if w.months_below_mkt >= max(m.market_quit_patience // 2, 1)
            ),
        })

    # ── Observation (mirrors firm_env.py exactly) ─────────────────────

    def _obs(self):
        firm  = self.rl_firm
        labor = len(firm.current_workers)

        profit_signal        = float(np.tanh(firm.profit / 5_000))
        profit_change_signal = float(np.tanh((firm.profit - self._prev_profit) / 2_000))

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

        patience = self.market_quit_patience
        at_risk  = sum(1 for w in firm.current_workers
                       if w.months_below_mkt >= patience // 2)
        at_risk_fraction = at_risk / max(labor, 1)

        obs = np.array([
            profit_signal, profit_change_signal, vmpl_gap, wage_vs_mkt,
            labor_ratio, vacancy_ratio, worker_change, wage_clock,
            prod_vs_mkt, cap_vs_mkt, survival_signal, mkt_employment,
            at_risk_fraction,
        ], dtype=np.float32)
        return np.clip(obs, -1.5, 1.5)

    def _action_mask(self):
        wage_ok = self._step % 12 == 0
        return np.array([True, wage_ok, wage_ok, wage_ok, wage_ok,
                         True, True, True], dtype=bool)

    # ── Step ──────────────────────────────────────────────────────────

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
            self.rl_action  = int(act[0])
            self._last_obs  = obs
        else:
            self.rl_action = 0

        # Track market-quit events
        employed_before = sum(1 for w in self.workers if w.employed)
        super().step()
        employed_after  = sum(1 for w in self.workers if w.employed)
        if employed_after < employed_before:
            self._market_quits_total += employed_before - employed_after

        self._step += 1
        self.actions.append(self.rl_action)
        self.datacollector.collect(self)


# ─────────────────────────────────────────────────────────────────────
# Styling helper
# ─────────────────────────────────────────────────────────────────────

def _ax(ax, facecolor=PANEL):
    ax.set_facecolor(facecolor)
    ax.tick_params(colors=TEXT, labelsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color("white")
    ax.grid(color=GRID, alpha=0.28, lw=0.5)


# ─────────────────────────────────────────────────────────────────────
# Components
# ─────────────────────────────────────────────────────────────────────

@solara.component
def InfoBanner(model):
    """Live status — updates each step."""
    update_counter.get()
    active = model.active_firms()
    employed = sum(1 for w in model.workers if w.employed)
    mkt_wage = float(np.mean([f.monthly_wage for f in active])) if active else 0
    at_risk  = sum(1 for w in model.rl_firm.current_workers
                   if w.months_below_mkt >= max(model.market_quit_patience // 2, 1))
    policy_label = "RL policy ACTIVE" if model._use_rl_policy else "Heuristic mode (no RL)"
    solara.Info(
        f"Month {model._step}  |  {policy_label}  |  "
        f"Active firms: {len(active)}  |  "
        f"Employed: {employed}/{len(model.workers)}  |  "
        f"Market wage: {mkt_wage:,.0f} THB  |  "
        f"RL at-risk workers: {at_risk}  |  "
        f"Patience: {model.market_quit_patience}mo  "
        f"Threshold: {model.market_quit_threshold:.0%}"
    )


@solara.component
def FirmTableRL(model):
    """Live per-firm snapshot — RL firm highlighted."""
    update_counter.get()
    active = model.active_firms()
    rows   = []
    for f in active:
        labor = len(f.current_workers)
        mpl   = f.marginal_product_labor(f.productivity, labor, f.alpha) if labor > 0 else 0
        vmpl  = mpl * f.output_price
        rows.append({
            "Firm":       f.uid + (" [RL]" if f.uid == RL_FIRM_ID else ""),
            "Profit":     round(f.profit, 0),
            "Workers":    labor,
            "Wage":       round(f.monthly_wage, 0),
            "VMPL":       round(vmpl, 0),
            "Capital":    round(f.capital, 1),
            "Vacancies":  f.vacancies,
            "Deficit Mo": f.deficit_months,
        })
    solara.Text(f"All Firms — month {model._step}  |  Active: {len(active)}")
    solara.DataFrame(pd.DataFrame(rows))


@solara.component
def WorkerTableRL(model):
    """Live worker table with market-quit counter."""
    update_counter.get()
    rows = []
    for w in model.workers:
        rows.append({
            "Worker":   w.uid,
            "Employer": w.employer.uid if w.employer else "—",
            "Employed": w.employed,
            "Wage":     w.monthly_wage if w.employed else 0,
            "Utility":  round(
                w.utility_if_work(w.monthly_wage) if w.employed
                else w.utility_if_not_work(), 3
            ),
            "Mkt-quit cnt": w.months_below_mkt,
        })
    df = pd.DataFrame(rows)
    employed = df["Employed"].sum()
    solara.Text(f"Workers — {employed} employed / {len(df)} total  |  month {model._step}")
    solara.DataFrame(df)


@solara.component
def RLObsPanel(model):
    """
    Horizontal bar chart of the 13 observation values the RL policy
    currently 'sees'.  Positive = green, negative = red, near-zero = grey.
    Great for understanding WHY the policy picked a certain action.
    """
    update_counter.get()
    obs    = model._last_obs
    labels = OBS_LABELS
    colors = [BETTER if v > 0.05 else (WORSE if v < -0.05 else DIM) for v in obs]

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=BG)
    _ax(ax)
    y = np.arange(len(labels))
    ax.barh(y, obs, color=colors, height=0.65, edgecolor=GRID, lw=0.4)
    ax.axvline(0, color=DIM, lw=1.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9, color=TEXT)
    ax.set_xlim(-1.6, 1.6)
    ax.set_xlabel("Observation value  (clipped to ±1.5)", fontsize=8, color=TEXT)

    action_name = ACT_NAMES[model.rl_action] if model._use_rl_policy else "heuristic"
    ax.set_title(f"RL Policy Observation  |  last action: {action_name}  (month {model._step})",
                 fontsize=10, fontweight="bold", color="white")
    # Annotate values
    for i, v in enumerate(obs):
        ha = "left" if v >= 0 else "right"
        ax.text(v + (0.04 if v >= 0 else -0.04), i, f"{v:.2f}",
                va="center", ha=ha, color=TEXT, fontsize=7.5)
    plt.tight_layout(pad=0.4)
    solara.FigureMatplotlib(fig)
    plt.close(fig)


@solara.component
def ActionBar(model):
    """Colour strip of the last 72 RL actions (6 wage cycles)."""
    update_counter.get()
    last = model.actions[-72:] if model.actions else []
    if not last:
        return
    fig, ax = plt.subplots(figsize=(11, 1.6), facecolor=BG)
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    for i, act in enumerate(last):
        ax.bar(i, 1, color=ACT_COLORS[act], width=1.0, align="edge")
    # Mark wage-cycle boundaries
    for t in range(0, len(last), 12):
        ax.axvline(t, color=DIM, lw=0.7, linestyle=":")
    ax.set_xlim(0, len(last))
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel(f"Last {len(last)} months  (month {model._step})", color=TEXT, fontsize=8)
    ax.set_title("RL Firm — Recent Actions  (dotted = annual wage window)",
                 color="white", fontsize=9, fontweight="bold", pad=4)
    patches = [mpatches.Patch(color=ACT_COLORS[k], label=ACT_NAMES[k]) for k in range(8)]
    ax.legend(handles=patches, ncol=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT,
              fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.6))
    plt.tight_layout(pad=0.3)
    solara.FigureMatplotlib(fig)
    plt.close(fig)


@solara.component
def ActionPieChart(model):
    """Pie chart of cumulative action distribution."""
    update_counter.get()
    if not model.actions:
        return
    counts = np.zeros(8, dtype=int)
    for a in model.actions:
        counts[a] += 1
    nonzero = [(c, k) for k, c in enumerate(counts) if c > 0]
    if not nonzero:
        return
    vals   = [c for c, _ in nonzero]
    labels = [f"{ACT_NAMES[k]} ({c})" for c, k in nonzero]
    colors = [ACT_COLORS[k] for _, k in nonzero]

    fig, ax = plt.subplots(figsize=(5, 4.5), facecolor=BG)
    ax.set_facecolor(BG)
    wedges, _ = ax.pie(vals, colors=colors, startangle=90,
                       wedgeprops=dict(edgecolor=BG, linewidth=1.5))
    ax.legend(wedges, labels, loc="lower center", ncol=2,
              facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT,
              fontsize=8, bbox_to_anchor=(0.5, -0.18))
    ax.set_title(f"Action Distribution  ({len(model.actions)} steps)",
                 color="white", fontsize=10, fontweight="bold")
    plt.tight_layout(pad=0.3)
    solara.FigureMatplotlib(fig)
    plt.close(fig)


@solara.component
def MarketQuitRisk(model):
    """
    Shows how close each RL firm worker is to quitting the market.
    Bar per worker coloured by months_below_mkt; red line = patience threshold.
    """
    update_counter.get()
    workers = list(model.rl_firm.current_workers)
    if not workers:
        solara.Text("RL firm has no workers.")
        return

    patience  = model.market_quit_patience
    half_pat  = patience // 2
    months    = [w.months_below_mkt for w in workers]
    bar_cols  = [AT_COL if m >= half_pat else (WARN if m > 0 else DIM) for m in months]

    fig, ax = plt.subplots(figsize=(10, 2.5), facecolor=BG)
    _ax(ax)
    x = np.arange(len(months))
    ax.bar(x, months, color=bar_cols, width=0.8)
    ax.axhline(half_pat, color=WARN,   lw=1.5, linestyle="--",
               label=f"Warning ({half_pat}mo)")
    ax.axhline(patience, color=AT_COL, lw=1.5, linestyle="--",
               label=f"Patience ({patience}mo)")
    ax.set_xlim(-0.5, len(months) - 0.5)
    ax.set_ylim(0, patience + 2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(w.uid) for w in workers], fontsize=7,
                       rotation=90, color=TEXT)
    ax.set_ylabel("Months below mkt", fontsize=8, color=TEXT)
    ax.set_title(f"RL Workers — Market-Quit Risk  (month {model._step})",
                 color="white", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
    plt.tight_layout(pad=0.4)
    solara.FigureMatplotlib(fig)
    plt.close(fig)


@solara.component
def WageBandChart(model):
    """
    Shows RL wage position within the market wage band (min / mean / max).
    Helps visualise whether the RL firm is a premium or discount payer.
    """
    update_counter.get()
    df = model.datacollector.get_model_vars_dataframe()
    if len(df) < 2:
        return
    steps = np.arange(1, len(df) + 1)

    fig, ax = plt.subplots(figsize=(9, 3.2), facecolor=BG)
    _ax(ax)

    ax.fill_between(steps, df["Wage Min"], df["Wage Max"],
                    color=H_COL, alpha=0.15, label="Market wage band")
    ax.plot(steps, df["Market Wage"],  color=H_COL,  lw=1.8, label="Market mean",   linestyle="--")
    ax.plot(steps, df["RL Wage"],      color=RL_COL, lw=2.2, label="RL firm wage")

    ax.set_xlabel("Month", fontsize=8)
    ax.set_ylabel("Wage (THB)", fontsize=8)
    ax.set_title("RL Wage vs Market Band  (fill = min–max range)",
                 color="white", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
    plt.tight_layout(pad=0.4)
    solara.FigureMatplotlib(fig)
    plt.close(fig)


@solara.component
def Scorecard(model):
    """Cumulative RL advantage vs heuristic baseline (%)."""
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
    for name, (rc, hc) in metrics.items():
        rl_m  = float(df[rc].mean())
        h_m   = float(df[hc].mean())
        pct   = (rl_m - h_m) / max(abs(h_m), 1.0) * 100
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
    ax.set_xlabel("RL advantage vs heuristic avg (%)", color=TEXT, fontsize=9)
    plt.tight_layout(pad=0.4)
    solara.FigureMatplotlib(fig)
    plt.close(fig)


@solara.component
def ProfitRank(model):
    """Live profit leaderboard — shows RL firm's rank among all active firms."""
    update_counter.get()
    active = model.active_firms()
    if not active:
        return
    ranked = sorted(active, key=lambda f: f.profit, reverse=True)
    rows   = [
        {
            "Rank":   i + 1,
            "Firm":   f.uid + (" [RL]" if f.uid == RL_FIRM_ID else ""),
            "Profit": f"{f.profit:,.0f}",
            "Wage":   f.monthly_wage,
            "Workers": len(f.current_workers),
        }
        for i, f in enumerate(ranked)
    ]
    rl_rank = next((r["Rank"] for r in rows if "[RL]" in r["Firm"]), "—")
    solara.Text(f"Profit Leaderboard  |  RL rank: {rl_rank} / {len(ranked)}  |  month {model._step}")
    solara.DataFrame(pd.DataFrame(rows))


@solara.component
def SurvivalChart(model):
    """Tracks active firms count and RL deficit months over time."""
    update_counter.get()
    df = model.datacollector.get_model_vars_dataframe()
    if len(df) < 2:
        return
    steps = np.arange(1, len(df) + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 4), facecolor=BG,
                                    gridspec_kw={"height_ratios": [2, 1]})
    _ax(ax1); _ax(ax2)

    ax1.plot(steps, df["Active Firms"], color=BETTER, lw=2.0)
    ax1.set_ylabel("Active Firms", fontsize=8)
    ax1.set_title("Market Health & RL Survival Pressure",
                  color="white", fontsize=10, fontweight="bold")

    ax2.fill_between(steps, 0, df["RL Deficit Months"],
                     color=WORSE, alpha=0.5)
    ax2.plot(steps, df["RL Deficit Months"], color=WORSE, lw=1.5)
    ax2.set_ylabel("RL Deficit\nMonths", fontsize=8)
    ax2.set_xlabel("Month", fontsize=8)

    plt.tight_layout(pad=0.4)
    solara.FigureMatplotlib(fig)
    plt.close(fig)


@solara.component
def AtRiskChart(model):
    """Time series of RL at-risk workers (approaching market-quit)."""
    update_counter.get()
    df = model.datacollector.get_model_vars_dataframe()
    if len(df) < 2:
        return
    steps = np.arange(1, len(df) + 1)

    fig, ax = plt.subplots(figsize=(9, 2.8), facecolor=BG)
    _ax(ax)
    ax.fill_between(steps, 0, df["At Risk Workers"], color=AT_COL, alpha=0.4)
    ax.plot(steps, df["At Risk Workers"], color=AT_COL, lw=2.0,
            label="Workers ≥ patience/2 months below market")
    ax.set_xlabel("Month", fontsize=8)
    ax.set_ylabel("At-Risk Workers", fontsize=8)
    ax.set_title("RL Firm — Workers Approaching Market-Quit",
                 color="white", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
    plt.tight_layout(pad=0.4)
    solara.FigureMatplotlib(fig)
    plt.close(fig)


# ── make_plot_component line charts ──────────────────────────────────

chart_profit     = make_plot_component({"RL Profit":  RL_COL, "Heuristic Profit":  H_COL})
chart_workers    = make_plot_component({"RL Workers": RL_COL, "Heuristic Workers": H_COL})
chart_wage       = make_plot_component({"RL Wage": RL_COL, "Heuristic Wage": H_COL, "Market Wage": MAC_COL})
chart_employment = make_plot_component("Employment %")


# ─────────────────────────────────────────────────────────────────────
# SolaraViz app
# ─────────────────────────────────────────────────────────────────────

model_params = {
    # ── Policy ──────────────────────────────────────────────────────
    "use_rl_policy_val":          Slider("Use RL Policy  (1=RL, 0=heuristic F0)",    1,    0,    1,    1),
    # ── Market structure ────────────────────────────────────────────
    "n_workers":                  Slider("Workers  (default 100)",                  100,   30,  300,   10),
    "n_firms":                    Slider("Firms  (default 10)",                      10,    3,   20,    1),
    # ── Production economics ─────────────────────────────────────────
    "output_price":               Slider("Output Price  (default 100)",             100,   30,  400,   10),
    "productivity_scale":         Slider("Productivity Scale  (default 1.0)",       1.0,  0.2,  4.0,  0.1),
    "rental_rate_val":            Slider("Capital Rental Rate  (default 500)",      500,   50, 2000,   50),
    "alpha_val":                  Slider("Labour Elasticity α×100  (default 65)",    65,   20,   90,    5),
    # ── Labour market ─────────────────────────────────────────────────
    "min_wage_val":               Slider("Min Wage THB  (default 7700)",           7700, 3000,20000,  500),
    "worker_search_prob_val":     Slider("Worker Search Prob %  (default 10)",       10,    1,   60,    1),
    # ── Market-quit mechanism ────────────────────────────────────────
    "market_quit_patience_val":   Slider("Market-Quit Patience months  (default 4)",  4,    1,   16,    1),
    "market_quit_threshold_val":  Slider("Market-Quit Threshold %  (default 91)",    91,   60,  100,    1),
    # ── Firm rules ────────────────────────────────────────────────────
    "max_vacancies_val":          Slider("RL Vacancy Cap  (default 5)",               5,    1,   20,    1),
    "deficit_exit_months_val":    Slider("Exit After N Deficit Months  (default 24)", 24,   3,   72,    3),
    # ── Starting conditions ───────────────────────────────────────────
    "equal_terms_val":            Slider("Equal Terms  (1=narrow spread, 0=wide)",    1,    0,    1,    1),
}

page = SolaraViz(
    ReformedDemoModel(),
    components=[
        # ── Overview ────────────────────────────────────────────────
        InfoBanner,
        Scorecard,
        ProfitRank,
        # ── RL brain ────────────────────────────────────────────────
        RLObsPanel,
        ActionBar,
        ActionPieChart,
        # ── Worker dynamics ─────────────────────────────────────────
        MarketQuitRisk,
        AtRiskChart,
        # ── Time series ─────────────────────────────────────────────
        chart_profit,
        chart_workers,
        WageBandChart,
        chart_employment,
        SurvivalChart,
        # ── Snapshot tables ─────────────────────────────────────────
        FirmTableRL,
        WorkerTableRL,
        # ── Distribution plots (from rl_vis.py) ─────────────────────
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
    name="Reformed RL Labor Market — Interactive Demo",
)
