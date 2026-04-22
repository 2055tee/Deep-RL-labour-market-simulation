#!/usr/bin/env python
# demo/demo_coop.py
#
# Interactive demo -- Cooperative: 3 RL firms share reward and are
# trained to collaborate.  Reformed rules apply.
#
# Run with:  solara run demo/demo_coop.py

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()

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
import importlib.util

# ── Load model module ─────────────────────────────────────────────────
def _load_mod(folder):
    spec = importlib.util.spec_from_file_location(
        f"model_rl_{folder}", ROOT / folder / "model_rl.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_MOD = _load_mod("cooperative")

# ── Load policy ───────────────────────────────────────────────────────
try:
    from sb3_contrib import MaskablePPO
    _POLICY = MaskablePPO.load(str(ROOT / "cooperative" / "coop_model_longrun"))
    print("[demo_coop] Cooperative policy loaded.")
except Exception as _e:
    _POLICY = None
    print(f"[demo_coop] Could not load coop model: {_e}")

N_RL_FIRMS = 3

# ── Palette ───────────────────────────────────────────────────────────
BG      = "white"
PANEL   = "#f0f0f0"
GRID    = "#cccccc"
TEXT    = "black"
DIM     = "#555555"
BETTER  = "#00c853"
WORSE   = "#ff1744"
WARN    = "#ffab40"
RL_COL  = "#4fc3f7"
H_COL   = "#ffb74d"
MAC_COL = "#ce93d8"

RL_FIRM_COLORS = ["#4fc3f7", "#ef5350", "#66bb6a"]

ACT_COLORS = {
    0: "#555577", 1: "#1565c0", 2: "#64b5f6",
    3: "#ffa726", 4: "#e53935", 5: "#43a047",
    6: "#8e24aa", 7: "#00bcd4",
}
ACT_NAMES = ["Hold", "Wage+300", "Wage+100", "Wage-100", "Wage-300",
             "Post Vac", "Fire", "Snap"]

OBS_LABELS = [
    "profit_signal", "profit_change", "vmpl_gap", "wage_vs_mkt",
    "labor_ratio", "vacancy_ratio", "worker_change", "wage_clock",
    "prod_vs_mkt", "cap_vs_mkt", "survival_signal",
    "team_profit_signal", "at_risk_fraction",
]

DEFAULTS = dict(
    n_workers=100, n_firms=10, output_price=100.0,
    productivity_scale=1.0, alpha_param=65, min_wage_val=7700,
    rental_rate_val=500, worker_search_prob_val=10,
    market_quit_patience_val=4, market_quit_threshold_val=91,
    max_vacancies_val=5, deficit_exit_months_val=24,
    equal_terms_val=1, seed_val=455,
)


# ─────────────────────────────────────────────────────────────────────
# Demo model
# ─────────────────────────────────────────────────────────────────────

def _make_model(n_workers, n_firms, output_price, productivity_scale,
                alpha_param, min_wage_val, rental_rate_val, worker_search_prob_val,
                market_quit_patience_val, market_quit_threshold_val,
                max_vacancies_val, deficit_exit_months_val, equal_terms_val,
                seed_val):

    Base   = _MOD.LaborMarketModel
    alpha  = float(alpha_param) / 100.0
    rr     = float(rental_rate_val)
    sp     = float(worker_search_prob_val) / 100.0

    class _DemoModel(Base):
        def __init__(self):
            super().__init__(
                N_workers=int(n_workers), N_firms=int(n_firms),
                n_rl_firms=N_RL_FIRMS,
                use_wage_gap_prob=True,
                equal_terms=bool(equal_terms_val),
                min_wage=int(min_wage_val),
                market_quit_patience=int(market_quit_patience_val),
                market_quit_threshold=float(market_quit_threshold_val) / 100.0,
                max_vacancies=int(max_vacancies_val),
                deficit_exit_months=int(deficit_exit_months_val),
                seed=int(seed_val),
            )
            self._mode_label = "Cooperative"

            for f in self.firms:
                f.output_price = float(output_price)
                f.productivity = f.productivity * float(productivity_scale)
                f.alpha        = alpha
                f.rental_rate  = rr
            for w in self.workers:
                w.job_search_prob = sp
            for f in self.firms:
                if f.uid not in self.rl_firm_ids:
                    f.set_initial_wage(gamma=0.8)

            self._policy        = _POLICY
            self.rl_firms       = self.firms[:N_RL_FIRMS]
            self._step          = 0
            self._prev_profit   = {f.uid: 0.0                    for f in self.rl_firms}
            self._prev_workers  = {f.uid: len(f.current_workers) for f in self.rl_firms}
            self.actions_each   = [[] for _ in range(N_RL_FIRMS)]
            self._last_obs_each = [np.zeros(13, dtype=np.float32) for _ in range(N_RL_FIRMS)]

            self.datacollector = DataCollector(model_reporters={
                "RL Avg Profit":     lambda m: float(np.mean([f.profit for f in m.rl_firms])),
                "Heuristic Profit":  lambda m: float(np.mean([f.profit for f in m.active_firms() if f not in m.rl_firms] or [0])),
                "RL F0 Profit":      lambda m: m.rl_firms[0].profit,
                "RL F1 Profit":      lambda m: m.rl_firms[1].profit if len(m.rl_firms) > 1 else 0,
                "RL F2 Profit":      lambda m: m.rl_firms[2].profit if len(m.rl_firms) > 2 else 0,
                "RL Avg Workers":    lambda m: float(np.mean([len(f.current_workers) for f in m.rl_firms])),
                "Heuristic Workers": lambda m: float(np.mean([len(f.current_workers) for f in m.active_firms() if f not in m.rl_firms] or [0])),
                "RL Avg Wage":       lambda m: float(np.mean([f.monthly_wage for f in m.rl_firms])),
                "Heuristic Wage":    lambda m: float(np.mean([f.monthly_wage for f in m.active_firms() if f not in m.rl_firms] or [0])),
                "Market Wage":       lambda m: float(np.mean([f.monthly_wage for f in m.active_firms()] or [0])),
                "Wage Min":          lambda m: float(np.min([f.monthly_wage for f in m.active_firms()] or [0])),
                "Wage Max":          lambda m: float(np.max([f.monthly_wage for f in m.active_firms()] or [0])),
                "Employment %":      lambda m: 100.0 * sum(1 for w in m.workers if w.employed) / max(len(m.workers), 1),
                "Active Firms":      lambda m: len(m.active_firms()),
                "RL Avg Deficit":    lambda m: float(np.mean([f.deficit_months for f in m.rl_firms])),
                "Wage Spread (RL)":  lambda m: float(np.max([f.monthly_wage for f in m.rl_firms]) -
                                                      np.min([f.monthly_wage for f in m.rl_firms])),
            })

        def _obs(self, idx):
            firm  = self.rl_firms[idx]
            labor = len(firm.current_workers)

            profit_signal        = float(np.tanh(firm.profit / 5_000))
            profit_change_signal = float(np.tanh(
                (firm.profit - self._prev_profit[firm.uid]) / 2_000))

            if labor > 0:
                mpl      = firm.marginal_product_labor(firm.productivity, labor, firm.alpha)
                vmpl     = mpl * firm.output_price
                vmpl_gap = float(np.tanh((vmpl - firm.monthly_wage) / max(firm.monthly_wage, 1.0)))
            else:
                vmpl_gap = 1.0

            all_wages   = [f.monthly_wage for f in self.firms]
            market_wage = float(np.mean(all_wages))
            wage_vs_mkt = float(np.tanh((firm.monthly_wage - market_wage) / max(market_wage, 1.0)))

            labor_ratio   = labor / 40.0
            vacancy_ratio = min(firm.vacancies, 5) / 5.0
            worker_change = float(np.tanh((labor - self._prev_workers[firm.uid]) / 3.0))
            wage_clock    = (self._step % 12) / 11.0

            avg_prod    = float(np.mean([f.productivity for f in self.firms]))
            avg_cap     = float(np.mean([f.capital      for f in self.firms]))
            prod_vs_mkt = float(np.tanh((firm.productivity - avg_prod) / max(avg_prod, 1.0)))
            cap_vs_mkt  = float(np.tanh((firm.capital      - avg_cap)  / max(avg_cap,  1.0)))

            survival_signal    = float(np.tanh(firm.deficit_months / 12.0))
            peer_profits       = [f.profit for f in self.rl_firms]
            team_profit_signal = float(np.tanh(float(np.mean(peer_profits)) / 5_000))

            patience = self.market_quit_patience
            at_risk  = sum(1 for w in firm.current_workers if w.months_below_mkt >= patience // 2)
            at_risk_fraction = at_risk / max(labor, 1)

            obs = np.array([
                profit_signal, profit_change_signal, vmpl_gap, wage_vs_mkt,
                labor_ratio, vacancy_ratio, worker_change, wage_clock,
                prod_vs_mkt, cap_vs_mkt, survival_signal,
                team_profit_signal, at_risk_fraction,
            ], dtype=np.float32)
            return np.clip(obs, -1.5, 1.5)

        def _action_mask(self):
            wage_ok = self._step % 12 == 0
            return np.array([True, wage_ok, wage_ok, wage_ok, wage_ok,
                             True, True, True], dtype=bool)

        def step(self):
            for f in self.rl_firms:
                self._prev_profit[f.uid]  = f.profit
                self._prev_workers[f.uid] = len(f.current_workers)

            mask = self._action_mask()
            for idx, firm in enumerate(self.rl_firms):
                if self._policy is not None:
                    obs = self._obs(idx)
                    act, _ = self._policy.predict(obs[np.newaxis], deterministic=True,
                                                  action_masks=mask[np.newaxis])
                    firm.rl_action           = int(act[0])
                    self._last_obs_each[idx] = obs
                else:
                    firm.rl_action = 0

            super().step()
            self._step += 1
            for idx, firm in enumerate(self.rl_firms):
                self.actions_each[idx].append(firm.rl_action)
            self.datacollector.collect(self)

    return _DemoModel()


class CoopDemoProxy:
    def __init__(self,
                 n_workers                 = DEFAULTS["n_workers"],
                 n_firms                   = DEFAULTS["n_firms"],
                 output_price              = DEFAULTS["output_price"],
                 productivity_scale        = DEFAULTS["productivity_scale"],
                 alpha_param               = DEFAULTS["alpha_param"],
                 min_wage_val              = DEFAULTS["min_wage_val"],
                 rental_rate_val           = DEFAULTS["rental_rate_val"],
                 worker_search_prob_val    = DEFAULTS["worker_search_prob_val"],
                 market_quit_patience_val  = DEFAULTS["market_quit_patience_val"],
                 market_quit_threshold_val = DEFAULTS["market_quit_threshold_val"],
                 max_vacancies_val         = DEFAULTS["max_vacancies_val"],
                 deficit_exit_months_val   = DEFAULTS["deficit_exit_months_val"],
                 equal_terms_val           = DEFAULTS["equal_terms_val"],
                 seed_val                  = DEFAULTS["seed_val"]):

        self._inner = _make_model(
            n_workers=n_workers, n_firms=n_firms,
            output_price=output_price, productivity_scale=productivity_scale,
            alpha_param=alpha_param, min_wage_val=min_wage_val,
            rental_rate_val=rental_rate_val,
            worker_search_prob_val=worker_search_prob_val,
            market_quit_patience_val=market_quit_patience_val,
            market_quit_threshold_val=market_quit_threshold_val,
            max_vacancies_val=max_vacancies_val,
            deficit_exit_months_val=deficit_exit_months_val,
            equal_terms_val=equal_terms_val,
            seed_val=seed_val,
        )
        self.schedule = self._inner.schedule

    def step(self):
        self._inner.step()

    def __getattr__(self, name):
        return getattr(self._inner, name)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _ax(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    ax.xaxis.label.set_color(TEXT); ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT); ax.grid(color=GRID, alpha=0.28, lw=0.5)


# ─────────────────────────────────────────────────────────────────────
# Components
# ─────────────────────────────────────────────────────────────────────

@solara.component
def InfoBanner(model):
    update_counter.get()
    m        = model._inner
    active   = m.active_firms()
    employed = sum(1 for w in m.workers if w.employed)
    mkt_wage = float(np.mean([f.monthly_wage for f in active])) if active else 0
    solara.Info(
        f"Cooperative  |  Month {m._step}  |  Active firms: {len(active)}  |  "
        f"Employed: {employed}/{len(m.workers)}  |  "
        f"Market wage: {mkt_wage:,.0f} THB  |  "
        f"Patience: {m.market_quit_patience}mo  Threshold: {m.market_quit_threshold:.0%}"
    )


@solara.component
def Scorecard(model):
    update_counter.get()
    m  = model._inner
    df = m.datacollector.get_model_vars_dataframe()
    if df.empty: return
    metrics = {"Profit":("RL Avg Profit","Heuristic Profit"),
               "Workers":("RL Avg Workers","Heuristic Workers"),
               "Wage":("RL Avg Wage","Heuristic Wage")}
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
                f"{p:+.1f}%", ha=ha, va="center", color=TEXT, fontsize=10, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels(labels, color=TEXT, fontsize=10)
    ax.set_xlim(-xlim, xlim)
    ax.axvspan(0, xlim, color=BETTER, alpha=0.05); ax.axvspan(-xlim, 0, color=WORSE, alpha=0.05)
    ax.set_title(f"Cumulative RL Scorecard  [Cooperative]  (month {m._step})",
                 color=TEXT, fontsize=10, fontweight="bold")
    ax.set_xlabel("RL avg vs Heuristic avg (%)", color=TEXT, fontsize=9)
    plt.tight_layout(pad=0.4); solara.FigureMatplotlib(fig); plt.close(fig)


@solara.component
def ProfitRank(model):
    update_counter.get()
    m      = model._inner
    active = m.active_firms()
    ranked = sorted(active, key=lambda f: f.profit, reverse=True)
    rl_set = {id(f) for f in m.rl_firms}
    rows   = [{"Rank": i+1,
               "Firm": f.uid + (" [RL]" if id(f) in rl_set else ""),
               "Profit": f"{f.profit:,.0f}", "Wage": f.monthly_wage,
               "Workers": len(f.current_workers)}
              for i, f in enumerate(ranked)]
    rl_ranks = [r["Rank"] for r in rows if "[RL]" in r["Firm"]]
    solara.Text(f"Profit Leaderboard  [Cooperative]  |  RL ranks: {rl_ranks}  |  month {m._step}")
    solara.DataFrame(pd.DataFrame(rows))


@solara.component
def RLObsPanels(model):
    update_counter.get()
    m = model._inner
    fig, axes = plt.subplots(N_RL_FIRMS, 1, figsize=(9, 3.5 * N_RL_FIRMS), facecolor=BG)
    if N_RL_FIRMS == 1: axes = [axes]
    for idx, ax in enumerate(axes):
        _ax(ax)
        obs    = m._last_obs_each[idx]
        colors = [BETTER if v > 0.05 else (WORSE if v < -0.05 else DIM) for v in obs]
        y = np.arange(len(OBS_LABELS))
        ax.barh(y, obs, color=colors, height=0.65, edgecolor=GRID, lw=0.4)
        ax.axvline(0, color=DIM, lw=1.2)
        ax.set_yticks(y); ax.set_yticklabels(OBS_LABELS, fontsize=8, color=TEXT)
        ax.set_xlim(-1.6, 1.6)
        last_act = ACT_NAMES[m.actions_each[idx][-1]] if m.actions_each[idx] else "--"
        ax.set_title(f"F{idx} [Cooperative]  last action: {last_act}",
                     color=RL_FIRM_COLORS[idx], fontsize=9, fontweight="bold")
        for i, v in enumerate(obs):
            ha = "left" if v >= 0 else "right"
            ax.text(v+(0.04 if v>=0 else -0.04), i, f"{v:.2f}",
                    va="center", ha=ha, color=TEXT, fontsize=7)
    fig.suptitle(f"RL Policy Observations  (month {m._step})",
                 color=TEXT, fontsize=11, fontweight="bold")
    plt.tight_layout(pad=0.4); solara.FigureMatplotlib(fig); plt.close(fig)


@solara.component
def ActionGrid(model):
    update_counter.get()
    m = model._inner
    if not any(m.actions_each[0]): return
    fig, axes = plt.subplots(N_RL_FIRMS, 1, figsize=(11, 1.2*N_RL_FIRMS+0.8), facecolor=BG)
    if N_RL_FIRMS == 1: axes = [axes]
    fig.suptitle(f"RL Firm Actions -- Last 72 months  [Cooperative]",
                 color=TEXT, fontsize=9, fontweight="bold", y=1.01)
    for idx, ax in enumerate(axes):
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID)
        last = m.actions_each[idx][-72:]
        for i, act in enumerate(last):
            ax.bar(i, 1, color=ACT_COLORS[act], width=1.0, align="edge")
        for t in range(0, len(last), 12):
            ax.axvline(t, color=DIM, lw=0.7, linestyle=":")
        ax.set_xlim(0, len(last)); ax.set_ylim(0, 1); ax.set_yticks([])
        ax.set_ylabel(f"F{idx}", color=RL_FIRM_COLORS[idx], fontsize=8, rotation=0, labelpad=20)
    axes[-1].set_xlabel(f"Month offset (current: {m._step})", color=TEXT, fontsize=8)
    patches = [mpatches.Patch(color=ACT_COLORS[k], label=ACT_NAMES[k]) for k in range(8)]
    axes[-1].legend(handles=patches, ncol=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT,
                    fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.7))
    plt.tight_layout(pad=0.3); solara.FigureMatplotlib(fig); plt.close(fig)


@solara.component
def ActionPieGrid(model):
    update_counter.get()
    m = model._inner
    if not any(m.actions_each[0]): return
    fig, axes = plt.subplots(1, N_RL_FIRMS, figsize=(5*N_RL_FIRMS, 4.5), facecolor=BG)
    if N_RL_FIRMS == 1: axes = [axes]
    for idx, ax in enumerate(axes):
        ax.set_facecolor(BG)
        counts = np.zeros(8, dtype=int)
        for a in m.actions_each[idx]: counts[a] += 1
        nonzero = [(c, k) for k, c in enumerate(counts) if c > 0]
        if not nonzero: continue
        vals   = [c for c, _ in nonzero]
        labels = [ACT_NAMES[k] for c, k in nonzero]
        colors = [ACT_COLORS[k] for _, k in nonzero]
        wedges, _ = ax.pie(vals, colors=colors, startangle=90,
                           wedgeprops=dict(edgecolor=BG, linewidth=1.5))
        ax.legend(wedges, labels, loc="lower center", ncol=2,
                  facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT,
                  fontsize=7, bbox_to_anchor=(0.5, -0.25))
        ax.set_title(f"F{idx}  ({len(m.actions_each[idx])} steps)",
                     color=RL_FIRM_COLORS[idx], fontsize=9, fontweight="bold")
    fig.suptitle("Action Distribution [Cooperative]",
                 color=TEXT, fontsize=11, fontweight="bold")
    plt.tight_layout(pad=0.3); solara.FigureMatplotlib(fig); plt.close(fig)


@solara.component
def RLWageSpreadChart(model):
    """Individual profits + wage spread.  Cooperative firms should converge wages."""
    update_counter.get()
    m  = model._inner
    df = m.datacollector.get_model_vars_dataframe()
    if len(df) < 2: return
    steps = np.arange(1, len(df)+1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 4.5), facecolor=BG,
                                    gridspec_kw={"height_ratios":[2,1]})
    _ax(ax1); _ax(ax2)
    for idx in range(N_RL_FIRMS):
        col = f"RL F{idx} Profit"
        if col in df.columns:
            ax1.plot(steps, df[col], color=RL_FIRM_COLORS[idx], lw=1.8, label=f"F{idx}")
    ax1.plot(steps, df["Heuristic Profit"], color=H_COL, lw=1.5, linestyle="--", label="Heuristic avg")
    ax1.set_ylabel("Profit (THB)", fontsize=8)
    ax1.set_title("Individual RL Firm Profits  [Cooperative]",
                  color=TEXT, fontsize=10, fontweight="bold")
    ax1.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
    ax2.fill_between(steps, 0, df["Wage Spread (RL)"], color=WARN, alpha=0.5)
    ax2.plot(steps, df["Wage Spread (RL)"], color=WARN, lw=1.5)
    ax2.set_ylabel("RL Wage\nSpread (THB)", fontsize=8); ax2.set_xlabel("Month", fontsize=8)
    ax2.set_title("Wage Spread Among RL Firms  (should stay low in cooperative mode)",
                  color=TEXT, fontsize=9)
    plt.tight_layout(pad=0.4); solara.FigureMatplotlib(fig); plt.close(fig)


@solara.component
def WageBandChart(model):
    update_counter.get()
    m  = model._inner
    df = m.datacollector.get_model_vars_dataframe()
    if len(df) < 2: return
    steps = np.arange(1, len(df)+1)
    fig, ax = plt.subplots(figsize=(9, 3.2), facecolor=BG)
    _ax(ax)
    ax.fill_between(steps, df["Wage Min"], df["Wage Max"], color=H_COL, alpha=0.15, label="Market wage band")
    ax.plot(steps, df["Market Wage"],  color=H_COL,  lw=1.8, label="Market mean", linestyle="--")
    ax.plot(steps, df["RL Avg Wage"],  color=RL_COL, lw=2.2, label="RL avg wage")
    ax.set_xlabel("Month", fontsize=8); ax.set_ylabel("Wage (THB)", fontsize=8)
    ax.set_title("RL Avg Wage vs Market Band", color=TEXT, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
    plt.tight_layout(pad=0.4); solara.FigureMatplotlib(fig); plt.close(fig)


@solara.component
def SurvivalChart(model):
    update_counter.get()
    m  = model._inner
    df = m.datacollector.get_model_vars_dataframe()
    if len(df) < 2: return
    steps = np.arange(1, len(df)+1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 4), facecolor=BG,
                                    gridspec_kw={"height_ratios":[2,1]})
    _ax(ax1); _ax(ax2)
    ax1.plot(steps, df["Active Firms"], color=BETTER, lw=2.0)
    ax1.set_ylabel("Active Firms", fontsize=8)
    ax1.set_title("Market Health", color=TEXT, fontsize=10, fontweight="bold")
    ax2.fill_between(steps, 0, df["RL Avg Deficit"], color=WORSE, alpha=0.5)
    ax2.plot(steps, df["RL Avg Deficit"], color=WORSE, lw=1.5)
    ax2.set_ylabel("RL Avg\nDeficit Mo", fontsize=8); ax2.set_xlabel("Month", fontsize=8)
    plt.tight_layout(pad=0.4); solara.FigureMatplotlib(fig); plt.close(fig)


@solara.component
def FirmTable(model):
    update_counter.get()
    m      = model._inner
    active = m.active_firms()
    rl_set = {id(f): i for i, f in enumerate(m.rl_firms)}
    rows   = [{"Firm": f.uid + (f" [RL {rl_set[id(f)]}]" if id(f) in rl_set else ""),
               "Profit": round(f.profit, 0), "Workers": len(f.current_workers),
               "Wage": round(f.monthly_wage, 0), "Capital": round(f.capital, 1),
               "Vacancies": f.vacancies, "Deficit Mo": f.deficit_months}
              for f in active]
    solara.Text(f"All Firms -- Cooperative  month {m._step}  |  Active: {len(active)}")
    solara.DataFrame(pd.DataFrame(rows))


@solara.component
def WorkerTable(model):
    update_counter.get()
    m    = model._inner
    rows = [{"Worker": w.uid, "Employed": w.employed,
              "Employer": w.employer.uid if w.employer else "--",
              "Wage": w.monthly_wage if w.employed else 0,
              "Utility": round(w.utility_if_work(w.monthly_wage) if w.employed
                               else w.utility_if_not_work(), 3),
              "Mkt-quit cnt": w.months_below_mkt}
             for w in m.workers]
    employed = sum(1 for w in m.workers if w.employed)
    solara.Text(f"Workers -- {employed}/{len(m.workers)} employed  |  month {m._step}")
    solara.DataFrame(pd.DataFrame(rows))


# ── make_plot_component charts ────────────────────────────────────────

chart_profit     = make_plot_component({"RL Avg Profit": RL_COL, "Heuristic Profit": H_COL})
chart_rl_firms   = make_plot_component({
    "RL F0 Profit": RL_FIRM_COLORS[0],
    "RL F1 Profit": RL_FIRM_COLORS[1],
    "RL F2 Profit": RL_FIRM_COLORS[2],
})
chart_workers    = make_plot_component({"RL Avg Workers": RL_COL, "Heuristic Workers": H_COL})
chart_employment = make_plot_component("Employment %")


# ─────────────────────────────────────────────────────────────────────
# SolaraViz app
# ─────────────────────────────────────────────────────────────────────

model_params = {
    # ── Market structure ─────────────────────────────────────────────
    "n_workers":                  Slider("Workers  (default 100)",                   100,   30,  300,   10),
    "n_firms":                    Slider("Firms  (default 10)",                       10,    3,   20,    1),
    # ── Production economics ─────────────────────────────────────────
    "output_price":               Slider("Output Price  (default 100)",              100,   30,  400,   10),
    "productivity_scale":         Slider("Productivity Scale  (default 1.0)",        1.0,  0.2,  4.0,  0.1),
    "rental_rate_val":            Slider("Capital Rental Rate  (default 500)",       500,   50, 2000,   50),
    "alpha_param":                Slider("Labour Elasticity alpha x100  (default 65)", 65,  20,   90,    5),
    # ── Labour market ─────────────────────────────────────────────────
    "min_wage_val":               Slider("Min Wage THB  (default 7700)",            7700, 3000,20000,  500),
    "worker_search_prob_val":     Slider("Worker Search Prob %  (default 10)",        10,    1,   60,    1),
    # ── Market-quit mechanism ─────────────────────────────────────────
    "market_quit_patience_val":   Slider("Market-Quit Patience months  (default 4)",   4,    1,   16,    1),
    "market_quit_threshold_val":  Slider("Market-Quit Threshold %  (default 91)",     91,   60,  100,    1),
    # ── Firm rules ────────────────────────────────────────────────────
    "max_vacancies_val":          Slider("RL Vacancy Cap  (default 5)",                5,    1,   20,    1),
    "deficit_exit_months_val":    Slider("Exit After N Deficit Months  (default 24)", 24,    3,   72,    3),
    # ── Starting conditions ───────────────────────────────────────────
    "equal_terms_val":            Slider("Equal Terms  (1=narrow spread, 0=wide)",     1,    0,    1,    1),
    "seed_val":                   Slider("Random Seed  (default 455)",               455,    0,  999,    1),
}

page = SolaraViz(
    CoopDemoProxy(),
    components=[
        InfoBanner,
        Scorecard,
        ProfitRank,
        RLObsPanels,
        ActionGrid,
        ActionPieGrid,
        RLWageSpreadChart,
        chart_profit,
        chart_rl_firms,
        chart_workers,
        WageBandChart,
        chart_employment,
        SurvivalChart,
        FirmTable,
        WorkerTable,
    ],
    model_params=model_params,
    name="RL Labor Market -- Cooperative Demo",
)
