#!/usr/bin/env python
# demo/demo_multi.py
#
# Interactive Mesa demo — Multi-firm scenarios:
#   Cooperative  (3 RL firms share reward, trained to collaborate)
#   Competitive  (3 RL firms race each other, trained to dominate peers)
#
# Run with:  solara run demo/demo_multi.py
#
# Mode slider:  0 = Cooperative   1 = Competitive
# Changing mode reloads the trained policy and resets the simulation.
#
# ⭐ = slider critically affects RL performance (model was trained on fixed values).
#     Parameters far outside the training range produce a yellow warning banner.
#
# Training defaults (safe zone):
#   output_price       = 100   ⭐  (safe: 70–150)
#   productivity_scale = 1.0   ⭐  (safe: 0.7–1.5)
#   labor share α      = 0.65      (safe: 0.50–0.80)
#   min wage           = 7700      (safe: 5000–12000)
#   workers            = 100       (safe: 60–160)

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT / "cooperative"))   # default; overridden per-mode below

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

# ── Policies (loaded once at module level) ────────────────────────────
try:
    from sb3_contrib import MaskablePPO
    _COOP_POLICY = MaskablePPO.load(str(ROOT / "cooperative" / "coop_model"))
    print("[demo_multi] Cooperative policy loaded.")
except Exception as _e:
    _COOP_POLICY = None
    print(f"[demo_multi] Could not load coop model: {_e}")

try:
    from sb3_contrib import MaskablePPO
    _COMP_POLICY = MaskablePPO.load(str(ROOT / "competitive" / "comp_model"))
    print("[demo_multi] Competitive policy loaded.")
except Exception as _e:
    _COMP_POLICY = None
    print(f"[demo_multi] Could not load comp model: {_e}")

# Import base model (both coop and comp share the same LaborMarketModel structure)
import importlib, importlib.util

def _load_model_module(folder):
    spec = importlib.util.spec_from_file_location(
        f"model_rl_{folder}", ROOT / folder / "model_rl.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_COOP_MOD = _load_model_module("cooperative")
_COMP_MOD = _load_model_module("competitive")

N_RL_FIRMS = 3

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

# Distinct colours for each of the 3 RL firms
RL_FIRM_COLORS = ["#4fc3f7", "#ef5350", "#66bb6a"]

ACT_COLORS = {
    0: "#555577", 1: "#1565c0", 2: "#64b5f6",
    3: "#ffa726", 4: "#e53935", 5: "#43a047", 6: "#8e24aa",
}
ACT_NAMES = ["Hold", "Wage+300", "Wage+100", "Wage-100", "Wage-300", "Post Vac", "Fire"]

DEFAULTS = dict(mode=0, output_price=100.0, productivity_scale=1.0,
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

class MultiDemoModel:
    """
    Wraps cooperative OR competitive LaborMarketModel with inline RL policy.
    Mode 0 = cooperative, 1 = competitive.
    SolaraViz re-instantiates this whenever a slider changes.

    Because Mesa's SolaraViz calls __init__ with slider kwargs and expects a
    Mesa Model, we subclass at runtime based on the mode value.
    """
    pass


def _make_multi_model(mode, output_price, productivity_scale, alpha_param, min_wage_val, n_workers):
    """
    Factory: returns a freshly constructed Mesa Model instance for the given mode.
    This pattern lets us switch between cooperative and competitive model classes
    while still satisfying SolaraViz's expectations.
    """
    mod    = _COOP_MOD if mode == 0 else _COMP_MOD
    policy = _COOP_POLICY if mode == 0 else _COMP_POLICY
    Base   = mod.LaborMarketModel

    class _DemoModel(Base):
        def __init__(self):
            super().__init__(N_workers=int(n_workers), N_firms=10, n_rl_firms=N_RL_FIRMS)

            self.params = dict(
                output_price=output_price,
                productivity_scale=productivity_scale,
                alpha_param=alpha_param,
                min_wage_val=int(min_wage_val),
                n_workers=int(n_workers),
            )
            self._mode_label = "Cooperative" if mode == 0 else "Competitive"

            # Apply custom parameters
            self.min_wage = int(min_wage_val)
            for firm in self.firms:
                firm.output_price = output_price
                firm.productivity = firm.productivity * productivity_scale
                firm.alpha        = alpha_param
            for firm in self.firms:
                firm.set_initial_wage(gamma=0.8)

            self._policy   = policy
            self.rl_firms  = self.firms[:N_RL_FIRMS]
            self._step     = 0
            self._mode     = mode

            self._prev_profit  = {f.uid: 0.0                    for f in self.rl_firms}
            self._prev_workers = {f.uid: len(f.current_workers) for f in self.rl_firms}

            # Per-firm action history (for ActionGrid)
            self.actions_each = [[] for _ in range(N_RL_FIRMS)]

            # Mesa DataCollector — drives make_plot_component charts
            self.datacollector = DataCollector(model_reporters={
                "RL Avg Profit":     lambda m: float(np.mean([f.profit         for f in m.rl_firms])),
                "Heuristic Profit":  lambda m: float(np.mean([f.profit         for f in m.firms if f not in m.rl_firms])),
                "RL F0 Profit":      lambda m: m.rl_firms[0].profit,
                "RL F1 Profit":      lambda m: m.rl_firms[1].profit,
                "RL F2 Profit":      lambda m: m.rl_firms[2].profit,
                "RL Avg Workers":    lambda m: float(np.mean([len(f.current_workers) for f in m.rl_firms])),
                "Heuristic Workers": lambda m: float(np.mean([len(f.current_workers) for f in m.firms if f not in m.rl_firms])),
                "RL Avg Wage":       lambda m: float(np.mean([f.monthly_wage   for f in m.rl_firms])),
                "Heuristic Wage":    lambda m: float(np.mean([f.monthly_wage   for f in m.firms if f not in m.rl_firms])),
                "Market Wage":       lambda m: float(np.mean([f.monthly_wage   for f in m.firms])),
                "Employment %":      lambda m: 100.0 * sum(1 for w in m.workers if w.employed) / max(len(m.workers), 1),
            })

        # ── Cooperative observation (mirrors cooperative/firm_env.py) ──────

        def _obs_coop(self, idx):
            firm  = self.rl_firms[idx]
            labor = len(firm.current_workers)

            profit_signal        = float(np.tanh(firm.profit / 20_000))
            profit_change_signal = float(np.tanh(
                (firm.profit - self._prev_profit[firm.uid]) / 5_000
            ))

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
            worker_change = float(np.tanh(
                (labor - self._prev_workers[firm.uid]) / 3.0
            ))
            wage_clock = (self._step % 12) / 11.0

            avg_prod    = float(np.mean([f.productivity for f in self.firms]))
            avg_cap     = float(np.mean([f.capital      for f in self.firms]))
            prod_vs_mkt = float(np.tanh((firm.productivity - avg_prod) / max(avg_prod, 1.0)))
            cap_vs_mkt  = float(np.tanh((firm.capital      - avg_cap)  / max(avg_cap,  1.0)))

            survival_signal = float(np.tanh(firm.deficit_months / 12.0))

            peer_profits       = [f.profit for f in self.rl_firms]
            team_profit_signal = float(np.tanh(float(np.mean(peer_profits)) / 20_000))

            peer_wages    = [f.monthly_wage for f in self.rl_firms if f is not firm]
            peer_avg_wage = float(np.mean(peer_wages)) if peer_wages else firm.monthly_wage
            wage_vs_peers = float(np.tanh(
                (firm.monthly_wage - peer_avg_wage) / max(peer_avg_wage, 1.0)
            ))

            obs = np.array([
                profit_signal, profit_change_signal, vmpl_gap, wage_vs_mkt,
                labor_ratio, vacancy_ratio, worker_change, wage_clock,
                prod_vs_mkt, cap_vs_mkt, survival_signal,
                team_profit_signal, wage_vs_peers,
            ], dtype=np.float32)
            return np.clip(obs, -1.5, 1.5)

        # ── Competitive observation (mirrors competitive/firm_env.py) ──────

        def _obs_comp(self, idx):
            firm  = self.rl_firms[idx]
            labor = len(firm.current_workers)

            profit_signal        = float(np.tanh(firm.profit / 20_000))
            profit_change_signal = float(np.tanh(
                (firm.profit - self._prev_profit[firm.uid]) / 5_000
            ))

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
            worker_change = float(np.tanh(
                (labor - self._prev_workers[firm.uid]) / 3.0
            ))
            wage_clock = (self._step % 12) / 11.0

            avg_prod    = float(np.mean([f.productivity for f in self.firms]))
            avg_cap     = float(np.mean([f.capital      for f in self.firms]))
            prod_vs_mkt = float(np.tanh((firm.productivity - avg_prod) / max(avg_prod, 1.0)))
            cap_vs_mkt  = float(np.tanh((firm.capital      - avg_cap)  / max(avg_cap,  1.0)))

            peers = [f for f in self.rl_firms if f is not firm]
            peer_avg_profit  = float(np.mean([f.profit              for f in peers])) if peers else 0.0
            peer_avg_wage    = float(np.mean([f.monthly_wage         for f in peers])) if peers else firm.monthly_wage
            peer_avg_workers = float(np.mean([len(f.current_workers) for f in peers])) if peers else labor

            profit_vs_peers  = float(np.tanh((firm.profit - peer_avg_profit) / 20_000))
            wage_vs_peers    = float(np.tanh(
                (firm.monthly_wage - peer_avg_wage) / max(peer_avg_wage, 1.0)
            ))
            workers_vs_peers = float(np.tanh((labor - peer_avg_workers) / 5.0))

            obs = np.array([
                profit_signal, profit_change_signal, vmpl_gap, wage_vs_mkt,
                labor_ratio, vacancy_ratio, worker_change, wage_clock,
                prod_vs_mkt, cap_vs_mkt,
                profit_vs_peers, wage_vs_peers, workers_vs_peers,
            ], dtype=np.float32)
            return np.clip(obs, -1.5, 1.5)

        def _action_mask(self):
            wage_ok = self._step % 12 == 0
            return np.array([True, wage_ok, wage_ok, wage_ok, wage_ok, True, True], dtype=bool)

        # ── Mesa step ──────────────────────────────────────────────────────

        def step(self):
            # Snapshot state before the round
            for f in self.rl_firms:
                self._prev_profit[f.uid]  = f.profit
                self._prev_workers[f.uid] = len(f.current_workers)

            mask = self._action_mask()

            # Query policy for each RL firm independently
            for idx, firm in enumerate(self.rl_firms):
                if self._policy is not None:
                    if self._mode == 0:
                        obs = self._obs_coop(idx)
                    else:
                        obs = self._obs_comp(idx)
                    act, _ = self._policy.predict(
                        obs[np.newaxis], deterministic=True,
                        action_masks=mask[np.newaxis],
                    )
                    firm.rl_action = int(act[0])
                else:
                    firm.rl_action = 0

            super().step()
            self._step += 1

            for idx, firm in enumerate(self.rl_firms):
                self.actions_each[idx].append(firm.rl_action)

            self.datacollector.collect(self)

    return _DemoModel()


# ─────────────────────────────────────────────────────────────────────
# Top-level Mesa Model proxy
# (SolaraViz needs a class, not a factory; we use a thin wrapper)
# ─────────────────────────────────────────────────────────────────────

class MultiDemoModelProxy:
    """
    Thin class that SolaraViz can instantiate.
    Delegates everything to the real model built by the factory.
    """

    def __init__(self,
                 mode               = DEFAULTS["mode"],
                 output_price       = DEFAULTS["output_price"],
                 productivity_scale = DEFAULTS["productivity_scale"],
                 alpha_param        = DEFAULTS["alpha_param"],
                 min_wage_val       = DEFAULTS["min_wage_val"],
                 n_workers          = DEFAULTS["n_workers"]):

        self._inner = _make_multi_model(
            mode=int(mode),
            output_price=output_price,
            productivity_scale=productivity_scale,
            alpha_param=alpha_param,
            min_wage_val=min_wage_val,
            n_workers=n_workers,
        )
        # Expose attributes SolaraViz expects
        self.schedule = self._inner.schedule

    def step(self):
        self._inner.step()

    # Proxy every attribute access to the inner model
    def __getattr__(self, name):
        return getattr(self._inner, name)


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
    update_counter.get()
    inner = model._inner
    solara.Info(
        f"Mode: {inner._mode_label}  "
        f"(0 = Cooperative — shared reward,  1 = Competitive — beat-your-peers reward)  |  "
        f"Training defaults — ⭐ Output Price: 100  |  ⭐ Productivity Scale: 1.0  |  "
        f"α: 0.65  |  Min Wage: 7700  |  Workers: 100"
    )
    for msg in _ood_msgs(inner.params):
        solara.Warning(msg)



@solara.component
def ActionGrid(model):
    """One action strip per RL firm (last 60 steps each)."""
    update_counter.get()
    inner = model._inner
    if not any(inner.actions_each[0]):
        return

    fig, axes = plt.subplots(N_RL_FIRMS, 1, figsize=(10, 1.0 * N_RL_FIRMS + 0.8), facecolor=BG)
    if N_RL_FIRMS == 1:
        axes = [axes]
    fig.suptitle(f"RL Firm Actions — Last 60 months  [{inner._mode_label}]",
                 color="white", fontsize=9, fontweight="bold", y=1.01)

    for idx, ax in enumerate(axes):
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)
        last = inner.actions_each[idx][-60:]
        for i, act in enumerate(last):
            ax.bar(i, 1, color=ACT_COLORS[act], width=1.0, align="edge")
        ax.set_xlim(0, len(last))
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel(f"F{idx}", color=RL_FIRM_COLORS[idx], fontsize=8, rotation=0, labelpad=20)

    axes[-1].set_xlabel(f"Month offset (current: {inner._step})", color=TEXT, fontsize=8)

    patches = [mpatches.Patch(color=ACT_COLORS[k], label=ACT_NAMES[k]) for k in range(7)]
    axes[-1].legend(handles=patches, ncol=7, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT,
                    fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.7))
    plt.tight_layout(pad=0.3)
    solara.FigureMatplotlib(fig)
    plt.close(fig)


@solara.component
def FirmTable(model):
    """Live per-firm snapshot (sortable)."""
    update_counter.get()
    inner  = model._inner
    firms  = inner.firms
    rl_set = {id(f): i for i, f in enumerate(inner.rl_firms)}
    df = pd.DataFrame([
        {
            "Firm":         f"F{i}" + (f" [RL {rl_set[id(f)]}]" if id(f) in rl_set else ""),
            "Type":         f"RL F{rl_set[id(f)]}" if id(f) in rl_set else "Heuristic",
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
    solara.Text(f"All Firms — {inner._mode_label}  month {inner._step}")
    solara.DataFrame(df)


@solara.component
def WorkerTable(model):
    """Live worker snapshot (employed/unemployed, wage, utility)."""
    update_counter.get()
    inner = model._inner
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
        for w in inner.workers
    ])
    solara.Text(f"All Workers — {inner._mode_label}  month {inner._step}")
    solara.DataFrame(df)


@solara.component
def Scorecard(model):
    update_counter.get()
    inner = model._inner
    df = inner.datacollector.get_model_vars_dataframe()
    if df.empty:
        return

    metrics = {
        "Profit":  ("RL Avg Profit",  "Heuristic Profit"),
        "Workers": ("RL Avg Workers", "Heuristic Workers"),
        "Wage":    ("RL Avg Wage",    "Heuristic Wage"),
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
    ax.set_title(f"Cumulative RL Scorecard  [{inner._mode_label}]  (month {inner._step})",
                 color="white", fontsize=10, fontweight="bold")
    ax.set_xlabel("RL avg vs Heuristic avg (%)", color=TEXT, fontsize=9)
    plt.tight_layout(pad=0.4)
    solara.FigureMatplotlib(fig)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# SolaraViz app
# ─────────────────────────────────────────────────────────────────────

chart_profit     = make_plot_component({"RL Avg Profit": RL_COL, "Heuristic Profit": H_COL})
chart_rl_firms   = make_plot_component({
    "RL F0 Profit": RL_FIRM_COLORS[0],
    "RL F1 Profit": RL_FIRM_COLORS[1],
    "RL F2 Profit": RL_FIRM_COLORS[2],
})
chart_workers    = make_plot_component({"RL Avg Workers": RL_COL, "Heuristic Workers": H_COL})
chart_wage       = make_plot_component({"RL Avg Wage": RL_COL, "Heuristic Wage": H_COL, "Market Wage": MAC_COL})
chart_employment = make_plot_component("Employment %")

model_params = {
    "mode":                Slider("Mode  (0=Cooperative  1=Competitive)",   0,     0,     1,      1),
    "output_price":        Slider("⭐ Output Price  (default 100)",          100,   50,    300,    10),
    "productivity_scale":  Slider("⭐ Productivity Scale  (default 1.0)",    1.0,   0.3,   3.0,    0.1),
    "alpha_param":         Slider("Labor Share α  (default 0.65)",           0.65,  0.30,  0.90,   0.05),
    "min_wage_val":        Slider("Min Wage THB  (default 7700)",            7700,  3000,  20000,  500),
    "n_workers":           Slider("Workers  (default 100)",                  100,   30,    200,    10),
}

page = SolaraViz(
    MultiDemoModelProxy(),
    components=[InfoBanner, FirmTable, WorkerTable,
                chart_profit, chart_rl_firms, chart_workers, chart_wage, chart_employment,
                ActionGrid, Scorecard],
    model_params=model_params,
    name="RL Labor Market — Multi-Firm Demo",
)
