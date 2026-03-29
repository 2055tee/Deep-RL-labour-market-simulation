# reformed/rl_vis.py
#
# Two sections:
#   1. ReformedMetricsCallback  — TensorBoard callback used during training
#   2. Solara chart components  — histograms/scatters adapted from viz_wage.py
#                                 RL firm (uid == "F0") is highlighted in red

import shutil
import numpy as np
from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback

import matplotlib.pyplot as plt
import pandas as pd
import solara
from mesa.visualization.utils import update_counter

# ─────────────────────────────────────────────────────────────────────
# 1.  TensorBoard callback
# ─────────────────────────────────────────────────────────────────────

ACTION_NAMES = {
    0: "hold",
    1: "wage_up_300",
    2: "wage_up_100",
    3: "wage_dn_100",
    4: "wage_dn_300",
    5: "post_vacancy",
    6: "fire_worker",
}

RL_FIRM_ID = "F0"


class ReformedMetricsCallback(BaseCallback):

    def __init__(self, log_dir="./tensorboard_logs", algo_name="Reformed_MaskablePPO",
                 keep_runs=3, verbose=0):
        super().__init__(verbose)
        self.log_dir       = Path(log_dir)
        self.algo_name     = algo_name
        self.keep_runs     = keep_runs
        self.action_counts = np.zeros(len(ACTION_NAMES), dtype=np.int64)

    def _on_training_start(self):
        try:
            runs = []
            for d in self.log_dir.iterdir():
                if d.is_dir() and d.name.startswith(self.algo_name + "_"):
                    try:
                        num = int(d.name.rsplit("_", 1)[-1])
                        runs.append((num, d))
                    except ValueError:
                        pass
            runs.sort(key=lambda x: x[0])
            for _, run_dir in runs[: -self.keep_runs]:
                shutil.rmtree(run_dir)
                print(f"[cleanup] removed old run: {run_dir.name}")
        except Exception:
            pass

    def _on_step(self) -> bool:
        env  = self.training_env.envs[0].env   # unwrap ActionMasker -> ReformedFirmEnv
        sim  = env.model
        firm = env.rl_firm

        actions = self.locals.get("actions")
        if actions is not None:
            self.action_counts[int(actions[0])] += 1

        labor = len(firm.current_workers)

        self.logger.record("firm/profit",         firm.profit)
        self.logger.record("firm/monthly_wage",   firm.monthly_wage)
        self.logger.record("firm/n_workers",      labor)
        self.logger.record("firm/vacancies",      firm.vacancies)
        self.logger.record("firm/reward",         firm.reward)
        self.logger.record("firm/capital",        firm.capital)
        self.logger.record("firm/deficit_months", firm.deficit_months)

        if labor > 0:
            mpl  = firm.marginal_product_labor(firm.productivity, labor, firm.alpha)
            vmpl = mpl * firm.output_price
            self.logger.record("firm/vmpl",     vmpl)
            self.logger.record("firm/vmpl_gap", vmpl - firm.monthly_wage)

        employed    = [w for w in sim.workers if w.employed]
        market_wage = np.mean([f.monthly_wage for f in sim.firms] or [0])
        avg_profit  = np.mean([f.profit       for f in sim.firms] or [0])
        employ_rate = len(employed) / len(sim.workers) if sim.workers else 0

        self.logger.record("economy/market_avg_wage",  market_wage)
        self.logger.record("economy/avg_profit_all",   avg_profit)
        self.logger.record("economy/employment_rate",  employ_rate)
        self.logger.record("economy/n_active_firms",   len(sim.active_firms()))

        if market_wage > 0:
            self.logger.record("economy/firm_wage_premium",
                               (firm.monthly_wage - market_wage) / market_wage)

        active      = sim.active_firms()
        avg_workers = float(np.mean([len(f.current_workers) for f in active])) if active else labor
        self.logger.record("firm/worker_mkt_share", (labor - avg_workers) / max(avg_workers, 1))

        total = self.action_counts.sum()
        if total > 0:
            for idx, name in ACTION_NAMES.items():
                self.logger.record(f"actions/{name}", self.action_counts[idx] / total)

        return True


# ─────────────────────────────────────────────────────────────────────
# 2.  Solara chart components  (adapted from viz_wage.py)
#     RL firm (uid == RL_FIRM_ID) is shown in red in all charts.
# ─────────────────────────────────────────────────────────────────────

_COL_RL  = "#e74c3c"   # red  — RL firm
_COL_H   = "#3498db"   # blue — heuristic firms


@solara.component
def FirmHistogram(model):
    update_counter.get()
    active = model.active_firms()
    sizes  = [len(f.current_workers) for f in active]
    rl_sz  = [len(f.current_workers) for f in active if f.uid == RL_FIRM_ID]

    fig, ax = plt.subplots(figsize=(8, 4))
    if sizes:
        ax.hist(sizes, bins=20, color=_COL_H, edgecolor="black", label="Heuristic")
    if rl_sz:
        ax.axvline(rl_sz[0], color=_COL_RL, lw=2, linestyle="--",
                   label=f"RL firm ({rl_sz[0]} workers)")
    ax.set_title("Distribution of Firm Sizes")
    ax.set_xlabel("Number of Workers")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = solara.FigureMatplotlib(fig)
    plt.close(fig)
    return out


@solara.component
def FirmWageHistogram(model):
    update_counter.get()
    active = model.active_firms()
    wages  = [f.monthly_wage for f in active if f.monthly_wage is not None]
    rl_w   = [f.monthly_wage for f in active if f.uid == RL_FIRM_ID and f.monthly_wage is not None]

    fig, ax = plt.subplots(figsize=(8, 4))
    if wages:
        ax.hist(wages, bins=20, color=_COL_H, edgecolor="black", label="Heuristic")
        if len(set(wages)) == 1:
            ax.set_xlim(wages[0] - 1, wages[0] + 1)
            ax.set_xticks(wages)
    if rl_w:
        ax.axvline(rl_w[0], color=_COL_RL, lw=2, linestyle="--",
                   label=f"RL firm ({rl_w[0]:,})")
    ax.set_title("Distribution of Firm Wages")
    ax.set_xlabel("Monthly Wage (THB)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = solara.FigureMatplotlib(fig)
    plt.close(fig)
    return out


@solara.component
def FirmProfitHistogram(model):
    update_counter.get()
    active  = model.active_firms()
    profits = [f.profit for f in active]
    rl_p    = [f.profit for f in active if f.uid == RL_FIRM_ID]

    fig, ax = plt.subplots(figsize=(8, 4))
    if profits:
        ax.hist(profits, bins=20, color=_COL_H, edgecolor="black", label="Heuristic")
    if rl_p:
        ax.axvline(rl_p[0], color=_COL_RL, lw=2, linestyle="--",
                   label=f"RL firm ({rl_p[0]:,.0f})")
    ax.set_title("Distribution of Firm Profits")
    ax.set_xlabel("Profit (THB)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = solara.FigureMatplotlib(fig)
    plt.close(fig)
    return out


@solara.component
def FirmCapitalHistogram(model):
    update_counter.get()
    active   = model.active_firms()
    capitals = [f.capital for f in active]
    rl_k     = [f.capital for f in active if f.uid == RL_FIRM_ID]

    fig, ax = plt.subplots(figsize=(8, 4))
    if capitals:
        ax.hist(capitals, bins=20, color=_COL_H, edgecolor="black", label="Heuristic")
    if rl_k:
        ax.axvline(rl_k[0], color=_COL_RL, lw=2, linestyle="--",
                   label=f"RL firm ({rl_k[0]:.1f})")
    ax.set_title("Distribution of Firm Capital")
    ax.set_xlabel("Capital")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = solara.FigureMatplotlib(fig)
    plt.close(fig)
    return out


@solara.component
def WageVsMPLScatter(model):
    update_counter.get()
    active = model.active_firms()
    wages, vmpls, colors = [], [], []
    for f in active:
        if f.monthly_wage is None:
            continue
        labor = len(f.current_workers)
        mpl   = f.marginal_product_labor(f.productivity, labor, f.alpha)
        vmpl  = mpl * f.output_price
        wages.append(f.monthly_wage)
        vmpls.append(vmpl)
        colors.append(_COL_RL if f.uid == RL_FIRM_ID else _COL_H)

    fig, ax = plt.subplots(figsize=(8, 4))
    if wages:
        ax.scatter(vmpls, wages, c=colors, edgecolors="black", alpha=0.8, s=60,
                   zorder=3)
        lo = min(min(vmpls), min(wages))
        hi = max(max(vmpls), max(wages))
        ax.plot([lo, hi], [lo, hi], color="#7f8c8d", linestyle="--", lw=1,
                label="wage = VMPL")
    ax.set_title("Wage vs Value of MPL")
    ax.set_xlabel("Value of MPL (price x MPL)")
    ax.set_ylabel("Monthly Wage (THB)")
    ax.legend(fontsize=8)
    # legend patch for RL
    from matplotlib.patches import Patch
    handles = [Patch(color=_COL_RL, label="RL firm"),
               Patch(color=_COL_H,  label="Heuristic")]
    ax.legend(handles=handles, fontsize=8)
    plt.tight_layout()
    out = solara.FigureMatplotlib(fig)
    plt.close(fig)
    return out


@solara.component
def CapitalVsProfitScatter(model):
    update_counter.get()
    active = model.active_firms()
    caps, profits, colors = [], [], []
    for f in active:
        caps.append(f.capital)
        profits.append(f.profit)
        colors.append(_COL_RL if f.uid == RL_FIRM_ID else _COL_H)

    fig, ax = plt.subplots(figsize=(8, 4))
    if caps:
        ax.scatter(caps, profits, c=colors, edgecolors="black", alpha=0.8, s=60)
    ax.axhline(0, color="#7f8c8d", linestyle="--", lw=1)
    ax.set_title("Capital vs Profit")
    ax.set_xlabel("Capital")
    ax.set_ylabel("Profit (THB)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=_COL_RL, label="RL firm"),
                       Patch(color=_COL_H,  label="Heuristic")], fontsize=8)
    plt.tight_layout()
    out = solara.FigureMatplotlib(fig)
    plt.close(fig)
    return out


@solara.component
def WorkerUtilityHistogram(model):
    update_counter.get()
    utilities = []
    for w in model.workers:
        if w.employed:
            utilities.append(w.utility_if_work(w.monthly_wage))
        else:
            utilities.append(w.utility_if_not_work())

    fig, ax = plt.subplots(figsize=(9, 4))
    if utilities:
        ax.hist(utilities, bins=20, color="#9b59b6", edgecolor="black")
    ax.set_title("Distribution of Worker Utility")
    ax.set_xlabel("Utility")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    out = solara.FigureMatplotlib(fig)
    plt.close(fig)
    return out


@solara.component
def WorkerWageHistogram(model):
    update_counter.get()
    wages = [w.monthly_wage for w in model.workers if w.employed and w.monthly_wage > 0]

    fig, ax = plt.subplots(figsize=(9, 4))
    if wages:
        ax.hist(wages, bins=20, color="#1abc9c", edgecolor="black")
        if len(set(wages)) == 1:
            ax.set_xlim(wages[0] - 1, wages[0] + 1)
            ax.set_xticks(wages)
    ax.set_title("Distribution of Worker Wages")
    ax.set_xlabel("Monthly Wage (THB)")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    out = solara.FigureMatplotlib(fig)
    plt.close(fig)
    return out


@solara.component
def FirmTable(model):
    update_counter.get()
    active = model.active_firms()
    df = pd.DataFrame([
        {
            "id":      f.uid + (" [RL]" if f.uid == RL_FIRM_ID else ""),
            "wage":    f.monthly_wage,
            "profit":  round(f.profit, 0),
            "capital": round(f.capital, 1),
            "workers": len(f.current_workers),
            "deficit": f.deficit_months,
        }
        for f in active
    ])
    solara.Text(f"Active firms: {len(active)}")
    return solara.DataFrame(df)


@solara.component
def WorkerTable(model):
    update_counter.get()
    df = pd.DataFrame([
        {
            "id":       w.uid,
            "employed": w.employed,
            "wage":     w.monthly_wage if w.employed else 0,
            "utility":  round(
                w.utility_if_work(w.monthly_wage) if w.employed
                else w.utility_if_not_work(), 4
            ),
            "mkt_wait": w.months_below_mkt,
        }
        for w in model.workers
    ])
    employed = df["employed"].sum()
    solara.Text(f"Workers: {employed} employed / {len(df)} total")
    return solara.DataFrame(df)
