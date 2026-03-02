from mesa.experimental.devs import ABMSimulator
from mesa.visualization import (
    Slider,
    SolaraViz,
    make_plot_component,
)
from mesa.visualization.utils import update_counter
from min_wage_model import LaborMarketModel , Worker, Firm
import solara
import matplotlib.pyplot as plt
import pandas as pd


@solara.component
def FirmHistogram(model: LaborMarketModel):
    update_counter.get()  # hook into the global update counter so the figure re-renders every step
    fig, ax = plt.subplots(figsize=(8, 4))
    firm_sizes = [len(firm.current_workers) for firm in model.firms]
    ax.hist(firm_sizes, bins=20, color="#3498db", edgecolor="black")
    ax.set_title(f"Distribution of Firm Sizes")
    ax.set_xlabel("Number of Workers")
    ax.set_ylabel("Frequency")
    fig.subplots_adjust(bottom=0.2)
    plt.tight_layout()
    out = solara.FigureMatplotlib(fig)
    plt.close(fig)
    return out
   
   
   


@solara.component
def FirmWageHistogram(model: LaborMarketModel):
    update_counter.get()
    wages = [firm.monthly_wage for firm in model.firms if firm.monthly_wage is not None]
    fig, ax = plt.subplots(figsize=(8, 4))
    if wages:
        ax.hist(wages, bins=20, color="#8e44ad", edgecolor="black")
    # If all wages are the same, make sure the x-axis shows that single wage value
    if wages and len(set(wages)) == 1:
        ax.set_xlim(wages[0] - 1, wages[0] + 1)  # Show a small range around the single wage value
        ax.set_xticks(wages)  # Set x-ticks to the single wage value
    ax.set_title("Distribution of Firm Wages")
    ax.set_xlabel("Monthly Wage")
    ax.set_ylabel("Frequency")
    fig.subplots_adjust(bottom=0.2)
    plt.tight_layout()
    out = solara.FigureMatplotlib(fig)
    plt.close(fig)
    return out
   
   
   


@solara.component
def FirmProfitHistogram(model: LaborMarketModel):
    update_counter.get()
    profits = [firm.profit for firm in model.firms]
    fig, ax = plt.subplots(figsize=(8, 4))
    if profits:
        ax.hist(profits, bins=20, color="#f39c12", edgecolor="black")
    ax.set_title("Distribution of Firm Profits")
    ax.set_xlabel("Profit")
    ax.set_ylabel("Frequency")
    fig.subplots_adjust(bottom=0.2)
    plt.tight_layout()
    out = solara.FigureMatplotlib(fig)
    plt.close(fig)
    return out
   
   
   


@solara.component
def FirmCapitalHistogram(model: LaborMarketModel):
    update_counter.get()
    capitals = [firm.capital for firm in model.firms]
    fig, ax = plt.subplots(figsize=(8, 4))
    if capitals:
        ax.hist(capitals, bins=20, color="#27ae60", edgecolor="black")
    ax.set_title("Distribution of Firm Capital")
    ax.set_xlabel("Capital")
    ax.set_ylabel("Frequency")
    fig.subplots_adjust(bottom=0.2)
    plt.tight_layout()
    out = solara.FigureMatplotlib(fig)
    plt.close(fig)
    return out
   
   
   


@solara.component
def WageVsMPLScatter(model: LaborMarketModel):
    update_counter.get()
    wages = []
    vmpls = []  # value of marginal product of labor (price * MPL)
    for firm in model.firms:
        if firm.monthly_wage is None:
            continue
        labor = len(firm.current_workers)
        mpl = firm.marginal_product_labor(firm.productivity, labor, firm.alpha)
        vmpl = mpl * firm.output_price
        wages.append(firm.monthly_wage)
        vmpls.append(vmpl)

    fig, ax = plt.subplots(figsize=(8, 4))
    if wages and vmpls:
        ax.scatter(vmpls, wages, c="#2980b9", edgecolors="black", alpha=0.7)
        diag_min = min(min(vmpls), min(wages))
        diag_max = max(max(vmpls), max(wages))
        ax.plot([diag_min, diag_max], [diag_min, diag_max], color="#7f8c8d", linestyle="--", linewidth=1)
    ax.set_title("Wage vs Value of MPL")
    ax.set_xlabel("Value of MPL (price × MPL)")
    ax.set_ylabel("Monthly Wage")
    fig.subplots_adjust(bottom=0.2)
    plt.tight_layout()
    out = solara.FigureMatplotlib(fig)
    plt.close(fig)
    return out
   
   
   


@solara.component
def CapitalVsProfitScatter(model: LaborMarketModel):
    update_counter.get()
    capitals = [firm.capital for firm in model.firms]
    profits = [firm.profit for firm in model.firms]

    fig, ax = plt.subplots(figsize=(8, 4))
    if capitals and profits:
        ax.scatter(capitals, profits, c="#c0392b", edgecolors="black", alpha=0.7)
    ax.set_title("Capital vs Profit")
    ax.set_xlabel("Capital")
    ax.set_ylabel("Profit")
    fig.subplots_adjust(bottom=0.2)
    plt.tight_layout()
    out = solara.FigureMatplotlib(fig)
    plt.close(fig)
    return out
   
   
@solara.component
def WorkerUtilityHistogram(model: LaborMarketModel):
    update_counter.get()
    utilities = []
    for worker in model.workers:
        if worker.employed:
            utilities.append(worker.utility_if_work(worker.monthly_wage))
        else:
            utilities.append(worker.utility_if_not_work())

    fig, ax = plt.subplots(figsize=(9, 4))
    if utilities:
        ax.hist(utilities, bins=20, color="#9b59b6", edgecolor="black")
    ax.set_title("Distribution of Worker Utility")
    ax.set_xlabel("Utility")
    ax.set_ylabel("Frequency")
    fig.subplots_adjust(bottom=0.2)
    plt.tight_layout()
    out = solara.FigureMatplotlib(fig)
    plt.close(fig)
    return out


@solara.component
def WorkerWageHistogram(model: LaborMarketModel):
    update_counter.get()
    wages = [w.monthly_wage for w in model.workers if w.employed and w.monthly_wage > 0]

    fig, ax = plt.subplots(figsize=(9, 4))
    if wages:
        ax.hist(wages, bins=20, color="#1abc9c", edgecolor="black")
    # If all wages are the same, make sure the x-axis shows that single wage value
    if wages and len(set(wages)) == 1:
        ax.set_xlim(wages[0] - 1, wages[0] + 1)  # Show a small range around the single wage value
        ax.set_xticks(wages)  # Set x-ticks to the single wage value
    ax.set_title("Distribution of Worker Wages")
    ax.set_xlabel("Monthly Wage")
    ax.set_ylabel("Frequency")
    fig.subplots_adjust(bottom=0.2)
    plt.tight_layout()
    out = solara.FigureMatplotlib(fig)
    plt.close(fig)
    return out

@solara.component
def FirmTable(model: LaborMarketModel):
    update_counter.get()
    df = pd.DataFrame([
        {
            "id": f.unique_id,
            "wage": f.monthly_wage,
            "profit": f.profit,
            "capital": f.capital,
            "labor": len(f.current_workers),
        }
        for f in model.firms
    ])
    return solara.DataFrame(df)
   
@solara.component
def WorkerTable(model: LaborMarketModel):
    update_counter.get()
    df = pd.DataFrame([
        {
            "id": w.unique_id,
            "employed": w.employed,
            "wage": w.monthly_wage if w.employed else 0,
            "utility": w.utility_if_work(w.monthly_wage) if w.employed else w.utility_if_not_work(),
        }
        for w in model.workers
    ])
    return solara.DataFrame(df)


# --- Agent portrayal ---
def worker_portrayal(agent):
    if agent is None:
        return

    portrayal = {
        "size": 5,
        "marker": "o",
        "zorder": 2,
    }

    if agent.employed:
        portrayal["color"] = "#2ecc71"  # employed - green
    else:
        portrayal["color"] = "#e74c3c"  # unemployed - red

    return portrayal

# -- Firm portrayal ---
def firm_portrayal(agent):
    if agent is None:
        return

    portrayal = {
        "size": 15,
        "marker": "s",
        "color": "#3498db",  # blue
        "zorder": 1,
    }
    
    if agent.profit >= 0:
        portrayal["edgecolor"] = "#2ecc71"  # profitable - green border
    else:
        portrayal["edgecolor"] = "#e74c3c"  # unprofitable - red border

    return portrayal

def create_model(simulator):
    model = LaborMarketModel(simulator=simulator)
    simulator.setup(model)  # <-- always setup
    return model


# --- Model parameters ---
model_params = {
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    "N_workers": Slider("Number of Workers", 100, 500, 5000, 50),
    "N_firms": Slider("Number of Firms", 10, 30, 200, 5),
    "min_wage": Slider("Minimum Wage", 350, 300, 600, 25),
}

# --- Visualization helpers ---
def post_process_lines(ax):
    # ax.legend(loc="center left", bbox_to_anchor=(1, 0.9))
    # Give the auto-generated line plots more breathing room
    fig = ax.figure
    fig.set_size_inches(7, 5)
    fig.subplots_adjust(bottom=0.2)
    
# --- Visualization components ---
lineplot_component = make_plot_component(
    measure="EmploymentRate",  # Directly specify the measure to plot
    post_process=post_process_lines,
)

# If you need to plot more measures like "AverageWage" or "AverageProfit", you can do this:
lineplot_component_wage = make_plot_component(
    measure="AverageWage",  # Specify the measure for the AverageWage plot
    post_process=post_process_lines,
)

lineplot_component_worker_utility = make_plot_component(
    measure="AverageWorkerUtility",
    post_process=post_process_lines,
)

# lineplot_component_profit = make_plot_component(
#     measure="AverageProfit",  # Specify the measure for the AverageProfit plot
#     post_process=post_process_lines,
# )

lineplot_component_firm_size = make_plot_component(
    measure="AvgFirmSize",  # Specify the measure for the Firm Size plot
    post_process=post_process_lines,
)    

lineplot_component_firm_capital = make_plot_component(
    measure="AvgFirmCapital",  # Specify the measure for the Firm Capital plot
    post_process=post_process_lines,
)

lineplot_component_min_wage = make_plot_component(
    measure="MinWage",  # Specify the measure for the Minimum Wage plot
    post_process=post_process_lines,
)

lineplot_component_avg_firm_wage = make_plot_component(
    measure="AverageFirmWage",
    post_process=post_process_lines,
)

lineplot_component_avg_profit = make_plot_component(
    measure="AverageProfit",
    post_process=post_process_lines,
)

lineplot_component_total_output = make_plot_component(
    measure="TotalOutput",
    post_process=post_process_lines,
)

lineplot_component_capital_stock = make_plot_component(
    measure="CapitalStock",
    post_process=post_process_lines,
)

# lineplot_component_machine_investment = make_plot_component(
#     measure="AverageMachineInvestment",  # Specify the measure for the Average Machine Investment plot
#     post_process=post_process_lines,
# )


# --- Simulation setup ---
simulator = ABMSimulator()
model = LaborMarketModel(simulator=simulator)


# Do not instantiate the model here; Solara/ABMSimulator will call `create_model(simulator)`
page = SolaraViz(
    simulator=simulator,
    model=model,
    model_params=model_params,
    components=[FirmTable, WorkerTable, lineplot_component, lineplot_component_wage, lineplot_component_worker_utility, lineplot_component_firm_size, lineplot_component_firm_capital,
                lineplot_component_min_wage, lineplot_component_avg_firm_wage, lineplot_component_avg_profit, lineplot_component_total_output, lineplot_component_capital_stock,
                FirmHistogram, FirmWageHistogram, FirmProfitHistogram, FirmCapitalHistogram,
                WageVsMPLScatter, CapitalVsProfitScatter, WorkerUtilityHistogram, WorkerWageHistogram],
)

page
