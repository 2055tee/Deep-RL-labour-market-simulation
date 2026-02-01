from mesa.experimental.devs import ABMSimulator
from mesa.visualization import (
    Slider,
    SolaraViz,
    make_plot_component,
)
from min_wage_model import LaborMarketModel , Worker, Firm

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
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.9))
    
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

lineplot_component_profit = make_plot_component(
    measure="AverageProfit",  # Specify the measure for the AverageProfit plot
    post_process=post_process_lines,
)

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




# --- Simulation setup ---
simulator = ABMSimulator()
model = LaborMarketModel(simulator=simulator)


# Do not instantiate the model here; Solara/ABMSimulator will call `create_model(simulator)`
page = SolaraViz(
    simulator=simulator,
    model=model,
    model_params=model_params,
    components=[lineplot_component, lineplot_component_wage, lineplot_component_profit, lineplot_component_firm_size, lineplot_component_firm_capital,
                lineplot_component_min_wage],
)

page
