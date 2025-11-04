from mesa.experimental.devs import ABMSimulator
from mesa.visualization import (
    Slider,
    SolaraViz,
    make_plot_component,
    make_space_component,
)
from min_wage_model import LaborMarketModel

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
    model = LaborMarketModel(N_workers=100, N_firms=10, min_wage=350, simulator=simulator)
    simulator.setup(model)  # <-- always setup
    return model

# --- Model parameters ---
model_params = {
    "N_workers": Slider("Number of Workers", 100, 50, 500, 10),
    "N_firms": Slider("Number of Firms", 10, 5, 50, 1),
    "min_wage": Slider("Minimum Wage", 350, 300, 600, 25),
}

# --- Visualization helpers ---
def post_process_space(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

def post_process_employment(ax):
    ax.legend(loc="upper left")

# --- Visualization components ---
space_component = make_space_component(
    agent_portrayal={ "Worker": worker_portrayal, "Firm": firm_portrayal },
    width=600,
    height=600,
    post_process=post_process_space,
)

employment_component = make_plot_component(
    data={"Employment Rate": "employment_rate"},
    width=600,
    height=400,
    post_process=post_process_employment,
)

# --- Simulation setup ---
simuulator = ABMSimulator()
model = LaborMarketModel(simuulator)

page = SolaraViz(
    simulator=simuulator,
    model=create_model,
    model_params=model_params,
    components=[space_component, employment_component],
)

page
