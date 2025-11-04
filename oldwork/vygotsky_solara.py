from mesa.experimental.devs import ABMSimulator
from mesa.visualization import (
    Slider,
    SolaraViz,
    make_plot_component,
    make_space_component,
)
from mesa_vygotsky_model import TutoringModel
from mesa_vygotsky_agents import StudentAgent


# --- Agent portrayal ---
def student_portrayal(agent):
    if agent is None:
        return

    portrayal = {
        "size": 50,
        "marker": "o",
        "zorder": 2,
    }

    if agent.happiness > 5:
        portrayal["color"] = "#2ecc71"  # happy - green
    else:
        portrayal["color"] = "#e74c3c"  # unhappy - red

    # highlight agent if potential > current
    if agent.potential_knowledge > agent.current_knowledge:
        portrayal["edgecolor"] = "#FFD700"
        portrayal["linewidth"] = 2
    else:
        portrayal["edgecolor"] = "#333333"

    return portrayal

def create_model(simulator):
    model = TutoringModel(N=25, simulator=simulator)
    simulator.setup(model)  # <-- always setup
    return model


# --- Model parameters ---
model_params = {
    "N": Slider("Number of Students", 25, 5, 100, 5),
    "max_knowledge_gain": Slider("Tutoring Effectiveness", 0.5, 0.1, 1.0, 0.1),
    "independent_study_rate": Slider("Independent Study Rate", 0.05, 0.01, 0.2, 0.01),
    "internalization_rate": Slider("Internalization Rate", 0.02, 0.0, 0.1, 0.01),
    "zpd_spread_rate": Slider("ZPD Spread Rate", 0.01, 0.0, 0.1, 0.01),
    "social_gain_h": Slider("Happiness Gain (Social)", 1.0, 0.0, 2.0, 0.1),
    "happiness_decay": Slider("Happiness Decay Rate", 0.05, 0.01, 0.2, 0.01),
}


# --- Visualization helpers ---
def post_process_space(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def post_process_lines(ax):
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.9))


# --- Create SolaraViz components ---
space_component = make_space_component(
    student_portrayal,
    draw_grid=False,
    post_process=post_process_space,
)

lineplot_component = make_plot_component(
    {
        "Average Knowledge": "#3498db",
        "Average ZPD Size": "#f39c12",
        "Average Happiness": "#2ecc71",
    },
    post_process=post_process_lines,
)


# --- Simulation setup ---
simulator = ABMSimulator()
model = TutoringModel(simulator=simulator)

page = SolaraViz(
    model,
    components=[space_component, lineplot_component],
    model_params=model_params,
    name="Vygotsky Peer Learning Simulation",
    simulator=simulator,
)

page  # show visualization
