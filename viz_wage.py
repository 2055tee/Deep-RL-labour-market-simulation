from mesa.visualization.modules import ChartModule, TextElement
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter

from min_wage_model import LaborMarketModel

# -----------------------------
# Custom Text display
# -----------------------------
class SummaryElement(TextElement):
    def render(self, model):
        return (
            f"Step: {model.schedule.time} <br>"
            f"Employment Rate: {model.compute_employment_rate():.2f}<br>"
            f"Average Wage: {model.compute_avg_wage():.2f}<br>"
            f"Average Profit: {model.compute_avg_profit():.2f}"
        )

# -----------------------------
# Charts
# -----------------------------
employment_chart = ChartModule(
    [{"Label": "EmploymentRate", "Color": "#1f77b4"}],
    data_collector_name="datacollector",
    canvas_height=200,
    canvas_width=500,
)

wage_chart = ChartModule(
    [{"Label": "AverageWage", "Color": "#2ca02c"}],
    data_collector_name="datacollector",
    canvas_height=200,
    canvas_width=500,
)

profit_chart = ChartModule(
    [{"Label": "AverageProfit", "Color": "#ff7f0e"}],
    data_collector_name="datacollector",
    canvas_height=200,
    canvas_width=500,
)

# -----------------------------
# User Controls
# -----------------------------
model_params = {
    "N_workers": UserSettableParameter("slider", "Number of Workers", 100, 10, 200, 10),
    "N_firms": UserSettableParameter("slider", "Number of Firms", 10, 1, 50, 1),
    "min_wage": UserSettableParameter("slider", "Minimum Wage", 350, 200, 600, 50),
}

# -----------------------------
# Launch Server
# -----------------------------
server = ModularServer(
    LaborMarketModel,
    [SummaryElement(), employment_chart, wage_chart, profit_chart],
    "Minimum Wage Economy Model",
    model_params,
)

server.port = 8521

if __name__ == "__main__":
    server.launch()