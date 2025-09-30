"""
MESA Visualization Server for Peer Simulation
"""
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ChartModule

from model import PeerSimulationModel


def agent_portrayal(agent):
    """
    Define how agents are portrayed in the visualization.
    
    Args:
        agent: The agent to portray
        
    Returns:
        Dictionary with portrayal properties
    """
    portrayal = {
        "Shape": "circle",
        "Filled": "true",
        "Layer": 0,
        "Color": "blue",
        "r": 0.5
    }
    return portrayal


# Create visualization elements
grid = CanvasGrid(agent_portrayal, 10, 10, 500, 500)

# Create the server
server = ModularServer(
    PeerSimulationModel,
    [grid],
    "Peer Simulation Model",
    {
        "n_agents": 10,
        "width": 10,
        "height": 10
    }
)

server.port = 8521  # Default MESA port


if __name__ == "__main__":
    server.launch()
