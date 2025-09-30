"""
Run the Peer Simulation Model
"""
from model import PeerSimulationModel


def run_simulation(steps=100):
    """
    Run the simulation for a specified number of steps.
    
    Args:
        steps: Number of steps to run the simulation
    """
    # Initialize model
    model = PeerSimulationModel(n_agents=10, width=10, height=10)
    
    # Run simulation
    for i in range(steps):
        model.step()
        if i % 10 == 0:
            print(f"Step {i} completed")
    
    print(f"Simulation completed after {steps} steps")
    
    # Get collected data
    model_data = model.datacollector.get_model_vars_dataframe()
    agent_data = model.datacollector.get_agent_vars_dataframe()
    
    return model, model_data, agent_data


if __name__ == "__main__":
    model, model_data, agent_data = run_simulation(steps=100)
    print("\nSimulation finished!")
    print("Model data shape:", model_data.shape if not model_data.empty else "No data")
    print("Agent data shape:", agent_data.shape if not agent_data.empty else "No data")
