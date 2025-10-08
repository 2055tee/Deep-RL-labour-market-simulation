import mesa
import pandas as pd
from mesa_vygotsky_agents import StudentAgent # Import the agent file
import random
from mesa.space import SingleGrid

# --- Helper Functions for Data Collection ---
def get_avg_knowledge(model):
    """Returns the average current knowledge of all agents."""
    return sum(a.current_knowledge for a in model.schedule.agents) / len(model.schedule.agents)

def get_avg_zpd_size(model):
    """Returns the average ZPD size (Potential - Current)."""
    zpd_sizes = [a.potential_knowledge - a.current_knowledge for a in model.schedule.agents]
    return sum(zpd_sizes) / len(zpd_sizes)

def get_avg_happiness(model):
    """Returns the average happiness of all agents."""
    return sum(a.happiness for a in model.schedule.agents) / len(model.schedule.agents)

# --- The Model Class ---
class TutoringModel(mesa.Model):
    """
    A model simulating peer tutoring based on Vygotsky's ZPD.
    """
    def __init__(
        self,
        N=25,
        max_knowledge_gain=0.5,
        independent_study_rate=0.05,
        internalization_rate=0.02,
        zpd_spread_rate=0.01,
        social_gain_h=1.0,
        happiness_decay=0.05,
        width=10,
        height=10,
        simulator=None,
    ):
        super().__init__()
        self.simulator = simulator
        self.simulator.setup(self)
        self.random = random.Random()
        self.num_agents = N
        self.schedule = mesa.time.RandomActivation(self)
        self.running = True

        # Model Parameters (can be adjusted via interface)
        self.max_knowledge_gain = max_knowledge_gain
        self.independent_study_rate = independent_study_rate
        self.internalization_rate = internalization_rate
        self.zpd_spread_rate = zpd_spread_rate
        self.social_gain_h = social_gain_h
        self.happiness_decay = happiness_decay


        # Create grid for visualization
        self.grid = SingleGrid(10, 10, torus=False)

        # Create agents and place them on the grid
        for i in range(self.num_agents):
            k = self.random.uniform(10, 50)
            p = self.random.uniform(k, 70)
            a = StudentAgent(i, self, k, p)
            self.schedule.add(a)
            x = i % 10
            y = i // 10
            self.grid.place_agent(a, (x, y))

        # Set up data collector
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Average Knowledge": get_avg_knowledge,
                "Average ZPD Size": get_avg_zpd_size,
                "Average Happiness": get_avg_happiness,
            }
        )
        self.datacollector.collect(self)

    def step(self): 
        """Advance the model by one step."""
        self.datacollector.collect(self)
        self.schedule.step()
        
        # Stop condition: if average knowledge reaches 90, the simulation is done
        if get_avg_knowledge(self) >= 90:
            self.running = False


# --- Main Run Logic ---
if __name__ == '__main__':
    # Define parameters for the simulation
    num_students = 25
    steps_to_run = 100

    # Initialize and run the model
    model = TutoringModel(num_students)
    for i in range(steps_to_run):
        if model.running:
            model.step()
        else:
            print(f"Simulation ended early at step {i} due to high average knowledge.")
            break

    # Collect and display results
    model_data = model.datacollector.get_model_vars_dataframe()
    agent_data = model.datacollector.get_agent_vars_dataframe()

    print("\n--- Model-Level Results (First 10 Steps) ---")
    print(model_data.head(10))

    print("\n--- Final Agent States (Sample) ---")
    final_agent_states = agent_data.xs(model.schedule.steps - 1, level="Step")
    print(final_agent_states.sample(5))

    # Optional: Save to CSV for plotting
    # model_data.to_csv("vygotsky_model_results.csv")