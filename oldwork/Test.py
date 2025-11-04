import random
import pandas as pd
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

# --- Agent Class ---
class StudentAgent(Agent):
    """An agent that represents a student with a knowledge score and a learning style."""
    def __init__(self, unique_id, model, learning_style):
        super().__init__(unique_id)
        self.model = model
        self.unique_id = unique_id
        self.random = random.Random(self.unique_id)  # Seeded random for reproducibility.
        self.learning_style = learning_style
        # A knowledge score from 0 to 100, initialized randomly.
        self.knowledge_score = random.randint(0, 100)

    def step(self):
        """A student agent's daily routine."""
        # 50% chance to self-study, 50% chance to find a peer to tutor.
        if self.random.random() < 0.5:
            self.self_study()
        else:
            self.tutor_peer()

    def self_study(self):
        """Increases the agent's knowledge score based on their learning style."""
        # The gain from self-study can be a parameter to tune based on dataset.
        self_study_gain = self.model.self_study_gain
        self.knowledge_score = min(100, self.knowledge_score + self_study_gain)

    def tutor_peer(self):
        """Randomly selects another agent to tutor and transfers some knowledge."""
        # Find all other agents in the model.
        other_agents = self.model.schedule.agents.copy()
        other_agents.remove(self)

        if not other_agents:
            return  # No other agents to tutor.

        # Select a random peer to tutor.
        peer = self.random.choice(other_agents)
        
        # Knowledge transfer is a function of the tutor's score and a fixed efficiency.
        knowledge_transfer = (self.knowledge_score - peer.knowledge_score) * self.model.tutoring_efficiency
        
        if knowledge_transfer > 0:
            peer.knowledge_score = min(100, peer.knowledge_score + knowledge_transfer)

# --- Model Class ---
class TutoringModel(Model):
    """A model of a peer tutoring system."""
    def __init__(self, num_students, self_study_gain, tutoring_efficiency):
        self.num_students = num_students
        self.self_study_gain = self_study_gain
        self.tutoring_efficiency = tutoring_efficiency
        self.random = random.Random()  # Required by Mesa 3.x schedulers
        self.schedule = RandomActivation(self)
        self.running = True

        # To align with the dataset, we can define learning styles.
        learning_styles = ['Visual', 'Auditory', 'Kinesthetic', 'Reading/Writing']

        # Create agents.
        for i in range(self.num_students):
            # Assign a random learning style from the defined list.
            style = random.choice(learning_styles)
            a = StudentAgent(i, self, style)
            self.schedule.add(a)

        # Data collector for analysis.
        self.datacollector = DataCollector(
            model_reporters={"Average_Knowledge": lambda m: sum(a.knowledge_score for a in m.schedule.agents) / m.num_students},
            agent_reporters={"Knowledge": "knowledge_score", "Learning_Style": "learning_style"}
        )

    def step(self):
        """Advance the model by one step."""
        self.datacollector.collect(self)
        self.schedule.step()

# --- Running the Simulation and Comparison ---
if __name__ == '__main__':
    # You will need to load the dataset for comparison.
    # Note: Replace 'path/to/your/dataset.csv' with the actual file path.
    # data = pd.read_csv('path/to/your/dataset.csv')
    # Use the number of students from the dataset.
    # num_students = len(data)
    
    # For this example, we will use a fixed number.
    NUM_STUDENTS = 50
    SIMULATION_STEPS = 100

    model = TutoringModel(num_students=NUM_STUDENTS, self_study_gain=1.0, tutoring_efficiency=0.1)
    
    for i in range(SIMULATION_STEPS):
        model.step()

    # Get the data from the simulation.
    model_data = model.datacollector.get_model_vars_dataframe()
    agent_data = model.datacollector.get_agent_vars_dataframe()
    
    print("Simulation Results (Model-level):\n", model_data.tail())
    print("\nSimulation Results (Agent-level):\n", agent_data.head())
    
    # --- How to use this for comparison with the dataset ---
    # 1. Parameter Tuning: Analyze the dataset to get a sense of real-world values.
    #    - What is the average `StudyHours`? This can inform your `self_study_gain`.
    #    - What is the distribution of `FinalGrade`? Use this as a target for your simulation's `knowledge_score` distribution.
    #    - How does `LearningStyle` correlate with `FinalGrade`? This could be used to adjust the learning gain for different agent types.

    # 2. Comparison Metrics: Once you have a final distribution of knowledge scores from your simulation run, you can compare it to the `FinalGrade` distribution in the dataset.
    #    - Plot histograms of both the simulation's final scores and the dataset's final grades side-by-side. 
    #    - Use statistical tests (e.g., a two-sample Kolmogorov-Smirnov test) to check if the distributions are similar.