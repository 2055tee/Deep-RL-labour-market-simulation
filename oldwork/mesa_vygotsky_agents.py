import mesa
import random

def get_avg_knowledge(model):
    """Returns the average current knowledge of all agents."""
    if not model.schedule.agents:
        return 0
    return sum(a.current_knowledge for a in model.schedule.agents) / len(model.schedule.agents)

def get_avg_zpd_size(model):
    """Returns the average ZPD size (Potential - Current)."""
    if not model.schedule.agents:
        return 0
    zpd_sizes = [a.potential_knowledge - a.current_knowledge for a in model.schedule.agents]
    return sum(zpd_sizes) / len(zpd_sizes)

def get_avg_happiness(model):
    """Returns the average happiness of all agents."""
    if not model.schedule.agents:
        return 0
    return sum(a.happiness for a in model.schedule.agents) / len(model.schedule.agents)

class StudentAgent(mesa.Agent):
    """A student agent with knowledge, potential, and happiness attributes."""

    def __init__(self, unique_id, model, initial_knowledge, initial_potential):
        super().__init__(model)
        self.unique_id = unique_id
        # Vygotsky's Core Attributes
        self.current_knowledge = initial_knowledge  # What the student can do alone
        self.potential_knowledge = initial_potential  # What the student can do with help
        self.happiness = random.uniform(0, 5)        # Motivation/Well-being (0 to 10)
        self.tutoring_efficacy = random.uniform(0.1, 0.5) # How well this agent can tutor (set once)

        # Ensure potential is always >= current_knowledge
        self.potential_knowledge = max(self.current_knowledge, self.potential_knowledge)

    def is_in_zpd(self):
        """Checks if the agent is in their Zone of Proximal Development."""
        # The ZPD is the gap between what they know and what they can potentially learn
        return self.potential_knowledge > self.current_knowledge

    def tutor_peer(self, tutee):
        """
        The tutor (self) scaffolds the tutee, increasing the tutee's current_knowledge.
        Knowledge gain is proportional to the size of the tutee's ZPD.
        """
        if tutee.is_in_zpd():
            # Calculate knowledge transfer based on ZPD size and tutor's efficacy (Scaffolding)
            zpd_size = tutee.potential_knowledge - tutee.current_knowledge
            knowledge_increase = self.tutoring_efficacy * zpd_size * self.model.max_knowledge_gain
            
            # Apply the increase, capping at the potential_knowledge
            tutee.current_knowledge += knowledge_increase
            tutee.current_knowledge = min(tutee.current_knowledge, tutee.potential_knowledge)
            
            # Social-Emotional Gain
            self.happiness += self.model.social_gain_h
            tutee.happiness += self.model.social_gain_h
            
            return True
        return False

    def study_alone(self):
        """Independent learning, a small, non-ZPD based gain."""
        self.current_knowledge += self.model.independent_study_rate
        self.happiness = max(0, self.happiness - self.model.happiness_decay)

    def internalize_knowledge(self):
        """
        Internalization: Consolidates potential into actual knowledge.
        Also models a passive, small increase in potential (ZPD expansion).
        """
        # Internalization rate applies to the gap
        internal_gain = (self.potential_knowledge - self.current_knowledge) * self.model.internalization_rate
        self.current_knowledge += internal_gain

        # ZPD expansion (natural aptitude/growth)
        self.potential_knowledge += self.model.zpd_spread_rate
        
        # Cap all values at 100 for simplicity
        self.current_knowledge = min(100, self.current_knowledge)
        self.potential_knowledge = min(100, self.potential_knowledge)
        self.happiness = min(10, self.happiness)

    def step(self):
        """The agent's main decision-making logic for one simulation step."""
        
        # 1. Internalize and grow potential first
        self.internalize_knowledge()

        # 2. Decide if the agent needs help (in ZPD)
        if self.is_in_zpd() and self.happiness > 2: # Only seek help if motivated
            
            # Find a more knowledgeable peer (potential MKO)
            # The agent is looking for a peer with higher current_knowledge to be a tutor
            tutor_candidates = [
                agent for agent in self.model.schedule.agents
                if agent.current_knowledge > self.current_knowledge and agent.unique_id != self.unique_id
            ]

            if tutor_candidates:
                # Randomly choose a tutor and attempt a session
                tutor = random.choice(tutor_candidates)
                
                # IMPORTANT: The *tutor* runs the tutoring method on the *tutee* (self)
                tutor.tutor_peer(self) 
            else:
                # If no tutor is available, their happiness decays slightly faster
                self.study_alone()
        else:
            # If not in ZPD or unmotivated, they study alone
            self.study_alone()