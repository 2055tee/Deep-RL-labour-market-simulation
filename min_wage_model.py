from mesa import Model, Agent
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import random
import numpy as np

# -------------------
# AGENTS
# -------------------
class Worker(Agent):
    def __init__(self, unique_id, model, productivity):
        super().__init__(model)
        self.unique_id = unique_id
        self.productivity = productivity
        self.employed = False
        self.wage = 0
        self.income = 0

    def step(self):
        # Worker just resets employment each round
        self.employed = False
        self.wage = 0


class Firm(Agent):
    def __init__(self, unique_id, model, capital, productivity):
        super().__init__(model)
        self.unique_id = unique_id  
        self.capital = capital
        self.productivity = productivity
        self.profit = 0

    def step(self):
        # Firms try to hire workers based on productivity and minimum wage
        available_workers = [w for w in self.model.schedule.agents if isinstance(w, Worker) and not w.employed]
        random.shuffle(available_workers)

        desired_workers = int(self.productivity * 10)  # crude "labor demand"
        min_wage = self.model.min_wage

        hired = []
        for worker in available_workers[:desired_workers]:
            if self.capital >= min_wage:
                # hire worker
                worker.employed = True
                worker.wage = min_wage
                worker.income = min_wage
                self.capital -= min_wage
                hired.append(worker)
            else:
                break

        # Firm profit = productivity * workers - wages paid
        self.profit = max(0, (self.productivity * len(hired) * 20) - len(hired) * min_wage)


# -------------------
# MODEL
# -------------------
class LaborMarketModel(Model):
    def __init__(self, N_workers=100, N_firms=10, min_wage=350):
        self.num_workers = N_workers
        self.num_firms = N_firms
        self.min_wage = min_wage
        self.random = random.Random()
        self.schedule = RandomActivation(self)

        # Create agents
        for i in range(self.num_workers):
            w = Worker(i, self, productivity=random.uniform(0.5, 1.5))
            self.schedule.add(w)

        for i in range(self.num_firms):
            f = Firm(f"F{i}", self, capital=random.uniform(5000, 10000), productivity=random.uniform(0.5, 2.0))
            self.schedule.add(f)

        self.datacollector = DataCollector(
            model_reporters={
                "EmploymentRate": self.compute_employment_rate,
                "AverageWage": self.compute_avg_wage,
                "AverageProfit": self.compute_avg_profit
            }
        )

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

    def compute_employment_rate(self):
        workers = [a for a in self.schedule.agents if isinstance(a, Worker)]
        employed = [w for w in workers if w.employed]
        return len(employed) / len(workers)

    def compute_avg_wage(self):
        wages = [w.wage for w in self.schedule.agents if isinstance(w, Worker) and w.employed]
        return np.mean(wages) if wages else 0

    def compute_avg_profit(self):
        profits = [f.profit for f in self.schedule.agents if isinstance(f, Firm)]
        return np.mean(profits) if profits else 0