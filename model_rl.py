# labor_model_rl.py

import random
import numpy as np
from mesa import Model, Agent
from mesa.time import StagedActivation

# =====================================================
# WORKER AGENT (RL CONTROLLED)
# =====================================================

class Worker(Agent):

    def __init__(self, uid, model, hours_worked,
                 non_labor_income, alpha):

        super().__init__(model)
        self.uid = uid

        self.employed = False
        self.employer = None

        self.hours_worked = hours_worked
        self.non_labor_income = non_labor_income
        self.alpha = alpha

        self.monthly_wage = 0

        # ----- RL memory -----
        self.rl_action = 0
        self.last_utility = 0

    # ---------------- Utility ----------------

    def cobb_douglas(self, c, l):
        c = max(c, 1e-6)
        l = max(l, 1e-6)
        return (c ** self.alpha) * (l ** (1 - self.alpha))

    def utility(self):
        leisure = self.model.MAX_HOURS - self.hours_worked
        consumption = self.monthly_wage + self.non_labor_income
        return self.cobb_douglas(consumption, leisure)

    # ---------------- RL ACTION ----------------

    def rl_decision(self):

        firms = self.model.firms

        # 0 = stay
        if self.rl_action == 0:
            return

        # 1 = apply randomly
        if self.rl_action == 1:
            firm = random.choice(firms)
            if firm.vacancies > 0:
                firm.applicants.append(self)

        # 2 = quit
        if self.rl_action == 2 and self.employed:
            self.employer.current_workers.remove(self)
            self.employed = False
            self.employer = None
            self.monthly_wage = 0

    def job_search_step(self):
        self.rl_decision()

    def step(self):
        new_u = self.utility()
        self.reward = new_u - self.last_utility
        self.last_utility = new_u

    def rl_stage(self):
        pass
    
    def hire_step(self):
        pass
    
    

# =====================================================
# FIRM AGENT (RL CONTROLLED)
# =====================================================

class Firm(Agent):

    def __init__(self, uid, model, capital):

        super().__init__(model)
        self.uid = uid

        self.capital = capital
        self.monthly_wage = random.randint(15000, 25000)

        self.current_workers = []
        self.applicants = []
        self.vacancies = 2

        # RL
        self.rl_action = 0
        self.profit = 0

    # ---------- RL decision ----------

    def rl_decision(self):

        if self.rl_action == 1:
            self.monthly_wage *= 1.05

        elif self.rl_action == 2:
            self.monthly_wage *= 0.95

        elif self.rl_action == 3:
            self.vacancies += 1

        elif self.rl_action == 4 and self.current_workers:
            w = random.choice(self.current_workers)
            self.current_workers.remove(w)
            w.employed = False
            w.monthly_wage = 0

    # ---------- Hiring ----------

    def hire_step(self):

        random.shuffle(self.applicants)

        hires = min(len(self.applicants), self.vacancies)

        for i in range(hires):
            w = self.applicants[i]
            w.employed = True
            w.employer = self
            w.monthly_wage = self.monthly_wage
            self.current_workers.append(w)

        self.applicants = []
        self.vacancies = 0

    # ---------- Production ----------

    def step(self):

        labor = len(self.current_workers)
        output = 10 * (labor ** 0.7)

        revenue = output * 100
        wage_cost = sum(w.monthly_wage for w in self.current_workers)

        self.profit = revenue - wage_cost
    
    def rl_stage(self):
        pass
    
    def job_search_step(self):
        pass


# =====================================================
# LABOR MARKET MODEL
# =====================================================

class LaborMarketModel(Model):

    def __init__(self, N_workers=50, N_firms=5):

        super().__init__()

        self.MAX_HOURS = 192

        self.schedule = StagedActivation(
            self,
            stage_list=[
                "rl_stage",
                "job_search_step",
                "hire_step",
                "step"
            ],
            shuffle=False
        )

        # create agents
        for i in range(N_workers):
            w = Worker(
                i,
                self,
                160,
                random.uniform(0, 3000),
                random.uniform(0.3, 0.7),
            )
            self.schedule.add(w)

        for i in range(N_firms):
            f = Firm(f"F{i}", self, capital=100)
            self.schedule.add(f)

        self.workers = [a for a in self.schedule.agents if isinstance(a, Worker)]
        self.firms = [a for a in self.schedule.agents if isinstance(a, Firm)]

    # ---------- RL stage ----------
    def rl_stage(self):

        for w in self.workers:
            w.rl_action = self.worker_actions[w.unique_id]

        for i, f in enumerate(self.firms):
            f.rl_action = self.firm_actions[i]
            f.rl_decision()

    def step(self):
        self.schedule.step()