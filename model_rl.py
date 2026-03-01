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
                 non_labor_income, consumption_rate):

        super().__init__(model)
        self.uid = uid

        self.employed = False
        self.employer = None

        self.hours_worked = hours_worked
        self.non_labor_income = non_labor_income
        self.alpha = consumption_rate
        self.job_search_prob = 0.1
        self.monthly_wage = 0

        # ----- RL memory -----
        self.rl_action = 0
        self.last_utility = 0

    # ---------------- Utility ----------------

    def cobb_douglas(self, c, l):
        c = max(c, 1e-6)
        l = max(l, 1e-6)
        return (c ** self.alpha) * (l ** (1 - self.alpha))

    def utility_if_work(self, wage):
        consumption = wage * self.hours_worked + self.non_labor_income
        leisure = self.model.MAX_HOURS - self.hours_worked
        return self.cobb_douglas_utility(consumption, leisure)

    def utility_if_not_work(self):
        consumption = self.non_labor_income
        leisure = self.model.MAX_HOURS
        return self.cobb_douglas_utility(consumption, leisure)

    def cobb_douglas_utility(self, consumption, leisure):
        # Avoid zero values (important for numerical stability)
        consumption = max(consumption, 1e-6)
        leisure = max(leisure, 1e-6) 
        return (consumption ** self.alpha) * (leisure ** (1 - self.alpha))

    # ---------------- RL ACTION ----------------

    def rl_decision(self):

        firms = self.model.firms

        # 0 = stay
        if self.rl_action == 0:
            return

        # 1 = apply randomly
        if self.rl_action == 1:
            if self.employed:
                firms_to_consider = random.sample([f for f in firms if f != self.employer], max(1, (len(firms) - 1) // 2))
            else:
                firms_to_consider = random.sample(firms, max(1, len(firms) // 2))
            self.search_for_jobs(firms_to_consider)

        # 2 = quit
        if self.rl_action == 2 and self.employed:
            self.employer.current_workers.remove(self)
            self.employed = False
            self.employer = None
            self.monthly_wage = 0
            
    def search_for_jobs(self, firms):
        if self.employed:
            if random.random() > self.ON_THE_JOB_SEARCH_PROB:
                return
            else:
                # Consider switching jobs
                # print(f"Worker {self.unique_id} is considering switching jobs.")
                acceptable_firms = []
                for firm in firms:
                    if firm.vacancies > 0:
                        # Consider the utility of working at this firm and compare to current job and switching cost
                        if self.utility_if_work(firm.monthly_wage) - self.calculate_switching_cost() > self.utility_if_work(self.monthly_wage):
                            acceptable_firms.append(firm)

                if acceptable_firms:
                    chosen_firm = max(
                        acceptable_firms,
                        key=lambda f: self.utility_if_work(f.monthly_wage)
                    )

                    # Apply to the chosen firm
                    chosen_firm.applicants.append(self)
                return  # After considering switching jobs, end the search for this step

                # TODO: Consider whether to make this worker search again every step after this for realism?


        acceptable_firms = []

        utility_if_not_work = self.utility_if_not_work()
        for firm in firms:
            if firm.vacancies > 0:
                # Consider the utility of working at this firm and compare to not working and switching cost
                if self.utility_if_work(firm.monthly_wage) - self.calculate_switching_cost() > utility_if_not_work:
                    acceptable_firms.append(firm)

        if acceptable_firms:
            chosen_firm = max(
                acceptable_firms,
                key=lambda f: self.utility_if_work(f.monthly_wage)
            )
            chosen_firm.applicants.append(self)

    def job_search_step(self):
        self.rl_decision()
        

    def step(self):
        # Calculate reward based on utility change
        current_utility = self.utility_if_work(self.monthly_wage) if self.employed else self.utility_if_not_work()
        self.reward = current_utility - self.last_utility
        self.last_utility = current_utility
        

    def rl_stage(self):
        pass
    
    def hire_step(self):
        pass
    
    def onboard_workers_step(self):
        pass
    
    def post_vacancies_step(self):
        pass
    
# =====================================================
# FIRM AGENT (RL CONTROLLED)
# =====================================================

class Firm(Agent):

    def __init__(self, uid, model, capital, rental_rate, productivity, output_price):

        super().__init__(model)
        self.uid = uid

        self.capital = capital
        self.rental_rate = rental_rate
        self.productivity = 60 * productivity
        self.output_price = output_price
        self.alpha = 0.65
        self.current_workers = []
        self.applicants = []
        self.vacancies = 2
        self.gamma = 0.8
        self.steps = 0
        self.pending_workers = []
        
        self.monthly_wage = 0

        # RL
        self.rl_action = 0
        self.profit = 0

    # ---------- RL decision ----------

    def marginal_product_labor(self, A, labor, alpha):
        # Marginal Product of Labor: MPL = dQ/dL = A * alpha * K^beta * L^(alpha-1)
        if labor == 0:
            return 0
        return A * alpha * (self.capital ** (1 - alpha)) * (labor ** (alpha - 1))
       
    def marginal_product_capital(self, A, labor, alpha):
        # Marginal Product of Capital: MPK = dQ/dK = A * (1-alpha) * K^(-alpha) * L^alpha
        if self.capital == 0:
            return 0
        return A * (1 - alpha) * (self.capital ** (-alpha)) * (labor ** alpha)
    
    def produce(self):
        labor = max(len(self.current_workers), 1e-6)
        
        output = output = self.productivity * (self.capital ** (1 - self.alpha)) * (labor ** self.alpha)
        return output
    
    def value_of_marginal_product(self, price, mp):
        return price * mp
    
    def adjust_capital(self, labor, rental_rate):
        # Run every 12 steps to adjust capital based on the value of the marginal product of capital
        mpk = self.marginal_product_capital(self.productivity, labor, self.alpha)
        vmpk = self.value_of_marginal_product(self.output_price, mpk)
        
        if vmpk > rental_rate:
            # invest more in capital
            self.capital *= 1.05  # increase capital by 5%
        elif vmpk < rental_rate:
            # reduce capital
            self.capital *= 0.95  # decrease capital by 5%
    
    def set_initial_wage(self):
        mpl = self.marginal_product_labor(self.productivity, len(self.current_workers), self.alpha)
        self.monthly_wage = max(mpl * self.output_price, 7700)
        self.monthly_wage = int(self.monthly_wage)
        
        for w in self.current_workers:
            w.monthly_wage = self.monthly_wage
            
    
    def rl_decision(self):

        if self.rl_action == 1:
            self.monthly_wage += 100
        elif self.rl_action == 2:
            self.monthly_wage -= 100
            self.monthly_wage = max(self.monthly_wage, 7700)

    # ---------- Hiring ----------

    def hire_step(self):
        random.shuffle(self.applicants)

        hires = min(len(self.applicants), self.vacancies)

        for i in range(hires):
            worker = self.applicants[i]
            worker.employed = True
            worker.employer = self
            self.pending_workers.append(worker)  # Add to pending to start next step
            self.vacancies -= 1

        self.applicants = []

    def onboard_workers_step(self):
        """Move workers from pending to current (1-step hiring delay)"""
        self.current_workers.extend(self.pending_workers)
        self.pending_workers = []
        # All workers should have the same wages based on the firm's current wage setting
        for w in self.current_workers:
            w.monthly_wage = self.monthly_wage
        
    def post_vacancies_step(self):
        self.applicants = []
        self.vacancies = 0

        current_assumed_labor = len(self.current_workers)
        while True:
            mpl = self.marginal_product_labor(
                self.productivity,
                current_assumed_labor + 1,  # Consider hiring one more worker
                self.alpha
            )
            vmp = self.output_price * mpl

            if vmp >= self.monthly_wage:
                self.vacancies += 1
                # Here we just increment the count; actual hiring is handled elsewhere
                current_assumed_labor += 1
            else:
                break
            

    # ---------- Production ----------

    def step(self):
        output = self.produce()
        revenue = output * self.output_price
        wage_cost = sum(w.monthly_wage for w in self.current_workers)
        capital_cost = self.capital * self.rental_rate
        self.profit = revenue - wage_cost - capital_cost
        
        if self.steps % 12 == 0:
            self.adjust_capital(len(self.current_workers), self.rental_rate)
        
        self.reward = self.profit
        
        self.steps += 1
    
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
            stage_list=["rl_stage",
                        "onboard_workers_step",
                        "post_vacancies_step",
                        "job_search_step",
                        "hire_step",
                        "step"],
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
            f = Firm(f"F{i}", self, capital=random.uniform(10,100), rental_rate=500,
                      productivity=random.uniform(0.8, 1.2), output_price=100)
            self.schedule.add(f)
        
        

        self.workers = [a for a in self.schedule.agents if isinstance(a, Worker)]
        self.firms = [a for a in self.schedule.agents if isinstance(a, Firm)]
        
        for firm in self.firms:
            initial_hires = random.randint(3, 5)
            for _ in range(initial_hires):
                if self.workers:
                    # random until find an unemployed worker
                    while True:
                        worker = random.choice(self.workers)
                        if not worker.employed:
                            worker.employed = True
                            worker.employer = firm
                            worker.monthly_wage = firm.monthly_wage
                            firm.current_workers.append(worker)
                            break
            firm.set_initial_wage()
                    
                    
                    

    # ---------- RL stage ----------
    def rl_stage(self):

        for w in self.workers:
            w.rl_action = self.worker_actions[w.unique_id]

        for i, f in enumerate(self.firms):
            f.rl_action = self.firm_actions[i]
            f.rl_decision()

    def step(self):
        self.schedule.step()