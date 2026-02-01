from mesa import Model, Agent
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from mesa.experimental.devs import ABMSimulator
from mesa.space import MultiGrid
import random
import mesa
import numpy as np

# -------------------
# AGENTS
# -------------------
class Worker(Agent):
    def __init__(self, unique_id, model, productivity, skill_level, savings,res_wage):
        super().__init__(model)
        self.unique_id = unique_id
        self.productivity = productivity
        self.employed = False
        # TODO: Implement reservation wage
        self.reservation_wage = res_wage
        self.wage = 0
        self.savings = savings
        self.skill_level = skill_level

    def step(self):

        if self.employed:
            return 
        else :
            all_available_firms = [a for a in self.model.schedule.agents if isinstance(a, Firm)]
            
            for f in all_available_firms:
                    f.applying_workers.append(self) 
            
                    

class Firm(Agent):
    def __init__(self, unique_id, model, capital, productivity, skill_requirement, fixed_cost, max_worker):
        super().__init__(model)
        self.unique_id = unique_id
        self.capital = capital
        self.productivity = productivity
        self.fixed_cost = fixed_cost
        self.skill_requirement = skill_requirement
        
        self.product_sales_price = 2
        self.applying_workers = []
        self.current_workers = []
        self.max_workers = max_worker # maximum number of workers firm can employ TODO: Base this on capital and productivity later
        self.current_profit = 0

    def step(self):
        # calculate total wage cost and average worker productivity
        total_wage_cost = sum([w.wage for w in self.current_workers])

        revenue = (self.production() * self.product_sales_price)
        self.current_profit = revenue - total_wage_cost - self.fixed_cost

        # update profit and capital
        self.capital += self.current_profit
        
        # payout to current workers
        for w in self.current_workers:
            w.savings += w.wage
        
        self.mrp_check()
        
        self.applying_workers = []

    def production(self):
        if len(self.current_workers) == 0:
            return 0
        return self.productivity * np.sqrt(len(self.current_workers))

    def MPL(self):
        L = len(self.current_workers)
        return self.productivity * (np.sqrt(L + 1) - np.sqrt(L))
        
    def MRP(self):
        L = len(self.current_workers)
        delta_q = self.productivity * (np.sqrt(L + 1) - np.sqrt(L))
        return self.product_sales_price * delta_q
    
    def mrp_check(self):
        mrp = self.MRP()
        print(f"Firm {self.unique_id} MRP: {mrp:.2f}, Min Wage: {self.model.min_wage:.2f}")
        if self.MRP() >= self.model.min_wage:
            self.hire_one_worker()
        if self.MRP() < self.model.min_wage and len(self.current_workers) > 0 and self.current_profit < 0:
            self.fire_worker()
    
    
    def hire_one_worker(self):
        for w in self.applying_workers:
            if w.employed:
                continue
            if len(self.current_workers) >= self.max_workers:
                print(f"Firm {self.unique_id} cannot hire more workers; at max capacity.")
                return False
            if w.skill_level >= self.skill_requirement and self.model.min_wage >= w.reservation_wage:
                w.employed = True
                w.wage = self.model.min_wage
                self.current_workers.append(w)
                print(f"Firm {self.unique_id} hired Worker {w.unique_id} at wage {w.wage}")
                return True
        return False
    
    def fire_worker(self):
        w = random.choice(self.current_workers)
        w.employed = False
        w.wage = 0
        self.current_workers.remove(w)
        print(f"Firm {self.unique_id} fired Worker {w.unique_id}")


# -------------------
# MODEL
# -------------------
class LaborMarketModel(Model):
    def __init__(self, N_workers=100, N_firms=10, min_wage=7700, simulator=None, seed=42):
        super().__init__(seed=seed)
        if simulator:
            self.simulator = simulator
            self.simulator.setup(self)
        self.num_workers = N_workers
        self.num_firms = N_firms
        self.min_wage = min_wage
        self.step_count = 0
        self.running = True # Needed for Mesa to know the model is running
        self.schedule = RandomActivation(self)
        random.seed(seed)
        
        # Create agents
        for i in range(self.num_workers):
            w = Worker(i, self, productivity=1, skill_level=2,
                       savings=20000, res_wage=7700) # monthly expenses
            self.schedule.add(w)
        for i in range(self.num_firms):
            f = Firm(f"F{i}", self, capital=random.uniform(250000, 750000), productivity=random.uniform(15000,20000),
                    skill_requirement=1, fixed_cost=random.uniform(1000, 1500), max_worker=100)
            self.schedule.add(f)

        # Helper attributes for Solara display
        self.average_wage = 0
        self.employment_rate = 0
        self.starting_min_wage = min_wage
        self.number_of_workers = N_workers
        self.number_of_firms = N_firms
        
            
        # Data Collector
        model_reporters={
            "EmploymentRate": self.compute_employment_rate,
            "AverageWage": self.compute_avg_wage,
            "AverageProfit": self.compute_avg_profit,
            "AvgFirmSize": self.get_firm_size,
            "AvgFirmCapital": self.get_avg_firm_capital,
            "MinWage": self.get_min_wage,
            # NEW: Collect lists of all values for later analysis/distribution plotting
            "AllFirmSizes": self.get_firm_sizes_list, 
            "AllFirmCapitals": self.get_firm_capitals_list,
            "AllEmployedWages": self.get_employed_wages_list,
            "AllFirmProfits": self.get_firm_profits_list,
        }
        
        self.datacollector = mesa.DataCollector(model_reporters)
        self.datacollector.collect(self)
    
    def step(self):
        # increase min wage over time   
        # print(f"min_wage increased to {self.min_wage} at step {self.step_count}")
        self.schedule.step()
        self.datacollector.collect(self)
        self.update_data()
        self.step_count += 1
        
    def steps(self):
        return self.step_count
    
    def update_data(self):
        self.average_wage = self.compute_avg_wage()
        self.employment_rate = self.compute_employment_rate()

    def compute_employment_rate(self):
        workers = [a for a in self.schedule.agents if isinstance(a, Worker)]
        employed = [w for w in workers if w.employed]
        return len(employed) / len(workers)

    def compute_avg_wage(self):
        wages = [w.wage for w in self.schedule.agents if isinstance(w, Worker)]
        return np.mean(wages) if wages else 0

    def compute_avg_profit(self):
        profits = [f.current_profit for f in self.schedule.agents if isinstance(f, Firm)]
        return np.mean(profits) if profits else 0

    def get_firm_size(self):
        # average number of workers per firm
        firm_sizes = [len(f.current_workers) for f in self.schedule.agents if isinstance(f, Firm)]
        return np.mean(firm_sizes) if firm_sizes else 0
    
    def get_avg_firm_capital(self):
        firm_capitals = [f.capital for f in self.schedule.agents if isinstance(f, Firm)]
        return np.mean(firm_capitals) if firm_capitals else 0
    
    def get_min_wage(self):
        return self.min_wage  # monthly minimum wage
    
    def get_firm_sizes_list(self):
        return [len(f.current_workers) for f in self.schedule.agents if isinstance(f, Firm)]
    
    def get_firm_capitals_list(self):
        return [f.capital for f in self.schedule.agents if isinstance(f, Firm)]
    
    def get_employed_wages_list(self):
        return [w.wage for w in self.schedule.agents if isinstance(w, Worker)]
    
    def get_firm_profits_list(self):
        return [f.current_profit for f in self.schedule.agents if isinstance(f, Firm)]
    