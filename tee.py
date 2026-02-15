from mesa import Model, Agent
from mesa.time import RandomActivation, StagedActivation
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
    def __init__(self, unique_id, model, productivity, hours_worked, non_labor_income, consumption_weight):
        super().__init__(model)
        self.unique_id = unique_id
        self.employed = False

        self.productivity = productivity
        self.non_labor_income = non_labor_income # monthly non-labor income V
        self.alpha = consumption_weight  # weight on share of income spent on consumption
        self.hours_worked = hours_worked
        self.wage = 0
        self.reservation_wage = 0
        
        self.probability_of_switching_job = 0.1  # probability of finding a job when already employed

    def calculate_leisure(self):
        return self.model.MAX_HOURS - self.hours_worked
    
    def cobb_douglas_utility(self, consumption, leisure):
        # Avoid zero values (important for numerical stability)
        consumption = max(consumption, 1e-6)
        leisure = max(leisure, 1e-6)
        return (consumption ** self.alpha) * (leisure ** (1 - self.alpha))
    
    def search_for_jobs(self, firms):
        if self.employed:
            if random.random() > self.probability_of_switching_job:
                return
            else:
                # Consider switching jobs
                print(f"Worker {self.unique_id} is considering switching jobs.")
                # TODO: Implement job switching logic later
                # TODO: Consider whether to make this worker search again every step after this for realism?
                pass

        acceptable_firms = []

        utility_if_not_work = self.utility_if_not_work()
        for firm in firms:
            if firm.vacancies > 0:
                if self.utility_if_work(firm.current_wage) > utility_if_not_work:
                    acceptable_firms.append(firm)

        if acceptable_firms:
            chosen_firm = max(
                acceptable_firms,
                key=lambda f: self.utility_if_work(f.current_wage)
            )
            chosen_firm.applicants.append(self)
    
    def utility_if_work(self, wage):
        consumption = wage * self.hours_worked + self.non_labor_income
        leisure = self.model.MAX_HOURS - self.hours_worked
        utility = (consumption ** self.alpha) * (leisure ** (1 - self.alpha))
        # print(f"Worker {self.unique_id} utility if work: {utility}, wage={wage}, consumption={consumption}, leisure={leisure}")
        return utility

    def utility_if_not_work(self):
        consumption = self.non_labor_income
        leisure = self.model.MAX_HOURS
        utility = (consumption ** self.alpha) * (leisure ** (1 - self.alpha))
        # print(f"Worker {self.unique_id} utility if not work: {utility}, consumption={consumption}, leisure={leisure}")
        return utility

    def job_search_step(self):
        all_firms = [a for a in self.model.schedule.agents if isinstance(a, Firm)]
        self.search_for_jobs(all_firms)
    
    # ------------------- Cancel mesa overlapping step call -------------------
    def pre_hiring_step(self):
        pass
    
    def firm_main_step(self):
        pass
    
    

class Firm(Agent):
    def __init__(self, unique_id, model, capital, productivity):
        super().__init__(model)
        self.unique_id = unique_id
        self.capital = capital
        self.productivity = productivity
        self.product_sales_price = 2
        self.applicants = []
        self.current_workers = []
        self.current_profit = 0
        self.machine_productivity = 0.5  # productivity factor for machines
        self.current_wage = 45  # initial wage offer
        self.expected_working_hours = 160
        
        self.machine_investment = capital * 0.5 
        self.capital = capital - self.machine_investment
        
        self.worker_diminishing_return = 0.65  # parameter for diminishing returns to labor
        self.machine_diminishing_return = 0.35  # parameter for diminishing returns to capital
        
        self.machine_depreciation_rate = 0.9  # x% depreciation per time step 
        self.machine_operation_cost_rate = 0.05  # y% operation cost per time step
        self.machine_reinvestment_rate = 0.05  # reinvest 20% of profits into machines each step
        self.machine_interest_cost_rate = 0.08  # interest cost on machine investment
        self.machine_price = 1500
        
        self.vacancies = 0
        self.same_wage_steps = 0  # counter for steps with the same wage offer
    
    
    def pre_hiring_step(self):
        self.applicants = []
        self.vacancies = 0
        
        self.mrp_check()
    

    def firm_main_step(self):
        # print(f"applicating_workers: {[w.unique_id for w in self.applicants]}")
        
        # Hire/Fire first
        self.hire_workers()
        
        # production phase with current workers
        self.output = self.production()
        # print(f"Firm {self.unique_id} produced output: {self.output:.2f}")
        
        # calculate revenue
        revenue = (self.output * self.product_sales_price)
        
        
        # wage distribution
        total_wage_cost = sum(w.wage * w.hours_worked for w in self.current_workers)
        
        # calculate profit
        self.current_profit = revenue - total_wage_cost
        print(f"Firm {self.unique_id} Revenue: {revenue:.2f}, Wage Cost: {total_wage_cost:.2f}, Profit: {self.current_profit:.2f}")
        self.capital += self.current_profit
        
        # Reinvest in machines        
        self.check_machine_reinvestment()
        
        # Clear applying workers list for next step
        self.applicants = []

    def production(self):
        l = sum(w.hours_worked for w in self.current_workers)  # effective labor input based on hours worked
        k = self.machine_investment
        return self.productivity * (l ** self.worker_diminishing_return) * (k ** self.machine_diminishing_return)

    def MPL(self, worker_count):
        l = (worker_count * self.expected_working_hours)  # effective labor input based on hours worked
        k = self.machine_investment
        return (self.productivity * self.worker_diminishing_return * (l ** (self.worker_diminishing_return - 1)) * (k ** self.machine_diminishing_return))
    
    def MPK(self):
        l = sum(w.hours_worked for w in self.current_workers)  # effective labor input based on hours worked
        k = self.machine_investment
        if k == 0:
            return 0
        return (self.productivity * self.machine_diminishing_return * (l ** self.worker_diminishing_return) * (k ** (self.machine_diminishing_return - 1)))
    
    def MRP(self, additional_workers=1):

        current_workers = len(self.current_workers)

        workers_before = current_workers
        workers_after = current_workers + additional_workers

        hours_before = workers_before * self.expected_working_hours
        hours_after = workers_after * self.expected_working_hours

        production_before = self.production_from_hours(hours_before)
        production_after = self.production_from_hours(hours_after)

        return (production_after - production_before) * self.product_sales_price
        
    def production_from_hours(self, total_hours):
        l = total_hours
        k = self.machine_investment
        return self.productivity * (l ** self.worker_diminishing_return) * (k ** self.machine_diminishing_return)
    
    def MRPK(self):
        l = len(self.current_workers)
        k_current = self.machine_investment
        k_next = k_current * self.machine_reinvestment_rate
        
        # Calculate Total Product with current capital
        if l == 0:
            return 0
            
        # Total Product with current capital
        tp_current = self.productivity * (l ** self.worker_diminishing_return) * (k_current ** self.machine_diminishing_return)
        
        # Total Product with one additional unit of capital
        tp_next = self.productivity * (l ** self.worker_diminishing_return) * (k_next ** self.machine_diminishing_return)
        
        # Change in output * price
        delta_q = tp_next - tp_current
        return self.product_sales_price * delta_q
            
    def check_machine_reinvestment(self):
        # check only every 12 steps (months)
        if self.model.step_count % 12 != 0:
            return
        
        # machine depreciation and operation costs
        operation_cost = self.machine_investment * self.machine_operation_cost_rate
        interest_cost = self.machine_investment * self.machine_interest_cost_rate
        depreciation_cost = self.machine_investment * (1 - self.machine_depreciation_rate)
        total_machine_cost = operation_cost + interest_cost + depreciation_cost
        # depriciate existing machines
        self.machine_investment *= self.machine_depreciation_rate
        self.current_profit -= total_machine_cost
        
        # Calculate the Marginal Revenue Product of ONE dollar of capital
        l = max(1, len(self.current_workers))
        k = self.machine_investment
        
        # Using the Calculus version for the 'slope' of the MPK
        # MPK = A * beta * L^alpha * K^(beta-1)
        mpk = self.productivity * self.machine_diminishing_return * (l ** self.worker_diminishing_return) * (k ** (self.machine_diminishing_return - 1))
        mrpk = self.product_sales_price * mpk
        
        # User Cost (r) per step: Interest + Depreciation + Ops
        r = self.machine_interest_cost_rate + (1 - self.machine_depreciation_rate) + self.machine_operation_cost_rate
        
        # STOP GROWING if the benefit (mrpk) is less than the cost (r)
        if mrpk > r:
            # Invest only enough to move toward the equilibrium
            potential_investment = self.current_profit * self.machine_reinvestment_rate
            if potential_investment > 0:
                self.machine_investment += potential_investment
                self.capital -= potential_investment
            
    def mrp_check(self):
        labor_cost = self.current_wage * self.model.worker_hours_worked_per_month
        
        new_hirings = 0
        
        # check current MRP against minimum wage and determine whether to fire worker or not
        if self.MRP() < labor_cost and len(self.current_workers) > 0 and self.current_profit < 0:
            self.fire_worker()
            return 0
        
        
        while True:
            mrp = self.MRP(new_hirings + 1)
            # print(f"Firm {self.unique_id} checking MRP for hiring {new_hirings + 1} workers: MRP = {mrp:.2f} vs Labor Cost = {self.current_wage * self.model.worker_hours_worked_per_month * (new_hirings + 1):.2f}")
            if mrp < self.current_wage * self.model.worker_hours_worked_per_month * (new_hirings + 1):
                break
            new_hirings += 1
        
        self.vacancies = new_hirings
        return new_hirings
      
            
    
    def hire_workers(self):
        # if hire worker at the same wage for 3 steps and no worker increase wage by 500
        # if self.vacancies <= 0:
        #     return
        # if self.vacancies > 0:
        #     if self.same_wage_steps >= 3:
        #         self.current_wage += 500
        #         self.same_wage_steps = 0
        #     else:
        #         self.same_wage_steps += 1
        
        
        # randomly hire from applying workers up to new_hirings
        random.shuffle(self.applicants)
        for w in self.applicants:
            if self.vacancies <= 0:
                break
            if not w.employed and w.reservation_wage <= self.current_wage:
                w.employed = True
                w.wage = self.current_wage
                self.current_workers.append(w)
                self.vacancies -= 1
                # print(f"Firm {self.unique_id} hired Worker {w.unique_id} at wage {w.wage}")
        
    
    def fire_worker(self):
        w = random.choice(self.current_workers)
        w.employed = False
        w.wage = 0
        self.current_workers.remove(w)
        # print(f"Firm {self.unique_id} fired Worker {w.unique_id}")

    # Avoid Mesa overlapping step call
    def job_search_step(self):
        pass


# -------------------
# MODEL
# -------------------
class LaborMarketModel(Model):
    def __init__(self, N_workers=1000, N_firms=10, min_wage=44, simulator=None, seed=42):
        super().__init__(seed=seed)
        if simulator:
            self.simulator = simulator
            self.simulator.setup(self)
        self.num_workers = N_workers
        self.MAX_HOURS = 192 # max working hours per month
        self.num_firms = N_firms
        self.min_wage = min_wage
        self.step_count = 0
        self.worker_hours_worked_per_month = 160
        self.running = True # Needed for Mesa to know the model is running
        self.schedule = StagedActivation(self, stage_list=["pre_hiring_step", "job_search_step", "firm_main_step"], shuffle=False)
        random.seed(seed)
        
        # Create agents
        for i in range(self.num_workers):
            w = Worker(i, self, productivity=1, 
                       non_labor_income=random.uniform(0, 1000), consumption_weight=random.uniform(0.3, 0.7), hours_worked=self.worker_hours_worked_per_month)
            self.schedule.add(w)
        for i in range(self.num_firms):
            f = Firm(f"F{i}", self, capital=random.uniform(45000, 75000), productivity=random.uniform(10, 12))
            self.schedule.add(f)

        def labor_supply(workers, wage):
            return sum(1 for w in workers if wage > w.reservation_wage)

        def labor_demand(firms):
            demand = 0
            for firm in firms:
                demand = firm.mrp_check()
            return demand

        # def find_market_clearing_wage(workers, firms, wage_grid):
        #     for wage in wage_grid:
        #         supply = labor_supply(workers, wage)
        #         demand = labor_demand(firms)
        #         if abs(supply - demand) <= 1:
                    # return wage

        # Find market clearing wage
        workers = [a for a in self.schedule.agents if isinstance(a, Worker)]
        firms = [a for a in self.schedule.agents if isinstance(a, Firm)]
        wage_grid = np.arange(20, 100, 2)  # hourly wage grid from 20 Baht to 100 Baht
        # market_clearing_wage = find_market_clearing_wage(workers, firms, wage_grid)
        # print(f"Estimated Market Clearing Wage: {market_clearing_wage}")

        # Helper attributes for Solara display
        self.average_wage = 0
        self.employment_rate = 0
        self.starting_min_wage = min_wage
        self.number_of_workers = N_workers
        self.number_of_firms = N_firms
        
            
        # Data Collector
        model_reporters={
            # Mesa reporters
            "EmploymentRate": self.compute_employment_rate,
            "AverageWage": self.compute_avg_wage,
            "AverageProfit": self.compute_avg_profit,
            "AvgFirmSize": self.get_firm_size,
            "AvgFirmCapital": self.get_avg_firm_capital,
            "MinWage": self.get_min_wage,
            "AverageMachineInvestment": self.get_avg_machine_investment,
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
    
    def get_avg_machine_investment(self):
        machine_investments = [f.machine_investment for f in self.schedule.agents if isinstance(f, Firm)]
        return np.mean(machine_investments) if machine_investments else 0
    