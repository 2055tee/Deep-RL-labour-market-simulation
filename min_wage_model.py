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
    def __init__(self, unique_id, model, productivity, skill_level, non_labor_income, consumption_weight):
        super().__init__(model)
        self.unique_id = unique_id
        self.employed = False

        self.productivity = productivity
        self.non_labor_income = non_labor_income # monthly non-labor income V
        self.alpha = consumption_weight  # weight on share of income spent on consumption
        self.hours_worked = 0  # hours worked in the current month
        self.wage = 0
        self.skill_level = skill_level

    def calculate_leisure(self):
        return self.model.MAX_HOURS - self.hours_worked
    
    def cobb_douglas_utility(self, consumption, leisure):
        # Avoid zero values (important for numerical stability)
        consumption = max(consumption, 1e-6)
        leisure = max(leisure, 1e-6) 
        return (consumption ** self.alpha) * (leisure ** (1 - self.alpha))
    
    def search_for_job(self):
        # Reservation wage based on non-labor income and preferences
        leisure_if_unemployed = self.model.MAX_HOURS
        consumption_if_unemployed = self.non_labor_income
        utility_if_unemployed = self.cobb_douglas_utility(consumption_if_unemployed, leisure_if_unemployed)

        # Find reservation wage by iterating over possible wages
        possible_wages = range(1000, 20000, 500)  # Monthly wage grid
        reservation_wage = 0

        for wage in possible_wages:
            hours = self.choose_hours(wage)
            consumption = (wage * hours) + self.non_labor_income
            leisure = self.model.MAX_HOURS - hours
            utility = self.cobb_douglas_utility(consumption, leisure)

            if utility >= utility_if_unemployed:
                reservation_wage = wage
                break

        self.reservation_wage = reservation_wage

    def choose_hours(self, wage):
        # Typical labor econ values:
        # beta ≈ 0.6–0.8 → consumption-focused
        # 1−beta reflects leisure preference

        possible_hours = [0, 40, 80, 120, 160]  # Possible working hours per month
        best_hours = 0
        best_utility = -float('inf')

        for h in possible_hours:
            consumption = (wage * h) + self.non_labor_income
            leisure = self.model.MAX_HOURS - h
            utility = self.cobb_douglas_utility(consumption, leisure)

            if utility > best_utility:
                best_utility = utility
                best_hours = h

        return best_hours

    def step(self):

        if self.employed:
            return 
        else :
            all_available_firms = [a for a in self.model.schedule.agents if isinstance(a, Firm)]
            
            for f in all_available_firms:
                    f.applying_workers.append(self)
            
class Firm(Agent):
    def __init__(self, unique_id, model, productivity, output_price, skill_requirement):
        super().__init__(model)
        self.unique_id = unique_id

        self.capital = 100000  # initial capital
        self.rental_rate = 0.01  # cost of capital rental per time step
        self.base_productivity = 500 
        self.productivity_multiplier = productivity
        self.output_price = output_price
        self.productivity = self.base_productivity * self.productivity_multiplier
        self.alpha = 0.65  # labor share

        self.skill_requirement = skill_requirement
        self.applying_workers = []
        self.current_workers = []

    def produce(self):
        labor = sum(w.hours_worked for w in self.current_workers) # total labor input in hours
        labor = max(labor, 1e-6)  # Avoid zero labor input

        # Cobb-Douglas production function: Q = A * K^(1-alpha) * L^alpha 
        output = self.productivity * (self.capital ** (1 - self.alpha)) * (labor ** self.alpha)
        return output
    
    
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
    
    def value_of_marginal_product(price, mp):
        return price * mp
    
    def adjust_capital(self, labor, rental_rate):
        mpk = self.marginal_product_capital(self.productivity, labor, self.alpha)
        vmpk = self.value_of_marginal_product(self.output_price, mpk)
        
        if vmpk > rental_rate:
            # invest more in capital
            self.capital *= 1.05  # increase capital by 5%
        elif vmpk < rental_rate:
            # reduce capital
            self.capital *= 0.95  # decrease capital by 5%

        # if vmpk > rental_rate:
        #     self.capital += self.investment_step
        # elif vmpk < rental_rate:
        #     self.capital = max(0, self.capital - self.investment_step)

    def post_vacancies(self, wage):
        # Calculate current labor input
        labor = sum(w.hours_worked for w in self.current_workers)
        labor = max(labor, 1e-6)  # Avoid zero labor input

        vacancies = 0
        mpl = self.marginal_product_labor(self.productivity, labor, self.alpha)
        vmp_l = self.value_of_marginal_product(self.output_price, mpl)
        while vmp_l >= wage:
            vacancies += 1
            labor += 40  # assume each vacancy adds 40 hours of labor
            mpl = self.marginal_product_labor(self.productivity, labor, self.alpha)
            vmp_l = self.value_of_marginal_product(self.output_price, mpl)
        return vacancies

        
    # hiring rule
    # if vmp_l >= wage:
    # hire_more()

    # Parameter	Typical value	            Why
    # A	        500–1000	                Scales output to wage levels
    # α	        0.6–0.8	                    Strong diminishing returns
    # price	    1.0	                        Normalization

    # def marginal_product_labor(self):
    #     num_workers = len(self.current_workers)
    #     return self.productivity / (num_workers + 1)


class Firm_Old(Agent):
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
        self.machine_productivity = 0.5  # productivity factor for machines
        
        self.machine_investment = capital * 0.5 
        self.capital = capital - self.machine_investment
        
        self.worker_diminishing_return = 0.65  # parameter for diminishing returns to labor
        self.machine_diminishing_return = 0.35  # parameter for diminishing returns to capital
        
        self.machine_depreciation_rate = 0.9  # x% depreciation per time step
        self.machine_operation_cost_rate = 0.05  # y% operation cost per time step
        self.machine_reinvestment_rate = 0.1  # reinvest 20% of profits into machines each step
        self.machine_interest_cost_rate = 0.08  # interest cost on machine investment
        self.machine_price = 1500
        

    def step(self):
        # Hire/Fire first
        self.mrp_check()
        
        # production phase with current workers
        self.output = self.production()
        
        # calculate revenue
        revenue = (self.output * self.product_sales_price)
        
        # machine depreciation and operation costs
        operation_cost = self.machine_investment * self.machine_operation_cost_rate
        interest_cost = self.machine_investment * self.machine_interest_cost_rate
        depreciation_cost = self.machine_investment * (1 - self.machine_depreciation_rate)
        total_machine_cost = operation_cost + interest_cost + depreciation_cost
        self.machine_investment *= self.machine_depreciation_rate
        
        # wage distribution
        total_wage_cost = sum([w.wage for w in self.current_workers])
        for w in self.current_workers:
            w.savings += w.wage
        
        # calculate profit
        self.current_profit = revenue - total_wage_cost - self.fixed_cost - total_machine_cost
        # print(f"Firm {self.unique_id} Revenue: {revenue:.2f}, Wage Cost: {total_wage_cost:.2f}, Machine Cost: {total_machine_cost:.2f}, Fixed Cost: {self.fixed_cost:.2f}, Profit: {self.current_profit:.2f}, Machine Investment: {self.machine_investment:.2f}")
        self.capital += self.current_profit
        
        # Reinvest in machines        
        self.check_machine_reinvestment()
        
        # Clear applying workers list for next step
        self.applying_workers = []

    def production(self):
        l = len(self.current_workers)
        k = self.machine_investment
        if len(self.current_workers) == 0 or self.machine_investment == 0:
            return 0
        return self.productivity * (l ** self.worker_diminishing_return) * (k ** self.machine_diminishing_return)

    def MPL(self):
        l = len(self.current_workers)
        k = self.machine_investment
        if l == 0:
            return 0
        return (self.productivity * self.worker_diminishing_return * (l ** (self.worker_diminishing_return - 1)) * (k ** self.machine_diminishing_return))
    
    def MPK(self):
        l = len(self.current_workers)
        k = self.machine_investment
        if k == 0:
            return 0
        return (self.productivity * self.machine_diminishing_return * (l ** self.worker_diminishing_return) * (k ** (self.machine_diminishing_return - 1)))
        
    def MRP(self):
        l_current = len(self.current_workers)
        l_next = l_current + 1
        k = self.machine_investment
        
        # Calculate Total Product with current workers
        if l_current == 0:
            tp_current = 0
        else:
            tp_current = self.productivity * (l_current ** self.worker_diminishing_return) * (k ** self.machine_diminishing_return)
            
        # Calculate Total Product if we add one more
        tp_next = self.productivity * (l_next ** self.worker_diminishing_return) * (k ** self.machine_diminishing_return)
        
        # Change in output * price
        delta_q = tp_next - tp_current
        # print(f"Firm {self.unique_id} MRP Calculation: TP Current: {tp_current:.2f}, TP Next: {tp_next:.2f}, Delta Q: {delta_q:.2f}, MRP: {self.product_sales_price * delta_q:.2f}")
        return self.product_sales_price * delta_q 
    
    def MRPK(self):
        l = len(self.current_workers)
        k_current = self.machine_investment
        k_next = k_current * 1.01
        
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
    
    # def check_machine_reinvestment(self):
    #     user_cost = self.machine_price * (self.machine_interest_cost_rate + self.machine_operation_cost_rate + (1 - self.machine_depreciation_rate))
    #     print(f"Firm {self.unique_id} MRPK: {self.MRPK():.2f}, User Cost: {user_cost:.2f}, Current Profit: {self.current_profit:.2f}")
    #     if self.MRPK() > user_cost and self.current_profit > 0:
    #         investment = self.current_profit * self.machine_reinvestment_rate
    #         self.machine_investment += investment
    #         self.capital -= investment
            
    def check_machine_reinvestment(self):
        # Calculate the Marginal Revenue Product of ONE dollar of capital
        l = len(self.current_workers)
        k = self.machine_investment
        
        if l == 0: return # No workers, no reason to buy machines
        
        # Using the Calculus version for the 'slope' of the MPK
        # MPK = A * beta * L^alpha * K^(beta-1)
        mpk = self.productivity * self.machine_diminishing_return * (l ** self.worker_diminishing_return) * (k ** (self.machine_diminishing_return - 1))
        mrpk = self.product_sales_price * mpk
        
        # User Cost (r) per step: Interest + Depreciation + Ops
        r = self.machine_interest_cost_rate + (1 - self.machine_depreciation_rate) + self.machine_operation_cost_rate
        
        # STOP GROWING if the benefit (mrpk) is less than the cost (r)
        if mrpk > r:
            # Invest only enough to move toward the equilibrium
            # Don't dump all profit at once
            potential_investment = self.current_profit * self.machine_reinvestment_rate
            if potential_investment > 0:
                self.machine_investment += potential_investment
                self.capital -= potential_investment
            
    def mrp_check(self):
        mrp = self.MRP()
        
        if len(self.current_workers) == 0:
            if mrp > (self.model.min_wage * 0.5): # Be a bit more lenient for the first hire
                self.hire_one_worker()
        elif mrp >= self.model.min_wage and self.capital > self.model.min_wage:
            self.hire_one_worker()
        elif mrp < self.model.min_wage and len(self.current_workers) > 0:
            self.fire_worker()
    
    def hire_one_worker(self):
        for w in self.applying_workers:
            if w.employed:
                continue
            if len(self.current_workers) >= self.max_workers:
                # print(f"Firm {self.unique_id} cannot hire more workers; at max capacity.")
                return False
            if w.skill_level >= self.skill_requirement and self.model.min_wage >= w.reservation_wage:
                w.employed = True
                w.wage = self.model.min_wage
                self.current_workers.append(w)
                # print(f"Firm {self.unique_id} hired Worker {w.unique_id} at wage {w.wage}")
                return True
        return False
    
    def fire_worker(self):
        w = random.choice(self.current_workers)
        w.employed = False
        w.wage = 0
        self.current_workers.remove(w)
        # print(f"Firm {self.unique_id} fired Worker {w.unique_id}")


# -------------------
# MODEL
# -------------------
class LaborMarketModel(Model):
    def __init__(self, N_workers=1000, N_firms=10, min_wage=7700, simulator=None, seed=42):
        super().__init__(seed=seed)
        if simulator:
            self.simulator = simulator
            self.simulator.setup(self)
        self.num_workers = N_workers
        self.num_firms = N_firms
        self.MAX_HOURS = 8 * 6 * 4  # (192 hours) Max working hours per month (assumed 8 hours/day * 6 days/week * 4 weeks)
        self.min_wage = min_wage
        self.step_count = 0
        self.running = True # Needed for Mesa to know the model is running
        self.schedule = RandomActivation(self)
        random.seed(seed)
        
        # Create agents
        for i in range(self.num_workers):
            w = Worker(i, self, productivity=1, skill_level=2, 
                       non_labor_income=random.uniform(0, 1000), consumption_weight=random.uniform(0.3, 0.7))
            self.schedule.add(w)
        for i in range(self.num_firms):
            f = Firm(f"F{i}", self, capital=random.uniform(45000, 75000), productivity=random.uniform(0.8, 1.2),
                    skill_requirement=1, output_price=1.0)
            self.schedule.add(f)

        def labor_supply(workers, wage):
            return sum(1 for w in workers if wage > w.reservation_wage)

        def labor_demand(firms, wage):
            demand = 0
            for firm in firms:
                while firm.output_price * firm.marginal_product() > wage:
                    demand += 1
                    firm.employment += 1
            return demand

        def find_market_clearing_wage(workers, firms, wage_grid):
            for wage in wage_grid:
                supply = labor_supply(workers, wage)
                demand = labor_demand(firms, wage)
                if abs(supply - demand) <= 1:
                    return wage

        # Find market clearing wage
        workers = [a for a in self.schedule.agents if isinstance(a, Worker)]
        firms = [a for a in self.schedule.agents if isinstance(a, Firm)]
        wage_grid = np.arange(20, 100, 2)  # hourly wage grid from 20 Baht to 100 Baht
        market_clearing_wage = find_market_clearing_wage(workers, firms, wage_grid)
        print(f"Estimated Market Clearing Wage: {market_clearing_wage}")

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
    