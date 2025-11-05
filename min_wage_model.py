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
    def __init__(self, unique_id, model, productivity, skill_level, savings, expenses):
        super().__init__(model)
        self.unique_id = unique_id
        self.productivity = productivity
        self.employed = False
        # TODO: Implement reservation wage
        self.reservation_wage = expenses + 100
        self.wage = 0
        self.welfare = self.wage - self.reservation_wage
        self.savings = savings
        self.monthly_expenses = expenses
        self.monthly_search = 3
        self.skill_level = skill_level
        self.loyalty = 0  # number of consecutive steps employed at the same firm

    def step(self):
        # Pay expenses
        self.savings -= self.monthly_expenses
        # TODO: What to do next if worker runs out of savings?
        # Apply for jobs if unemployed
        # Also worker shouldnt spend more than their wage no?
        # if wage is higher than expenses then they can save more and try to remove debt

        if self.employed:
            return  # already employed, nothing to do
        else :
            all_available_firms = [a for a in self.model.schedule.agents if isinstance(a, Firm)]
            # shuffle and pick firms to apply
            random.shuffle(all_available_firms)
            firms_to_apply = all_available_firms[:self.monthly_search]
            for f in firms_to_apply:
                if self.skill_level >= f.skill_requirement:
                    f.applying_workers.append(self)
            
                    

class Firm(Agent):
    def __init__(self, unique_id, model, capital, productivity, skill_requirement, product_sales_price, fixed_cost):
        super().__init__(model)
        self.unique_id = unique_id
        self.capital = capital
        self.productivity = productivity

        self.work_days_per_step = 22
        # TODO: Implement hiring and vacancy costs
        self.hiring_cost_per_worker = 2000 # cost to hire one worker
        self.vacancy_cost_per_step = 200 # cost per open vacancy per step
        self.hiring_margin_threshold = 0.1  # hire if MRP exceeds marginal cost by this fraction TODO: Maybe make this a model parameter to be tuned later

        # TODO: Implement wage setting mechanism
        # Example A: offered_wage = self.model.min_wage * (1 + 0.1 * self.productivity) not quite understand
        # Example B: (Derive wage from a market wage concept)
        #   market_wage = np.mean([firm.last_offered_wage for firm in firms])
        #   offered_wage = max(min_wage, market_wage * (1 + 0.1 * self.profit_margin))
        self.fixed_cost = fixed_cost
        self.skill_requirement = max(1, skill_requirement)
        # TODO: For Nino: Adjust product sales price based on market conditions
        # Make PRICE decline slightly as total output rises (captures price competition):
        # PRICE = base_price * (1 / (1 + alpha * total_output))
        # alpha small (0.0001–0.01) controls sensitivity.
        # Read up more on this.

        self.base_output_per_worker_per_day = 10  # Example value
        self.product_sales_price = product_sales_price
        self.applying_workers = []
        self.current_workers = []
        self.max_workers = 10 # maximum number of workers firm can employ TODO: Base this on capital and productivity later
        self.previous_profit = 0 # TODO: Might be removed later if still unused
        self.current_profit = 0 
        self.vacancies = 0
        self.threshold_profit = 2000 # minimum profit to consider hiring TODO: Make this a model parameter later and tune based on capital and productivity
        self.productivity_effectiveness = 1.0 # change to adjust hiring based on productivity (number ^ 0.8)
        self.initialize_hires(num_hires=5)

    # initialize hired workers with random workers stats
    def initialize_hires(self, num_hires):
        available_workers = [w for w in self.model.schedule.agents if isinstance(w, Worker) and not w.employed]
        random.shuffle(available_workers)
        hires = available_workers[:num_hires]
        for w in hires:
            w.employed = True
            w.wage = self.model.min_wage * self.work_days_per_step  # pay for a month
            self.current_workers.append(w)
        # print(f"Firm {self.unique_id} initial hires: {[w.unique_id for w in hires]}")
    

    def step(self):
         # calculate total wage cost and average worker productivity
        total_wage_cost = sum([w.wage for w in self.current_workers])
        total_worker_productivity = sum([w.productivity for w in self.current_workers])
        avg_worker_productivity = total_worker_productivity / len(self.current_workers) if self.current_workers else 0

        # calculate revenue and profit
        ### Output per worker per day = firm_productivity × worker_productivity × base_output_per_worker_per_day
        ### Total output = that × work_days × num_workers
        ### Revenue = total_output × price_per_product
        revenue = ( self.productivity * avg_worker_productivity *
                   self.work_days_per_step * self.base_output_per_worker_per_day * len(self.current_workers) *
                   self.product_sales_price)
        self.current_profit = revenue - total_wage_cost - self.fixed_cost - (self.vacancies * self.vacancy_cost_per_step)


        # update profit and capital
        self.previous_profit = self.current_profit
        self.capital += self.current_profit
        
        # payout to current workers
        for w in self.current_workers:
            w.savings += w.wage
            # Captial has already been reduced by wage cost in profit calculation
            
            # TODO: Also should wages be paid in the worker agent step instead? (for easier worker expense calculation too maybe?)
            # We can but have to make sure that the worker get paid before Firm steps or else the firm will fired them and they get no wage
            
       
        # TODO: Prevent runaway growth by possibly calculating market saturation for phase 2!

        # Calculate MRP to decide on hiring
        labor_diminishing_factor = 1 / (1 + 0.01 * (len(self.current_workers) + 1)) # Example diminishing factor TODO: Tune this parameter (Maybe make 0.01 an alpha parameter of the model later)
        expected_output_from_one_more_worker = self.base_output_per_worker_per_day * self.work_days_per_step *\
            avg_worker_productivity * labor_diminishing_factor * self.productivity
        mrp = expected_output_from_one_more_worker * self.product_sales_price
        marginal_cost_of_hiring = self.model.min_wage * self.work_days_per_step + self.hiring_cost_per_worker # wage + hiring cost
        print(f"Firm {self.unique_id} MRP: {mrp:.2f} vs Marginal Cost: {marginal_cost_of_hiring:.2f}")

        # Forecast hiring 1..k workers (greedy incremental)
        k = 0
        expected_capital = self.capital
        while True:
            # Recalculate MRP for each iteration
            labor_diminishing_factor = 1 / (1 + 0.01 * (len(self.current_workers) + 1 + k)) # Example diminishing factor TODO: Tune this parameter
            expected_output_from_one_more_worker = self.base_output_per_worker_per_day * self.work_days_per_step *\
            avg_worker_productivity * labor_diminishing_factor * self.productivity
            mrp = expected_output_from_one_more_worker * self.product_sales_price
            # check affordability
            if expected_capital < marginal_cost_of_hiring:
                break
            # estimate incremental profit for 1 more worker
            incr_revenue = mrp
            incr_profit = incr_revenue / marginal_cost_of_hiring - 1  # profit margin
            if incr_profit > self.hiring_margin_threshold and (len(self.current_workers) + k) < self.max_workers: # only hire if profit margin exceeds threshold
                k += 1
                expected_capital += incr_profit  # optimistic reinvest
            else:
                break

        # If forecast suggests hiring k>0, create k vacancies (or hire immediately if matching)
        if k > 0:
            self.vacancies = k
        else:
            self.vacancies = 0

        # Hire or fire based on profit
        # 1. If profit > threshold and vacancies > 0, hire from applicants
        # 2. If profit < 0, consider firing least worthy worker

        # fire least worthy worker if profit decreased
        if self.current_profit < 0 and len(self.current_workers) > 0:
            # test if kick out worker will profit increase or decrease
            worker_worth = {}
            for w in self.current_workers:
                projected_wage_cost = total_wage_cost - w.wage
                projected_total_worker_productivity = total_worker_productivity - w.productivity
                projected_revenue = ( self.productivity * (projected_total_worker_productivity / (len(self.current_workers) - 1) if len(self.current_workers) > 1 else 0) *
                                   self.work_days_per_step * self.base_output_per_worker_per_day * (len(self.current_workers) - 1) *
                                   self.product_sales_price)
                projected_profit = projected_revenue - projected_wage_cost - self.fixed_cost - ((self.vacancies + 1) * self.vacancy_cost_per_step)
                profit_change = projected_profit - self.current_profit
                worker_worth[w] = profit_change
            # kick out 1 worker that gives least profit increase (or most profit decrease)
            worker_to_kick = max(worker_worth, key=worker_worth.get)
            print(f"Firm {self.unique_id} firing Worker {worker_to_kick.unique_id} worth {worker_worth[worker_to_kick]:.2f}")
            self.current_workers.remove(worker_to_kick)
            worker_to_kick.employed = False
            worker_to_kick.wage = 0
            self.vacancies += 1  # create vacancy due to firing
        
        print(f"profit for Firm {self.unique_id}: {self.current_profit} with {len(self.current_workers)} workers")
        
        # bonus to worker if stay for 12 steps (1 year)
        for w in self.current_workers:
            w.loyalty += 1
            if w.loyalty % 12 == 0:
                bonus = 50
                if self.capital >= bonus:
                    w.savings += bonus
                    self.capital -= bonus
        
        # hire new workers from applicants if profit high enough
        if self.current_profit > self.threshold_profit and self.vacancies > 0:
            # refresh applicants to skip any who may have been hired by other firms
            self.applying_workers = [w for w in self.applying_workers if not w.employed]
            while self.vacancies > 0:
                if self.applying_workers:
                    # hire the most productive applicant who is still unemployed
                    best_applicant = max(self.applying_workers, key=lambda w: w.productivity)
                    # double-check applicant still unemployed (race condition with other firms)
                    if best_applicant.employed:
                        # remove and continue
                        try:
                            self.applying_workers.remove(best_applicant)
                        except ValueError:
                            pass
                    else:
                        if self.capital >= marginal_cost_of_hiring:
                            best_applicant.employed = True
                            best_applicant.wage = self.model.min_wage * self.work_days_per_step  # pay for a month
                            self.current_workers.append(best_applicant)

                            try:
                                self.applying_workers.remove(best_applicant)
                            except ValueError:
                                pass

                            # deduct hiring cost
                            self.capital -= self.hiring_cost_per_worker
                            self.vacancies -= 1
                            # reset loyalty counter on new hire
                            best_applicant.loyalty = 0
                        else:
                            # cannot afford more hires
                            break
                else:
                    # no more applicants
                    break

        # clear applicants for next step
        self.applying_workers = []
        
        


# -------------------
# MODEL
# -------------------
class LaborMarketModel(Model):
    def __init__(self, N_workers=100, N_firms=10, min_wage=350, simulator=None):
        self.simulator = simulator
        if self.simulator:
            self.simulator.setup(self)
        self.num_workers = N_workers
        self.num_firms = N_firms
        self.min_wage = min_wage
        self.step_count = 0
        self.random = random.Random()
        self.running = True # Needed for Mesa to know the model is running
        self.schedule = RandomActivation(self)
        
        # Create agents
        for i in range(self.num_workers):
            w = Worker(i, self, productivity=random.uniform(0.5, 1.5), skill_level=random.uniform(1.0, 3.0), 
                       savings=random.randint(10000, 30000), expenses=random.randint(4500, 9000)) # monthly expenses
            self.schedule.add(w)
        # TODO: Skill requirement should be related to worker skill level distribution
        for i in range(self.num_firms):
            f = Firm(f"F{i}", self, capital=random.uniform(250000, 750000), productivity=random.uniform(0.75, 2.0), 
                     skill_requirement=random.uniform(0.5,2.0), product_sales_price=self.min_wage*random.uniform(1.1,1.75), 
                     fixed_cost=random.uniform(500, 1000))
                    # TODO: Base fixed_cost on capital and productivity and/or unit cost later
            self.schedule.add(f)
            
        # Data Collector
        model_reporters={
            "EmploymentRate": self.compute_employment_rate,
            "AverageWage": self.compute_avg_wage,
            "AverageProfit": self.compute_avg_profit,
            "FirmSize": self.get_firm_size,
            "AvgFirmCapital": self.get_avg_firm_capital,
        }
        
        self.datacollector = mesa.DataCollector(model_reporters)
        self.datacollector.collect(self)
    
    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)
        self.step_count += 1
        
    def steps(self):
        return self.step_count

    def compute_employment_rate(self):
        workers = [a for a in self.schedule.agents if isinstance(a, Worker)]
        employed = [w for w in workers if w.employed]
        print(f"Employed: {len(employed)} / {len(workers)}")
        return len(employed) / len(workers)

    def compute_avg_wage(self):
        wages = [w.wage for w in self.schedule.agents if isinstance(w, Worker) and w.employed]
        return np.mean(wages) if wages else 0

    def compute_avg_profit(self):
        profits = [f.previous_profit for f in self.schedule.agents if isinstance(f, Firm)]
        return np.mean(profits) if profits else 0

    def get_firm_size(self):
        # average number of workers per firm
        firm_sizes = [len(f.current_workers) for f in self.schedule.agents if isinstance(f, Firm)]
        return np.mean(firm_sizes) if firm_sizes else 0
    
    def get_avg_firm_capital(self):
        firm_capitals = [f.capital for f in self.schedule.agents if isinstance(f, Firm)]
        return np.mean(firm_capitals) if firm_capitals else 0