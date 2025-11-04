from mesa import Model, Agent
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import random
import numpy as np

# -------------------
# AGENTS
# -------------------
class Worker(Agent):
    def __init__(self, unique_id, model, productivity, skill_level,):
        super().__init__(model)
        self.unique_id = unique_id
        self.productivity = productivity
        self.employed = False
        self.wage = 0
        self.saving = 0
        self.monthly_search = 3
        self.skill_level = skill_level
        self.loyalty = 0  # number of consecutive steps employed

    def step(self):
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
    def __init__(self, unique_id, model, capital, productivity, skill_requirement, product_sales_price):
        super().__init__(model)
        self.unique_id = unique_id
        self.capital = capital
        self.productivity = productivity
        self.skill_requirement = max(1, skill_requirement)
        self.product_sales_price = product_sales_price
        self.applying_workers = []
        self.current_workers = []
        self.previous_profit = 0
        self.current_profit = 0
        self.vacancies = 0
        self.threshold_profit = 2000 # minimum profit to consider hiring
        self.productivity_effectiveness = 1.0 # change to adjust hiring based on productivity (number ^ 0.8)
        self.initialize_hires(num_hires=5)
    
    # initialize hired workers with random workers stat
    def initialize_hires(self, num_hires):
        available_workers = [w for w in self.model.schedule.agents if isinstance(w, Worker) and not w.employed]
        random.shuffle(available_workers)
        hires = available_workers[:num_hires]
        for w in hires:
            w.employed = True
            w.wage = self.model.min_wage
            self.current_workers.append(w)
        # print(f"Firm {self.unique_id} initial hires: {[w.unique_id for w in hires]}")
    

    def step(self):
        # payout to current workers
        for w in self.current_workers:
            w.saving += w.wage
        
        # calculate profit
        total_wage_cost = sum([w.wage for w in self.current_workers])
        total_worker_productivity = sum([w.productivity for w in self.current_workers])
        self.current_profit = (total_worker_productivity * self.product_sales_price) - total_wage_cost
        
        print(f"profit for Firm {self.unique_id}: {self.current_profit} with {len(self.current_workers)} workers")
        # fire least worthy worker if profit decreased
        if self.previous_profit < 0 and len(self.current_workers) > 0:
            total_wage_cost = sum([w.wage for w in self.current_workers])
            total_worker_productivity = sum([w.productivity for w in self.current_workers])
            # test if kick out worker will profit increase or decrease
            worker_worth = {}
            for w in self.current_workers:
                projected_wage_cost = total_wage_cost - w.wage
                projected_total_worker_productivity = total_worker_productivity - w.productivity
                projected_profit = (projected_total_worker_productivity * self.product_sales_price) - projected_wage_cost
                profit_change = projected_profit - self.current_profit
                worker_worth[w] = profit_change
            # kick out 1 workers that give least profit increase (or most profit decrease)
            # print(f"Worker worth for Firm {self.unique_id}: " + ", ".join([f"W{w.unique_id}:{worth:.2f}" for w, worth in worker_worth.items()]))
            worker_to_kick = max(worker_worth, key=worker_worth.get)
            self.current_workers.remove(worker_to_kick)
            worker_to_kick.employed = False
            worker_to_kick.wage = 0
        
        
        
        # hire new workers from applicants

        x = 1
        while self.current_profit > x * self.threshold_profit:
            # refresh applicants to skip any who may have been hired by other firms
            self.applying_workers = [w for w in self.applying_workers if not w.employed]
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
                    if self.capital >= self.model.min_wage:
                        best_applicant.employed = True
                        best_applicant.wage = self.model.min_wage
                        best_applicant.saving += self.model.min_wage
                        self.capital -= self.model.min_wage
                        self.current_workers.append(best_applicant)
                        try:
                            self.applying_workers.remove(best_applicant)
                        except ValueError:
                            pass
                        self.loyalty = 0  # reset loyalty counter on new hire
                    else:
                        # cannot afford more hires
                        break
            else:
                break
            x += 1
        
        # bonus to worker if stay for 12 steps
        for w in self.current_workers:
            w.loyalty += 1
            if w.loyalty % 12 == 0:
                bonus = 50
                if self.capital >= bonus:
                    w.saving += bonus
                    self.capital -= bonus

        # update profit and capital
        self.previous_profit = self.current_profit
        self.capital += self.current_profit
        
        
        
        # bonus , payout , new worker , fired , calculate profit
        # payout , calculate profit , fire worker , new worker , bonus
        
        
        
        
        
        # Firms try to hire workers based on productivity and minimum wage
        # available_workers = [w for w in self.model.schedule.agents if isinstance(w, Worker) and not w.employed]
        # random.shuffle(available_workers)

        # desired_workers = int(self.productivity * 10)  # crude "labor demand"
        # min_wage = self.model.min_wage

        # hired = []
        # for worker in available_workers[:desired_workers]:
        #     if self.capital >= min_wage:
        #         # hire worker
        #         worker.employed = True
        #         worker.wage = min_wage
        #         worker.saving = min_wage
        #         self.capital -= min_wage
        #         hired.append(worker)
        #     else:
        #         break
        
        # print(f"Firm {self.unique_id} , hired list: {[worker.unique_id for worker in hired]} ")
        # Firm profit = productivity * workers - wages paid
        # self.profit = (self.productivity * len(hired) * 20) - len(hired) * min_wage
        # if self.profit > 0:
        #     print(f"Firm {self.unique_id} hired {len(hired)} workers at wage {min_wage} each. with profit {self.profit}")
        


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
            w = Worker(i, self, productivity=random.uniform(0.5, 1.5), skill_level=random.uniform(1.0, 3.0))
            self.schedule.add(w)

        for i in range(self.num_firms):
            f = Firm(f"F{i}", self, capital=random.uniform(5000, 10000), productivity=random.uniform(0.75, 2.0), skill_requirement=random.uniform(0.5,2.0), product_sales_price=self.min_wage*random.uniform(1.1,1.75))
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
        profits = [f.previous_profit for f in self.schedule.agents if isinstance(f, Firm)]
        return np.mean(profits) if profits else 0