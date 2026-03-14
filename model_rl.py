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
        self.steps = 0

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
            # check if employee is in list of current workers for safety
            if self in self.employer.current_workers:
                self.employer.current_workers.remove(self)
                self.employed = False
                self.employer = None
                self.monthly_wage = 0

    def search_for_jobs(self, firms):
        if self.employed:
            if random.random() > self.job_search_prob:
                return
            else:
                # Consider switching jobs
                # print(f"Worker {self.unique_id} is considering switching jobs.")
                acceptable_firms = []
                for firm in firms:
                    if firm.vacancies > 0:
                        # Consider the utility of working at this firm and compare to current job and switching cost
                        if self.utility_if_work(firm.monthly_wage) - (self.monthly_wage * 0.05) > self.utility_if_work(self.monthly_wage):
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
                if self.utility_if_work(firm.monthly_wage) - (self.monthly_wage * 0.05) > utility_if_not_work:
                    acceptable_firms.append(firm)

        if acceptable_firms:
            chosen_firm = max(
                acceptable_firms,
                key=lambda f: self.utility_if_work(f.monthly_wage)
            )
            chosen_firm.applicants.append(self)

    def job_search_step(self):
        # Quit if the outside option (not working) yields higher utility
        if self.employed and self.utility_if_not_work() > self.utility_if_work(self.monthly_wage):
            if self.employer:
                self.employer.handle_quit(self)
            self.employed = False
            self.employer = None
            self.monthly_wage = 0
            self.daily_wage = 0
            # print(f"Worker {self.unique_id} quit their job because the outside option is better. Utility if not work: {self.utility_if_not_work():.2f}, utility if work: {self.utility_if_work(self.monthly_wage):.2f}")
            return

        all_firms = [a for a in self.model.schedule.agents if isinstance(a, Firm)]
        # Assume workers have limited information and only consider 10% of firms randomly each step or at least 2 firms to avoid zero consideration, this applies to both employed and unemployed workers. This is a simplification to reflect real-world frictions in job search and information.
        # If worker is already employed, the considered firms do not include their current employer

        if self.employed:
            firms_to_consider = random.sample([f for f in all_firms if f != self.employer], max(2, (len(all_firms) - 1) // 10))
        else:
            firms_to_consider = random.sample(all_firms, max(2, len(all_firms) // 10))

        self.search_for_jobs(firms_to_consider)
        
        # self.rl_decision()

    def step(self):
        # Calculate reward based on utility change
        if self.steps % self.model.DECISION_INTERVAL == 0:
            current_utility = self.utility_if_work(self.monthly_wage) if self.employed else self.utility_if_not_work()
            self.reward = current_utility - self.last_utility
            self.last_utility = current_utility
        else:
            self.reward = 0
        
        self.steps += 1

    def rl_stage(self):
        # if self.uid == self.model.rl_worker_id:
        #     self.rl_action = self.model.worker_actions
        #     self.rl_decision()
        pass
    
    def hire_step(self):
        pass
    
    def onboard_workers_step(self):
        pass
    
    def post_vacancies_step(self):
        pass
    
    def adjust_employment_step(self):
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
        self.last_profit = 0
        self.deficit_months = 0
        self.reward = 0
        self.vacancy_duration = 0
        
        self.monthly_wage = 0
        self.fixed_wage_floor = None  # set once from initial VMPL and held constant thereafter
        self.quits_last_month = 0

        # RL
        self.rl_action = 0
        self.profit = 0

    # ---------- RL decision ----------
    def set_initial_wage(self, gamma):
        # Set initial wage based on MPL
        labor = len(self.current_workers)  # number of workers
        mpl = self.marginal_product_labor(self.productivity, labor, self.alpha)
        vmpl = mpl * self.output_price
        self.fixed_wage_floor = max(self.model.min_wage, 0.7 * vmpl)
        # gamma is the fraction of MPL paid to workers (0.7 to 0.9 typical)
        self.monthly_wage = gamma * vmpl
        self.monthly_wage = max(self.monthly_wage, self.wage_floor())
        # Make wage an integer for realism (since we're modeling in THB)
        self.monthly_wage = int(self.monthly_wage)

        # print(f"Firm {self.unique_id} initial wage set to {self.monthly_wage:.2f} based on MPL of {mpl:.2f}")
        self.daily_wage = self.monthly_wage / 20  # assuming 20 working days per month

        # Set worker wages accordingly
        for w in self.current_workers:
            w.monthly_wage = self.monthly_wage
            w.daily_wage = self.daily_wage
            
    def produce(self):
        labor = max(len(self.current_workers), 1e-6)
        
        output = output = self.productivity * (self.capital ** (1 - self.alpha)) * (labor ** self.alpha)
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
             
    def rl_decision(self):
        # 0 = hold
        # 1 = wage +300  (aggressive raise)   — wage-only: restricted to every 12 steps
        # 2 = wage +100  (soft raise)          — wage-only: restricted to every 12 steps
        # 3 = wage -100  (soft cut)            — wage-only: restricted to every 12 steps
        # 4 = wage -300  (aggressive cut)      — wage-only: restricted to every 12 steps
        # 5 = post 1 vacancy                   — allowed every step (mirrors adjust_employment_step)
        # 6 = fire 1 worker                    — allowed every step (mirrors adjust_employment_step)
        wage_action = self.rl_action in (1, 2, 3, 4)
        if wage_action and self.steps % 12 != 0:
            return  # wage can only change annually, matching heuristic firms

        if self.rl_action == 1:
            self.monthly_wage += 300
            for w in self.current_workers:
                w.monthly_wage = self.monthly_wage
        elif self.rl_action == 2:
            self.monthly_wage += 100
            for w in self.current_workers:
                w.monthly_wage = self.monthly_wage
        elif self.rl_action == 3:
            self.monthly_wage = max(self.monthly_wage - 100, self.wage_floor())
            for w in self.current_workers:
                w.monthly_wage = self.monthly_wage
        elif self.rl_action == 4:
            self.monthly_wage = max(self.monthly_wage - 300, self.wage_floor())
            for w in self.current_workers:
                w.monthly_wage = self.monthly_wage
        elif self.rl_action == 5:
            self.vacancies += 1
        elif self.rl_action == 6:
            if self.current_workers:
                fired_worker = random.choice(self.current_workers)
                self.current_workers.remove(fired_worker)
                fired_worker.employed = False
                fired_worker.employer = None
                fired_worker.monthly_wage = 0

    # ---------- Hiring ----------

    def hire_step(self):
        random.shuffle(self.applicants)

        hires = min(len(self.applicants), self.vacancies)

        for i in range(hires):
            worker = self.applicants[i]
            previous_employer = worker.employer
            if previous_employer and previous_employer is not self:
                previous_employer.handle_quit(worker)
            worker.employed = True
            worker.employer = self
            self.pending_workers.append(worker)  # Add to pending to start next step
            self.vacancies -= 1

        self.applicants = []

        # Track persistence of unfilled roles
        # print(f"Firm {self.unique_id} hired {hires} workers. Remaining vacancies: {self.vacancies}")
        if self.vacancies > 0:
            self.vacancy_duration += 1
        else:
            self.vacancy_duration = 0

    def onboard_workers_step(self):
        """Move workers from pending to current (1-step hiring delay)"""
        self.current_workers.extend(self.pending_workers)
        self.pending_workers = []
        # All workers should have the same wages based on the firm's current wage setting
        for w in self.current_workers:
            w.monthly_wage = self.monthly_wage
            
    def adjust_employment_step(self):
        if self.uid == self.model.rl_firm_id:
            return
        
        """Single-stage employment adjustment before worker job search."""
        fired_anyone = self.fire_for_profit()
        if fired_anyone:
            # If we downsized this step, do not open vacancies until next period
            self.vacancies = 0
            self.applicants = []
            self.vacancy_duration = 0
            return
        
        # No firing, so consider expanding by posting vacancies
        self.post_vacancies()

    # ---------- Production ----------

    def step(self):
        self.last_worker_count = len(self.current_workers)
        # print(f"firm id {self.uid} number worker {len(self.current_workers)} profit : {self.profit}")
        output = self.produce()
        self.last_output = output
        revenue = output * self.output_price
        
        wage_cost = sum(w.monthly_wage for w in self.current_workers)
        
        capital_cost = self.capital * self.rental_rate
        self.profit = revenue - wage_cost - capital_cost
        
        if self.profit < 0:
            self.deficit_months += 1
        else:
            self.deficit_months = 0

        if self.deficit_months >= self.model.deficit_exit_months:
            # print(f"Firm {self.unique_id} flagged for exit after {self.deficit_months} deficit months.")
            self.model.queue_firm_exit(self)
        
        if self.steps % 12 == 0:
            self.adjust_capital(len(self.current_workers), self.rental_rate)
            if self.uid != self.model.rl_firm_id:
                self.optimize_wage_annual()
        
        profit_change = self.profit - self.last_profit

        # Smooth, continuous reward: level signal (70%) + trend signal (30%)
        # tanh keeps rewards bounded in (-1, 1) with no discontinuity at profit=0.
        # Scale 20000 ≈ one firm's monthly revenue range; 5000 ≈ meaningful change.
        self.reward = (
            0.7 * float(np.tanh(self.profit       / 20_000)) +
            0.3 * float(np.tanh(profit_change     /  5_000))
        )

        # Update state
        self.last_profit = self.profit

        self.steps += 1
        
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
    
    def value_of_marginal_product(self, price, mp):
        return price * mp

    def wage_floor(self):
        """Fixed per-firm floor set once from initial VMPL during wage initialization."""
        if self.fixed_wage_floor is None:
            return self.model.min_wage
        return self.fixed_wage_floor
    
    def compute_vacancy_rate(self):
        positions = len(self.current_workers) + self.vacancies
        return self.vacancies / positions if positions > 0 else 0

    def compute_quit_rate(self):
        baseline_workers = self.last_worker_count + self.quits_last_month
        baseline_workers = max(baseline_workers, 1)
        return self.quits_last_month / baseline_workers
    
    def adjust_capital(self, labor, rental_rate):
        # Run every 12 steps to adjust capital based on the value of the marginal product of capital
        mpk = self.marginal_product_capital(self.productivity, labor, self.alpha)
        vmpk = self.value_of_marginal_product(self.output_price, mpk)
        # print(f"Firm {self.unique_id} VMPK: {vmpk:.2f}, Rental Rate: {rental_rate:.2f}")
        
        if vmpk > rental_rate:
            # invest more in capital
            self.capital *= 1.05  # increase capital by 5%
        elif vmpk < rental_rate:
            # reduce capital
            self.capital *= 0.95  # decrease capital by 5%

    def post_vacancies(self):
        self.applicants = []
        self.vacancies = 0

        current_labor = len(self.current_workers)
        mpl = self.marginal_product_labor(
            self.productivity,
            current_labor + 1,  # Consider exactly one additional worker
            self.alpha
        )
        vmp = self.output_price * mpl
        # print(f"Firm {self.unique_id} vacancy check: labor={current_labor}, VMP={vmp:.2f}, wage={self.monthly_wage:.2f}")

        if vmp >= self.monthly_wage:
            self.vacancies = 1  # limit expansion to one vacancy per step

        # print(f"Firm {self.unique_id} posted {self.vacancies} vacancies.")

    def hire(self):
        random.shuffle(self.applicants)

        hires = min(len(self.applicants), self.vacancies)

        for i in range(hires):
            worker = self.applicants[i]
            previous_employer = worker.employer
            if previous_employer and previous_employer is not self:
                previous_employer.handle_quit(worker)
            worker.employed = True
            worker.employer = self
            self.pending_workers.append(worker)  # Add to pending to start next step
            self.vacancies -= 1

        self.applicants = []

        # Track persistence of unfilled roles
        # print(f"Firm {self.unique_id} hired {hires} workers. Remaining vacancies: {self.vacancies}")
        if self.vacancies > 0:
            self.vacancy_duration += 1
        else:
            self.vacancy_duration = 0

    def handle_quit(self, worker):
        if worker in self.current_workers:
            self.current_workers.remove(worker)
            # print(f"Quit: Worker {worker.unique_id} left Firm {self.unique_id} at wage {worker.monthly_wage:.2f}")
        worker.employer = None
        worker.employed = False
        worker.monthly_wage = 0
        worker.daily_wage = 0
        self.quits_last_month += 1

    def fire_one_worker(self):
        """Remove one worker and update their employment state."""
        if not self.current_workers:
            return False
        worker = self.current_workers.pop()
        worker.employed = False
        worker.employer = None
        worker.monthly_wage = 0
        worker.daily_wage = 0
        # print(f"Fired: Worker {worker.unique_id} from Firm {self.unique_id}")
        return True

    def exit_and_release_workers(self):
        """Release all workers when the firm exits the market."""
        for w in list(self.current_workers):
            w.employed = False
            w.employer = None
            w.monthly_wage = 0
            w.daily_wage = 0
        self.current_workers = []
        self.pending_workers = []
        self.vacancies = 0
        self.applicants = []
    
    def fire_for_profit(self):
        """Allow firing each step when it strictly raises profit."""
        baseline_profit = self.compute_profit()
        fired_anyone = False

        while len(self.current_workers) > 0:
            profit_if_fire = self.compute_profit(labor_override=len(self.current_workers) - 1)
            if profit_if_fire > baseline_profit:
                self.fire_one_worker()
                baseline_profit = profit_if_fire
                fired_anyone = True
            else:
                break

        if fired_anyone:
            self.profit = baseline_profit
            self.daily_wage = self.monthly_wage / 20
            for w in self.current_workers:
                w.monthly_wage = self.monthly_wage
                w.daily_wage = self.daily_wage

        return fired_anyone

    def optimize_wage_annual(self):
        """Annual wage adjustment based on profit and labor market conditions."""
        # Wage moves only if it improves profit given current headcount
        self.adjust_wage(current_profit=self.compute_profit())
        self.profit = self.compute_profit()

    def adjust_wage(self, base_delta=0.02, target_quit=0.02, quit_gamma=0.05, current_profit=None):
        """Three-stage wage logic: (1) vacancy/quit pressure, (2) profit projection, (3) VMPL gap nudges."""
        vacancy_rate = self.compute_vacancy_rate()
        quit_rate = self.compute_quit_rate()
        # print(f"Firm {self.unique_id} adjusting wage: Vacancy Rate: {vacancy_rate:.2f}, Quit Rate: {quit_rate:.2f}, Current Wage: {self.monthly_wage:.2f}, Current Profit: {current_profit:.2f}")

        current_wage = self.monthly_wage
        current_profit = self.profit if current_profit is None else current_profit
        new_wage = current_wage

        # 1) Vacancy/quit pressure first: if we cannot hire (open roles persist), lift wages immediately
        cannot_hire = self.vacancies > 0 and self.vacancy_duration > 0
        if cannot_hire:
            # Scale more aggressively with vacancy pressure
            delta = base_delta * (1 + vacancy_rate + quit_rate)
            new_wage = max(current_wage * (1 + delta * (1 + self.vacancy_duration)), self.wage_floor())
            # print(f"Firm {self.unique_id} cannot hire, increasing wage aggressively to {new_wage:.2f} (delta: {delta:.4f}, vacancy duration: {self.vacancy_duration})")
        else:
            # 2) Profit projection branch when hiring is not blocked; use a calmer scaler
            delta = base_delta
            candidate_up = max(current_wage * (1 + delta), self.wage_floor())
            candidate_down = max(current_wage * (1 - delta), self.wage_floor())
            profit_up = self.compute_profit(candidate_up)
            profit_down = self.compute_profit(candidate_down)

            if profit_up > current_profit and profit_up >= profit_down:
                new_wage = candidate_up
                # print(f"Firm {self.unique_id} profit projection suggests increasing wage to {new_wage:.2f} (profit up: {profit_up:.2f})")
            elif profit_down > current_profit:
                new_wage = candidate_down
                # print(f"Firm {self.unique_id} profit projection suggests decreasing wage to {new_wage:.2f} (profit down: {profit_down:.2f})")

            quit_adjustment = quit_gamma * (quit_rate - target_quit)
            new_wage *= (1 + quit_adjustment)
            # print(f"Firm {self.unique_id} adjusting wage by quit rate: new wage {new_wage:.2f} (quit adjustment: {quit_adjustment:.4f})")

        # 3) VMPL wage-gap correction: probabilistic nudge upward when wage < VMPL
        labor = len(self.current_workers)
        if labor > 0:
            mpl = self.marginal_product_labor(self.productivity, labor, self.alpha)
            vmpl = mpl * self.output_price
            wage_gap = max(vmpl - new_wage, 0)
            if wage_gap > 0:
                # Probability rises with the gap but is capped
                gap_ratio = wage_gap / max(vmpl, 1e-6)
                bump_prob = min(0.9, gap_ratio)
                if random.random() < bump_prob:
                    new_wage *= (1 + delta * (1 + gap_ratio))
                    # print(f"Firm {self.unique_id} adjusting wage by VMPL gap: new wage {new_wage:.2f} (gap ratio: {gap_ratio:.4f})")

        # Apply bounds and propagate to workers
        self.monthly_wage = int(max(new_wage, self.wage_floor()))
        self.daily_wage = self.monthly_wage / 20
        for w in self.current_workers:
            w.monthly_wage = self.monthly_wage
            w.daily_wage = self.daily_wage
    
    def rl_stage(self):
        if self.uid == self.model.rl_firm_id:
            self.rl_action = self.model.rl_action
            self.rl_decision()
        else:
            pass

    def compute_profit(self, wage=None, labor_override=None):
            """Estimate profit for hypothetical wage and/or labor without mutating state."""
            labor = len(self.current_workers) if labor_override is None else labor_override
            floor = self.wage_floor()
            wage_to_use = self.monthly_wage if wage is None else wage
            wage_to_use = max(wage_to_use, floor)
            labor = max(labor, 1e-6)
            output = self.productivity * (self.capital ** (1 - self.alpha)) * (labor ** self.alpha)
            revenue = output * self.output_price
            total_wage_cost = wage_to_use * labor
            capital_cost = self.capital * self.rental_rate
            return revenue - total_wage_cost - capital_cost
        
    def job_search_step(self):
        pass


# =====================================================
# LABOR MARKET MODEL
# =====================================================

class LaborMarketModel(Model):

    def __init__(self, N_workers=100, N_firms=10):

        super().__init__()

        self.MAX_HOURS = 192
        self.DECISION_INTERVAL = 3
        self.rl_firm_id = "F0"
        self.rl_worker_id = 0
        self.min_wage = 7700
        self.deficit_exit_months = 24
        self.pending_firm_exits = []
        
        self.schedule = StagedActivation(
            self,
            stage_list=["rl_stage",
                        "onboard_workers_step",
                        "adjust_employment_step",
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
            firm.set_initial_wage(gamma=0.8)
    
    def queue_firm_exit(self, firm):
        if firm not in self.pending_firm_exits:
            self.pending_firm_exits.append(firm)

    def step(self):
        self.schedule.step()