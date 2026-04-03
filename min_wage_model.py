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
    def __init__(self, unique_id, model, hours_worked, non_labor_income, consumption_weight):
        super().__init__(model)
        self.unique_id = unique_id
        self.employed = False
        self.employer = None

        self.non_labor_income = non_labor_income # monthly non-labor income V
        self.alpha = consumption_weight  # weight on share of income spent on consumption
        self.hours_worked = hours_worked  # hours worked in the current month (currently fixed at 160)
        self.monthly_wage = 0  # monthly wage, set when employed
        self.daily_wage = 0  # daily wage, derived from monthly wage

        self.ON_THE_JOB_SEARCH_PROB = 0.1  # probability of searching for a new job while employed (currently set to 10% per month, can be tuned based on literature estimates of job-to-job transition rates)

    def calculate_leisure(self):
        return self.model.MAX_HOURS - self.hours_worked
    
    def calculate_switching_cost(self):
        # Switching cost is a percentage of monthly wage, curently set at 5% but can be tuned as needed (2-10% of monthly wage). This represents the cost (in time, effort, risk) of switching jobs.
        return self.monthly_wage * 0.05  # 5% of monthly wage as switching cost, can be tuned as needed

    def cobb_douglas_utility(self, consumption, leisure):
        # Avoid zero values (important for numerical stability)
        consumption = max(consumption, 1e-6)
        leisure = max(leisure, 1e-6) 
        return (consumption ** self.alpha) * (leisure ** (1 - self.alpha))
    
    def search_for_jobs(self, firms):
        if self.employed:
            if random.random() > self.ON_THE_JOB_SEARCH_PROB:
                return
            else:
                # Consider switching jobs
                print(f"Worker {self.unique_id} is considering switching jobs.")
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
                    print(f"Worker {self.unique_id} applied to Firm {chosen_firm.unique_id} for a potential switch from Firm {self.employer.unique_id}.")
                    print(f"Worker {self.unique_id} current wage: {self.monthly_wage}, potential new wage: {chosen_firm.monthly_wage}, switching cost: {self.calculate_switching_cost():.2f}")
                    print(f"Worker {self.unique_id} utility if stay: {self.utility_if_work(self.monthly_wage):.2f}, utility if switch: {self.utility_if_work(chosen_firm.monthly_wage):.2f}")
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

    def utility_if_work(self, wage):
        consumption = wage * self.hours_worked + self.non_labor_income
        leisure = self.model.MAX_HOURS - self.hours_worked
        return self.cobb_douglas_utility(consumption, leisure)

    def utility_if_not_work(self):
        consumption = self.non_labor_income
        leisure = self.model.MAX_HOURS
        return self.cobb_douglas_utility(consumption, leisure)


    # def choose_hours(self, wage):
    #     # Typical labor econ values:
    #     # beta ≈ 0.6–0.8 → consumption-focused
    #     # 1−beta reflects leisure preference

    #     possible_hours = [0, 40, 80, 120, 160]  # Possible working hours per month
    #     best_hours = 0
    #     best_utility = -float('inf')

    #     for h in possible_hours:
    #         consumption = (wage * h) + self.non_labor_income
    #         leisure = self.model.MAX_HOURS - h
    #         utility = self.cobb_douglas_utility(consumption, leisure)

    #         if utility > best_utility:
    #             best_utility = utility
    #             best_hours = h

    #     return best_hours

    def job_search_step(self):
        # Quit if the outside option (not working) yields higher utility
        if self.employed and self.utility_if_not_work() > self.utility_if_work(self.monthly_wage):
            if self.employer:
                self.employer.handle_quit(self)
            self.employed = False
            self.employer = None
            self.monthly_wage = 0
            self.daily_wage = 0
            print(f"Worker {self.unique_id} quit their job because the outside option is better. Utility if not work: {self.utility_if_not_work():.2f}, utility if work: {self.utility_if_work(self.monthly_wage):.2f}")
            return

        all_firms = [a for a in self.model.schedule.agents if isinstance(a, Firm)]
        # Assume workers have limited information and only consider 10% of firms randomly each step or at least 2 firms to avoid zero consideration, this applies to both employed and unemployed workers. This is a simplification to reflect real-world frictions in job search and information.
        # If worker is already employed, the considered firms do not include their current employer

        if self.employed:
            firms_to_consider = random.sample([f for f in all_firms if f != self.employer], max(2, (len(all_firms) - 1) // 10))
        else:
            firms_to_consider = random.sample(all_firms, max(2, len(all_firms) // 10))

        self.search_for_jobs(firms_to_consider)

    def adjust_employment_step(self):
        pass

    def hire_step(self):
        pass

    def onboard_workers_step(self):
        pass
    # def step(self):

    #     if self.employed:
    #         return 
    #     else :
    #         all_available_firms = [a for a in self.model.schedule.agents if isinstance(a, Firm)]
            
    #         for f in all_available_firms:
    #                 f.applicants.append(self)
            
class Firm(Agent):
    def __init__(self, unique_id, model, capital, rental_rate, productivity, output_price):
        super().__init__(model)
        self.unique_id = unique_id

        self.capital = capital  # initial capital
        self.rental_rate = rental_rate  # cost of capital rental (THB per unit of capital per month)
        self.base_productivity = 60  # TODO: Tune this parameter to scale output to realistic wage levels (currently set to 60, can be adjusted based on calibration)
        self.productivity_multiplier = productivity
        self.output_price = output_price  # Fixed market price for the firm's product (set to 50 THB per unit for now, can be adjusted based on calibration)
        self.monthly_wage = None # set based on initial MPL and updated over time, this is the wage paid to workers
        self.daily_wage = None # derived from monthly wage, this is the wage paid to workers per day (assuming 20 working days per month)
        self.fixed_wage_floor = None  # set once from initial VMPL and held constant thereafter
        self.productivity = self.base_productivity * self.productivity_multiplier
        self.alpha = 0.65  # labor share
        self.profit = 0  # track latest profit for visualization
        self.last_output = 0  # track latest physical output for aggregation

        # Wage and turnover tracking
        self.prev_profit = 0
        self.quits_last_month = 0
        self.last_worker_count = 0
        self.vacancy_duration = 0

        self.vacancies = 0
        self.applicants = []
        self.current_workers = []
        self.pending_workers = []  # Workers hired but waiting 1 step before starting work
        self.deficit_months = 0  # consecutive months of negative profit

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

        print(f"Firm {self.unique_id} initial wage set to {self.monthly_wage:.2f} based on MPL of {mpl:.2f}")
        self.daily_wage = self.monthly_wage / 20  # assuming 20 working days per month

        # Set worker wages accordingly
        for w in self.current_workers:
            w.monthly_wage = self.monthly_wage
            w.daily_wage = self.daily_wage

    def produce(self):
        labor = len(self.current_workers)  # number of workers
        labor = max(labor, 1e-6)  # Avoid zero labor input

        # Cobb-Douglas production function: Q = A * K^(1-alpha) * L^alpha 
        output = self.productivity * (self.capital ** (1 - self.alpha)) * (labor ** self.alpha)
        return output
    
    def marginal_product_labor(self, A, labor, alpha):
        # Marginal Product of Labor: MPL = dQ/dL = A * alpha * K^beta * L^(alpha-1)
        labor = max(labor, 1e-6)  # Avoid zero labor input
        return A * alpha * (self.capital ** (1 - alpha)) * (labor ** (alpha - 1))
        
    def marginal_product_capital(self, A, labor, alpha):
        # Marginal Product of Capital: MPK = dQ/dK = A * (1-alpha) * K^(-alpha) * L^alpha
        self.capital = max(self.capital, 1e-6)  # Avoid zero capital input
        return A * (1 - alpha) * (self.capital ** (-alpha)) * (labor ** alpha)
    
    def value_of_marginal_product(self, price, mp):
        return price * mp

    def wage_floor(self):
        """Fixed per-firm floor set once from initial VMPL during wage initialization."""
        if self.fixed_wage_floor is None:
            return self.model.min_wage
        return self.fixed_wage_floor

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
        print(f"Firm {self.unique_id} VMPK: {vmpk:.2f}, Rental Rate: {rental_rate:.2f}")
        
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
        print(f"Firm {self.unique_id} vacancy check: labor={current_labor}, VMP={vmp:.2f}, wage={self.monthly_wage:.2f}")

        if vmp >= self.monthly_wage:
            self.vacancies = 1  # limit expansion to one vacancy per step

        print(f"Firm {self.unique_id} posted {self.vacancies} vacancies.")

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
        print(f"Firm {self.unique_id} hired {hires} workers. Remaining vacancies: {self.vacancies}")
        if self.vacancies > 0:
            self.vacancy_duration += 1
        else:
            self.vacancy_duration = 0

    def handle_quit(self, worker):
        if worker in self.current_workers:
            self.current_workers.remove(worker)
            print(f"Quit: Worker {worker.unique_id} left Firm {self.unique_id} at wage {worker.monthly_wage:.2f}")
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
        print(f"Fired: Worker {worker.unique_id} from Firm {self.unique_id}")
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
        print(f"Firm {self.unique_id} adjusting wage: Vacancy Rate: {vacancy_rate:.2f}, Quit Rate: {quit_rate:.2f}, Current Wage: {self.monthly_wage:.2f}, Current Profit: {current_profit:.2f}")

        current_wage = self.monthly_wage
        current_profit = self.profit if current_profit is None else current_profit
        new_wage = current_wage

        # 1) Vacancy/quit pressure first: if we cannot hire (open roles persist), lift wages immediately
        cannot_hire = self.vacancies > 0 and self.vacancy_duration > 0
        if cannot_hire:
            # Scale more aggressively with vacancy pressure
            delta = base_delta * (1 + vacancy_rate + quit_rate)
            new_wage = max(current_wage * (1 + delta * (1 + self.vacancy_duration)), self.wage_floor())
            print(f"Firm {self.unique_id} cannot hire, increasing wage aggressively to {new_wage:.2f} (delta: {delta:.4f}, vacancy duration: {self.vacancy_duration})")
        else:
            # 2) Profit projection branch when hiring is not blocked; use a calmer scaler
            delta = base_delta
            candidate_up = max(current_wage * (1 + delta), self.wage_floor())
            candidate_down = max(current_wage * (1 - delta), self.wage_floor())
            profit_up = self.compute_profit(candidate_up)
            profit_down = self.compute_profit(candidate_down)

            if profit_up > current_profit and profit_up >= profit_down:
                new_wage = candidate_up
                print(f"Firm {self.unique_id} profit projection suggests increasing wage to {new_wage:.2f} (profit up: {profit_up:.2f})")
            elif profit_down > current_profit:
                new_wage = candidate_down
                print(f"Firm {self.unique_id} profit projection suggests decreasing wage to {new_wage:.2f} (profit down: {profit_down:.2f})")

            quit_adjustment = quit_gamma * (quit_rate - target_quit)
            new_wage *= (1 + quit_adjustment)
            print(f"Firm {self.unique_id} adjusting wage by quit rate: new wage {new_wage:.2f} (quit adjustment: {quit_adjustment:.4f})")

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
                    print(f"Firm {self.unique_id} adjusting wage by VMPL gap: new wage {new_wage:.2f} (gap ratio: {gap_ratio:.4f})")

        # Apply bounds and propagate to workers
        self.monthly_wage = int(max(new_wage, self.wage_floor()))
        self.daily_wage = self.monthly_wage / 20
        for w in self.current_workers:
            w.monthly_wage = self.monthly_wage
            w.daily_wage = self.daily_wage

    def onboard_workers_step(self):
        """Move workers from pending to current (1-step hiring delay)"""
        self.current_workers.extend(self.pending_workers)
        self.pending_workers = []
        # All workers should have the same wages based on the firm's current wage setting
        for w in self.current_workers:
            w.monthly_wage = self.monthly_wage
            w.daily_wage = self.daily_wage

    def adjust_employment_step(self):
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

    def hire_step(self):
        self.hire()

    def step(self):
        self.last_worker_count = len(self.current_workers)
        # Production phase
        output = self.produce()
        self.last_output = output
        revenue = output * self.output_price

        # Wage cost
        total_wage_cost = sum(w.monthly_wage for w in self.current_workers)
        
        # Capital rental cost
        capital_cost = self.capital * self.rental_rate

        # Profit calculation
        profit = revenue - total_wage_cost - capital_cost
        self.profit = profit

        # Track consecutive deficits for exit decisions
        if profit < 0:
            self.deficit_months += 1
        else:
            self.deficit_months = 0

        if self.deficit_months >= self.model.deficit_exit_months:
            print(f"Firm {self.unique_id} flagged for exit after {self.deficit_months} deficit months.")
            self.model.queue_firm_exit(self)

        # Adjust capital and wage every 12 steps but not in the first step to allow initial conditions to stabilize
        if self.model.step_count % 12 == 0 and self.model.step_count > 0:
            total_labor = len(self.current_workers)
            self.adjust_capital(total_labor, self.rental_rate)
            self.optimize_wage_annual()

        self.prev_profit = self.profit
        self.quits_last_month = 0

    def job_search_step(self):
        pass

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


# -------------------
# MODEL
# -------------------
class LaborMarketModel(Model):
    def __init__(self, N_workers=300, N_firms=20, min_wage=7700, simulator=None, seed=42):
        super().__init__(seed=seed)
        if simulator:
            self.simulator = simulator
            self.simulator.setup(self)
        self.step_count = 0
        self.running = True # Needed for Mesa to know the model is running
        self.schedule = StagedActivation(self, stage_list=["onboard_workers_step", "adjust_employment_step", "job_search_step", "hire_step", "step"], shuffle=False)  #TODO Update list of stages later
        random.seed(seed)

        self.num_workers = N_workers
        self.num_firms = N_firms
        self.deficit_exit_months = 24  # default: exit after 24 consecutive deficit months (2 years)
        self.MAX_HOURS = 8 * 6 * 4  # (192 hours) Max working hours per month (assumed 8 hours/day * 6 days/week * 4 weeks)
        self.min_wage = min_wage
        self.total_profit = 0  # TODO: Track total profit in the economy for analysis
        self.pending_firm_exits = []
        
        # Create agents
        for i in range(self.num_workers):
            w = Worker(i, self, hours_worked=160, non_labor_income=random.uniform(0, 5000), 
                       consumption_weight=random.uniform(0.3, 0.7))
            self.schedule.add(w)
        for i in range(self.num_firms):
            f = Firm(f"F{i}", self, capital=random.uniform(10,100), rental_rate=500,
                      productivity=random.uniform(0.8, 1.2), output_price=100)  #TODO: Tune these parameters to scale output and wages to realistic levels (currently set to produce wages in the range of 20,000-40,000 THB per month, can be adjusted based on calibration)
            self.schedule.add(f)

        self.next_firm_id = self.num_firms

        # Add initial workers to firms. Each firm gets between 3-5 workers to start
        all_workers = [a for a in self.schedule.agents if isinstance(a, Worker)]
        all_firms = [a for a in self.schedule.agents if isinstance(a, Firm)]
        random.shuffle(all_workers)
        for firm in all_firms:
            initial_hires = random.randint(3, 5)
            for _ in range(initial_hires):
                if all_workers:
                    worker = all_workers.pop()
                    worker.employed = True
                    worker.employer = firm
                    # Worker wage is set based on firm's initial wage setting
                    firm.current_workers.append(worker)

        # Set initial wages for firms and their workers
        for firm in all_firms:
            firm.set_initial_wage(gamma=0.8)  # Pay 80% of MPL initially

        self.workers = [a for a in self.schedule.agents if isinstance(a, Worker)]
        self.firms = [a for a in self.schedule.agents if isinstance(a, Firm)]

        # def labor_supply(workers, wage):
        #     return sum(1 for w in workers if wage > w.reservation_wage)

        # def labor_demand(firms, wage):
        #     demand = 0
        #     for firm in firms:
        #         while firm.output_price * firm.marginal_product() > wage:
        #             demand += 1
        #             firm.employment += 1
        #     return demand

        # def find_market_clearing_wage(workers, firms, wage_grid):
        #     for wage in wage_grid:
        #         supply = labor_supply(workers, wage)
        #         demand = labor_demand(firms, wage)
        #         if abs(supply - demand) <= 1:
        #             return wage

        # Find market clearing wage
        # workers = [a for a in self.schedule.agents if isinstance(a, Worker)]
        # firms = [a for a in self.schedule.agents if isinstance(a, Firm)]
        # wage_grid = np.arange(20, 100, 2)  # hourly wage grid from 20 Baht to 100 Baht
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
            "AverageFirmWage": self.get_avg_firm_wage,
            "AverageWorkerUtility": self.compute_avg_worker_utility,
            "CompetitiveWage": self.compute_competitive_wage,
            "AvgFirmSize": self.get_firm_size,
            "AvgFirmCapital": self.get_avg_firm_capital,
            "VacancyRate": self.compute_vacancy_rate_model,
            "UnemploymentRate": self.compute_unemployment_rate,
            "MinWage": self.get_min_wage,
            "TotalOutput": self.get_total_output,
            "CapitalStock": self.get_capital_stock,
            "CapitalPerWorker": self.get_capital_per_worker,
            # "AverageMachineInvestment": self.get_avg_machine_investment,
            # NEW: Collect lists of all values for later analysis/distribution plotting
            "AllFirmSizes": self.get_firm_sizes_list, 
            "AllFirmCapitals": self.get_firm_capitals_list,
            # "AllEmployedWages": self.get_employed_wages_list,
            # "AllFirmProfits": self.get_firm_profits_list,
        }
        
        self.datacollector = mesa.DataCollector(model_reporters)
        self.datacollector.collect(self)

    def queue_firm_exit(self, firm):
        if firm not in self.pending_firm_exits:
            self.pending_firm_exits.append(firm)

    def create_new_firm(self):
        firm_id = f"F{self.next_firm_id}"
        self.next_firm_id += 1
        f = Firm(firm_id, self, capital=random.uniform(10,100), rental_rate=500,
                  productivity=random.uniform(0.8, 1.2), output_price=100)
        self.schedule.add(f)
        f.set_initial_wage(gamma=0.8)
        self.firms.append(f)
        print(f"Created replacement firm {firm_id} with initial wage {f.monthly_wage:.2f}")
        return f

    def process_firm_turnover(self):
        if not self.pending_firm_exits:
            return

        exiting = list(self.pending_firm_exits)
        self.pending_firm_exits = []

        for firm in exiting:
            if firm not in self.schedule.agents:
                continue
            firm.exit_and_release_workers()
            self.schedule.remove(firm)
            if firm in self.firms:
                self.firms.remove(firm)
            print(f"Firm {firm.unique_id} exited after sustained deficits.")

        # Replace exited firms one-for-one to keep market size stable
        for _ in exiting:
            self.create_new_firm()
    
    def step(self):
        # increase min wage over time   
        # print(f"min_wage increased to {self.min_wage} at step {self.step_count}")
        self.schedule.step()
        self.process_firm_turnover()
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

    def compute_unemployment_rate(self):
        return 1 - self.compute_employment_rate()

    def compute_avg_wage(self):
        wages = [w.monthly_wage for w in self.schedule.agents if isinstance(w, Worker) and w.employed and w.monthly_wage > 0]
        return np.mean(wages) if wages else 0

    def compute_avg_profit(self):
        profits = [f.profit for f in self.schedule.agents if isinstance(f, Firm)]
        return np.mean(profits) if profits else 0

    def compute_vacancy_rate_model(self):
        firms = [a for a in self.schedule.agents if isinstance(a, Firm)]
        total_vacancies = sum(f.vacancies for f in firms)
        filled = sum(len(f.current_workers) for f in firms)
        positions = total_vacancies + filled
        return total_vacancies / positions if positions > 0 else 0

    def compute_avg_worker_utility(self):
        utilities = []
        for w in self.schedule.agents:
            if isinstance(w, Worker):
                if w.employed:
                    utilities.append(w.utility_if_work(w.monthly_wage))
                else:
                    utilities.append(w.utility_if_not_work())
        return np.mean(utilities) if utilities else 0

    def compute_competitive_wage(self):
        # Employment-weighted value of marginal product of labor (VMP)
        weighted_sum = 0
        labor_sum = 0
        for f in self.schedule.agents:
            if isinstance(f, Firm):
                labor = len(f.current_workers)
                if labor == 0:
                    continue
                mpl = f.marginal_product_labor(f.productivity, labor, f.alpha)
                vmp = mpl * f.output_price
                weighted_sum += vmp * labor
                labor_sum += labor
        return (weighted_sum / labor_sum) if labor_sum > 0 else 0

    def get_firm_size(self):
        # average number of workers per firm
        firm_sizes = [len(f.current_workers) for f in self.schedule.agents if isinstance(f, Firm)]
        return np.mean(firm_sizes) if firm_sizes else 0
    
    def get_avg_firm_capital(self):
        firm_capitals = [f.capital for f in self.schedule.agents if isinstance(f, Firm)]
        return np.mean(firm_capitals) if firm_capitals else 0

    def get_avg_firm_wage(self):
        wages = [f.monthly_wage for f in self.schedule.agents if isinstance(f, Firm) and f.monthly_wage is not None]
        return np.mean(wages) if wages else 0
    
    def get_min_wage(self):
        return self.min_wage  # monthly minimum wage
    
    def get_firm_sizes_list(self):
        return [len(f.current_workers) for f in self.schedule.agents if isinstance(f, Firm)]
    
    def get_firm_capitals_list(self):
        return [f.capital for f in self.schedule.agents if isinstance(f, Firm)]

    def get_total_output(self):
        outputs = [f.last_output for f in self.schedule.agents if isinstance(f, Firm)]
        return float(np.sum(outputs)) if outputs else 0.0

    def get_capital_stock(self):
        capitals = [f.capital for f in self.schedule.agents if isinstance(f, Firm)]
        return float(np.sum(capitals)) if capitals else 0.0

    def get_capital_per_worker(self):
        capitals = [f.capital for f in self.schedule.agents if isinstance(f, Firm)]
        total_capital = float(np.sum(capitals)) if capitals else 0.0
        employed_workers = len([w for w in self.schedule.agents if isinstance(w, Worker) and w.employed])
        return total_capital / employed_workers if employed_workers > 0 else 0.0
    
    # def get_employed_wages_list(self):
    #     return [w.monthly_wage for w in self.schedule.agents if isinstance(w, Worker) and w.employed]
    
    # def get_firm_profits_list(self):
    #     return [f.current_profit for f in self.schedule.agents if isinstance(f, Firm)]
    
    # def get_avg_machine_investment(self):
    #     machine_investments = [f.machine_investment for f in self.schedule.agents if isinstance(f, Firm)]
    #     return np.mean(machine_investments) if machine_investments else 0
