# min_wage_model.py
#
# Heuristic-only labor market model (no RL).
# Structural improvements merged from reformed/model.py:
#   - Market-quit: workers leave probabilistically when wage < 91% of market
#     for 4+ months (sigmoid: months=1->12%, months=2->27%, months=4->73%)
#   - Workers see max(3, N//4) firms per search (was max(2, N//10))
#   - Switching cost is utility-proportional (was raw THB — unit bug)
#   - Firm exit/replacement: bankrupt firms exit after deficit_exit_months
#   - Per-firm step counter for wage review timing
#   - active_firms() filters out exited firms
#
# Mesa/DataCollector/Solara compatible — viz_wage.py and benchmark_heuristic.py
# both import LaborMarketModel, Worker, Firm from this file.

from mesa import Model, Agent
from mesa.time import StagedActivation
import mesa
import random
import numpy as np


MAX_VACANCIES           = 5
MARKET_QUIT_THRESHOLD   = 0.91   # quit if wage < 91% of market wage
MARKET_QUIT_PATIENCE    = 4      # months below threshold before quitting


# ─────────────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────────────

class Worker(Agent):

    def __init__(self, unique_id, model, hours_worked, non_labor_income, consumption_weight):
        super().__init__(model)
        self.unique_id        = unique_id
        self.employed         = False
        self.employer         = None
        self.hours_worked     = hours_worked
        self.non_labor_income = non_labor_income
        self.alpha            = consumption_weight
        self.job_search_prob  = 0.1            # base on-the-job search probability
        self.monthly_wage     = 0
        self.daily_wage       = 0
        self.steps            = 0
        self.months_below_mkt = 0              # market-quit counter

    # ── Utility ──────────────────────────────────────────────────────

    def cobb_douglas_utility(self, consumption, leisure):
        return max(consumption, 1e-6) ** self.alpha * max(leisure, 1e-6) ** (1 - self.alpha)

    def utility_if_work(self, wage):
        consumption = wage * self.hours_worked + self.non_labor_income
        leisure     = self.model.MAX_HOURS - self.hours_worked
        return self.cobb_douglas_utility(consumption, leisure)

    def utility_if_not_work(self):
        return self.cobb_douglas_utility(self.non_labor_income, self.model.MAX_HOURS)

    # ── Job search helpers ───────────────────────────────────────────

    def _firms_to_consider(self):
        """See ~25% of active firms per search (was ~10%)."""
        pool = self.model.active_firms()
        if self.employed:
            pool = [f for f in pool if f is not self.employer]
        n = max(3, len(pool) // 4)
        return random.sample(pool, min(n, len(pool)))

    def search_for_jobs(self, firms_to_consider):
        if self.employed:
            u_now     = self.utility_if_work(self.monthly_wage)
            threshold = u_now * 0.05          # require 5% relative utility gain
            candidates = [
                f for f in firms_to_consider
                if f.vacancies > 0
                and self.utility_if_work(f.monthly_wage) - u_now > threshold
            ]
        else:
            u_out = self.utility_if_not_work()
            candidates = [
                f for f in firms_to_consider
                if f.vacancies > 0
                and self.utility_if_work(f.monthly_wage) > u_out
            ]

        if candidates:
            best = max(candidates, key=lambda f: self.utility_if_work(f.monthly_wage))
            best.applicants.append(self)

    # ── Mesa stages ──────────────────────────────────────────────────

    def job_search_step(self):
        # Market-quit: probabilistic drain when wage is below market threshold
        # sigmoid probability: months=1->~12%, months=2->~27%, months=4->~73%
        if self.employed:
            active   = self.model.active_firms()
            mkt_wage = float(np.mean([f.monthly_wage for f in active])) if active else self.monthly_wage
            if self.monthly_wage < self.model.market_quit_threshold * mkt_wage:
                self.months_below_mkt += 1
                x         = self.months_below_mkt - self.model.market_quit_patience / 2.0
                quit_prob = 1.0 / (1.0 + np.exp(-x))
                if random.random() < quit_prob:
                    if self.employer:
                        self.employer.handle_quit(self)
                    self.employed         = False
                    self.employer         = None
                    self.monthly_wage     = 0
                    self.daily_wage       = 0
                    self.months_below_mkt = 0
                    return
            else:
                self.months_below_mkt = 0

        # Utility quit: outside option dominates current job
        if self.employed and self.utility_if_not_work() > self.utility_if_work(self.monthly_wage):
            if self.employer:
                self.employer.handle_quit(self)
            self.employed     = False
            self.employer     = None
            self.monthly_wage = 0
            self.daily_wage   = 0
            return

        firms = self._firms_to_consider()
        if not firms:
            return

        if self.employed:
            if random.random() > self.job_search_prob:
                return

        self.search_for_jobs(firms)

    def adjust_employment_step(self): pass
    def hire_step(self):              pass
    def onboard_workers_step(self):   pass

    def step(self):
        self.steps += 1


# ─────────────────────────────────────────────────────────────────────
# Firm
# ─────────────────────────────────────────────────────────────────────

class Firm(Agent):

    def __init__(self, unique_id, model, capital, rental_rate, productivity, output_price):
        super().__init__(model)
        self.unique_id         = unique_id
        self.capital           = capital
        self.rental_rate       = rental_rate
        self.productivity      = 60 * productivity
        self.output_price      = output_price
        self.alpha             = 0.65
        self.current_workers   = []
        self.applicants        = []
        self.vacancies         = 2
        self.pending_workers   = []
        self.deficit_months    = 0
        self.vacancy_duration  = 0
        self.monthly_wage      = 0
        self.daily_wage        = 0
        self.fixed_wage_floor  = None
        self.quits_last_month  = 0
        self.last_worker_count = 0
        self.profit            = 0
        self.last_profit       = 0
        self.last_output       = 0     # tracked for DataCollector TotalOutput
        self.active            = True
        self.steps             = 0

    # ── Economics ────────────────────────────────────────────────────

    def produce(self):
        labor = max(len(self.current_workers), 1e-6)
        return self.productivity * (self.capital ** (1 - self.alpha)) * (labor ** self.alpha)

    def marginal_product_labor(self, A, labor, alpha):
        labor = max(labor, 1e-6)
        return A * alpha * (self.capital ** (1 - alpha)) * (labor ** (alpha - 1))

    def marginal_product_capital(self, A, labor, alpha):
        if self.capital == 0:
            return 0.0
        return A * (1 - alpha) * (self.capital ** (-alpha)) * (labor ** alpha)

    def compute_profit(self, wage=None, labor_override=None):
        labor = len(self.current_workers) if labor_override is None else labor_override
        w     = max(self.monthly_wage if wage is None else wage, self.wage_floor())
        labor = max(labor, 1e-6)
        out   = self.productivity * (self.capital ** (1 - self.alpha)) * (labor ** self.alpha)
        return out * self.output_price - w * labor - self.capital * self.rental_rate

    def wage_floor(self):
        return self.fixed_wage_floor if self.fixed_wage_floor is not None else self.model.min_wage

    def set_initial_wage(self, gamma=0.8):
        labor = max(len(self.current_workers), 1)
        mpl   = self.marginal_product_labor(self.productivity, labor, self.alpha)
        vmpl  = mpl * self.output_price
        self.fixed_wage_floor = int(max(self.model.min_wage, 0.7 * vmpl))
        self.monthly_wage     = max(int(gamma * vmpl), self.wage_floor())
        self.daily_wage       = self.monthly_wage / 20
        for w in self.current_workers:
            w.monthly_wage = self.monthly_wage
            w.daily_wage   = self.daily_wage

    # ── Vacancy / hiring ─────────────────────────────────────────────

    def post_vacancies(self):
        self.applicants = []
        self.vacancies  = 0
        labor = len(self.current_workers)
        mpl   = self.marginal_product_labor(self.productivity, labor + 1, self.alpha)
        if self.output_price * mpl >= self.monthly_wage:
            self.vacancies = 1

    def onboard_workers_step(self):
        self.current_workers.extend(self.pending_workers)
        self.pending_workers = []
        for w in self.current_workers:
            w.monthly_wage = self.monthly_wage
            w.daily_wage   = self.daily_wage

    def hire_step(self):
        random.shuffle(self.applicants)
        hires = min(len(self.applicants), self.vacancies)
        for i in range(hires):
            worker = self.applicants[i]
            if worker.employer and worker.employer is not self:
                worker.employer.handle_quit(worker)
            worker.employed  = True
            worker.employer  = self
            self.pending_workers.append(worker)
            self.vacancies  -= 1
        self.applicants = []
        if self.vacancies > 0:
            self.vacancy_duration += 1
        else:
            self.vacancy_duration = 0

    def handle_quit(self, worker):
        if worker in self.current_workers:
            self.current_workers.remove(worker)
        worker.employer     = None
        worker.employed     = False
        worker.monthly_wage = 0
        worker.daily_wage   = 0
        self.quits_last_month += 1

    def fire_one_worker(self):
        if not self.current_workers:
            return False
        w = self.current_workers.pop()
        w.employed     = False
        w.employer     = None
        w.monthly_wage = 0
        w.daily_wage   = 0
        return True

    def fire_for_profit(self):
        baseline = self.compute_profit()
        fired    = False
        while self.current_workers:
            if self.compute_profit(labor_override=len(self.current_workers) - 1) > baseline:
                self.fire_one_worker()
                baseline = self.compute_profit()
                fired    = True
            else:
                break
        if fired:
            self.profit = baseline
            for w in self.current_workers:
                w.monthly_wage = self.monthly_wage
                w.daily_wage   = self.monthly_wage / 20
        return fired

    def adjust_employment_step(self):
        if not self.active:
            return
        if self.fire_for_profit():
            self.vacancies        = 0
            self.applicants       = []
            self.vacancy_duration = 0
        else:
            self.post_vacancies()

    def job_search_step(self): pass

    # ── Wage ─────────────────────────────────────────────────────────

    def compute_quit_rate(self):
        base = max(self.last_worker_count + self.quits_last_month, 1)
        return self.quits_last_month / base

    def compute_vacancy_rate(self):
        total = len(self.current_workers) + self.vacancies
        return self.vacancies / total if total > 0 else 0

    def adjust_wage(self, base_delta=0.02, target_quit=0.02, quit_gamma=0.05):
        vacancy_rate   = self.compute_vacancy_rate()
        quit_rate      = self.compute_quit_rate()
        current_wage   = self.monthly_wage
        current_profit = self.profit
        new_wage       = current_wage

        cannot_hire = self.vacancies > 0 and self.vacancy_duration > 0
        if cannot_hire:
            delta    = base_delta * (1 + vacancy_rate + quit_rate)
            new_wage = max(current_wage * (1 + delta * (1 + self.vacancy_duration)), self.wage_floor())
        else:
            delta = base_delta
            up    = max(current_wage * (1 + delta), self.wage_floor())
            down  = max(current_wage * (1 - delta), self.wage_floor())
            p_up  = self.compute_profit(up)
            p_dn  = self.compute_profit(down)
            if p_up > current_profit and p_up >= p_dn:
                new_wage = up
            elif p_dn > current_profit:
                new_wage = down
            new_wage *= (1 + quit_gamma * (quit_rate - target_quit))

        # VMPL gap nudge: probabilistic upward push when wage < VMPL
        labor = len(self.current_workers)
        if labor > 0:
            mpl  = self.marginal_product_labor(self.productivity, labor, self.alpha)
            vmpl = mpl * self.output_price
            gap  = max(vmpl - new_wage, 0)
            if gap > 0:
                ratio = gap / max(vmpl, 1e-6)
                if random.random() < min(0.9, ratio):
                    new_wage *= (1 + delta * (1 + ratio))

        self.monthly_wage = int(max(new_wage, self.wage_floor()))
        self.daily_wage   = self.monthly_wage / 20
        for w in self.current_workers:
            w.monthly_wage = self.monthly_wage
            w.daily_wage   = self.daily_wage

    def optimize_wage_annual(self):
        self.adjust_wage()
        self.profit = self.compute_profit()

    def adjust_capital(self):
        labor = len(self.current_workers)
        mpk   = self.marginal_product_capital(self.productivity, labor, self.alpha)
        vmpk  = self.output_price * mpk
        if vmpk > self.rental_rate:
            self.capital *= 1.05
        elif vmpk < self.rental_rate:
            self.capital *= 0.95

    # ── Mesa step ────────────────────────────────────────────────────

    def step(self):
        if not self.active:
            return
        self.last_worker_count = len(self.current_workers)
        output           = self.produce()
        self.last_output = output
        wage_cost        = sum(w.monthly_wage for w in self.current_workers)
        capital_cost     = self.capital * self.rental_rate
        self.profit      = output * self.output_price - wage_cost - capital_cost

        if self.profit < 0:
            self.deficit_months += 1
        else:
            self.deficit_months = 0

        if self.deficit_months >= self.model.deficit_exit_months:
            self.model.queue_firm_exit(self)

        if self.steps % 12 == 0:
            self.adjust_capital()
            self.optimize_wage_annual()

        self.last_profit      = self.profit
        self.quits_last_month = 0
        self.steps           += 1

    def exit_and_release_workers(self):
        for w in list(self.current_workers):
            w.employed     = False
            w.employer     = None
            w.monthly_wage = 0
            w.daily_wage   = 0
        self.current_workers = []
        self.pending_workers = []
        self.vacancies       = 0
        self.applicants      = []
        self.active          = False


# ─────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────

class LaborMarketModel(Model):

    def __init__(self, N_workers=100, N_firms=10,
                 min_wage=7700,
                 simulator=None,
                 market_quit_threshold=None,
                 market_quit_patience=None,
                 max_vacancies=None,
                 deficit_exit_months=24,
                 equal_terms=False,
                 seed=42):
        super().__init__()

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if simulator:
            self.simulator = simulator
            self.simulator.setup(self)

        self.step_count          = 0
        self.running             = True
        self.MAX_HOURS           = 192
        self.min_wage            = int(min_wage)
        self.deficit_exit_months = deficit_exit_months
        self.equal_terms         = equal_terms
        self.pending_firm_exits  = []
        self._firm_counter       = N_firms

        self.market_quit_threshold = (market_quit_threshold if market_quit_threshold is not None
                                      else MARKET_QUIT_THRESHOLD)
        self.market_quit_patience  = (market_quit_patience  if market_quit_patience  is not None
                                      else MARKET_QUIT_PATIENCE)
        self.max_vacancies         = (max_vacancies if max_vacancies is not None
                                      else MAX_VACANCIES)

        self.schedule = StagedActivation(
            self,
            stage_list=[
                "onboard_workers_step",
                "adjust_employment_step",
                "job_search_step",
                "hire_step",
                "step",
            ],
            shuffle=False,
        )

        for i in range(N_workers):
            w = Worker(i, self,
                       hours_worked=160,
                       non_labor_income=random.uniform(0, 3000),
                       consumption_weight=random.uniform(0.3, 0.7))
            self.schedule.add(w)

        for i in range(N_firms):
            if self.equal_terms:
                cap  = random.uniform(44, 66)
                prod = random.uniform(0.9, 1.1)
            else:
                cap  = random.uniform(10, 100)
                prod = random.uniform(0.8, 1.2)
            f = Firm(f"F{i}", self,
                     capital=cap,
                     rental_rate=500,
                     productivity=prod,
                     output_price=100)
            self.schedule.add(f)

        self.workers = [a for a in self.schedule.agents if isinstance(a, Worker)]
        self.firms   = [a for a in self.schedule.agents if isinstance(a, Firm)]

        # Initial hires: shuffle worker pool, pop sequentially into firms
        all_workers = list(self.workers)
        random.shuffle(all_workers)
        for firm in self.firms:
            for _ in range(random.randint(3, 5)):
                if all_workers:
                    w = all_workers.pop()
                    w.employed  = True
                    w.employer  = firm
                    firm.current_workers.append(w)
            firm.set_initial_wage(gamma=0.8)

        # Solara display helpers
        self.average_wage       = 0
        self.employment_rate    = 0
        self.starting_min_wage  = min_wage
        self.number_of_workers  = N_workers
        self.number_of_firms    = N_firms

        # DataCollector
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "EmploymentRate":       self.compute_employment_rate,
                "AverageWage":          self.compute_avg_wage,
                "AverageProfit":        self.compute_avg_profit,
                "AverageFirmWage":      self.get_avg_firm_wage,
                "AverageWorkerUtility": self.compute_avg_worker_utility,
                "CompetitiveWage":      self.compute_competitive_wage,
                "AvgFirmSize":          self.get_firm_size,
                "AvgFirmCapital":       self.get_avg_firm_capital,
                "VacancyRate":          self.compute_vacancy_rate_model,
                "UnemploymentRate":     self.compute_unemployment_rate,
                "MinWage":              self.get_min_wage,
                "TotalOutput":          self.get_total_output,
                "CapitalStock":         self.get_capital_stock,
                "CapitalPerWorker":     self.get_capital_per_worker,
                "AllFirmSizes":         self.get_firm_sizes_list,
                "AllFirmCapitals":      self.get_firm_capitals_list,
            }
        )
        self.datacollector.collect(self)

    # ── Firm lifecycle ────────────────────────────────────────────────

    def active_firms(self):
        return [f for f in self.firms if f.active]

    def queue_firm_exit(self, firm):
        if firm not in self.pending_firm_exits:
            self.pending_firm_exits.append(firm)

    def _spawn_replacement_firm(self):
        uid = f"F{self._firm_counter}"
        self._firm_counter += 1
        f = Firm(uid, self,
                 capital=random.uniform(10, 100),
                 rental_rate=500,
                 productivity=random.uniform(0.8, 1.2),
                 output_price=100)
        f.monthly_wage = self.min_wage
        f.daily_wage   = self.min_wage / 20
        self.schedule.add(f)
        self.firms.append(f)

    def _process_exits(self):
        for firm in self.pending_firm_exits:
            firm.exit_and_release_workers()
            self._spawn_replacement_firm()
        self.pending_firm_exits = []

    # ── Mesa step ────────────────────────────────────────────────────

    def step(self):
        self._process_exits()
        self.schedule.step()
        self.datacollector.collect(self)
        self.update_data()
        self.step_count += 1

    def update_data(self):
        self.average_wage    = self.compute_avg_wage()
        self.employment_rate = self.compute_employment_rate()

    # ── DataCollector reporters ───────────────────────────────────────

    def compute_employment_rate(self):
        workers  = [a for a in self.schedule.agents if isinstance(a, Worker)]
        employed = [w for w in workers if w.employed]
        return len(employed) / len(workers) if workers else 0

    def compute_unemployment_rate(self):
        return 1 - self.compute_employment_rate()

    def compute_avg_wage(self):
        wages = [w.monthly_wage for w in self.schedule.agents
                 if isinstance(w, Worker) and w.employed and w.monthly_wage > 0]
        return np.mean(wages) if wages else 0

    def compute_avg_profit(self):
        profits = [f.profit for f in self.schedule.agents if isinstance(f, Firm)]
        return np.mean(profits) if profits else 0

    def compute_vacancy_rate_model(self):
        firms     = [a for a in self.schedule.agents if isinstance(a, Firm)]
        total_vac = sum(f.vacancies for f in firms)
        filled    = sum(len(f.current_workers) for f in firms)
        positions = total_vac + filled
        return total_vac / positions if positions > 0 else 0

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
        weighted_sum = 0
        labor_sum    = 0
        for f in self.schedule.agents:
            if isinstance(f, Firm):
                labor = len(f.current_workers)
                if labor == 0:
                    continue
                mpl = f.marginal_product_labor(f.productivity, labor, f.alpha)
                vmp = mpl * f.output_price
                weighted_sum += vmp * labor
                labor_sum    += labor
        return (weighted_sum / labor_sum) if labor_sum > 0 else 0

    def get_firm_size(self):
        sizes = [len(f.current_workers) for f in self.schedule.agents if isinstance(f, Firm)]
        return np.mean(sizes) if sizes else 0

    def get_avg_firm_capital(self):
        caps = [f.capital for f in self.schedule.agents if isinstance(f, Firm)]
        return np.mean(caps) if caps else 0

    def get_avg_firm_wage(self):
        wages = [f.monthly_wage for f in self.schedule.agents
                 if isinstance(f, Firm) and f.monthly_wage is not None]
        return np.mean(wages) if wages else 0

    def get_min_wage(self):
        return self.min_wage

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
        total_capital    = self.get_capital_stock()
        employed_workers = len([w for w in self.schedule.agents
                                 if isinstance(w, Worker) and w.employed])
        return total_capital / employed_workers if employed_workers > 0 else 0.0
