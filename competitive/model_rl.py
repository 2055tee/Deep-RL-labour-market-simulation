# cooperative/model_rl.py  — supports multiple RL firms (rl_firm_ids set)

import random
import numpy as np
from mesa import Model, Agent
from mesa.time import StagedActivation


class Worker(Agent):

    def __init__(self, uid, model, hours_worked, non_labor_income, consumption_rate):
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
        self.rl_action = 0
        self.last_utility = 0

    def cobb_douglas_utility(self, consumption, leisure):
        consumption = max(consumption, 1e-6)
        leisure = max(leisure, 1e-6)
        return (consumption ** self.alpha) * (leisure ** (1 - self.alpha))

    def utility_if_work(self, wage):
        consumption = wage * self.hours_worked + self.non_labor_income
        leisure = self.model.MAX_HOURS - self.hours_worked
        return self.cobb_douglas_utility(consumption, leisure)

    def utility_if_not_work(self):
        return self.cobb_douglas_utility(self.non_labor_income, self.model.MAX_HOURS)

    def search_for_jobs(self, firms):
        if self.employed:
            if random.random() > self.job_search_prob:
                return
            acceptable_firms = []
            for firm in firms:
                if firm.vacancies > 0:
                    if self.utility_if_work(firm.monthly_wage) - (self.monthly_wage * 0.05) > self.utility_if_work(self.monthly_wage):
                        acceptable_firms.append(firm)
            if acceptable_firms:
                chosen_firm = max(acceptable_firms, key=lambda f: self.utility_if_work(f.monthly_wage))
                chosen_firm.applicants.append(self)
            return

        acceptable_firms = []
        utility_if_not_work = self.utility_if_not_work()
        for firm in firms:
            if firm.vacancies > 0:
                if self.utility_if_work(firm.monthly_wage) - (self.monthly_wage * 0.05) > utility_if_not_work:
                    acceptable_firms.append(firm)
        if acceptable_firms:
            chosen_firm = max(acceptable_firms, key=lambda f: self.utility_if_work(f.monthly_wage))
            chosen_firm.applicants.append(self)

    def job_search_step(self):
        if self.employed and self.utility_if_not_work() > self.utility_if_work(self.monthly_wage):
            if self.employer:
                self.employer.handle_quit(self)
            self.employed = False
            self.employer = None
            self.monthly_wage = 0
            return

        all_firms = [a for a in self.model.schedule.agents if isinstance(a, Firm)]
        if self.employed:
            firms_to_consider = random.sample(
                [f for f in all_firms if f != self.employer],
                max(2, (len(all_firms) - 1) // 10)
            )
        else:
            firms_to_consider = random.sample(all_firms, max(2, len(all_firms) // 10))
        self.search_for_jobs(firms_to_consider)

    def step(self):
        if self.steps % self.model.DECISION_INTERVAL == 0:
            current_utility = self.utility_if_work(self.monthly_wage) if self.employed else self.utility_if_not_work()
            self.reward = current_utility - self.last_utility
            self.last_utility = current_utility
        else:
            self.reward = 0
        self.steps += 1

    def rl_stage(self):
        pass

    def hire_step(self):
        pass

    def onboard_workers_step(self):
        pass

    def post_vacancies_step(self):
        pass

    def adjust_employment_step(self):
        pass


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
        self.fixed_wage_floor = None
        self.quits_last_month = 0
        self.rl_action = 0
        self.profit = 0

    def set_initial_wage(self, gamma):
        labor = len(self.current_workers)
        mpl = self.marginal_product_labor(self.productivity, labor, self.alpha)
        vmpl = mpl * self.output_price
        self.fixed_wage_floor = max(self.model.min_wage, 0.7 * vmpl)
        self.monthly_wage = gamma * vmpl
        self.monthly_wage = max(self.monthly_wage, self.wage_floor())
        self.monthly_wage = int(self.monthly_wage)
        self.daily_wage = self.monthly_wage / 20
        for w in self.current_workers:
            w.monthly_wage = self.monthly_wage
            w.daily_wage = self.daily_wage

    def produce(self):
        labor = max(len(self.current_workers), 1e-6)
        return self.productivity * (self.capital ** (1 - self.alpha)) * (labor ** self.alpha)

    def marginal_product_labor(self, A, labor, alpha):
        if labor == 0:
            return 0
        return A * alpha * (self.capital ** (1 - alpha)) * (labor ** (alpha - 1))

    def marginal_product_capital(self, A, labor, alpha):
        if self.capital == 0:
            return 0
        return A * (1 - alpha) * (self.capital ** (-alpha)) * (labor ** alpha)

    def value_of_marginal_product(self, price, mp):
        return price * mp

    def rl_decision(self):
        wage_action = self.rl_action in (1, 2, 3, 4)
        if wage_action and self.steps % 12 != 0:
            return
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
            self.pending_workers.append(worker)
            self.vacancies -= 1
        self.applicants = []
        if self.vacancies > 0:
            self.vacancy_duration += 1
        else:
            self.vacancy_duration = 0

    def onboard_workers_step(self):
        self.current_workers.extend(self.pending_workers)
        self.pending_workers = []
        for w in self.current_workers:
            w.monthly_wage = self.monthly_wage

    def adjust_employment_step(self):
        # RL firms handle vacancies/firing via actions 5 & 6
        if self.uid in self.model.rl_firm_ids:
            return
        fired_anyone = self.fire_for_profit()
        if fired_anyone:
            self.vacancies = 0
            self.applicants = []
            self.vacancy_duration = 0
            return
        self.post_vacancies()

    def step(self):
        self.last_worker_count = len(self.current_workers)
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
            self.model.queue_firm_exit(self)

        if self.steps % 12 == 0:
            self.adjust_capital(len(self.current_workers), self.rental_rate)
            if self.uid not in self.model.rl_firm_ids:
                self.optimize_wage_annual()

        profit_change = self.profit - self.last_profit
        self.reward = (
            0.7 * float(np.tanh(self.profit / 20_000)) +
            0.3 * float(np.tanh(profit_change / 5_000))
        )
        self.last_profit = self.profit
        self.steps += 1

    def rl_stage(self):
        # rl_action is set directly on the firm by the env before model.step()
        if self.uid in self.model.rl_firm_ids:
            self.rl_decision()

    def wage_floor(self):
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
        mpk = self.marginal_product_capital(self.productivity, labor, self.alpha)
        vmpk = self.value_of_marginal_product(self.output_price, mpk)
        if vmpk > rental_rate:
            self.capital *= 1.05
        elif vmpk < rental_rate:
            self.capital *= 0.95

    def post_vacancies(self):
        self.applicants = []
        self.vacancies = 0
        current_labor = len(self.current_workers)
        mpl = self.marginal_product_labor(self.productivity, current_labor + 1, self.alpha)
        vmp = self.output_price * mpl
        if vmp >= self.monthly_wage:
            self.vacancies = 1

    def handle_quit(self, worker):
        if worker in self.current_workers:
            self.current_workers.remove(worker)
        worker.employer = None
        worker.employed = False
        worker.monthly_wage = 0
        worker.daily_wage = 0
        self.quits_last_month += 1

    def fire_one_worker(self):
        if not self.current_workers:
            return False
        worker = self.current_workers.pop()
        worker.employed = False
        worker.employer = None
        worker.monthly_wage = 0
        worker.daily_wage = 0
        return True

    def exit_and_release_workers(self):
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
        self.adjust_wage(current_profit=self.compute_profit())
        self.profit = self.compute_profit()

    def adjust_wage(self, base_delta=0.02, target_quit=0.02, quit_gamma=0.05, current_profit=None):
        vacancy_rate = self.compute_vacancy_rate()
        quit_rate = self.compute_quit_rate()
        current_wage = self.monthly_wage
        current_profit = self.profit if current_profit is None else current_profit
        new_wage = current_wage

        cannot_hire = self.vacancies > 0 and self.vacancy_duration > 0
        if cannot_hire:
            delta = base_delta * (1 + vacancy_rate + quit_rate)
            new_wage = max(current_wage * (1 + delta * (1 + self.vacancy_duration)), self.wage_floor())
        else:
            delta = base_delta
            candidate_up   = max(current_wage * (1 + delta), self.wage_floor())
            candidate_down = max(current_wage * (1 - delta), self.wage_floor())
            profit_up   = self.compute_profit(candidate_up)
            profit_down = self.compute_profit(candidate_down)

            if profit_up > current_profit and profit_up >= profit_down:
                new_wage = candidate_up
            elif profit_down > current_profit:
                new_wage = candidate_down

            quit_adjustment = quit_gamma * (quit_rate - target_quit)
            new_wage *= (1 + quit_adjustment)

        labor = len(self.current_workers)
        if labor > 0:
            mpl = self.marginal_product_labor(self.productivity, labor, self.alpha)
            vmpl = mpl * self.output_price
            wage_gap = max(vmpl - new_wage, 0)
            if wage_gap > 0:
                gap_ratio = wage_gap / max(vmpl, 1e-6)
                bump_prob = min(0.9, gap_ratio)
                if random.random() < bump_prob:
                    new_wage *= (1 + delta * (1 + gap_ratio))

        self.monthly_wage = int(max(new_wage, self.wage_floor()))
        self.daily_wage = self.monthly_wage / 20
        for w in self.current_workers:
            w.monthly_wage = self.monthly_wage
            w.daily_wage = self.daily_wage

    def compute_profit(self, wage=None, labor_override=None):
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


class LaborMarketModel(Model):

    def __init__(self, N_workers=100, N_firms=10, n_rl_firms=3):

        super().__init__()

        self.MAX_HOURS = 192
        self.DECISION_INTERVAL = 3
        self.min_wage = 7700
        self.deficit_exit_months = 24
        self.pending_firm_exits = []
        self.n_rl_firms = n_rl_firms

        # rl_firm_ids: set of UIDs for all RL-controlled firms (set by env after init)
        self.rl_firm_ids = {f"F{i}" for i in range(n_rl_firms)}

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

        for i in range(N_workers):
            w = Worker(i, self, 160, random.uniform(0, 3000), random.uniform(0.3, 0.7))
            self.schedule.add(w)

        for i in range(N_firms):
            f = Firm(f"F{i}", self,
                     capital=random.uniform(10, 100),
                     rental_rate=500,
                     productivity=random.uniform(0.8, 1.2),
                     output_price=100)
            self.schedule.add(f)

        self.workers = [a for a in self.schedule.agents if isinstance(a, Worker)]
        self.firms   = [a for a in self.schedule.agents if isinstance(a, Firm)]

        for firm in self.firms:
            initial_hires = random.randint(3, 5)
            for _ in range(initial_hires):
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
