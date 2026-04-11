# cooperative/model_rl.py
#
# Multi-RL labor market model — Reformed rules applied to N RL firms.
#
# Reformed features carried over from reformed/model.py:
#   - Market-quit: sigmoid drain when wage < threshold * market wage
#   - Option 3: workers see max(3, N_active//4) firms per search step
#   - Option 4: wage-gap probability boost (use_wage_gap_prob flag)
#   - Option 5: utility-proportional switching cost (5% relative gain required)
#   - Snap action (7): set wage to nearest-100 market mean
#   - Vacancy cap (max_vacancies) per RL firm
#   - Firm replacement on exit (heuristic firm spawned)
#   - RL firms never exit (protected from deficit_exit_months)
#   - Pure profit reward: 0.7*tanh(profit/5000) + 0.3*tanh(Δprofit/2000)
#
# Key difference from reformed/model.py:
#   - rl_firm_ids  is a set of UIDs (one per RL agent in coop/comp setup)
#   - rl_action is set directly on each Firm by the env (not via model.rl_action)

import random
import numpy as np
from mesa import Model, Agent
from mesa.time import StagedActivation


MAX_VACANCIES           = 5
MARKET_QUIT_THRESHOLD   = 0.91
MARKET_QUIT_PATIENCE    = 4


# ─────────────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────────────

class Worker(Agent):

    def __init__(self, uid, model, hours_worked, non_labor_income, consumption_rate):
        super().__init__(model)
        self.uid              = uid
        self.employed         = False
        self.employer         = None
        self.hours_worked     = hours_worked
        self.non_labor_income = non_labor_income
        self.alpha            = consumption_rate
        self.job_search_prob  = 0.1
        self.monthly_wage     = 0
        self.daily_wage       = 0
        self.steps            = 0
        self.months_below_mkt = 0   # market-quit counter

    def _u(self, consumption, leisure):
        return max(consumption, 1e-6) ** self.alpha * max(leisure, 1e-6) ** (1 - self.alpha)

    def utility_if_work(self, wage):
        c = wage * self.hours_worked + self.non_labor_income
        l = self.model.MAX_HOURS - self.hours_worked
        return self._u(c, l)

    def utility_if_not_work(self):
        return self._u(self.non_labor_income, self.model.MAX_HOURS)

    def _firms_to_consider(self):
        """Option 3: see ~25% of active firms (min 3)."""
        pool = self.model.active_firms()
        if self.employed:
            pool = [f for f in pool if f is not self.employer]
        n = max(3, len(pool) // 4)
        return random.sample(pool, min(n, len(pool)))

    def search_for_jobs(self, firms_to_consider):
        if self.employed:
            u_now = self.utility_if_work(self.monthly_wage)
            # Option 5: require 5% relative utility gain to switch
            threshold = u_now * 0.05
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

    def job_search_step(self):
        # ── Market-quit: sigmoid drain when below threshold * market wage ─
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

        # ── Utility quit ─────────────────────────────────────────────
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
            # Option 4: boost search probability when below market wage
            if self.model.use_wage_gap_prob:
                active   = self.model.active_firms()
                mkt_wage = float(np.mean([f.monthly_wage for f in active])) if active else self.monthly_wage
                shortfall   = max(0.0, (mkt_wage - self.monthly_wage) / max(mkt_wage, 1.0))
                search_prob = self.job_search_prob + shortfall * 0.5
            else:
                search_prob = self.job_search_prob
            if random.random() > search_prob:
                return

        self.search_for_jobs(firms)

    def rl_stage(self): pass
    def hire_step(self): pass
    def onboard_workers_step(self): pass
    def adjust_employment_step(self): pass

    def step(self):
        self.steps += 1


# ─────────────────────────────────────────────────────────────────────
# Firm
# ─────────────────────────────────────────────────────────────────────

class Firm(Agent):

    def __init__(self, uid, model, capital, rental_rate, productivity, output_price):
        super().__init__(model)
        self.uid               = uid
        self.capital           = capital
        self.rental_rate       = rental_rate
        self.productivity      = 60 * productivity
        self.output_price      = output_price
        self.alpha             = 0.65
        self.current_workers   = []
        self.applicants        = []
        self.vacancies         = 2
        self.pending_workers   = []
        self.last_profit       = 0
        self.deficit_months    = 0
        self.vacancy_duration  = 0
        self.monthly_wage      = 0
        self.daily_wage        = 0
        self.fixed_wage_floor  = None
        self.quits_last_month  = 0
        self.last_worker_count = 0
        self.profit            = 0
        self.reward            = 0.0
        self.active            = True
        self.rl_action         = 0
        self.steps             = 0

    # ── Economics ────────────────────────────────────────────────────

    def produce(self):
        labor = max(len(self.current_workers), 1e-6)
        return self.productivity * (self.capital ** (1 - self.alpha)) * (labor ** self.alpha)

    def marginal_product_labor(self, A, labor, alpha):
        if labor == 0:
            return 0.0
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

    # ── RL decision ──────────────────────────────────────────────────

    def rl_decision(self):
        """
        0 = hold
        1 = wage +300   (annual only)
        2 = wage +100   (annual only)
        3 = wage -100   (annual only)
        4 = wage -300   (annual only)
        5 = post 1 vacancy (every step, capped at max_vacancies)
        6 = fire 1 worker  (every step)
        7 = snap to market (every step)
        """
        wage_action = self.rl_action in (1, 2, 3, 4)
        if wage_action and self.steps % 12 != 0:
            return

        if self.rl_action == 1:
            self.monthly_wage += 300
            self._broadcast_wage()
        elif self.rl_action == 2:
            self.monthly_wage += 100
            self._broadcast_wage()
        elif self.rl_action == 3:
            self.monthly_wage = max(self.monthly_wage - 100, self.wage_floor())
            self._broadcast_wage()
        elif self.rl_action == 4:
            self.monthly_wage = max(self.monthly_wage - 300, self.wage_floor())
            self._broadcast_wage()
        elif self.rl_action == 5:
            if self.vacancies < self.model.max_vacancies:
                self.vacancies += 1
        elif self.rl_action == 6:
            if self.current_workers:
                w = self.current_workers.pop()
                w.employed     = False
                w.employer     = None
                w.monthly_wage = 0
                w.daily_wage   = 0
        elif self.rl_action == 7:
            # Snap wage to market mean (exclude self), rounded to nearest 100
            others   = [f.monthly_wage for f in self.model.active_firms() if f is not self]
            mkt_wage = float(np.mean(others)) if others else self.monthly_wage
            snapped  = int(round(mkt_wage / 100.0) * 100)
            self.monthly_wage = max(snapped, self.wage_floor())
            self._broadcast_wage()

    def _broadcast_wage(self):
        self.daily_wage = self.monthly_wage / 20
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
        # RL firms control their own employment via actions
        if self.uid in self.model.rl_firm_ids:
            return
        if self.fire_for_profit():
            self.vacancies        = 0
            self.applicants       = []
            self.vacancy_duration = 0
        else:
            self.post_vacancies()

    # ── RL stage ─────────────────────────────────────────────────────

    def rl_stage(self):
        # rl_action is set directly on the firm by the env
        if self.uid in self.model.rl_firm_ids and self.active:
            self.rl_decision()

    def job_search_step(self): pass

    # ── Wage heuristic ────────────────────────────────────────────────

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
        output       = self.produce()
        wage_cost    = sum(w.monthly_wage for w in self.current_workers)
        capital_cost = self.capital * self.rental_rate
        self.profit  = output * self.output_price - wage_cost - capital_cost

        if self.profit < 0:
            self.deficit_months += 1
        else:
            self.deficit_months = 0

        # Only heuristic firms can exit; RL firms are kept alive
        if self.deficit_months >= self.model.deficit_exit_months:
            if self.uid not in self.model.rl_firm_ids:
                self.model.queue_firm_exit(self)

        if self.steps % 12 == 0:
            self.adjust_capital()
            if self.uid not in self.model.rl_firm_ids:
                self.optimize_wage_annual()

        # Pure profit reward — market-quit handles low-wage punishment naturally
        profit_change = self.profit - self.last_profit
        self.reward = (
            0.7 * float(np.tanh(self.profit       / 5_000)) +
            0.3 * float(np.tanh(profit_change     / 2_000))
        )

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

    def __init__(self, N_workers=100, N_firms=10, n_rl_firms=3,
                 use_wage_gap_prob=True,
                 equal_terms=False,
                 min_wage=7700,
                 market_quit_threshold=None,
                 market_quit_patience=None,
                 max_vacancies=None,
                 deficit_exit_months=24,
                 seed=None):
        super().__init__()

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.MAX_HOURS             = 192
        self.min_wage              = int(min_wage)
        self.deficit_exit_months   = deficit_exit_months
        self.use_wage_gap_prob     = use_wage_gap_prob
        self.equal_terms           = equal_terms
        self.n_rl_firms            = n_rl_firms
        self.rl_firm_ids           = {f"F{i}" for i in range(n_rl_firms)}
        self.pending_firm_exits    = []
        self._firm_counter         = N_firms
        self.market_quit_threshold = market_quit_threshold if market_quit_threshold is not None else MARKET_QUIT_THRESHOLD
        self.market_quit_patience  = market_quit_patience  if market_quit_patience  is not None else MARKET_QUIT_PATIENCE
        self.max_vacancies         = max_vacancies          if max_vacancies          is not None else MAX_VACANCIES

        self.schedule = StagedActivation(
            self,
            stage_list=[
                "rl_stage",
                "onboard_workers_step",
                "adjust_employment_step",
                "job_search_step",
                "hire_step",
                "step",
            ],
            shuffle=False,
        )

        for i in range(N_workers):
            w = Worker(i, self, 160,
                       random.uniform(0, 3000),
                       random.uniform(0.3, 0.7))
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

        # Initial hires
        for firm in self.firms:
            for _ in range(random.randint(3, 5)):
                pool = [w for w in self.workers if not w.employed]
                if pool:
                    w = random.choice(pool)
                    w.employed  = True
                    w.employer  = firm
                    firm.current_workers.append(w)
            firm.set_initial_wage(gamma=0.8)

        # RL firms start at market mean wage (rounded to nearest 100)
        for uid in self.rl_firm_ids:
            rl_firm = next((f for f in self.firms if f.uid == uid), None)
            if rl_firm is not None:
                others   = [f.monthly_wage for f in self.firms if f is not rl_firm]
                mkt_mean = int(round(float(np.mean(others)) / 100.0) * 100) if others else self.min_wage
                rl_firm.fixed_wage_floor = self.min_wage
                rl_firm.monthly_wage     = max(mkt_mean, self.min_wage)
                rl_firm.daily_wage       = rl_firm.monthly_wage / 20
                for w in rl_firm.current_workers:
                    w.monthly_wage = rl_firm.monthly_wage
                    w.daily_wage   = rl_firm.daily_wage

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

    def step(self):
        self._process_exits()
        self.schedule.step()
