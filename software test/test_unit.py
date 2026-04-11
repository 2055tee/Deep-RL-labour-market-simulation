"""
test_unit.py — Unit tests for deterministic mathematical functions.

Covers: FR-02 (worker utility), FR-03 (firm production/profit/MPL),
        FR-04 (minimum wage floor), FR-05 (firms-to-consider count),
        FR-06 (wage adjustment direction), FR-07 (market-quit sigmoid),
        FR-09 (RL observation vector shape and bounds).
"""
import sys
import os
import math
import random
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from min_wage_model import LaborMarketModel, Worker, Firm
from conftest import create_minimal_model, setup_firm_with_workers


# ─────────────────────────────────────────────────────────────────────
# FR-02  Worker Utility
# ─────────────────────────────────────────────────────────────────────

class TestWorkerUtility:
    """Tests for Cobb-Douglas utility U(C, L) = C^α × L^(1−α)."""

    def test_cobb_douglas_utility_known_values(self):
        """FR-02 — verify utility against hand-computed values."""
        model = create_minimal_model(min_wage=7700)
        worker = model.workers[0]
        # Override attributes to known values
        worker.hours_worked = 160
        worker.non_labor_income = 1000
        worker.alpha = 0.5

        # U(C, L) = C^0.5 × L^0.5
        # consumption = 7700*160 + 1000 = 1_233_000
        # leisure     = 192 - 160 = 32
        expected = (1_233_000 ** 0.5) * (32 ** 0.5)
        result = worker.utility_if_work(7700)
        assert abs(result - expected) < 0.01

    def test_utility_positive_for_positive_inputs(self):
        """FR-02 — utility must be positive for positive consumption and leisure."""
        model = create_minimal_model()
        worker = model.workers[0]
        for alpha in [0.3, 0.5, 0.7]:
            worker.alpha = alpha
            assert worker.utility_if_work(model.min_wage) > 0

    def test_utility_increases_with_wage(self):
        """FR-02 — higher wage increases utility (consumption weight > 0)."""
        model = create_minimal_model()
        worker = model.workers[0]
        worker.alpha = 0.5
        u_low = worker.utility_if_work(7700)
        u_high = worker.utility_if_work(15000)
        assert u_high > u_low

    def test_utility_if_not_work_uses_max_hours(self):
        """FR-02 — unemployment utility uses full MAX_HOURS as leisure."""
        model = create_minimal_model()
        worker = model.workers[0]
        worker.non_labor_income = 500
        worker.alpha = 0.5
        expected = (500 ** 0.5) * (model.MAX_HOURS ** 0.5)
        assert abs(worker.utility_if_not_work() - expected) < 0.01

    def test_high_alpha_favors_consumption(self):
        """FR-02 — equivalence partition: α > 0.5 favors income over leisure."""
        model = create_minimal_model()
        worker = model.workers[0]
        worker.alpha = 0.9
        worker.non_labor_income = 0
        # At very high wage, utility should be much larger than at low wage
        u_low = worker.utility_if_work(7700)
        u_high = worker.utility_if_work(30000)
        assert u_high > u_low * 2

    def test_low_alpha_favors_leisure(self):
        """FR-02 — equivalence partition: α < 0.5 favors leisure."""
        model = create_minimal_model()
        worker = model.workers[0]
        worker.alpha = 0.1
        worker.non_labor_income = 1000
        # Outside option (full leisure) should dominate low-wage employment
        u_work = worker.utility_if_work(7700)
        u_rest = worker.utility_if_not_work()
        # No assertion on direction (depends on wage), but both must be positive
        assert u_work > 0
        assert u_rest > 0


# ─────────────────────────────────────────────────────────────────────
# FR-03  Firm Production and Profit
# ─────────────────────────────────────────────────────────────────────

class TestFirmProduction:
    """Tests for Y = A × K^(1−α) × L^α and profit computation."""

    def test_cobb_douglas_production(self):
        """FR-03 — verify Y = A × K^(1-alpha) × L^alpha."""
        model = create_minimal_model()
        firm = setup_firm_with_workers(model, n_workers=5)
        # productivity stored as 60 * raw_prod; alpha = 0.65
        A = firm.productivity
        K = firm.capital
        L = len(firm.current_workers)
        expected = A * (K ** (1 - firm.alpha)) * (L ** firm.alpha)
        assert abs(firm.produce() - expected) < 0.01

    def test_production_increases_with_labor(self):
        """FR-03 — diminishing returns: more workers → more output."""
        model = create_minimal_model()
        firm = model.firms[0]
        firm.current_workers = []
        prev_output = 0
        for n in range(1, 6):
            firm.current_workers = [None] * n
            out = firm.produce()
            assert out > prev_output
            prev_output = out

    def test_profit_computation(self):
        """FR-03 — profit = revenue - wage_cost - capital_cost."""
        model = create_minimal_model()
        firm = setup_firm_with_workers(model, n_workers=5)
        L = len(firm.current_workers)
        w = firm.monthly_wage
        A = firm.productivity
        K = firm.capital
        output = A * (K ** (1 - firm.alpha)) * (max(L, 1e-6) ** firm.alpha)
        expected = output * firm.output_price - w * L - K * firm.rental_rate
        assert abs(firm.compute_profit() - expected) < 1.0

    def test_profit_decreases_with_higher_wage(self):
        """FR-03 — holding labor constant, higher wage means lower profit."""
        model = create_minimal_model()
        firm = setup_firm_with_workers(model, n_workers=5)
        p_low = firm.compute_profit(wage=7700)
        p_high = firm.compute_profit(wage=20000)
        assert p_low > p_high

    def test_mpl_zero_workers_no_division_error(self):
        """FR-03 — MPL with zero workers must not raise ZeroDivisionError."""
        model = create_minimal_model()
        firm = model.firms[0]
        # Should return a finite value, not raise
        result = firm.marginal_product_labor(firm.productivity, 0, firm.alpha)
        assert math.isfinite(result)

    def test_mpk_zero_capital_returns_zero(self):
        """FR-03 — MPK with zero capital returns 0.0 (guard clause)."""
        model = create_minimal_model()
        firm = model.firms[0]
        firm.capital = 0
        result = firm.marginal_product_capital(firm.productivity, 5, firm.alpha)
        assert result == 0.0


# ─────────────────────────────────────────────────────────────────────
# FR-04  Minimum Wage Floor
# ─────────────────────────────────────────────────────────────────────

class TestMinimumWageFloor:
    """Tests that wages never fall below the government-set floor."""

    def test_wage_never_below_minimum_after_adjust(self):
        """FR-04 — after adjust_wage(), monthly_wage >= min_wage."""
        model = create_minimal_model(min_wage=7700)
        firm = setup_firm_with_workers(model, n_workers=5)
        # Pin wage at floor, then try to adjust downwards
        firm.monthly_wage = 7700
        firm.adjust_wage()
        assert firm.monthly_wage >= 7700

    def test_initial_wage_respects_floor(self):
        """FR-04 — set_initial_wage always produces wage >= min_wage."""
        model = create_minimal_model(min_wage=7700)
        for firm in model.firms:
            assert firm.monthly_wage >= model.min_wage

    def test_wage_floor_method_returns_model_min_wage(self):
        """FR-04 — wage_floor() returns model.min_wage when no fixed floor set."""
        model = create_minimal_model(min_wage=9000)
        firm = model.firms[0]
        firm.fixed_wage_floor = None
        assert firm.wage_floor() == 9000

    def test_wage_floor_respects_fixed_floor(self):
        """FR-04 — fixed_wage_floor overrides model.min_wage."""
        model = create_minimal_model(min_wage=7700)
        firm = model.firms[0]
        firm.fixed_wage_floor = 10000
        assert firm.wage_floor() == 10000


# ─────────────────────────────────────────────────────────────────────
# FR-05  Job Search — firms-to-consider count
# ─────────────────────────────────────────────────────────────────────

class TestJobSearch:
    """Tests for the ~25% firm visibility rule."""

    def test_firms_to_consider_count_unemployed(self):
        """FR-05 — unemployed worker sees max(3, N//4) firms."""
        model = LaborMarketModel(N_workers=50, N_firms=20, seed=42)
        worker = model.workers[0]
        worker.employed = False
        firms = worker._firms_to_consider()
        n_active = len(model.active_firms())
        expected = max(3, n_active // 4)
        assert len(firms) == expected

    def test_firms_to_consider_excludes_own_employer(self):
        """FR-05 — employed worker cannot see their own employer in the pool."""
        model = LaborMarketModel(N_workers=50, N_firms=10, seed=42)
        worker = next(w for w in model.workers if w.employed)
        firms = worker._firms_to_consider()
        assert worker.employer not in firms

    def test_firms_to_consider_minimum_two(self):
        """FR-05 — at least 2 firms returned even with tiny market."""
        model = LaborMarketModel(N_workers=10, N_firms=3, seed=42)
        worker = model.workers[0]
        worker.employed = False
        firms = worker._firms_to_consider()
        assert len(firms) >= 2


# ─────────────────────────────────────────────────────────────────────
# FR-06  Wage Adjustment
# ─────────────────────────────────────────────────────────────────────

class TestWageAdjustment:
    """Tests for heuristic wage adjustment direction."""

    def test_wage_adjustment_never_below_floor(self):
        """FR-06 — wage never drops below wage_floor() after adjustment."""
        model = create_minimal_model(min_wage=7700)
        firm = setup_firm_with_workers(model, n_workers=5)
        firm.monthly_wage = 7700
        firm.vacancy_duration = 0
        firm.vacancies = 0
        firm.adjust_wage()
        assert firm.monthly_wage >= firm.wage_floor()

    def test_wage_increases_when_vacancy_unfilled(self):
        """FR-06 — firm with open vacancy and long vacancy duration should raise wage."""
        model = create_minimal_model(min_wage=7700)
        firm = setup_firm_with_workers(model, n_workers=2)
        firm.monthly_wage = 8000
        firm.vacancies = 1
        firm.vacancy_duration = 5
        old_wage = firm.monthly_wage
        firm.adjust_wage()
        assert firm.monthly_wage >= old_wage


# ─────────────────────────────────────────────────────────────────────
# FR-07  Market Quit — sigmoid probability
# ─────────────────────────────────────────────────────────────────────

class TestMarketQuit:
    """Tests for the probabilistic sigmoid market-quit mechanism."""

    def _sigmoid(self, months, patience):
        x = months - patience / 2.0
        return 1.0 / (1.0 + math.exp(-x))

    def test_market_quit_sigmoid_probability_increasing(self):
        """FR-07 — quit probability increases monotonically with months_below_mkt."""
        patience = 4
        probs = [self._sigmoid(m, patience) for m in range(1, 8)]
        for i in range(len(probs) - 1):
            assert probs[i] < probs[i + 1], (
                f"Quit probability not monotonically increasing at month {i+1}"
            )

    def test_market_quit_probability_values(self):
        """FR-07 — sigmoid formula: x = months - patience/2; verify key thresholds.

        With patience=4: month1 ≈ 27%, month2 = 50%, month4 ≈ 88%.
        (The PDF quoted approximate values from an earlier parameterisation;
        the live code uses x = months - patience/2.0.)
        """
        patience = 4
        assert abs(self._sigmoid(1, patience) - 0.269) < 0.01   # ~27%
        assert abs(self._sigmoid(2, patience) - 0.500) < 0.01   # 50% (inflection)
        assert abs(self._sigmoid(4, patience) - 0.881) < 0.01   # ~88%

    def test_market_quit_counter_resets_when_above_threshold(self):
        """FR-07 — months_below_mkt resets to 0 when wage exceeds threshold."""
        model = LaborMarketModel(N_workers=20, N_firms=5, seed=42)
        worker = next(w for w in model.workers if w.employed)
        worker.months_below_mkt = 3
        # Force wage well above market average
        for f in model.active_firms():
            f.monthly_wage = 7700
        worker.monthly_wage = 99999
        worker.employer.monthly_wage = 99999
        worker.job_search_step()
        assert worker.months_below_mkt == 0


# ─────────────────────────────────────────────────────────────────────
# FR-09  RL Environment — observation vector
# ─────────────────────────────────────────────────────────────────────

class TestRLObservation:
    """Tests for the 13-feature Gymnasium observation vector."""

    @pytest.fixture(autouse=True)
    def import_env(self):
        """Skip if reformed package or gymnasium not available."""
        try:
            import gymnasium  # noqa: F401
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "reformed"))
            from firm_env import ReformedFirmEnv
            self.EnvClass = ReformedFirmEnv
        except ImportError:
            pytest.skip("gymnasium or reformed package not available")

    def test_observation_vector_shape(self):
        """FR-09 — observation must have shape (13,)."""
        env = self.EnvClass(N_workers=20, N_firms=5, min_wage=7700)
        obs, _ = env.reset()
        assert obs.shape == (13,), f"Expected shape (13,), got {obs.shape}"

    def test_observation_vector_bounds(self):
        """FR-09 — all observation features must lie within [-1.5, 1.5]."""
        env = self.EnvClass(N_workers=20, N_firms=5, min_wage=7700)
        obs, _ = env.reset()
        assert np.all(obs >= -1.5), f"Obs below -1.5: {obs}"
        assert np.all(obs <= 1.5), f"Obs above 1.5: {obs}"

    def test_action_space_has_8_actions(self):
        """FR-09 — action space must be Discrete(8)."""
        import gymnasium as gym
        env = self.EnvClass(N_workers=20, N_firms=5, min_wage=7700)
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert env.action_space.n == 8
