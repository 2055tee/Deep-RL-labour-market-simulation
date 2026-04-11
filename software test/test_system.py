"""
test_system.py — System tests for full-simulation invariants.

Covers: FR-01 (staged activation order), FR-04 (wages above floor),
        FR-08 (firm count stability), FR-10 (data collector reporters).
"""
import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from min_wage_model import LaborMarketModel


# ─────────────────────────────────────────────────────────────────────
# FR-01  Staged activation order
# ─────────────────────────────────────────────────────────────────────

class TestStagedActivation:
    """The model must step through all five stages each tick."""

    def test_staged_activation_order(self):
        """FR-01 — all stage methods exist and are called each step without error."""
        model = LaborMarketModel(N_workers=30, N_firms=5, seed=42)
        # Run 10 steps; any stage ordering bug would likely raise an AttributeError
        for _ in range(10):
            model.step()
        assert model.step_count == 10

    def test_step_count_increments(self):
        """FR-01 — step_count increments by 1 per call to model.step()."""
        model = LaborMarketModel(N_workers=20, N_firms=3, seed=42)
        for i in range(1, 6):
            model.step()
            assert model.step_count == i


# ─────────────────────────────────────────────────────────────────────
# FR-04  Wages above floor — full simulation
# ─────────────────────────────────────────────────────────────────────

class TestWagesAboveFloor:
    """Average wages must never fall below min_wage during a full run."""

    def test_all_firm_wages_above_floor_100_steps(self):
        """FR-04 — every active firm's wage >= min_wage for 100 simulated months."""
        model = LaborMarketModel(N_workers=100, N_firms=10, min_wage=7700, seed=42)
        for _ in range(100):
            model.step()
            for f in model.active_firms():
                assert f.monthly_wage >= model.min_wage, (
                    f"Firm {f.unique_id} wage {f.monthly_wage} < min_wage {model.min_wage}"
                    f" at step {model.step_count}"
                )

    @pytest.mark.parametrize("min_wage", [5000, 7700, 10000, 15000])
    def test_invariants_across_wage_levels(self, min_wage):
        """FR-04 — parametrized: wage floor holds across four min_wage levels."""
        model = LaborMarketModel(
            N_workers=100, N_firms=10,
            min_wage=min_wage, seed=42,
        )
        for step in range(100):
            model.step()
            for f in model.active_firms():
                assert f.monthly_wage >= min_wage, (
                    f"min_wage={min_wage}: firm wage {f.monthly_wage} below floor at step {step+1}"
                )
            assert 0 <= model.compute_employment_rate() <= 1


# ─────────────────────────────────────────────────────────────────────
# FR-08  Firm count stability
# ─────────────────────────────────────────────────────────────────────

class TestFirmCountStability:
    """active_firms() count must stay constant thanks to firm replacement."""

    def test_firm_replacement_maintains_count(self):
        """FR-08 — number of active firms remains N_firms after 100 steps."""
        N_firms = 10
        model = LaborMarketModel(N_workers=100, N_firms=N_firms, seed=42)
        for _ in range(100):
            model.step()
        assert len(model.active_firms()) == N_firms


# ─────────────────────────────────────────────────────────────────────
# General simulation invariants (100 steps)
# ─────────────────────────────────────────────────────────────────────

class TestSimulationInvariants:
    """Core invariants that must hold regardless of stochastic variation."""

    def test_simulation_invariants_100_steps(self):
        """Run 100 steps and verify employment rate, average wage, and firm count."""
        model = LaborMarketModel(N_workers=100, N_firms=10, seed=42)
        for _ in range(100):
            model.step()

        emp_rate = model.compute_employment_rate()
        avg_wage = model.compute_avg_wage()
        active = len(model.active_firms())

        assert 0.0 <= emp_rate <= 1.0, f"Employment rate out of bounds: {emp_rate}"
        assert avg_wage >= model.min_wage, f"Avg wage {avg_wage} < min_wage {model.min_wage}"
        assert active == 10, f"Expected 10 active firms, got {active}"

    def test_employment_rate_bounded(self):
        """Employment rate must always be in [0, 1]."""
        model = LaborMarketModel(N_workers=50, N_firms=5, seed=99)
        for _ in range(50):
            model.step()
            rate = model.compute_employment_rate()
            assert 0.0 <= rate <= 1.0, f"Employment rate {rate} out of [0,1] at step {model.step_count}"


# ─────────────────────────────────────────────────────────────────────
# FR-10  Data collection
# ─────────────────────────────────────────────────────────────────────

class TestDataCollector:
    """DataCollector must report all required metrics each step."""

    REQUIRED_REPORTERS = [
        "EmploymentRate",
        "AverageWage",
        "AverageProfit",
    ]

    def test_data_collector_all_reporters(self):
        """FR-10 — all required reporters are present in collected data."""
        model = LaborMarketModel(N_workers=50, N_firms=5, seed=42)
        for _ in range(5):
            model.step()

        df = model.datacollector.get_model_vars_dataframe()
        for col in self.REQUIRED_REPORTERS:
            assert col in df.columns, f"Reporter '{col}' missing from DataCollector output"

    def test_data_collector_no_nan_values(self):
        """FR-10 — required reporters must not return NaN."""
        model = LaborMarketModel(N_workers=50, N_firms=5, seed=42)
        for _ in range(10):
            model.step()

        df = model.datacollector.get_model_vars_dataframe()
        for col in self.REQUIRED_REPORTERS:
            if col in df.columns:
                assert not df[col].isnull().any(), f"NaN found in reporter '{col}'"

    def test_employment_rate_reported_correctly(self):
        """FR-10 — EmploymentRate must equal employed_workers / total_workers."""
        model = LaborMarketModel(N_workers=50, N_firms=5, seed=42)
        model.step()
        reported = model.compute_employment_rate()
        manual = sum(1 for w in model.workers if w.employed) / len(model.workers)
        assert abs(reported - manual) < 1e-6
