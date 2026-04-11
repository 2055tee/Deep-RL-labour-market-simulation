"""
test_acceptance.py — Acceptance / model-validation tests.

These are directional (economic theory) checks rather than exact-match tests.
They verify the simulation produces outcomes consistent with established labor
economics theory (Borjas, monopsony model).

Run with:  pytest tests/test_acceptance.py -v --timeout=120
"""
import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from conftest import run_simulation


# ─────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────

def _tail_mean(df, col, last_n=50):
    """Average of the last *last_n* rows of column *col*."""
    return float(np.mean(df[col].values[-last_n:]))


# ─────────────────────────────────────────────────────────────────────
# Acceptance tests
# ─────────────────────────────────────────────────────────────────────

class TestEconomicPredictions:
    """Model must produce outcomes consistent with economic theory."""

    @pytest.mark.slow
    def test_higher_min_wage_reduces_employment(self):
        """Standard competitive prediction: higher min wage → lower employment.

        Averages the final 50 steps of a 200-step run to remove transients.
        """
        low  = run_simulation(min_wage=5000,  steps=200, seed=42)
        high = run_simulation(min_wage=15000, steps=200, seed=42)

        emp_low  = _tail_mean(low,  "EmploymentRate")
        emp_high = _tail_mean(high, "EmploymentRate")

        assert emp_low > emp_high, (
            f"Expected lower employment at higher min wage: "
            f"emp@5000={emp_low:.3f}, emp@15000={emp_high:.3f}"
        )

    @pytest.mark.slow
    def test_higher_min_wage_raises_average_wage(self):
        """Mechanical prediction: binding minimum wage must push up average wage."""
        low  = run_simulation(min_wage=5000,  steps=200, seed=42)
        high = run_simulation(min_wage=15000, steps=200, seed=42)

        wage_low  = _tail_mean(low,  "AverageWage")
        wage_high = _tail_mean(high, "AverageWage")

        assert wage_high > wage_low, (
            f"Expected higher avg wage at higher min wage: "
            f"wage@5000={wage_low:.0f}, wage@15000={wage_high:.0f}"
        )

    @pytest.mark.slow
    def test_average_wage_always_at_or_above_min_wage(self):
        """Model invariant: average wage ≥ min_wage at all times."""
        for min_wage in [5000, 7700, 10000]:
            df = run_simulation(min_wage=min_wage, steps=200, seed=42)
            below = (df["AverageWage"] < min_wage).sum()
            assert below == 0, (
                f"AverageWage dipped below min_wage={min_wage} in {below} steps"
            )
