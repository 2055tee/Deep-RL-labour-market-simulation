"""
conftest.py — shared fixtures and factory functions for all test stages.
"""
import sys
import os
import random
import pytest

# Ensure the project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from min_wage_model import LaborMarketModel, Worker, Firm


# ── Factory helpers ───────────────────────────────────────────────────

def create_minimal_model(min_wage=7700, n_workers=10, n_firms=3, seed=42):
    """Return a small seeded LaborMarketModel."""
    return LaborMarketModel(
        N_workers=n_workers,
        N_firms=n_firms,
        min_wage=min_wage,
        seed=seed,
    )


def setup_firm_with_workers(model, n_workers=5):
    """Return the first firm in *model* after assigning n_workers to it."""
    firm = model.firms[0]
    # Detach workers from any existing employer
    available = [w for w in model.workers if not w.employed][:n_workers]
    for w in available:
        w.employed = True
        w.employer = firm
        firm.current_workers.append(w)
    firm.set_initial_wage(gamma=0.8)
    return firm


def run_simulation(min_wage, steps, seed=42):
    """Run the full model for *steps* and return the DataCollector DataFrame."""
    model = LaborMarketModel(
        N_workers=100,
        N_firms=10,
        min_wage=min_wage,
        seed=seed,
    )
    for _ in range(steps):
        model.step()
    return model.datacollector.get_model_vars_dataframe()


# ── pytest fixtures ───────────────────────────────────────────────────

@pytest.fixture
def small_model():
    return LaborMarketModel(
        N_workers=10,
        N_firms=3,
        min_wage=7700,
        seed=42,
    )


@pytest.fixture
def firm_with_workers(small_model):
    return setup_firm_with_workers(small_model, n_workers=5)
