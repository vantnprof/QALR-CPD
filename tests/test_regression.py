"""Regression checks of the solvers and the loop: the same checks, on the same instances, as `python main.py check`."""
import numpy as np
import pytest

from src import checks

CHECKS = {
    "1": "vectorized exact search equals the per-mask loop",
    "2": "revision bit flipping equals the first-version implementation",
    "3": "Proposition 2 removed-energy identity at a stationary point",
    "4": "diagonal-only QUBO equals the magnitude threshold (Mag-IL)",
    "5": "a keep-all pruning call reproduces fixed-rank ALS sweep by sweep",
    "6": "SA and tabu return feasible, near-exact masks",
    "7": "seed s runs on tensor s for manifest datasets",
    "8": "reproduction mode (final-sweep call, original exact-search loop)",
    "9": "QAOA honours the non-empty rule; legacy flag reproduces the first version",
}


@pytest.fixture(scope="module")
def results():
    checks.results.clear()
    rng = np.random.default_rng(12345)          # same sequence as src.checks.main
    checks.check_es(rng)
    checks.check_bf(rng)
    checks.check_prop2_and_mag(rng)
    checks.check_keep_all()
    checks.check_sa_tabu(rng)
    checks.check_seed_per_tensor()
    checks.check_reproduction_mode()
    checks.check_qaoa()
    return list(checks.results)


@pytest.mark.parametrize("num", list(CHECKS), ids=list(CHECKS.values()))
def test_check(results, num):
    outcome = [ok for name, ok in results if name.split()[0] == num]
    assert outcome, f"check {num} did not run"
    assert all(outcome), f"check {num} failed: {CHECKS[num]}"
