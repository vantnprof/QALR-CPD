"""QALR-CPD source code: quantum-optimization-compatible adaptive low-rank CP decomposition.

CP alternating least squares started from an overfactored model; every few sweeps a QUBO over the components (its
quadratic matrix is the components' Gram matrix) decides which components to remove. The QUBO can be solved by exact
search, bit flipping, simulated annealing, tabu search or QAOA.

Subpackages
    src.cp           CP-ALS with in-loop pruning (CPFactorization)
    src.pruning      pruning QUBO and its solvers
    src.data         dataset loading and the synthetic generator
    src.experiments  experiment configuration, runner and stages
    src.analysis     plateau protocol, CORCONDIA, tables and figures
Command line: ``python main.py --help`` from the repository root.
"""
__version__ = "1.0.0"
