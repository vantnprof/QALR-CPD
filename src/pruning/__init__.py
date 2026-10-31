"""The pruning QUBO and its solvers.

qubo        QUBO terms of a CP snapshot (Gram matrix, linear term, removal form)
exhaustive  exact search over all masks (vectorized; QALR_ES_IMPL=loop restores the per-mask loop)
bitflip     single-flip descent (BF)
qaoa        QAOA on Qiskit Aer or IBM hardware, with the uniform non-empty rule
threshold   magnitude threshold
solvers     one entry point for every solver used in the paper (ES, BF, MS-BF, SA, tabu, QAOA, Mag-IL, Diag-x)
"""
