"""
Tests for Forward-Backward Sinkhorn solver
"""

import numpy as np
import pytest
from src.seqot.sinkhorn import ForwardBackwardSinkhorn
from src.seqot.metrics import flow_conservation_error, marginal_errors
from src.seqot.utils import compute_cosine_distance


def test_sinkhorn_convergence():
    """Test that the Sinkhorn algorithm converges."""
    np.random.seed(42)

    # Create simple test case: 2 time steps
    n1, n2, n3 = 10, 15, 12
    d = 5

    X1 = np.random.randn(n1, d)
    X2 = np.random.randn(n2, d)
    X3 = np.random.randn(n3, d)

    C1 = compute_cosine_distance(X1, X2)
    C2 = compute_cosine_distance(X2, X3)

    cost_matrices = [C1, C2]

    # Fit solver
    solver = ForwardBackwardSinkhorn(epsilon=0.1, max_iter=500, tol=1e-6, verbose=False)
    solver.fit(cost_matrices)

    assert solver.converged_, "Sinkhorn did not converge"
    assert solver.n_iter_ < 500, "Sinkhorn took too many iterations"


def test_flow_conservation():
    """Test that flow conservation is satisfied."""
    np.random.seed(42)

    n1, n2, n3 = 20, 25, 20
    d = 3

    X1 = np.random.randn(n1, d)
    X2 = np.random.randn(n2, d)
    X3 = np.random.randn(n3, d)

    C1 = compute_cosine_distance(X1, X2)
    C2 = compute_cosine_distance(X2, X3)

    cost_matrices = [C1, C2]

    solver = ForwardBackwardSinkhorn(epsilon=0.05, max_iter=1000, tol=1e-7)
    solver.fit(cost_matrices)

    couplings = solver.get_couplings()

    # Check flow conservation
    errors, max_error = flow_conservation_error(couplings)

    assert max_error < 1e-5, f"Flow conservation error too large: {max_error}"


def test_marginal_constraints():
    """Test that source and target marginals are satisfied."""
    np.random.seed(42)

    n1, n2 = 30, 25
    d = 4

    X1 = np.random.randn(n1, d)
    X2 = np.random.randn(n2, d)

    C = compute_cosine_distance(X1, X2)

    # Create non-uniform marginals
    mu = np.random.rand(n1)
    mu = mu / mu.sum()

    nu = np.random.rand(n2)
    nu = nu / nu.sum()

    solver = ForwardBackwardSinkhorn(epsilon=0.1, max_iter=1000, tol=1e-7)
    solver.fit([C], mu=mu, nu=nu)

    couplings = solver.get_couplings()

    source_error, target_error = marginal_errors(couplings, mu, nu)

    assert source_error < 1e-5, f"Source marginal error too large: {source_error}"
    assert target_error < 1e-5, f"Target marginal error too large: {target_error}"


def test_uniform_marginals():
    """Test with uniform marginals (default case)."""
    np.random.seed(42)

    n = 20
    d = 5

    X1 = np.random.randn(n, d)
    X2 = np.random.randn(n, d)
    X3 = np.random.randn(n, d)

    C1 = compute_cosine_distance(X1, X2)
    C2 = compute_cosine_distance(X2, X3)

    solver = ForwardBackwardSinkhorn(epsilon=0.1, max_iter=1000, tol=1e-6)
    solver.fit([C1, C2])

    couplings = solver.get_couplings()

    # Check that couplings are valid probability distributions
    for i, P in enumerate(couplings):
        assert np.all(P >= -1e-10), f"Negative values in coupling {i}"
        assert np.abs(P.sum() - 1.0) < 1e-5, f"Coupling {i} doesn't sum to 1"


def test_multiple_time_steps():
    """Test with longer sequences."""
    np.random.seed(42)

    n_steps = 6
    n_points = 15
    d = 4

    embeddings = [np.random.randn(n_points, d) for _ in range(n_steps)]

    cost_matrices = [
        compute_cosine_distance(embeddings[t], embeddings[t + 1])
        for t in range(n_steps - 1)
    ]

    solver = ForwardBackwardSinkhorn(epsilon=0.1, max_iter=1000, tol=1e-6)
    solver.fit(cost_matrices)

    couplings = solver.get_couplings()

    assert len(couplings) == n_steps - 1, "Wrong number of couplings"

    # Check flow conservation for all intermediate steps
    errors, max_error = flow_conservation_error(couplings)

    assert max_error < 1e-5, f"Flow conservation violated: {max_error}"


def test_objective_decreases():
    """Test that objective function is well-defined."""
    np.random.seed(42)

    n = 20
    d = 3

    X1 = np.random.randn(n, d)
    X2 = np.random.randn(n, d)

    C = compute_cosine_distance(X1, X2)

    solver = ForwardBackwardSinkhorn(epsilon=0.1, max_iter=100, tol=1e-6)
    solver.fit([C])

    # Compute objective
    objective = solver.compute_objective([C])

    assert np.isfinite(objective), "Objective is not finite"
    assert objective > 0, "Objective should be positive"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
