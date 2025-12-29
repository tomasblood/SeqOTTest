"""
Evaluation metrics for Global SeqOT and alignment quality
"""

import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances


def flow_conservation_error(couplings):
    """
    Compute the flow conservation error for intermediate time steps.

    For each intermediate time t, we check:
        ||P^(t)^T @ 1 - P^(t+1) @ 1||_1

    Parameters
    ----------
    couplings : list of np.ndarray
        Transport matrices [P^(1), ..., P^(N-1)]

    Returns
    -------
    errors : list of float
        Flow conservation errors for each intermediate step
    max_error : float
        Maximum flow conservation error
    """
    errors = []

    for t in range(len(couplings) - 1):
        # Outgoing flow from step t
        outgoing = np.sum(couplings[t], axis=0)

        # Incoming flow to step t+1
        incoming = np.sum(couplings[t + 1], axis=1)

        # L1 error
        error = np.sum(np.abs(outgoing - incoming))
        errors.append(error)

    max_error = max(errors) if errors else 0.0

    return errors, max_error


def marginal_errors(couplings, mu, nu):
    """
    Check if source and target marginals are satisfied.

    Parameters
    ----------
    couplings : list of np.ndarray
        Transport matrices
    mu : np.ndarray
        Source distribution
    nu : np.ndarray
        Target distribution

    Returns
    -------
    source_error : float
        L1 error in source marginal
    target_error : float
        L1 error in target marginal
    """
    # Source marginal: P^(1) @ 1 should equal mu
    source_marginal = np.sum(couplings[0], axis=1)
    source_error = np.sum(np.abs(source_marginal - mu))

    # Target marginal: P^(N-1)^T @ 1 should equal nu
    target_marginal = np.sum(couplings[-1], axis=0)
    target_error = np.sum(np.abs(target_marginal - nu))

    return source_error, target_error


def alignment_error(X_source, X_target, X_aligned):
    """
    Measure alignment quality using multiple metrics.

    Parameters
    ----------
    X_source : np.ndarray, shape (n, d)
        Source embeddings
    X_target : np.ndarray, shape (n, d)
        Target embeddings (ground truth)
    X_aligned : np.ndarray, shape (n, d)
        Aligned embeddings

    Returns
    -------
    metrics : dict
        Dictionary containing various alignment metrics
    """
    # Euclidean distance
    euclidean_dist = np.mean(euclidean_distances(X_aligned, X_target))
    euclidean_error = np.mean(np.linalg.norm(X_aligned - X_target, axis=1))

    # Cosine distance
    cosine_dist = np.mean(cosine_distances(X_aligned, X_target))

    # Procrustes distance (optimal rotation)
    R, scale = orthogonal_procrustes(X_aligned, X_target)
    X_procrustes = X_aligned @ R
    procrustes_error = np.mean(np.linalg.norm(X_procrustes - X_target, axis=1))

    # Correlation of corresponding dimensions
    correlations = [np.corrcoef(X_aligned[:, i], X_target[:, i])[0, 1]
                   for i in range(X_aligned.shape[1])]
    mean_correlation = np.mean(correlations)

    return {
        "euclidean_distance": euclidean_dist,
        "euclidean_error": euclidean_error,
        "cosine_distance": cosine_dist,
        "procrustes_error": procrustes_error,
        "mean_correlation": mean_correlation,
    }


def evaluate_alignment(embeddings_source, embeddings_target, embeddings_aligned,
                      method_name="Unknown"):
    """
    Comprehensive evaluation of an alignment method.

    Parameters
    ----------
    embeddings_source : list of np.ndarray
        Source embeddings at each time step
    embeddings_target : list of np.ndarray
        Target (ground truth) embeddings at each time step
    embeddings_aligned : list of np.ndarray
        Aligned embeddings at each time step
    method_name : str
        Name of the alignment method for reporting

    Returns
    -------
    results : dict
        Comprehensive evaluation results
    """
    n_steps = len(embeddings_aligned)

    step_errors = []
    for t in range(n_steps):
        errors = alignment_error(
            embeddings_source[t],
            embeddings_target[t],
            embeddings_aligned[t]
        )
        step_errors.append(errors)

    # Aggregate metrics
    results = {
        "method": method_name,
        "n_steps": n_steps,
        "step_errors": step_errors,
        "mean_euclidean_error": np.mean([e["euclidean_error"] for e in step_errors]),
        "mean_cosine_distance": np.mean([e["cosine_distance"] for e in step_errors]),
        "mean_procrustes_error": np.mean([e["procrustes_error"] for e in step_errors]),
        "mean_correlation": np.mean([e["mean_correlation"] for e in step_errors]),
    }

    return results


def compute_transport_cost(couplings, cost_matrices):
    """
    Compute the total transport cost (without regularization).

    Parameters
    ----------
    couplings : list of np.ndarray
        Transport matrices
    cost_matrices : list of np.ndarray
        Cost matrices

    Returns
    -------
    total_cost : float
        Sum of <P^(t), C^(t)>
    """
    total_cost = 0.0
    for P, C in zip(couplings, cost_matrices):
        total_cost += np.sum(P * C)
    return total_cost


def sparsity_metric(couplings):
    """
    Measure how sparse/focused the transport plans are.

    A lower Gini coefficient indicates more uniform (diffuse) transport,
    while a higher value indicates concentrated (sparse) transport.

    Parameters
    ----------
    couplings : list of np.ndarray
        Transport matrices

    Returns
    -------
    gini_coefficients : list of float
        Gini coefficient for each coupling matrix
    mean_gini : float
        Average Gini coefficient
    """
    def gini(x):
        """Compute Gini coefficient of array x."""
        x = x.flatten()
        x = np.sort(x)
        n = len(x)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * x)) / (n * np.sum(x)) - (n + 1) / n

    gini_coefficients = [gini(P) for P in couplings]
    mean_gini = np.mean(gini_coefficients)

    return gini_coefficients, mean_gini


def tunneling_score(couplings, bridge_indices):
    """
    Measure how much mass goes through specific "bridge" points.

    This is used to validate the tunneling behavior: in Global SeqOT,
    we expect mass to concentrate on semantically relevant intermediate
    papers, rather than diffusing uniformly.

    Parameters
    ----------
    couplings : list of np.ndarray
        Transport matrices [P^(0→1), P^(1→2), ..., P^(N-2→N-1)]
    bridge_indices : list of list of int
        For each time step [0, 1, ..., N-1], the indices of bridge points

    Returns
    -------
    bridge_mass : list of float
        Total mass flowing through bridge points at each step
    """
    bridge_mass = []

    for t, P in enumerate(couplings):
        # P transports from time t to time t+1
        # We want to measure mass going INTO bridge points at time t+1
        target_bridges = bridge_indices[t + 1]

        # Mass entering target bridges (sum over columns for bridge points)
        mass = np.sum(P[:, target_bridges])

        bridge_mass.append(mass)

    return bridge_mass
