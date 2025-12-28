"""
Utility functions for Global SeqOT
"""

import numpy as np
from scipy.special import logsumexp
from sklearn.metrics.pairwise import cosine_distances


def compute_cosine_distance(X, Y):
    """
    Compute cosine distance matrix between two sets of embeddings.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples_X, n_features)
        Source embeddings
    Y : np.ndarray, shape (n_samples_Y, n_features)
        Target embeddings

    Returns
    -------
    C : np.ndarray, shape (n_samples_X, n_samples_Y)
        Cosine distance matrix
    """
    return cosine_distances(X, Y)


def log_kernel_product(log_u, log_K, log_v, axis=None):
    """
    Compute diag(u) @ K @ diag(v) in log domain with stabilization.

    This computes: log(u_i * K_ij * v_j) = log_u_i + log_K_ij + log_v_j
    Then sums along the specified axis using logsumexp for numerical stability.

    Parameters
    ----------
    log_u : np.ndarray, shape (n,)
        Log of left scaling vector
    log_K : np.ndarray, shape (n, m)
        Log of kernel matrix
    log_v : np.ndarray, shape (m,)
        Log of right scaling vector
    axis : int or None
        Axis along which to sum (0 for column sums, 1 for row sums)

    Returns
    -------
    result : np.ndarray
        Log of the result after summation
    """
    # Broadcast: (n, 1) + (n, m) + (1, m)
    log_product = log_u[:, None] + log_K + log_v[None, :]

    if axis is None:
        return logsumexp(log_product)
    else:
        return logsumexp(log_product, axis=axis)


def log_row_sums(log_u, log_K, log_v):
    """
    Compute row sums of diag(u) @ K @ diag(v) in log domain.

    Returns
    -------
    row_sums : np.ndarray, shape (n,)
        Log of row sums
    """
    return log_kernel_product(log_u, log_K, log_v, axis=1)


def log_col_sums(log_u, log_K, log_v):
    """
    Compute column sums of diag(u) @ K @ diag(v) in log domain.

    Returns
    -------
    col_sums : np.ndarray, shape (m,)
        Log of column sums
    """
    return log_kernel_product(log_u, log_K, log_v, axis=0)


def safe_log(x, epsilon=1e-100):
    """
    Compute log with numerical safety.

    Parameters
    ----------
    x : np.ndarray
        Input array
    epsilon : float
        Small constant to avoid log(0)

    Returns
    -------
    log_x : np.ndarray
        Logarithm of input
    """
    return np.log(np.maximum(x, epsilon))


def safe_exp(log_x):
    """
    Compute exp with numerical safety.

    Parameters
    ----------
    log_x : np.ndarray
        Log of values

    Returns
    -------
    x : np.ndarray
        Exponential of input
    """
    return np.exp(np.clip(log_x, -500, 500))


def normalize_distribution(dist):
    """
    Normalize a distribution to sum to 1.

    Parameters
    ----------
    dist : np.ndarray
        Input distribution

    Returns
    -------
    normalized : np.ndarray
        Normalized distribution
    """
    dist = np.asarray(dist, dtype=np.float64)
    total = np.sum(dist)
    if total > 0:
        return dist / total
    else:
        return np.ones_like(dist) / len(dist)
