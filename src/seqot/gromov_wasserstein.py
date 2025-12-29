"""
Gromov-Wasserstein distance and Sequential Gromov-Wasserstein
for embedding alignment based on internal geometry comparison
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from .utils import safe_log, safe_exp, normalize_distribution


def compute_distance_matrix(X, metric='euclidean'):
    """
    Compute pairwise distance matrix within a set of points.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Points in d-dimensional space
    metric : str
        Distance metric ('euclidean', 'sqeuclidean', 'cosine')

    Returns
    -------
    D : np.ndarray, shape (n, n)
        Pairwise distance matrix
    """
    return cdist(X, X, metric=metric)


def gromov_wasserstein_loss(D1, D2, P):
    """
    Compute the Gromov-Wasserstein loss for a given transport plan.

    L = Σ_ijkl (D1[i,k] - D2[j,l])² P[i,j] P[k,l]

    Parameters
    ----------
    D1 : np.ndarray, shape (n, n)
        Distance matrix for source space
    D2 : np.ndarray, shape (m, m)
        Distance matrix for target space
    P : np.ndarray, shape (n, m)
        Transport plan

    Returns
    -------
    loss : float
        GW loss value
    """
    n, m = P.shape

    # Compute the quadratic term efficiently
    # L = tr(D1^T @ P @ D2 @ P^T) - 2 * tr(D1 @ P @ D2^T @ P^T)
    # Simplified using einsum for efficiency

    # Method 1: Direct computation (clear but potentially slow)
    loss = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(n):
                for l in range(m):
                    diff = D1[i, k] - D2[j, l]
                    loss += diff ** 2 * P[i, j] * P[k, l]

    return loss


def gromov_wasserstein_cost_matrix(D1, D2, P):
    """
    Compute the GW cost matrix (gradient of GW loss w.r.t. P).

    This is used in the entropic GW algorithm.

    Parameters
    ----------
    D1 : np.ndarray, shape (n, n)
        Distance matrix for source space
    D2 : np.ndarray, shape (m, m)
        Distance matrix for target space
    P : np.ndarray, shape (n, m)
        Current transport plan

    Returns
    -------
    L : np.ndarray, shape (n, m)
        GW cost matrix
    """
    n, m = P.shape

    # Efficient computation using matrix operations
    # L[i,j] = Σ_kl (D1[i,k] - D2[j,l])² P[k,l]
    # = Σ_k Σ_l (D1[i,k]² + D2[j,l]² - 2*D1[i,k]*D2[j,l]) P[k,l]
    # = Σ_k D1[i,k]² Σ_l P[k,l] + Σ_l D2[j,l]² Σ_k P[k,l] - 2 Σ_k Σ_l D1[i,k]*D2[j,l]*P[k,l]

    # Term 1: Σ_k D1[i,k]² * (row sum of P)
    P_row_sum = P.sum(axis=1, keepdims=True)  # (n, 1)
    term1 = (D1 ** 2) @ P_row_sum  # (n, 1)
    term1 = np.repeat(term1, m, axis=1)  # (n, m)

    # Term 2: Σ_l D2[j,l]² * (column sum of P)
    P_col_sum = P.sum(axis=0, keepdims=True)  # (1, m)
    term2 = P_col_sum @ (D2 ** 2).T  # (1, m)
    term2 = np.repeat(term2, n, axis=0)  # (n, m)

    # Term 3: -2 * D1 @ P @ D2^T
    term3 = -2 * D1 @ P @ D2.T  # (n, m)

    L = term1 + term2 + term3

    return L


def entropic_gromov_wasserstein(D1, D2, mu=None, nu=None, epsilon=0.1,
                                max_iter=100, tol=1e-6, verbose=False):
    """
    Compute entropic Gromov-Wasserstein distance between two metric spaces.

    Uses the block-coordinate descent algorithm for entropic GW.

    Parameters
    ----------
    D1 : np.ndarray, shape (n, n)
        Distance matrix for source space
    D2 : np.ndarray, shape (m, m)
        Distance matrix for target space
    mu : np.ndarray, shape (n,), optional
        Source distribution (uniform if None)
    nu : np.ndarray, shape (m,), optional
        Target distribution (uniform if None)
    epsilon : float
        Entropic regularization parameter
    max_iter : int
        Maximum number of iterations
    tol : float
        Convergence tolerance
    verbose : bool
        Print progress

    Returns
    -------
    P : np.ndarray, shape (n, m)
        Optimal transport plan
    log : dict
        Optimization log
    """
    n = D1.shape[0]
    m = D2.shape[0]

    # Initialize distributions
    if mu is None:
        mu = np.ones(n) / n
    else:
        mu = normalize_distribution(mu)

    if nu is None:
        nu = np.ones(m) / m
    else:
        nu = normalize_distribution(nu)

    # Initialize transport plan
    P = mu[:, None] * nu[None, :]

    # Main iteration
    log = {'loss': [], 'error': []}

    for iter_num in range(max_iter):
        P_old = P.copy()

        # Step 1: Compute GW cost matrix
        L = gromov_wasserstein_cost_matrix(D1, D2, P)

        # Step 2: Sinkhorn step with current cost matrix
        K = np.exp(-L / epsilon)

        # Sinkhorn iterations (inner loop)
        u = np.ones(n)
        v = np.ones(m)

        for _ in range(20):  # Fixed number of inner Sinkhorn iterations
            u = mu / (K @ v + 1e-100)
            v = nu / (K.T @ u + 1e-100)

        P = u[:, None] * K * v[None, :]

        # Check convergence
        error = np.max(np.abs(P - P_old))
        loss = gromov_wasserstein_loss(D1, D2, P)

        log['loss'].append(loss)
        log['error'].append(error)

        if verbose and iter_num % 10 == 0:
            print(f"Iteration {iter_num}: loss = {loss:.4e}, error = {error:.4e}")

        if error < tol:
            if verbose:
                print(f"Converged after {iter_num + 1} iterations")
            break

    return P, log


class SequentialGromovWasserstein:
    """
    Sequential Gromov-Wasserstein for temporal embedding alignment.

    This extends the Forward-Backward Sinkhorn algorithm to work with
    Gromov-Wasserstein distances, which compare internal geometries
    rather than point-to-point distances.

    Parameters
    ----------
    epsilon : float
        Entropic regularization parameter
    max_outer_iter : int
        Maximum outer iterations (for GW updates)
    max_inner_iter : int
        Maximum inner iterations (for Sinkhorn)
    tol : float
        Convergence tolerance
    metric : str
        Distance metric for computing distance matrices
    verbose : bool
        Print progress
    """

    def __init__(self, epsilon=0.1, max_outer_iter=50, max_inner_iter=100,
                 tol=1e-6, metric='euclidean', verbose=False):
        self.epsilon = epsilon
        self.max_outer_iter = max_outer_iter
        self.max_inner_iter = max_inner_iter
        self.tol = tol
        self.metric = metric
        self.verbose = verbose

        # Will be set during fit
        self.distance_matrices_ = None
        self.couplings_ = None
        self.converged_ = False
        self.n_iter_ = 0

    def fit(self, embeddings, mu=None, nu=None):
        """
        Fit sequential GW alignment.

        Parameters
        ----------
        embeddings : list of np.ndarray
            List of embeddings [X_1, X_2, ..., X_N]
        mu : np.ndarray, optional
            Source distribution
        nu : np.ndarray, optional
            Target distribution

        Returns
        -------
        self
        """
        n_steps = len(embeddings)

        # Compute distance matrices for each time step
        if self.verbose:
            print("Computing distance matrices...")

        self.distance_matrices_ = [
            compute_distance_matrix(X, metric=self.metric)
            for X in embeddings
        ]

        # Initialize marginals
        if mu is None:
            mu = np.ones(embeddings[0].shape[0]) / embeddings[0].shape[0]
        else:
            mu = normalize_distribution(mu)

        if nu is None:
            nu = np.ones(embeddings[-1].shape[0]) / embeddings[-1].shape[0]
        else:
            nu = normalize_distribution(nu)

        # For now, use greedy sequential GW (not global optimization)
        # TODO: Implement global Forward-Backward for GW
        if self.verbose:
            print(f"\nSolving {n_steps - 1} sequential GW problems...")

        self.couplings_ = []

        for t in range(n_steps - 1):
            if self.verbose:
                print(f"\nStep {t + 1}/{n_steps - 1}: {embeddings[t].shape[0]} → {embeddings[t + 1].shape[0]} points")

            D1 = self.distance_matrices_[t]
            D2 = self.distance_matrices_[t + 1]

            # Use uniform marginals for intermediate steps
            mu_t = mu if t == 0 else None
            nu_t = nu if t == n_steps - 2 else None

            P, log = entropic_gromov_wasserstein(
                D1, D2, mu_t, nu_t,
                epsilon=self.epsilon,
                max_iter=self.max_outer_iter,
                tol=self.tol,
                verbose=self.verbose
            )

            self.couplings_.append(P)

        self.converged_ = True
        self.n_iter_ = self.max_outer_iter

        return self

    def get_couplings(self):
        """Return the learned transport couplings."""
        if self.couplings_ is None:
            raise RuntimeError("Must call fit() first")
        return self.couplings_

    def transform(self, embeddings):
        """
        Transform embeddings using learned couplings.

        Parameters
        ----------
        embeddings : list of np.ndarray
            Original embeddings

        Returns
        -------
        aligned_embeddings : list of np.ndarray
            Aligned embeddings
        """
        if self.couplings_ is None:
            raise RuntimeError("Must call fit() first")

        aligned = [embeddings[0].copy()]

        for t, P in enumerate(self.couplings_):
            # Normalize coupling
            P_norm = P / (P.sum(axis=1, keepdims=True) + 1e-10)

            # Transport embeddings
            X_next = P_norm @ embeddings[t + 1]
            aligned.append(X_next)

        return aligned

    def fit_transform(self, embeddings, mu=None, nu=None):
        """Fit and transform in one step."""
        self.fit(embeddings, mu, nu)
        return self.transform(embeddings)
