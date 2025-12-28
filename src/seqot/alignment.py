"""
Embedding alignment methods: Global SeqOT, Procrustes, and Aligned UMAP
"""

import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA

from .sinkhorn import ForwardBackwardSinkhorn
from .utils import compute_cosine_distance, normalize_distribution


class GlobalSeqOTAlignment:
    """
    Global Sequential Optimal Transport alignment.

    This method aligns a sequence of embedding spaces by solving a global
    multi-marginal optimal transport problem that minimizes the total
    transport cost across all time steps while maintaining flow conservation.

    Parameters
    ----------
    epsilon : float, default=0.01
        Entropic regularization parameter
    max_iter : int, default=1000
        Maximum iterations for Sinkhorn
    tol : float, default=1e-6
        Convergence tolerance
    verbose : bool, default=False
        Print progress information
    """

    def __init__(self, epsilon=0.01, max_iter=1000, tol=1e-6, verbose=False):
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

        self.solver_ = None
        self.couplings_ = None
        self.cost_matrices_ = None

    def fit_transform(self, embeddings, mu=None, nu=None):
        """
        Align a sequence of embeddings using Global SeqOT.

        Parameters
        ----------
        embeddings : list of np.ndarray
            List of embedding matrices [X_1, X_2, ..., X_N]
            Each X_t has shape (n_t, d_t)
        mu : np.ndarray, optional
            Source distribution over first time step
        nu : np.ndarray, optional
            Target distribution over last time step

        Returns
        -------
        aligned_embeddings : list of np.ndarray
            Aligned embeddings maintaining the global structure
        """
        n_steps = len(embeddings)

        # Compute cost matrices (cosine distance between consecutive steps)
        self.cost_matrices_ = []
        for t in range(n_steps - 1):
            C = compute_cosine_distance(embeddings[t], embeddings[t + 1])
            self.cost_matrices_.append(C)

        # Solve the global SeqOT problem
        self.solver_ = ForwardBackwardSinkhorn(
            epsilon=self.epsilon,
            max_iter=self.max_iter,
            tol=self.tol,
            verbose=self.verbose
        )

        self.solver_.fit(self.cost_matrices_, mu=mu, nu=nu)
        self.couplings_ = self.solver_.get_couplings()

        # Align embeddings by transporting them through the optimal couplings
        aligned_embeddings = self._transport_embeddings(embeddings)

        return aligned_embeddings

    def _transport_embeddings(self, embeddings):
        """
        Transport embeddings forward using the learned couplings.

        The idea: Use the transport plan to propagate embeddings from
        time t to time t+1, creating a consistent global alignment.
        """
        n_steps = len(embeddings)
        aligned = [embeddings[0].copy()]  # Start with original first step

        for t in range(n_steps - 1):
            P = self.couplings_[t]

            # Normalize each row to get conditional distribution
            P_normalized = P / (P.sum(axis=1, keepdims=True) + 1e-10)

            # Transport: X_{t+1}^aligned = P_normalized @ X_{t+1}
            # This creates a weighted average of target embeddings
            # based on the optimal transport plan
            X_next_aligned = P_normalized @ embeddings[t + 1]

            # Alternatively, we can use the transport to create a mapping
            # Here we use barycentric projection
            aligned.append(X_next_aligned)

        return aligned

    def get_couplings(self):
        """Return the learned transport couplings."""
        if self.couplings_ is None:
            raise RuntimeError("Must call fit_transform first")
        return self.couplings_

    def get_intermediate_distributions(self):
        """
        Get the learned intermediate distributions.

        Returns
        -------
        distributions : list of np.ndarray
            The marginal distributions at each intermediate time step
        """
        if self.couplings_ is None:
            raise RuntimeError("Must call fit_transform first")

        distributions = []

        # First distribution is from source
        distributions.append(np.sum(self.couplings_[0], axis=1))

        # Intermediate distributions
        for t in range(len(self.couplings_) - 1):
            dist = np.sum(self.couplings_[t], axis=0)
            distributions.append(dist)

        # Last distribution is from target
        distributions.append(np.sum(self.couplings_[-1], axis=0))

        return distributions


class ProcrustesAlignment:
    """
    Sequential Procrustes alignment (greedy baseline).

    Aligns each pair of consecutive embeddings independently using
    orthogonal Procrustes analysis. This is a greedy approach that
    doesn't consider global structure.

    Parameters
    ----------
    center : bool, default=True
        Center embeddings before alignment
    scale : bool, default=True
        Scale embeddings to unit variance
    """

    def __init__(self, center=True, scale=True):
        self.center = center
        self.scale = scale

        self.transformations_ = None

    def fit_transform(self, embeddings):
        """
        Align embeddings using sequential Procrustes.

        Parameters
        ----------
        embeddings : list of np.ndarray
            List of embedding matrices [X_1, X_2, ..., X_N]

        Returns
        -------
        aligned_embeddings : list of np.ndarray
            Sequentially aligned embeddings
        """
        n_steps = len(embeddings)
        aligned = [embeddings[0].copy()]
        self.transformations_ = []

        for t in range(n_steps - 1):
            X_source = aligned[t]
            X_target = embeddings[t + 1]

            # Ensure same dimensionality (use PCA if needed)
            if X_source.shape[1] != X_target.shape[1]:
                min_dim = min(X_source.shape[1], X_target.shape[1])
                pca_source = PCA(n_components=min_dim)
                pca_target = PCA(n_components=min_dim)
                X_source = pca_source.fit_transform(X_source)
                X_target = pca_target.fit_transform(X_target)

            # Ensure same number of points (use nearest neighbor matching)
            if X_source.shape[0] != X_target.shape[0]:
                min_n = min(X_source.shape[0], X_target.shape[0])
                X_source = X_source[:min_n]
                X_target = X_target[:min_n]

            # Center
            if self.center:
                X_source_mean = X_source.mean(axis=0)
                X_target_mean = X_target.mean(axis=0)
                X_source = X_source - X_source_mean
                X_target = X_target - X_target_mean
            else:
                X_source_mean = 0
                X_target_mean = 0

            # Scale
            if self.scale:
                X_source_scale = np.std(X_source)
                X_target_scale = np.std(X_target)
                X_source = X_source / (X_source_scale + 1e-10)
                X_target = X_target / (X_target_scale + 1e-10)
            else:
                X_source_scale = 1.0
                X_target_scale = 1.0

            # Compute optimal rotation
            R, _ = orthogonal_procrustes(X_source, X_target)

            # Transform
            X_next_aligned = X_source @ R

            # Reverse scaling and centering
            X_next_aligned = X_next_aligned * X_target_scale + X_target_mean

            self.transformations_.append({
                'R': R,
                'source_mean': X_source_mean,
                'target_mean': X_target_mean,
                'source_scale': X_source_scale,
                'target_scale': X_target_scale,
            })

            aligned.append(X_next_aligned)

        return aligned


class AlignedUMAPAlignment:
    """
    Aligned UMAP baseline (requires umap-learn).

    Uses UMAP's built-in alignment capabilities to align embeddings.
    This serves as a baseline comparison for the Global SeqOT method.

    Note: This is a simplified version. Full implementation would use
    umap.AlignedUMAP for temporal alignment.

    Parameters
    ----------
    n_components : int, default=2
        Dimensionality of aligned space
    n_neighbors : int, default=15
        Number of neighbors for UMAP
    """

    def __init__(self, n_components=2, n_neighbors=15):
        self.n_components = n_components
        self.n_neighbors = n_neighbors

        self.models_ = None

    def fit_transform(self, embeddings):
        """
        Align embeddings using UMAP alignment.

        Parameters
        ----------
        embeddings : list of np.ndarray
            List of embedding matrices

        Returns
        -------
        aligned_embeddings : list of np.ndarray
            UMAP-aligned embeddings
        """
        try:
            import umap
        except ImportError:
            raise ImportError(
                "umap-learn is required for AlignedUMAPAlignment. "
                "Install with: pip install umap-learn"
            )

        # For now, use sequential UMAP with the same random state
        # A full implementation would use umap.AlignedUMAP
        aligned = []
        self.models_ = []

        # Fit first embedding
        model_0 = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            random_state=42
        )
        X_0_aligned = model_0.fit_transform(embeddings[0])
        aligned.append(X_0_aligned)
        self.models_.append(model_0)

        # Align subsequent embeddings
        for t in range(1, len(embeddings)):
            model_t = umap.UMAP(
                n_components=self.n_components,
                n_neighbors=self.n_neighbors,
                random_state=42
            )
            X_t_aligned = model_t.fit_transform(embeddings[t])

            # Align to previous using Procrustes
            R, _ = orthogonal_procrustes(X_t_aligned, aligned[t - 1])
            X_t_aligned = X_t_aligned @ R

            aligned.append(X_t_aligned)
            self.models_.append(model_t)

        return aligned
