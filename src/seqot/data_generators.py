"""
Synthetic data generators for testing Global SeqOT
"""

import numpy as np
from sklearn.datasets import make_blobs


def generate_rotating_embeddings(n_steps=5, n_points=100, n_dims=2,
                                 rotation_angle=np.pi/6, noise=0.1,
                                 random_state=42):
    """
    Generate embeddings that rotate smoothly over time.

    This creates a sequence of point clouds that rotate in a known way,
    allowing us to test if the alignment can recover the rotation.

    Parameters
    ----------
    n_steps : int
        Number of time steps
    n_points : int
        Number of points per time step
    n_dims : int
        Dimensionality of embeddings
    rotation_angle : float
        Rotation angle per step (radians)
    noise : float
        Gaussian noise level
    random_state : int
        Random seed

    Returns
    -------
    embeddings : list of np.ndarray
        List of rotated embeddings
    true_correspondences : list of np.ndarray
        Ground truth point correspondences
    """
    rng = np.random.RandomState(random_state)

    # Generate initial configuration
    X_0 = rng.randn(n_points, n_dims)

    embeddings = [X_0]
    true_correspondences = [np.arange(n_points)]

    for t in range(1, n_steps):
        # 2D rotation matrix
        angle = rotation_angle * t
        if n_dims == 2:
            R = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])
        else:
            # Higher dimensional rotation (rotate in first 2 dims)
            R = np.eye(n_dims)
            R[:2, :2] = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])

        # Apply rotation and add noise
        X_t = X_0 @ R.T + rng.randn(n_points, n_dims) * noise

        embeddings.append(X_t)
        true_correspondences.append(np.arange(n_points))

    return embeddings, true_correspondences


def generate_concept_drift(n_steps=5, n_points_per_cluster=50, n_dims=10,
                          drift_rate=0.5, noise=0.1, random_state=42):
    """
    Generate embeddings with concept drift.

    Simulates a scenario where concepts gradually shift over time,
    like "Deep Learning" evolving into "Transformers".

    Parameters
    ----------
    n_steps : int
        Number of time steps
    n_points_per_cluster : int
        Number of points per concept cluster
    n_dims : int
        Dimensionality of embeddings
    drift_rate : float
        Rate of concept drift (0=no drift, 1=complete change)
    noise : float
        Gaussian noise level
    random_state : int
        Random seed

    Returns
    -------
    embeddings : list of np.ndarray
        List of embeddings with concept drift
    cluster_centers : list of np.ndarray
        True cluster centers at each time step
    """
    rng = np.random.RandomState(random_state)

    # Start with 3 distinct concepts
    n_clusters = 3
    centers_0 = rng.randn(n_clusters, n_dims) * 5

    embeddings = []
    cluster_centers = [centers_0]

    # Generate first time step
    X_0, labels_0 = make_blobs(
        n_samples=n_points_per_cluster * n_clusters,
        n_features=n_dims,
        centers=centers_0,
        cluster_std=noise,
        random_state=random_state
    )
    embeddings.append(X_0)

    # Generate subsequent time steps with drift
    for t in range(1, n_steps):
        # Drift centers
        drift_direction = rng.randn(n_clusters, n_dims)
        drift_direction = drift_direction / np.linalg.norm(drift_direction, axis=1, keepdims=True)

        centers_t = cluster_centers[-1] + drift_rate * drift_direction

        X_t, labels_t = make_blobs(
            n_samples=n_points_per_cluster * n_clusters,
            n_features=n_dims,
            centers=centers_t,
            cluster_std=noise,
            random_state=random_state + t
        )

        embeddings.append(X_t)
        cluster_centers.append(centers_t)

    return embeddings, cluster_centers


def generate_tunneling_scenario(n_bridge_points=10, n_distractor_points=90,
                                n_dims=10, random_state=42):
    """
    Generate the "tunneling" test scenario.

    Creates embeddings where:
    - Source (t=0): Starting concept
    - Bridge (t=1,2): Semantically relevant intermediate papers
    - Target (t=3): Final concept

    The bridge points are the "correct" path, while distractors are noise.
    Global SeqOT should tunnel through bridges, while greedy methods
    should diffuse into distractors.

    Parameters
    ----------
    n_bridge_points : int
        Number of semantically relevant bridge points
    n_distractor_points : int
        Number of distractor points at intermediate steps
    n_dims : int
        Dimensionality of embeddings
    random_state : int
        Random seed

    Returns
    -------
    embeddings : list of np.ndarray
        [source, intermediate1, intermediate2, target]
    bridge_indices : list of list of int
        Indices of bridge points at each step
    """
    rng = np.random.RandomState(random_state)

    # Source: Single tight cluster
    source_center = np.zeros(n_dims)
    source = source_center + rng.randn(20, n_dims) * 0.1

    # Target: Single tight cluster far away
    target_center = np.ones(n_dims) * 5
    target = target_center + rng.randn(20, n_dims) * 0.1

    # Intermediate 1: Bridge points + Distractors
    # Bridge points: Along the path from source to target
    bridge1_centers = source_center + 0.33 * (target_center - source_center)
    bridge1 = bridge1_centers + rng.randn(n_bridge_points, n_dims) * 0.2

    # Distractors: Random points scattered around
    distractors1 = rng.randn(n_distractor_points, n_dims) * 3

    intermediate1 = np.vstack([bridge1, distractors1])
    bridge_indices_1 = list(range(n_bridge_points))

    # Intermediate 2: Similar structure
    bridge2_centers = source_center + 0.67 * (target_center - source_center)
    bridge2 = bridge2_centers + rng.randn(n_bridge_points, n_dims) * 0.2

    distractors2 = rng.randn(n_distractor_points, n_dims) * 3

    intermediate2 = np.vstack([bridge2, distractors2])
    bridge_indices_2 = list(range(n_bridge_points))

    embeddings = [source, intermediate1, intermediate2, target]
    bridge_indices = [
        list(range(len(source))),  # All source points
        bridge_indices_1,
        bridge_indices_2,
        list(range(len(target)))   # All target points
    ]

    return embeddings, bridge_indices


def generate_branching_evolution(n_steps=5, n_points=100, n_dims=10,
                                 branch_point=2, random_state=42):
    """
    Generate embeddings with branching evolution.

    Simulates a scenario where a concept splits into multiple branches
    (e.g., "Neural Networks" branching into "CNNs" and "RNNs").

    Parameters
    ----------
    n_steps : int
        Number of time steps
    n_points : int
        Number of points per time step
    n_dims : int
        Dimensionality of embeddings
    branch_point : int
        Time step at which branching occurs
    random_state : int
        Random seed

    Returns
    -------
    embeddings : list of np.ndarray
        List of embeddings with branching
    branch_labels : list of np.ndarray
        Branch membership labels
    """
    rng = np.random.RandomState(random_state)

    embeddings = []
    branch_labels = []

    # Before branching: Single cluster
    center = np.zeros(n_dims)

    for t in range(branch_point):
        X_t = center + rng.randn(n_points, n_dims) * 0.5
        labels_t = np.zeros(n_points, dtype=int)

        embeddings.append(X_t)
        branch_labels.append(labels_t)

    # After branching: Two clusters
    center_A = center + np.array([2.0] + [0.0] * (n_dims - 1))
    center_B = center + np.array([-2.0] + [0.0] * (n_dims - 1))

    for t in range(branch_point, n_steps):
        n_A = n_points // 2
        n_B = n_points - n_A

        X_A = center_A + rng.randn(n_A, n_dims) * 0.5
        X_B = center_B + rng.randn(n_B, n_dims) * 0.5

        X_t = np.vstack([X_A, X_B])
        labels_t = np.array([0] * n_A + [1] * n_B)

        embeddings.append(X_t)
        branch_labels.append(labels_t)

    return embeddings, branch_labels


def generate_varying_size_embeddings(sizes=[50, 100, 75, 120, 90],
                                     n_dims=10, noise=0.2, random_state=42):
    """
    Generate embeddings with varying number of points per time step.

    This tests the algorithm's ability to handle imbalanced transport
    (different numbers of points at each time step).

    Parameters
    ----------
    sizes : list of int
        Number of points at each time step
    n_dims : int
        Dimensionality of embeddings
    noise : float
        Gaussian noise level
    random_state : int
        Random seed

    Returns
    -------
    embeddings : list of np.ndarray
        List of embeddings with varying sizes
    """
    rng = np.random.RandomState(random_state)

    embeddings = []
    base_center = np.zeros(n_dims)

    for t, size in enumerate(sizes):
        # Drift center slightly
        center_t = base_center + rng.randn(n_dims) * 0.5 * t

        X_t = center_t + rng.randn(size, n_dims) * noise

        embeddings.append(X_t)

    return embeddings
